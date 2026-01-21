import os
import json
import random
import csv
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
from collections import Counter
import Levenshtein

from .config import DEFAULT_CONFIG
from .model import CharTransformerMLM
from .dataset import NgramDataset, PairsDataset
from .utils import (
    Logger, build_vocab, CharLMCorrector, evaluate_ocr_confidence,
    evaluate_ocr_with_cer, masked_accuracy, filter_words, load_allowed_chars,
    log_random_examples,
)


def load_pairs(path, allowed_chars, eval_ratio=0.15, seed=42):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            inc, cor = row["incorrect"].strip().lower(), row["correct"].strip().lower()
            if len(inc) < 4 or len(inc) != len(cor) or inc == cor:
                continue
            if not all(ch in allowed_chars for ch in inc + cor):
                continue
            pairs.append((inc, cor))
    random.seed(seed)
    random.shuffle(pairs)
    n = int(len(pairs) * eval_ratio)
    return pairs[:n], pairs[n:]


def build_substitutions(pairs):
    subs = []
    for inc, cor in pairs:
        dist = Levenshtein.distance(inc, cor)
        if dist == 1 or dist == 2:
            if len(inc) == len(cor):
                for ci, cc in zip(inc, cor):
                    if ci != cc:
                        subs.append((ci, cc))
    counter = Counter(subs)
    return {f"{ci}→{cc}": count for (ci, cc), count in counter.items()}


def train(config=None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    os.makedirs(cfg["exp_dir"], exist_ok=True)
    ckpt_dir = os.path.join(cfg["exp_dir"], "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    logger = Logger(os.path.join(cfg["exp_dir"], "train.log"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log(f"Device: {device}")
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    allowed_chars = load_allowed_chars(cfg["charset_path"])
    
    words_path = cfg.get("words_path")
    if words_path and os.path.exists(words_path):
        with open(words_path, encoding="utf-8") as f:
            raw_words = [w.strip() for w in f if w.strip()]
        words = filter_words(raw_words, min_len=cfg["min_len"], allowed_chars=allowed_chars)
    else:
        words = []
    
    lexicon = set(words)
    if cfg.get("max_words") and len(words) > cfg["max_words"]:
        words = random.sample(words, cfg["max_words"])
    logger.log(f"Words: {len(words):,}")

    c2i, i2c, chars = build_vocab(words)
    vocab_size = len(chars)
    with open(os.path.join(cfg["exp_dir"], "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(chars, f, ensure_ascii=False)

    pairs_path = cfg.get("pairs_path")
    eval_pairs, train_pairs = [], []
    substitutions = {}
    if pairs_path and os.path.exists(pairs_path):
        eval_pairs, train_pairs = load_pairs(pairs_path, allowed_chars, cfg["eval_ratio"], cfg["seed"])
        logger.log(f"Train pairs: {len(train_pairs)}, Eval pairs: {len(eval_pairs)}")
        
        logger.log("Building substitutions dictionary...")
        substitutions = build_substitutions(train_pairs)
        sorted_subs = dict(sorted(substitutions.items(), key=lambda x: x[1], reverse=True))
        sub_path = os.path.join(cfg["exp_dir"], "substitutions.json")
        with open(sub_path, "w", encoding="utf-8") as f:
            json.dump(sorted_subs, f, ensure_ascii=False, indent=2)
        logger.log(f"Substitutions: {len(sorted_subs)}, saved to {sub_path}")
        top_subs = list(sorted_subs.items())[:10]
        for sub, count in top_subs:
            logger.log(f"  {sub}: {count}")
        substitutions = sorted_subs

    datasets = []
    text_path = cfg.get("text_path")
    if text_path and os.path.exists(text_path):
        datasets.append(NgramDataset(
            text_path, c2i, cfg["max_len"], cfg["span_min"], cfg["span_max"],
            cfg["spans_min"], cfg["spans_max"], cfg["ngram_probs"], cfg["steps_per_epoch"], cfg["mask_prob"]
        ))
        logger.log(f"NgramDataset: {text_path}")

    if train_pairs:
        datasets.append(PairsDataset(train_pairs, c2i, cfg["max_len"], cfg["min_len"]))
        logger.log(f"PairsDataset: {len(train_pairs)} pairs")

    if not datasets:
        raise ValueError("No datasets: set text_path or pairs_path")

    train_loader = DataLoader(
        ConcatDataset(datasets), batch_size=cfg["batch_size"], shuffle=True, num_workers=0, pin_memory=True
    )

    for name, pairs, fname in [("eval", eval_pairs, "ocr_eval_pairs.tsv")]:
        path = os.path.join(cfg["exp_dir"], fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write("incorrect\tcorrect\n")
            for inc, cor in pairs:
                f.write(f"{inc}\t{cor}\n")

    model = CharTransformerMLM(
        vocab_size, cfg["emb_size"], cfg["max_len"], cfg["n_layers"],
        cfg["n_heads"], cfg["ffn_size"], cfg["dropout"], c2i["<PAD>"]
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg["epochs"] * len(train_loader))
    )

    start_epoch = 1
    if cfg.get("checkpoint") and os.path.exists(cfg["checkpoint"]):
        logger.log(f"Loading checkpoint: {cfg['checkpoint']}")
        ckpt = torch.load(cfg["checkpoint"], map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        logger.log(f"Resuming from epoch {start_epoch}")

    corrector = CharLMCorrector(
        model, c2i, i2c, device, cfg["max_len"],
        cfg["mask_threshold"], cfg["apply_threshold"], cfg["max_edits"], lexicon,
        substitutions=substitutions, sub_threshold=cfg.get("sub_threshold", 100)
    )

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        total_loss = total_acc = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1), ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            if cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            acc = masked_accuracy(logits, y)
            total_loss += loss.item()
            total_acc += acc
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{acc*100:.1f}%")

        logger.log(f"[Epoch {epoch}] loss={total_loss/len(train_loader):.4f} acc={total_acc/len(train_loader)*100:.2f}%")
        log_random_examples(model, (x, y), logits, c2i, i2c, logger)

        if eval_pairs:
            model.eval()
            if epoch == 1 or epoch % 5 == 0:
                corr, incorr = evaluate_ocr_confidence(model, eval_pairs, c2i, device, cfg["max_len"])
                logger.log(f"Confidence: correct={corr}, incorrect={incorr}")

            stats = evaluate_ocr_with_cer(corrector, eval_pairs, os.path.join(cfg["exp_dir"], f"ocr_epoch_{epoch}.csv"))
            for k, v in stats.items():
                logger.log(f"{k}: {v:.4f}")

        torch.save({"model": model.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict()},
                   os.path.join(ckpt_dir, f"charlm_epoch_{epoch}.pt"))

    return model, (c2i, i2c, chars)
