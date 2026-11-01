import os
import json
import random
import csv
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import Counter
from manuscript._internal.edit_distance import levenshtein_distance

from .config import DEFAULT_CONFIG
from .model import CharTransformerMLM
from .dataset import NgramDataset, PairsDataset
from .utils import (
    Logger, build_vocab, CharLMCorrector, evaluate_ocr_confidence,
    evaluate_ocr_with_cer, masked_accuracy, filter_words, load_allowed_chars,
    log_random_examples,
)


def load_pairs(path, allowed_chars, eval_ratio=0.15, seed=42, max_edits=3):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            inc, cor = row["incorrect"].strip().lower(), row["correct"].strip().lower()
            if len(inc) < 4 or len(inc) != len(cor) or inc == cor:
                continue
            if not all(ch in allowed_chars for ch in inc + cor):
                continue
            
            num_diffs = sum(1 for ci, cc in zip(inc, cor) if ci != cc)
            if num_diffs > max_edits:
                continue
            
            pairs.append((inc, cor))
    random.seed(seed)
    random.shuffle(pairs)
    n = int(len(pairs) * eval_ratio)
    return pairs[:n], pairs[n:]


def build_substitutions(pairs):
    subs = []
    for inc, cor in pairs:
        dist = levenshtein_distance(inc, cor)
        if dist == 1 or dist == 2:
            if len(inc) == len(cor):
                for ci, cc in zip(inc, cor):
                    if ci != cc:
                        subs.append((ci, cc))
    counter = Counter(subs)
    return {f"{ci}→{cc}": count for (ci, cc), count in counter.items()}


class MixedDataset(Dataset):
    def __init__(self, ngram_dataset, pairs_dataset, pairs_ratio, steps):
        self.ngram_dataset = ngram_dataset
        self.pairs_dataset = pairs_dataset
        self.pairs_ratio = pairs_ratio
        self.steps = steps
    
    def __len__(self):
        return self.steps
    
    def __getitem__(self, idx):
        if random.random() < self.pairs_ratio:
            idx = random.randint(0, len(self.pairs_dataset) - 1)
            return self.pairs_dataset[idx]
        else:
            return self.ngram_dataset[idx]


def train(config=None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    for key in ("epochs", "batch_size", "accumulation_steps", "max_len", "steps_per_epoch"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    checkpoint = None
    if cfg.get("checkpoint"):
        checkpoint = torch.load(cfg["checkpoint"], map_location="cpu", weights_only=False)
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

    if not allowed_chars:
        raise ValueError("charset must contain letters")
    c2i, i2c, chars = build_vocab([*words, "".join(sorted(allowed_chars))])
    if checkpoint is not None:
        saved_chars = checkpoint.get("vocab")
        if saved_chars is None:
            vocab_path = os.path.join(os.path.dirname(os.path.dirname(cfg["checkpoint"])), "vocab.json")
            if not os.path.isfile(vocab_path):
                raise ValueError("Resume requires the checkpoint vocabulary")
            with open(vocab_path, encoding="utf-8") as stream:
                saved_chars = json.load(stream)
        chars = saved_chars
        c2i = {ch: index for index, ch in enumerate(chars)}
        i2c = dict(enumerate(chars))
    vocab_size = len(chars)
    with open(os.path.join(cfg["exp_dir"], "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(chars, f, ensure_ascii=False)

    pairs_path = cfg.get("pairs_path")
    eval_pairs, train_pairs = [], []
    substitutions = {}
    if pairs_path and os.path.exists(pairs_path):
        eval_pairs, train_pairs = load_pairs(pairs_path, allowed_chars, cfg["eval_ratio"], cfg["seed"], cfg.get("max_pairs_edits", 3))
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

    ngram_dataset = None
    pairs_dataset = None
    
    text_path = cfg.get("text_path")
    if text_path and os.path.exists(text_path):
        ngram_dataset = NgramDataset(
            text_path, c2i, cfg["max_len"], cfg["span_min"], cfg["span_max"],
            cfg["spans_min"], cfg["spans_max"], cfg["ngram_probs"], cfg["steps_per_epoch"], cfg["mask_prob"]
        )
        logger.log(f"NgramDataset: {text_path}")

    if train_pairs:
        pairs_dataset = PairsDataset(train_pairs, c2i, cfg["max_len"], cfg["min_len"])
        logger.log(f"PairsDataset: {len(train_pairs)} pairs")

    if ngram_dataset and pairs_dataset:
        pairs_ratio = cfg.get("pairs_ratio", 0.5)
        dataset = MixedDataset(ngram_dataset, pairs_dataset, pairs_ratio, cfg["steps_per_epoch"])
        logger.log(f"MixedDataset: pairs_ratio={pairs_ratio:.2f} ({pairs_ratio*100:.0f}% OCR)")
    elif ngram_dataset:
        dataset = ngram_dataset
        logger.log("Using NgramDataset only")
    elif pairs_dataset:
        dataset = pairs_dataset
        logger.log("Using PairsDataset only")
    else:
        raise ValueError("No datasets: set text_path or pairs_path")

    train_loader = DataLoader(
        dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0, pin_memory=device == "cuda"
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

    if cfg.get("compile_model", False) and hasattr(torch, "compile"):
        logger.log("Compiling model with torch.compile")
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"] * math.ceil(len(train_loader) / cfg["accumulation_steps"])
    )
    
    scaler = torch.cuda.amp.GradScaler() if cfg.get("use_amp", False) and device == "cuda" else None
    if scaler:
        logger.log("Using mixed precision (AMP)")

    start_epoch = 1
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        if scaler and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        logger.log(f"Resuming from epoch {start_epoch}")

    corrector = CharLMCorrector(
        model, c2i, i2c, device, cfg["max_len"],
        cfg["mask_threshold"], cfg["apply_threshold"], cfg["max_edits"], lexicon
    )

    if checkpoint is not None:
        if "random_state" in checkpoint:
            random.setstate(checkpoint["random_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if device == "cuda" and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        model.train()
        total_loss = total_acc = 0.0
        accumulation_steps = cfg["accumulation_steps"]
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        optimizer.zero_grad()
        for step_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            group_start = (step_idx // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, len(train_loader) - group_start)
            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100)
                scaler.scale(loss / group_size).backward()
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1), ignore_index=-100)
                (loss / group_size).backward()

            if (step_idx + 1) % accumulation_steps == 0 or step_idx + 1 == len(train_loader):
                if scaler:
                    scaler.unscale_(optimizer)
                if cfg["grad_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                if scaler:
                    scale = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    updated = scaler.get_scale() >= scale
                else:
                    optimizer.step()
                    updated = True
                optimizer.zero_grad()
                if updated:
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

        ckpt_dict = {
            "model": model.state_dict(), "epoch": epoch, "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "vocab": chars,
            "random_state": random.getstate(), "torch_rng_state": torch.get_rng_state(),
        }
        if device == "cuda":
            ckpt_dict["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        if scaler:
            ckpt_dict["scaler"] = scaler.state_dict()
        torch.save(ckpt_dict, os.path.join(ckpt_dir, f"charlm_epoch_{epoch}.pt"))

    return model, (c2i, i2c, chars)
