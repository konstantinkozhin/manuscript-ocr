"""Prepare, check and run controlled 20-epoch TRBA experiments on Cyrillic OCR."""

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT = REPO_ROOT / "experiments" / "cyrillic_50"
PARSEQ_SETTINGS = {
    "cnn_backbone": "seresnetlite31v2", "decoder_type": "parseq",
    "decoder_layers": 2, "decoder_heads": 8, "decoder_ffn": 1024,
    "decoder_dropout": 0.1, "parseq_permutations": 6, "parseq_refine_iters": 1,
}
VARIANT_OVERRIDES = {
    "parseq_linear": dict(PARSEQ_SETTINGS),
    "tps": {"cnn_backbone": "seresnet31lite", "decoder_type": "attention",
            "transformation": "tps", "tps_num_fiducial": 20},
}
VARIANTS = tuple(VARIANT_OVERRIDES)


def resolve_config(args):
    """Accept a variant name and prepare missing first-run files automatically."""
    requested = Path(args.config)
    if str(requested) in VARIANTS:
        path = DEFAULT_EXPERIMENT / f"{requested}.json"
    elif requested.is_absolute() or requested.is_file():
        path = requested.resolve()
    else:
        path = (REPO_ROOT / requested).resolve()
    if path.is_file():
        return path
    known_variant = path.parent == DEFAULT_EXPERIMENT and path.stem in VARIANTS
    if known_variant and getattr(args, "resume", None):
        run_dir = (
            getattr(args, "run_dir", None) or DEFAULT_EXPERIMENT / "runs" / path.stem
        )
        saved = Path(run_dir).resolve() / "config.json"
        if saved.is_file():
            return saved
        raise FileNotFoundError(
            f"No saved run for {path.stem}. Start without --resume first."
        )
    if known_variant and args.command == "train":
        run_dir = DEFAULT_EXPERIMENT / "runs" / path.stem
        if run_dir.exists():
            raise FileNotFoundError(f"Missing {path}; existing run found. Use --resume.")
        parent_path = next((candidate for candidate in (
            DEFAULT_EXPERIMENT / "runs/parseq_linear/config.json",
            DEFAULT_EXPERIMENT / "parseq_linear.json",
        ) if candidate.is_file()), None)
        if parent_path is not None:
            # Preserve the winner's training settings and the existing split.
            cfg = json.loads(parent_path.read_text(encoding="utf-8"))
            cfg.update(VARIANT_OVERRIDES[path.stem])
            cfg.update(exp_dir=str(run_dir), pretrain_weights=None, resume_from=None)
            path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Created {path.name} using {parent_path} data and settings.", flush=True)
            return path
        if (DEFAULT_EXPERIMENT / "runs").exists():
            raise FileNotFoundError(
                f"Missing {path}. Existing runs found; their preparation files will not be overwritten."
            )
        check_memory(0)
        args.memory_checked = True
        print(
            f"Preparing the Cyrillic dataset and {len(VARIANTS)} experiment configs for the first launch...",
            flush=True,
        )
        prepare(
            argparse.Namespace(
                data=Path(r"C:\shared\orig_cyrillic"),
                output=DEFAULT_EXPERIMENT,
                seed=42,
                batch_size=48,
                num_workers=0,
                refresh=True,
            )
        )
        return path
    raise FileNotFoundError(
        f"Config not found: {path}. Use a variant name: {', '.join(VARIANTS)}"
    )


def check_memory(workers=0):
    """Avoid repeating the observed Windows commit-limit crash on this machine."""
    if platform.system() != "Windows":
        return
    import ctypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong)
            for name in (
                "total_phys",
                "avail_phys",
                "total_commit",
                "avail_commit",
                "total_virtual",
                "avail_virtual",
                "avail_extended_virtual",
            )
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("Cannot read Windows memory status")
    available = status.avail_commit / 1024**3
    required = 4 + 2 * workers
    print(
        f"Available Windows commit: {available:.1f} GiB; required preflight reserve: {required} GiB",
        flush=True,
    )
    if available < required:
        raise RuntimeError(
            "Insufficient Windows commit headroom. Close memory-heavy apps or enable a system-managed pagefile; use num_workers=0."
        )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def archive_unstarted_run(exp):
    """Retain failed startup logs so the same launch command can be retried."""
    exp = exp.resolve()
    metrics = exp / "metrics_epoch.csv"
    if not metrics.is_file() or not (exp / "config.json").is_file():
        return False
    if any(
        p.suffix.lower() in (".pth", ".pt", ".ckpt", ".onnx") for p in exp.rglob("*")
    ):
        return False
    with metrics.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        if not header or header[0] != "epoch" or any(row for row in reader):
            return False
    parent = exp.parent.resolve()
    archive = parent / f"{exp.name}.failed_{time.time_ns()}"
    # Both resolved paths must stay directly within the explicitly selected
    # experiment parent; rename only this checkpoint-free failed run.
    if exp.parent != parent or archive.resolve().parent != parent:
        raise ValueError("Unexpected experiment archive path")
    exp.rename(archive)
    print(
        f"Previous launch did not finish epoch 1. Logs preserved in {archive.name}; restarting epoch 1.",
        flush=True,
    )
    return True


def read_tsv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t", quoting=csv.QUOTE_NONE))
    if not rows or any(len(row) != 2 or not row[0] for row in rows):
        raise ValueError(f"Expected filename/tab/label rows: {path}")
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError(f"Duplicate filenames in {path}")
    # Match OCRDatasetAttn normalization explicitly, before splitting/auditing.
    return [
        [name, label.replace("\u00a0", " ").strip().replace("\ufeff", "")]
        for name, label in rows
    ]


def prepare(args):
    from PIL import Image
    from manuscript.recognizers._trba.data.transforms import DEFAULT_AUG_PARAMS

    root, out = args.data.resolve(), args.output.resolve()
    if out.exists() and (not args.refresh or (out / "runs").exists()):
        raise FileExistsError(
            f"Preparation directory already exists: {out}; choose --output"
        )
    rows = read_tsv(root / "train.tsv")
    source_count = len(rows)
    empty_labels = [row[0] for row in rows if not row[1]]
    rows = [row for row in rows if row[1]]
    test_rows = read_tsv(root / "test.tsv")
    if any(not row[1] for row in test_rows):
        raise ValueError(
            "Test has empty references; define evaluation policy before proceeding"
        )

    def inspect(item):
        split, row = item
        path = (root / split / row[0]).resolve()
        path.relative_to(root / split)
        raw = path.read_bytes()
        with Image.open(io.BytesIO(raw)) as img:
            img.load()  # Fail before training on truncated/unreadable images.
            width, height = img.size
        return hashlib.sha256(raw).hexdigest(), width, height

    print("Checking all images and exact byte duplicates...", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        info = list(pool.map(inspect, [("train", r) for r in rows]))
        test_info = list(pool.map(inspect, [("test", r) for r in test_rows]))
    groups = defaultdict(list)
    for row, (digest, _, _) in zip(rows, info):
        groups[digest].append(row)
    # Respect the supplied dataset split: all nonempty train.tsv rows train,
    # and the complete test.tsv is used for validation every epoch.
    train, val = rows, test_rows
    if not train or not val:
        raise ValueError("Both training and validation sets must be nonempty")
    charset = ["<PAD>", "<SOS>", "<EOS>", "<BLANK>"] + sorted(
        set("".join(r[1] for r in rows))
    )
    unknown_test = sorted(set("".join(r[1] for r in test_rows)) - set(charset))
    if unknown_test:
        raise ValueError(
            f"Test contains characters outside training vocabulary: {unknown_test!r}"
        )
    max_len = max(len(r[1]) for r in rows)
    required = [len(r[1]) + sum(a == b for a, b in zip(r[1], r[1][1:])) for r in rows]
    # Multiples of 64, with sufficient time steps even at width stride 8.
    img_w = max(256, math.ceil(max(required) / 8) * 64)
    clean_test = [
        row for row, entry in zip(test_rows, test_info) if entry[0] not in groups
    ]
    for name, subset in (
        ("train", train),
        ("val", val),
        ("test", test_rows),
        ("test_no_train_duplicates", clean_test),
    ):
        path = out / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["filename", "text"])
            writer.writerows(subset)
    (out / "charset.txt").write_text("\n".join(charset) + "\n", encoding="utf-8")
    write_json(
        out / "annotation_issues.json",
        {
            "conflicting_labels": [
                group for group in groups.values() if len({r[1] for r in group}) > 1
            ],
            "test_overlaps_source_train": [
                row for row, entry in zip(test_rows, test_info) if entry[0] in groups
            ],
            "policy": "Conflicts retained as supplied; no label guessed or corrected automatically",
        },
    )
    base = dict(DEFAULT_AUG_PARAMS)
    base.update(
        train_csvs=[str(out / "train.csv")],
        train_roots=[str(root / "train")],
        val_csvs=[str(out / "val.csv")],
        val_roots=[str(root / "test")],
        charset_path=str(out / "charset.txt"),
        encoding="utf-8",
        img_h=64,
        img_w=img_w,
        max_len=max_len,
        hidden_size=256,
        num_encoder_layers=2,
        cnn_backbone="seresnet31lite",
        cnn_out_channels=512,
        batch_size=args.batch_size,
        epochs=20,
        seed=args.seed,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        cudnn_benchmark=False,
        optimizer="AdamW",
        lr=0.0005,
        weight_decay=0.0001,
        scheduler="OneCycleLR",
        ctc_weight=0.3,
        ctc_weight_decay_epochs=15,
        ctc_weight_min=0.03,
        max_grad_norm=5.0,
        val_interval=1,
        pretrain_weights=None,
        resume_from=None,
        train_proportions=None,
        text_mosaic_prob=0.0,
        freeze_cnn="none",
        freeze_enc_rnn="none",
        freeze_attention="none",
    )
    for name, delta in VARIANT_OVERRIDES.items():
        cfg = dict(base, **delta)
        cfg["exp_dir"] = str(out / "runs" / name)
        write_json(out / f"{name}.json", cfg)
    audit = {
        "data_root": str(root),
        "seed": args.seed,
        "source_train": source_count,
        "excluded_empty_train_labels": empty_labels,
        "train": len(train),
        "val": len(val),
        "test": len(test_rows),
        "test_no_train_duplicates": len(clean_test),
        "vocabulary_source": "train.tsv only",
        "tokens": len(charset),
        "max_train_label_length": max_len,
        "max_ctc_steps_required": max(required),
        "img_h": 64,
        "img_w": img_w,
        "ctc_infeasible_at_width128_stride8": sum(n > 16 for n in required),
        "ctc_infeasible_at_width256_stride8": sum(n > 32 for n in required),
        "exact_duplicate_train_rows": len(rows) - len(groups),
        "conflicting_label_hash_groups": sum(
            len({r[1] for r in g}) > 1 for g in groups.values()
        ),
        "train_test_shared_image_hashes": len(set(groups) & {i[0] for i in test_info}),
        "split_policy": "source folders: train.tsv + train/ for training; complete test.tsv + test/ for validation; no holdout from train",
        "validation_source": "test.tsv",
        "validation_is_test": True,
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("train.tsv", "test.tsv")
        },
        "split_sha256": {
            name: hashlib.sha256((out / name).read_bytes()).hexdigest()
            for name in (
                "train.csv",
                "val.csv",
                "test.csv",
                "test_no_train_duplicates.csv",
                "charset.txt",
            )
        },
    }
    write_json(out / "audit.json", audit)
    print(json.dumps(audit, ensure_ascii=True, indent=2))


def load_model(cfg, device):
    from manuscript.recognizers._trba.model.decoder_parseq import decoder_config
    from manuscript.recognizers._trba.data.transforms import load_charset
    from manuscript.recognizers._trba.model.model import TRBAModel

    itos, stoi = load_charset(cfg["charset_path"])
    model = TRBAModel(
        len(itos),
        hidden_size=cfg["hidden_size"],
        num_encoder_layers=cfg["num_encoder_layers"],
        img_h=cfg["img_h"],
        img_w=cfg["img_w"],
        cnn_backbone=cfg["cnn_backbone"],
        transformation=cfg.get("transformation", "none"),
        tps_num_fiducial=cfg.get("tps_num_fiducial", 20),
        **decoder_config(cfg),
        cnn_out_channels=cfg["cnn_out_channels"],
        sos_id=stoi["<SOS>"],
        eos_id=stoi["<EOS>"],
        pad_id=stoi["<PAD>"],
        blank_id=stoi["<BLANK>"],
    ).to(device)
    return model, itos, stoi


def smoke(args):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from manuscript.recognizers._trba.data.dataset import OCRDatasetAttn
    from manuscript.recognizers._trba.data.transforms import get_train_transform
    from manuscript.recognizers._trba.training.utils import set_seed

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the GPU preflight")
    set_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = cfg.get("cudnn_benchmark", False)
    np.random.seed(cfg["seed"])
    model, _, stoi = load_model(cfg, "cuda")
    transform = get_train_transform(cfg, cfg["img_h"], cfg["img_w"])
    if hasattr(transform, "set_random_seed"):
        transform.set_random_seed(cfg["seed"])
    dataset = OCRDatasetAttn(
        cfg["train_csvs"][0],
        cfg["train_roots"][0],
        stoi,
        transform=transform,
        max_len=cfg["max_len"],
        validate_image=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        collate_fn=dataset.make_collate_attn(stoi, cfg["max_len"]),
    )
    imgs, text, targets, lengths = [
        t.cuda(non_blocking=True) for t in next(iter(loader))
    ]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        foreach=cfg.get("optimizer_foreach", False),
    )
    scaler = torch.amp.GradScaler("cuda")
    times = []
    successful_steps = 0
    overflow_steps = 0
    torch.cuda.reset_peak_memory_stats()
    for attempt in range(20):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            result = model(imgs, text, batch_max_length=cfg["max_len"])
            attn = model.compute_attention_loss(result, targets)
            ctc = model.compute_ctc_loss(result["ctc_logits"], targets, lengths)
            loss = 0.7 * attn + 0.3 * ctc
        if not torch.isfinite(loss) or ctc.item() <= 0:
            raise RuntimeError(
                f"Invalid loss: attention={attn.item()}, ctc={ctc.item()}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
        if not torch.isfinite(norm):
            # GradScaler may need to lower its initial scale. Match training's
            # skipped-update behavior, but require five successful updates.
            scaler.step(optimizer)
            scaler.update()
            overflow_steps += 1
            continue
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        successful_steps += 1
        if successful_steps > 2:
            times.append(time.perf_counter() - started)
        if successful_steps == 5:
            break
    if successful_steps < 5:
        raise RuntimeError("Gradients remain nonfinite after AMP scale calibration")
    peak = torch.cuda.max_memory_allocated() / 1024**3
    model.eval()
    latency = {}
    with torch.inference_mode(), torch.autocast("cuda"):
        for batch in (1, cfg["batch_size"]):
            x = imgs[:batch]
            for _ in range(3):
                model.forward_attention(
                    x, batch_max_length=cfg["max_len"] + 1, onnx_mode=True
                )
            elapsed = []
            for _ in range(10):
                torch.cuda.synchronize()
                started = time.perf_counter()
                model.forward_attention(
                    x, batch_max_length=cfg["max_len"] + 1, onnx_mode=True
                )
                torch.cuda.synchronize()
                elapsed.append((time.perf_counter() - started) * 1000)
            latency[str(batch)] = {
                "median_ms": float(np.median(elapsed)),
                "p95_ms": float(np.percentile(elapsed, 95)),
            }
    report = {
        "config": str(args.config.resolve()),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "python": platform.python_version(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "loss": loss.item(),
        "attention_loss": attn.item(),
        "ctc_loss": ctc.item(),
        "successful_updates": successful_steps,
        "amp_calibration_skips": overflow_steps,
        "peak_allocated_gib": peak,
        "median_train_step_seconds": float(np.median(times)),
        "attention_gpu_fp16_fixed_steps": cfg["max_len"] + 1,
        "latency_by_batch": latency,
        "note": "5 updates on one real batch; warmed timings exclude data loading; not accuracy or full-epoch timing",
    }
    write_json(args.config.with_suffix(".smoke.json"), report)
    print(json.dumps(report, indent=2))


def train(args):
    from manuscript.recognizers._trba.training.train import Config, run_training

    source = json.loads(args.config.read_text(encoding="utf-8"))
    if args.run_dir:
        source["exp_dir"] = str(args.run_dir.resolve())
    if args.resume:
        checkpoint = (
            Path(source["exp_dir"]) / "last_ckpt.pth"
            if args.resume == "last"
            else Path(args.resume).resolve()
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"No checkpoint: {checkpoint}. If epoch 1 never finished, start a fresh --run-dir."
            )
        if checkpoint.parent.resolve() != Path(source["exp_dir"]).resolve():
            raise ValueError(
                "Checkpoint must belong to this experiment; use the matching config or --run-dir"
            )
        source["resume_from"] = str(checkpoint)
    cfg = Config(source)
    exp = Path(cfg.exp_dir)
    if exp.exists() and any(exp.iterdir()) and not getattr(cfg, "resume_from", None):
        if not archive_unstarted_run(exp):
            raise FileExistsError(
                f"Run already exists: {exp}; use --resume to continue it"
            )
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to silently train on CPU")
    if not isinstance(cfg.epochs, int) or cfg.epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    exp.mkdir(parents=True, exist_ok=True)
    snapshot = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], capture_output=True, check=True
    )
    suffix = (
        time.strftime(".resume_%Y%m%d_%H%M%S")
        if getattr(cfg, "resume_from", None)
        else ""
    )
    (exp / f"source{suffix}.patch").write_bytes(snapshot.stdout)
    (exp / f"runner{suffix}.py").write_bytes(Path(__file__).read_bytes())
    # git diff omits newly added, untracked architecture files. Keep the exact
    # model sources for each run/resume as well, so the comparison is reviewable.
    model_snapshot = exp / f"model_sources{suffix}"
    model_snapshot.mkdir(parents=True, exist_ok=True)
    for model_source in (REPO_ROOT / "src/manuscript/recognizers/_trba/model").iterdir():
        if model_source.suffix in {".py", ".txt"}:
            (model_snapshot / model_source.name).write_bytes(model_source.read_bytes())
    write_json(
        exp / f"environment{suffix}.json",
        {
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(),
            "python": platform.python_version(),
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        },
    )
    started = time.perf_counter()
    result = run_training(cfg, device="cuda")
    result["invocation_wall_seconds"] = time.perf_counter() - started
    result["resume_from"] = getattr(cfg, "resume_from", None)
    write_json(exp / f"result{suffix}.json", result)


def refinement_report(refs, before, after):
    """Compare decoded strings (after EOS removal), including harmful corrections."""
    from manuscript.recognizers._trba.training.metrics import compute_cer, compute_wer

    if not refs or len(refs) != len(before) or len(refs) != len(after):
        raise ValueError("Refinement evaluation requires equal, nonempty sample lists")
    def metrics(hyps):
        return {"accuracy": sum(r == h for r, h in zip(refs, hyps)) / len(refs),
                "cer": compute_cer(refs, hyps), "wer": compute_wer(refs, hyps)}
    return {
        "before": metrics(before), "after": metrics(after),
        "fixed": sum(b != r and a == r for r, b, a in zip(refs, before, after)),
        "broken": sum(b == r and a != r for r, b, a in zip(refs, before, after)),
        "both_correct": sum(b == r and a == r for r, b, a in zip(refs, before, after)),
        "both_wrong": sum(b != r and a != r for r, b, a in zip(refs, before, after)),
        "changed": sum(b != a for b, a in zip(before, after)),
    }


def evaluate_model(args):
    import torch
    from torch.utils.data import DataLoader
    from manuscript.recognizers._trba.data.dataset import OCRDatasetAttn
    from manuscript.recognizers._trba.data.transforms import (
        decode_tokens,
        get_val_transform,
    )
    from manuscript.recognizers._trba.training.metrics import compute_cer, compute_wer

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    requested_device = getattr(args, "device", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if requested_device == "auto" else requested_device
    if device == "cpu":
        torch.set_num_threads(getattr(args, "cpu_threads", 2))
    model, itos, stoi = load_model(cfg, device)
    compare_refinement = getattr(args, "compare_refinement", False)
    if compare_refinement and model.decoder_type != "parseq":
        raise ValueError("--compare-refinement requires a PARSeq model")
    state = torch.load(args.weights, map_location="cpu", weights_only=True)
    model.load_state_dict(state.get("model_state", state), strict=True)
    model.eval()
    data_dir = Path(cfg["train_csvs"][0]).parent
    images = (
        cfg["val_roots"][0]
        if args.split == "val"
        else str(Path(cfg["train_roots"][0]).parent / "test")
    )
    ds = OCRDatasetAttn(
        str(data_dir / f"{args.split}.csv"),
        images,
        stoi,
        transform=get_val_transform(cfg["img_h"], cfg["img_w"]),
        max_len=cfg["max_len"],
        validate_image=False,
    )
    loader = DataLoader(
        ds,
        batch_size=getattr(args, "batch_size", None) or cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        collate_fn=ds.make_collate_attn(stoi, cfg["max_len"]),
    )
    refs, hyps, before_hyps = [], [], []
    with torch.inference_mode():
        for imgs, _, targets, _ in loader:
            if compare_refinement:
                memory = model._decoder_memory(imgs.to(device))
                stages = model.attention_decoder.greedy_decode(
                    memory, model.evaluation_steps(cfg["max_len"]), return_stages=True
                )
                predictions = stages[-1].argmax(-1)
                before_hyps.extend(decode_tokens(
                    pred, itos, stoi["<PAD>"], stoi["<EOS>"], stoi["<BLANK>"]
                ) for pred in stages[0].argmax(-1).cpu())
            else:
                _, predictions = model.forward_attention(
                    imgs.to(device), batch_max_length=model.evaluation_steps(cfg["max_len"])
                )
            for target, pred in zip(targets, predictions.cpu()):
                refs.append(
                    decode_tokens(
                        target, itos, stoi["<PAD>"], stoi["<EOS>"], stoi["<BLANK>"]
                    )
                )
                hyps.append(
                    decode_tokens(
                        pred, itos, stoi["<PAD>"], stoi["<EOS>"], stoi["<BLANK>"]
                    )
                )
    report = {
        "split": args.split,
        "samples": len(refs),
        "weights": str(args.weights.resolve()),
        "accuracy": sum(r == h for r, h in zip(refs, hyps)) / len(refs),
        "cer": compute_cer(refs, hyps),
        "wer": compute_wer(refs, hyps),
        "precision": "FP32",
        "decoder": model.decoder_type + " greedy",
        "device": device,
    }
    if compare_refinement:
        report["refine_iters"] = model.attention_decoder.refine_iters
        report["refinement"] = refinement_report(refs, before_hyps, hyps)
    target = args.weights.parent / f"{args.weights.stem}.{args.split}"
    if compare_refinement:
        target = target.with_suffix(target.suffix + ".refinement")
    write_json(target.with_suffix(target.suffix + ".json"), report)
    with target.with_suffix(target.suffix + ".csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        if compare_refinement:
            writer.writerow(["filename", "reference", "before", "after", "fixed", "broken"])
            writer.writerows(
                (sample[0], ref, before, after, before != ref and after == ref,
                 before == ref and after != ref)
                for sample, ref, before, after in zip(ds.samples, refs, before_hyps, hyps)
            )
        else:
            writer.writerow(["filename", "reference", "prediction"])
            writer.writerows(
                (sample[0], ref, hyp) for sample, ref, hyp in zip(ds.samples, refs, hyps)
            )
    print(json.dumps(report, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--data", type=Path, default=Path(r"C:\shared\orig_cyrillic"))
    prep.add_argument("--output", type=Path, default=DEFAULT_EXPERIMENT)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--batch-size", type=int, default=48)
    prep.add_argument("--num-workers", type=int, default=0)
    prep.add_argument(
        "--refresh",
        action="store_true",
        help="Regenerate preparation files only if no runs directory exists",
    )
    for name in ("smoke", "train", "evaluate"):
        cmd = sub.add_parser(name)
        cmd.add_argument(
            "config",
            type=Path,
            nargs="?",
            default=Path("parseq_linear"),
            help=f"Variant name ({'/'.join(VARIANTS)}) or JSON path",
        )
        if name == "train":
            cmd.add_argument(
                "--resume",
                nargs="?",
                const="last",
                metavar="CHECKPOINT",
                help="Resume last_ckpt.pth, or an explicitly supplied checkpoint",
            )
            cmd.add_argument(
                "--run-dir", type=Path, help="Use a separate directory for a fresh run"
            )
            cmd.add_argument(
                "--prepare-only",
                action="store_true",
                help="Prepare configs without starting training",
            )
        if name == "evaluate":
            cmd.add_argument("--weights", type=Path, required=True)
            cmd.add_argument("--compare-refinement", action="store_true",
                             help="Save PARSeq predictions and metrics before/after refinement")
            cmd.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
            cmd.add_argument("--batch-size", type=int)
            cmd.add_argument("--cpu-threads", type=int, default=2)
            cmd.add_argument(
                "--split",
                choices=("val", "test", "test_no_train_duplicates"),
                default="test",
            )
    args = parser.parse_args(argv)
    if args.command != "prepare":
        try:
            args.config = resolve_config(args)
        except (FileNotFoundError, FileExistsError) as error:
            parser.error(str(error))
        if getattr(args, "prepare_only", False):
            print(f"Ready: {args.config}. Training was not started.")
            return
    workers = (
        args.num_workers
        if args.command == "prepare"
        else json.loads(args.config.read_text(encoding="utf-8")).get("num_workers", 0)
    )
    if not getattr(args, "memory_checked", False):
        check_memory(workers)
    {"prepare": prepare, "smoke": smoke, "train": train, "evaluate": evaluate_model}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
