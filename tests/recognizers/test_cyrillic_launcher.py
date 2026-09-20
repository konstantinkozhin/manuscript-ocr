"""Launcher recovery checks; never start a training job."""

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import pytest




def test_refinement_report_counts_fixes_and_regressions(launcher):
    report = launcher.refinement_report(
        ["a", "b", "c", "d", "e"], ["x", "b", "c", "x", "x"],
        ["a", "x", "c", "y", "x"],
    )
    assert {k: report[k] for k in ("fixed", "broken", "both_correct", "both_wrong", "changed")} == {
        "fixed": 1, "broken": 1, "both_correct": 1, "both_wrong": 2, "changed": 3,
    }
    assert report["before"]["accuracy"] == report["after"]["accuracy"] == 0.4
    with pytest.raises(ValueError):
        launcher.refinement_report(["a"], [], ["a"])


def test_refinement_evaluation_writes_paired_predictions(launcher, tmp_path):
    import torch
    from PIL import Image
    charset = tmp_path / "charset.txt"
    charset.write_text("<PAD>\n<SOS>\n<EOS>\n<BLANK>\na\nb\nc\nd\n")
    Image.new("RGB", (64, 32), "white").save(tmp_path / "crop.png")
    (tmp_path / "val.csv").write_text("filename,text\ncrop.png,ab\n")
    cfg = dict(charset_path=str(charset), cnn_backbone="seresnetlite31v2",
               cnn_out_channels=32, hidden_size=16, num_encoder_layers=2,
               img_h=32, img_w=64, max_len=4, decoder_type="parseq", decoder_heads=2,
               decoder_ffn=32, batch_size=2,
               train_csvs=[str(tmp_path / "train.csv")], val_roots=[str(tmp_path)])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    net, _, _ = launcher.load_model(cfg, "cpu")
    weights = tmp_path / "weights.pth"
    torch.save(net.state_dict(), weights)
    launcher.evaluate_model(argparse.Namespace(
        config=config_path, weights=weights, device="cpu", cpu_threads=2,
        batch_size=1, compare_refinement=True, split="val",
    ))
    report = json.loads((tmp_path / "weights.val.refinement.json").read_text())
    with (tmp_path / "weights.val.refinement.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == report["samples"] == 1
    assert Path(rows[0]["filename"]) == tmp_path / "crop.png"
    assert rows[0]["reference"] == "ab"
    assert report["refinement"] == launcher.refinement_report(
        [rows[0]["reference"]], [rows[0]["before"]], [rows[0]["after"]])


@pytest.fixture
def launcher(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "scripts" / "cyrillic_experiments.py"
    spec = importlib.util.spec_from_file_location("cyrillic_launcher_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "DEFAULT_EXPERIMENT", tmp_path / "experiments" / "cyrillic_50")
    monkeypatch.setattr(module, "check_memory", lambda workers: None)
    return module


@pytest.mark.parametrize("config", ["parseq_linear", "tps", "experiments/cyrillic_50/parseq_linear.json"])
def test_first_launch_prepares_missing_config(launcher, monkeypatch, config):
    calls = []

    def prepare(args):
        calls.append(args)
        args.output.mkdir(parents=True, exist_ok=True)
        for variant in launcher.VARIANTS:
            (args.output / f"{variant}.json").write_text("{}")

    monkeypatch.setattr(launcher, "prepare", prepare)
    args = argparse.Namespace(config=Path(config), command="train", resume=None)
    result = launcher.resolve_config(args)
    assert result.is_file()
    assert len(calls) == 1
    assert calls[0].batch_size == 48
    assert launcher.resolve_config(args) == result
    assert len(calls) == 1


def test_resume_uses_saved_run_config_without_rebuilding_data(launcher, monkeypatch):
    saved = launcher.DEFAULT_EXPERIMENT / "runs" / "parseq_linear" / "config.json"
    saved.parent.mkdir(parents=True)
    saved.write_text("{}")
    monkeypatch.setattr(launcher, "prepare", lambda args: pytest.fail("Must not rebuild a saved run"))
    args = argparse.Namespace(config=Path("parseq_linear"), command="train", resume="last", run_dir=None)
    assert launcher.resolve_config(args) == saved


def test_missing_custom_path_does_not_start_default_experiment(launcher, monkeypatch):
    monkeypatch.setattr(launcher, "prepare", lambda args: pytest.fail("Unknown path must not prepare data"))
    args = argparse.Namespace(config=Path("typo.json"), command="train", resume=None)
    with pytest.raises(FileNotFoundError, match="Config not found"):
        launcher.resolve_config(args)


def test_retry_preserves_logs_from_failed_first_epoch(launcher):
    exp = launcher.DEFAULT_EXPERIMENT / "runs" / "parseq_linear"
    exp.mkdir(parents=True)
    (exp / "config.json").write_text("{}")
    (exp / "metrics_epoch.csv").write_text("epoch,train_loss\n")
    (exp / "train.log").write_text("startup failed")
    assert launcher.archive_unstarted_run(exp)
    assert not exp.exists()
    archived = list(exp.parent.glob("parseq_linear.failed_*"))
    assert len(archived) == 1
    assert (archived[0] / "train.log").read_text() == "startup failed"


@pytest.mark.parametrize("saved", ["checkpoint", "metrics"])
def test_retry_never_archives_a_trained_run(launcher, saved):
    exp = launcher.DEFAULT_EXPERIMENT / "runs" / "parseq_linear"
    exp.mkdir(parents=True)
    (exp / "config.json").write_text("{}")
    (exp / "metrics_epoch.csv").write_text("epoch,train_loss\n" + ("1,0.5\n" if saved == "metrics" else ""))
    if saved == "checkpoint":
        (exp / "last_ckpt.pth").write_bytes(b"checkpoint")
    assert not launcher.archive_unstarted_run(exp)
    assert exp.exists()


def test_prepare_uses_complete_source_train_and_test_for_validation(launcher, tmp_path):
    from PIL import Image

    root = tmp_path / "data"
    for split in ("train", "test"):
        (root / split).mkdir(parents=True)
    (root / "train.tsv").write_text("a.png\ta\nb.png\tb\nempty.png\t \n", encoding="utf-8")
    (root / "test.tsv").write_text("c.png\tab\nd.png\ta\n", encoding="utf-8")
    Image.new("RGB", (16, 16), "white").save(root / "train/a.png")
    Image.new("RGB", (16, 16), "black").save(root / "train/b.png")
    Image.new("RGB", (16, 16), "gray").save(root / "test/c.png")
    # Preserve the supplied split even if it contains a duplicate; audit it.
    (root / "test/d.png").write_bytes((root / "train/a.png").read_bytes())
    out = launcher.DEFAULT_EXPERIMENT
    launcher.prepare(argparse.Namespace(
        data=root, output=out, refresh=False, seed=42, batch_size=48, num_workers=0,
    ))

    def read_csv(name):
        with (out / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    assert read_csv("train.csv") == [
        {"filename": "a.png", "text": "a"}, {"filename": "b.png", "text": "b"},
    ]
    assert read_csv("val.csv") == read_csv("test.csv") == [
        {"filename": "c.png", "text": "ab"}, {"filename": "d.png", "text": "a"},
    ]
    assert len(read_csv("test_no_train_duplicates.csv")) == 1
    for variant in launcher.VARIANTS:
        cfg = json.loads((out / f"{variant}.json").read_text())
        assert cfg["train_roots"] == [str(root / "train")]
        assert cfg["val_roots"] == [str(root / "test")]
        assert cfg["val_csvs"] == [str(out / "val.csv")]
        assert cfg["batch_size"] == 48
        assert cfg["epochs"] == 20
        assert cfg["pretrain_weights"] is None and cfg["resume_from"] is None
        for key, value in launcher.VARIANT_OVERRIDES[variant].items():
            assert cfg[key] == value
    audit = json.loads((out / "audit.json").read_text())
    assert audit["validation_is_test"] is True
    assert audit["train_test_shared_image_hashes"] == 1
    assert audit["excluded_empty_train_labels"] == ["empty.png"]




def test_missing_tps_config_does_not_recreate_existing_tps_run(launcher):
    root = launcher.DEFAULT_EXPERIMENT
    (root / "runs" / "tps").mkdir(parents=True)
    (root / "parseq_linear.json").write_text("{}")
    args = argparse.Namespace(config=Path("tps"), command="train", resume=None)
    with pytest.raises(FileNotFoundError, match="Use --resume"):
        launcher.resolve_config(args)


