"""Smoke tests for TRBA run_training orchestration."""

import contextlib
import csv
import sys
from pathlib import Path
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

import manuscript.recognizers._trba.training.train as train_module


class FakeLogger:
    def __init__(self):
        self.infos = []
        self.warnings = []

    def info(self, message):
        self.infos.append(str(message))

    def warning(self, message):
        self.warnings.append(str(message))


class FakeWriter:
    instances = []

    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        self.scalars = []
        self.images = []
        self.closed = False
        type(self).instances.append(self)

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, float(value), step))

    def add_image(self, tag, image, step):
        self.images.append((tag, step))

    def close(self):
        self.closed = True


class FakeScaler:
    def __init__(self, *args, **kwargs):
        pass

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

    def state_dict(self):
        return {"scale": 1.0}

    def load_state_dict(self, state):
        self.loaded = state


class FakeProgress:
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def set_postfix(self, **kwargs):
        self.postfix = kwargs


class FakeOCRDataset(torch.utils.data.Dataset):
    transforms = []

    def __init__(
        self,
        csv_path,
        root,
        stoi,
        img_height,
        img_max_width,
        transform,
        encoding,
        max_len=None,
        strict_max_len=None,
        text_mosaic_prob=None,
        text_mosaic_n_words=None,
        text_mosaic_gap_ratio=None,
    ):
        self.csv_path = csv_path
        self.root = root
        self.stoi = stoi
        self.transform = transform
        self.max_len = max_len or 4
        self.items = list(range(4))
        type(self).transforms.append(transform)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return idx

    @staticmethod
    def make_collate_attn(stoi, max_len, drop_blank=True):
        def _collate(batch):
            batch_size = len(batch)
            imgs = torch.zeros(batch_size, 3, 8, 16)
            text_in = torch.full((batch_size, max_len), stoi["<PAD>"], dtype=torch.long)
            target_y = torch.full((batch_size, max_len), stoi["<PAD>"], dtype=torch.long)
            text_in[:, 0] = stoi["<SOS>"]
            text_in[:, 1] = stoi["a"]
            target_y[:, 0] = stoi["a"]
            target_y[:, 1] = stoi["<EOS>"]
            lengths = torch.full((batch_size,), 2, dtype=torch.long)
            return imgs, text_in, target_y, lengths

        return _collate


class TinyAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.attention_cell = nn.Linear(4, 4)
        self.generator = nn.Linear(4, num_classes)


class TinyTRBAModel(nn.Module):
    def __init__(
        self,
        num_classes,
        hidden_size,
        num_encoder_layers,
        img_h,
        img_w,
        cnn_in_channels,
        cnn_out_channels,
        cnn_backbone,
        sos_id,
        eos_id,
        pad_id,
        blank_id,
        use_ctc_head,
    ):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(cnn_in_channels, 4, 1), nn.BatchNorm2d(4))
        self.enc_rnn = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
        self.attn = TinyAttention(num_classes)
        self.proj = nn.Linear(cnn_in_channels, num_classes)

    def forward(self, imgs, text=None, is_train=True, batch_max_length=4):
        steps = text.size(1) if text is not None else batch_max_length
        pooled = imgs.mean(dim=(2, 3))
        logits = self.proj(pooled).unsqueeze(1).repeat(1, steps, 1)
        return {
            "attention_logits": logits,
            "ctc_logits": logits,
            "attention_preds": logits.argmax(dim=-1),
        }

    def compute_ctc_loss(self, ctc_logits, target_y, lengths):
        return ctc_logits.mean() * 0 + torch.tensor(
            0.25, device=ctc_logits.device, dtype=ctc_logits.dtype
        )


def _patch_runtime(monkeypatch):
    FakeWriter.instances.clear()
    FakeOCRDataset.transforms.clear()
    logger = FakeLogger()
    train_transform_marker = object()
    val_transform_marker = object()
    checkpoint_calls = []
    weight_calls = []
    validation_calls = []
    export_calls = []

    fake_recognizers = ModuleType("manuscript.recognizers")

    class FakeTRBA:
        @staticmethod
        def export(**kwargs):
            export_calls.append(kwargs)
            Path(kwargs["output_path"]).write_bytes(b"onnx")

    fake_recognizers.TRBA = FakeTRBA
    monkeypatch.setitem(sys.modules, "manuscript.recognizers", fake_recognizers)
    monkeypatch.setattr(train_module, "setup_logger", lambda exp_dir: logger)
    monkeypatch.setattr(train_module, "SummaryWriter", FakeWriter)
    monkeypatch.setattr(train_module, "OCRDatasetAttn", FakeOCRDataset)
    monkeypatch.setattr(train_module, "TRBAModel", TinyTRBAModel)
    monkeypatch.setattr(
        train_module,
        "load_charset",
        lambda _path: (
            ["<PAD>", "<SOS>", "<EOS>", "<BLANK>", "a"],
            {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<BLANK>": 3, "a": 4},
        ),
    )
    monkeypatch.setattr(train_module, "get_train_transform", lambda *a, **k: train_transform_marker)
    monkeypatch.setattr(train_module, "get_val_transform", lambda *a, **k: val_transform_marker)
    monkeypatch.setattr(train_module, "log_augmentation_previews_tensorboard", lambda *a, **k: None)
    monkeypatch.setattr(train_module, "visualize_predictions_tensorboard", lambda *a, **k: None)
    monkeypatch.setattr(train_module, "load_pretrained_weights", lambda *a, **k: {"ok": True})
    monkeypatch.setattr(train_module, "set_seed", lambda seed: None)
    monkeypatch.setattr(train_module, "_autocast", contextlib.nullcontext)
    monkeypatch.setattr(train_module.amp, "GradScaler", FakeScaler)
    monkeypatch.setattr(train_module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(train_module, "tqdm", lambda iterable, **kwargs: FakeProgress(iterable))
    monkeypatch.setattr(
        train_module,
        "DataLoader",
        lambda dataset, **kwargs: torch.utils.data.DataLoader(
            dataset,
            **{**kwargs, "pin_memory": False},
        ),
    )
    monkeypatch.setattr(
        train_module,
        "run_validation",
        lambda **kwargs: validation_calls.append(kwargs) or {
            "avg_loss": 0.4,
            "acc": 0.75,
            "cer": 0.1,
            "wer": 0.2,
        },
    )

    def _save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, global_step, best_val_loss, best_val_acc, itos, stoi, config, log_dir):
        checkpoint_calls.append(path)
        torch.save({"epoch": epoch, "global_step": global_step}, path)

    def _save_weights(path, model):
        weight_calls.append(path)
        Path(path).write_bytes(b"weights")

    monkeypatch.setattr(train_module, "save_checkpoint", _save_checkpoint)
    monkeypatch.setattr(train_module, "save_weights", _save_weights)

    return {
        "logger": logger,
        "train_transform": train_transform_marker,
        "val_transform": val_transform_marker,
        "checkpoint_calls": checkpoint_calls,
        "weight_calls": weight_calls,
        "validation_calls": validation_calls,
        "export_calls": export_calls,
    }


class TestTRBARunTraining:
    def test_run_training_smoke_split_val_writes_checkpoints_and_exports(self, monkeypatch, tmp_path):
        runtime = _patch_runtime(monkeypatch)
        charset_path = tmp_path / "charset.txt"
        charset_path.write_text("<PAD>\n<SOS>\n<EOS>\n<BLANK>\na\n", encoding="utf-8")

        cfg = train_module.Config(
            {
                "exp_dir": str(tmp_path / "exp"),
                "train_csvs": ["train.csv"],
                "train_roots": ["images"],
                "val_csvs": [None],
                "val_roots": [None],
                "charset_path": str(charset_path),
                "epochs": 1,
                "batch_size": 2,
                "lr": 1e-3,
                "optimizer": "AdamW",
                "scheduler": "None",
                "max_len": 4,
                "hidden_size": 8,
                "num_encoder_layers": 1,
                "cnn_in_channels": 3,
                "cnn_out_channels": 4,
                "cnn_backbone": "seresnet31-lite",
                "val_size": 2,
                "pretrain_weights": "none",
                "val_interval": 1,
            }
        )

        result = train_module.run_training(cfg, device="cpu")
        exp_dir = Path(result["exp_dir"])
        metrics_csv = exp_dir / "metrics_epoch.csv"
        writer = FakeWriter.instances[-1]

        assert result["val_acc"] == 0.75
        assert result["val_loss"] == 0.4
        assert FakeOCRDataset.transforms == [
            None,
            runtime["train_transform"],
            runtime["val_transform"],
        ]
        assert writer.closed is True
        assert any(tag == "Loss/train_step" for tag, _, _ in writer.scalars)
        assert len(runtime["validation_calls"]) == 2
        assert (exp_dir / "config.json").exists()
        assert (exp_dir / "charset.txt").exists()
        assert metrics_csv.exists()
        with metrics_csv.open(encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 2
        assert rows[1][0] == "1"
        assert rows[1][2] == "0.400000"
        assert any(str(path).endswith("last_ckpt.pth") for path in runtime["checkpoint_calls"])
        assert any(str(path).endswith("best_loss_ckpt.pth") for path in runtime["checkpoint_calls"])
        assert any(str(path).endswith("best_acc_ckpt.pth") for path in runtime["checkpoint_calls"])
        assert any(str(path).endswith("best_acc_weights.pth") for path in runtime["weight_calls"])
        assert runtime["export_calls"]
        assert runtime["export_calls"][0]["output_path"].endswith("best_acc_model.onnx")
        assert (exp_dir / "best_acc_model.onnx").exists()

    def test_run_training_resume_falls_back_to_weights_only(self, monkeypatch, tmp_path):
        runtime = _patch_runtime(monkeypatch)
        charset_path = tmp_path / "charset.txt"
        charset_path.write_text("<PAD>\n<SOS>\n<EOS>\n<BLANK>\na\n", encoding="utf-8")
        resume_path = tmp_path / "exp" / "resume.ckpt"
        resume_path.parent.mkdir(parents=True, exist_ok=True)
        resume_path.write_bytes(b"resume")
        calls = []

        def _load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None):
            calls.append((optimizer is not None, scheduler is not None, scaler is not None))
            if optimizer is not None:
                raise RuntimeError("broken optimizer state")
            return {
                "epoch": 0,
                "global_step": 5,
                "best_val_loss": 0.9,
                "best_val_acc": 0.1,
            }

        monkeypatch.setattr(train_module, "load_checkpoint", _load_checkpoint)

        cfg = train_module.Config(
            {
                "exp_dir": str(tmp_path / "exp"),
                "train_csvs": ["train.csv"],
                "train_roots": ["images"],
                "charset_path": str(charset_path),
                "resume_from": str(resume_path),
                "epochs": 1,
                "batch_size": 2,
                "lr": 1e-3,
                "optimizer": "AdamW",
                "scheduler": "None",
                "max_len": 4,
                "hidden_size": 8,
                "num_encoder_layers": 1,
                "cnn_in_channels": 3,
                "cnn_out_channels": 4,
                "cnn_backbone": "seresnet31-lite",
                "val_size": 2,
                "pretrain_weights": "none",
                "val_interval": 1,
            }
        )

        result = train_module.run_training(cfg, device="cpu")

        assert result["exp_dir"] == str(tmp_path / "exp")
        assert calls == [(True, False, True), (False, False, False)]
        assert any("weights only" in warning for warning in runtime["logger"].warnings)
