import importlib
import sys
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest
import torch

import manuscript.recognizers._trba as trba_module
from manuscript.recognizers._trba.data.dataset import OCRDatasetAttn
from manuscript.recognizers._trba.training.utils import is_loss_explosion


pytestmark = pytest.mark.requires_torch


def _import_train_module(monkeypatch):
    fake_tensorboard = ModuleType("torch.utils.tensorboard")
    fake_tensorboard.SummaryWriter = object
    monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", fake_tensorboard)
    sys.modules.pop("manuscript.recognizers._trba.training.train", None)
    return importlib.import_module("manuscript.recognizers._trba.training.train")


def test_sample_batch_input_size_respects_minimums(monkeypatch):
    monkeypatch.setattr(
        "manuscript.recognizers._trba.data.dataset.random.uniform",
        lambda _a, _b: 0.85,
    )

    height, width = OCRDatasetAttn.sample_batch_input_size(
        img_height=28,
        img_width=128,
        resolution_jitter=0.15,
        min_img_height=24,
        min_img_width=132,
    )

    assert height == 24
    assert width == 132


def test_make_collate_attn_applies_batch_resolution_jitter(monkeypatch):
    monkeypatch.setattr(
        "manuscript.recognizers._trba.data.dataset.random.uniform",
        lambda _a, _b: 0.9,
    )

    def fake_get_train_transform(_params, img_h, img_w, **_kwargs):
        def _transform(*, image):
            assert image.ndim == 3
            return {"image": torch.zeros(3, img_h, img_w)}

        return _transform

    monkeypatch.setattr(
        "manuscript.recognizers._trba.data.dataset.get_train_transform",
        fake_get_train_transform,
    )

    stoi = {
        "<PAD>": 0,
        "<SOS>": 1,
        "<EOS>": 2,
        "a": 3,
        "b": 4,
    }
    collate = OCRDatasetAttn.make_collate_attn(
        stoi=stoi,
        max_len=4,
        drop_blank=True,
        batch_img_size=(64, 256),
        resolution_jitter=0.15,
        min_img_height=24,
        min_img_width=132,
        image_transform_params={},
        is_train=True,
    )

    batch = [
        (np.zeros((32, 80, 3), dtype=np.uint8), "ab"),
        (np.ones((28, 72, 3), dtype=np.uint8) * 255, "a"),
    ]
    imgs, text_in, target_y, lengths = collate(batch)

    assert imgs.shape == (2, 3, 58, 230)
    assert text_in.shape == (2, 5)
    assert target_y.shape == (2, 5)
    assert lengths.tolist() == [3, 2]


def test_is_loss_explosion_detects_large_jump_and_nonfinite():
    assert not is_loss_explosion(current_loss=2.0, reference_loss=1.5, factor=10.0)
    assert is_loss_explosion(current_loss=25.0, reference_loss=2.0, factor=10.0)
    assert is_loss_explosion(current_loss=float("inf"), reference_loss=2.0, factor=10.0)


def test_joint_loss_is_computed_in_float32(monkeypatch):
    train_module = _import_train_module(monkeypatch)
    captured = {}

    class FakeModel:
        def compute_ctc_loss(self, ctc_logits, targets, target_lengths):
            captured["dtype"] = ctc_logits.dtype
            captured["targets_shape"] = tuple(targets.shape)
            captured["lengths"] = target_lengths.tolist()
            return ctc_logits.mean()

    result = {
        "attention_logits": torch.tensor(
            [[[0.0, 3.0, -1.0], [0.0, -1.0, 3.0], [3.0, 0.0, -1.0]]],
            dtype=torch.float16,
        ),
        "ctc_logits": torch.randn(1, 4, 3, dtype=torch.float16),
    }
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    target_y = torch.tensor([[1, 2, 0]], dtype=torch.long)
    lengths = torch.tensor([2], dtype=torch.long)

    total_loss, attn_loss, ctc_loss = train_module._compute_joint_loss(
        result=result,
        criterion=criterion,
        target_y=target_y,
        lengths=lengths,
        model=FakeModel(),
        ctc_weight=0.3,
    )

    assert captured == {
        "dtype": torch.float32,
        "targets_shape": (1, 3),
        "lengths": [2],
    }
    assert total_loss.dtype == torch.float32
    assert attn_loss.dtype == torch.float32
    assert ctc_loss.dtype == torch.float32
    assert torch.isfinite(total_loss)


def test_rollback_checkpoint_uses_fresh_grad_scaler(monkeypatch):
    train_module = _import_train_module(monkeypatch)

    fresh_scaler = object()
    captured = {}

    monkeypatch.setattr(train_module, "_make_grad_scaler", lambda: fresh_scaler)

    def fake_load_checkpoint(
        path,
        model=None,
        optimizer=None,
        scheduler=None,
        scaler=None,
        **_kwargs,
    ):
        captured["path"] = path
        captured["model"] = model
        captured["optimizer"] = optimizer
        captured["scheduler"] = scheduler
        captured["scaler"] = scaler
        return {"global_step": 123}

    monkeypatch.setattr(train_module, "load_checkpoint", fake_load_checkpoint)

    ckpt, scaler = train_module._load_rollback_checkpoint(
        "rollback_ckpt.pth",
        model="model",
        optimizer="optimizer",
        scheduler="scheduler",
    )

    assert ckpt == {"global_step": 123}
    assert scaler is fresh_scaler
    assert captured == {
        "path": "rollback_ckpt.pth",
        "model": "model",
        "optimizer": "optimizer",
        "scheduler": "scheduler",
        "scaler": fresh_scaler,
    }


def test_trba_train_passes_new_stability_parameters(monkeypatch):
    captured = {}

    def fake_run_training(cfg, device="cuda"):
        captured["cfg"] = cfg
        captured["device"] = device
        return {"status": "ok"}

    monkeypatch.setattr(trba_module, "_TRAINING_AVAILABLE", True)
    monkeypatch.setattr(
        trba_module,
        "Config",
        lambda payload: SimpleNamespace(**payload),
    )
    monkeypatch.setattr(trba_module, "run_training", fake_run_training)

    result = trba_module.TRBA.train(
        train_csvs="train.csv",
        train_roots="train_images",
        device="cpu",
        auto_rollback_on_loss_explosion=False,
        loss_explosion_factor=12.0,
        loss_explosion_max_retries=4,
        batch_resolution_jitter=0.15,
        batch_resolution_min_h=30,
        batch_resolution_min_w=140,
    )

    assert result == {"status": "ok"}
    assert captured["device"] == "cpu"
    assert captured["cfg"].auto_rollback_on_loss_explosion is False
    assert captured["cfg"].loss_explosion_factor == 12.0
    assert captured["cfg"].loss_explosion_max_retries == 4
    assert captured["cfg"].batch_resolution_jitter == 0.15
    assert captured["cfg"].batch_resolution_min_h == 30
    assert captured["cfg"].batch_resolution_min_w == 140


def test_make_collate_attn_keeps_fixed_validation_size(monkeypatch):
    def fake_get_val_transform(img_h, img_w):
        def _transform(*, image):
            assert image.ndim == 3
            return {"image": torch.ones(3, img_h, img_w)}

        return _transform

    monkeypatch.setattr(
        "manuscript.recognizers._trba.data.dataset.get_val_transform",
        fake_get_val_transform,
    )

    stoi = {
        "<PAD>": 0,
        "<SOS>": 1,
        "<EOS>": 2,
        "a": 3,
    }
    collate = OCRDatasetAttn.make_collate_attn(
        stoi=stoi,
        max_len=4,
        drop_blank=True,
        batch_img_size=(64, 256),
        is_train=False,
    )

    batch = [
        (np.zeros((20, 90, 3), dtype=np.uint8), "a"),
        (np.ones((25, 60, 3), dtype=np.uint8) * 255, "a"),
    ]
    imgs, text_in, target_y, lengths = collate(batch)

    assert imgs.shape == (2, 3, 64, 256)
    assert text_in.shape == (2, 5)
    assert target_y.shape == (2, 5)
    assert lengths.tolist() == [2, 2]


def test_trba_train_defaults_batch_resolution_jitter_to_enabled(monkeypatch):
    captured = {}

    def fake_run_training(cfg, device="cuda"):
        captured["cfg"] = cfg
        captured["device"] = device
        return {"status": "ok"}

    monkeypatch.setattr(trba_module, "_TRAINING_AVAILABLE", True)
    monkeypatch.setattr(
        trba_module,
        "Config",
        lambda payload: SimpleNamespace(**payload),
    )
    monkeypatch.setattr(trba_module, "run_training", fake_run_training)

    trba_module.TRBA.train(
        train_csvs="train.csv",
        train_roots="train_images",
        device="cpu",
    )

    assert captured["device"] == "cpu"
    assert captured["cfg"].auto_rollback_on_loss_explosion is True
    assert captured["cfg"].batch_resolution_jitter == 0.12
