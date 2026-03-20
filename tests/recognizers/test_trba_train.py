from types import SimpleNamespace

import pytest
import torch

import manuscript.recognizers._trba as trba_module
from manuscript.recognizers._trba.data.dataset import OCRDatasetAttn
from manuscript.recognizers._trba.training.utils import is_loss_explosion


pytestmark = pytest.mark.requires_torch


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
    )

    batch = [
        (torch.zeros(3, 64, 256), "ab"),
        (torch.ones(3, 64, 256), "a"),
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
