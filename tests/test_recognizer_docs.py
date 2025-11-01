"""
Тесты для проверки примеров из RECOGNIZERS.md документации
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

import manuscript.recognizers._trba as trba_module
from manuscript.recognizers import TRBA


def test_recognizer_doc_single_image(monkeypatch):
    """Тест примера распознавания одного изображения"""
    init_called = {}
    predict_called = {}

    def fake_init(self, *args, **kwargs):
        init_called.update(kwargs)

    def fake_predict(self, images, **kwargs):
        predict_called.update(kwargs)
        assert images == "data/word_images/word_001.jpg" or isinstance(images, list)
        return [("hello", 0.95)]

    monkeypatch.setattr(trba_module.TRBA, "__init__", fake_init, raising=False)
    monkeypatch.setattr(trba_module.TRBA, "predict", fake_predict, raising=False)

    recognizer = TRBA()
    results = recognizer.predict("data/word_images/word_001.jpg")
    text, confidence = results[0]

    assert text == "hello"
    assert confidence == 0.95


def test_recognizer_doc_batch_beam_search(monkeypatch):
    """Тест пакетного распознавания с beam search"""
    init_called = {}
    predict_called = {}

    def fake_init(self, *args, **kwargs):
        init_called.update(kwargs)

    def fake_predict(self, images, **kwargs):
        predict_called.update(kwargs)
        return [
            ("word1", 0.92),
            ("word2", 0.88),
            ("word3", 0.95),
        ]

    monkeypatch.setattr(trba_module.TRBA, "__init__", fake_init, raising=False)
    monkeypatch.setattr(trba_module.TRBA, "predict", fake_predict, raising=False)

    recognizer = TRBA(
        model_path="path/to/custom_model.pth",
        config_path="path/to/custom_config.json",
        device="cuda",
    )

    image_paths = [
        "data/words/word_001.jpg",
        "data/words/word_002.jpg",
        "data/words/word_003.jpg",
    ]

    results = recognizer.predict(
        images=image_paths,
        batch_size=16,
        mode="beam",
        beam_size=10,
        temperature=1.5,
        alpha=0.9,
    )

    assert len(results) == 3
    assert init_called.get("model_path") == "path/to/custom_model.pth"
    assert init_called.get("device") == "cuda"
    assert predict_called.get("batch_size") == 16
    assert predict_called.get("mode") == "beam"
    assert predict_called.get("beam_size") == 10


def test_recognizer_doc_greedy_decoding(monkeypatch):
    """Тест быстрого распознавания с greedy декодированием"""
    init_called = {}
    predict_called = {}

    def fake_init(self, *args, **kwargs):
        init_called.update(kwargs)

    def fake_predict(self, images, **kwargs):
        predict_called.update(kwargs)
        return [("fast_text", 0.87)]

    monkeypatch.setattr(trba_module.TRBA, "__init__", fake_init, raising=False)
    monkeypatch.setattr(trba_module.TRBA, "predict", fake_predict, raising=False)

    recognizer = TRBA(device="auto")

    img_rgb = np.zeros((64, 256, 3), dtype=np.uint8)

    results = recognizer.predict(images=img_rgb, mode="greedy", batch_size=1)

    text, confidence = results[0]
    assert text == "fast_text"
    assert predict_called.get("mode") == "greedy"


def test_recognizer_doc_training_single_dataset(monkeypatch):
    """Тест примера обучения на одном датасете"""
    train_called = {}

    def fake_train(*args, **kwargs):
        train_called.update(kwargs)
        return "path/to/best_model.pth"

    monkeypatch.setattr(
        trba_module.TRBA, "train", staticmethod(fake_train), raising=False
    )

    best_model = TRBA.train(
        train_csvs="data/train.csv",
        train_roots="data/train_images",
        val_csvs="data/val.csv",
        val_roots="data/val_images",
        exp_dir="./experiments/trba_exp1",
        epochs=50,
        batch_size=64,
        img_h=64,
        img_w=256,
        max_len=25,
        lr=1e-3,
        optimizer="Adam",
    )

    assert best_model == "path/to/best_model.pth"
    assert train_called["train_csvs"] == "data/train.csv"
    assert train_called["epochs"] == 50
    assert train_called["batch_size"] == 64
    assert train_called["optimizer"] == "Adam"


def test_recognizer_doc_training_multiple_datasets(monkeypatch):
    """Тест обучения на нескольких датасетах с пропорциями"""
    train_called = {}

    def fake_train(*args, **kwargs):
        train_called.update(kwargs)
        return "path/to/best_model.pth"

    monkeypatch.setattr(
        trba_module.TRBA, "train", staticmethod(fake_train), raising=False
    )

    train_csvs = [
        "data/dataset1/train.csv",
        "data/dataset2/train.csv",
        "data/dataset3/train.csv",
    ]
    train_roots = [
        "data/dataset1/images",
        "data/dataset2/images",
        "data/dataset3/images",
    ]
    train_proportions = [0.5, 0.3, 0.2]

    best_model = TRBA.train(
        train_csvs=train_csvs,
        train_roots=train_roots,
        train_proportions=train_proportions,
        val_csvs="data/val.csv",
        val_roots="data/val_images",
        exp_dir="./experiments/multi_dataset",
        epochs=100,
        batch_size=32,
        lr=5e-4,
        optimizer="AdamW",
        weight_decay=1e-4,
        scheduler="CosineAnnealingLR",
    )

    assert best_model == "path/to/best_model.pth"
    assert train_called["train_csvs"] == train_csvs
    assert train_called["train_proportions"] == train_proportions
    assert train_called["optimizer"] == "AdamW"
    assert train_called["scheduler"] == "CosineAnnealingLR"


def test_recognizer_doc_finetuning(monkeypatch):
    """Тест fine-tuning с замороженными слоями"""
    train_called = {}

    def fake_train(*args, **kwargs):
        train_called.update(kwargs)
        return "path/to/best_model.pth"

    monkeypatch.setattr(
        trba_module.TRBA, "train", staticmethod(fake_train), raising=False
    )

    best_model = TRBA.train(
        train_csvs="data/finetune.csv",
        train_roots="data/finetune_images",
        val_csvs="data/val.csv",
        val_roots="data/val_images",
        exp_dir="./experiments/finetune_frozen_cnn",
        pretrain_weights="default",
        freeze_cnn="all",
        freeze_enc_rnn="none",
        freeze_attention="none",
        epochs=20,
        batch_size=64,
        lr=1e-4,
        optimizer="Adam",
    )

    assert best_model == "path/to/best_model.pth"
    assert train_called["pretrain_weights"] == "default"
    assert train_called["freeze_cnn"] == "all"
    assert train_called["freeze_enc_rnn"] == "none"
    assert train_called["lr"] == 1e-4


def test_recognizer_doc_resume_training(monkeypatch):
    """Тест возобновления обучения"""
    train_called = {}

    def fake_train(*args, **kwargs):
        train_called.update(kwargs)
        return "path/to/best_model.pth"

    monkeypatch.setattr(
        trba_module.TRBA, "train", staticmethod(fake_train), raising=False
    )

    best_model = TRBA.train(
        train_csvs="data/train.csv",
        train_roots="data/train_images",
        val_csvs="data/val.csv",
        val_roots="data/val_images",
        resume_path="experiments/trba_exp1/checkpoints/last.pth",
        epochs=100,
        batch_size=32,
        save_every=5,
    )

    assert best_model == "path/to/best_model.pth"
    assert train_called["resume_path"] == "experiments/trba_exp1/checkpoints/last.pth"
    assert train_called["save_every"] == 5


def test_recognizer_doc_dual_validation(monkeypatch):
    """Тест двойной валидации (greedy + beam)"""
    train_called = {}

    def fake_train(*args, **kwargs):
        train_called.update(kwargs)
        return "path/to/best_model.pth"

    monkeypatch.setattr(
        trba_module.TRBA, "train", staticmethod(fake_train), raising=False
    )

    best_model = TRBA.train(
        train_csvs="data/train.csv",
        train_roots="data/train_images",
        val_csvs="data/val.csv",
        val_roots="data/val_images",
        exp_dir="./experiments/dual_val",
        dual_validate=True,
        beam_size=8,
        beam_alpha=0.9,
        beam_temperature=1.7,
        epochs=50,
        batch_size=64,
        eval_every=1,
    )

    assert best_model == "path/to/best_model.pth"
    assert train_called["dual_validate"] is True
    assert train_called["beam_size"] == 8


def test_recognizer_doc_custom_architecture(monkeypatch):
    """Тест настройки размеров изображения и архитектуры"""
    train_called = {}

    def fake_train(*args, **kwargs):
        train_called.update(kwargs)
        return "path/to/best_model.pth"

    monkeypatch.setattr(
        trba_module.TRBA, "train", staticmethod(fake_train), raising=False
    )

    best_model = TRBA.train(
        train_csvs="data/train.csv",
        train_roots="data/train_images",
        val_csvs="data/val.csv",
        val_roots="data/val_images",
        exp_dir="./experiments/custom_arch",
        img_h=128,
        img_w=512,
        max_len=40,
        hidden_size=512,
        epochs=100,
        batch_size=16,
        lr=1e-3,
    )

    assert best_model == "path/to/best_model.pth"
    assert train_called["img_h"] == 128
    assert train_called["img_w"] == 512
    assert train_called["max_len"] == 40
    assert train_called["hidden_size"] == 512


def test_recognizer_doc_custom_charset(monkeypatch):
    """Тест использования кастомного charset"""
    train_called = {}
    init_called = {}

    def fake_train(*args, **kwargs):
        train_called.update(kwargs)
        return "path/to/best_model.pth"

    def fake_init(self, *args, **kwargs):
        init_called.update(kwargs)

    monkeypatch.setattr(
        trba_module.TRBA, "train", staticmethod(fake_train), raising=False
    )
    monkeypatch.setattr(trba_module.TRBA, "__init__", fake_init, raising=False)

    # Обучение
    best_model = TRBA.train(
        train_csvs="data/train.csv",
        train_roots="data/train_images",
        val_csvs="data/val.csv",
        val_roots="data/val_images",
        exp_dir="./experiments/custom_charset",
        charset_path="data/custom_charset.txt",
        encoding="utf-8",
        epochs=50,
        batch_size=64,
    )

    assert train_called["charset_path"] == "data/custom_charset.txt"
    assert train_called["encoding"] == "utf-8"

    # Инференс
    recognizer = TRBA(
        model_path="experiments/custom_charset/checkpoints/best_acc_weights.pth",
        config_path="experiments/custom_charset/checkpoints/config.json",
        charset_path="data/custom_charset.txt",
    )

    assert init_called["charset_path"] == "data/custom_charset.txt"
