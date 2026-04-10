"""Tests for CharLM train/export orchestration."""

import json
import os
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from manuscript.correctors import CharLM


class TestCharLMTrain:
    def test_train_is_static_method(self):
        assert isinstance(CharLM.__dict__["train"], staticmethod)

    def test_train_builds_config_and_returns_checkpoint_path(self):
        captured = {}
        fake_train_module = ModuleType("manuscript.correctors._charlm.train")
        fake_config_module = ModuleType("manuscript.correctors._charlm.config")
        fake_config_module.DEFAULT_CONFIG = {"default_only": 123, "batch_size": 999}

        def fake_run_training(config):
            captured["config"] = config

        fake_train_module.train = fake_run_training

        with patch.dict(
            sys.modules,
            {
                "manuscript.correctors._charlm.train": fake_train_module,
                "manuscript.correctors._charlm.config": fake_config_module,
            },
        ):
            with patch.object(CharLM, "export") as mock_export:
                result = CharLM.train(
                    words_path="words.txt",
                    text_path="text.txt",
                    pairs_path="pairs.csv",
                    charset_path="charset.txt",
                    exp_dir="exp_charlm_test",
                    epochs=3,
                    checkpoint="resume.pt",
                    custom_flag=True,
                )

        config = captured["config"]
        assert config["default_only"] == 123
        assert config["words_path"] == "words.txt"
        assert config["text_path"] == "text.txt"
        assert config["pairs_path"] == "pairs.csv"
        assert config["charset_path"] == "charset.txt"
        assert config["exp_dir"] == "exp_charlm_test"
        assert config["epochs"] == 3
        assert config["checkpoint"] == "resume.pt"
        assert config["custom_flag"] is True
        mock_export.assert_called_once_with(
            weights_path=os.path.join(
                "exp_charlm_test", "checkpoints", "charlm_epoch_3.pt"
            ),
            vocab_path=os.path.join("exp_charlm_test", "vocab.json"),
            output_path=os.path.join("exp_charlm_test", "charlm_epoch_3.onnx"),
            max_len=config["max_len"],
            emb_size=config["emb_size"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            ffn_size=config["ffn_size"],
            opset_version=14,
            simplify=True,
        )
        assert result == os.path.join(
            "exp_charlm_test", "checkpoints", "charlm_epoch_3.pt"
        )

    def test_train_ignores_export_failures(self):
        fake_train_module = ModuleType("manuscript.correctors._charlm.train")
        fake_config_module = ModuleType("manuscript.correctors._charlm.config")
        fake_config_module.DEFAULT_CONFIG = {}

        def fake_run_training(config):
            return None

        fake_train_module.train = fake_run_training

        with patch.dict(
            sys.modules,
            {
                "manuscript.correctors._charlm.train": fake_train_module,
                "manuscript.correctors._charlm.config": fake_config_module,
            },
        ):
            with patch.object(CharLM, "export", side_effect=RuntimeError("boom")):
                result = CharLM.train(
                    charset_path="charset.txt",
                    exp_dir="exp_charlm_test",
                    epochs=1,
                )

        assert result == os.path.join(
            "exp_charlm_test", "checkpoints", "charlm_epoch_1.pt"
        )


class TestCharLMExport:
    def test_export_is_static_method(self):
        assert isinstance(CharLM.__dict__["export"], staticmethod)

    @pytest.mark.parametrize("missing_field", ["weights", "vocab"])
    def test_export_requires_existing_files(self, tmp_path, missing_field):
        weights_path = tmp_path / "model.pt"
        vocab_path = tmp_path / "vocab.json"
        output_path = tmp_path / "model.onnx"

        weights_path.write_bytes(b"weights")
        vocab_path.write_text(json.dumps(["<PAD>", "<MASK>", "<UNK>", "a"]), encoding="utf-8")

        if missing_field == "weights":
            weights_path.unlink()
            expected = "Weights not found"
        else:
            vocab_path.unlink()
            expected = "Vocab not found"

        with pytest.raises(FileNotFoundError, match=expected):
            CharLM.export(
                weights_path=str(weights_path),
                vocab_path=str(vocab_path),
                output_path=str(output_path),
            )

    def test_export_creates_onnx_file_and_simplifies(self, tmp_path):
        weights_path = tmp_path / "model.pt"
        vocab_path = tmp_path / "vocab.json"
        output_path = tmp_path / "exports" / "model.onnx"

        weights_path.write_bytes(b"weights")
        vocab_path.write_text(
            json.dumps(["<PAD>", "<MASK>", "<UNK>", "a", "b"]),
            encoding="utf-8",
        )

        class FakeCharTransformerMLM(nn.Module):
            def __init__(self, vocab_size, **kwargs):
                super().__init__()
                self.vocab_size = vocab_size
                self.loaded_state = None

            def load_state_dict(self, state_dict):
                self.loaded_state = state_dict

            def forward(self, x):
                return torch.zeros(x.shape[0], x.shape[1], self.vocab_size)

        fake_model_module = ModuleType("manuscript.correctors._charlm.model")
        fake_model_module.CharTransformerMLM = FakeCharTransformerMLM
        fake_onnx = ModuleType("onnx")
        fake_onnx.load = MagicMock(return_value=object())
        fake_onnx.save = MagicMock()
        fake_onnxsim = ModuleType("onnxsim")
        fake_onnxsim.simplify = MagicMock(return_value=(MagicMock(), True))

        def _write_onnx(*args, **kwargs):
            output_path.write_bytes(b"fake onnx")

        with patch.dict(
            sys.modules,
            {
                "manuscript.correctors._charlm.model": fake_model_module,
                "onnx": fake_onnx,
                "onnxsim": fake_onnxsim,
            },
        ):
            with patch.object(
                torch, "load", return_value={"model": {"weight": torch.tensor(1.0)}}
            ):
                with patch.object(torch.onnx, "export", side_effect=_write_onnx) as mock_export:
                    CharLM.export(
                        weights_path=str(weights_path),
                        vocab_path=str(vocab_path),
                        output_path=str(output_path),
                        simplify=True,
                    )

        assert mock_export.called
        assert fake_onnx.load.called
        assert fake_onnxsim.simplify.called
        assert fake_onnx.save.called
        assert output_path.exists()
