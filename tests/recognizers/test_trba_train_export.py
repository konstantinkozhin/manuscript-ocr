"""Tests for TRBA train/export orchestration."""

import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

import manuscript.recognizers._trba as trba_module
from manuscript.recognizers import TRBA


class TestTRBATrain:
    def test_train_is_static_method(self):
        assert isinstance(TRBA.__dict__["train"], staticmethod)

    def test_train_raises_with_import_guard_details(self, monkeypatch):
        monkeypatch.setattr(trba_module, "_TRAINING_AVAILABLE", False)
        monkeypatch.setattr(
            trba_module, "_TRAINING_IMPORT_ERROR", ModuleNotFoundError("tensorboard")
        )
        monkeypatch.setattr(
            trba_module,
            "_TRAINING_IMPORT_TRACEBACK",
            "Traceback (most recent call last):\nModuleNotFoundError: tensorboard",
        )

        with pytest.raises(ImportError, match="Training dependencies not available") as exc:
            TRBA.train(train_csvs="train.csv", train_roots="images")

        assert "Original import error traceback" in str(exc.value)
        assert "tensorboard" in str(exc.value)

    def test_train_validates_train_input_lengths(self, monkeypatch):
        monkeypatch.setattr(trba_module, "_TRAINING_AVAILABLE", True)

        with pytest.raises(
            ValueError, match="train_csvs and train_roots must contain the same number"
        ):
            TRBA.train(
                train_csvs=["a.csv", "b.csv"],
                train_roots=["images"],
            )

    def test_train_requires_val_arguments_together(self, monkeypatch):
        monkeypatch.setattr(trba_module, "_TRAINING_AVAILABLE", True)

        with pytest.raises(
            ValueError, match="val_csvs and val_roots must both be provided"
        ):
            TRBA.train(
                train_csvs="train.csv",
                train_roots="images",
                val_csvs="val.csv",
            )

    def test_train_builds_config_and_forwards_to_runner(self, monkeypatch):
        monkeypatch.setattr(trba_module, "_TRAINING_AVAILABLE", True)
        captured = {}

        class FakeConfig(dict):
            def __init__(self, payload):
                super().__init__(payload)
                captured["payload"] = payload

        def fake_run_training(config, device=None):
            captured["config"] = config
            captured["device"] = device
            return "best_model.pth"

        monkeypatch.setattr(trba_module, "Config", FakeConfig)
        monkeypatch.setattr(trba_module, "run_training", fake_run_training)

        result = TRBA.train(
            train_csvs=["train_a.csv", "train_b.csv"],
            train_roots=["images_a", "images_b"],
            val_csvs=["val.csv"],
            val_roots=["val_images"],
            exp_dir="exp_trba",
            train_proportions=[0.7, 0.3],
            resume_from="resume.ckpt",
            save_interval=5,
            device="cpu",
            freeze_cnn="all",
            freeze_enc_rnn="first",
            freeze_attention="all",
            pretrain_weights="weights.pth",
            custom_flag=True,
        )

        payload = captured["payload"]
        assert result == "best_model.pth"
        assert captured["device"] == "cpu"
        assert payload["train_csvs"] == ["train_a.csv", "train_b.csv"]
        assert payload["train_roots"] == ["images_a", "images_b"]
        assert payload["val_csvs"] == ["val.csv"]
        assert payload["val_roots"] == ["val_images"]
        assert payload["exp_dir"] == "exp_trba"
        assert payload["train_proportions"] == [0.7, 0.3]
        assert payload["resume_from"] == "resume.ckpt"
        assert payload["save_interval"] == 5
        assert payload["freeze_cnn"] == "all"
        assert payload["freeze_enc_rnn"] == "first"
        assert payload["freeze_attention"] == "all"
        assert payload["pretrain_weights"] == "weights.pth"
        assert payload["custom_flag"] is True
        assert payload["charset_path"].endswith("configs\\charset.txt") or payload[
            "charset_path"
        ].endswith("configs/charset.txt")


class TestTRBAExport:
    def test_export_is_static_method(self):
        assert isinstance(TRBA.__dict__["export"], staticmethod)

    @pytest.mark.parametrize("missing_field", ["weights", "config", "charset"])
    def test_export_requires_existing_files(self, tmp_path, missing_field):
        weights_path = tmp_path / "model.pth"
        config_path = tmp_path / "model.json"
        charset_path = tmp_path / "charset.txt"
        output_path = tmp_path / "model.onnx"

        weights_path.write_bytes(b"weights")
        config_path.write_text("{}", encoding="utf-8")
        charset_path.write_text("<PAD>\n<SOS>\n<EOS>\n<BLANK>\na\n", encoding="utf-8")

        if missing_field == "weights":
            weights_path.unlink()
            expected = "Weights file not found"
        elif missing_field == "config":
            config_path.unlink()
            expected = "Config file not found"
        else:
            charset_path.unlink()
            expected = "Charset file not found"

        with pytest.raises(FileNotFoundError, match=expected):
            TRBA.export(
                weights_path=str(weights_path),
                config_path=str(config_path),
                charset_path=str(charset_path),
                output_path=str(output_path),
            )

    def test_export_creates_onnx_file_and_simplifies(self, tmp_path):
        weights_path = tmp_path / "model.pth"
        config_path = tmp_path / "model.json"
        charset_path = tmp_path / "charset.txt"
        output_path = tmp_path / "model.onnx"

        weights_path.write_bytes(b"weights")
        config_path.write_text(
            json.dumps(
                {
                    "max_len": 10,
                    "img_h": 32,
                    "img_w": 128,
                    "hidden_size": 64,
                    "num_encoder_layers": 2,
                    "cnn_in_channels": 3,
                    "cnn_out_channels": 32,
                    "cnn_backbone": "seresnet31-lite",
                }
            ),
            encoding="utf-8",
        )
        charset_path.write_text(
            "<PAD>\n<SOS>\n<EOS>\n<BLANK>\na\nb\n",
            encoding="utf-8",
        )

        class FakeTRBAModel(nn.Module):
            def __init__(self, num_classes, **kwargs):
                super().__init__()
                self.num_classes = num_classes
                self.loaded_state = None
                self.loaded_strict = None

            def load_state_dict(self, state_dict, strict=False):
                self.loaded_state = state_dict
                self.loaded_strict = strict

        class FakeTRBAONNXWrapper(nn.Module):
            def __init__(self, model, max_length):
                super().__init__()
                self.num_classes = model.num_classes
                self.max_length = max_length

            def forward(self, x):
                return torch.zeros(x.shape[0], self.max_length, self.num_classes)

        fake_model_module = ModuleType("manuscript.recognizers._trba.model.model")
        fake_model_module.TRBAModel = FakeTRBAModel
        fake_model_module.TRBAONNXWrapper = FakeTRBAONNXWrapper
        fake_onnx = ModuleType("onnx")
        fake_onnx.load = MagicMock(return_value=object())
        fake_onnx.save = MagicMock()
        fake_onnx.checker = type("Checker", (), {"check_model": MagicMock()})()
        fake_onnxsim = ModuleType("onnxsim")
        fake_onnxsim.simplify = MagicMock(return_value=(MagicMock(), True))

        expected_output = torch.zeros(1, 11, 6).numpy()

        def _write_onnx(*args, **kwargs):
            output_path.write_bytes(b"fake onnx")

        class FakeSession:
            def run(self, *_args, **_kwargs):
                return [expected_output]

        with patch.dict(
            sys.modules,
            {
                "manuscript.recognizers._trba.model.model": fake_model_module,
                "onnx": fake_onnx,
                "onnxsim": fake_onnxsim,
            },
        ):
            with patch.object(
                trba_module,
                "load_charset",
                return_value=(
                    ["<PAD>", "<SOS>", "<EOS>", "<BLANK>", "a", "b"],
                    {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<BLANK>": 3, "a": 4, "b": 5},
                ),
            ):
                with patch.object(
                    torch,
                    "load",
                    return_value={"model_state_dict": {"weight": torch.tensor(1.0)}},
                ):
                    with patch.object(torch.onnx, "export", side_effect=_write_onnx) as mock_export:
                        with patch.object(
                            trba_module.ort,
                            "InferenceSession",
                            return_value=FakeSession(),
                        ):
                            TRBA.export(
                                weights_path=str(weights_path),
                                config_path=str(config_path),
                                charset_path=str(charset_path),
                                output_path=str(output_path),
                                simplify=True,
                            )

        assert mock_export.called
        assert fake_onnx.load.called
        assert fake_onnx.checker.check_model.called
        assert fake_onnxsim.simplify.called
        assert fake_onnx.save.called
        assert output_path.exists()
