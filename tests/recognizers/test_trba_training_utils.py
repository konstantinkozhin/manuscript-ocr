"""Reliability tests for TRBA training metrics and utility helpers."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

import manuscript.recognizers._trba.training.metrics as metrics_module
import manuscript.recognizers._trba.training.utils as utils_module


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_decoder = nn.Linear(2, 2, bias=False)
        self.encoder = nn.Linear(2, 2, bias=False)

    def forward(self, x):
        return self.encoder(self.attention_decoder(x))


class DummyScaler:
    def __init__(self):
        self.loaded = None

    def state_dict(self):
        return {"scale": 8.0}

    def load_state_dict(self, state):
        self.loaded = state


class FakeMetric:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def compute(self, predictions, references):
        self.calls.append((predictions, references))
        return self.value


class TestTRBAMetrics:
    def test_get_cer_metric_lazy_loads_once(self, monkeypatch):
        fake_metric = FakeMetric(0.25)
        calls = []
        monkeypatch.setattr(metrics_module, "_cer_metric", None)
        monkeypatch.setattr(
            metrics_module.evaluate,
            "load",
            lambda name: calls.append(name) or fake_metric,
        )

        assert metrics_module.get_cer_metric() is fake_metric
        assert metrics_module.get_cer_metric() is fake_metric
        assert calls == ["cer"]

    def test_compute_cer_and_wer_replace_empty_strings(self, monkeypatch):
        fake_cer = FakeMetric(0.1)
        fake_wer = FakeMetric(0.2)
        monkeypatch.setattr(metrics_module, "_cer_metric", fake_cer)
        monkeypatch.setattr(metrics_module, "_wer_metric", fake_wer)

        assert metrics_module.compute_cer(["", "ab"], ["", "a"]) == 0.1
        assert metrics_module.compute_wer(["", "ab"], ["", "a"]) == 0.2
        assert fake_cer.calls[0] == ([" ", "a"], [" ", "ab"])
        assert fake_wer.calls[0] == ([" ", "a"], [" ", "ab"])

    def test_compute_accuracy_handles_empty_and_exact_match(self):
        assert metrics_module.compute_accuracy([], []) == 0.0
        assert metrics_module.compute_accuracy(["a", "b"], ["a", "c"]) == 0.5


class TestTRBATrainingUtils:
    def test_save_and_load_checkpoint_roundtrip(self, tmp_path):
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scaler = DummyScaler()
        path = tmp_path / "checkpoint.pt"

        utils_module.save_checkpoint(
            path=str(path),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=3,
            global_step=10,
            best_val_loss=0.4,
            best_val_acc=0.8,
            itos=["a"],
            stoi={"a": 0},
            config={"epochs": 3},
            log_dir="logs",
        )

        new_model = TinyModel()
        new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=1)
        new_scaler = DummyScaler()

        metadata = utils_module.load_checkpoint(
            str(path),
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            scaler=new_scaler,
            map_location="cpu",
        )

        assert metadata["epoch"] == 3
        assert metadata["global_step"] == 10
        assert new_scaler.loaded == {"scale": 8.0}
        for p_old, p_new in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p_old, p_new)

    def test_load_checkpoint_filters_mismatched_shapes(self, tmp_path, capsys):
        src_model = TinyModel()
        dst_model = nn.Linear(3, 1, bias=False)
        path = tmp_path / "mismatch.pt"
        torch.save({"model_state": src_model.state_dict()}, path)

        metadata = utils_module.load_checkpoint(
            str(path),
            model=dst_model,
            map_location="cpu",
            strict=False,
        )

        captured = capsys.readouterr().out
        assert "Shape mismatches" in captured or "Missing keys" in captured
        assert "model_state" in metadata

    def test_save_weights_writes_state_dict(self, tmp_path):
        model = TinyModel()
        path = tmp_path / "weights.pt"

        utils_module.save_weights(str(path), model)

        loaded = torch.load(path, map_location="cpu", weights_only=False)
        assert set(loaded.keys()) == set(model.state_dict().keys())

    def test_set_seed_makes_random_generators_repeatable(self):
        utils_module.set_seed(123)
        a = torch.rand(2)
        b = utils_module.random.random()

        utils_module.set_seed(123)
        assert torch.allclose(torch.rand(2), a)
        assert utils_module.random.random() == b

    def test_extract_and_prefix_helpers(self):
        payload = {"weight": torch.tensor([1.0])}
        assert utils_module._extract_model_state({"model_state": payload}) is payload
        assert utils_module._extract_model_state({"state_dict": payload}) is payload
        assert utils_module._extract_model_state({"model": payload}) is payload
        stripped = utils_module._maybe_strip_prefix(
            {"module.layer.weight": torch.tensor([1.0])}
        )
        assert list(stripped.keys()) == ["layer.weight"]

    def test_build_compatible_state_dict_reports_stats(self):
        model = TinyModel()
        filtered, stats = utils_module.build_compatible_state_dict(
            model,
            {
                "module.attention_decoder.weight": model.attention_decoder.weight.detach().clone(),
                "module.encoder.weight": torch.randn(3, 3),
                "module.missing.weight": torch.randn(1),
            },
        )

        assert "attention_decoder.weight" in filtered
        assert stats["num_loaded"] == 1
        assert stats["num_missing"] == 1
        assert stats["num_shape_mismatch"] == 1

    def test_load_pretrained_weights_handles_local_failure(self, monkeypatch):
        model = TinyModel()
        monkeypatch.setattr(utils_module.torch, "load", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("boom")))

        result = utils_module.load_pretrained_weights(model, "weights.pt", map_location="cpu")

        assert result["ok"] is False
        assert "boom" in result["error"]

    def test_load_pretrained_weights_migrates_legacy_attn_prefix(self, monkeypatch):
        model = TinyModel()
        raw_state = {
            "attn.weight": model.attention_decoder.weight.detach().clone(),
            "encoder.weight": model.encoder.weight.detach().clone(),
        }
        monkeypatch.setattr(utils_module.torch, "load", lambda *args, **kwargs: {"model_state": raw_state})

        result = utils_module.load_pretrained_weights(
            model,
            "weights.pt",
            map_location="cpu",
        )

        assert result["ok"] is True
        assert result["num_loaded"] == 2

    def test_load_pretrained_weights_supports_urls(self, monkeypatch):
        model = TinyModel()
        raw_state = {"model": {"encoder.weight": model.encoder.weight.detach().clone()}}
        monkeypatch.setattr(
            utils_module.torch.hub,
            "load_state_dict_from_url",
            lambda *args, **kwargs: raw_state,
        )

        result = utils_module.load_pretrained_weights(
            model,
            "https://example.com/model.pt",
            map_location="cpu",
            logger=SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None),
        )

        assert result["ok"] is True
        assert result["num_loaded"] == 1
