"""Helper-level reliability tests for EAST training utilities."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

import manuscript.detectors._east.train_utils as train_utils


class TinyModel(nn.Module):
    def __init__(self, in_features=2, out_features=2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


class DummyScaler:
    def __init__(self):
        self.loaded = None

    def state_dict(self):
        return {"scale": 2.0}

    def load_state_dict(self, state):
        self.loaded = state


def test_is_full_state_checkpoint_detects_required_keys():
    assert train_utils._is_full_state_checkpoint(
        {
            "model_state": {},
            "optimizer_state": {},
            "scheduler_state": {},
        }
    )
    assert not train_utils._is_full_state_checkpoint({"model_state": {}})


def test_extract_model_state_supports_common_layouts():
    payload = {"weight": torch.tensor([1.0])}

    assert train_utils._extract_model_state({"model_state": payload}) is payload
    assert train_utils._extract_model_state({"state_dict": payload}) is payload
    assert train_utils._extract_model_state({"model": payload}) is payload
    assert train_utils._extract_model_state(payload) is payload


def test_check_architecture_compatibility_reports_missing_and_mismatch():
    model = TinyModel(2, 2)

    ok, msg = train_utils._check_architecture_compatibility(
        model,
        {"other.weight": torch.randn(2, 2)},
    )
    assert ok is False
    assert "No common keys" in msg

    ok, msg = train_utils._check_architecture_compatibility(
        model,
        {"linear.weight": torch.randn(3, 2), "linear.bias": torch.randn(2)},
    )
    assert ok is False
    assert "Shape mismatches" in msg

    ok, msg = train_utils._check_architecture_compatibility(
        model,
        model.state_dict(),
    )
    assert ok is True
    assert msg == ""


def test_load_weights_only_loads_matching_checkpoint(tmp_path):
    src_model = TinyModel(2, 2)
    dst_model = TinyModel(2, 2)
    path = tmp_path / "weights.pt"
    torch.save({"model_state": src_model.state_dict()}, path)

    train_utils._load_weights_only(dst_model, str(path), device=torch.device("cpu"))

    for src_param, dst_param in zip(src_model.parameters(), dst_model.parameters()):
        assert torch.allclose(src_param, dst_param)


def test_load_weights_only_rejects_architecture_mismatch(tmp_path):
    src_model = TinyModel(2, 3)
    dst_model = TinyModel(2, 2)
    path = tmp_path / "weights.pt"
    torch.save({"state_dict": src_model.state_dict()}, path)

    with pytest.raises(ValueError, match="Architecture mismatch"):
        train_utils._load_weights_only(dst_model, str(path), device=torch.device("cpu"))


def test_find_preview_dataset_walks_nested_datasets():
    class PreviewDataset:
        def __len__(self):
            return 1

        def preview_augmentation(self, idx, name):
            return None

    nested = SimpleNamespace(datasets=[SimpleNamespace(), PreviewDataset()])
    wrapped = SimpleNamespace(dataset=nested)

    found = train_utils._find_preview_dataset(wrapped)
    assert hasattr(found, "preview_augmentation")


def test_log_augmentation_previews_writes_only_existing_previews():
    class PreviewDataset:
        def __len__(self):
            return 1

        def list_augmentations(self):
            return ["first", "second"]

        def preview_augmentation(self, idx, name):
            if name == "first":
                return torch.zeros(4, 4, 3, dtype=torch.uint8).numpy()
            return None

    class Writer:
        def __init__(self):
            self.calls = []

        def add_image(self, tag, image, step, dataformats):
            self.calls.append((tag, image.shape, step, dataformats))

    writer = Writer()
    train_utils._log_augmentation_previews(writer, PreviewDataset())

    assert writer.calls == [("Augmentations/first", (4, 4, 3), 0, "HWC")]


def test_build_scheduler_supports_expected_modes():
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    scheduler, step_mode = train_utils._build_scheduler(
        optimizer, "none", lr=0.1, num_epochs=5
    )
    assert scheduler is None
    assert step_mode == "none"

    scheduler, step_mode = train_utils._build_scheduler(
        optimizer, "cosine_restart", lr=0.1, num_epochs=5, params={"t0": 2}
    )
    assert scheduler is not None
    assert step_mode == "batch"

    scheduler, step_mode = train_utils._build_scheduler(
        optimizer, "linear", lr=0.1, num_epochs=5, params={"final_factor": 0.2}
    )
    assert scheduler is not None
    assert step_mode == "epoch"

    scheduler, step_mode = train_utils._build_scheduler(
        optimizer, "plateau", lr=0.1, num_epochs=5
    )
    assert scheduler is not None
    assert step_mode == "val"


def test_build_scheduler_rejects_unknown_mode():
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="Unknown lr_scheduler"):
        train_utils._build_scheduler(optimizer, "mystery", lr=0.1, num_epochs=5)
