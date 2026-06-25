import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

import manuscript.detectors._eastv2 as eastv2_module
from manuscript.detectors import EASTV2
from manuscript.detectors._eastv2.utils import decode_instance_maps, labels_to_polygons

try:
    import torch
    from manuscript.detectors._eastv2.dataset import EASTV2Dataset
    from manuscript.detectors._eastv2.loss import EASTV2Loss
    from manuscript.detectors._eastv2.model import EASTV2Model, EASTV2OutputHead

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    EASTV2Dataset = None
    EASTV2Loss = None
    EASTV2Model = None
    EASTV2OutputHead = None


def test_eastv2_public_import():
    assert EASTV2 is not None
    assert isinstance(EASTV2.__dict__["train"], staticmethod)


def test_decode_instance_maps_splits_two_words():
    score = np.zeros((64, 64), dtype=np.float32)
    boundary = np.zeros_like(score)
    center = np.zeros_like(score)

    score[16:40, 8:28] = 1.0
    score[16:40, 36:56] = 1.0
    center[28, 18] = 1.0
    center[28, 46] = 1.0
    boundary[:, 31:33] = 1.0

    labels = decode_instance_maps(
        score,
        boundary,
        center,
        score_thresh=0.5,
        boundary_thresh=0.5,
        center_thresh=0.5,
    )

    assert labels.max() == 2
    polygons = labels_to_polygons(labels)
    assert len(polygons) == 2


def test_eastv2_train_matches_east_style_orchestration(monkeypatch, tmp_path):
    class _FakeDataset:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.dataset_name = kwargs["dataset_name"]

        def __len__(self):
            return 1

    class _FakeConcatDataset:
        def __init__(self, datasets):
            self.datasets = list(datasets)

        def __len__(self):
            return sum(len(ds) for ds in self.datasets)

    class _FakeModel:
        map_scale = 0.25

        def to(self, device):
            self.device = device
            return self

    monkeypatch.setattr(eastv2_module, "torch", SimpleNamespace(device=lambda name: name))
    monkeypatch.setattr(eastv2_module, "ConcatDataset", _FakeConcatDataset)
    monkeypatch.setattr(eastv2_module, "EASTV2Dataset", _FakeDataset)
    monkeypatch.setattr(eastv2_module, "EASTV2Model", MagicMock(return_value=_FakeModel()))

    mock_train = MagicMock(return_value=object())
    monkeypatch.setattr(eastv2_module, "_run_training", mock_train)
    result = EASTV2.train(
        train_images=str(tmp_path / "train_images"),
        train_anns=str(tmp_path / "train.json"),
        val_images=str(tmp_path / "test_images"),
        val_anns=str(tmp_path / "test.json"),
        experiment_root=str(tmp_path / "experiments"),
        model_name="resnet_quad",
        epochs=3,
        batch_size=1,
        target_size=1440,
        val_interval=3,
        device="cuda",
    )

    assert result is mock_train.return_value
    kwargs = mock_train.call_args.kwargs
    assert kwargs["num_epochs"] == 3
    assert kwargs["batch_size"] == 1
    assert kwargs["target_size"] == 1440
    assert kwargs["val_interval"] == 3
    assert kwargs["resume"] is False
    assert kwargs["val_dataset_names"] == [f"{tmp_path.name}/test"]
    assert isinstance(kwargs["train_dataset"], _FakeConcatDataset)
    assert kwargs["augmentation_config"]["boundary_width"] == 2


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_eastv2_output_head_shapes():
    head = EASTV2OutputHead()
    x = torch.randn(2, 32, 32, 48)

    score, boundary, center = head(x)

    assert score.shape == (2, 1, 32, 48)
    assert boundary.shape == (2, 1, 32, 48)
    assert center.shape == (2, 1, 32, 48)
    assert 0 <= score.min() and score.max() <= 1


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_eastv2_model_forward_shapes():
    model = EASTV2Model(pretrained_backbone=False)
    x = torch.randn(1, 3, 256, 256)

    out = model(x)

    assert {"score", "boundary", "center"}.issubset(out.keys())
    assert out["score"].shape == (1, 1, 64, 64)
    assert out["boundary"].shape == (1, 1, 64, 64)
    assert out["center"].shape == (1, 1, 64, 64)
    assert out["score_logits"].shape == (1, 1, 64, 64)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_eastv2_loss_backward():
    loss_fn = EASTV2Loss()
    gt_score = torch.ones(1, 1, 8, 8)
    gt_boundary = torch.zeros(1, 1, 8, 8)
    gt_center = torch.zeros(1, 1, 8, 8)
    pred_score = torch.full((1, 1, 8, 8), 0.8, requires_grad=True)
    pred_boundary = torch.full((1, 1, 8, 8), 0.2, requires_grad=True)
    pred_center = torch.full((1, 1, 8, 8), 0.1, requires_grad=True)

    loss = loss_fn(
        gt_score,
        pred_score,
        gt_boundary,
        pred_boundary,
        gt_center,
        pred_center,
    )
    loss.backward()

    assert loss.ndim == 0
    assert pred_score.grad is not None
    assert pred_boundary.grad is not None
    assert pred_center.grad is not None
    assert "score" in loss_fn.last_losses


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_eastv2_loss_is_safe_inside_autocast():
    loss_fn = EASTV2Loss()
    gt_score = torch.ones(1, 1, 8, 8)
    gt_boundary = torch.zeros(1, 1, 8, 8)
    gt_center = torch.zeros(1, 1, 8, 8)
    pred_score = torch.full((1, 1, 8, 8), 0.8, requires_grad=True)
    pred_boundary = torch.full((1, 1, 8, 8), 0.2, requires_grad=True)
    pred_center = torch.full((1, 1, 8, 8), 0.1, requires_grad=True)

    if hasattr(torch, "amp"):
        with torch.amp.autocast("cpu"):
            loss = loss_fn(
                gt_score,
                pred_score,
                gt_boundary,
                pred_boundary,
                gt_center,
                pred_center,
            )
    else:
        loss = loss_fn(
            gt_score,
            pred_score,
            gt_boundary,
            pred_boundary,
            gt_center,
            pred_center,
        )

    loss.backward()

    assert torch.isfinite(loss)
    assert pred_score.grad is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_eastv2_dataset_targets(tmp_path):
    annotations = {
        "images": [{"id": 1, "file_name": "test.jpg", "width": 128, "height": 128}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "segmentation": [[16, 16, 56, 16, 56, 40, 16, 40]],
            },
            {
                "id": 2,
                "image_id": 1,
                "segmentation": [[72, 16, 112, 16, 112, 40, 72, 40]],
            },
        ],
    }
    ann_file = tmp_path / "annotations.json"
    ann_file.write_text(json.dumps(annotations), encoding="utf-8")

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "test.jpg"), np.zeros((128, 128, 3), dtype=np.uint8))

    dataset = EASTV2Dataset(
        str(img_dir),
        str(ann_file),
        target_size=128,
        map_scale=0.25,
        color_jitter=None,
    )
    img, target = dataset[0]

    assert img.shape == (3, 128, 128)
    assert target["score_map"].shape == (1, 32, 32)
    assert target["boundary_map"].shape == (1, 32, 32)
    assert target["center_map"].shape == (1, 32, 32)
    assert target["instance_map"].shape == (32, 32)
    assert target["score_map"].sum() > 0
    assert target["boundary_map"].sum() > 0
    assert target["center_map"].max() == pytest.approx(1.0)
    assert int(target["instance_map"].max()) == 2
