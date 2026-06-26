import json

import cv2
import numpy as np
import pytest

from manuscript.detectors import PSE
from manuscript.detectors._pse.dataset import PSEDataset, shrink_polygon
from manuscript.detectors._pse.utils import labels_to_polygons, pse

try:
    import torch
    from manuscript.detectors._pse.loss import PSELoss
    from manuscript.detectors._pse.model import PSEModel

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    PSEModel = None
    PSELoss = None


def test_pse_public_import():
    assert PSE is not None
    assert isinstance(PSE.__dict__["train"], staticmethod)


def test_shrink_polygon_reduces_area():
    poly = np.array([[0, 0], [100, 0], [100, 40], [0, 40]], dtype=np.float32)
    shrunk = shrink_polygon(poly, rate=0.5)
    assert shrunk.shape[0] >= 4
    assert cv2.contourArea(shrunk.astype(np.float32)) < cv2.contourArea(poly)


def test_pse_expands_small_kernels_to_text():
    kernels = np.zeros((3, 32, 64), dtype=np.uint8)
    kernels[0, 8:20, 5:25] = 1
    kernels[0, 8:20, 35:55] = 1
    kernels[1, 10:18, 8:22] = 1
    kernels[1, 10:18, 38:52] = 1
    kernels[2, 12:16, 11:19] = 1
    kernels[2, 12:16, 41:49] = 1

    labels = pse(kernels, min_area=2)
    assert labels.max() == 2
    polygons = labels_to_polygons(labels, kernels[0].astype(np.float32), min_score=0.0)
    assert len(polygons) == 2


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_pse_model_shapes():
    model = PSEModel(pretrained_backbone=False, kernel_num=7)
    out = model(torch.randn(1, 3, 256, 256))
    assert out["logits"].shape == (1, 7, 64, 64)
    assert out["maps"].shape == (1, 7, 64, 64)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_pse_loss_backward():
    loss_fn = PSELoss()
    logits = torch.randn(1, 7, 16, 16, requires_grad=True)
    gt_text = torch.zeros(1, 16, 16)
    gt_text[:, 4:12, 4:12] = 1
    gt_kernels = torch.zeros(1, 6, 16, 16)
    gt_kernels[:, :, 6:10, 6:10] = 1
    training_mask = torch.ones(1, 16, 16)

    loss = loss_fn(logits, gt_text, gt_kernels, training_mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_pse_dataset_targets(tmp_path):
    annotations = {
        "images": [{"id": 1, "file_name": "test.jpg", "width": 128, "height": 128}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "segmentation": [[16, 16, 56, 16, 56, 40, 16, 40]],
            }
        ],
    }
    ann_file = tmp_path / "annotations.json"
    ann_file.write_text(json.dumps(annotations), encoding="utf-8")

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    cv2.imwrite(str(img_dir / "test.jpg"), np.zeros((128, 128, 3), dtype=np.uint8))

    dataset = PSEDataset(
        str(img_dir),
        str(ann_file),
        target_size=128,
        map_scale=0.25,
        color_jitter=None,
        flip_prob=0.0,
        small_rotate_prob=0.0,
        shear_prob=0.0,
        random_crop_prob=0.0,
        perspective_prob=0.0,
        blur_prob=0.0,
        motion_blur_prob=0.0,
        noise_prob=0.0,
        jpeg_prob=0.0,
        shading_prob=0.0,
        gamma_prob=0.0,
        downscale_prob=0.0,
        negative_prob=0.0,
        hsv_prob=0.0,
        cutout_prob=0.0,
        elastic_prob=0.0,
        fog_prob=0.0,
    )
    img, target = dataset[0]
    assert img.shape == (3, 128, 128)
    assert target["text_map"].shape == (32, 32)
    assert target["kernel_maps"].shape == (6, 32, 32)
    assert target["text_map"].sum() > target["kernel_maps"][-1].sum() > 0
