import pytest
import numpy as np

# Import functions that don't require torch
from manuscript.detectors._east.utils import (
    decode_quads_from_maps,
    expand_boxes,
    create_collage,
)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


# --- Tests for decode_quads_from_maps ---
def test_decode_quads_empty_score_map():
    """Test decode_quads_from_maps with empty score map (all zeros)"""
    score_map = np.zeros((64, 64), dtype=np.float32)
    geo_map = np.zeros((64, 64, 8), dtype=np.float32)

    quads = decode_quads_from_maps(
        score_map=score_map, geo_map=geo_map, score_thresh=0.5, scale=4.0
    )

    assert quads.shape == (0, 9)
    assert quads.dtype == np.float32


def test_decode_quads_single_detection():
    """Test decode_quads_from_maps with single high-confidence pixel"""
    score_map = np.zeros((64, 64), dtype=np.float32)
    geo_map = np.zeros((64, 64, 8), dtype=np.float32)

    # Set one pixel with high score
    score_map[32, 32] = 0.9
    geo_map[32, 32, :] = [-10, -10, 10, -10, 10, 10, -10, 10]

    quads = decode_quads_from_maps(
        score_map=score_map, geo_map=geo_map, score_thresh=0.5, scale=4.0
    )

    assert quads.shape[0] >= 1
    assert quads.shape[1] == 9
    assert quads[0, 8] == pytest.approx(0.9, abs=0.01)


# --- Tests for expand_boxes ---
def test_expand_boxes_no_expansion():
    """Test expand_boxes with zero expansion parameters"""
    quads = np.array(
        [
            [10, 10, 50, 10, 50, 50, 10, 50, 0.9],
            [60, 60, 100, 60, 100, 100, 60, 100, 0.8],
        ],
        dtype=np.float32,
    )

    expanded = expand_boxes(quads, expand_w=0.0, expand_h=0.0)

    assert np.allclose(expanded, quads)


def test_expand_boxes_with_expansion():
    """Test expand_boxes with non-zero expansion"""
    quads = np.array([[10, 10, 50, 10, 50, 50, 10, 50, 0.9]], dtype=np.float32)

    expanded = expand_boxes(quads, expand_w=0.1, expand_h=0.1)

    assert expanded.shape == quads.shape
    assert expanded[0, 8] == quads[0, 8]
    assert not np.allclose(expanded[:, :8], quads[:, :8])


# --- Tests for create_collage ---
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_create_collage_basic():
    """Test create_collage with basic inputs"""
    img_tensor = torch.rand(3, 128, 128) * 2 - 1
    gt_score_map = torch.rand(128, 128)
    gt_geo_map = torch.rand(128, 128, 8)
    gt_quads = np.array([[10, 10, 50, 10, 50, 50, 10, 50, 0.9]])

    collage = create_collage(
        img_tensor=img_tensor,
        gt_score_map=gt_score_map,
        gt_geo_map=gt_geo_map,
        gt_quads=gt_quads,
        cell_size=64,
    )

    assert collage.shape == (128, 640, 3)
    assert collage.dtype == np.uint8


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
def test_create_collage_with_predictions():
    """Test create_collage with both GT and predictions"""
    img_tensor = torch.rand(3, 128, 128) * 2 - 1
    gt_score_map = torch.rand(128, 128)
    gt_geo_map = torch.rand(128, 128, 8)
    gt_quads = np.array([[10, 10, 50, 10, 50, 50, 10, 50, 0.9]])

    pred_score_map = torch.rand(128, 128)
    pred_geo_map = torch.rand(128, 128, 8)
    pred_quads = np.array([[15, 15, 55, 15, 55, 55, 15, 55, 0.85]])

    collage = create_collage(
        img_tensor=img_tensor,
        gt_score_map=gt_score_map,
        gt_geo_map=gt_geo_map,
        gt_quads=gt_quads,
        pred_score_map=pred_score_map,
        pred_geo_map=pred_geo_map,
        pred_quads=pred_quads,
        cell_size=64,
    )

    assert collage.shape == (128, 640, 3)
    assert collage.dtype == np.uint8
