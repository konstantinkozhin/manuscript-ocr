import pytest
import numpy as np

torch = pytest.importorskip("torch")
import torch.nn as nn
from manuscript.detectors._east.train_utils import (
    dice_coefficient as _dice_coefficient,
    _custom_collate_fn,
)
TORCH_AVAILABLE = True


# --- Tests for _dice_coefficient ---
def test_dice_coefficient_perfect_match():
    """Test Dice coefficient with perfect match"""
    pred = torch.ones(2, 1, 4, 4)
    target = torch.ones(2, 1, 4, 4)

    dice = _dice_coefficient(pred, target)

    # With perfect match dice = 1.0
    assert dice.shape == (2,)
    assert torch.allclose(dice, torch.ones(2))


def test_dice_coefficient_no_overlap():
    """Test Dice coefficient with no overlap"""
    pred = torch.zeros(2, 1, 4, 4)
    pred[0, 0, 0, 0] = 1.0

    target = torch.zeros(2, 1, 4, 4)
    target[0, 0, 3, 3] = 1.0

    dice = _dice_coefficient(pred, target)

    # Without overlap dice is close to 0
    assert dice.shape == (2,)
    assert dice[0] < 0.1  # First sample almost 0


def test_dice_coefficient_partial_overlap():
    """Test Dice coefficient with partial overlap"""
    pred = torch.zeros(1, 1, 4, 4)
    pred[0, 0, :2, :2] = 1.0  # Top-left quadrant

    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 1:3, 1:3] = 1.0  # Partial overlap

    dice = _dice_coefficient(pred, target)

    assert dice.shape == (1,)
    assert 0 < dice[0] < 1


def test_dice_coefficient_batch_processing():
    """Test Dice coefficient with different batch sizes"""
    for batch_size in [1, 2, 4, 8]:
        pred = torch.rand(batch_size, 1, 8, 8)
        target = torch.rand(batch_size, 1, 8, 8)

        dice = _dice_coefficient(pred, target)

        assert dice.shape == (batch_size,)
        assert torch.all(dice >= 0)
        assert torch.all(dice <= 1)


def test_dice_coefficient_different_shapes():
    """Test Dice coefficient with different shapes"""
    shapes = [(4, 4), (8, 8), (16, 16), (32, 32)]

    for h, w in shapes:
        pred = torch.rand(2, 1, h, w)
        target = torch.rand(2, 1, h, w)

        dice = _dice_coefficient(pred, target)

        assert dice.shape == (2,)


def test_dice_coefficient_multi_channel():
    """Test Dice coefficient with multiple channels"""
    pred = torch.rand(2, 3, 4, 4)  # 3 channels
    target = torch.rand(2, 3, 4, 4)

    dice = _dice_coefficient(pred, target)

    # Should work regardless of number of channels
    assert dice.shape == (2,)


def test_dice_coefficient_zeros():
    """Test Dice coefficient with zero tensors"""
    pred = torch.zeros(2, 1, 4, 4)
    target = torch.zeros(2, 1, 4, 4)

    dice = _dice_coefficient(pred, target)

    # Thanks to eps, there should be no division by zero
    assert torch.isfinite(dice).all()


def test_dice_coefficient_numerical_stability():
    """Test numerical stability Dice coefficient"""
    # Very small values
    pred = torch.ones(2, 1, 4, 4) * 1e-8
    target = torch.ones(2, 1, 4, 4) * 1e-8

    dice = _dice_coefficient(pred, target, eps=1e-6)

    assert torch.isfinite(dice).all()


def test_dice_coefficient_custom_eps():
    """Test Dice coefficient with custom eps"""
    pred = torch.rand(2, 1, 4, 4)
    target = torch.rand(2, 1, 4, 4)

    dice1 = _dice_coefficient(pred, target, eps=1e-6)
    dice2 = _dice_coefficient(pred, target, eps=1e-3)

    # Different eps values may give slightly different results
    assert dice1.shape == dice2.shape


def test_dice_coefficient_gradient():
    """Test that gradients flow through Dice coefficient"""
    pred = torch.rand(2, 1, 4, 4, requires_grad=True)
    target = torch.rand(2, 1, 4, 4)

    dice = _dice_coefficient(pred, target)
    loss = (1 - dice).mean()
    loss.backward()

    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


def test_dice_coefficient_binary_masks():
    """Test Dice coefficient with binary masks"""
    pred = torch.zeros(1, 1, 8, 8)
    pred[0, 0, :4, :4] = 1.0

    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, :4, :4] = 1.0

    dice = _dice_coefficient(pred, target)

    # Perfect match
    assert torch.allclose(dice, torch.ones(1))


def test_dice_coefficient_soft_predictions():
    """Test Dice coefficient with soft predictions"""
    pred = torch.rand(2, 1, 4, 4)  # Values in [0, 1]
    target = torch.rand(2, 1, 4, 4)

    dice = _dice_coefficient(pred, target)

    assert dice.shape == (2,)
    assert torch.all(dice >= 0)


# --- Tests for _custom_collate_fn ---


def test_custom_collate_fn_basic():
    """Test basic functionality _custom_collate_fn"""
    # Create a batch of 2 samples
    img1 = torch.rand(3, 64, 64)
    img2 = torch.rand(3, 64, 64)

    target1 = {
        "score_map": torch.rand(1, 16, 16),
        "geo_map": torch.rand(8, 16, 16),
        "quads": torch.rand(5, 8),
    }
    target2 = {
        "score_map": torch.rand(1, 16, 16),
        "geo_map": torch.rand(8, 16, 16),
        "quads": torch.rand(3, 8),
    }

    batch = [(img1, target1), (img2, target2)]

    images, targets = _custom_collate_fn(batch)

    # Check images
    assert images.shape == (2, 3, 64, 64)

    # Check targets
    assert "score_map" in targets
    assert "geo_map" in targets
    assert "quads" in targets

    assert targets["score_map"].shape == (2, 1, 16, 16)
    assert targets["geo_map"].shape == (2, 8, 16, 16)
    assert len(targets["quads"]) == 2
    assert targets["quads"][0].shape == (5, 8)
    assert targets["quads"][1].shape == (3, 8)


def test_custom_collate_fn_single_sample():
    """Test _custom_collate_fn with single sample"""
    img = torch.rand(3, 64, 64)
    target = {
        "score_map": torch.rand(1, 16, 16),
        "geo_map": torch.rand(8, 16, 16),
        "quads": torch.rand(2, 8),
    }

    batch = [(img, target)]

    images, targets = _custom_collate_fn(batch)

    assert images.shape == (1, 3, 64, 64)
    assert targets["score_map"].shape == (1, 1, 16, 16)
    assert targets["geo_map"].shape == (1, 8, 16, 16)
    assert len(targets["quads"]) == 1


def test_custom_collate_fn_multiple_samples():
    """Test _custom_collate_fn with multiple samples"""
    batch_size = 4
    batch = []

    for _ in range(batch_size):
        img = torch.rand(3, 128, 128)
        target = {
            "score_map": torch.rand(1, 32, 32),
            "geo_map": torch.rand(8, 32, 32),
            "quads": torch.rand(np.random.randint(1, 10), 8),
        }
        batch.append((img, target))

    images, targets = _custom_collate_fn(batch)

    assert images.shape == (batch_size, 3, 128, 128)
    assert targets["score_map"].shape == (batch_size, 1, 32, 32)
    assert targets["geo_map"].shape == (batch_size, 8, 32, 32)
    assert len(targets["quads"]) == batch_size


def test_custom_collate_fn_empty_quads():
    """Test _custom_collate_fn with empty quads"""
    img = torch.rand(3, 64, 64)
    target = {
        "score_map": torch.rand(1, 16, 16),
        "geo_map": torch.rand(8, 16, 16),
        "quads": torch.empty(0, 8),  # Empty boxes
    }

    batch = [(img, target)]

    images, targets = _custom_collate_fn(batch)

    assert images.shape == (1, 3, 64, 64)
    assert len(targets["quads"]) == 1
    assert targets["quads"][0].shape == (0, 8)


def test_custom_collate_fn_different_rbox_counts():
    """Test _custom_collate_fn with different number of boxes"""
    img1 = torch.rand(3, 64, 64)
    target1 = {
        "score_map": torch.rand(1, 16, 16),
        "geo_map": torch.rand(8, 16, 16),
        "quads": torch.rand(10, 8),  # 10 boxes
    }

    img2 = torch.rand(3, 64, 64)
    target2 = {
        "score_map": torch.rand(1, 16, 16),
        "geo_map": torch.rand(8, 16, 16),
        "quads": torch.rand(2, 8),  # 2 boxes
    }

    batch = [(img1, target1), (img2, target2)]

    images, targets = _custom_collate_fn(batch)

    # quads should be a list of tensors with different lengths
    assert len(targets["quads"]) == 2
    assert targets["quads"][0].shape[0] == 10
    assert targets["quads"][1].shape[0] == 2


def test_custom_collate_fn_preserves_dtype():
    """Test that _custom_collate_fn preserves data types"""
    img = torch.rand(3, 64, 64, dtype=torch.float32)
    target = {
        "score_map": torch.rand(1, 16, 16, dtype=torch.float32),
        "geo_map": torch.rand(8, 16, 16, dtype=torch.float32),
        "quads": torch.rand(5, 8, dtype=torch.float32),
    }

    batch = [(img, target)]

    images, targets = _custom_collate_fn(batch)

    assert images.dtype == torch.float32
    assert targets["score_map"].dtype == torch.float32
    assert targets["geo_map"].dtype == torch.float32


def test_custom_collate_fn_stack_consistency():
    """Test stack consistency for images and maps"""
    batch = []
    for i in range(3):
        img = torch.rand(3, 64, 64) * i  # Different values
        target = {
            "score_map": torch.rand(1, 16, 16) * i,
            "geo_map": torch.rand(8, 16, 16) * i,
            "quads": torch.rand(2, 8),
        }
        batch.append((img, target))

    images, targets = _custom_collate_fn(batch)

    # Check that stack works correctly
    for i in range(3):
        # Mean value should increase with i
        if i > 0:
            assert images[i].mean() > images[0].mean()


def test_custom_collate_fn_different_image_channels():
    """Test _custom_collate_fn with different number of channels"""
    # Usually 3 channels (RGB)
    img = torch.rand(3, 64, 64)
    target = {
        "score_map": torch.rand(1, 16, 16),
        "geo_map": torch.rand(8, 16, 16),
        "quads": torch.rand(5, 8),
    }

    batch = [(img, target)]

    images, targets = _custom_collate_fn(batch)

    assert images.shape[1] == 3  # RGB channels


def test_custom_collate_fn_large_batch():
    """Test _custom_collate_fn with large batch"""
    batch_size = 16
    batch = []

    for _ in range(batch_size):
        img = torch.rand(3, 256, 256)
        target = {
            "score_map": torch.rand(1, 64, 64),
            "geo_map": torch.rand(8, 64, 64),
            "quads": torch.rand(5, 8),
        }
        batch.append((img, target))

    images, targets = _custom_collate_fn(batch)

    assert images.shape == (batch_size, 3, 256, 256)
    assert targets["score_map"].shape == (batch_size, 1, 64, 64)
    assert targets["geo_map"].shape == (batch_size, 8, 64, 64)
    assert len(targets["quads"]) == batch_size


# --- Additional tests ---
def test_dice_coefficient_edge_case_single_pixel():
    """Test Dice coefficient with single active pixel"""
    pred = torch.zeros(1, 1, 8, 8)
    pred[0, 0, 4, 4] = 1.0

    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 4, 4] = 1.0

    dice = _dice_coefficient(pred, target)

    # Perfect match of a single pixel
    assert torch.allclose(dice, torch.ones(1))


def test_dice_coefficient_asymmetric_overlap():
    """Test Dice coefficient with asymmetric overlap"""
    pred = torch.zeros(1, 1, 8, 8)
    pred[0, 0, :4, :4] = 1.0  # 16 pixels

    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, :2, :2] = 1.0  # 4 pixels

    dice = _dice_coefficient(pred, target)

    # Should be between 0 and 1
    assert 0 < dice[0] < 1


def test_dice_coefficient_continuous_values():
    """Test Dice coefficient with continuous values"""
    pred = torch.rand(3, 1, 16, 16) * 0.8 + 0.1  # In the range [0.1, 0.9]
    target = torch.rand(3, 1, 16, 16) * 0.8 + 0.1

    dice = _dice_coefficient(pred, target)

    assert dice.shape == (3,)
    assert torch.all(dice >= 0)
    assert torch.all(dice <= 1)


def test_custom_collate_fn_maintains_rbox_structure():
    """Test что _custom_collate_fn maintains structure quads"""
    batch = []
    expected_shapes = [(7, 8), (3, 8), (5, 8)]

    for shape in expected_shapes:
        img = torch.rand(3, 64, 64)
        target = {
            "score_map": torch.rand(1, 16, 16),
            "geo_map": torch.rand(8, 16, 16),
            "quads": torch.rand(*shape),
        }
        batch.append((img, target))

    images, targets = _custom_collate_fn(batch)

    for i, expected_shape in enumerate(expected_shapes):
        assert targets["quads"][i].shape == expected_shape


def test_dice_coefficient_high_precision():
    """Test Dice coefficient with high precision"""
    # Create nearly identical tensors
    pred = torch.ones(2, 1, 4, 4)
    target = torch.ones(2, 1, 4, 4) * 0.9999

    dice = _dice_coefficient(pred, target)

    # Dice should be very close to 1
    assert dice[0] > 0.99


def test_custom_collate_fn_geo_map_channels():
    """Test that _custom_collate_fn correctly processes 8 channels geo_map"""
    img = torch.rand(3, 64, 64)
    target = {
        "score_map": torch.rand(1, 16, 16),
        "geo_map": torch.rand(8, 16, 16),  # 8 channels for QUAD offsets
        "quads": torch.rand(5, 8),
    }

    batch = [(img, target)]

    images, targets = _custom_collate_fn(batch)

    # geo_map should have 8 channels
    assert targets["geo_map"].shape[1] == 8
