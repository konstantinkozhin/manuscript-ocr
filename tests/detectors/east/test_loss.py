import pytest

torch = pytest.importorskip("torch")
from manuscript.detectors._east.loss import compute_dice_loss, EASTLoss
TORCH_AVAILABLE = True


# --- Tests for compute_dice_loss ---
def test_compute_dice_loss_perfect_match():
    """Test Dice loss for a perfect match"""
    gt = torch.ones(2, 1, 4, 4)
    pred = torch.ones(2, 1, 4, 4)

    loss = compute_dice_loss(gt, pred)

    # For a perfect match, loss should be close to 0
    assert loss < 0.01


def test_compute_dice_loss_no_overlap():
    """Test Dice loss for no overlap"""
    gt = torch.zeros(2, 1, 4, 4)
    gt[0, 0, 0, 0] = 1.0
    pred = torch.zeros(2, 1, 4, 4)
    pred[0, 0, 3, 3] = 1.0

    loss = compute_dice_loss(gt, pred)

    # For no overlap, loss should be close to 1
    assert loss > 0.5


def test_compute_dice_loss_partial_overlap():
    """Test Dice loss for partial overlap"""
    gt = torch.zeros(1, 1, 4, 4)
    gt[0, 0, :2, :2] = 1.0  # Top-left quadrant

    pred = torch.zeros(1, 1, 4, 4)
    pred[0, 0, 1:3, 1:3] = 1.0  # Overlaps with gt

    loss = compute_dice_loss(gt, pred)

    # Should be between 0 and 1
    assert 0 < loss < 1


def test_compute_dice_loss_stability():
    """Test Dice loss stability (protection against division by zero)"""
    gt = torch.zeros(1, 1, 4, 4)
    pred = torch.zeros(1, 1, 4, 4)

    loss = compute_dice_loss(gt, pred)

    # Should work without errors due to epsilon
    assert torch.isfinite(loss)


def test_compute_dice_loss_different_scales():
    """Test Dice loss with different intensity values"""
    gt = torch.ones(1, 1, 4, 4) * 0.5
    pred = torch.ones(1, 1, 4, 4) * 0.5

    loss = compute_dice_loss(gt, pred)

    # Dice loss depends on absolute values, not just overlap
    # inter = 0.5*0.5*16 = 4, union = 0.5*16 + 0.5*16 + eps = 16 + eps
    # loss = 1 - 2*4/(16+eps) = 1 - 0.5 = 0.5
    assert abs(loss - 0.5) < 0.01


# --- Tests for EASTLoss initialization ---
def test_east_loss_initialization_default():
    """Test EASTLoss initialization with default parameters"""
    loss_fn = EASTLoss()

    assert loss_fn.use_ohem == False
    assert loss_fn.ohem_ratio == 0.5
    assert loss_fn.use_focal_geo == False
    assert loss_fn.focal_gamma == 2.0


def test_east_loss_initialization_ohem():
    """Test EASTLoss initialization with OHEM"""
    loss_fn = EASTLoss(use_ohem=True, ohem_ratio=0.3)

    assert loss_fn.use_ohem == True
    assert loss_fn.ohem_ratio == 0.3


def test_east_loss_initialization_focal():
    """Test EASTLoss initialization with Focal loss"""
    loss_fn = EASTLoss(use_focal_geo=True, focal_gamma=3.0)

    assert loss_fn.use_focal_geo == True
    assert loss_fn.focal_gamma == 3.0


# --- Tests for EASTLoss forward ---
def test_east_loss_forward_basic():
    """Test basic forward pass for EASTLoss"""
    loss_fn = EASTLoss()

    gt_score = torch.ones(2, 1, 8, 8)
    pred_score = torch.ones(2, 1, 8, 8) * 0.9
    gt_geo = torch.ones(2, 8, 8, 8)
    pred_geo = torch.ones(2, 8, 8, 8) * 0.95

    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert loss >= 0


def test_east_loss_forward_no_positives():
    """Test forward when there are no positive pixels"""
    loss_fn = EASTLoss()

    gt_score = torch.zeros(2, 1, 8, 8)  # No positives
    pred_score = torch.ones(2, 1, 8, 8)
    gt_geo = torch.ones(2, 8, 8, 8)
    pred_geo = torch.ones(2, 8, 8, 8)

    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)

    # Should return 0 with requires_grad=True
    assert loss == 0.0
    assert loss.requires_grad


def test_east_loss_forward_perfect_prediction():
    """Test forward with perfect prediction"""
    loss_fn = EASTLoss()

    gt_score = torch.ones(1, 1, 4, 4)
    pred_score = torch.ones(1, 1, 4, 4)
    gt_geo = torch.ones(1, 8, 4, 4)
    pred_geo = torch.ones(1, 8, 4, 4)

    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)

    # With perfect prediction, loss should be close to 0
    assert loss < 0.1


def test_east_loss_forward_with_ohem():
    """Test forward with OHEM"""
    loss_fn = EASTLoss(use_ohem=True, ohem_ratio=0.5)

    gt_score = torch.ones(2, 1, 8, 8)
    pred_score = torch.rand(2, 1, 8, 8)
    gt_geo = torch.ones(2, 8, 8, 8)
    pred_geo = torch.rand(2, 8, 8, 8)

    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)

    assert isinstance(loss, torch.Tensor)
    assert loss >= 0


def test_east_loss_forward_with_focal():
    """Test forward with Focal geometry loss"""
    loss_fn = EASTLoss(use_focal_geo=True, focal_gamma=2.0)

    gt_score = torch.ones(2, 1, 8, 8)
    pred_score = torch.rand(2, 1, 8, 8)
    gt_geo = torch.ones(2, 8, 8, 8)
    pred_geo = torch.rand(2, 8, 8, 8)

    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)

    assert isinstance(loss, torch.Tensor)
    assert loss >= 0


def test_east_loss_forward_with_ohem_and_focal():
    """Test forward with OHEM and Focal simultaneously"""
    loss_fn = EASTLoss(
        use_ohem=True, ohem_ratio=0.3, use_focal_geo=True, focal_gamma=3.0
    )

    gt_score = torch.ones(2, 1, 8, 8)
    pred_score = torch.rand(2, 1, 8, 8)
    gt_geo = torch.ones(2, 8, 8, 8)
    pred_geo = torch.rand(2, 8, 8, 8)

    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)

    assert isinstance(loss, torch.Tensor)
    assert loss >= 0


def test_east_loss_gradient_flow():
    """Test that gradients flow through the loss"""
    loss_fn = EASTLoss()

    gt_score = torch.ones(1, 1, 4, 4)
    pred_score = torch.rand(1, 1, 4, 4, requires_grad=True)
    gt_geo = torch.ones(1, 8, 4, 4)
    pred_geo = torch.rand(1, 8, 4, 4, requires_grad=True)

    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
    loss.backward()

    # Check that gradients exist
    assert pred_score.grad is not None
    assert pred_geo.grad is not None
    assert pred_score.grad.abs().sum() > 0
    assert pred_geo.grad.abs().sum() > 0


def test_east_loss_partial_gt_mask():
    """Test forward with partial gt_score mask"""
    loss_fn = EASTLoss()

    gt_score = torch.zeros(2, 1, 8, 8)
    gt_score[:, :, :4, :4] = 1.0  # Only top-left quadrant
    pred_score = torch.rand(2, 1, 8, 8)
    gt_geo = torch.ones(2, 8, 8, 8)
    pred_geo = torch.rand(2, 8, 8, 8)

    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)

    assert loss > 0


def test_east_loss_different_batch_sizes():
    """Test forward with different batch sizes"""
    loss_fn = EASTLoss()

    for batch_size in [1, 2, 4, 8]:
        gt_score = torch.ones(batch_size, 1, 4, 4)
        pred_score = torch.rand(batch_size, 1, 4, 4)
        gt_geo = torch.ones(batch_size, 8, 4, 4)
        pred_geo = torch.rand(batch_size, 8, 4, 4)

        loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
        assert loss >= 0


def test_east_loss_large_prediction_error():
    """Test forward with large prediction error"""
    loss_fn = EASTLoss()

    gt_score = torch.ones(1, 1, 4, 4)
    pred_score = torch.zeros(1, 1, 4, 4)
    gt_geo = torch.ones(1, 8, 4, 4)
    pred_geo = torch.zeros(1, 8, 4, 4)

    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)

    # With large prediction error, loss should be significant
    assert loss > 1.0


def test_east_loss_ohem_ratio_extreme_values():
    """Test OHEM with extreme ratio values"""
    # Very low ratio
    loss_fn_low = EASTLoss(use_ohem=True, ohem_ratio=0.01)

    gt_score = torch.ones(2, 1, 8, 8)
    pred_score = torch.rand(2, 1, 8, 8)
    gt_geo = torch.ones(2, 8, 8, 8)
    pred_geo = torch.rand(2, 8, 8, 8)

    loss_low = loss_fn_low(gt_score, pred_score, gt_geo, pred_geo)
    assert torch.isfinite(loss_low)

    # Very high ratio
    loss_fn_high = EASTLoss(use_ohem=True, ohem_ratio=0.99)
    loss_high = loss_fn_high(gt_score, pred_score, gt_geo, pred_geo)
    assert torch.isfinite(loss_high)


def test_east_loss_focal_gamma_variations():
    """Test Focal loss with different gamma values"""
    gt_score = torch.ones(1, 1, 8, 8)
    pred_score = torch.rand(1, 1, 8, 8)
    gt_geo = torch.ones(1, 8, 8, 8)
    pred_geo = torch.rand(1, 8, 8, 8)

    losses = []
    for gamma in [0.5, 1.0, 2.0, 5.0]:
        loss_fn = EASTLoss(use_focal_geo=True, focal_gamma=gamma)
        loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
        losses.append(loss.item())
        assert torch.isfinite(loss)

    # All losses should be valid
    assert all(l >= 0 for l in losses)


def test_east_loss_geo_channels_independence():
    """Test that all 8 geometry channels are considered"""
    loss_fn = EASTLoss()

    gt_score = torch.ones(1, 1, 4, 4)
    pred_score = torch.ones(1, 1, 4, 4)
    gt_geo = torch.ones(1, 8, 4, 4)

    # Prediction with error in only one channel
    pred_geo_one_channel = torch.ones(1, 8, 4, 4)
    pred_geo_one_channel[:, 0, :, :] = 0  # Error in the first channel

    loss_one = loss_fn(gt_score, pred_score, gt_geo, pred_geo_one_channel)

    # Prediction with error in all channels
    pred_geo_all_channels = torch.zeros(1, 8, 4, 4)

    loss_all = loss_fn(gt_score, pred_score, gt_geo, pred_geo_all_channels)

    # Loss with error in all channels should be greater
    assert loss_all > loss_one


def test_east_loss_device_cuda():
    """Test that loss works on CUDA (if available)"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    loss_fn = EASTLoss().cuda()

    gt_score = torch.ones(1, 1, 4, 4).cuda()
    pred_score = torch.rand(1, 1, 4, 4).cuda()
    gt_geo = torch.ones(1, 8, 4, 4).cuda()
    pred_geo = torch.rand(1, 8, 4, 4).cuda()

    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)

    assert loss.device.type == "cuda"


def test_east_loss_device_cpu():
    """Test that loss works on CPU"""
    loss_fn = EASTLoss()

    gt_score = torch.ones(1, 1, 4, 4)
    pred_score = torch.rand(1, 1, 4, 4)
    gt_geo = torch.ones(1, 8, 4, 4)
    pred_geo = torch.rand(1, 8, 4, 4)

    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)

    assert loss.device.type == "cpu"


def test_east_loss_numerical_stability():
    """Test numerical stability with extreme values"""
    loss_fn = EASTLoss()

    # Very large values
    gt_score = torch.ones(1, 1, 4, 4) * 1e6
    pred_score = torch.ones(1, 1, 4, 4) * 1e6
    gt_geo = torch.ones(1, 8, 4, 4) * 1e6
    pred_geo = torch.ones(1, 8, 4, 4) * 1e6

    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
    assert torch.isfinite(loss)

    # Very small values
    gt_score_small = torch.ones(1, 1, 4, 4) * 1e-6
    pred_score_small = torch.ones(1, 1, 4, 4) * 1e-6
    gt_geo_small = torch.ones(1, 8, 4, 4) * 1e-6
    pred_geo_small = torch.ones(1, 8, 4, 4) * 1e-6

    loss_small = loss_fn(gt_score_small, pred_score_small, gt_geo_small, pred_geo_small)
    assert torch.isfinite(loss_small)


def test_east_loss_single_positive_pixel():
    """Test with a single positive pixel"""
    loss_fn = EASTLoss()

    gt_score = torch.zeros(1, 1, 8, 8)
    gt_score[0, 0, 4, 4] = 1.0  # Only one pixel
    pred_score = torch.rand(1, 1, 8, 8)
    gt_geo = torch.ones(1, 8, 8, 8)
    pred_geo = torch.rand(1, 8, 8, 8)

    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)

    assert torch.isfinite(loss)
    assert loss >= 0


def test_east_loss_ohem_single_batch():
    """Test OHEM with batch_size=1"""
    loss_fn = EASTLoss(use_ohem=True, ohem_ratio=0.5)

    gt_score = torch.ones(1, 1, 8, 8)
    pred_score = torch.rand(1, 1, 8, 8)
    gt_geo = torch.ones(1, 8, 8, 8)
    pred_geo = torch.rand(1, 8, 8, 8)

    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)

    assert torch.isfinite(loss)


def test_east_loss_ohem_multi_batch():
    """Test OHEM with multiple batches"""
    loss_fn = EASTLoss(use_ohem=True, ohem_ratio=0.5)

    gt_score = torch.ones(4, 1, 8, 8)
    pred_score = torch.rand(4, 1, 8, 8)
    gt_geo = torch.ones(4, 8, 8, 8)
    pred_geo = torch.rand(4, 8, 8, 8)

    loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)

    assert torch.isfinite(loss)


def test_compute_dice_loss_gradient():
    """Test that gradients flow through dice loss"""
    gt = torch.ones(1, 1, 4, 4)
    pred = torch.rand(1, 1, 4, 4, requires_grad=True)

    loss = compute_dice_loss(gt, pred)
    loss.backward()

    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0


def test_east_loss_comparison_ohem_vs_standard():
    """Test comparison of OHEM vs standard loss"""
    gt_score = torch.ones(2, 1, 8, 8)
    pred_score = torch.rand(2, 1, 8, 8)
    gt_geo = torch.ones(2, 8, 8, 8)
    pred_geo = torch.rand(2, 8, 8, 8)

    loss_standard = EASTLoss(use_ohem=False)
    loss_ohem = EASTLoss(use_ohem=True, ohem_ratio=0.5)

    loss_std_val = loss_standard(gt_score, pred_score, gt_geo, pred_geo)
    loss_ohem_val = loss_ohem(gt_score, pred_score, gt_geo, pred_geo)

    # Both should be valid
    assert torch.isfinite(loss_std_val)
    assert torch.isfinite(loss_ohem_val)

    # Values may differ
    assert loss_std_val >= 0
    assert loss_ohem_val >= 0


def test_east_loss_different_spatial_sizes():
    """Test with different spatial sizes"""
    loss_fn = EASTLoss()

    sizes = [(4, 4), (8, 8), (16, 16), (32, 32)]

    for h, w in sizes:
        gt_score = torch.ones(1, 1, h, w)
        pred_score = torch.rand(1, 1, h, w)
        gt_geo = torch.ones(1, 8, h, w)
        pred_geo = torch.rand(1, 8, h, w)

        loss = loss_fn(gt_score, pred_score, gt_geo, pred_geo)
        assert torch.isfinite(loss)
        assert loss >= 0
