import pytest
from pathlib import Path
import numpy as np

from manuscript.api.detector import BaseDetector
from manuscript.detectors import EAST


def test_east_inherits_base_detector():
    assert issubclass(EAST, BaseDetector)

class TestEASTInitialization:
    """Tests initialization EAST"""

    def test_initialization_default_parameters(self):
        """Test initialization with default parameters"""
        detector = EAST()

        # Check basic attributes
        assert detector is not None
        assert hasattr(detector, "predict")
        assert hasattr(detector, "onnx_session")
        assert detector.target_size == 1280
        assert detector.score_thresh == 0.6
        assert detector.expand_ratio_w == 1.4
        assert detector.expand_ratio_h == 1.5

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters"""
        detector = EAST(
            target_size=640,
            score_thresh=0.8,
            expand_ratio_w=1.0,
            expand_ratio_h=1.0,
            iou_threshold=0.1,
        )

        assert detector.target_size == 640
        assert detector.score_thresh == 0.8
        assert detector.expand_ratio_w == 1.0
        assert detector.expand_ratio_h == 1.0
        assert detector.iou_threshold == 0.1

    def test_initialization_device_auto_selection(self):
        """Test automatic device selection"""
        detector = EAST()
        
        # device should be cuda or cpu
        assert detector.device in ["cuda", "cpu"]

    def test_initialization_device_explicit_cpu(self):
        """Test explicit selection of CPU"""
        detector = EAST(device="cpu")
        assert detector.device == "cpu"

    def test_initialization_nonexistent_weights(self, local_east_artifacts):
        """Test errors with nonexistent weights"""
        local_east_artifacts.side_effect = ValueError("unknown model")
        with pytest.raises(ValueError):
            EAST(weights="nonexistent_model.onnx", config=Path("nonexistent_model.onnx").with_suffix(".json"))

    def test_initialization_resolves_default_preset(self, local_east_artifacts):
        """Test that model is downloaded if path not specified"""
        # Model should be downloaded or already downloaded
        detector = EAST()
        
        local_east_artifacts.assert_called_once_with("east_50_g1", "EAST", force_download=False)

        # Check that weights path exists
        assert detector.weights is not None
        assert Path(detector.weights).exists()
        
        # Check that default model exists in cache
        default_path = Path(detector.weights)
        assert default_path.exists()

class TestEASTScalingMethods:
    """Tests for scaling methods и трансформации"""

    def test_scale_boxes_to_original_empty(self):
        """Test scaling empty array boxes"""
        detector = EAST(target_size=1280)
        boxes = np.array([])
        
        scaled = detector._scale_boxes_to_original(boxes, (480, 640))
        
        assert len(scaled) == 0

    def test_scale_boxes_to_original_basic(self):
        """Test basic scaling"""
        detector = EAST(target_size=1280)
        
        # Single box: 8 coordinates + 1 score
        boxes = np.array([
            [100, 100, 200, 100, 200, 200, 100, 200, 0.9]
        ])
        
        # Original image 480x640
        scaled = detector._scale_boxes_to_original(boxes, (480, 640))
        
        # Check that coordinates have changed
        assert not np.allclose(scaled[:, :8], boxes[:, :8])
        # Score should not change
        assert scaled[0, 8] == 0.9

    def test_scale_boxes_to_original_correct_scaling(self):
        """Test correctness of scaling"""
        detector = EAST(target_size=1000)
        
        # Box on a 1000x1000 image at point (100, 100)
        boxes = np.array([
            [100, 100, 200, 100, 200, 200, 100, 200, 0.9]
        ])
        
        # Scale to an image of size 500x1000
        # scale_x = 1000/1000 = 1.0, scale_y = 500/1000 = 0.5
        scaled = detector._scale_boxes_to_original(boxes, (500, 1000))
        
        # X coordinates do not change (scale_x=1.0)
        assert scaled[0, 0] == 100
        # Y coordinates are halved (scale_y=0.5)
        assert scaled[0, 1] == 50

    def test_convert_to_axis_aligned_empty(self):
        """Test conversion of an empty array"""
        detector = EAST()
        quads = np.array([])
        
        aligned = detector._convert_to_axis_aligned(quads)
        
        assert len(aligned) == 0

    def test_convert_to_axis_aligned_basic(self):
        """Test conversion of a square to an axis-aligned rectangle"""
        detector = EAST()
        
        # Rotated square
        quads = np.array([
            [50, 100, 150, 50, 200, 150, 100, 200, 0.9]
        ])
        
        aligned = detector._convert_to_axis_aligned(quads)
        
        # Should become axis-aligned (rectangle aligned with axes)
        coords = aligned[0, :8].reshape(4, 2)
        
        # Check that this is a rectangle with parallel sides
        # Two points should have the same x (left side)
        # Two points should have the same x (right side)
        x_coords = sorted(coords[:, 0])
        assert x_coords[0] == x_coords[1]  # Left side
        assert x_coords[2] == x_coords[3]  # Right side

    def test_convert_to_axis_aligned_multiple_quads(self):
        """Test conversion of multiple quads"""
        detector = EAST()
        
        quads = np.array([
            [0, 0, 100, 0, 100, 100, 0, 100, 0.9],
            [200, 200, 300, 200, 300, 300, 200, 300, 0.8],
        ])
        
        aligned = detector._convert_to_axis_aligned(quads)
        
        assert aligned.shape == quads.shape
        # Scores do not change
        assert aligned[0, 8] == 0.9
        assert aligned[1, 8] == 0.8

class TestEASTGeometricUtils:
    """Tests for geometric utilities"""

    def test_polygon_area_batch_empty(self):
        """Test calculation of area for an empty array"""
        areas = EAST._polygon_area_batch(np.array([]))
        assert len(areas) == 0

    def test_polygon_area_batch_square(self):
        """Test calculation of area for a square"""
        # Square 100x100
        polys = np.array([
            [[0, 0], [100, 0], [100, 100], [0, 100]]
        ])
        
        areas = EAST._polygon_area_batch(polys)
        
        assert len(areas) == 1
        assert abs(areas[0] - 10000) < 1  # Area ~10000

    def test_polygon_area_batch_multiple(self):
        """Test calculation of areas for multiple polygons"""
        polys = np.array([
            [[0, 0], [10, 0], [10, 10], [0, 10]],  # 100
            [[0, 0], [20, 0], [20, 20], [0, 20]],  # 400
        ])
        
        areas = EAST._polygon_area_batch(polys)
        
        assert len(areas) == 2
        assert abs(areas[0] - 100) < 1
        assert abs(areas[1] - 400) < 1

    def test_is_quad_inside_true(self):
        """Test that a small quad is inside a big one"""
        detector = EAST()
        
        inner = np.array([[10, 10], [20, 10], [20, 20], [10, 20]])
        outer = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        
        result = detector._is_quad_inside(inner, outer)
        
        assert result is True

    def test_is_quad_inside_false(self):
        """Test that a quad is outside"""
        detector = EAST()
        
        inner = np.array([[150, 150], [160, 150], [160, 160], [150, 160]])
        outer = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        
        result = detector._is_quad_inside(inner, outer)
        
        assert result is False

    def test_remove_fully_contained_boxes_empty(self):
        """Test removal of nested boxes from an empty array"""
        detector = EAST()
        quads = np.array([])
        
        result = detector._remove_fully_contained_boxes(quads)
        
        assert len(result) == 0

    def test_remove_fully_contained_boxes_single(self):
        """Test with a single box"""
        detector = EAST()
        quads = np.array([
            [0, 0, 100, 0, 100, 100, 0, 100, 0.9]
        ])
        
        result = detector._remove_fully_contained_boxes(quads)
        
        assert len(result) == 1

    def test_remove_fully_contained_boxes_nested(self):
        """Test removal of nested box"""
        detector = EAST()
        
        # Big box and small one inside
        quads = np.array([
            [0, 0, 100, 0, 100, 100, 0, 100, 0.9],      # Big
            [10, 10, 20, 10, 20, 20, 10, 20, 0.8],      # Small inside
        ])
        
        result = detector._remove_fully_contained_boxes(quads)
        
        # Only the big one should remain
        assert len(result) == 1

    def test_remove_area_anomalies_disabled(self):
        """Test that anomalies are not removed if the flag is disabled"""
        detector = EAST(remove_area_anomalies=False)
        
        quads = np.array([
            [0, 0, 10, 0, 10, 10, 0, 10, 0.9],
            [0, 0, 1000, 0, 1000, 1000, 0, 1000, 0.9],  # Huge
        ])
        
        result = detector._remove_area_anomalies(quads)
        
        # All remain
        assert len(result) == 2

    def test_remove_area_anomalies_too_few_boxes(self):
        """Test that anomalies are not removed if there are too few boxes"""
        detector = EAST(remove_area_anomalies=True, anomaly_min_box_count=10)
        
        quads = np.array([
            [0, 0, 10, 0, 10, 10, 0, 10, 0.9],
            [0, 0, 1000, 0, 1000, 1000, 0, 1000, 0.9],
        ])
        
        result = detector._remove_area_anomalies(quads)
        
        # All remain (less than 10 boxes)
        assert len(result) == 2

class TestEASTTTA:
    """Tests for Test-Time Augmentation (TTA) functionality."""

    def test_tta_default_disabled(self):
        """Test that TTA is disabled by default."""
        detector = EAST()
        assert detector.use_tta is False
        assert detector.tta_iou_thresh == 0.1

    def test_tta_enabled_initialization(self):
        """Test EAST initialization with TTA enabled."""
        detector = EAST(use_tta=True, tta_iou_thresh=0.15)
        assert detector.use_tta is True
        assert detector.tta_iou_thresh == 0.15

    def test_box_iou_calculation(self):
        """Test IoU calculation between two boxes."""
        detector = EAST()
        
        # Identical boxes - IoU should be 1.0
        box1 = (0, 0, 100, 100)
        iou = detector._box_iou(box1, box1)
        assert abs(iou - 1.0) < 1e-6
        
        # Non-overlapping boxes - IoU should be 0.0
        box1 = (0, 0, 50, 50)
        box2 = (100, 100, 150, 150)
        iou = detector._box_iou(box1, box2)
        assert iou == 0.0
        
        # Partial overlap - 50% overlap
        box1 = (0, 0, 100, 100)
        box2 = (50, 0, 150, 100)
        iou = detector._box_iou(box1, box2)
        # Intersection: 50x100 = 5000
        # Union: 10000 + 10000 - 5000 = 15000
        # IoU: 5000/15000 = 0.333...
        expected_iou = 5000 / 15000
        assert abs(iou - expected_iou) < 1e-6

    def test_merge_tta_boxes_perfect_match(self):
        """Test merging boxes that perfectly match."""
        detector = EAST(use_tta=True, tta_iou_thresh=0.1)
        
        # Same boxes in both views
        boxes_orig = [((10, 10, 100, 50), 0.9)]
        boxes_flipped = [((10, 10, 100, 50), 0.8)]
        
        merged = detector._merge_tta_boxes(boxes_orig, boxes_flipped)
        
        assert len(merged) == 1
        box, score = merged[0]
        assert box == (10, 10, 100, 50)
        assert abs(score - 0.85) < 1e-6  # Average of 0.9 and 0.8

    def test_merge_tta_boxes_no_match(self):
        """Test merging boxes that don't match (no overlap)."""
        detector = EAST(use_tta=True, tta_iou_thresh=0.1)
        
        # Non-overlapping boxes
        boxes_orig = [((10, 10, 50, 50), 0.9)]
        boxes_flipped = [((200, 200, 250, 250), 0.8)]
        
        merged = detector._merge_tta_boxes(boxes_orig, boxes_flipped)
        
        # No matches found - TTA returns only matched boxes
        assert len(merged) == 0

    def test_merge_tta_boxes_partial_overlap(self):
        """Test merging boxes with partial overlap."""
        detector = EAST(use_tta=True, tta_iou_thresh=0.1)
        
        # Overlapping boxes
        boxes_orig = [((0, 0, 100, 100), 0.9)]
        boxes_flipped = [((80, 10, 120, 90), 0.8)]  # Overlapping with different y
        
        # IoU = (20*80) / (100*100 + 40*80 - 20*80) = 1600 / (10000 + 3200 - 1600) = 1600/11600 ≈ 0.138
        # This exceeds 0.1 threshold, so should merge
        merged = detector._merge_tta_boxes(boxes_orig, boxes_flipped)
        
        assert len(merged) == 1
        box, score = merged[0]
        # Merged box: x extended, y from original
        assert box == (0, 0, 120, 100)  # x: min(0,80)=0, max(100,120)=120, y from orig: 0, 100
        assert abs(score - 0.85) < 1e-6



@pytest.fixture(autouse=True)
def local_east_artifacts(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from unittest.mock import Mock
    weights = tmp_path / "model.onnx"
    config = tmp_path / "model.json"
    weights.write_bytes(b"fake")
    config.write_text(json.dumps({"region_classes": {"0": {"type": "word"}}}))
    resolve = Mock(return_value={"weights": weights, "config": config})
    monkeypatch.setattr("manuscript.models.resolve", resolve)
    session = SimpleNamespace(
        get_inputs=lambda: [SimpleNamespace(name="images")],
        get_outputs=lambda: [SimpleNamespace(name="score"), SimpleNamespace(name="geometry")],
        run=lambda *args: [np.zeros((1, 1, 8, 8), np.float32), np.zeros((1, 8, 8, 8), np.float32)],
    )
    monkeypatch.setattr(EAST, "_create_onnx_session", lambda self: session)
    return resolve


@pytest.mark.parametrize("source", ["array", "path", "pil"])
def test_predict_restores_original_coordinates_and_region_type(tmp_path, monkeypatch, source):
    from PIL import Image
    image = np.zeros((50, 100, 3), np.uint8)
    detector = EAST(target_size=100, expand_ratio_w=0, expand_ratio_h=0)
    quad = np.array([[10, 20, 30, 20, 30, 40, 10, 40, .8]], np.float32)
    monkeypatch.setattr(detector, "_run_inference_on_image", lambda image: (quad, None, None))
    if source == "path":
        path = tmp_path / "image.png"
        Image.fromarray(image).save(path)
        image = path
    elif source == "pil": image = Image.fromarray(image)
    page = detector.predict(image)
    assert (page.image_width, page.image_height) == (100, 50)
    assert len(page.children) == 1 and page.children[0].type == "word"
    np.testing.assert_allclose(page.children[0].polygon, [[10, 10], [30, 10], [30, 20], [10, 20]])
    assert page.children[0].detection_confidence == pytest.approx(.8)


def test_score_threshold_changes_decoded_detections():
    detector = EAST(target_size=32, expand_ratio_w=0, expand_ratio_h=0, score_thresh=.6)
    detector._initialize_session()
    score = np.zeros((1, 1, 8, 8), np.float32)
    score[0, 0, 3, 3] = .7
    geometry = np.zeros((1, 8, 8, 8), np.float32)
    geometry[0, :, 3, 3] = [-1, -1, 1, -1, 1, 1, -1, 1]
    detector.onnx_session.run = lambda *args: [score, geometry]
    image = np.zeros((32, 32, 3), np.uint8)
    assert len(detector.predict(image).children) == 1
    detector.score_thresh = .8
    assert not detector.predict(image).children


@pytest.mark.parametrize("matches", [False, True])
def test_tta_mirrors_coordinates_and_retains_only_matched_quads(monkeypatch, matches):
    detector = EAST(target_size=100, use_tta=True, expand_ratio_w=0, expand_ratio_h=0)
    image = np.arange(50 * 100 * 3, dtype=np.uint8).reshape(50, 100, 3)
    original = np.array([[10, 20, 30, 20, 30, 40, 10, 40, .9]], np.float32)
    flipped = np.array([[70, 20, 90, 20, 90, 40, 70, 40, .8]], np.float32) if matches else np.empty((0, 9), np.float32)
    calls = []
    def infer(pixels):
        calls.append(pixels.copy())
        return (original if len(calls) == 1 else flipped, None, None)
    monkeypatch.setattr(detector, "_run_inference_on_image", infer)
    page = detector.predict(image)
    np.testing.assert_array_equal(calls[1], image[:, ::-1])
    assert len(page.children) == int(matches)
    if matches:
        np.testing.assert_allclose(page.children[0].polygon, [[10, 10], [30, 10], [30, 20], [10, 20]])


def test_unknown_preset_is_not_silently_replaced(local_east_artifacts):
    local_east_artifacts.side_effect = ValueError("unknown model")
    with pytest.raises(ValueError, match="unknown model"):
        EAST(weights="unknown")


@pytest.mark.parametrize("source", [object(), "/nonexistent/image.png"])
def test_invalid_image_source_fails(source):
    with pytest.raises((TypeError, FileNotFoundError)):
        EAST(target_size=32).predict(source)


def test_configured_line_level_is_used_in_prediction(monkeypatch):
    detector = EAST(target_size=32, config={"region_classes": {"0": {"type": "line"}}},
                    expand_ratio_w=0, expand_ratio_h=0)
    quad = np.array([[4, 4, 12, 4, 12, 12, 4, 12, .8]], np.float32)
    monkeypatch.setattr(detector, "_run_inference_on_image", lambda image: (quad, None, None))
    page = detector.predict(np.zeros((32, 32, 3), np.uint8))
    assert page.children[0].type == "line"
