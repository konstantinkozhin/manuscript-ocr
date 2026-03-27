import pytest
from pathlib import Path
import numpy as np
import cv2

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from manuscript.api.detector import BaseDetector
from manuscript.detectors import EAST
from manuscript.data import Page


def test_east_inherits_base_detector():
    assert issubclass(EAST, BaseDetector)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
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

    def test_initialization_nonexistent_weights(self):
        """Test errors with nonexistent weights"""
        with pytest.raises(ValueError):
            EAST(weights="nonexistent_model.onnx")

    def test_initialization_downloads_default_model(self):
        """Test that model is downloaded if path not specified"""
        # Model should be downloaded or already downloaded
        detector = EAST()
        
        # Check that weights path exists
        assert detector.weights is not None
        assert Path(detector.weights).exists()
        
        # Check that default model exists in cache
        default_path = Path.home() / ".manuscript" / "weights" / "east_50_g1.onnx"
        assert default_path.exists()

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
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

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
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
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
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
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
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

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestEASTPredict:
    """Tests for the predict method"""

    @pytest.fixture
    def test_image(self):
        """Creates a test image"""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    @pytest.fixture
    def test_image_path(self, tmp_path, test_image):
        """Creates a temporary file with an image"""
        img_path = tmp_path / "test_image.jpg"
        cv2.imwrite(str(img_path), cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
        return str(img_path)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_predict_with_path(self, test_image_path):
        """Test predict with file path"""
        detector = EAST()
        
        result = detector.predict(test_image_path)
        
        assert isinstance(result, Page)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_predict_with_numpy_array(self, test_image):
        """Test predict with numpy array"""
        detector = EAST()
        
        result = detector.predict(test_image)
        
        assert isinstance(result, Page)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_predict_returns_page_structure(self, test_image):
        """Test that predict returns correct Page structure"""
        detector = EAST()
        
        result = detector.predict(test_image)
        page = result
        
        assert hasattr(page, "blocks")
        assert isinstance(page.blocks, list)
        
        if len(page.blocks) > 0:
            block = page.blocks[0]
            assert hasattr(block, "lines")
            assert isinstance(block.lines, list)
            if len(block.lines) > 0:
                assert isinstance(block.lines[0].text_spans, list)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_predict_different_thresholds(self, test_image):
        """Test with different thresholds"""
        detector_low = EAST(score_thresh=0.1)
        detector_high = EAST(score_thresh=0.9)
        
        result_low = detector_low.predict(test_image)
        result_high = detector_high.predict(test_image)
        
        # Both should work without errors
        assert isinstance(result_low, Page)
        assert isinstance(result_high, Page)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_predict_axis_aligned_vs_original(self, test_image):
        """Test axis_aligned_output flag"""
        detector_aligned = EAST(axis_aligned_output=True)
        detector_original = EAST(axis_aligned_output=False)
        
        result_aligned = detector_aligned.predict(test_image)
        result_original = detector_original.predict(test_image)
        
        # Both should return results
        assert isinstance(result_aligned, Page)
        assert isinstance(result_original, Page)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestEASTErrorHandling:
    """Tests error handling"""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_predict_nonexistent_file(self):
        """Test errors with nonexistent file"""
        detector = EAST()
        
        with pytest.raises(FileNotFoundError):
            detector.predict("nonexistent_image.jpg")
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_predict_invalid_type(self):
        """Test errors with invalid input type"""
        detector = EAST()
        
        with pytest.raises((TypeError, AttributeError)):
            detector.predict(12345)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_predict_corrupted_image_path(self, tmp_path):
        """Test with a corrupted image file"""
        # Create a text file instead of an image
        fake_img = tmp_path / "fake.jpg"
        fake_img.write_text("not an image")
        
        detector = EAST()
        
        # Should raise an error when reading
        with pytest.raises((cv2.error, ValueError, OSError)):
            detector.predict(str(fake_img))

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestEASTEdgeCases:
    """Tests edge cases"""
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_predict_very_small_image(self):
        """Test with very small image"""
        detector = EAST()
        
        # Image 10x10
        small_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        
        result = detector.predict(small_img)
        
        # Should not raise errors
        assert isinstance(result, Page)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_predict_very_large_image(self):
        """Test with very large image"""
        detector = EAST()
        
        # Image 2000x3000
        large_img = np.random.randint(0, 255, (2000, 3000, 3), dtype=np.uint8)
        
        result = detector.predict(large_img)
        
        # Should not raise errors
        assert isinstance(result, Page)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_predict_grayscale_image(self):
        """Test with grayscale image"""
        detector = EAST()
        
        # Grayscale image (2D)
        gray_img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        
        # Should raise an error or automatically convert
        try:
            result = detector.predict(gray_img)
            # If it worked, check the result
            assert isinstance(result, Page)
        except (ValueError, AttributeError):
            # Expected error for grayscale
            pass
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_predict_black_image(self):
        """Test with completely black image"""
        detector = EAST()
        
        black_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        page = detector.predict(black_img)
        
        # Most likely there will be no detections, but there should be no errors
        assert isinstance(page, Page)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_predict_white_image(self):
        """Test with completely white image"""
        detector = EAST()
        
        white_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        
        page = detector.predict(white_img)
        
        assert isinstance(page, Page)
        
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_predict_with_extreme_expand_ratios(self):
        """Test with extreme expand ratios"""
        detector = EAST(expand_ratio_w=0.0, expand_ratio_h=0.0)
        
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result = detector.predict(test_img)
        
        # Should not raise errors even with zero expansion
        assert isinstance(result, Page)

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestEASTIntegration:
    """Integration tests with real image"""

    @pytest.fixture
    def example_image_path(self):
        """Path to example image if it exists"""
        repo_root = Path(__file__).parent.parent.parent.parent
        image_path = repo_root / "example" / "ocr_example_image.jpg"
        
        if not image_path.exists():
            pytest.skip("Test image example/ocr_example_image.jpg not found")
        
        return str(image_path)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_different_target_sizes(self, example_image_path):
        """Test with different shapes target_size"""
        sizes = [640, 1280, 1920]
        
        for size in sizes:
            detector = EAST(target_size=size)
            result = detector.predict(example_image_path)
            
            # All should run without errors
            assert isinstance(result, Page)

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
    def test_consistency_across_runs(self):
        """Test consistency across runs"""
        detector = EAST(score_thresh=0.5)
        
        # Create a fixed image
        np.random.seed(42)
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Run twice
        result1 = detector.predict(img)
        result2 = detector.predict(img)
        
        # The number of detections should be the same
        words1 = sum(len(line.text_spans) for b in result1.blocks for line in b.lines)
        words2 = sum(len(line.text_spans) for b in result2.blocks for line in b.lines)
        
        assert words1 == words2


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
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

    def test_tta_predict_runs_without_error(self):
        """Test that predict with TTA runs without errors."""
        detector = EAST(use_tta=True, tta_iou_thresh=0.1)
        
        # Create a simple test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some white rectangles to simulate text
        img[100:150, 100:300] = 255
        img[200:250, 150:350] = 255
        
        result = detector.predict(img)
        
        assert isinstance(result, Page)

    def test_tta_consistency(self):
        """Test that TTA produces consistent results."""
        detector = EAST(use_tta=True, score_thresh=0.5)
        
        np.random.seed(42)
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result1 = detector.predict(img)
        result2 = detector.predict(img)
        
        words1 = sum(len(line.text_spans) for block in result1.blocks for line in block.lines)
        words2 = sum(len(line.text_spans) for block in result2.blocks for line in block.lines)
        
        assert words1 == words2

    def test_tta_vs_no_tta(self):
        """Test that TTA and non-TTA can produce different results."""
        # This is more of a sanity check - TTA might or might not find more boxes
        detector_tta = EAST(use_tta=True, score_thresh=0.5)
        detector_no_tta = EAST(use_tta=False, score_thresh=0.5)
        
        # Create image with content
        np.random.seed(42)
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result_tta = detector_tta.predict(img)
        result_no_tta = detector_no_tta.predict(img)
        
        # Both should return valid Page objects
        assert isinstance(result_tta, Page)
        assert isinstance(result_no_tta, Page)

    def test_tta_with_axis_aligned_false(self):
        """Test that TTA works correctly with axis_aligned_output=False."""
        detector = EAST(use_tta=True, axis_aligned_output=False, score_thresh=0.5)
        
        # Create a simple test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img[100:150, 100:300] = 255
        
        result = detector.predict(img)
        
        assert isinstance(result, Page)
        
        # Check that polygons are preserved (not converted to axis-aligned rectangles)
        # With axis_aligned_output=False, we should get the original 4-point polygons
        for block in result.blocks:
            for line in block.lines:
                for text_span in line.text_spans:
                    # Each polygon should have 4 points
                    assert len(text_span.polygon) == 4
                    # Each point should have 2 coordinates
                    for point in text_span.polygon:
                        assert len(point) == 2

    def test_tta_axis_aligned_true_vs_false(self):
        """Test that axis_aligned_output affects TTA output correctly."""
        detector_aligned = EAST(use_tta=True, axis_aligned_output=True, score_thresh=0.5)
        detector_not_aligned = EAST(use_tta=True, axis_aligned_output=False, score_thresh=0.5)
        
        # Create test image
        np.random.seed(42)
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        result_aligned = detector_aligned.predict(img)
        result_not_aligned = detector_not_aligned.predict(img)
        
        # Both should return valid results
        assert isinstance(result_aligned, Page)
        assert isinstance(result_not_aligned, Page)
        
        # Get all words from both results
        def get_all_words(page):
            return [text_span for block in page.blocks for line in block.lines for text_span in line.text_spans]
        
        words_aligned = get_all_words(result_aligned)
        words_not_aligned = get_all_words(result_not_aligned)
        
        # Should have the same number of words
        assert len(words_aligned) == len(words_not_aligned)




