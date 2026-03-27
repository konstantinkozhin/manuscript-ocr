import pytest
import json
import numpy as np
import cv2
from pathlib import Path

try:
    import torch
    from manuscript.detectors._east.dataset import (
        order_vertices_clockwise,
        shrink_poly,
        EASTDataset,
    )

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    order_vertices_clockwise = None
    shrink_poly = None
    EASTDataset = None


def _quad(x0, y0, x1, y1):
    return np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        dtype=np.float32,
    )


# ============================================================================
# Tests for geometric functions
# ============================================================================
class TestGeometricFunctions:
    """Tests for helper geometric functions"""

    def test_order_vertices_clockwise_square(self):
        """Test ordering square vertices clockwise"""
        # Unordered square points
        poly = [[100, 100], [100, 0], [0, 0], [0, 100]]
        ordered = order_vertices_clockwise(poly)

        # Check the format
        assert ordered.shape == (4, 2)
        assert ordered.dtype == np.float32

        # Check order: TL, TR, BR, BL
        # top-left should be to the left of top-right
        assert ordered[0][0] < ordered[1][0]
        # bottom-right should be below top-right
        assert ordered[2][1] > ordered[1][1]
        # bottom-left should be to the left of bottom-right
        assert ordered[3][0] < ordered[2][0]

    def test_order_vertices_clockwise_rectangle(self):
        """Test ordering rectangle vertices clockwise"""
        # Rectangle 200x100
        poly = [[0, 0], [200, 0], [200, 100], [0, 100]]
        ordered = order_vertices_clockwise(poly)

        # Check that top-left is first
        assert np.allclose(ordered[0], [0, 0])
        # And bottom-right is in the correct position
        assert np.allclose(ordered[2], [200, 100])

    def test_order_vertices_clockwise_rotated(self):
        """Test ordering rotated square vertices"""
        # Square rotated by 45 degrees
        poly = [[50, 0], [100, 50], [50, 100], [0, 50]]
        ordered = order_vertices_clockwise(poly)

        assert ordered.shape == (4, 2)
        # Ensure vertices are ordered
        assert len(ordered) == 4

    def test_shrink_poly_basic(self):
        """Test basic polygon shrinking"""
        # Square 100x100
        quad = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)

        shrunk = shrink_poly(quad, shrink_ratio=0.3)

        # Check format
        assert shrunk.shape == (4, 2)
        assert shrunk.dtype == np.float32

        # Shrunk square should be smaller than original
        # top-left moved right and down
        assert shrunk[0][0] > quad[0][0]
        assert shrunk[0][1] > quad[0][1]

        # bottom-right moved left and up
        assert shrunk[2][0] < quad[2][0]
        assert shrunk[2][1] < quad[2][1]

    def test_shrink_poly_different_ratios(self):
        """Test shrinking with different ratios"""
        quad = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)

        shrunk_small = shrink_poly(quad, shrink_ratio=0.1)
        shrunk_large = shrink_poly(quad, shrink_ratio=0.5)

        # Larger ratio = more shrinkage
        # Distance from top-left to center should be smaller when shrink_ratio is larger
        center = np.array([50, 50])
        dist_small = np.linalg.norm(shrunk_small[0] - center)
        dist_large = np.linalg.norm(shrunk_large[0] - center)

        assert dist_large < dist_small

    def test_shrink_poly_invalid_vertices(self):
        """Test error with incorrect number of vertices"""
        # Triangle (3 vertices)
        triangle = np.array([[0, 0], [100, 0], [50, 100]], dtype=np.float32)

        with pytest.raises(ValueError, match="Expected quadrilateral with 4 vertices"):
            shrink_poly(triangle)

    def test_shrink_poly_clockwise_order(self):
        """Test shrink works with unordered vertices"""
        # Vertices in random order
        quad = np.array([[100, 0], [100, 100], [0, 100], [0, 0]], dtype=np.float32)

        shrunk = shrink_poly(quad, shrink_ratio=0.2)

        # Should not raise any errors
        assert shrunk.shape == (4, 2)


# ============================================================================
# Tests for EASTDataset
# ============================================================================
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestEASTDataset:
    """Tests for EASTDataset class"""

    @pytest.fixture
    def simple_dataset(self, tmp_path):
        """Creates a simple dataset with one image"""
        annotations = {
            "images": [{"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "segmentation": [[10, 10, 100, 10, 100, 50, 10, 50]],
                }
            ],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.jpg"), img)

        return str(img_dir), str(ann_file)

    def test_east_dataset_initialization(self, simple_dataset):
        """Test basic dataset initialization"""
        img_dir, ann_file = simple_dataset

        dataset = EASTDataset(
            images_folder=img_dir,
            coco_annotation_file=ann_file,
            target_size=512,
            score_geo_scale=0.25,
        )

        assert len(dataset) == 1
        assert dataset.target_size == 512
        assert dataset.score_geo_scale == 0.25
        assert dataset.images_folder == img_dir
        assert dataset.quad_source == "auto"

    def test_east_dataset_invalid_quad_source(self, simple_dataset):
        """Test validation for quad_source mode."""
        img_dir, ann_file = simple_dataset

        with pytest.raises(ValueError, match="quad_source"):
            EASTDataset(
                images_folder=img_dir,
                coco_annotation_file=ann_file,
                quad_source="unknown",
            )

    def test_east_dataset_len(self, simple_dataset):
        """Test __len__ method"""
        img_dir, ann_file = simple_dataset
        dataset = EASTDataset(img_dir, ann_file)

        assert len(dataset) == 1
        assert dataset.__len__() == 1

    def test_east_dataset_getitem(self, simple_dataset):
        """Test getting an item from the dataset"""
        img_dir, ann_file = simple_dataset
        dataset = EASTDataset(img_dir, ann_file, target_size=512, score_geo_scale=0.25)

        img_tensor, target = dataset[0]

        # Check image tensor
        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.shape == (3, 512, 512)  # C, H, W

        # Check target dictionary
        assert isinstance(target, dict)
        assert "score_map" in target
        assert "geo_map" in target
        assert "quads" in target

        # Check map dimensions
        assert target["score_map"].shape[1:] == (128, 128)  # 512 * 0.25
        assert target["geo_map"].shape == (8, 128, 128)

        # Check quads format (N, 8) where 8 = 4 points * 2 coordinates
        assert target["quads"].shape[1] == 8

    def test_east_dataset_different_target_size(self, simple_dataset):
        """Test with different image sizes"""
        img_dir, ann_file = simple_dataset

        dataset_512 = EASTDataset(img_dir, ann_file, target_size=512)
        dataset_1024 = EASTDataset(img_dir, ann_file, target_size=1024)

        img_512, _ = dataset_512[0]
        img_1024, _ = dataset_1024[0]

        assert img_512.shape == (3, 512, 512)
        assert img_1024.shape == (3, 1024, 1024)

    def test_east_dataset_filter_invalid(self, tmp_path):
        """Test filtering of invalid annotations"""
        annotations = {
            "images": [
                {"id": 1, "file_name": "valid.jpg", "width": 640, "height": 480},
                {"id": 2, "file_name": "invalid.jpg", "width": 640, "height": 480},
                {"id": 3, "file_name": "no_ann.jpg", "width": 640, "height": 480},
            ],
            "annotations": [
                # Valid annotation (4 points)
                {
                    "id": 1,
                    "image_id": 1,
                    "segmentation": [[10, 10, 100, 10, 100, 50, 10, 50]],
                },
                # Invalid (less than 4 points)
                {"id": 2, "image_id": 2, "segmentation": [[10, 10, 100, 10]]},
                # id 3 has no annotations
            ],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()

        for fname in ["valid.jpg", "invalid.jpg", "no_ann.jpg"]:
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / fname), img)

        with pytest.warns(UserWarning, match="found.*images without valid quads"):
            dataset = EASTDataset(str(img_dir), str(ann_file))

        # Only 1 valid image should remain
        assert len(dataset) == 1

    def test_east_dataset_missing_image(self, tmp_path):
        """Missing images should trigger warnings and return a dummy sample."""
        annotations = {
            "images": [
                {"id": 1, "file_name": "missing.jpg", "width": 640, "height": 480}
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "segmentation": [[10, 10, 100, 10, 100, 50, 10, 50]],
                }
            ],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        # Do NOT create the image

        dataset = EASTDataset(str(img_dir), str(ann_file))

        with pytest.warns(UserWarning, match="returning dummy sample"):
            img_tensor, target = dataset[0]

        assert isinstance(img_tensor, torch.Tensor)
        assert img_tensor.shape == (3, 512, 512)
        assert torch.count_nonzero(img_tensor) == 0
        assert target["score_map"].shape == (1, 128, 128)
        assert target["geo_map"].shape == (8, 128, 128)
        assert target["quads"].shape == (0, 8)

    def test_east_dataset_multiple_quads(self, tmp_path):
        """Test with multiple quads on a single image"""
        annotations = {
            "images": [
                {"id": 1, "file_name": "multi.jpg", "width": 640, "height": 480}
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "segmentation": [[10, 10, 50, 10, 50, 30, 10, 30]],
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "segmentation": [[100, 100, 200, 100, 200, 150, 100, 150]],
                },
                {
                    "id": 3,
                    "image_id": 1,
                    "segmentation": [[300, 200, 400, 200, 400, 300, 300, 300]],
                },
            ],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "multi.jpg"), img)

        dataset = EASTDataset(str(img_dir), str(ann_file))
        _, target = dataset[0]

        # Should have 3 quads
        assert target["quads"].shape[0] == 3

    def test_east_dataset_empty_annotations(self, tmp_path):
        """Test with image without annotations"""
        annotations = {
            "images": [
                {"id": 1, "file_name": "empty.jpg", "width": 640, "height": 480}
            ],
            "annotations": [],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "empty.jpg"), img)

        with pytest.warns(UserWarning):
            dataset = EASTDataset(str(img_dir), str(ann_file))

        # Should be empty
        assert len(dataset) == 0

    def test_east_dataset_custom_transform(self, simple_dataset):
        """Test custom transform"""
        import torchvision.transforms as transforms

        img_dir, ann_file = simple_dataset

        # Custom transform without normalization
        custom_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.ToTensor(),
            ]
        )

        dataset = EASTDataset(img_dir, ann_file, transform=custom_transform)

        img_tensor, _ = dataset[0]
        assert img_tensor.shape == (3, 512, 512)
        # Without normalization values should be in [0, 1]
        assert img_tensor.min() >= 0
        assert img_tensor.max() <= 1

    def test_east_dataset_dataset_name(self, simple_dataset):
        """Test dataset_name attribute"""
        img_dir, ann_file = simple_dataset

        # Automatic name from folder
        dataset = EASTDataset(img_dir, ann_file)
        assert dataset.dataset_name == Path(img_dir).stem

        # Custom name
        dataset_custom = EASTDataset(img_dir, ann_file, dataset_name="my_dataset")
        assert dataset_custom.dataset_name == "my_dataset"

    def test_compute_quad_maps(self, simple_dataset):
        """Test generation of score and geo maps"""
        img_dir, ann_file = simple_dataset
        dataset = EASTDataset(img_dir, ann_file, target_size=512, score_geo_scale=0.25)

        # Create a square
        quad = np.array([[50, 50], [150, 50], [150, 150], [50, 150]], dtype=np.float32)

        score_map, geo_map = dataset.compute_quad_maps([quad])

        # Check dimensions
        assert score_map.shape == (128, 128)
        assert geo_map.shape == (8, 128, 128)

        # Should have at least one positive value in score_map (inside the quad)
        assert np.sum(score_map) > 0
        assert score_map.max() == 1.0

    def test_compute_quad_maps_multiple_quads(self, simple_dataset):
        """Test map generation with multiple quads"""
        img_dir, ann_file = simple_dataset
        dataset = EASTDataset(img_dir, ann_file, target_size=512, score_geo_scale=0.25)

        quad1 = np.array([[20, 20], [80, 20], [80, 60], [20, 60]], dtype=np.float32)
        quad2 = np.array(
            [[100, 100], [200, 100], [200, 180], [100, 180]], dtype=np.float32
        )

        score_map, geo_map = dataset.compute_quad_maps([quad1, quad2])

        # Both regions should be marked
        assert np.sum(score_map) > 0
        assert score_map.shape == (128, 128)
        assert geo_map.shape == (8, 128, 128)

    def test_compute_quad_maps_empty(self, simple_dataset):
        """Test map generation without quads"""
        img_dir, ann_file = simple_dataset
        dataset = EASTDataset(img_dir, ann_file, target_size=512, score_geo_scale=0.25)

        score_map, geo_map = dataset.compute_quad_maps([])

        # Should have zero maps
        assert score_map.shape == (128, 128)
        assert geo_map.shape == (8, 128, 128)
        assert np.sum(score_map) == 0

    def test_east_dataset_segmentation_variants(self, tmp_path):
        """Test various segmentation formats"""
        annotations = {
            "images": [{"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}],
            "annotations": [
                # Variant 1: simple list
                {
                    "id": 1,
                    "image_id": 1,
                    "segmentation": [[10, 10, 50, 10, 50, 30, 10, 30]],
                },
                # Variant 2: annotation without segmentation
                {"id": 2, "image_id": 1},
                # Variant 3: empty segmentation
                {"id": 3, "image_id": 1, "segmentation": []},
            ],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.jpg"), img)

        dataset = EASTDataset(str(img_dir), str(ann_file))
        _, target = dataset[0]

        # Should have at least 1 valid quad
        assert target["quads"].shape[0] >= 1

    def test_east_dataset_multipart_segmentation(self, tmp_path):
        """Multipart polygon annotations should be parsed consistently."""
        annotations = {
            "images": [{"id": 1, "file_name": "test.jpg", "width": 400, "height": 400}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "segmentation": [
                        [10, 10, 80, 10, 80, 40, 10, 40],
                        [120, 60, 220, 60, 220, 100, 120, 100],
                    ],
                }
            ],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.jpg"), img)

        dataset = EASTDataset(str(img_dir), str(ann_file), target_size=400)

        assert len(dataset) == 1
        _, target = dataset[0]
        assert target["quads"].shape[0] == 2

    def test_east_dataset_quad_source_auto_preserves_4point_polygons(self, tmp_path):
        """quad_source='auto' should keep 4-point polygons without minAreaRect."""
        polygon = [100, 100, 220, 120, 200, 210, 80, 190]
        annotations = {
            "images": [{"id": 1, "file_name": "test.jpg", "width": 400, "height": 400}],
            "annotations": [{"id": 1, "image_id": 1, "segmentation": [polygon]}],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.jpg"), img)

        dataset = EASTDataset(
            str(img_dir),
            str(ann_file),
            target_size=400,
            quad_source="auto",
        )
        _, quads = dataset._load_image_and_quads(0)

        expected = order_vertices_clockwise(np.array(polygon, dtype=np.float32).reshape(4, 2))
        assert len(quads) == 1
        assert np.allclose(quads[0], expected)

    def test_east_dataset_quad_source_min_area_rect_forces_fit(self, tmp_path):
        """quad_source='min_area_rect' should still fit a rectangle."""
        polygon = [100, 100, 220, 120, 200, 210, 80, 190]
        annotations = {
            "images": [{"id": 1, "file_name": "test.jpg", "width": 400, "height": 400}],
            "annotations": [{"id": 1, "image_id": 1, "segmentation": [polygon]}],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.jpg"), img)

        dataset = EASTDataset(
            str(img_dir),
            str(ann_file),
            target_size=400,
            quad_source="min_area_rect",
        )
        _, quads = dataset._load_image_and_quads(0)

        original_quad = order_vertices_clockwise(np.array(polygon, dtype=np.float32).reshape(4, 2))
        assert len(quads) == 1
        assert not np.allclose(quads[0], original_quad)

    def test_east_dataset_quad_source_as_is_skips_non_quads(self, tmp_path):
        """quad_source='as_is' should ignore polygons that are not 4-point quads."""
        annotations = {
            "images": [{"id": 1, "file_name": "test.jpg", "width": 400, "height": 400}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "segmentation": [[10, 10, 80, 10, 80, 40, 10, 40]],
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "segmentation": [[150, 150, 220, 140, 260, 190, 210, 240, 140, 220]],
                },
            ],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.jpg"), img)

        dataset = EASTDataset(
            str(img_dir),
            str(ann_file),
            target_size=400,
            quad_source="as_is",
        )
        _, target = dataset[0]

        assert len(dataset) == 1
        assert target["quads"].shape[0] == 1

    def test_east_dataset_scaling(self, tmp_path):
        """Test correct coordinate scaling"""
        # Image 640x480, target_size=512
        annotations = {
            "images": [{"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    # Square in original coordinates
                    "segmentation": [[100, 100, 200, 100, 200, 200, 100, 200]],
                }
            ],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128
        cv2.imwrite(str(img_dir / "test.jpg"), img)

        dataset = EASTDataset(str(img_dir), str(ann_file), target_size=512)
        _, target = dataset[0]

        # Check that quads were scaled
        assert target["quads"].shape[0] == 1
        # Coordinates should be in range [0, 512]
        quad = target["quads"][0].reshape(4, 2)
        assert quad.min() >= 0
        assert quad.max() <= 512


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestEASTDatasetEdgeCases:
    """Edge case tests for EASTDataset"""

    def test_very_small_quad(self, tmp_path):
        """Test with a very small square"""
        annotations = {
            "images": [{"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "segmentation": [[10, 10, 11, 10, 11, 11, 10, 11]],  # 1x1 px
                }
            ],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.jpg"), img)

        dataset = EASTDataset(str(img_dir), str(ann_file))
        # Should process without errors
        img_tensor, target = dataset[0]
        assert img_tensor.shape == (3, 512, 512)

    def test_quad_at_image_boundary(self, tmp_path):
        """Test with quad at image boundary"""
        annotations = {
            "images": [{"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "segmentation": [
                        [0, 0, 100, 0, 100, 100, 0, 100]
                    ],  # At image corner
                }
            ],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.jpg"), img)

        dataset = EASTDataset(str(img_dir), str(ann_file))
        img_tensor, target = dataset[0]

        # Should process correctly
        assert target["quads"].shape[0] >= 1

    def test_different_score_geo_scales(self, simple_dataset):
        """Test with different score_geo_scale values"""
        img_dir, ann_file = simple_dataset

        dataset_025 = EASTDataset(img_dir, ann_file, score_geo_scale=0.25)
        dataset_050 = EASTDataset(img_dir, ann_file, score_geo_scale=0.5)

        _, target_025 = dataset_025[0]
        _, target_050 = dataset_050[0]

        # Map dimensions should differ
        assert target_025["score_map"].shape[1:] == (128, 128)  # 512 * 0.25
        assert target_050["score_map"].shape[1:] == (256, 256)  # 512 * 0.5

    @pytest.fixture
    def simple_dataset(self, tmp_path):
        """Creates a simple dataset with one image"""
        annotations = {
            "images": [{"id": 1, "file_name": "test.jpg", "width": 640, "height": 480}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "segmentation": [[10, 10, 100, 10, 100, 50, 10, 50]],
                }
            ],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.jpg"), img)

        return str(img_dir), str(ann_file)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestEASTDatasetAugmentations:
    @pytest.fixture
    def simple_dataset(self, tmp_path):
        annotations = {
            "images": [{"id": 1, "file_name": "test.jpg", "width": 128, "height": 128}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "segmentation": [[16, 16, 48, 16, 48, 48, 16, 48]],
                }
            ],
        }

        ann_file = tmp_path / "annotations.json"
        ann_file.write_text(json.dumps(annotations), encoding="utf-8")

        img_dir = tmp_path / "images"
        img_dir.mkdir()
        img = np.full((128, 128, 3), 100, dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.jpg"), img)

        return str(img_dir), str(ann_file)

    def _make_aug_dataset(self, simple_dataset):
        img_dir, ann_file = simple_dataset
        dataset = EASTDataset(img_dir, ann_file, target_size=64, score_geo_scale=0.25)
        dataset.image_ids = [0, 1, 2, 3]
        return dataset

    def test_apply_mosaic_keeps_quads_in_bounds(self, simple_dataset, monkeypatch):
        dataset = self._make_aug_dataset(simple_dataset)
        dataset.mosaic_prob = 1.0

        monkeypatch.setattr(
            dataset,
            "_load_image_and_quads",
            lambda idx: (
                np.full((64, 64, 3), 40 + idx * 40, dtype=np.uint8),
                [_quad(8, 8, 24, 24)],
            ),
        )
        monkeypatch.setattr(np.random, "uniform", lambda *args, **kwargs: 0.5)
        randint_values = iter([1, 2, 3])
        monkeypatch.setattr(
            np.random, "randint", lambda *args, **kwargs: next(randint_values)
        )

        img, quads = dataset._apply_mosaic(0, force=True)

        pts = np.stack(quads, axis=0)
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8
        assert len(quads) == 4
        assert np.all(np.isfinite(pts))
        assert pts.min() >= 0
        assert pts.max() <= 63

    def test_apply_cutmix_merges_expected_regions(self, simple_dataset, monkeypatch):
        dataset = self._make_aug_dataset(simple_dataset)
        dataset.cutmix_prob = 1.0
        dataset.cutmix_alpha = 1.0

        def fake_load(idx):
            if idx == 0:
                return (
                    np.full((64, 64, 3), 10, dtype=np.uint8),
                    [_quad(40, 40, 56, 56), _quad(20, 20, 40, 40)],
                )
            return (
                np.full((64, 64, 3), 200, dtype=np.uint8),
                [_quad(4, 4, 12, 12), _quad(40, 40, 52, 52)],
            )

        monkeypatch.setattr(dataset, "_load_image_and_quads", fake_load)
        monkeypatch.setattr(np.random, "beta", lambda *args, **kwargs: 0.75)
        randint_values = iter([0, 0, 1])
        monkeypatch.setattr(
            np.random, "randint", lambda *args, **kwargs: next(randint_values)
        )

        img, quads = dataset._apply_cutmix(0, force=True)

        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8
        assert tuple(img[0, 0]) == (200, 200, 200)
        assert tuple(img[50, 50]) == (10, 10, 10)
        assert len(quads) == 2
        assert any(np.allclose(q, _quad(40, 40, 56, 56)) for q in quads)
        assert any(np.allclose(q, _quad(4, 4, 12, 12)) for q in quads)

    def test_apply_ricap_returns_finite_quads(self, simple_dataset, monkeypatch):
        dataset = self._make_aug_dataset(simple_dataset)
        dataset.ricap_prob = 1.0
        dataset.ricap_beta = 0.3

        monkeypatch.setattr(
            dataset,
            "_load_image_and_quads",
            lambda idx: (
                np.full((64, 64, 3), 30 + idx * 30, dtype=np.uint8),
                [_quad(4, 4, 16, 16)],
            ),
        )
        beta_values = iter([0.5, 0.5])
        monkeypatch.setattr(np.random, "beta", lambda *args, **kwargs: next(beta_values))
        randint_values = iter([1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0])
        monkeypatch.setattr(
            np.random, "randint", lambda *args, **kwargs: next(randint_values)
        )

        img, quads = dataset._apply_ricap(0, force=True)

        pts = np.stack(quads, axis=0)
        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8
        assert len(quads) == 4
        assert np.all(np.isfinite(pts))
        assert pts.min() >= 0
        assert pts.max() <= 63

    def test_apply_resizemix_keeps_shape_and_expected_quads(
        self, simple_dataset, monkeypatch
    ):
        dataset = self._make_aug_dataset(simple_dataset)
        dataset.resizemix_prob = 1.0
        dataset.resizemix_scale_range = (0.5, 0.5)

        def fake_load(idx):
            if idx == 0:
                return (
                    np.full((64, 64, 3), 10, dtype=np.uint8),
                    [_quad(40, 40, 56, 56), _quad(12, 12, 28, 28)],
                )
            return (
                np.full((64, 64, 3), 220, dtype=np.uint8),
                [_quad(4, 4, 12, 12)],
            )

        monkeypatch.setattr(dataset, "_load_image_and_quads", fake_load)
        monkeypatch.setattr(np.random, "uniform", lambda *args, **kwargs: 0.5)
        randint_values = iter([0, 0, 1])
        monkeypatch.setattr(
            np.random, "randint", lambda *args, **kwargs: next(randint_values)
        )

        img, quads = dataset._apply_resizemix(0, force=True)

        assert img.shape == (64, 64, 3)
        assert img.dtype == np.uint8
        assert tuple(img[0, 0]) == (220, 220, 220)
        assert tuple(img[50, 50]) == (10, 10, 10)
        assert len(quads) == 2
        assert any(np.allclose(q, _quad(40, 40, 56, 56)) for q in quads)
        assert any(np.allclose(q, _quad(2, 2, 6, 6)) for q in quads)

    def test_photometric_helpers_preserve_shape_dtype_and_finiteness(
        self, simple_dataset, monkeypatch
    ):
        dataset = self._make_aug_dataset(simple_dataset)

        img = np.zeros((16, 16, 3), dtype=np.uint8)
        img[:, 8:] = 200

        dataset.cutout_prob = 1.0
        dataset.cutout_num_holes = 1
        dataset.cutout_hole_size_range = (0.25, 0.25)
        monkeypatch.setattr(np.random, "uniform", lambda *args, **kwargs: 0.25)
        randint_values = iter([0, 0])
        monkeypatch.setattr(
            np.random, "randint", lambda *args, **kwargs: next(randint_values)
        )
        cutout = dataset._apply_cutout(img, force=True)
        expected_fill = img.mean(axis=(0, 1)).astype(np.uint8)
        assert cutout.shape == img.shape
        assert cutout.dtype == np.uint8
        assert np.all(cutout[:4, :4] == expected_fill)

        dataset.elastic_prob = 1.0
        monkeypatch.setattr(
            np.random, "rand", lambda *shape: np.full(shape, 0.5, dtype=np.float32)
        )
        elastic = dataset._apply_elastic(img, force=True)
        assert elastic.shape == img.shape
        assert elastic.dtype == np.uint8
        assert np.array_equal(elastic, img)

        dataset.fog_prob = 1.0
        dataset.fog_direction = "random"
        monkeypatch.setattr(np.random, "uniform", lambda *args, **kwargs: 0.25)
        monkeypatch.setattr(np.random, "choice", lambda choices: "left")
        fog = dataset._apply_fog(img, force=True)
        assert fog.shape == img.shape
        assert fog.dtype == np.uint8
        assert np.all(np.isfinite(fog))
        assert fog[:, 0].mean() > img[:, 0].mean()

    def test_preview_augmentation_and_list(self, simple_dataset):
        img_dir, ann_file = simple_dataset
        dataset = EASTDataset(img_dir, ann_file, target_size=64)

        names = dataset.list_augmentations()
        assert "mosaic" in names
        assert "fog" in names
        assert "negative" in names

        preview = dataset.preview_augmentation(0, "negative")
        assert preview.shape == (64, 64, 3)
        assert preview.dtype == np.uint8
        assert dataset.preview_augmentation(0, "unknown") is None
