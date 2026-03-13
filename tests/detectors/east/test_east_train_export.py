"""
Tests for EAST.train() and EAST.export() static methods.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

try:
    import torch
    import torch.nn as nn
    from manuscript.detectors._east import EAST

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None
    EAST = None

try:
    import onnx
    import onnxsim

    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    onnx = None
    onnxsim = None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestEASTTrain:
    """Tests for EAST.train() static method."""

    def test_train_is_static_method(self):
        """Test that train is a static method."""
        assert isinstance(EAST.__dict__["train"], staticmethod)

    def test_train_minimal_parameters(self, tmp_path):
        """Test train with minimal required parameters."""
        # Create mock training data
        train_img_dir = tmp_path / "train" / "images"
        train_img_dir.mkdir(parents=True)
        (train_img_dir / "img1.jpg").write_bytes(b"fake image")

        train_ann_file = tmp_path / "train" / "annotations.json"
        train_ann_file.write_text(
            json.dumps(
                {
                    "images": [
                        {"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100}
                    ],
                    "annotations": [],
                }
            )
        )

        val_img_dir = tmp_path / "val" / "images"
        val_img_dir.mkdir(parents=True)
        (val_img_dir / "img2.jpg").write_bytes(b"fake image")

        val_ann_file = tmp_path / "val" / "annotations.json"
        val_ann_file.write_text(
            json.dumps(
                {
                    "images": [
                        {"id": 2, "file_name": "img2.jpg", "width": 100, "height": 100}
                    ],
                    "annotations": [],
                }
            )
        )

        experiment_root = tmp_path / "experiments"

        # Mock the internal _run_training function
        with patch("manuscript.detectors._east._run_training") as mock_train:
            mock_model = MagicMock(spec=nn.Module)
            mock_train.return_value = mock_model

            result = EAST.train(
                train_images=str(train_img_dir),
                train_anns=str(train_ann_file),
                val_images=str(val_img_dir),
                val_anns=str(val_ann_file),
                experiment_root=str(experiment_root),
                epochs=1,
                batch_size=1,
            )

            # Verify _run_training was called
            assert mock_train.called
            assert result == mock_model

    def test_train_with_multiple_datasets(self, tmp_path):
        """Test train with multiple training datasets."""
        # Create multiple dataset directories
        datasets = []
        for i in range(2):
            img_dir = tmp_path / f"train{i}" / "images"
            img_dir.mkdir(parents=True)
            (img_dir / f"img{i}.jpg").write_bytes(b"fake")

            ann_file = tmp_path / f"train{i}" / "annotations.json"
            ann_file.write_text(
                json.dumps(
                    {
                        "images": [
                            {
                                "id": i,
                                "file_name": f"img{i}.jpg",
                                "width": 100,
                                "height": 100,
                            }
                        ],
                        "annotations": [],
                    }
                )
            )
            datasets.append((str(img_dir), str(ann_file)))

        val_img_dir = tmp_path / "val" / "images"
        val_img_dir.mkdir(parents=True)
        (val_img_dir / "val.jpg").write_bytes(b"fake")

        val_ann_file = tmp_path / "val" / "annotations.json"
        val_ann_file.write_text(
            json.dumps(
                {
                    "images": [
                        {"id": 99, "file_name": "val.jpg", "width": 100, "height": 100}
                    ],
                    "annotations": [],
                }
            )
        )

        experiment_root = tmp_path / "experiments"

        with patch("manuscript.detectors._east._run_training") as mock_train:
            mock_model = MagicMock()
            mock_train.return_value = mock_model

            result = EAST.train(
                train_images=[ds[0] for ds in datasets],
                train_anns=[ds[1] for ds in datasets],
                val_images=str(val_img_dir),
                val_anns=str(val_ann_file),
                experiment_root=str(experiment_root),
                epochs=1,
            )

            assert mock_train.called
            # Check that multiple datasets were passed
            # _run_training receives train_dataset (ConcatDataset) as first positional arg
            call_args, call_kwargs = mock_train.call_args
            # Verify it was called (datasets are created internally, not passed directly)
            assert len(call_args) > 0 or "train_dataset" in call_kwargs

    def test_train_with_custom_parameters(self, tmp_path):
        """Test train with custom training parameters."""
        train_img_dir = tmp_path / "train" / "images"
        train_img_dir.mkdir(parents=True)
        (train_img_dir / "img.jpg").write_bytes(b"fake")

        train_ann = tmp_path / "train" / "ann.json"
        train_ann.write_text(json.dumps({"images": [], "annotations": []}))

        val_img_dir = tmp_path / "val" / "images"
        val_img_dir.mkdir(parents=True)
        (val_img_dir / "val.jpg").write_bytes(b"fake")

        val_ann = tmp_path / "val" / "ann.json"
        val_ann.write_text(json.dumps({"images": [], "annotations": []}))

        with patch("manuscript.detectors._east._run_training") as mock_train:
            mock_model = MagicMock()
            mock_train.return_value = mock_model

            EAST.train(
                train_images=str(train_img_dir),
                train_anns=str(train_ann),
                val_images=str(val_img_dir),
                val_anns=str(val_ann),
                experiment_root=str(tmp_path / "exp"),
                model_name="custom_model",
                backbone_name="resnet101",
                pretrained_backbone=False,
                freeze_first=False,
                target_size=512,
                epochs=10,
                batch_size=4,
                lr=0.0001,
                use_sam=False,
                use_ohem=False,
            )

            # Verify custom parameters were passed to _run_training
            call_args, call_kwargs = mock_train.call_args
            # These parameters are passed through to _run_training
            assert call_kwargs["backbone_name"] == "resnet101"
            assert call_kwargs["pretrained_backbone"] is False
            assert call_kwargs["target_size"] == 512
            assert call_kwargs["batch_size"] == 4
            assert call_kwargs["lr"] == 0.0001
            # model_name is used for experiment_dir, check that
            assert "experiment_dir" in call_kwargs
            assert "custom_model" in call_kwargs["experiment_dir"]

    def test_train_passes_quad_source_via_augmentation_config(self, tmp_path):
        """Test quad_source is forwarded into dataset/training augmentation config."""
        train_img_dir = tmp_path / "train" / "images"
        train_img_dir.mkdir(parents=True)
        (train_img_dir / "img.jpg").write_bytes(b"fake")

        train_ann = tmp_path / "train" / "ann.json"
        train_ann.write_text(json.dumps({"images": [], "annotations": []}))

        val_img_dir = tmp_path / "val" / "images"
        val_img_dir.mkdir(parents=True)
        (val_img_dir / "val.jpg").write_bytes(b"fake")

        val_ann = tmp_path / "val" / "ann.json"
        val_ann.write_text(json.dumps({"images": [], "annotations": []}))

        with patch("manuscript.detectors._east._run_training") as mock_train:
            mock_model = MagicMock()
            mock_train.return_value = mock_model

            EAST.train(
                train_images=str(train_img_dir),
                train_anns=str(train_ann),
                val_images=str(val_img_dir),
                val_anns=str(val_ann),
                experiment_root=str(tmp_path / "exp"),
                augmentation_config={"quad_source": "as_is"},
                epochs=1,
            )

            _, call_kwargs = mock_train.call_args
            assert call_kwargs["augmentation_config"]["quad_source"] == "as_is"

    def test_train_with_resume_from(self, tmp_path):
        """Test train with resume_from parameter."""
        # Create checkpoint file
        checkpoint_path = tmp_path / "checkpoint.pth"
        checkpoint_path.write_bytes(b"fake checkpoint")

        train_img_dir = tmp_path / "train" / "images"
        train_img_dir.mkdir(parents=True)
        (train_img_dir / "img.jpg").write_bytes(b"fake")

        train_ann = tmp_path / "train" / "ann.json"
        train_ann.write_text(json.dumps({"images": [], "annotations": []}))

        val_img_dir = tmp_path / "val" / "images"
        val_img_dir.mkdir(parents=True)
        (val_img_dir / "val.jpg").write_bytes(b"fake")

        val_ann = tmp_path / "val" / "ann.json"
        val_ann.write_text(json.dumps({"images": [], "annotations": []}))

        with patch("manuscript.detectors._east._run_training") as mock_train:
            mock_model = MagicMock()
            mock_train.return_value = mock_model

            EAST.train(
                train_images=str(train_img_dir),
                train_anns=str(train_ann),
                val_images=str(val_img_dir),
                val_anns=str(val_ann),
                experiment_root=str(tmp_path / "exp"),
                resume_from=str(checkpoint_path),
                epochs=5,
            )

            call_args, call_kwargs = mock_train.call_args
            # Resume is handled internally - just verify training was called
            assert call_kwargs is not None


@pytest.mark.skipif(
    not TORCH_AVAILABLE or not ONNX_AVAILABLE, reason="PyTorch and ONNX not installed"
)
class TestEASTExport:
    """Tests for EAST.export() static method."""

    def test_export_is_static_method(self):
        """Test that export is a static method."""
        assert isinstance(EAST.__dict__["export"], staticmethod)

    def test_export_file_not_found(self, tmp_path):
        """Test export raises FileNotFoundError for missing weights."""
        weights_path = tmp_path / "nonexistent.pth"
        output_path = tmp_path / "model.onnx"

        with pytest.raises(FileNotFoundError, match="Weights file not found"):
            EAST.export(
                weights_path=str(weights_path),
                output_path=str(output_path),
            )

    def test_export_creates_onnx_file(self, tmp_path):
        """Test that export creates an ONNX file."""
        weights_path = tmp_path / "model.pth"
        output_path = tmp_path / "model.onnx"

        # Create minimal checkpoint that won't trigger simplify
        state_dict = {
            "model_state_dict": {},
        }
        torch.save(state_dict, weights_path)

        # Fully mock the export process
        with patch("manuscript.detectors._east.EASTModel"):
            with patch("torch.onnx.export") as mock_export:

                def create_file(*args, **kwargs):
                    output_path.write_bytes(b"fake onnx")

                mock_export.side_effect = create_file

                with patch("onnx.load"):
                    with patch("onnx.checker.check_model"):
                        with patch("onnxruntime.InferenceSession"):
                            # Don't call simplify
                            with patch("onnxsim.simplify") as mock_simplify:
                                mock_model = MagicMock()
                                mock_model.SerializeToString.return_value = (
                                    b"simplified"
                                )
                                mock_simplify.return_value = (mock_model, True)

                                EAST.export(
                                    weights_path=str(weights_path),
                                    output_path=str(output_path),
                                    simplify=False,  # Don't simplify
                                )

                                assert mock_export.called
                                # Verify simplify NOT called when False
                                assert not mock_simplify.called
