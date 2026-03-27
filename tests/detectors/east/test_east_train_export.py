"""Tests for EAST train/export orchestration."""

import json
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

import manuscript.detectors._east as east_module
from manuscript.detectors import EAST


def _write_coco_dataset(root: Path, stem: str) -> tuple[str, str]:
    images_dir = root / stem / "images"
    images_dir.mkdir(parents=True)
    (images_dir / f"{stem}.jpg").write_bytes(b"fake image")

    ann_path = root / stem / "annotations.json"
    ann_path.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": f"{stem}.jpg",
                        "width": 100,
                        "height": 100,
                    }
                ],
                "annotations": [],
            }
        ),
        encoding="utf-8",
    )
    return str(images_dir), str(ann_path)


class _FakeDataset:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.dataset_name = kwargs["dataset_name"]


class _FakeConcatDataset:
    def __init__(self, datasets):
        self.datasets = list(datasets)


@pytest.fixture
def east_train_stack(monkeypatch):
    monkeypatch.setattr(east_module, "_TORCH_AVAILABLE", True)
    monkeypatch.setattr(
        east_module,
        "torch",
        SimpleNamespace(
            device=lambda name: f"device:{name}",
            cuda=SimpleNamespace(is_available=lambda: False),
        ),
    )
    monkeypatch.setattr(east_module, "ConcatDataset", _FakeConcatDataset)
    monkeypatch.setattr(east_module, "EASTDataset", _FakeDataset)

    class FakeModel:
        score_scale = 0.25

        def __init__(self):
            self.device = None

        def to(self, device):
            self.device = device
            return self

    model_ctor = MagicMock(side_effect=lambda **_: FakeModel())
    monkeypatch.setattr(east_module, "EASTModel", model_ctor)
    return model_ctor


class TestEASTTrain:
    def test_train_is_static_method(self):
        assert isinstance(EAST.__dict__["train"], staticmethod)

    def test_train_requires_training_dependencies(self, monkeypatch, tmp_path):
        monkeypatch.setattr(east_module, "_TORCH_AVAILABLE", False)
        train_images, train_anns = _write_coco_dataset(tmp_path, "train")
        val_images, val_anns = _write_coco_dataset(tmp_path, "val")

        with pytest.raises(ImportError, match="PyTorch is required for training"):
            EAST.train(
                train_images=train_images,
                train_anns=train_anns,
                val_images=val_images,
                val_anns=val_anns,
            )

    def test_train_minimal_parameters(self, east_train_stack, tmp_path):
        train_images, train_anns = _write_coco_dataset(tmp_path, "train")
        val_images, val_anns = _write_coco_dataset(tmp_path, "val")

        with patch.object(east_module, "_run_training", autospec=True) as mock_train:
            mock_model = MagicMock(spec=nn.Module)
            mock_train.return_value = mock_model

            result = EAST.train(
                train_images=train_images,
                train_anns=train_anns,
                val_images=val_images,
                val_anns=val_anns,
                experiment_root=str(tmp_path / "experiments"),
                epochs=1,
                batch_size=1,
            )

        assert result == mock_model
        kwargs = mock_train.call_args.kwargs
        assert kwargs["batch_size"] == 1
        assert kwargs["num_epochs"] == 1
        assert kwargs["target_size"] == 1024
        assert kwargs["resume"] is False
        assert isinstance(kwargs["train_dataset"], _FakeConcatDataset)
        assert isinstance(kwargs["val_dataset"], _FakeConcatDataset)
        assert kwargs["val_dataset_names"] == ["val/annotations"]

    def test_train_combines_multiple_datasets_and_unique_names(
        self, east_train_stack, tmp_path
    ):
        train_a = _write_coco_dataset(tmp_path / "set_a", "train")
        train_b = _write_coco_dataset(tmp_path / "set_b", "train")
        val_a = _write_coco_dataset(tmp_path / "val_a", "eval")
        val_b = _write_coco_dataset(tmp_path / "val_b", "eval")

        with patch.object(east_module, "_run_training", autospec=True) as mock_train:
            mock_train.return_value = MagicMock()

            EAST.train(
                train_images=[train_a[0], train_b[0]],
                train_anns=[train_a[1], train_b[1]],
                val_images=[val_a[0], val_b[0]],
                val_anns=[val_a[1], val_b[1]],
                experiment_root=str(tmp_path / "experiments"),
                model_name="custom_model",
                augmentation_config={"quad_source": "as_is"},
                epochs=2,
                batch_size=4,
                lr=1e-4,
                backbone_name="resnet101",
                pretrained_backbone=False,
            )

        kwargs = mock_train.call_args.kwargs
        assert [ds.dataset_name for ds in kwargs["train_dataset"].datasets] == [
            "train/annotations",
            "train/annotations_2",
        ]
        assert kwargs["val_dataset_names"] == ["eval/annotations", "eval/annotations_2"]
        assert kwargs["augmentation_config"]["quad_source"] == "as_is"
        assert kwargs["backbone_name"] == "resnet101"
        assert kwargs["pretrained_backbone"] is False
        assert "custom_model" in kwargs["experiment_dir"]

    def test_train_resolve_resume_from_experiment_checkpoint(
        self, east_train_stack, tmp_path
    ):
        train_images, train_anns = _write_coco_dataset(tmp_path, "train")
        val_images, val_anns = _write_coco_dataset(tmp_path, "val")

        exp_dir = tmp_path / "existing_exp"
        checkpoints_dir = exp_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True)
        checkpoint = checkpoints_dir / "last_state.pt"
        checkpoint.write_bytes(b"checkpoint")
        (exp_dir / "training_config.json").write_text("{}", encoding="utf-8")

        with patch.object(east_module, "_run_training", autospec=True) as mock_train:
            mock_train.return_value = MagicMock()

            EAST.train(
                train_images=train_images,
                train_anns=train_anns,
                val_images=val_images,
                val_anns=val_anns,
                experiment_root=str(tmp_path / "experiments"),
                model_name="new_exp",
                resume_from=str(checkpoint),
                epochs=1,
            )

        kwargs = mock_train.call_args.kwargs
        assert kwargs["resume"] is True
        assert kwargs["resume_state_path"] == str(checkpoint)
        assert kwargs["experiment_dir"] == str(exp_dir.resolve())

    def test_train_weights_only_resume_keeps_default_experiment_dir(
        self, east_train_stack, tmp_path
    ):
        train_images, train_anns = _write_coco_dataset(tmp_path, "train")
        val_images, val_anns = _write_coco_dataset(tmp_path, "val")
        weights_only = tmp_path / "best.pth"
        weights_only.write_bytes(b"weights")

        with patch.object(east_module, "_run_training", autospec=True) as mock_train:
            mock_train.return_value = MagicMock()

            EAST.train(
                train_images=train_images,
                train_anns=train_anns,
                val_images=val_images,
                val_anns=val_anns,
                experiment_root=str(tmp_path / "experiments"),
                model_name="new_exp",
                resume_from=str(weights_only),
                epochs=1,
            )

        kwargs = mock_train.call_args.kwargs
        assert kwargs["resume"] is True
        assert kwargs["resume_state_path"] == str(weights_only)
        assert kwargs["experiment_dir"] == str(
            (tmp_path / "experiments" / "new_exp").resolve()
        )

    def test_train_validates_input_lengths(self, east_train_stack, tmp_path):
        train_images, train_anns = _write_coco_dataset(tmp_path, "train")
        val_images, val_anns = _write_coco_dataset(tmp_path, "val")

        with pytest.raises(AssertionError, match="train_images and train_anns"):
            EAST.train(
                train_images=[train_images, train_images],
                train_anns=[train_anns],
                val_images=val_images,
                val_anns=val_anns,
            )


class TestEASTExport:
    def test_export_is_static_method(self):
        assert isinstance(EAST.__dict__["export"], staticmethod)

    def test_export_requires_training_dependencies(self, monkeypatch, tmp_path):
        monkeypatch.setattr(east_module, "_TORCH_AVAILABLE", False)

        with pytest.raises(ImportError, match="PyTorch is required for exporting"):
            EAST.export(
                weights_path=str(tmp_path / "weights.pth"),
                output_path=str(tmp_path / "model.onnx"),
            )

    def test_export_file_not_found(self, monkeypatch, tmp_path):
        monkeypatch.setattr(east_module, "_TORCH_AVAILABLE", True)

        with pytest.raises(FileNotFoundError, match="Weights file not found"):
            EAST.export(
                weights_path=str(tmp_path / "missing.pth"),
                output_path=str(tmp_path / "model.onnx"),
            )

    def test_export_rejects_backbone_mismatch(self, monkeypatch, tmp_path):
        monkeypatch.setattr(east_module, "_TORCH_AVAILABLE", True)
        weights_path = tmp_path / "model.pth"
        weights_path.write_bytes(b"checkpoint")

        with patch.object(
            east_module.torch,
            "load",
            return_value={"backbone.extractor.layer3.10.conv1.weight": object()},
        ):
            with pytest.raises(ValueError, match="Backbone mismatch"):
                EAST.export(
                    weights_path=str(weights_path),
                    output_path=str(tmp_path / "model.onnx"),
                    backbone_name="resnet50",
                )

    def test_export_creates_onnx_file_and_simplifies(self, monkeypatch, tmp_path):
        monkeypatch.setattr(east_module, "_TORCH_AVAILABLE", True)
        weights_path = tmp_path / "model.pth"
        output_path = tmp_path / "model.onnx"
        weights_path.write_bytes(b"checkpoint")

        score_map = torch.zeros(1, 1, 64, 64)
        geo_map = torch.zeros(1, 8, 64, 64)

        class FakeEastModel:
            def eval(self):
                return self

            def __call__(self, x):
                return {"score": score_map, "geometry": geo_map}

        def _write_onnx(*args, **kwargs):
            output_path.write_bytes(b"fake onnx")

        class FakeSession:
            def run(self, *_args, **_kwargs):
                return [score_map.numpy(), geo_map.numpy()]

        fake_onnx = ModuleType("onnx")
        fake_onnx.load = MagicMock(return_value=object())
        fake_onnx.save = MagicMock()
        fake_onnx.checker = SimpleNamespace(check_model=MagicMock())
        fake_onnxsim = ModuleType("onnxsim")
        fake_onnxsim.simplify = MagicMock(return_value=(MagicMock(), True))
        fake_onnxruntime = ModuleType("onnxruntime")
        fake_onnxruntime.InferenceSession = MagicMock(return_value=FakeSession())

        with patch.object(
            east_module.torch,
            "load",
            return_value={"backbone.extractor.layer3.10.conv1.weight": object()},
        ):
            with patch.object(east_module, "EASTModel", return_value=FakeEastModel()):
                with patch.object(east_module.torch.onnx, "export", side_effect=_write_onnx) as mock_export:
                    with patch.dict(
                        sys.modules,
                        {
                            "onnx": fake_onnx,
                            "onnxsim": fake_onnxsim,
                            "onnxruntime": fake_onnxruntime,
                        },
                    ):
                        EAST.export(
                            weights_path=str(weights_path),
                            output_path=str(output_path),
                            backbone_name=None,
                            input_size=64,
                            simplify=True,
                        )

        assert mock_export.called
        assert fake_onnx.load.called
        assert fake_onnx.checker.check_model.called
        assert fake_onnxsim.simplify.called
        assert fake_onnx.save.called
        assert output_path.exists()

    def test_export_skips_simplify_when_disabled(self, monkeypatch, tmp_path):
        monkeypatch.setattr(east_module, "_TORCH_AVAILABLE", True)
        weights_path = tmp_path / "model.pth"
        output_path = tmp_path / "model.onnx"
        weights_path.write_bytes(b"checkpoint")

        score_map = torch.zeros(1, 1, 32, 32)
        geo_map = torch.zeros(1, 8, 32, 32)

        class FakeEastModel:
            def eval(self):
                return self

            def __call__(self, x):
                return {"score": score_map, "geometry": geo_map}

        def _write_onnx(*args, **kwargs):
            output_path.write_bytes(b"fake onnx")

        class FakeSession:
            def run(self, *_args, **_kwargs):
                return [score_map.numpy(), geo_map.numpy()]

        fake_onnx = ModuleType("onnx")
        fake_onnx.load = MagicMock(return_value=object())
        fake_onnx.checker = SimpleNamespace(check_model=MagicMock())
        fake_onnxsim = ModuleType("onnxsim")
        fake_onnxsim.simplify = MagicMock()
        fake_onnxruntime = ModuleType("onnxruntime")
        fake_onnxruntime.InferenceSession = MagicMock(return_value=FakeSession())

        with patch.object(
            east_module.torch,
            "load",
            return_value={"backbone.extractor.layer3.0.conv1.weight": object()},
        ):
            with patch.object(east_module, "EASTModel", return_value=FakeEastModel()):
                with patch.object(east_module.torch.onnx, "export", side_effect=_write_onnx):
                    with patch.dict(
                        sys.modules,
                        {
                            "onnx": fake_onnx,
                            "onnxsim": fake_onnxsim,
                            "onnxruntime": fake_onnxruntime,
                        },
                    ):
                        EAST.export(
                            weights_path=str(weights_path),
                            output_path=str(output_path),
                            backbone_name="resnet50",
                            input_size=32,
                            simplify=False,
                        )

        assert not fake_onnxsim.simplify.called
