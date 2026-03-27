"""Smoke tests for EAST training loop reliability."""

import contextlib
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

import manuscript.detectors._east.train_utils as train_utils


class TinyEastDataset(torch.utils.data.Dataset):
    def __init__(self, length=1):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = torch.full((3, 16, 16), 0.25, dtype=torch.float32)
        score_map = torch.full((1, 8, 8), 0.5, dtype=torch.float32)
        geo_map = torch.zeros((8, 8, 8), dtype=torch.float32)
        quads = torch.zeros((0, 8), dtype=torch.float32)
        return image, {"score_map": score_map, "geo_map": geo_map, "quads": quads}


class TinyEastModel(nn.Module):
    score_scale = 4

    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(3, 9, kernel_size=1)

    def forward(self, x):
        output = self.backbone(x)
        return {"score": output[:, :1], "geometry": output[:, 1:]}


class FakeLoss(nn.Module):
    def forward(self, gt_s, ps, gt_g, pg):
        return ((ps - gt_s) ** 2).mean() + ((pg - gt_g) ** 2).mean()


class FakeWriter:
    instances = []

    def __init__(self, log_dir, purge_step=None):
        self.log_dir = log_dir
        self.purge_step = purge_step
        self.scalars = []
        self.images = []
        self.closed = False
        type(self).instances.append(self)

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, float(value), step))

    def add_image(self, tag, image, step, dataformats=None):
        self.images.append((tag, step, dataformats))

    def close(self):
        self.closed = True


class FakeScaler:
    def __init__(self, *args, **kwargs):
        self.loaded = None

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None

    def state_dict(self):
        return {"scale": 1.0}

    def load_state_dict(self, state):
        self.loaded = state


def _patch_runtime(monkeypatch):
    FakeWriter.instances.clear()
    monkeypatch.setattr(train_utils, "SummaryWriter", FakeWriter)
    monkeypatch.setattr(train_utils, "_log_augmentation_previews", lambda *a, **k: None)
    monkeypatch.setattr(train_utils, "EASTLoss", lambda **kwargs: FakeLoss())
    monkeypatch.setattr(
        train_utils.toptim,
        "RAdam",
        lambda params, lr: torch.optim.SGD(params, lr=lr),
    )
    monkeypatch.setattr(train_utils, "tqdm", lambda iterable, **kwargs: iterable)
    monkeypatch.setattr(train_utils.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(train_utils.torch.amp, "GradScaler", FakeScaler)
    monkeypatch.setattr(
        train_utils.torch.amp,
        "autocast",
        lambda **kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(
        train_utils,
        "DataLoader",
        lambda dataset, **kwargs: torch.utils.data.DataLoader(
            dataset,
            **{**kwargs, "pin_memory": False, "persistent_workers": False},
        ),
    )


class TestRunTraining:
    def test_run_training_smoke_writes_checkpoints_and_exports(self, monkeypatch, tmp_path):
        _patch_runtime(monkeypatch)
        export_calls = []
        fake_detectors = ModuleType("manuscript.detectors")

        class FakeEAST:
            @staticmethod
            def export(**kwargs):
                export_calls.append(kwargs)
                Path(kwargs["output_path"]).write_bytes(b"onnx")

        fake_detectors.EAST = FakeEAST
        monkeypatch.setitem(sys.modules, "manuscript.detectors", fake_detectors)

        model = TinyEastModel()
        returned = train_utils._run_training(
            experiment_dir=str(tmp_path / "exp"),
            model=model,
            train_dataset=TinyEastDataset(1),
            val_dataset=TinyEastDataset(1),
            device=torch.device("cpu"),
            num_epochs=2,
            batch_size=1,
            accumulation_steps=1,
            lr=0.01,
            lr_scheduler="plateau",
            lr_scheduler_params={"factor": 0.5, "patience": 1},
            grad_clip=1.0,
            early_stop=0,
            use_sam=False,
            sam_type="sam",
            use_lookahead=False,
            use_ema=True,
            use_multiscale=False,
            use_ohem=False,
            ohem_ratio=3.0,
            use_focal_geo=False,
            focal_gamma=2.0,
            log_collage=False,
        )

        exp_dir = tmp_path / "exp"
        ckpt_dir = exp_dir / "checkpoints"
        writer = FakeWriter.instances[-1]

        assert returned is not model
        assert isinstance(returned, nn.Module)
        assert writer.closed is True
        assert any(tag == "Loss/Train" for tag, _, _ in writer.scalars)
        assert any(tag == "Loss/Val" for tag, _, _ in writer.scalars)
        assert json.loads((exp_dir / "training_config.json").read_text(encoding="utf-8"))["optimizer"] == "RAdam"
        assert (ckpt_dir / "best_loss.pth").exists()
        assert (ckpt_dir / "best_dice.pth").exists()
        assert (ckpt_dir / "last.pth").exists()
        assert (ckpt_dir / "last_state.pt").exists()
        assert (ckpt_dir / "best_model.onnx").exists()
        assert export_calls and export_calls[0]["weights_path"].endswith("best_loss.pth")

    def test_run_training_resume_missing_state_raises(self, monkeypatch, tmp_path):
        _patch_runtime(monkeypatch)

        with pytest.raises(FileNotFoundError, match="Resume requested"):
            train_utils._run_training(
                experiment_dir=str(tmp_path / "exp"),
                model=TinyEastModel(),
                train_dataset=TinyEastDataset(1),
                val_dataset=TinyEastDataset(1),
                device=torch.device("cpu"),
                num_epochs=1,
                batch_size=1,
                accumulation_steps=1,
                lr=0.01,
                lr_scheduler="none",
                lr_scheduler_params=None,
                grad_clip=1.0,
                early_stop=1,
                use_sam=False,
                sam_type="sam",
                use_lookahead=False,
                use_ema=False,
                use_multiscale=False,
                use_ohem=False,
                ohem_ratio=3.0,
                use_focal_geo=False,
                focal_gamma=2.0,
                resume=True,
                resume_state_path=str(tmp_path / "missing.pt"),
                log_collage=False,
            )

    def test_run_training_resume_full_state_skips_when_epoch_exceeded(self, monkeypatch, tmp_path):
        _patch_runtime(monkeypatch)
        export_calls = []
        fake_detectors = ModuleType("manuscript.detectors")

        class FakeEAST:
            @staticmethod
            def export(**kwargs):
                export_calls.append(kwargs)

        fake_detectors.EAST = FakeEAST
        monkeypatch.setitem(sys.modules, "manuscript.detectors", fake_detectors)

        model = TinyEastModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        resume_path = tmp_path / "exp" / "checkpoints" / "last_state.pt"
        resume_path.parent.mkdir(parents=True)
        torch.save(
            {
                "epoch": 1,
                "model_state": model.state_dict(),
                "ema_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": {"scale": 1.0},
                "best_val_loss": 0.5,
                "best_val_dice": 0.6,
                "patience_loss": 1,
                "patience_dice": 2,
            },
            resume_path,
        )

        returned = train_utils._run_training(
            experiment_dir=str(tmp_path / "exp"),
            model=TinyEastModel(),
            train_dataset=TinyEastDataset(1),
            val_dataset=TinyEastDataset(1),
            device=torch.device("cpu"),
            num_epochs=1,
            batch_size=1,
            accumulation_steps=1,
            lr=0.01,
            lr_scheduler="plateau",
            lr_scheduler_params={"factor": 0.5},
            grad_clip=1.0,
            early_stop=1,
            use_sam=False,
            sam_type="sam",
            use_lookahead=False,
            use_ema=True,
            use_multiscale=False,
            use_ohem=False,
            ohem_ratio=3.0,
            use_focal_geo=False,
            focal_gamma=2.0,
            resume=True,
            resume_state_path=str(resume_path),
            log_collage=False,
        )

        writer = FakeWriter.instances[-1]
        assert isinstance(returned, nn.Module)
        assert writer.purge_step == 2
        assert writer.closed is True
        assert export_calls == []

    def test_run_training_validates_val_dataset_names_length(self, monkeypatch, tmp_path):
        _patch_runtime(monkeypatch)

        with pytest.raises(ValueError, match="val_dataset_names length must match"):
            train_utils._run_training(
                experiment_dir=str(tmp_path / "exp"),
                model=TinyEastModel(),
                train_dataset=TinyEastDataset(1),
                val_dataset=TinyEastDataset(1),
                device=torch.device("cpu"),
                num_epochs=1,
                batch_size=1,
                accumulation_steps=1,
                lr=0.01,
                lr_scheduler="none",
                lr_scheduler_params=None,
                grad_clip=1.0,
                early_stop=1,
                use_sam=False,
                sam_type="sam",
                use_lookahead=False,
                use_ema=False,
                use_multiscale=False,
                use_ohem=False,
                ohem_ratio=3.0,
                use_focal_geo=False,
                focal_gamma=2.0,
                val_datasets=[TinyEastDataset(1), TinyEastDataset(1)],
                val_dataset_names=["only_one"],
                log_collage=False,
            )
