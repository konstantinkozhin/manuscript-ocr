from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


def test_trba_epoch_scheduler_state_is_saved_after_step(monkeypatch, tmp_path):
    import torch
    import manuscript.recognizers._trba.training.train as module
    from tests.recognizers.test_trba_training_smoke import _patch_runtime
    from manuscript.recognizers._trba.training.utils import save_checkpoint

    _patch_runtime(monkeypatch)
    records = []

    def record(path, model, optimizer, scheduler, scaler, epoch, *rest):
        records.append((Path(path).name, epoch, scheduler.state_dict()["last_epoch"]))
        return save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, *rest)

    monkeypatch.setattr(module, "save_checkpoint", record)
    charset = tmp_path / "charset.txt"
    charset.write_text("<PAD>\n<SOS>\n<EOS>\n<BLANK>\na\n")
    cfg = module.Config(
        dict(
            exp_dir=str(tmp_path / "exp"),
            train_csvs=["train.csv"],
            train_roots=["images"],
            charset_path=str(charset),
            epochs=2,
            batch_size=2,
            lr=0.001,
            optimizer="AdamW",
            scheduler="CosineAnnealingLR",
            max_len=4,
            hidden_size=8,
            num_encoder_layers=1,
            cnn_in_channels=3,
            cnn_out_channels=4,
            cnn_backbone="seresnet31-lite",
            val_size=2,
            pretrain_weights="none",
            val_interval=1,
        )
    )
    module.run_training(cfg, device="cpu")
    assert all(epoch == state for _, epoch, state in records), records


def test_trba_detects_its_own_checkpoint_for_automatic_resume(monkeypatch, tmp_path):
    import manuscript.recognizers._trba.training.train as module
    from tests.recognizers.test_trba_training_smoke import _patch_runtime

    _patch_runtime(monkeypatch)
    charset = tmp_path / "charset.txt"
    charset.write_text("<PAD>\n<SOS>\n<EOS>\n<BLANK>\na\n")
    settings = dict(
        exp_dir=str(tmp_path / "exp"),
        train_csvs=["train.csv"],
        train_roots=["images"],
        charset_path=str(charset),
        epochs=1,
        batch_size=2,
        lr=0.001,
        optimizer="AdamW",
        scheduler="None",
        max_len=4,
        hidden_size=8,
        num_encoder_layers=1,
        cnn_in_channels=3,
        cnn_out_channels=4,
        cnn_backbone="seresnet31-lite",
        val_size=2,
        pretrain_weights="none",
        val_interval=1,
    )
    module.run_training(module.Config(settings), device="cpu")
    calls = []
    monkeypatch.setattr(
        module, "load_checkpoint", lambda *a, **k: calls.append(a[0]) or {"epoch": 1}
    )
    module.run_training(module.Config(settings), device="cpu")
    assert calls, "last_ckpt.pth exists, but automatic resume did not load it"


def training_config(tmp_path, **options):
    import manuscript.recognizers._trba.training.train as module

    charset = tmp_path / "charset.txt"
    charset.write_text("<PAD>\n<SOS>\n<EOS>\n<BLANK>\na\n", encoding="utf-8")
    settings = dict(
        exp_dir=str(tmp_path / "exp"),
        train_csvs=["train.csv"],
        train_roots=["images"],
        charset_path=str(charset),
        epochs=2,
        batch_size=2,
        lr=0.001,
        optimizer="AdamW",
        scheduler="CosineAnnealingLR",
        max_len=4,
        hidden_size=8,
        num_encoder_layers=1,
        cnn_in_channels=3,
        cnn_out_channels=4,
        cnn_backbone="seresnet31-lite",
        val_size=2,
        pretrain_weights="none",
        val_interval=1,
    )
    settings.update(options)
    return module.Config(settings)


@pytest.mark.parametrize(
    "scheduler", ["CosineAnnealingLR", "ReduceLROnPlateau", "OneCycleLR"]
)
def test_training_resume_preserves_weights_lr_and_scheduler(
    monkeypatch, tmp_path, scheduler
):
    import shutil
    import manuscript.recognizers._trba.training.train as module
    from manuscript.recognizers._trba.training import utils
    from tests.recognizers.test_trba_training_smoke import _patch_runtime

    _patch_runtime(monkeypatch)
    monkeypatch.setattr(module, "set_seed", utils.set_seed)
    monkeypatch.setattr(module, "save_weights", utils.save_weights)
    snapshot = tmp_path / "epoch1.pth"
    records = []

    def save(path, *args):
        utils.save_checkpoint(path, *args)
        if Path(path).name == "last_ckpt.pth":
            records.append(torch.load(path, weights_only=False))
            if args[4] == 1:
                shutil.copy2(path, snapshot)

    monkeypatch.setattr(module, "save_checkpoint", save)
    cfg = training_config(tmp_path, scheduler=scheduler)
    module.run_training(cfg, device="cpu")
    expected = records[-1]
    monkeypatch.setattr(module, "load_checkpoint", utils.load_checkpoint)
    cfg.resume_from = str(snapshot)
    module.run_training(cfg, device="cpu")
    actual = records[-1]
    assert actual["scheduler_state"] == expected["scheduler_state"]
    assert (
        actual["optimizer_state"]["param_groups"]
        == expected["optimizer_state"]["param_groups"]
    )
    for name, value in expected["model_state"].items():
        torch.testing.assert_close(actual["model_state"][name], value, rtol=0, atol=0)


def test_save_interval_is_independent_of_validation(monkeypatch, tmp_path):
    import manuscript.recognizers._trba.training.train as module
    from tests.recognizers.test_trba_training_smoke import _patch_runtime

    _patch_runtime(monkeypatch)
    epochs = []
    original = module.save_checkpoint

    def save(path, *args):
        if Path(path).name == "last_ckpt.pth":
            epochs.append(args[4])
        original(path, *args)

    monkeypatch.setattr(module, "save_checkpoint", save)
    module.run_training(
        training_config(tmp_path, epochs=5, save_interval=2, val_interval=3),
        device="cpu",
    )
    assert epochs == [2, 4, 5]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_save_interval_is_rejected(monkeypatch, tmp_path, value):
    import manuscript.recognizers._trba.training.train as module
    from tests.recognizers.test_trba_training_smoke import _patch_runtime

    _patch_runtime(monkeypatch)
    with pytest.raises(ValueError, match="save_interval"):
        module.run_training(
            training_config(tmp_path, save_interval=value), device="cpu"
        )
