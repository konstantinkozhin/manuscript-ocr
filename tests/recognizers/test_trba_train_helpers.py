"""Reliability tests for TRBA training helpers."""

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import Dataset

import manuscript.recognizers._trba.training.train as train_module


class FakeOCRDataset(Dataset):
    def __init__(self, _csv_path, _root, _stoi, img_height, img_max_width, transform, encoding):
        self.items = [0] * 10
        self.img_height = img_height
        self.img_max_width = img_max_width
        self.transform = transform
        self.encoding = encoding

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class TinyValidationModel(torch.nn.Module):
    def __init__(self, attention_preds):
        super().__init__()
        self.attention_preds = attention_preds

    def forward(self, imgs, text=None, is_train=True, batch_max_length=None):
        batch_size = imgs.size(0)
        seq_len = self.attention_preds.size(1)
        vocab_size = 6
        if is_train:
            attention_logits = torch.zeros(batch_size, seq_len, vocab_size)
            ctc_logits = torch.zeros(batch_size, seq_len, vocab_size)
            for row_idx in range(batch_size):
                for col_idx in range(seq_len):
                    token_id = int(text[row_idx, col_idx].item())
                    attention_logits[row_idx, col_idx, token_id] = 5.0
                    ctc_logits[row_idx, col_idx, token_id] = 5.0
            return {
                "attention_logits": attention_logits,
                "ctc_logits": ctc_logits,
            }
        return {"attention_preds": self.attention_preds.clone()}

    def compute_ctc_loss(self, _ctc_logits, _target_y, _lengths):
        return torch.tensor(0.25)


class FakeWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, float(value), step))


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(str(message))


class TestCTCDecode:
    def test_ctc_greedy_decode_removes_repeats_and_blanks(self):
        logits = torch.tensor(
            [
                [
                    [9.0, 0.0, 0.0],
                    [0.0, 9.0, 0.0],
                    [0.0, 9.0, 0.0],
                    [0.0, 0.0, 9.0],
                    [9.0, 0.0, 0.0],
                    [0.0, 0.0, 9.0],
                ],
                [
                    [9.0, 0.0, 0.0],
                    [0.0, 9.0, 0.0],
                    [9.0, 0.0, 0.0],
                    [0.0, 9.0, 0.0],
                    [9.0, 0.0, 0.0],
                    [9.0, 0.0, 0.0],
                ],
            ]
        )

        decoded = train_module.ctc_greedy_decode(logits, blank_id=0)

        assert decoded.tolist() == [[1, 2, 2], [1, 1, -1]]

    def test_get_ctc_weight_for_epoch_decays_linearly(self):
        assert train_module.get_ctc_weight_for_epoch(1, initial_weight=0.3, decay_epochs=4) == 0.3
        assert train_module.get_ctc_weight_for_epoch(3, initial_weight=0.3, decay_epochs=4) == 0.15
        assert train_module.get_ctc_weight_for_epoch(10, initial_weight=0.3, decay_epochs=4) == 0.0
        assert train_module.get_ctc_weight_for_epoch(5, initial_weight=0.2, decay_epochs=0) == 0.2


class TestConfig:
    def test_config_resume_from_directory_merges_saved_config_and_overrides(self, tmp_path):
        exp_dir = tmp_path / "exp1"
        exp_dir.mkdir()
        (exp_dir / "last_ckpt.pth").write_bytes(b"weights")
        (exp_dir / "config.json").write_text(
            json.dumps({"epochs": 5, "img_h": 32, "keep_me": True}),
            encoding="utf-8",
        )

        cfg = train_module.Config(
            {
                "resume_from": str(exp_dir),
                "epochs": None,
                "img_h": 64,
                "new_flag": "x",
            }
        )

        assert cfg.resume_from == str(exp_dir / "last_ckpt.pth")
        assert cfg.exp_dir == str(exp_dir)
        assert cfg.epochs == 5
        assert cfg.img_h == 64
        assert cfg.keep_me is True
        assert cfg.new_flag == "x"

    def test_config_resume_requires_existing_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Путь для резюме не найден"):
            train_module.Config({"resume_from": str(tmp_path / "missing")})

    def test_config_save_writes_json(self, tmp_path):
        cfg = train_module.Config({"exp_dir": str(tmp_path / "exp2"), "epochs": 2})
        out_path = tmp_path / "saved" / "config.json"

        cfg.save(str(out_path))

        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["epochs"] == 2


class TestSplitTrainVal:
    def test_split_train_val_builds_subsets_and_applies_transforms(self, monkeypatch):
        monkeypatch.setattr(train_module, "OCRDatasetAttn", FakeOCRDataset)
        train_transform = object()
        val_transform = object()

        train_sets, val_sets = train_module.split_train_val(
            csvs=["train.csv"],
            roots=["images"],
            stoi={"a": 1},
            img_h=32,
            img_w=128,
            train_transform=train_transform,
            val_transform=val_transform,
            val_size=3,
        )

        assert len(train_sets) == 1
        assert len(val_sets) == 1
        assert len(train_sets[0]) == 7
        assert len(val_sets[0]) == 3
        assert train_sets[0].dataset.transform is train_transform
        assert val_sets[0].dataset.transform is val_transform

    def test_split_train_val_rejects_dataset_smaller_than_val_split(self, monkeypatch):
        class TinyDataset(FakeOCRDataset):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.items = [0, 1]

        monkeypatch.setattr(train_module, "OCRDatasetAttn", TinyDataset)

        with pytest.raises(ValueError, match="[Вв] датасете"):
            train_module.split_train_val(
                csvs=["tiny.csv"],
                roots=["images"],
                stoi={"a": 1},
                img_h=32,
                img_w=128,
                train_transform=None,
                val_transform=None,
                val_size=3,
            )


class TestRunValidation:
    def test_run_validation_aggregates_metrics_and_logs(self, monkeypatch):
        monkeypatch.setattr(train_module, "_autocast", contextlib.nullcontext)
        monkeypatch.setattr(train_module.torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(train_module, "compute_cer", lambda refs, hyps: 0.0)
        monkeypatch.setattr(train_module, "compute_wer", lambda refs, hyps: 0.0)

        writer = FakeWriter()
        logger = FakeLogger()
        itos = ["<PAD>", "<SOS>", "<EOS>", "<BLANK>", "a", "b"]
        stoi = {token: idx for idx, token in enumerate(itos)}
        target_y = torch.tensor([[4, 2, 0], [5, 2, 0]])
        text_in = torch.tensor([[1, 4, 0], [1, 5, 0]])
        imgs = torch.zeros(2, 3, 8, 16)
        lengths = torch.tensor([2, 2])
        val_loaders = [[(imgs, text_in, target_y, lengths)]]
        model = TinyValidationModel(attention_preds=target_y.clone())

        result = train_module.run_validation(
            model=model,
            val_loaders=val_loaders,
            criterion=lambda logits, target: torch.tensor(0.5),
            itos=itos,
            stoi=stoi,
            device=torch.device("cpu"),
            pin_memory=False,
            max_len=3,
            ctc_weight=0.25,
            epoch=1,
            writer=writer,
            logger=logger,
        )

        assert result["avg_loss"] == pytest.approx(0.4375)
        assert result["acc"] == 1.0
        assert result["cer"] == 0.0
        assert result["wer"] == 0.0
        assert any(tag == "Loss/val_epoch" for tag, _, _ in writer.scalars)
        assert any("TOTAL" in message for message in logger.messages)
