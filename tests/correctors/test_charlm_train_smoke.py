"""Smoke tests for CharLM training helpers and one-step training."""

import csv
import importlib
import json
import sys
from types import ModuleType

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn


def _load_charlm_train_module(monkeypatch):
    fake_lev = ModuleType("Levenshtein")

    def _distance(a, b):
        return abs(len(a) - len(b)) + sum(ca != cb for ca, cb in zip(a, b))

    fake_lev.distance = _distance
    monkeypatch.setitem(sys.modules, "Levenshtein", fake_lev)
    for name in (
        "manuscript.correctors._charlm.utils",
        "manuscript.correctors._charlm.dataset",
        "manuscript.correctors._charlm.train",
    ):
        sys.modules.pop(name, None)
    return importlib.import_module("manuscript.correctors._charlm.train")


class TinyMaskedDataset(torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[1], dict):
            self.c2i = args[1]
        else:
            self.c2i = kwargs["c2i"]
        if len(args) >= 3 and isinstance(args[2], int):
            self.max_len = args[2]
        else:
            self.max_len = kwargs["max_len"]
        self.steps = kwargs.get("steps")
        if self.steps is None:
            self.steps = args[8] if len(args) >= 9 else 1

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        x = torch.full((self.max_len,), self.c2i["<PAD>"], dtype=torch.long)
        y = torch.full((self.max_len,), -100, dtype=torch.long)
        x[0] = self.c2i["<MASK>"]
        y[0] = self.c2i["a"]
        return x, y


class TinyCharModel(nn.Module):
    def __init__(self, vocab_size, *args, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 4)
        self.proj = nn.Linear(4, vocab_size)

    def forward(self, x):
        return self.proj(self.embedding(x))


class FakeLogger:
    def __init__(self, path):
        self.path = path
        self.messages = []

    def log(self, message):
        self.messages.append(str(message))


class FakeCorrector:
    def __init__(self, *args, **kwargs):
        pass


class TestCharLMTrainHelpers:
    def test_load_pairs_filters_bad_rows(self, monkeypatch, tmp_path):
        train_module = _load_charlm_train_module(monkeypatch)
        pairs_path = tmp_path / "pairs.csv"
        with pairs_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["incorrect", "correct"])
            writer.writeheader()
            writer.writerow({"incorrect": "abce", "correct": "abde"})
            writer.writerow({"incorrect": "abc", "correct": "abc"})
            writer.writerow({"incorrect": "abcf", "correct": "zzzq"})
            writer.writerow({"incorrect": "short", "correct": "longer"})

        eval_pairs, train_pairs = train_module.load_pairs(
            str(pairs_path),
            allowed_chars={"a", "b", "c", "d", "e", "f"},
            eval_ratio=0.5,
            seed=0,
            max_edits=2,
        )

        assert eval_pairs == []
        assert train_pairs == [("abce", "abde")]

    def test_build_substitutions_counts_single_char_replacements(self, monkeypatch):
        train_module = _load_charlm_train_module(monkeypatch)

        result = train_module.build_substitutions(
            [
                ("cot", "cat"),
                ("cot", "cat"),
                ("dog", "dug"),
            ]
        )

        oa_key = next(key for key in result if key.startswith("o") and key.endswith("a"))
        ou_key = next(key for key in result if key.startswith("o") and key.endswith("u"))
        assert result[oa_key] == 2
        assert result[ou_key] == 1

    def test_mixed_dataset_switches_between_sources(self, monkeypatch):
        train_module = _load_charlm_train_module(monkeypatch)
        monkeypatch.setattr(train_module.random, "random", lambda: 0.2)
        monkeypatch.setattr(train_module.random, "randint", lambda a, b: b)

        mixed = train_module.MixedDataset(
            ngram_dataset=["ng0", "ng1"],
            pairs_dataset=["p0", "p1"],
            pairs_ratio=0.5,
            steps=3,
        )

        assert len(mixed) == 3
        assert mixed[0] == "p1"

    def test_train_runs_one_cpu_step_and_writes_checkpoint(self, monkeypatch, tmp_path):
        train_module = _load_charlm_train_module(monkeypatch)
        exp_dir = tmp_path / "exp"
        text_path = tmp_path / "text.txt"
        words_path = tmp_path / "words.txt"
        charset_path = tmp_path / "charset.txt"
        text_path.write_text("abba cab", encoding="utf-8")
        words_path.write_text("abba\ncab\n", encoding="utf-8")
        charset_path.write_text("a\nb\nc\n \n", encoding="utf-8")

        monkeypatch.setattr(train_module, "Logger", FakeLogger)
        monkeypatch.setattr(train_module, "load_allowed_chars", lambda path: {"a", "b", "c", " "})
        monkeypatch.setattr(
            train_module,
            "build_vocab",
            lambda words: (
                {"<PAD>": 0, "<MASK>": 1, "<UNK>": 2, "a": 3, "b": 4, "c": 5, " ": 6},
                {0: "<PAD>", 1: "<MASK>", 2: "<UNK>", 3: "a", 4: "b", 5: "c", 6: " "},
                ["<PAD>", "<MASK>", "<UNK>", "a", "b", "c", " "],
            ),
        )
        monkeypatch.setattr(train_module, "NgramDataset", TinyMaskedDataset)
        monkeypatch.setattr(train_module, "PairsDataset", TinyMaskedDataset)
        monkeypatch.setattr(train_module, "CharTransformerMLM", TinyCharModel)
        monkeypatch.setattr(train_module, "CharLMCorrector", FakeCorrector)
        monkeypatch.setattr(train_module, "log_random_examples", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            train_module,
            "DataLoader",
            lambda dataset, **kwargs: torch.utils.data.DataLoader(
                dataset,
                **{**kwargs, "pin_memory": False},
            ),
        )

        model, vocab = train_module.train(
            {
                "exp_dir": str(exp_dir),
                "words_path": str(words_path),
                "text_path": str(text_path),
                "pairs_path": None,
                "charset_path": str(charset_path),
                "epochs": 1,
                "batch_size": 1,
                "max_len": 6,
                "emb_size": 8,
                "n_layers": 1,
                "n_heads": 1,
                "ffn_size": 16,
                "dropout": 0.0,
                "steps_per_epoch": 1,
                "use_amp": False,
                "compile_model": False,
                "grad_clip": 0.0,
            }
        )

        assert isinstance(model, nn.Module)
        assert vocab[0]["a"] == 3
        assert json.loads((exp_dir / "vocab.json").read_text(encoding="utf-8"))[3] == "a"
        assert (exp_dir / "checkpoints" / "charlm_epoch_1.pt").exists()
