"""Reliability tests for CharLM helper and dataset modules."""

import importlib

import pytest

torch = pytest.importorskip("torch")


def _load_charlm_modules(monkeypatch):
    return (importlib.import_module("manuscript.correctors._charlm.utils"),
            importlib.import_module("manuscript.correctors._charlm.dataset"))


class FakeMLM:
    def __init__(self, c2i):
        self.c2i = c2i

    def eval(self):
        return self

    def __call__(self, batch):
        bsz, seqlen = batch.shape
        vocab = max(self.c2i.values()) + 1
        logits = torch.zeros(bsz, seqlen, vocab)
        for i in range(min(bsz, 3)):
            if i == 1:
                logits[i, i, self.c2i["a"]] = 10.0
                logits[i, i, self.c2i["o"]] = -10.0
            else:
                target = self.c2i["c"] if i == 0 else self.c2i["t"]
                logits[i, i, target] = 10.0
        return logits


class TestCharLMUtils:
    def test_build_vocab_encode_and_filter_helpers(self, monkeypatch, tmp_path):
        utils_module, _ = _load_charlm_modules(monkeypatch)

        c2i, i2c, chars = utils_module.build_vocab(["ab", "bc"], include_space=True)
        assert chars[:3] == ["<PAD>", "<MASK>", "<UNK>"]
        assert " " in c2i
        assert i2c[c2i["a"]] == "a"

        encoded = utils_module.encode_str("az", c2i, max_len=4)
        assert encoded[0] == c2i["a"]
        assert encoded[1] == c2i["<UNK>"]
        assert encoded[2:] == [c2i["<PAD>"], c2i["<PAD>"]]

        charset_path = tmp_path / "charset.txt"
        charset_path.write_text("a\nб\nab\n1\n", encoding="utf-8")
        allowed = utils_module.load_allowed_chars(str(charset_path))
        assert allowed == {"a", "б"}
        assert utils_module.filter_words(["ab!", "c", "bbb"], min_len=2, allowed_chars={"a", "b"}) == ["ab", "bbb"]

    def test_choose_spans_and_masked_accuracy(self, monkeypatch):
        utils_module, _ = _load_charlm_modules(monkeypatch)

        monkeypatch.setattr(utils_module.random, "randint", lambda a, b: a)
        spans = utils_module.choose_spans(8, span_min=1, span_max=2, spans_min=1, spans_max=1)
        assert spans
        assert min(spans) >= 1
        assert max(spans) <= 6
        assert utils_module.choose_spans(3, 1, 2, 1, 1) == []

        logits = torch.tensor([[[0.0, 1.0], [2.0, 0.0]]])
        targets = torch.tensor([[1, -100]])
        assert utils_module.masked_accuracy(logits, targets) == 1.0
        assert utils_module.masked_accuracy(logits, torch.full_like(targets, -100)) == 0.0

    def test_reconstruct_word_and_corrector_restore_case(self, monkeypatch):
        utils_module, _ = _load_charlm_modules(monkeypatch)
        c2i = {"<PAD>": 0, "<MASK>": 1, "<UNK>": 2, "c": 3, "a": 4, "o": 5, "t": 6}
        i2c = {v: k for k, v in c2i.items()}
        model = FakeMLM(c2i)

        result, trace, confidences = utils_module.reconstruct_word(
            model,
            "cot",
            c2i,
            i2c,
            device="cpu",
            max_len=5,
            mask_threshold=0.9,
            apply_threshold=0.9,
            max_edits=2,
            return_trace=True,
            return_p_cur=True,
        )
        assert result == "cat"
        assert trace
        assert confidences

        corrector = utils_module.CharLMCorrector(
            model=model,
            c2i=c2i,
            i2c=i2c,
            device="cpu",
            max_len=5,
            mask_threshold=0.9,
            apply_threshold=0.9,
            max_edits=2,
            min_word_len=2,
        )
        assert corrector.correct_word("Cot!") == "Cat!"

    def test_cer_and_evaluate_ocr_with_cer(self, monkeypatch, tmp_path):
        utils_module, _ = _load_charlm_modules(monkeypatch)

        class FakeCorrector:
            def correct_word(self, text, return_trace=False, return_p_cur=False):
                return ("cat", [{"pos": 1}], [(0, 0.1, None)]) if return_trace or return_p_cur else "cat"

        csv_path = tmp_path / "report.csv"
        stats = utils_module.evaluate_ocr_with_cer(
            FakeCorrector(),
            [("cot", "cat")],
            csv_path=str(csv_path),
        )

        assert stats["exact_match"] == 1.0
        assert stats["cer_after"] == 0.0
        assert csv_path.exists()


class TestCharLMDatasets:
    def test_ngram_dataset_masks_tokens(self, monkeypatch, tmp_path):
        utils_module, dataset_module = _load_charlm_modules(monkeypatch)
        text_path = tmp_path / "text.txt"
        text_path.write_text("one two three four", encoding="utf-8")
        c2i = {"<PAD>": 0, "<MASK>": 1, "<UNK>": 2, "o": 3, "n": 4, "e": 5, " ": 6, "t": 7, "w": 8, "h": 9, "r": 10, "f": 11, "u": 12}

        monkeypatch.setattr(dataset_module.random, "choices", lambda *a, **k: [1])
        monkeypatch.setattr(dataset_module.random, "randint", lambda a, b: a)
        monkeypatch.setattr(dataset_module.random, "random", lambda: 0.0)
        monkeypatch.setattr(dataset_module, "choose_spans", lambda *a, **k: [1])

        ds = dataset_module.NgramDataset(
            text_path=str(text_path),
            c2i=c2i,
            max_len=8,
            span_min=1,
            span_max=2,
            spans_min=1,
            spans_max=1,
            steps=3,
            mask_prob=1.0,
        )

        x, y = ds[0]
        assert len(ds) == 3
        assert x.shape == (8,)
        assert y.shape == (8,)
        assert (y != -100).any()
        assert x[1].item() == c2i["<MASK>"]

    def test_ngram_dataset_rejects_short_text(self, monkeypatch, tmp_path):
        _, dataset_module = _load_charlm_modules(monkeypatch)
        text_path = tmp_path / "text.txt"
        text_path.write_text("one two", encoding="utf-8")

        with pytest.raises(ValueError, match="Text too short"):
            dataset_module.NgramDataset(
                text_path=str(text_path),
                c2i={"<PAD>": 0, "<MASK>": 1, "<UNK>": 2},
                max_len=8,
                span_min=1,
                span_max=2,
                spans_min=1,
                spans_max=1,
            )

    def test_pairs_dataset_marks_differences(self, monkeypatch):
        _, dataset_module = _load_charlm_modules(monkeypatch)
        c2i = {"<PAD>": 0, "<MASK>": 1, "<UNK>": 2, "a": 3, "b": 4, "c": 5}

        ds = dataset_module.PairsDataset([("abc", "acc")], c2i=c2i, max_len=5)
        x, y = ds[0]

        assert x.tolist()[:3] == [3, 1, 5]
        assert y.tolist()[:3] == [-100, 5, -100]


@pytest.mark.parametrize("return_details", [False, True])
def test_reconstruct_word_preserves_text_outside_model_context(return_details):
    from manuscript.correctors._charlm.utils import reconstruct_word
    from unittest.mock import Mock
    model = Mock()
    result = reconstruct_word(model, "abcdefgh", {}, {}, "cpu", 4, .05, .95, 2,
                              return_trace=return_details)
    assert result == (("abcdefgh", [], []) if return_details else "abcdefgh")
    model.assert_not_called()
    model.eval.assert_not_called()


def test_pairs_dataset_excludes_pairs_without_targets_inside_context():
    from manuscript.correctors._charlm.dataset import PairsDataset
    from manuscript.correctors._charlm.utils import build_vocab

    c2i, _, _ = build_vocab(["abc"])
    dataset = PairsDataset(
        [("aaaab", "aaaac"), ("aaaa", "aaaa"), ("abca", "acca")], c2i, max_len=4,
    )
    assert len(dataset) == 1
    _, targets = dataset[0]
    assert (targets != -100).any()


def test_pairs_dataset_rejects_insertion_or_deletion_pairs():
    from manuscript.correctors._charlm.dataset import PairsDataset
    with pytest.raises(ValueError, match="equal lengths"):
        PairsDataset([("abc", "ab")], {}, max_len=4)


def test_torch_correction_preserves_unknown_characters():
    from manuscript.correctors._charlm.utils import build_vocab, reconstruct_word

    c2i, i2c, _ = build_vocab(["cat", "cot"])
    result, trace, confidences = reconstruct_word(
        FakeMLM(c2i), "?ot", c2i, i2c, "cpu", 5, .9, .9, 2,
        return_trace=True, return_p_cur=True,
    )
    assert result == "?at"
    assert all(item["pos"] != 0 for item in trace)
    assert all(position != 0 for position, _, _ in confidences)
