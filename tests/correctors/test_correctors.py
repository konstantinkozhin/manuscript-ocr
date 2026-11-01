"""Deterministic CharLM inference tests: no downloads or real model required."""
import json
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from manuscript.correctors import CharLM
from manuscript.data import Page, TextRegion


@pytest.fixture
def corrector(tmp_path):
    weights = tmp_path / "model.onnx"
    vocab = tmp_path / "vocab.json"
    weights.write_bytes(b"test")
    vocab.write_text(json.dumps(["<PAD>", "<MASK>", "<UNK>", *"abcdefghijklmnopqrstuvwxyz"]))
    model = CharLM(weights=weights, vocab=vocab, device="cpu", min_word_len=3, max_len=8)
    model.onnx_session = Mock()
    model.onnx_session.get_inputs.return_value = [SimpleNamespace(name="input")]
    model.onnx_session.get_outputs.return_value = [SimpleNamespace(name="logits")]
    return model


def set_predictions(model, word, replacements):
    """Each masked position predicts its source character unless overridden."""
    size = len(model.c2i)
    logits = np.full((len(word), model.max_len, size), -20., dtype=np.float32)
    for i, char in enumerate(word):
        logits[i, i, model.c2i.get(char, 2)] = 20.
    for i, char in replacements.items():
        logits[i, i] = -20.
        logits[i, i, model.c2i[char]] = 20.
    model.onnx_session.run.return_value = [logits]


def test_local_artifacts_load_vocab_and_normalize_lexicon(corrector, tmp_path):
    assert corrector.c2i["<PAD>"] == 0
    vocab = tmp_path / "vocab.json"
    model = CharLM(weights=corrector.weights, vocab=vocab, lexicon={"CAT", "dog"})
    assert model.lexicon == frozenset({"cat", "dog"})
    lexicon = tmp_path / "words.txt"
    lexicon.write_text("CAT\n\ndog\n")
    model = CharLM(weights=corrector.weights, vocab=vocab, lexicon=lexicon)
    assert model.lexicon == frozenset({"cat", "dog"})


def test_default_preset_supplies_its_own_artifacts(corrector, monkeypatch):
    resolve = Mock(return_value={"weights": corrector.weights, "vocab": corrector.vocab_path})
    monkeypatch.setattr("manuscript.models.resolve", resolve)
    monkeypatch.setattr("manuscript.models.info", lambda *args: {"artifacts": {}})
    model = CharLM(device="cpu")
    resolve.assert_called_once_with("prereform_charlm_g1", "CharLM", force_download=False)
    assert model.vocab_path == corrector.vocab_path
    assert model.c2i == corrector.c2i


def test_lazy_session_created_once(corrector, monkeypatch):
    corrector.onnx_session = None
    constructor = Mock(return_value=Mock())
    monkeypatch.setattr("onnxruntime.InferenceSession", constructor)
    corrector._initialize_session()
    corrector._initialize_session()
    constructor.assert_called_once_with(corrector.weights, providers=["CPUExecutionProvider"])


def test_prediction_preserves_tree_and_selected_levels(corrector):
    page = Page(image_width=10, image_height=10)
    line = TextRegion(type="line", text="cot")
    word = TextRegion(type="word", text="Cot!", detection_confidence=.8, recognition_confidence=.9)
    line.children = [word, TextRegion(type="word")]
    page.children = [line]
    set_predictions(corrector, "cot", {1: "a"})
    result = corrector.predict(page, image=np.zeros((10, 10, 3), np.uint8))
    output = result.children[0].children[0]
    assert output.text == "Cat!" and line.text == "cot" and word.text == "Cot!"
    assert output.parent is result.children[0]
    assert (output.detection_confidence, output.recognition_confidence) == (.8, .9)
    assert result.children[0].children[1].text is None


def test_empty_page_does_not_initialize_backend(corrector):
    corrector.onnx_session = None
    corrector._initialize_session = Mock()
    result = corrector.predict(Page(image_width=10, image_height=10))
    assert not result.children
    corrector._initialize_session.assert_not_called()


@pytest.mark.parametrize("text", ["Cat!", "CAT!", "cAt!"])
def test_case_and_punctuation_are_preserved(corrector, text):
    set_predictions(corrector, "cat", {1: "o"})
    assert corrector._correct_word(text) == text.replace("a", "o").replace("A", "O")


def test_short_and_dictionary_words_skip_inference(corrector):
    corrector.lexicon = frozenset({"cat"})
    assert corrector._correct_word("Cat, ox!") == "Cat, ox!"
    corrector.onnx_session.run.assert_not_called()


@pytest.mark.parametrize("max_edits,expected", [(0, "cot"), (1, "cat"), (2, "car")])
def test_edit_limit(corrector, max_edits, expected):
    corrector.max_edits = max_edits
    set_predictions(corrector, "cot", {1: "a", 2: "r"})
    assert corrector._correct_single_word("cot") == expected


@pytest.mark.parametrize("setting,value", [("mask_threshold", 0), ("apply_threshold", .95)])
def test_thresholds_can_prevent_edits(corrector, setting, value):
    setattr(corrector, setting, value)
    set_predictions(corrector, "cot", {1: "a"})
    if setting == "apply_threshold":
        corrector.mask_threshold = .2
        logits = corrector.onnx_session.run.return_value[0]
        logits[1, 1, corrector.c2i["a"]] = 2.
        logits[1, 1, corrector.c2i["o"]] = 0.
    assert corrector._correct_single_word("cot") == "cot"


@pytest.mark.parametrize("lexicon,expected", [({"cat"}, "cat"), ({"dog"}, "cot")])
def test_candidate_must_belong_to_lexicon(corrector, lexicon, expected):
    corrector.lexicon = frozenset(lexicon)
    set_predictions(corrector, "cot", {1: "a"})
    assert corrector._correct_single_word("cot") == expected


@pytest.mark.parametrize("token", ["<PAD>", "<MASK>", "<UNK>"])
def test_special_tokens_are_never_inserted(corrector, token):
    set_predictions(corrector, "cot", {1: token})
    assert corrector._correct_single_word("cot") == "cot"


def test_unknown_character_is_not_replaced(corrector):
    set_predictions(corrector, "c?t", {1: "a"})
    assert corrector._correct_single_word("c?t") == "c?t"


def test_long_word_is_preserved_without_inference(corrector):
    word = "abcdefghijklmnopqrstuvwxyz"
    assert corrector._correct_single_word(word) == word
    assert corrector._correct_word(word.upper()) == word.upper()
    corrector.onnx_session.run.assert_not_called()


def test_inference_failure_warns_and_preserves_text(corrector):
    corrector.onnx_session.run.side_effect = RuntimeError("vocab mismatch")
    with pytest.warns(RuntimeWarning, match="Returning original text"):
        assert corrector._correct_word("Cot!") == "Cot!"


def test_local_corrector_does_not_use_default_preset_vocab(tmp_path):
    model = object.__new__(CharLM)
    model.weights = str(tmp_path / "model.onnx")
    (tmp_path / "vocab.json").write_text('{}')
    with pytest.raises(ValueError, match="explicit vocab"):
        model._resolve_vocab(None)



def test_line_correction_override_does_not_correct_child_words(corrector):
    corrector._configure_correction_levels(["line"])
    page = Page(image_width=10, image_height=10)
    line = TextRegion(type="line", text="Cot")
    word = TextRegion(type="word", text="Cot")
    line.children = [word]
    page.children = [line]
    set_predictions(corrector, "cot", {1: "a"})
    result = corrector.predict(page)
    assert result.children[0].text == "Cat"
    assert result.children[0].children[0].text == "Cot"
    assert line.text == "Cot" and word.text == "Cot"
