import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def _charlm_config(tmp_path, **overrides):
    (tmp_path / "words.txt").write_text("abba\ncaba\n")
    (tmp_path / "text.txt").write_text("abba caba abba")
    (tmp_path / "charset.txt").write_text("a\nb\nc\n")
    return dict(
        exp_dir=str(tmp_path / "exp"),
        words_path=str(tmp_path / "words.txt"),
        text_path=str(tmp_path / "text.txt"),
        pairs_path=None,
        charset_path=str(tmp_path / "charset.txt"),
        epochs=1,
        batch_size=1,
        max_len=8,
        emb_size=8,
        n_layers=1,
        n_heads=2,
        ffn_size=16,
        dropout=0.0,
        steps_per_epoch=1,
        use_amp=False,
        compile_model=False,
        grad_clip=0.0,
        **overrides,
    )


def test_charlm_one_batch_produces_an_optimizer_update(tmp_path):
    import torch
    from manuscript.correctors._charlm.train import train

    config = _charlm_config(tmp_path)
    train(config)
    checkpoint = torch.load(
        Path(config["exp_dir"]) / "checkpoints/charlm_epoch_1.pt", weights_only=True
    )
    assert checkpoint["optimizer"][
        "state"
    ], "No optimizer update with default accumulation_steps=2"


def test_charlm_text_only_training_retains_training_alphabet(tmp_path):
    from manuscript.correctors._charlm.train import train

    config = _charlm_config(tmp_path, accumulation_steps=1)
    config["words_path"] = None
    train(config)
    vocab = json.loads((Path(config["exp_dir"]) / "vocab.json").read_text())
    assert set("abc") <= set(vocab), vocab


def test_charlm_real_export_can_be_loaded_by_ort(tmp_path):
    pytest.importorskip("onnx")
    import torch
    import onnxruntime as ort
    from manuscript.correctors import CharLM
    from manuscript.correctors._charlm.model import CharTransformerMLM

    vocab = ["<PAD>", "<MASK>", "<UNK>", "a", "b", " "]
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))
    model = CharTransformerMLM(6, 8, 8, 1, 2, 16, 0, 0).eval()
    torch.save({"model": model.state_dict()}, tmp_path / "model.pt")
    CharLM.export(
        tmp_path / "model.pt",
        tmp_path / "vocab.json",
        tmp_path / "model.onnx",
        max_len=8,
        emb_size=8,
        n_layers=1,
        n_heads=2,
        ffn_size=16,
        simplify=False,
    )
    session = ort.InferenceSession(
        str(tmp_path / "model.onnx"), providers=["CPUExecutionProvider"]
    )
    for batch in (1, 2):
        ids = torch.randint(1, 6, (batch, 8))
        with torch.no_grad():
            expected = model(ids).numpy()
        actual = session.run(None, {"input": ids.numpy()})[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("batches", [1, 2, 3])
def test_accumulation_updates_weights_and_advances_schedule(
    tmp_path, monkeypatch, batches
):
    import manuscript.correctors._charlm.train as module

    config = _charlm_config(tmp_path)
    config["steps_per_epoch"] = batches
    created = []
    original = module.CharTransformerMLM

    def create(*args, **kwargs):
        model = original(*args, **kwargs)
        created.append(
            {key: value.clone() for key, value in model.state_dict().items()}
        )
        return model

    monkeypatch.setattr(module, "CharTransformerMLM", create)
    model, _ = module.train(config)
    saved = torch.load(
        Path(config["exp_dir"]) / "checkpoints/charlm_epoch_1.pt", weights_only=True
    )
    steps = (batches + 1) // 2
    assert all(
        int(state["step"]) == steps for state in saved["optimizer"]["state"].values()
    )
    assert saved["scheduler"]["last_epoch"] == steps
    assert saved["optimizer"]["param_groups"][0]["lr"] == pytest.approx(0.0)
    assert any(
        not torch.equal(value, created[0][key])
        for key, value in model.state_dict().items()
    )


def test_charlm_resume_matches_uninterrupted_training(tmp_path):
    from manuscript.correctors._charlm.train import train

    config = _charlm_config(tmp_path, accumulation_steps=2)
    config.update(epochs=2, steps_per_epoch=3)
    model, _ = train(config)
    directory = Path(config["exp_dir"])
    expected = torch.load(
        directory / "checkpoints/charlm_epoch_2.pt", weights_only=True
    )
    resumed_config = {
        **config,
        "checkpoint": str(directory / "checkpoints/charlm_epoch_1.pt"),
        "exp_dir": str(tmp_path / "resumed"),
    }
    resumed, _ = train(resumed_config)
    actual = torch.load(
        tmp_path / "resumed/checkpoints/charlm_epoch_2.pt", weights_only=True
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)
    assert actual["scheduler"] == expected["scheduler"]
    assert actual["optimizer"]["param_groups"] == expected["optimizer"]["param_groups"]
    assert actual["vocab"] == expected["vocab"]


def test_charlm_pairs_only_uses_charset_vocabulary(tmp_path):
    from manuscript.correctors._charlm.train import train

    config = _charlm_config(tmp_path, accumulation_steps=1)
    pairs = tmp_path / "pairs.csv"
    pairs.write_text("incorrect,correct\nabba,abca\n", encoding="utf-8")
    config.update(
        words_path=None, text_path=None, pairs_path=str(pairs), eval_ratio=0.0
    )
    model, (c2i, _, _) = train(config)
    assert all(c2i[letter] != c2i["<UNK>"] for letter in "abc")
    assert torch.isfinite(model(torch.tensor([[c2i["a"]] * 8]))).all()


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_accumulation_fails_before_training(tmp_path, value):
    from manuscript.correctors._charlm.train import train

    with pytest.raises(ValueError, match="accumulation_steps"):
        train({"exp_dir": str(tmp_path / "exp"), "accumulation_steps": value})
    assert not (tmp_path / "exp").exists()


def test_accumulation_matches_batches_including_the_final_partial_group(
    tmp_path, monkeypatch
):
    import manuscript.correctors._charlm.train as module
    from tests.correctors.test_charlm_train_smoke import TinyMaskedDataset

    monkeypatch.setattr(module, "NgramDataset", TinyMaskedDataset)
    config = _charlm_config(tmp_path)
    config.update(steps_per_epoch=3, accumulation_steps=2, weight_decay=0.0)
    accumulated, _ = module.train(config)
    batched_config = {
        **config,
        "exp_dir": str(tmp_path / "batched"),
        "batch_size": 2,
        "accumulation_steps": 1,
    }
    batched, _ = module.train(batched_config)
    for name, value in batched.state_dict().items():
        torch.testing.assert_close(
            accumulated.state_dict()[name], value, rtol=1e-5, atol=1e-6
        )


def test_pairs_only_training_rejects_pairs_without_context_targets(tmp_path):
    from manuscript.correctors._charlm.train import train

    config = _charlm_config(tmp_path)
    pairs = tmp_path / "pairs.csv"
    pairs.write_text("incorrect,correct\naaaab,aaaac\n", encoding="utf-8")
    config.update(words_path=None, text_path=None, pairs_path=str(pairs),
                  eval_ratio=0.0, max_len=4)
    with pytest.raises(ValueError, match="No datasets"):
        train(config)
    assert not (Path(config["exp_dir"]) / "checkpoints/charlm_epoch_1.pt").exists()


@pytest.mark.parametrize("name", ["max_len", "steps_per_epoch"])
def test_empty_training_context_or_epoch_is_rejected(tmp_path, name):
    from manuscript.correctors._charlm.train import train

    with pytest.raises(ValueError, match=name):
        train({"exp_dir": str(tmp_path / "exp"), name: 0})
    assert not (tmp_path / "exp").exists()
