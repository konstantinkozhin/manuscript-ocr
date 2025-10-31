import numpy as np
from pathlib import Path

import manuscript.detectors._east.infer as infer
from manuscript.detectors import EAST
from manuscript.detectors._types import Word, Block, Page


def _make_page(text: str) -> Page:
    word = Word(
        polygon=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        detection_confidence=0.95,
        text=text,
        recognition_confidence=0.9,
    )
    block = Block(words=[word])
    return Page(blocks=[block])


def test_detector_doc_single_image(monkeypatch):
    init_called = {}

    def fake_init(self, *args, **kwargs):
        init_called.update(kwargs)

    def fake_predict(self, image, vis=True, return_maps=False, profile=False):
        assert image == "data/samples/page_001.jpg"
        assert vis is True
        return {
            "page": _make_page("hello"),
            "vis_image": None,
            "score_map": None,
            "geo_map": None,
        }

    monkeypatch.setattr(infer.EAST, "__init__", fake_init, raising=False)
    monkeypatch.setattr(infer.EAST, "predict", fake_predict, raising=False)

    detector = EAST(score_thresh=0.75)
    result = detector.predict("data/samples/page_001.jpg", vis=True)
    page = result["page"]
    vis_image = result["vis_image"]

    assert page.blocks[0].words[0].text == "hello"
    assert vis_image is None
    assert init_called.get("score_thresh") == 0.75


def test_detector_doc_folder_processing(monkeypatch, tmp_path):
    samples = {
        "first.jpg": "first",
        "second.jpg": "second",
    }

    for name in samples:
        (tmp_path / name).write_bytes(b"")

    def fake_init(self, *args, **kwargs):
        pass

    def fake_predict(self, image, vis=True, return_maps=False, profile=False):
        name = Path(image).name
        return {
            "page": _make_page(samples[name]),
            "vis_image": None,
            "score_map": None,
            "geo_map": None,
        }

    monkeypatch.setattr(infer.EAST, "__init__", fake_init, raising=False)
    monkeypatch.setattr(infer.EAST, "predict", fake_predict, raising=False)

    detector = EAST(score_thresh=0.6)
    texts = {}
    for image_path in tmp_path.glob("*.jpg"):
        result = detector.predict(image_path)
        page = result["page"]
        text = " ".join((word.text or "") for block in page.blocks for word in block.words)
        texts[image_path.name] = text

    assert texts == samples


def test_detector_doc_custom_weights(monkeypatch, tmp_path):
    weights_path = tmp_path / "east_weights.pth"
    weights_path.write_bytes(b"dummy")

    init_kwargs = {}

    def fake_init(self, *args, **kwargs):
        init_kwargs.update(kwargs)

    def fake_predict(self, image, vis=True, return_maps=False, profile=False):
        score_map = np.zeros((2, 2), dtype=np.float32) if return_maps else None
        geo_map = np.zeros((2, 2, 5), dtype=np.float32) if return_maps else None
        return {
            "page": _make_page("custom"),
            "vis_image": None,
            "score_map": score_map,
            "geo_map": geo_map,
        }

    monkeypatch.setattr(infer.EAST, "__init__", fake_init, raising=False)
    monkeypatch.setattr(infer.EAST, "predict", fake_predict, raising=False)

    detector = EAST(
        weights_path=weights_path,
        target_size=1024,
        score_thresh=0.5,
        expand_ratio_w=0.85,
        expand_ratio_h=0.85,
    )

    result = detector.predict("data/custom/page.png", vis=False, return_maps=True)

    assert init_kwargs["weights_path"] == weights_path
    assert result["score_map"].shape == (2, 2)
    assert result["geo_map"].shape == (2, 2, 5)


def test_detector_doc_training(monkeypatch, tmp_path):
    captured = {}

    def fake_train(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "best-model-path"

    monkeypatch.setattr(
        infer.EAST, "train", staticmethod(fake_train), raising=False
    )

    train_images = [
        "data/train_main/images",
        "data/train_extra/images",
    ]
    train_anns = [
        "data/train_main/annotations.json",
        "data/train_extra/annotations.json",
    ]
    val_images = ["data/val/images"]
    val_anns = ["data/val/annotations.json"]

    best_model = EAST.train(
        train_images=train_images,
        train_anns=train_anns,
        val_images=val_images,
        val_anns=val_anns,
        experiment_root="./experiments",
        model_name="east_doc_example",
        target_size=1024,
        epochs=20,
        batch_size=4,
        score_geo_scale=None,
    )

    assert best_model == "best-model-path"
    assert captured["args"] == ()
    assert captured["kwargs"]["train_images"] == train_images
    assert captured["kwargs"]["train_anns"] == train_anns
    assert captured["kwargs"]["val_images"] == val_images
    assert captured["kwargs"]["val_anns"] == val_anns
    assert captured["kwargs"]["experiment_root"] == "./experiments"
    assert captured["kwargs"]["model_name"] == "east_doc_example"
