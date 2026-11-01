from tests.support.tree import make_page, make_region
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
import types
from PIL import Image
from manuscript.recognizers import TRBA
from manuscript.data import Page


class TestTRBAInitialization:
    """Tests for TRBA initialization"""



    def test_trba_default_weights(self):
        """Test default preset"""
        assert TRBA.default_weights_name == "trba_lite_g1"

    def test_trba_lite_g2_preset_registered(self):
        """TRBA exposes trba_lite_g2 across all preset registries."""
        from manuscript.models import info

        assert "TRBA" in info("trba_lite_g2")["model_classes"]

    @patch("manuscript.api.base.BaseArtifactModel._download_http")
    def test_trba_initialization_with_local_file(self, mock_download, tmp_path):
        """Test initialization with local files"""
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model.json"
        charset_file = tmp_path / "model.txt"
        weights_file.write_text("mock_onnx")
        config_file.write_text(
            '{"recognition_levels": ["word"], "max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}'
        )
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
        recognizer = TRBA(
            weights=str(weights_file),
            config=str(config_file),
            charset=str(charset_file),
            device="cpu",
        )
        assert recognizer.weights == str(weights_file.absolute())
        assert recognizer.device == "cpu"
        assert recognizer.config_path == str(config_file.absolute())
        assert recognizer.charset_path == str(charset_file.absolute())
        mock_download.assert_not_called()

    def test_trba_device_auto_selection(self, tmp_path):
        """Test automatic device selection"""
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model.json"
        charset_file = tmp_path / "model.txt"
        weights_file.write_text("mock_onnx")
        config_file.write_text(
            '{"recognition_levels": ["word"], "max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}'
        )
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
        recognizer = TRBA(
            weights=str(weights_file),
            config=str(config_file),
            charset=str(charset_file),
            device=None,
        )
        assert recognizer.device in ["cpu", "cuda"]

    def test_trba_explicit_cpu_device(self, tmp_path):
        """Test explicit CPU selection"""
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model.json"
        charset_file = tmp_path / "model.txt"
        weights_file.write_text("mock_onnx")
        config_file.write_text(
            '{"recognition_levels": ["word"], "max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}'
        )
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
        recognizer = TRBA(
            weights=str(weights_file),
            config=str(config_file),
            charset=str(charset_file),
            device="cpu",
        )
        assert recognizer.device == "cpu"


class TestTRBAConfigResolution:
    """Tests for config and charset file resolution"""

    def test_config_and_charset_explicit_paths(self, tmp_path):
        """Test explicit config and charset paths"""
        weights_file = tmp_path / "my_model.onnx"
        config_file = tmp_path / "my_model.json"
        charset_file = tmp_path / "my_model.txt"
        weights_file.write_text("mock_onnx")
        config_file.write_text(
            '{"recognition_levels": ["word"], "max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}'
        )
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
        recognizer = TRBA(weights=str(weights_file), device="cpu", config=Path(str(weights_file)).with_suffix(".json"), charset=Path(str(weights_file)).with_suffix(".txt"))
        assert recognizer.config_path == str(config_file.absolute())
        assert recognizer.charset_path == str(charset_file.absolute())

    def test_config_missing_fails(self, tmp_path):
        weights = tmp_path / "model.onnx"
        weights.write_bytes(b"test")
        with pytest.raises(FileNotFoundError, match="config"):
            TRBA(weights=str(weights), device="cpu", config=Path(str(weights)).with_suffix(".json"), charset=Path(str(weights)).with_suffix(".txt"))

    def test_charset_missing_fails(self, tmp_path):
        weights = tmp_path / "model.onnx"
        weights.write_bytes(b"test")
        weights.with_suffix(".json").write_text('{"recognition_levels":["word"],"img_h":32,"img_w":256}')
        with pytest.raises(FileNotFoundError, match="charset"):
            TRBA(weights=str(weights), device="cpu", config=Path(str(weights)).with_suffix(".json"), charset=Path(str(weights)).with_suffix(".txt"))

    def test_explicit_charset_parameter(self, tmp_path):
        """Test that explicit charset parameter is used when provided"""
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model.json"
        charset_file = tmp_path / "custom_charset.txt"
        weights_file.write_text("mock_onnx")
        config_file.write_text(
            '{"recognition_levels": ["word"], "max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}'
        )
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
        recognizer = TRBA(
            weights=str(weights_file),
            config=str(config_file),
            charset=str(charset_file),
            device="cpu",
        )
        assert recognizer.charset_path == str(charset_file.absolute())


class TestTRBAPreprocessing:
    """Tests for image preprocessing"""

    def test_preprocess_image_from_numpy(self, tmp_path):
        """Test preprocessing from numpy array"""
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model.json"
        charset_file = tmp_path / "model.txt"
        weights_file.write_text("mock_onnx")
        config_file.write_text(
            '{"recognition_levels": ["word"], "max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}'
        )
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
        recognizer = TRBA(
            weights=str(weights_file),
            config=str(config_file),
            charset=str(charset_file),
            device="cpu",
        )
        img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        preprocessed = recognizer._preprocess_image(img)
        assert preprocessed.shape == (1, 3, 64, 256)
        assert preprocessed.dtype == np.float32
        assert preprocessed.min() >= -1.5
        assert preprocessed.max() <= 1.5

    def test_preprocess_image_from_pil(self, tmp_path):
        """Test preprocessing from PIL Image"""
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model.json"
        charset_file = tmp_path / "model.txt"
        weights_file.write_text("mock_onnx")
        config_file.write_text(
            '{"recognition_levels": ["word"], "max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}'
        )
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
        recognizer = TRBA(
            weights=str(weights_file),
            config=str(config_file),
            charset=str(charset_file),
            device="cpu",
        )
        pil_img = Image.new("RGB", (200, 100), color=(255, 0, 0))
        preprocessed = recognizer._preprocess_image(pil_img)
        assert preprocessed.shape == (1, 3, 64, 256)
        assert preprocessed.dtype == np.float32

    def test_prepare_crop_rotates_vertical_image(self, tmp_path):
        """Test that TRBA rotates vertical crops before recognition."""
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model.json"
        charset_file = tmp_path / "model.txt"
        weights_file.write_text("mock_onnx")
        config_file.write_text(
            '{"recognition_levels": ["word"], "max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}'
        )
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
        recognizer = TRBA(
            weights=str(weights_file),
            config=str(config_file),
            charset=str(charset_file),
            device="cpu",
            rotate_threshold=1.5,
        )
        vertical_crop = np.zeros((100, 50, 3), dtype=np.uint8)
        vertical_crop[0, 0] = [255, 0, 0]
        result = recognizer._prepare_crop(vertical_crop)
        assert result.shape == (50, 100, 3)
        assert np.array_equal(result[0, 99], [255, 0, 0])

    def test_prepare_crop_can_be_disabled(self, tmp_path):
        """Test that rotation can be disabled in TRBA."""
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model.json"
        charset_file = tmp_path / "model.txt"
        weights_file.write_text("mock_onnx")
        config_file.write_text(
            '{"recognition_levels": ["word"], "max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}'
        )
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
        recognizer = TRBA(
            weights=str(weights_file),
            config=str(config_file),
            charset=str(charset_file),
            device="cpu",
            rotate_threshold=0,
        )
        vertical_crop = np.zeros((100, 50, 3), dtype=np.uint8)
        result = recognizer._prepare_crop(vertical_crop)
        assert result.shape == (100, 50, 3)

    def test_preprocess_does_not_use_prepare_crop(self, tmp_path):
        """Prepared-crop orientation should happen before model preprocessing."""
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model.json"
        charset_file = tmp_path / "model.txt"
        weights_file.write_text("mock_onnx")
        config_file.write_text(
            '{"recognition_levels": ["word"], "max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}'
        )
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
        recognizer = TRBA(
            weights=str(weights_file),
            config=str(config_file),
            charset=str(charset_file),
            device="cpu",
        )
        vertical_crop = np.zeros((100, 50, 3), dtype=np.uint8)
        with patch.object(
            recognizer, "_prepare_crop", wraps=recognizer._prepare_crop
        ) as mocked:
            recognizer._preprocess_image(vertical_crop)
            mocked.assert_not_called()


class TestTRBAAPI:
    """Tests for the public API"""

    def test_trba_callable(self, tmp_path):
        """Test that TRBA can be called as a function (via __call__)"""
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model.json"
        charset_file = tmp_path / "model.txt"
        weights_file.write_text("mock_onnx")
        config_file.write_text(
            '{"recognition_levels": ["word"], "max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}'
        )
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
        recognizer = TRBA(
            weights=str(weights_file),
            config=str(config_file),
            charset=str(charset_file),
            device="cpu",
        )
        assert callable(recognizer)



class TestTRBAPredictPageInterface:
    """Tests for Page-based recognizer API."""

    def _create_recognizer(self, tmp_path, min_text_size: int = 5, **kwargs):
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model.json"
        charset_file = tmp_path / "model.txt"
        weights_file.write_text("mock_onnx")
        config_file.write_text(
            '{"recognition_levels": ["word"], "max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}'
        )
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
        return TRBA(
            weights=str(weights_file),
            config=str(config_file),
            charset=str(charset_file),
            device="cpu",
            min_text_size=min_text_size,
            **kwargs,
        )

    @staticmethod
    def _create_page() -> Page:
        words = [
            make_region(
                "word",
                polygon=[(10.0, 10.0), (80.0, 10.0), (80.0, 40.0), (10.0, 40.0)],
                detection_confidence=0.95,
            ),
            make_region(
                "word",
                polygon=[(90.0, 10.0), (160.0, 10.0), (160.0, 40.0), (90.0, 40.0)],
                detection_confidence=0.92,
            ),
        ]
        return make_page(
            children=[
                make_region("block", children=[make_region("line", children=words)])
            ]
        )

    def test_predict_accepts_page_and_returns_page(self, tmp_path, monkeypatch):
        calls = {}

        def fake_predict_text_images(regions, batch_size=32):
            calls["count"] = len(regions)
            calls["batch_size"] = batch_size
            return [
                {"text": "word1", "confidence": 0.9},
                {"text": "word2", "confidence": 0.85},
            ]

        recognizer = self._create_recognizer(tmp_path)
        monkeypatch.setattr(
            recognizer, "_predict_text_images", fake_predict_text_images
        )
        page = self._create_page()
        image = np.zeros((64, 180, 3), dtype=np.uint8)
        result = recognizer.predict(page, image=image)
        assert isinstance(result, Page)
        assert result is not page
        assert result.children[0].children[0].children[0].text == "word1"
        assert result.children[0].children[0].children[1].text == "word2"


class TestTRBABatching:
    _create_page = staticmethod(TestTRBAPredictPageInterface._create_page)

    def _create_recognizer(self, tmp_path, **kwargs):
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model.json"
        charset_file = tmp_path / "model.txt"
        weights_file.write_text("mock_onnx")
        config_file.write_text(
            '{"recognition_levels": ["word"], "max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}'
        )
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
        return TRBA(
            weights=str(weights_file),
            config=str(config_file),
            charset=str(charset_file),
            device="cpu",
            **kwargs,
        )

    @pytest.mark.parametrize('runtime, device, input_batch, requested, count, expected_batches', [
        ('native', 'cpu', 'batch', 4, 6, [4, 2]),
        ('native', 'cpu', 'batch', 128, 1, [1]),
        ('native', 'cuda', 'batch', 128, 1, [128]),
        ('native', 'cuda', 'batch', 4, 6, [4, 4]),
        ('native', 'coreml', 'batch', 128, 1, [128]),
        ('web', 'gpu', 'batch', 128, 1, [1]),
        ('web', 'cpu', 'batch', 4, 6, [4, 2]),
        ('native', 'cpu', 4, 4, 6, [4, 4]),
        ('native', 'cpu', 4, 1, 6, [4, 4]),
        ('web', 'gpu', 4, 4, 6, [4, 4]),
    ])
    def test_run_inference_batches_padding_policy(
        self, tmp_path, monkeypatch, runtime, device, input_batch, requested, count,
        expected_batches,
    ):
        recognizer = self._create_recognizer(tmp_path, batch_size=requested)
        recognizer.runtime = runtime
        recognizer.device = device

        class FakeSession:

            def __init__(self):
                self.calls = []

            def get_inputs(self):
                return [
                    types.SimpleNamespace(name="input", shape=[input_batch, 3, 64, 256])
                ]

            def get_outputs(self):
                return [types.SimpleNamespace(name="output", shape=["batch", 5, 6])]

            def run(self, output_names, input_feed):
                batch = next(iter(input_feed.values()))
                self.calls.append(batch.shape[0])
                logits = np.zeros((batch.shape[0], 2, 6), dtype=np.float32)
                logits[:, 0, 3] = 1.0
                logits[:, 1, 2] = 1.0
                return [logits]

        recognizer.onnx_session = FakeSession()
        monkeypatch.setattr(recognizer, "_supports_multi_batch_inference", True)
        monkeypatch.setattr(
            recognizer,
            "_preprocess_image",
            lambda image: np.zeros((1, 3, 64, 256), dtype=np.float32),
        )
        images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(count)]
        predictions = recognizer._predict_word_images(images, batch_size=requested)
        assert len(predictions) == count
        assert recognizer.onnx_session.calls == expected_batches
        assert [prediction["text"] for prediction in predictions] == ["a"] * count

    def test_predict_uses_constructor_batch_size_by_default(
        self, tmp_path, monkeypatch
    ):
        calls = {}

        def fake_predict_text_images(regions, batch_size=32):
            calls["count"] = len(regions)
            calls["batch_size"] = batch_size
            return [
                {"text": "word1", "confidence": 0.9},
                {"text": "word2", "confidence": 0.85},
            ]

        recognizer = self._create_recognizer(tmp_path, batch_size=64)
        monkeypatch.setattr(
            recognizer, "_predict_text_images", fake_predict_text_images
        )
        page = self._create_page()
        image = np.zeros((64, 180, 3), dtype=np.uint8)
        recognizer.predict(page, image=image)
        assert calls == {"count": 2, "batch_size": 64}

    def test_predict_batch_size_argument_overrides_constructor_value(
        self, tmp_path, monkeypatch
    ):
        calls = {}

        def fake_predict_text_images(regions, batch_size=32):
            calls["count"] = len(regions)
            calls["batch_size"] = batch_size
            return [
                {"text": "word1", "confidence": 0.9},
                {"text": "word2", "confidence": 0.85},
            ]

        recognizer = self._create_recognizer(tmp_path, batch_size=64)
        monkeypatch.setattr(
            recognizer, "_predict_text_images", fake_predict_text_images
        )
        page = self._create_page()
        image = np.zeros((64, 180, 3), dtype=np.uint8)
        recognizer.predict(page, image=image, batch_size=8)
        assert calls == {"count": 2, "batch_size": 8}

    def test_predict_without_image_returns_copy(self, tmp_path):
        recognizer = self._create_recognizer(tmp_path)
        page = self._create_page()
        result = recognizer.predict(page, image=None)
        assert isinstance(result, Page)
        assert result is not page
        assert result.children[0].children[0].children[0].text is None
        assert result.children[0].children[0].children[1].text is None

    def test_predict_respects_min_text_size(self, tmp_path, monkeypatch):
        calls = {}

        def fake_predict_text_images(regions, batch_size=32):
            calls["count"] = len(regions)
            return [{"text": "big", "confidence": 0.88}]

        recognizer = self._create_recognizer(tmp_path, min_text_size=5)
        monkeypatch.setattr(
            recognizer, "_predict_text_images", fake_predict_text_images
        )
        page = make_page(
            children=[
                make_region(
                    "block",
                    children=[
                        make_region(
                            "line",
                            children=[
                                make_region(
                                    "word",
                                    polygon=[
                                        (10.0, 10.0),
                                        (12.0, 10.0),
                                        (12.0, 12.0),
                                        (10.0, 12.0),
                                    ],
                                    detection_confidence=0.9,
                                ),
                                make_region(
                                    "word",
                                    polygon=[
                                        (20.0, 10.0),
                                        (100.0, 10.0),
                                        (100.0, 40.0),
                                        (20.0, 40.0),
                                    ],
                                    detection_confidence=0.9,
                                ),
                            ],
                        )
                    ],
                )
            ]
        )
        image = np.zeros((64, 180, 3), dtype=np.uint8)
        result = recognizer.predict(page, image=image)
        assert result.children[0].children[0].children[0].text is None
        assert result.children[0].children[0].children[1].text == "big"
        assert calls == {"count": 1}

    def test_default_bbox_preparer_rotates_before_prediction(
        self, tmp_path, monkeypatch
    ):
        captured = {}

        def fake_predict_text_images(regions, batch_size=32):
            captured["shape"] = regions[0].image.shape
            captured["pixel"] = regions[0].image[0, 99].tolist()
            return [{"text": "rot", "confidence": 0.91}]

        recognizer = self._create_recognizer(tmp_path, rotate_threshold=1.5)
        monkeypatch.setattr(
            recognizer, "_predict_text_images", fake_predict_text_images
        )
        page = make_page(
            children=[
                make_region(
                    "block",
                    children=[
                        make_region(
                            "line",
                            children=[
                                make_region(
                                    "word",
                                    polygon=[
                                        (10.0, 10.0),
                                        (60.0, 10.0),
                                        (60.0, 110.0),
                                        (10.0, 110.0),
                                    ],
                                    detection_confidence=0.95,
                                )
                            ],
                        )
                    ],
                )
            ]
        )
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        image[10, 10] = [255, 0, 0]
        result = recognizer.predict(page, image=image)
        assert captured["shape"] == (50, 100, 3)
        assert captured["pixel"] == [255, 0, 0]
        assert result.children[0].children[0].children[0].text == "rot"

    def test_polygon_mask_preset_masks_background(self, tmp_path, monkeypatch):
        captured = {}

        def fake_predict_text_images(regions, batch_size=32):
            captured["image"] = regions[0].image.copy()
            return [{"text": "mask", "confidence": 0.9}]

        recognizer = self._create_recognizer(tmp_path, region_preparer="polygon_mask")
        monkeypatch.setattr(
            recognizer, "_predict_text_images", fake_predict_text_images
        )
        page = make_page(
            children=[
                make_region(
                    "block",
                    children=[
                        make_region(
                            "line",
                            children=[
                                make_region(
                                    "word",
                                    polygon=[
                                        (20.0, 10.0),
                                        (50.0, 10.0),
                                        (40.0, 40.0),
                                        (10.0, 40.0),
                                    ],
                                    detection_confidence=0.95,
                                )
                            ],
                        )
                    ],
                )
            ]
        )
        image = np.full((64, 64, 3), [10, 20, 30], dtype=np.uint8)
        recognizer.predict(page, image=image)
        assert captured["image"][0, 0].tolist() == [255, 255, 255]
        assert captured["image"][15, 15].tolist() == [10, 20, 30]

    def test_quad_warp_preset_uses_natural_rectified_size(self, tmp_path, monkeypatch):
        captured = {}

        def fake_predict_text_images(regions, batch_size=32):
            captured["shape"] = regions[0].image.shape
            return [{"text": "warp", "confidence": 0.9}]

        recognizer = self._create_recognizer(tmp_path, region_preparer="quad_warp")
        monkeypatch.setattr(
            recognizer, "_predict_text_images", fake_predict_text_images
        )
        polygon = [(10.0, 10.0), (70.0, 20.0), (60.0, 50.0), (0.0, 40.0)]
        page = make_page(
            children=[
                make_region(
                    "block",
                    children=[
                        make_region(
                            "line",
                            children=[
                                make_region(
                                    "word", polygon=polygon, detection_confidence=0.9
                                )
                            ],
                        )
                    ],
                )
            ]
        )
        image = np.full((64, 80, 3), 200, dtype=np.uint8)
        recognizer.predict(page, image=image)
        assert captured["shape"][:2] == (32, 61)

    def test_quad_warp_preset_falls_back_to_bbox_for_non_quad_polygon(
        self, tmp_path, monkeypatch
    ):
        captured = {}

        def fake_predict_text_images(regions, batch_size=32):
            captured["shape"] = regions[0].image.shape
            captured["meta"] = dict(regions[0].meta)
            return [{"text": "fallback", "confidence": 0.9}]

        recognizer = self._create_recognizer(tmp_path, region_preparer="quad_warp")
        monkeypatch.setattr(
            recognizer, "_predict_text_images", fake_predict_text_images
        )
        polygon = [(10.0, 10.0), (40.0, 10.0), (70.0, 20.0), (60.0, 50.0), (20.0, 55.0)]
        page = make_page(
            children=[
                make_region(
                    "block",
                    children=[
                        make_region(
                            "line",
                            children=[
                                make_region(
                                    "word", polygon=polygon, detection_confidence=0.9
                                )
                            ],
                        )
                    ],
                )
            ]
        )
        image = np.full((80, 96, 3), 200, dtype=np.uint8)
        result = recognizer.predict(page, image=image)
        assert captured["shape"][:2] == (45, 60)
        assert captured["meta"]["region_preparer"] == "quad_warp"
        assert captured["meta"]["fallback_to_bbox"] is True
        assert result.children[0].children[0].children[0].text == "fallback"

    def test_custom_region_preparer_hook(self, tmp_path, monkeypatch):
        captured = {}

        def region_preparer(page, image, recognizer=None, options=None):
            words = page.children[0].children[0].children
            return [
                {
                    "text_span": words[0],
                    "image": image[10:40, 10:80],
                    "polygon": np.array(words[0].polygon, dtype=np.float32),
                    "meta": {"source": "custom"},
                },
                {
                    "text_span": words[1],
                    "image": image[10:40, 90:160],
                    "polygon": np.array(words[1].polygon, dtype=np.float32),
                },
            ]

        def fake_predict_text_images(regions, batch_size=32):
            captured["metas"] = [region.meta for region in regions]
            captured["batch_size"] = batch_size
            return [
                {"text": "custom1", "confidence": 0.8},
                {"text": "custom2", "confidence": 0.7},
            ]

        recognizer = self._create_recognizer(tmp_path, region_preparer=region_preparer)
        monkeypatch.setattr(
            recognizer, "_predict_text_images", fake_predict_text_images
        )
        page = self._create_page()
        image = np.zeros((64, 180, 3), dtype=np.uint8)
        result = recognizer.predict(page, image=image)
        assert captured["batch_size"] == 128
        assert captured["metas"][0]["source"] == "custom"
        assert result.children[0].children[0].children[0].text == "custom1"
        assert result.children[0].children[0].children[1].text == "custom2"

    def test_predict_can_save_debug_regions(self, tmp_path, monkeypatch):

        def fake_predict_text_images(regions, batch_size=32):
            return [
                {"text": "word1", "confidence": 0.9},
                {"text": "word2", "confidence": 0.85},
            ]

        recognizer = self._create_recognizer(tmp_path)
        monkeypatch.setattr(
            recognizer, "_predict_text_images", fake_predict_text_images
        )
        page = self._create_page()
        image = np.zeros((64, 180, 3), dtype=np.uint8)
        image[10:40, 10:80] = [10, 20, 30]
        image[10:40, 90:160] = [40, 50, 60]
        debug_dir = tmp_path / "recognizer_crops"
        recognizer.predict(page, image=image, debug_save_dir=debug_dir)
        crop_files = sorted(debug_dir.glob("*.png"))
        assert [path.name for path in crop_files] == ["0000.png", "0001.png"]
        assert (debug_dir / "index.json").exists()
        index_data = json.loads((debug_dir / "index.json").read_text(encoding="utf-8"))
        assert len(index_data) == 2
        assert index_data[0]["region_preparer"] == "bbox"
        assert index_data[0]["prediction"]["text"] == "word1"
        assert index_data[1]["prediction"]["confidence"] == 0.85

    def test_predict_uses_default_debug_dir_from_constructor(
        self, tmp_path, monkeypatch
    ):

        def fake_predict_text_images(regions, batch_size=32):
            return [
                {"text": "word1", "confidence": 0.9},
                {"text": "word2", "confidence": 0.85},
            ]

        debug_dir = tmp_path / "recognizer_crops"
        recognizer = self._create_recognizer(tmp_path, debug_save_dir=debug_dir)
        monkeypatch.setattr(
            recognizer, "_predict_text_images", fake_predict_text_images
        )
        page = self._create_page()
        image = np.zeros((64, 180, 3), dtype=np.uint8)
        recognizer.predict(page, image=image)
        assert (debug_dir / "0000.png").exists()
        assert (debug_dir / "0001.png").exists()
        assert (debug_dir / "index.json").exists()

    def test_removed_recognizer_debug_dir_argument_raises(self, tmp_path):
        with pytest.raises(TypeError, match="recognizer_debug_dir"):
            self._create_recognizer(
                tmp_path, recognizer_debug_dir=tmp_path / "recognizer_crops"
            )

    def test_invalid_region_preparer_preset_raises(self, tmp_path):
        with patch("manuscript.api.base.BaseArtifactModel._download_http"):
            weights_file = tmp_path / "model.onnx"
            config_file = tmp_path / "model.json"
            charset_file = tmp_path / "model.txt"
            weights_file.write_text("mock_onnx")
            config_file.write_text(
                '{"recognition_levels": ["word"], "max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}'
            )
            charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
            try:
                TRBA(
                    weights=str(weights_file),
                    config=str(config_file),
                    charset=str(charset_file),
                    device="cpu",
                    region_preparer="unknown",
                )
                assert False, "Expected ValueError for invalid region_preparer"
            except ValueError as exc:
                assert "region_preparer" in str(exc)

    def test_removed_region_predictor_argument_raises(self, tmp_path):
        with patch("manuscript.api.base.BaseArtifactModel._download_http"):
            weights_file = tmp_path / "model.onnx"
            config_file = tmp_path / "model.json"
            charset_file = tmp_path / "model.txt"
            weights_file.write_text("mock_onnx")
            config_file.write_text(
                '{"recognition_levels": ["word"], "max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}'
            )
            charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
            try:
                TRBA(
                    weights=str(weights_file),
                    config=str(config_file),
                    charset=str(charset_file),
                    device="cpu",
                    region_predictor=lambda *_args, **_kwargs: [],
                )
                assert False, "Expected TypeError for removed region_predictor"
            except TypeError as exc:
                assert "region_predictor has been removed" in str(exc)


def test_local_recognizer_does_not_discover_adjacent_artifacts(tmp_path):
    model = object.__new__(TRBA)
    model.weights = str(tmp_path / "model.onnx")
    (tmp_path / "model.json").write_text('{}')
    (tmp_path / "inference.yml").write_text('unused: true')
    (tmp_path / "model.txt").write_text('a')
    with pytest.raises(FileNotFoundError, match="explicit config"):
        model._resolve_config(None)
    with pytest.raises(FileNotFoundError, match="explicit charset"):
        model._resolve_charset(None)



def test_local_trba_requires_explicit_config_even_with_charset(tmp_path):
    weights = tmp_path / 'model.onnx'
    weights.write_bytes(b'test')
    charset = tmp_path / 'charset.txt'
    charset.write_text('<PAD>\n<SOS>\n<EOS>\na\n')
    with pytest.raises(FileNotFoundError, match='config artifact or explicit config'):
        TRBA(weights=weights, charset=charset, device='cpu')
