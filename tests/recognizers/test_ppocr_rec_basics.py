import json
import numpy as np

from manuscript.api.recognizer import BaseRecognizer
from manuscript.recognizers import PPOCRRec


class TestPPOCRRecInitialization:
    def test_import_and_base_contract(self):
        assert PPOCRRec is not None
        assert issubclass(PPOCRRec, BaseRecognizer)
        assert hasattr(PPOCRRec, "predict")

    def test_initialization_with_local_files(self, tmp_path):
        weights_file = tmp_path / "model.onnx"
        charset_file = tmp_path / "custom_dict.txt"

        weights_file.write_text("mock_onnx")
        weights_file.with_suffix(".json").write_text(json.dumps({
            "rec_image_shape": [3, 48, 320], "PreProcess": {},
        }))
        charset_file.write_text("a\nb\nc\n")

        recognizer = PPOCRRec(
            weights=str(weights_file),
            charset=str(charset_file),
            device="cpu",
        )

        assert recognizer.weights == str(weights_file.absolute())
        assert recognizer.charset_path == str(charset_file.absolute())
        assert recognizer.rec_image_shape == [3, 48, 320]
        assert recognizer.characters == ["blank", "a", "b", "c"]

    def test_initialization_from_inference_yaml(self, tmp_path):
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "inference.yml"

        weights_file.write_text("mock_onnx")
        weights_file.with_suffix(".json").write_text(json.dumps({
            "rec_image_shape": [3, 48, 320], "PreProcess": {},
        }))
        config_file.write_text(
            "\n".join(
                [
                    "Global:",
                    "  use_space_char: true",
                    "PreProcess:",
                    "  transform_ops:",
                    "  - DecodeImage:",
                    "      channel_first: false",
                    "      img_mode: BGR",
                    "  - MultiLabelEncode:",
                    "      gtc_encode: NRTRLabelEncode",
                    "  - RecResizeImg:",
                    "      image_shape: [3, 48, 320]",
                    "PostProcess:",
                    "  name: CTCLabelDecode",
                    "  character_dict: ['a', 'b']",
                ]
            ),
            encoding="utf-8",
        )

        recognizer = PPOCRRec(
            weights=str(weights_file),
            config=str(config_file),
            device="cpu",
        )

        assert recognizer.config_path == str(config_file.absolute())
        assert recognizer.use_space_char is True
        assert recognizer.characters == ["blank", "a", "b", " "]


class TestPPOCRRecPreprocessing:
    def test_preprocess_image(self, tmp_path):
        weights_file = tmp_path / "model.onnx"
        charset_file = tmp_path / "custom_dict.txt"

        weights_file.write_text("mock_onnx")
        weights_file.with_suffix(".json").write_text(json.dumps({
            "rec_image_shape": [3, 48, 320], "PreProcess": {},
        }))
        charset_file.write_text("a\nb\nc\n")

        recognizer = PPOCRRec(
            weights=str(weights_file),
            charset=str(charset_file),
            device="cpu",
        )

        img = np.random.randint(0, 255, (64, 128, 3), dtype=np.uint8)
        preprocessed = recognizer._preprocess_image(img)

        assert preprocessed.shape == (1, 3, 48, 320)
        assert preprocessed.dtype == np.float32
        assert preprocessed.min() >= -1.0
        assert preprocessed.max() <= 1.0


class TestPPOCRRecDecode:
    def test_ctc_decode(self, tmp_path):
        weights_file = tmp_path / "model.onnx"
        charset_file = tmp_path / "custom_dict.txt"

        weights_file.write_text("mock_onnx")
        weights_file.with_suffix(".json").write_text(json.dumps({
            "rec_image_shape": [3, 48, 320], "PreProcess": {},
        }))
        charset_file.write_text("a\nb\n")

        recognizer = PPOCRRec(
            weights=str(weights_file),
            charset=str(charset_file),
            device="cpu",
        )

        logits = np.array(
            [
                [
                    [0.9, 0.1, 0.0],  # blank
                    [0.1, 0.9, 0.0],  # a
                    [0.1, 0.8, 0.1],  # a repeated
                    [0.1, 0.1, 0.8],  # b
                ]
            ],
            dtype=np.float32,
        )

        predictions = recognizer._decode_recognition_logits(logits)

        assert len(predictions) == 1
        assert predictions[0].text == "ab"
        assert 0.0 <= predictions[0].confidence <= 1.0
