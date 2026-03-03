from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image

from manuscript.recognizers import TRBA


class TestTRBAInitialization:
    """Tests for TRBA initialization"""

    def test_trba_import(self):
        """Test TRBA import"""
        assert TRBA is not None
        assert hasattr(TRBA, 'predict')
        assert hasattr(TRBA, 'train')
        assert hasattr(TRBA, 'export')

    def test_trba_has_basemodel_attributes(self):
        """Test that TRBA inherits from BaseModel"""
        assert hasattr(TRBA, 'default_weights_name')
        assert hasattr(TRBA, 'pretrained_registry')
        assert hasattr(TRBA, 'config_registry')
        assert hasattr(TRBA, 'charset_registry')

    def test_trba_default_weights(self):
        """Test default preset"""
        assert TRBA.default_weights_name == "trba_lite_g1"

    @patch('manuscript.api.base.BaseModel._download_http')
    def test_trba_initialization_with_local_file(self, mock_download, tmp_path):
        """Test initialization with local files"""
        # Create mock files
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model.json"
        charset_file = tmp_path / "model.txt"
        
        weights_file.write_text("mock_onnx")
        config_file.write_text('{"max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}')
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
        
        # Initialization should proceed without downloading
        recognizer = TRBA(
            weights=str(weights_file),
            config=str(config_file),
            charset=str(charset_file),
            device="cpu"
        )
        
        assert recognizer.weights == str(weights_file.absolute())
        assert recognizer.device == "cpu"
        assert recognizer.config_path == str(config_file.absolute())
        assert recognizer.charset_path == str(charset_file.absolute())
        
        # Download should not be called for local files
        mock_download.assert_not_called()

    def test_trba_device_auto_selection(self, tmp_path):
        """Test automatic device selection"""
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model.json"
        charset_file = tmp_path / "model.txt"
        
        weights_file.write_text("mock_onnx")
        config_file.write_text('{"max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}')
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
        
        recognizer = TRBA(
            weights=str(weights_file),
            config=str(config_file),
            charset=str(charset_file),
            device=None  # Auto-select
        )
        
        # Should select cpu or cuda depending on availability
        assert recognizer.device in ["cpu", "cuda"]

    def test_trba_explicit_cpu_device(self, tmp_path):
        """Test explicit CPU selection"""
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model.json"
        charset_file = tmp_path / "model.txt"
        
        weights_file.write_text("mock_onnx")
        config_file.write_text('{"max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}')
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
        
        recognizer = TRBA(
            weights=str(weights_file),
            config=str(config_file),
            charset=str(charset_file),
            device="cpu"
        )
        
        assert recognizer.device == "cpu"


class TestTRBAConfigResolution:
    """Tests for config and charset file resolution"""

    def test_config_inferred_from_weights_name(self, tmp_path):
        """Test automatic inference of config from weights filename"""
        weights_file = tmp_path / "my_model.onnx"
        config_file = tmp_path / "my_model.json"
        charset_file = tmp_path / "my_model.txt"
        
        weights_file.write_text("mock_onnx")
        config_file.write_text('{"max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}')
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
        
        # Do not specify config and charset - they should be inferred automatically
        recognizer = TRBA(weights=str(weights_file), device="cpu")
        
        assert recognizer.config_path == str(config_file.absolute())
        assert recognizer.charset_path == str(charset_file.absolute())

    def test_config_fallback_to_default_preset(self, tmp_path):
        """Test fallback to default preset config when not found next to weights"""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock_onnx")
        # Do not create config file - should use default preset
        
        # This will download the default preset config (trba_lite_g1)
        recognizer = TRBA(weights=str(weights_file), device="cpu")
        
        # Should fallback to default preset config (trba_lite_g1)
        assert recognizer.config_path is not None
        assert Path(recognizer.config_path).exists()
        # Should be the default preset config
        assert "trba_lite_g1" in recognizer.config_path

    def test_charset_fallback_to_default_preset(self, tmp_path):
        """Test fallback to default preset charset when not found next to weights"""
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model.json"
        
        weights_file.write_text("mock_onnx")
        config_file.write_text('{"max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}')
        # Do not create charset file - should use default preset
        
        # This will download the default preset charset (trba_lite_g1)
        recognizer = TRBA(weights=str(weights_file), config=str(config_file), device="cpu")
        
        # Should fallback to default preset charset (trba_lite_g1)
        assert recognizer.charset_path is not None
        assert Path(recognizer.charset_path).exists()
        # Should be the default preset charset
        assert "trba_lite_g1" in recognizer.charset_path

    def test_explicit_charset_parameter(self, tmp_path):
        """Test that explicit charset parameter is used when provided"""
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model.json"
        charset_file = tmp_path / "custom_charset.txt"
        
        weights_file.write_text("mock_onnx")
        config_file.write_text('{"max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}')
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
        
        # Explicit charset should be used
        recognizer = TRBA(
            weights=str(weights_file),
            config=str(config_file),
            charset=str(charset_file),
            device="cpu"
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
        config_file.write_text('{"max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}')
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
        
        recognizer = TRBA(
            weights=str(weights_file),
            config=str(config_file),
            charset=str(charset_file),
            device="cpu"
        )
        
        # Create a test image
        img = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        # Preprocessing
        preprocessed = recognizer._preprocess_image(img)
        
        # Check format
        assert preprocessed.shape == (1, 3, 64, 256)  # [batch, channels, height, width]
        assert preprocessed.dtype == np.float32
        assert preprocessed.min() >= -1.5  # After normalization
        assert preprocessed.max() <= 1.5

    def test_preprocess_image_from_pil(self, tmp_path):
        """Test preprocessing from PIL Image"""
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model.json"
        charset_file = tmp_path / "model.txt"
        
        weights_file.write_text("mock_onnx")
        config_file.write_text('{"max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}')
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
        
        recognizer = TRBA(
            weights=str(weights_file),
            config=str(config_file),
            charset=str(charset_file),
            device="cpu"
        )
        
        # Create a PIL image
        pil_img = Image.new('RGB', (200, 100), color=(255, 0, 0))
        
        # Preprocessing 
        preprocessed = recognizer._preprocess_image(pil_img)
        
        # Check format
        assert preprocessed.shape == (1, 3, 64, 256)
        assert preprocessed.dtype == np.float32

    def test_prepare_crop_rotates_vertical_image(self, tmp_path):
        """Test that TRBA rotates vertical crops before recognition."""
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model.json"
        charset_file = tmp_path / "model.txt"

        weights_file.write_text("mock_onnx")
        config_file.write_text('{"max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}')
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
        config_file.write_text('{"max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}')
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

    def test_preprocess_uses_prepare_crop(self, tmp_path):
        """Test that preprocessing calls crop-orientation logic."""
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model.json"
        charset_file = tmp_path / "model.txt"

        weights_file.write_text("mock_onnx")
        config_file.write_text('{"max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}')
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")

        recognizer = TRBA(
            weights=str(weights_file),
            config=str(config_file),
            charset=str(charset_file),
            device="cpu",
        )

        vertical_crop = np.zeros((100, 50, 3), dtype=np.uint8)
        with patch.object(recognizer, "_prepare_crop", wraps=recognizer._prepare_crop) as mocked:
            recognizer._preprocess_image(vertical_crop)
            mocked.assert_called_once()


class TestTRBAAPI:
    """Tests for the public API"""

    def test_trba_callable(self, tmp_path):
        """Test that TRBA can be called as a function (via __call__)"""
        weights_file = tmp_path / "model.onnx"
        config_file = tmp_path / "model.json"
        charset_file = tmp_path / "model.txt"
        
        weights_file.write_text("mock_onnx")
        config_file.write_text('{"max_len": 25, "hidden_size": 256, "img_h": 64, "img_w": 256}')
        charset_file.write_text("<PAD>\n<SOS>\n<EOS>\na\nb\nc")
        
        recognizer = TRBA(
            weights=str(weights_file),
            config=str(config_file),
            charset=str(charset_file),
            device="cpu"
        )
        
        # Should be callable via BaseModel.__call__
        assert callable(recognizer)

    def test_static_methods_exist(self):
        """Test that static methods are accessible"""
        assert hasattr(TRBA, 'train')
        assert callable(TRBA.train)
        assert hasattr(TRBA, 'export')
        assert callable(TRBA.export)
