"""Tests for BaseArtifactModel class."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import onnxruntime as ort

import manuscript.api.base as base_module
from manuscript.api.base import BaseArtifactModel


# ============================================================================
# Mock implementations for testing abstract class
# ============================================================================
class ConcreteModel(BaseArtifactModel):
    """Concrete implementation of BaseArtifactModel for testing."""
    
    default_weights_name = "default_model.onnx"
    pretrained_registry = {
        "preset1": "https://example.com/model1.onnx",
        "preset2": "github://owner/repo/v1.0/model2.onnx",
        "nested_preset": "preset1",  # Redirects to preset1
    }
    
    def _initialize_session(self):
        """Mock session initialization."""
        self.session = Mock()
        self.session.initialized = True
    
    def predict(self, x):
        """Mock prediction."""
        if self.session is None:
            self._initialize_session()
        return {"result": x * 2}


class MinimalModel(BaseArtifactModel):
    """Minimal model without default weights or registry."""
    
    def _initialize_session(self):
        self.session = Mock()
    
    def predict(self, x):
        return x


# ============================================================================
# Tests for device resolution
# ============================================================================
class TestDeviceResolution:
    """Tests for device selection logic."""
    
    def test_explicit_cpu_device(self, tmp_path):
        """Test explicit CPU device selection."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        model = ConcreteModel(weights=str(weights_file), device="cpu")
        assert model.device == "cpu"
    
    def test_explicit_cuda_device(self, tmp_path):
        """Test explicit CUDA device selection."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        model = ConcreteModel(weights=str(weights_file), device="cuda")
        assert model.device == "cuda"
    
    @patch("onnxruntime.get_device")
    def test_auto_device_gpu_available(self, mock_get_device, tmp_path):
        """Test automatic GPU selection when available."""
        mock_get_device.return_value = "GPU"
        
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        model = ConcreteModel(weights=str(weights_file), device=None)
        assert model.device == "cuda"
    
    @patch("onnxruntime.get_device")
    def test_auto_device_cpu_fallback(self, mock_get_device, tmp_path):
        """Test automatic CPU fallback when GPU unavailable."""
        mock_get_device.side_effect = Exception("No GPU")
        
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        model = ConcreteModel(weights=str(weights_file), device=None)
        assert model.device == "cpu"
    
    @patch("onnxruntime.get_device")
    def test_auto_device_cpu_when_cpu_reported(self, mock_get_device, tmp_path):
        """Test CPU selection when onnxruntime reports CPU."""
        mock_get_device.return_value = "CPU"
        
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        model = ConcreteModel(weights=str(weights_file), device=None)
        assert model.device == "cpu"
    
    def test_runtime_providers_cpu(self, tmp_path):
        """Test runtime providers for CPU device."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        model = ConcreteModel(weights=str(weights_file), device="cpu")
        providers = model.runtime_providers()
        
        assert providers == ["CPUExecutionProvider"]
    
    def test_runtime_providers_cuda(self, tmp_path):
        """Test runtime providers for CUDA device."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        model = ConcreteModel(weights=str(weights_file), device="cuda")
        providers = model.runtime_providers()
        
        assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]


# ============================================================================
# Tests for weight resolution
# ============================================================================
class TestWeightResolution:
    """Tests for weight file resolution logic."""
    
    def test_local_file_absolute_path(self, tmp_path):
        """Test loading from absolute local path."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock weights")
        
        model = ConcreteModel(weights=str(weights_file))
        assert Path(model.weights).exists()
        assert Path(model.weights).is_absolute()
    
    def test_local_file_relative_path(self, tmp_path, monkeypatch):
        """Test loading from relative local path."""
        # Create file in temp dir and cd there
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock weights")
        
        monkeypatch.chdir(tmp_path)
        
        model = ConcreteModel(weights="model.onnx")
        assert Path(model.weights).exists()
    
    def test_local_file_with_tilde(self, tmp_path):
        """Test path expansion with ~ (home directory)."""
        # Create file in user's home
        home_dir = Path.home()
        test_file = home_dir / ".test_manuscript_model.onnx"
        
        try:
            test_file.write_text("mock")
            
            model = ConcreteModel(weights="~/.test_manuscript_model.onnx")
            assert Path(model.weights).exists()
        finally:
            if test_file.exists():
                test_file.unlink()
    
    def test_default_weights_when_none(self, tmp_path):
        """Test using default weights when None specified."""
        # Create a mock weights file that will be found
        weights_file = tmp_path / "temp_model.onnx"
        weights_file.write_text("mock weights")
        
        # Mock _resolve_weights to use our test file for default
        with patch.object(ConcreteModel, 'default_weights_name', str(weights_file)):
            model = ConcreteModel(weights=None)
            assert Path(model.weights).exists()
    
    def test_no_default_weights_raises_error(self):
        """Test error when no default weights defined."""
        with pytest.raises(ValueError, match="must define default_weights_name"):
            MinimalModel(weights=None)
    
    def test_preset_from_registry(self, tmp_path):
        """Test loading preset from registry."""
        with patch.object(ConcreteModel, '_download_http') as mock_download:
            mock_file = tmp_path / "downloaded.onnx"
            mock_file.write_text("mock")
            mock_download.return_value = str(mock_file)
            
            model = ConcreteModel(weights="preset1")
            
            # Should call download with URL from registry
            mock_download.assert_called_once_with("https://example.com/model1.onnx")
            assert Path(model.weights).exists()
    
    def test_nested_preset_resolution(self, tmp_path):
        """Test preset that redirects to another preset."""
        with patch.object(ConcreteModel, '_download_http') as mock_download:
            mock_file = tmp_path / "downloaded.onnx"
            mock_file.write_text("mock")
            mock_download.return_value = str(mock_file)
            
            model = ConcreteModel(weights="nested_preset")
            
            # Should resolve nested_preset → preset1 → URL
            mock_download.assert_called_once_with("https://example.com/model1.onnx")
    
    def test_unknown_weights_raises_error(self):
        """Test error for unknown weight specification."""
        with pytest.raises(ValueError, match="Unknown weights"):
            ConcreteModel(weights="nonexistent_file_or_preset.onnx")
    
    @patch("manuscript.api.base.urllib.request.urlretrieve")
    def test_http_url_download(self, mock_urlretrieve, tmp_path):
        """Test downloading from HTTP URL."""
        url = "https://example.com/model.onnx"
        
        # Ensure cache file doesn't exist
        cache_dir = Path.home() / ".manuscript" / "weights"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_file = cache_dir / "model.onnx"
        
        # Remove if exists from previous tests
        if cached_file.exists():
            cached_file.unlink()
        
        try:
            def fake_download(url, tmp_path):
                # Write to the temp path that will be moved to cache
                Path(tmp_path).write_text("downloaded content")
            
            mock_urlretrieve.side_effect = fake_download
            
            model = ConcreteModel(weights=url)
            
            # Should have downloaded to cache
            assert "model.onnx" in model.weights
            mock_urlretrieve.assert_called_once()
        finally:
            if cached_file.exists():
                cached_file.unlink()
    
    @patch("manuscript.api.base.urllib.request.urlretrieve")
    def test_https_url_download(self, mock_urlretrieve, tmp_path):
        """Test downloading from HTTPS URL."""
        url = "https://secure.example.com/model.onnx"
        
        cache_dir = Path.home() / ".manuscript" / "weights"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_file = cache_dir / "model.onnx"
        
        # Remove if exists from previous tests
        if cached_file.exists():
            cached_file.unlink()
        
        try:
            def fake_download(url, tmp_path):
                Path(tmp_path).write_text("downloaded content")
        
            mock_urlretrieve.side_effect = fake_download
            
            model = ConcreteModel(weights=url)
            assert "model.onnx" in model.weights
        finally:
            if cached_file.exists():
                cached_file.unlink()
    
    @patch.object(ConcreteModel, '_download_http')
    def test_github_url_resolution(self, mock_download_http, tmp_path):
        """Test GitHub URL resolution to proper download URL."""
        github_spec = "github://owner/repo/v1.0/model.onnx"
        
        mock_file = tmp_path / "model.onnx"
        mock_file.write_text("mock")
        mock_download_http.return_value = str(mock_file)
        
        model = ConcreteModel(weights=github_spec)
        
        # Should convert to GitHub releases URL
        expected_url = "https://github.com/owner/repo/releases/download/v1.0/model.onnx"
        mock_download_http.assert_called_once_with(expected_url)
    
    @patch.object(ConcreteModel, '_download_http')
    def test_github_url_with_nested_path(self, mock_download_http, tmp_path):
        """Test GitHub URL with nested path."""
        github_spec = "github://owner/repo/v2.0/models/subdir/model.onnx"
        
        mock_file = tmp_path / "model.onnx"
        mock_file.write_text("mock")
        mock_download_http.return_value = str(mock_file)
        
        model = ConcreteModel(weights=github_spec)
        
        expected_url = "https://github.com/owner/repo/releases/download/v2.0/models/subdir/model.onnx"
        mock_download_http.assert_called_once_with(expected_url)
    
    def test_gdrive_url_resolution(self, tmp_path):
        """Test Google Drive URL resolution with gdown."""
        gdrive_spec = "gdrive:1234567890abcdef"

        mock_file = tmp_path / "model.onnx"
        mock_file.write_text("mock")
        cache_root = tmp_path / "cache_home"
        cached_file = cache_root / ".manuscript" / "weights" / "1234567890abcdef.bin"

        with patch.object(base_module.Path, "home", return_value=cache_root):
            with patch.dict("sys.modules", {"gdown": MagicMock()}):
                import gdown

                gdown.download.return_value = str(mock_file)
                model = ConcreteModel(weights=gdrive_spec)

        # Verify gdown.download was called with correct file_id
        gdown.download.assert_called_once()
        call_kwargs = gdown.download.call_args.kwargs
        assert call_kwargs['id'] == '1234567890abcdef'
        assert call_kwargs['quiet'] == False
        assert model.weights == str(cached_file)
        assert cached_file.read_text(encoding="utf-8") == "mock"


# ============================================================================
# Tests for extra artifact resolution
# ============================================================================
class TestExtraArtifactResolution:
    """Tests for _resolve_extra_artifact method."""
    
    def test_resolve_local_artifact(self, tmp_path):
        """Test resolving local artifact file."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        artifact = tmp_path / "config.json"
        artifact.write_text('{"key": "value"}')
        
        model = ConcreteModel(weights=str(weights_file))
        
        result = model._resolve_extra_artifact(
            str(artifact),
            default_name=None,
            registry={},
            description="config"
        )
        
        assert Path(result).exists()
        assert "config.json" in result
    
    def test_resolve_default_artifact(self, tmp_path):
        """Test using default artifact when None specified."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        default_artifact = tmp_path / "default.json"
        default_artifact.write_text('{"default": true}')
        
        model = ConcreteModel(weights=str(weights_file))
        
        result = model._resolve_extra_artifact(
            None,
            default_name=str(default_artifact),
            registry={},
            description="config"
        )
        
        assert Path(result).exists()
    
    def test_resolve_artifact_from_registry(self, tmp_path):
        """Test resolving artifact from preset registry."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        artifact_file = tmp_path / "artifact.txt"
        artifact_file.write_text("content")
        
        model = ConcreteModel(weights=str(weights_file))
        
        registry = {
            "preset_artifact": str(artifact_file)
        }
        
        result = model._resolve_extra_artifact(
            "preset_artifact",
            default_name=None,
            registry=registry,
            description="artifact"
        )
        
        assert Path(result).exists()
    
    @patch.object(ConcreteModel, '_download_http')
    def test_resolve_artifact_from_url(self, mock_download, tmp_path):
        """Test downloading artifact from URL."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        mock_file = tmp_path / "artifact.txt"
        mock_file.write_text("downloaded")
        mock_download.return_value = str(mock_file)
        
        model = ConcreteModel(weights=str(weights_file))
        
        result = model._resolve_extra_artifact(
            "https://example.com/artifact.txt",
            default_name=None,
            registry={},
            description="artifact"
        )
        
        mock_download.assert_called_once_with("https://example.com/artifact.txt")
    
    @patch.object(ConcreteModel, '_download_http')
    def test_resolve_artifact_from_github(self, mock_download, tmp_path):
        """Test downloading artifact from GitHub."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        mock_file = tmp_path / "artifact.txt"
        mock_file.write_text("downloaded")
        mock_download.return_value = str(mock_file)
        
        model = ConcreteModel(weights=str(weights_file))
        
        result = model._resolve_extra_artifact(
            "github://user/repo/v1.0/artifact.txt",
            default_name=None,
            registry={},
            description="artifact"
        )
        
        expected_url = "https://github.com/user/repo/releases/download/v1.0/artifact.txt"
        mock_download.assert_called_once_with(expected_url)
    
    def test_resolve_artifact_from_gdrive(self, tmp_path):
        """Test downloading artifact from Google Drive with gdown."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")

        mock_file = tmp_path / "artifact.txt"
        mock_file.write_text("downloaded")
        cache_root = tmp_path / "cache_home"
        cached_file = cache_root / ".manuscript" / "weights" / "ABCDEFG123.bin"

        with patch.object(base_module.Path, "home", return_value=cache_root):
            with patch.dict("sys.modules", {"gdown": MagicMock()}):
                import gdown

                gdown.download.return_value = str(mock_file)
                model = ConcreteModel(weights=str(weights_file))

                result = model._resolve_extra_artifact(
                    "gdrive:ABCDEFG123",
                    default_name=None,
                    registry={},
                    description="artifact"
                )

        # Verify gdown.download was called with correct file_id
        gdown.download.assert_called_once()
        call_kwargs = gdown.download.call_args.kwargs
        assert call_kwargs['id'] == 'ABCDEFG123'
        assert result == str(cached_file)
        assert cached_file.read_text(encoding="utf-8") == "downloaded"
    
    def test_artifact_no_default_raises_error(self, tmp_path):
        """Test error when no default artifact and None specified."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        model = ConcreteModel(weights=str(weights_file))
        
        with pytest.raises(ValueError, match="no default.*defined"):
            model._resolve_extra_artifact(
                None,
                default_name=None,
                registry={},
                description="config"
            )
    
    def test_unknown_artifact_raises_error(self, tmp_path):
        """Test error for unknown artifact specification."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        model = ConcreteModel(weights=str(weights_file))
        
        with pytest.raises(ValueError, match="Unknown.*artifact"):
            model._resolve_extra_artifact(
                "nonexistent.txt",
                default_name=None,
                registry={},
                description="artifact"
            )


# ============================================================================
# Tests for cache directory
# ============================================================================
class TestCacheDirectory:
    """Tests for cache directory management."""
    
    def test_cache_dir_creation(self, tmp_path):
        """Test cache directory is created."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        model = ConcreteModel(weights=str(weights_file))
        
        cache_dir = model._cache_dir
        
        assert cache_dir.exists()
        assert cache_dir.is_dir()
        assert ".manuscript" in str(cache_dir)
        assert "weights" in str(cache_dir)
    
    def test_cache_dir_in_home(self, tmp_path):
        """Test cache directory is in user home."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        model = ConcreteModel(weights=str(weights_file))
        
        cache_dir = model._cache_dir
        
        assert str(Path.home()) in str(cache_dir)


# ============================================================================
# Tests for download helpers
# ============================================================================
class TestDownloadHelpers:
    """Tests for download helper methods."""
    
    @patch("urllib.request.urlretrieve")
    def test_download_http_new_file(self, mock_urlretrieve, tmp_path):
        """Test downloading a new file via HTTP."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        url = "https://example.com/newfile.onnx"
        
        def fake_download(url, tmp_path):
            Path(tmp_path).write_text("new content")
        
        mock_urlretrieve.side_effect = fake_download
        
        model = ConcreteModel(weights=str(weights_file))
        
        result = model._download_http(url)
        
        assert "newfile.onnx" in result
        assert Path(result).exists()
    
    @patch("urllib.request.urlretrieve")
    def test_download_http_cached_file(self, mock_urlretrieve, tmp_path):
        """Test using cached file instead of re-downloading."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        url = "https://example.com/cached.onnx"
        
        # Pre-create cached file
        cache_dir = Path.home() / ".manuscript" / "weights"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_file = cache_dir / "cached.onnx"
        
        try:
            cached_file.write_text("already cached")
            
            model = ConcreteModel(weights=str(weights_file))
            
            result = model._download_http(url)
            
            # Should NOT download again
            mock_urlretrieve.assert_not_called()
            assert result == str(cached_file)
        finally:
            if cached_file.exists():
                cached_file.unlink()

    def test_download_http_force_download_replaces_cached_file(self, tmp_path):
        """Test force_download refreshes an existing cached HTTP artifact."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock", encoding="utf-8")

        cache_root = tmp_path / "cache_home"
        cached_file = cache_root / ".manuscript" / "weights" / "cached.onnx"
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        cached_file.write_text("stale", encoding="utf-8")

        def fake_urlretrieve(url, target):
            Path(target).write_text("fresh", encoding="utf-8")

        with patch.object(base_module.Path, "home", return_value=cache_root):
            with patch(
                "manuscript.api.base.urllib.request.urlretrieve",
                side_effect=fake_urlretrieve,
            ) as mock_urlretrieve:
                model = ConcreteModel(
                    weights=str(weights_file),
                    force_download=True,
                )
                result = model._download_http("https://example.com/cached.onnx")

        assert result == str(cached_file)
        assert cached_file.read_text(encoding="utf-8") == "fresh"
        assert mock_urlretrieve.call_count == 1
    
    def test_download_github_url_construction(self, tmp_path):
        """Test GitHub URL is correctly constructed."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        model = ConcreteModel(weights=str(weights_file))
        
        with patch.object(model, '_download_http') as mock_download:
            mock_download.return_value = str(tmp_path / "downloaded.onnx")
            
            spec = "github://owner/repo/v1.0.0/weights.onnx"
            result = model._download_github(spec)
            
            expected = "https://github.com/owner/repo/releases/download/v1.0.0/weights.onnx"
            mock_download.assert_called_once_with(expected)
    
    def test_download_gdrive_url_construction(self, tmp_path):
        """Test Google Drive download with gdown."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")

        downloaded_file = tmp_path / "downloaded.onnx"
        downloaded_file.write_text("mock_downloaded")
        cache_root = tmp_path / "cache_home"
        cached_file = cache_root / ".manuscript" / "weights" / "1Ab2Cd3Ef4Gh5.bin"

        with patch.object(base_module.Path, "home", return_value=cache_root):
            with patch.dict("sys.modules", {"gdown": MagicMock()}):
                import gdown

                gdown.download.return_value = str(downloaded_file)
                model = ConcreteModel(weights=str(weights_file))

                spec = "gdrive:1Ab2Cd3Ef4Gh5"
                result = model._download_gdrive(spec)

        # Verify gdown.download was called with correct file_id
        gdown.download.assert_called_once()
        call_kwargs = gdown.download.call_args.kwargs
        assert call_kwargs['id'] == '1Ab2Cd3Ef4Gh5'
        assert call_kwargs['quiet'] == False
        assert result == str(cached_file)
        assert cached_file.read_text(encoding="utf-8") == "mock_downloaded"
    
    def test_download_gdrive_fallback_without_gdown(self, tmp_path):
        """Test Google Drive fallback to HTTP when gdown not available."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        model = ConcreteModel(weights=str(weights_file))
        
        # Mock gdown as not available
        with patch.dict('sys.modules', {'gdown': None}):
            with patch.object(model, '_download_http') as mock_download_http:
                mock_download_http.return_value = str(tmp_path / "downloaded.onnx")
                
                spec = "gdrive:1Ab2Cd3Ef4Gh5"
                result = model._download_gdrive(spec)
                
                # Should fallback to direct URL via _download_http
                expected = "https://drive.google.com/uc?export=download&id=1Ab2Cd3Ef4Gh5"
                mock_download_http.assert_called_once_with(expected)


# ============================================================================
# Tests for model initialization and usage
# ============================================================================
class TestModelInitialization:
    """Tests for model initialization and basic usage."""
    
    def test_model_initialization_with_kwargs(self, tmp_path):
        """Test extra kwargs are stored."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        model = ConcreteModel(
            weights=str(weights_file),
            device="cpu",
            custom_param1="value1",
            custom_param2=42
        )
        
        assert model.extra_config["custom_param1"] == "value1"
        assert model.extra_config["custom_param2"] == 42

    def test_model_initialization_with_force_download(self, tmp_path):
        """Test force_download is stored separately from extra kwargs."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock", encoding="utf-8")

        model = ConcreteModel(
            weights=str(weights_file),
            force_download=True,
            custom_param="value",
        )

        assert model.force_download is True
        assert "force_download" not in model.extra_config
        assert model.extra_config["custom_param"] == "value"
    
    def test_session_initially_none(self, tmp_path):
        """Test session is None on initialization."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        model = ConcreteModel(weights=str(weights_file))
        
        assert model.session is None
    
    def test_initialize_session_called(self, tmp_path):
        """Test _initialize_session creates session."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        model = ConcreteModel(weights=str(weights_file))
        model._initialize_session()
        
        assert model.session is not None
        assert model.session.initialized is True
    
    def test_predict_method(self, tmp_path):
        """Test predict method works."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        model = ConcreteModel(weights=str(weights_file))
        result = model.predict(5)
        
        assert result == {"result": 10}
        assert model.session is not None
    
    def test_call_method_delegates_to_predict(self, tmp_path):
        """Test __call__ delegates to predict."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        model = ConcreteModel(weights=str(weights_file))
        result = model(3)
        
        assert result == {"result": 6}
    
    def test_train_not_implemented_by_default(self, tmp_path):
        """Test train raises NotImplementedError by default."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        # Train is a static method in BaseArtifactModel
        with pytest.raises(NotImplementedError, match="does not support training"):
            ConcreteModel.train()
    
    def test_export_not_implemented_by_default(self, tmp_path):
        """Test export raises NotImplementedError by default."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        # Export is a static method in BaseArtifactModel
        with pytest.raises(NotImplementedError, match="does not support export"):
            ConcreteModel.export()


# ============================================================================
# Integration tests
# ============================================================================
class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_workflow_local_weights(self, tmp_path):
        """Test complete workflow with local weights."""
        # Create weights file
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock weights")
        
        # Initialize model
        model = ConcreteModel(weights=str(weights_file), device="cpu")
        
        # Verify setup
        assert model.device == "cpu"
        assert Path(model.weights).exists()
        assert model.session is None
        
        # Make prediction
        result = model.predict(10)
        
        # Verify prediction and session
        assert result == {"result": 20}
        assert model.session is not None
    
    @patch.object(ConcreteModel, '_download_http')
    def test_full_workflow_preset_weights(self, mock_download, tmp_path):
        """Test complete workflow with preset weights."""
        # Setup mock download
        mock_file = tmp_path / "preset.onnx"
        mock_file.write_text("preset weights")
        mock_download.return_value = str(mock_file)
        
        # Initialize with preset
        model = ConcreteModel(weights="preset1", device="cuda")
        
        # Verify
        assert model.device == "cuda"
        assert Path(model.weights).exists()
        mock_download.assert_called_once()
        
        # Make prediction
        result = model(5)
        assert result == {"result": 10}
    
    def test_model_with_multiple_artifacts(self, tmp_path):
        """Test model using multiple artifacts."""
        # Create files
        weights = tmp_path / "model.onnx"
        config = tmp_path / "config.json"
        
        weights.write_text("weights")
        config.write_text('{"setting": "value"}')
        
        # Initialize
        model = ConcreteModel(weights=str(weights))
        
        # Resolve extra artifact
        config_path = model._resolve_extra_artifact(
            str(config),
            default_name=None,
            registry={},
            description="config"
        )
        
        assert Path(config_path).exists()
        assert Path(model.weights).exists()
