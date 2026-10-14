"""Tests for BaseArtifactModel class."""

from tests.support.modules import patch_modules

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

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
    
    @pytest.mark.parametrize("url", [
        "http://example.com/model.onnx",
        "https://example.com/model.onnx",
    ])
    def test_url_weights_download_to_cache(self, tmp_path, monkeypatch, url):
        calls = []

        def download(source, target):
            calls.append(source)
            Path(target).write_bytes(b"downloaded weights")

        monkeypatch.setattr(base_module, "tqdm", None)
        monkeypatch.setattr(base_module.urllib.request, "urlretrieve", download)
        model = ConcreteModel(weights=url)
        assert calls == [url]
        assert model.weights == str(model._http_cache_path(url))
        assert Path(model.weights).read_bytes() == b"downloaded weights"

    @pytest.mark.parametrize("spec,expected_url", [
        (
            "github://owner/repo/v1.0/model.onnx",
            "https://github.com/owner/repo/releases/download/v1.0/model.onnx",
        ),
        (
            "github://owner/repo/v2.0/models/subdir/model.onnx",
            "https://github.com/owner/repo/releases/download/v2.0/models/subdir/model.onnx",
        ),
    ])
    def test_github_weights_resolution(self, tmp_path, spec, expected_url):
        weights = tmp_path / "downloaded.onnx"
        weights.write_bytes(b"weights")
        with patch.object(ConcreteModel, "_download_http", return_value=str(weights)) as download:
            model = ConcreteModel(weights=spec)
        download.assert_called_once_with(expected_url)
        assert model.weights == str(weights)

    def test_gdrive_url_resolution(self, tmp_path):
        """Test Google Drive URL resolution with gdown."""
        gdrive_spec = "gdrive:1234567890abcdef"

        mock_file = tmp_path / "model.onnx"
        mock_file.write_text("mock")
        cache_root = tmp_path / "cache_home"
        cached_file = cache_root / ".manuscript" / "weights" / "1234567890abcdef.bin"

        with patch.object(base_module.Path, "home", return_value=cache_root):
            with patch_modules({"gdown": MagicMock()}):
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
        assert result == str(mock_file)
    
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
        assert result == str(mock_file)
    
    def test_resolve_artifact_from_gdrive(self, tmp_path):
        """Test downloading artifact from Google Drive with gdown."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")

        mock_file = tmp_path / "artifact.txt"
        mock_file.write_text("downloaded")
        cache_root = tmp_path / "cache_home"
        cached_file = cache_root / ".manuscript" / "weights" / "ABCDEFG123.bin"

        with patch.object(base_module.Path, "home", return_value=cache_root):
            with patch_modules({"gdown": MagicMock()}):
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

    @pytest.fixture(autouse=True)
    def plain_http_downloads(self, monkeypatch):
        # Progress and fallback behavior have dedicated tests in runtime_edges.
        monkeypatch.setattr(base_module, "tqdm", None)
    
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
        
        assert result == str(model._http_cache_path(url))
        assert Path(result).read_text() == "new content"
        assert mock_urlretrieve.call_count == 1
        assert mock_urlretrieve.call_args.args[0] == url
        assert not Path(mock_urlretrieve.call_args.args[1]).exists()
    
    @patch("urllib.request.urlretrieve")
    def test_download_http_cached_file(self, mock_urlretrieve, tmp_path):
        """Test using cached file instead of re-downloading."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")
        
        url = "https://example.com/cached.onnx"
        
        # Pre-create cached file
        model = ConcreteModel(weights=str(weights_file))
        cached_file = model._http_cache_path(url)
        
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
                cached_file = model._http_cache_path("https://example.com/cached.onnx")
                cached_file.write_text("stale", encoding="utf-8")
                result = model._download_http("https://example.com/cached.onnx")

        assert result == str(cached_file)
        assert cached_file.read_text(encoding="utf-8") == "fresh"
        assert mock_urlretrieve.call_count == 1
    
    def test_download_gdrive_url_construction(self, tmp_path):
        """Test Google Drive download with gdown."""
        weights_file = tmp_path / "model.onnx"
        weights_file.write_text("mock")

        downloaded_file = tmp_path / "downloaded.onnx"
        downloaded_file.write_text("mock_downloaded")
        cache_root = tmp_path / "cache_home"
        cached_file = cache_root / ".manuscript" / "weights" / "1Ab2Cd3Ef4Gh5.bin"

        with patch.object(base_module.Path, "home", return_value=cache_root):
            with patch_modules({"gdown": MagicMock()}):
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
        with patch_modules({'gdown': None}):
            with patch.object(model, '_download_http') as mock_download_http:
                mock_download_http.return_value = str(tmp_path / "downloaded.onnx")
                
                spec = "gdrive:1Ab2Cd3Ef4Gh5"
                result = model._download_gdrive(spec)
                
                # Should fallback to direct URL via _download_http
                expected = "https://drive.google.com/uc?export=download&id=1Ab2Cd3Ef4Gh5"
                mock_download_http.assert_called_once_with(expected)
                assert result == mock_download_http.return_value


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


def test_canonical_registry_forwards_class_and_download_policy(tmp_path, monkeypatch):
    class RegisteredModel(ConcreteModel):
        registry_model_class = "TestRegisteredModel"

    weights = tmp_path / "weights.onnx"
    config = tmp_path / "config.json"
    weights.write_bytes(b"weights")
    config.write_text('{}')
    resolve = Mock(return_value={"weights": weights, "config": config})
    monkeypatch.setattr("manuscript.models.resolve", resolve)
    model = RegisteredModel(weights="preset", force_download=True)
    resolve.assert_called_once_with("preset", "TestRegisteredModel", force_download=True)
    assert model.weights == str(weights)
    assert model._resolved_model_artifacts["config"] == config
    resolve.return_value = {"config": config}
    with pytest.raises(ValueError, match="no inference weights"):
        RegisteredModel(weights="missing_weights")


def test_onnx_session_uses_providers_options_and_runtime_setup(tmp_path, monkeypatch):
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"test")
    model = ConcreteModel(weights=str(weights), device="cpu")
    prepare = Mock()
    log = Mock()
    session = Mock()
    constructor = Mock(return_value=session)
    monkeypatch.setattr(model, "_prepare_runtime_dependencies", prepare)
    monkeypatch.setattr(model, "_log_device_info", log)
    monkeypatch.setattr(base_module.ort, "InferenceSession", constructor)
    options = object()
    assert model._create_onnx_session(sess_options=options) is session
    constructor.assert_called_once_with(str(weights), providers=["CPUExecutionProvider"], sess_options=options)
    prepare.assert_called_once_with()
    log.assert_called_once_with(session)


@pytest.mark.parametrize("description,role", [("model config", "config"), ("vocab", "vocab")])
def test_extra_registry_artifact_forwards_role_and_download_policy(tmp_path, monkeypatch, description, role):
    class RegisteredModel(ConcreteModel):
        registry_model_class = "TestRegisteredModel"

    weights = tmp_path / "weights.onnx"
    weights.write_bytes(b"weights")
    artifact = tmp_path / "artifact.json"
    resolve = Mock(return_value={role: artifact})
    monkeypatch.setattr("manuscript.models.resolve", resolve)
    model = RegisteredModel(weights=weights, force_download=True)
    result = model._resolve_extra_artifact(
        "preset", default_name=None, registry={}, description=description,
    )
    resolve.assert_called_once_with(
        "preset", "TestRegisteredModel", artifact=role, force_download=True,
    )
    assert result == str(artifact)


@pytest.mark.parametrize("missing", ["absolute", "relative", "path_object"])
def test_missing_explicit_extra_path_does_not_resolve_as_preset(tmp_path, monkeypatch, missing):
    class RegisteredModel(ConcreteModel):
        registry_model_class = "TestRegisteredModel"

    weights = tmp_path / "weights.onnx"
    weights.write_bytes(b"weights")
    resolve = Mock()
    monkeypatch.setattr("manuscript.models.resolve", resolve)
    monkeypatch.chdir(tmp_path)
    value = {
        "absolute": str(tmp_path / "missing.json"),
        "relative": "configs/missing.json",
        "path_object": Path("missing.json"),
    }[missing]
    model = RegisteredModel(weights=weights)
    with pytest.raises(FileNotFoundError, match="config file not found"):
        model._resolve_extra_artifact(value, default_name=None, registry={}, description="config")
    resolve.assert_not_called()
