from abc import ABC, abstractmethod
import contextlib
import io
import os
from pathlib import Path
import sys
from typing import Dict, Optional
import shutil
import tempfile
import time
import urllib.error
import urllib.request

import onnxruntime as ort

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class BaseArtifactModel(ABC):
    """
    Shared infrastructure for artifact-backed OCR stages.

    The class provides the common mechanics used by detectors, recognizers,
    layouts, and correctors:

    - artifact loading from local files, URLs, GitHub releases, Google Drive,
      and preset registries
    - device selection and ONNX Runtime provider resolution
    - unified ``predict`` / ``train`` / ``export`` surface
    """

    default_weights_name: Optional[str] = None
    pretrained_registry: Dict[str, str] = {}

    def __init__(
        self,
        weights: Optional[str] = None,
        device: Optional[str] = None,
        force_download: bool = False,
        **kwargs,
    ):
        self.device = self._resolve_device(device)
        self.force_download = bool(force_download)
        self.weights = self._resolve_weights(weights)
        self.extra_config = kwargs
        self.session = None
        self._runtime_deps_preloaded = False
        self._runtime_preload_message: Optional[str] = None
        self._runtime_dll_dir_handles = []

    # -------------------------------------------------------------------------
    # DEVICE
    # -------------------------------------------------------------------------
    def _resolve_device(self, device: Optional[str]) -> str:
        if device is not None:
            return device

        try:
            if ort.get_device().upper() == "GPU":
                return "cuda"
        except Exception:
            pass

        return "cpu"

    def runtime_providers(self):
        """
        Get ONNX Runtime execution providers based on device.
        
        Returns appropriate providers for:
        - CUDA (NVIDIA GPU): CUDAExecutionProvider
        - CoreML (Apple Silicon): CoreMLExecutionProvider
        - CPU: CPUExecutionProvider
        
        Note: GPU/CoreML providers require separate installation:
        - CUDA: pip install onnxruntime-gpu
        - Apple Silicon: pip install onnxruntime-silicon
        """
        if self.device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif self.device == "coreml":
            return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]

    def _prepare_runtime_dependencies(self) -> None:
        if self.device != "cuda" or self._runtime_deps_preloaded:
            return

        self._runtime_deps_preloaded = True
        messages = []

        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            nvidia_root = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
            dll_dirs = []
            for package_name in (
                "cudnn",
                "cublas",
                "cuda_runtime",
                "cufft",
                "nvjitlink",
            ):
                candidate = nvidia_root / package_name / "bin"
                if candidate.exists():
                    dll_dirs.append(candidate)

            added_dirs = []
            add_errors = []
            for dll_dir in dll_dirs:
                try:
                    handle = os.add_dll_directory(str(dll_dir))
                    self._runtime_dll_dir_handles.append(handle)
                    added_dirs.append(str(dll_dir))
                except Exception as exc:
                    add_errors.append(f"{dll_dir}: {exc}")

            if added_dirs:
                messages.append("Added Windows DLL directories:")
                messages.extend(f"  {path}" for path in added_dirs)
            if add_errors:
                messages.append("Failed to add some Windows DLL directories:")
                messages.extend(f"  {item}" for item in add_errors)

        if not hasattr(ort, "preload_dlls"):
            if messages:
                self._runtime_preload_message = "\n".join(messages)
            return

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
                ort.preload_dlls(directory="")
        except Exception as exc:
            preload_message = f"onnxruntime.preload_dlls failed: {exc}"
            if messages:
                messages.append(preload_message)
                self._runtime_preload_message = "\n".join(messages)
            else:
                self._runtime_preload_message = preload_message
            return

        captured = list(messages)
        stdout_text = stdout_buffer.getvalue().strip()
        stderr_text = stderr_buffer.getvalue().strip()
        if stdout_text:
            captured.append(stdout_text)
        if stderr_text:
            captured.append(stderr_text)
        if captured:
            self._runtime_preload_message = "\n".join(captured)
    
    def _log_device_info(self, session):
        """
        Log information about requested and actual execution providers.
        
        Parameters
        ----------
        session : onnxruntime.InferenceSession
            The initialized ONNX Runtime session
        """
        requested_providers = self.runtime_providers()
        actual_providers = session.get_providers()
        
        print(f"[{self.__class__.__name__}] Device configuration:")
        print(f"  Requested device: {self.device}")
        print(f"  Requested providers: {requested_providers}")
        print(f"  Active providers: {actual_providers}")
        
        # Check if primary provider is available
        if self.device == "cuda" and "CUDAExecutionProvider" not in actual_providers:
            print(f"  Warning: CUDA requested but not available. Falling back to CPU.")
            available_providers = set()
            try:
                available_providers = set(ort.get_available_providers())
            except Exception:
                pass

            if "CUDAExecutionProvider" not in available_providers:
                print(f"      Install onnxruntime-gpu for CUDA support: pip install onnxruntime-gpu")
            else:
                print("      CUDAExecutionProvider is installed, but its CUDA/cuDNN DLLs did not initialize.")
                print("      Install CUDA 12 / cuDNN 9 runtime DLLs or let onnxruntime preload them.")
                if self._runtime_preload_message:
                    print("      preload_dlls output:")
                    for line in self._runtime_preload_message.splitlines():
                        print(f"        {line}")
        elif self.device == "coreml" and "CoreMLExecutionProvider" not in actual_providers:
            print(f"  Warning: CoreML requested but not available. Falling back to CPU.")
            print(f"      Install onnxruntime-silicon for CoreML support: pip install onnxruntime-silicon")
        else:
            primary_provider = actual_providers[0] if actual_providers else "Unknown"
            print(f"  Running on: {primary_provider}")

    # -------------------------------------------------------------------------
    # WEIGHT RESOLUTION (main artifact)
    # -------------------------------------------------------------------------
    def _resolve_weights(self, weights: Optional[str]) -> str:
        if weights is None:
            if not self.default_weights_name:
                raise ValueError(
                    f"{self.__class__.__name__} must define default_weights_name"
                )
            weights = self.default_weights_name

        w = str(weights)

        # 1. Local file
        if Path(w).expanduser().exists():
            return str(Path(w).expanduser().absolute())

        # 2. URL
        if w.startswith(("http://", "https://")):
            return self._download_http(w)

        # 3. GitHub
        if w.startswith("github://"):
            return self._download_github(w)

        # 4. Google Drive
        if w.startswith("gdrive:"):
            return self._download_gdrive(w)

        # 5. Preset registry
        if w in self.pretrained_registry:
            return self._resolve_weights(self.pretrained_registry[w])

        raise ValueError(
            f"Unknown weights '{weights}'. Supported: local path, URL, "
            f"github://.., gdrive:ID, presets={list(self.pretrained_registry.keys())}"
        )

    # -------------------------------------------------------------------------
    # GENERIC EXTRA ARTIFACT RESOLUTION
    # -------------------------------------------------------------------------
    def _resolve_extra_artifact(
        self,
        value: Optional[str],
        *,
        default_name: Optional[str],
        registry: Dict[str, str],
        description: str = "artifact",
    ) -> str:
        """
        Universal resolver for auxiliary artifacts (config, charset, vocab, etc.).

        Supports:
            - None → use default_name
            - local path
            - URL
            - github://
            - gdrive:
            - preset (lookup in registry)
        """

        # 0) Default
        if value is None:
            if default_name is None:
                raise ValueError(
                    f"{self.__class__.__name__}: no default {description} defined."
                )
            value = default_name

        v = str(value)

        # 1) Local file
        if Path(v).expanduser().exists():
            return str(Path(v).expanduser().absolute())

        # 2) URL
        if v.startswith(("http://", "https://")):
            return self._download_http(v)

        # 3) GitHub
        if v.startswith("github://"):
            return self._download_github(v)

        # 4) Google Drive
        if v.startswith("gdrive:"):
            return self._download_gdrive(v)

        # 5) Preset
        if v in registry:
            return self._resolve_extra_artifact(
                registry[v],
                default_name=default_name,
                registry=registry,
                description=description,
            )

        raise ValueError(
            f"Unknown {description} '{value}'. "
            f"Supported: local file, URL, github://, gdrive:, presets={list(registry.keys())}"
        )

    # -------------------------------------------------------------------------
    # DOWNLOAD HELPERS
    # -------------------------------------------------------------------------
    @property
    def _cache_dir(self) -> Path:
        d = Path.home() / ".manuscript" / "weights"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _download_http(self, url: str) -> str:
        """Download file from HTTP/HTTPS URL"""
        file = self._cache_dir / Path(url).name
        if file.exists() and not self.force_download:
            return str(file)

        action = "Re-downloading" if file.exists() else "Downloading"
        print(f"{action} {Path(url).name} from {url}")
        max_attempts = 5
        retry_statuses = {429, 500, 502, 503, 504}
        base_backoff_seconds = 1.0
        last_error = None

        for attempt in range(1, max_attempts + 1):
            tmp = tempfile.NamedTemporaryFile(delete=False).name

            try:
                if tqdm is not None:
                    try:
                        # Get file size
                        with urllib.request.urlopen(url) as response:
                            total_size = int(response.headers.get("content-length", 0))

                        with tqdm(
                            total=total_size,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=Path(url).name,
                            ncols=80,
                        ) as pbar:

                            def reporthook(block_num, block_size, total_size):
                                downloaded = block_num * block_size
                                if downloaded < total_size:
                                    pbar.update(block_size)
                                else:
                                    pbar.update(total_size - pbar.n)

                            urllib.request.urlretrieve(url, tmp, reporthook=reporthook)
                    except Exception as e:
                        # Fallback to simple download if progress bar fails
                        print(f"Progress bar error ({e}), downloading without progress...")
                        urllib.request.urlretrieve(url, tmp)
                else:
                    # No tqdm available, simple download
                    urllib.request.urlretrieve(url, tmp)

                # Overwrite cached files only after a successful download.
                if file.exists():
                    shutil.copy2(tmp, file)
                else:
                    shutil.move(tmp, file)
                print(f" Downloaded to {file}")
                return str(file)

            except urllib.error.HTTPError as e:
                last_error = e
                if e.code not in retry_statuses or attempt == max_attempts:
                    raise
                delay = base_backoff_seconds * (2 ** (attempt - 1))
                print(
                    f"Download failed with HTTP {e.code}. "
                    f"Retrying in {delay:.1f}s ({attempt}/{max_attempts - 1})..."
                )
                time.sleep(delay)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_error = e
                if attempt == max_attempts:
                    raise
                delay = base_backoff_seconds * (2 ** (attempt - 1))
                print(
                    f"Download failed ({e}). "
                    f"Retrying in {delay:.1f}s ({attempt}/{max_attempts - 1})..."
                )
                time.sleep(delay)
            finally:
                tmp_path = Path(tmp)
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Failed to download: {url}")

    def _download_github(self, spec: str) -> str:
        payload = spec.replace("github://", "").strip()
        owner, repo, tag, *path_parts = payload.split("/")
        url = f"https://github.com/{owner}/{repo}/releases/download/{tag}/{'/'.join(path_parts)}"
        return self._download_http(url)

    def _download_gdrive(self, spec: str) -> str:
        """Download file from Google Drive with progress bar."""
        file_id = spec.split("gdrive:", 1)[1]
        file: Optional[Path] = None
        tmp: Optional[str] = None
        downloaded_path: Optional[Path] = None

        # Check if gdown is available for better GDrive support
        try:
            import gdown

            # Extract filename from cache or use file_id
            file = self._cache_dir / f"{file_id}.bin"  # Will be renamed after download
            if file.exists() and not self.force_download:
                return str(file)

            action = "Re-downloading" if file.exists() else "Downloading"
            print(f"{action} from Google Drive (ID: {file_id})")
            tmp = tempfile.NamedTemporaryFile(delete=False).name
            output = gdown.download(id=file_id, output=tmp, quiet=False)

            if output is None:
                raise RuntimeError(f"Failed to download from Google Drive: {file_id}")

            downloaded_path = Path(output)
            if file.exists():
                shutil.copy2(downloaded_path, file)
            else:
                shutil.move(downloaded_path, file)
            return str(file)
        except ImportError:
            # Fallback to direct URL (may not work for large files)
            print(
                "Warning: gdown not installed. Using direct URL (may fail for large files)"
            )
            print("Install gdown for better Google Drive support: pip install gdown")
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            return self._download_http(url)
        finally:
            if tmp:
                tmp_file = Path(tmp)
                if tmp_file.exists():
                    try:
                        tmp_file.unlink()
                    except OSError:
                        pass
            if (
                downloaded_path
                and downloaded_path != file
                and downloaded_path.exists()
            ):
                try:
                    downloaded_path.unlink()
                except OSError:
                    pass

    # -------------------------------------------------------------------------
    # BACKEND INITIALIZATION
    # -------------------------------------------------------------------------
    @abstractmethod
    def _initialize_session(self): ...

    # -------------------------------------------------------------------------
    # INFERENCE
    # -------------------------------------------------------------------------
    @abstractmethod
    def predict(self, *args, **kwargs): ...

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    # -------------------------------------------------------------------------
    # OPTIONAL API
    # -------------------------------------------------------------------------
    @staticmethod
    def train(*args, **kwargs):
        raise NotImplementedError("This model does not support training.")

    @staticmethod
    def export(*args, **kwargs):
        raise NotImplementedError("This model does not support export.")


__all__ = ["BaseArtifactModel"]
