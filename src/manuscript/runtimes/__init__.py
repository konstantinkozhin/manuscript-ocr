"""Common session creation for native ONNX Runtime and ONNX Runtime Web."""
from ._native import get_native_runtime


def create_session(weights, *, runtime, device, providers=None,
                   log_severity_level=None, external_data=None, **native_options):
    """Translate common options without exposing backend objects to models."""
    if log_severity_level is not None and (
        isinstance(log_severity_level, bool) or not isinstance(log_severity_level, int)
        or not 0 <= log_severity_level <= 4
    ):
        raise ValueError("log_severity_level must be an integer from 0 to 4")
    if runtime == "native":
        from ._native import create_native_session
        return create_native_session(weights, providers, log_severity_level=log_severity_level, **native_options)
    if runtime == "web":
        if native_options:
            raise ValueError("Native ONNX session options are not supported by the web runtime.")
        from .web import WebSession
        return WebSession(weights, device=device, external_data=external_data,
                          log_severity_level=log_severity_level)
    raise ValueError("runtime must be 'native' or 'web'")


__all__ = ["create_session", "get_native_runtime"]
