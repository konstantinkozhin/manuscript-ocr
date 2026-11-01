"""Native ONNX Runtime access, imported only when execution is requested."""
import importlib


def get_native_runtime():
    """Return the installed CPU or CUDA runtime; both expose the same module."""
    try:
        return importlib.import_module("onnxruntime")
    except ModuleNotFoundError as exc:
        if exc.name != "onnxruntime":
            raise
        raise ImportError(
            'Model execution requires an ONNX runtime. Install "manuscript-ocr[cpu]" '
            'or "manuscript-ocr[gpu]". Browser execution requires Pyodide and '
            'the ONNX Runtime Web bridge.'
        ) from exc


def create_native_session(weights, providers, *, log_severity_level=None, **options):
    ort = get_native_runtime()
    if log_severity_level is not None:
        session_options = options.get("sess_options")
        if session_options is None:
            session_options = ort.SessionOptions()
            options["sess_options"] = session_options
        session_options.log_severity_level = log_severity_level
    return ort.InferenceSession(str(weights), providers=providers, **options)
