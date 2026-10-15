"""Experimental synchronous session adapter for ONNX Runtime Web via Pyodide JSPI.

The host must register the JavaScript module ``manuscript_ort_web`` using the
bundled bridge. Enter Python through ``pyodide.runPythonAsync`` or
``callPromising``. This preserves the existing synchronous predict methods.
"""
import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np


_TYPES = {
    "float32": "float", "float64": "double", "float16": "float16",
    "int8": "int8", "uint8": "uint8", "int16": "int16", "uint16": "uint16",
    "int32": "int32", "uint32": "uint32", "int64": "int64", "uint64": "uint64",
    "bool": "bool",
}


def _context():
    try:
        from pyodide.ffi import can_run_sync, run_sync, to_js
    except ImportError as exc:
        raise RuntimeError("runtime='web' requires Python running in Pyodide.") from exc
    if not can_run_sync():
        raise RuntimeError(
            "Web inference requires JSPI and an asynchronous Python entry point. "
            "Use a JSPI-capable browser and pyodide.runPythonAsync()/callPromising()."
        )
    try:
        bridge = importlib.import_module("manuscript_ort_web")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Register manuscript_ort_web using bridge.mjs before inference.") from exc
    return bridge, run_sync, to_js


class WebSession:
    """ORT-like session for local ONNX bytes in Pyodide's virtual filesystem.

    Accepts numeric tensors and explicit external-data files. A sibling
    <model>.onnx.data file is discovered automatically. Native SessionOptions
    are not supported.
    """
    def __init__(self, weights, device="cpu", external_data=None, log_severity_level=None):
        if device not in ("cpu", "gpu", "webgpu"):
            raise ValueError("Web runtime device must be 'cpu', 'gpu' or 'webgpu'.")
        bridge, run_sync, to_js = _context()
        data = np.frombuffer(Path(weights).read_bytes(), dtype=np.uint8)
        self._bridge = bridge
        from js import Object
        if external_data is None:
            companion = Path(str(weights) + ".data")
            external_data = [companion] if companion.is_file() else []
        if isinstance(external_data, (str, Path)):
            external_data = [external_data]
        # A mapping can preserve graph-relative names for multi-file exports.
        entries = external_data.items() if isinstance(external_data, dict) else ((Path(path).name, path) for path in external_data)
        files = [{"path": str(name), "data": np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)} for name, path in entries]
        self._session = run_sync(bridge.create_session(to_js(data), "wasm" if device == "cpu" else "webgpu", to_js(files, dict_converter=Object.fromEntries), log_severity_level))
        try:
            self._inputs = self._metadata(self._session.inputs)
            self._outputs = self._metadata(self._session.outputs)
        except Exception:
            run_sync(bridge.release_session(self._session))
            raise
        self._device = device
        self._closed = False

    @staticmethod
    def _metadata(items):
        result = []
        for item in items.to_py():
            dtype = item["type"]
            if dtype not in _TYPES:
                raise TypeError(f"Unsupported web tensor type: {dtype}")
            result.append(SimpleNamespace(name=item["name"], type=f"tensor({_TYPES[dtype]})", shape=item["shape"]))
        return result

    def get_inputs(self):
        return list(self._inputs)

    def get_outputs(self):
        return list(self._outputs)

    def get_providers(self):
        # Requested backend, not a claim that every operator ran on GPU.
        return ["WASMExecutionProvider" if self._device == "cpu" else "WebGPUExecutionProvider"]

    def run(self, output_names, input_feed):
        if self._closed:
            raise RuntimeError("Web session has been closed.")
        bridge, run_sync, to_js = _context()
        from js import Object
        descriptors = []
        for name, value in input_feed.items():
            array = np.asarray(value)
            if not array.flags.c_contiguous:
                array = np.ascontiguousarray(array)
            dtype = str(array.dtype)
            if dtype not in _TYPES:
                raise TypeError(f"Unsupported web input dtype: {dtype}")
            descriptors.append({"name": name, "type": dtype, "shape": list(array.shape), "data": array.reshape(-1).view(np.uint16) if dtype == "float16" else array.reshape(-1)})
        names = [item.name for item in self._outputs] if output_names is None else list(output_names)
        raw = run_sync(bridge.run_session(self._session, to_js(descriptors, dict_converter=Object.fromEntries), to_js(names)))
        arrays = []
        for item in raw:
            dtype = str(item.type)
            if dtype not in _TYPES:
                raise TypeError(f"Unsupported web output dtype: {dtype}")
            values = np.asarray(item.data.to_py())
            values = values.astype(np.uint16, copy=False).view(np.float16) if dtype == "float16" else values.astype(dtype, copy=False)
            arrays.append(values.reshape(tuple(item.shape.to_py())).copy())
        return arrays

    def close(self):
        if not self._closed:
            _, run_sync, _ = _context()
            run_sync(self._bridge.release_session(self._session))
            self._closed = True
