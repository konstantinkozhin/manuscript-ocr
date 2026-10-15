# Browser runtime

Manuscript can run its Python pipeline in Pyodide and execute compatible ONNX
models through ONNX Runtime Web. This integration is experimental. It requires
a browser with JSPI support and an asynchronous Python entry point.

## Installation

- `manuscript-ocr`: shared Python components, without ONNX Runtime.
- `manuscript-ocr[cpu]`: native ONNX Runtime, including CoreML on macOS.
- `manuscript-ocr[gpu]`: native ONNX Runtime GPU for NVIDIA CUDA.
- `manuscript-ocr[web]`: browser target; the host page supplies ONNX Runtime Web.
- `[trocr]`: additional tokenizer dependency for TrOCR, combined with a runtime
  extra, e.g. `[cpu,trocr]`. TrOCR tokenization in Pyodide has not been verified.
- `[fast]`: optional Numba acceleration for EAST geometry on native Python.
- `[gdrive]`: optional gdown support for Google Drive downloads.

Install only one native ONNX Runtime distribution. When switching from CPU to
GPU, uninstall `onnxruntime` first. The legacy `[silicon]` extra is an alias for
the native CPU runtime; select `device="coreml"` explicitly.

## Host page

Serve the wheel, model files and their configurations over HTTPS (or localhost).
Extract `manuscript/runtimes/bridge.mjs` from the wheel and serve it alongside the
host JavaScript. Install `onnxruntime-web` separately; version 1.23.2 was tested.
Configure its WASM asset paths for your site.

After creating the Pyodide interpreter, register the bridge once:

```javascript
import * as ort from 'onnxruntime-web/webgpu';
import {registerOrtWeb} from './bridge.mjs';

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
registerOrtWeb(pyodide, ort);
await pyodide.loadPackage('micropip');
await pyodide.runPythonAsync(`
import micropip
await micropip.install("manuscript-ocr[web] @ https://your-site.example/manuscript_ocr-0.1.13-py3-none-any.whl")
`);
```

A bundler resolves the JavaScript package import above. A page without a bundler
can import the served `ort.webgpu.min.mjs` directly. Pyodide loads compatible
Python dependencies; native wheels from a desktop environment cannot be reused.

## Python models

Place models and their matching configurations in Pyodide's filesystem, then
construct the existing processors. For example:

```python
from manuscript import Pipeline
from manuscript.detectors._ppocr import PPOCR
from manuscript.recognizers import PPOCRRec

with PPOCR(weights="/models/detector.onnx", config="/models/detector.json",
           runtime="web", device="gpu") as detector, \
     PPOCRRec(weights="/models/recognizer.onnx", config="/models/recognizer.json",
              runtime="web", device="gpu", recognition_levels=["line"]) as recognizer:
    pipeline = Pipeline(detector=detector, layout=None, recognizer=recognizer)
    result = pipeline.predict("/images/document.png")
```

Use compatible line detection and recognition configurations for this example.
Call this code through `runPythonAsync()` or `callPromising()`, including calls
to `close()`. `runtime="auto"` selects Web in Pyodide and native execution in
ordinary Python. In Web, `device="cpu"` uses WASM and `device="gpu"` or
`"webgpu"` requests WebGPU. Graph compatibility must be checked for each model.

`close()` releases model sessions; the next prediction initializes them again.
Failed initialization releases sessions created during that attempt.

ONNX external data must accompany its graph. A sibling `<model>.onnx.data` is
found automatically; registry artifacts also supply declared companions.
For other names, pass `web_external_data={"name-in-graph": "/models/data.bin"}`.

Browser HTTP requests use Fetch and obey CORS. The default model cache lives in
Pyodide's virtual filesystem; persistence across page reloads is the host's
responsibility. `parallel=True` in Stack and recognizer ensembles requires
native Python threads; use `parallel=False` in Pyodide.
