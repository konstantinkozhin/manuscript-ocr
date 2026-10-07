Release Notes
=============

0.1.13
------

Model management
~~~~~~~~~~~~~~~~

Presets and TRBA pretraining now use a shared mirrored registry. Model bundles
store weights, supporting files and a declared license together, validate class,
file size and SHA-256, and reuse verified legacy cache files. Public helpers
download, inspect, verify and remove bundles. Downloads display progress.
Version compatibility mismatches warn and allow execution to continue.
See :doc:`model_registry`.

Inference fixes
~~~~~~~~~~~~~~~

TRBA pads the last incomplete batch to the effective ONNX batch size and returns
only the original predictions. This avoids severe CUDA inference slowdowns on
page-scale runs. Pipeline now respects the recognizer's configured batch size.

On Windows, CUDA initialization loads NVIDIA runtime DLLs from
``site-packages/nvidia`` and retains DLL directory handles. This fixes cuDNN
frontend sublibrary loading. Training imports and heavy public exports are lazy,
so runtime inference avoids unnecessary PyTorch imports.

Training and export
~~~~~~~~~~~~~~~~~~~

TRBA supports a PARSeq decoder, the ``SEResNet31LiteV2`` backbone and optional
TPS rectification. Attention remains the default decoder. Training and ONNX
export preserve architecture settings and use decoder-specific loss and decoding
length. See :doc:`api/recognizers` for configuration.

Checkpoint loading reports keys filtered during compatibility loading.
Checkpoint and weight writes are atomic, CPU Torch RNG state is saved and
restored, and seed setup also seeds NumPy. These changes do not guarantee full
reproducibility across devices or all random generators.

Regression tests cover partial ONNX batches, architecture configuration,
training, export and experiment launching.
