Getting Started
===============

Installation
------------

**Basic installation** (inference only):

.. code-block:: bash

    pip install manuscript-ocr

**Installation with training support** (includes PyTorch):

.. code-block:: bash

    pip install manuscript-ocr[dev]

This installs additional dependencies for model training:

- PyTorch and TorchVision
- ONNX export tools
- Training utilities (albumentations, tensorboard, etc.)
- Development tools (pytest, black, flake8, etc.)

**GPU acceleration** (NVIDIA CUDA):

.. code-block:: bash

    pip install manuscript-ocr
    pip install onnxruntime-gpu

**Apple Silicon acceleration** (CoreML):

.. code-block:: bash

    pip install manuscript-ocr
    pip install onnxruntime-silicon

Quick Start
-----------

Basic usage example:

.. code-block:: python

    from manuscript import Pipeline

    # Create pipeline
    pipeline = Pipeline()

    # Process image
    result = pipeline.predict("document.jpg")

    # Get recognized text
    text = pipeline.get_text(result["page"])
    print(text)

Main Components
---------------

- :class:`~manuscript.Pipeline` - High-level OCR pipeline
- :class:`~manuscript.detectors.EAST` - Text detector
- :class:`~manuscript.layouts.SimpleSorting` - Layout ordering model
- :class:`~manuscript.recognizers.TRBA` - Text recognizer
- :class:`~manuscript.correctors.CharLM` - Character-level text corrector
- :class:`~manuscript.data.Page` - Page data structure
- :class:`~manuscript.data.Block` - Block data structure
- :class:`~manuscript.data.Line` - Line data structure
- :class:`~manuscript.data.Word` - Word data structure

Pipeline Defaults
-----------------

``Pipeline()`` uses:

- ``EAST`` detector
- ``SimpleSorting`` layout
- ``TRBA`` recognizer
- no corrector (``None``)
