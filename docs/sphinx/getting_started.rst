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

Example Notebooks
-----------------

Current example notebooks are available in the repository ``notebooks``
folder:

- `End-to-end inference <https://github.com/konstantinkozhin/manuscript-ocr/blob/main/notebooks/%D0%9F%D1%80%D0%B8%D0%BC%D0%B5%D1%80_%D0%B8%D0%BD%D1%84%D0%B5%D1%80%D0%B5%D0%BD%D1%81%D0%B0_endtoend.ipynb>`_
- `Pipeline with YOLO detector <https://github.com/konstantinkozhin/manuscript-ocr/blob/main/notebooks/%D0%9F%D1%80%D0%B8%D0%BC%D0%B5%D1%80_%D0%B7%D0%B0%D0%BF%D1%83%D1%81%D0%BA%D0%B0_pipeline_%D1%81_yolo.ipynb>`_
- `Pipeline with TrOCR recognizer <https://github.com/konstantinkozhin/manuscript-ocr/blob/main/notebooks/%D0%9F%D1%80%D0%B8%D0%BC%D0%B5%D1%80_%D0%B7%D0%B0%D0%BF%D1%83%D1%81%D0%BA%D0%B0_pipeline_%D1%81_trocr.ipynb>`_
- `Pipeline with Yandex Speller <https://github.com/konstantinkozhin/manuscript-ocr/blob/main/notebooks/%D0%9F%D1%80%D0%B8%D0%BC%D0%B5%D1%80_%D0%B7%D0%B0%D0%BF%D1%83%D1%81%D0%BA%D0%B0_pipeline_%D1%81_yandex_speller.ipynb>`_
- `Gradio demo launch <https://github.com/konstantinkozhin/manuscript-ocr/blob/main/notebooks/%D0%9F%D1%80%D0%B8%D0%BC%D0%B5%D1%80_%D0%B7%D0%B0%D0%BF%D1%83%D1%81%D0%BA%D0%B0_%D0%B4%D0%B5%D0%BC%D0%BE.ipynb>`_
- `Detector training launch <https://github.com/konstantinkozhin/manuscript-ocr/blob/main/notebooks/%D0%9F%D1%80%D0%B8%D0%BC%D0%B5%D1%80_%D0%B7%D0%B0%D0%BF%D1%83%D1%81%D0%BA%D0%B0_%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D1%8F_%D0%B4%D0%B5%D1%82%D0%B5%D0%BA%D1%82%D0%BE%D1%80%D0%B0.ipynb>`_
- `Recognition training launch <https://github.com/konstantinkozhin/manuscript-ocr/blob/main/notebooks/%D0%9F%D1%80%D0%B8%D0%BC%D0%B5%D1%80_%D0%B7%D0%B0%D0%BF%D1%83%D1%81%D0%BA%D0%B0_%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D1%8F_%D1%80%D0%B0%D1%81%D0%BF%D0%BE%D0%B7%D0%BD%D0%B0%D0%B2%D0%B0%D0%BD%D0%B8%D1%8F.ipynb>`_
- `Corrector training launch <https://github.com/konstantinkozhin/manuscript-ocr/blob/main/notebooks/%D0%9F%D1%80%D0%B8%D0%BC%D0%B5%D1%80_%D0%B7%D0%B0%D0%BF%D1%83%D1%81%D0%BA%D0%B0_%D0%BE%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D1%8F_%D0%BA%D0%BE%D1%80%D1%80%D0%B5%D0%BA%D1%82%D0%BE%D1%80%D0%B0.ipynb>`_

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
- :class:`~manuscript.data.TextSpan` - Smallest OCR text region

Model Zoo
---------

For the list of built-in presets and release artifacts documented for this
documentation version, see :doc:`model_zoo`.
