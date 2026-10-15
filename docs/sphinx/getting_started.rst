Getting Started
===============

Installation
------------

Basic installation (inference only)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install "manuscript-ocr[cpu]"

The base package (``pip install manuscript-ocr``) installs shared components
without an inference runtime. Choose ``[cpu]``, ``[gpu]`` or ``[web]``;
TrOCR additionally requires ``[trocr]``, for example ``[cpu,trocr]``.

GPU acceleration (NVIDIA CUDA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When switching from CPU, first remove ``onnxruntime``. Then install the GPU extra:

.. code-block:: bash

    pip uninstall onnxruntime
    pip install "manuscript-ocr[gpu]"

If CUDA/cuDNN libraries are not installed, also run:

.. code-block:: bash

    pip install "onnxruntime-gpu[cuda,cudnn]"

In notebooks, restart the kernel after installation.

You can switch models and pipeline components explicitly with the ``device``
parameter, for example ``device="cuda"`` for NVIDIA GPU or ``device="cpu"``
for CPU:

.. code-block:: python

    from manuscript.detectors import EAST
    from manuscript.recognizers import TRBA
    from manuscript.correctors import CharLM

    detector = EAST(device="cuda")
    recognizer = TRBA(device="cuda")
    corrector = CharLM(device="cuda")

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

For stage configuration, custom processors, ensembles and result handling,
see :doc:`pipeline_usage`.

.. _example-notebooks:

Example Notebooks
-----------------

Notebooks for version 0.1.13:

- `Training the EAST detector <https://github.com/konstantinkozhin/manuscript-ocr/blob/v_0_1_13/notebooks/v_0_1_13/%D0%9E%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5_%D0%B4%D0%B5%D1%82%D0%B5%D0%BA%D1%82%D0%BE%D1%80%D0%B0_EAST.ipynb>`_.
- `Training the TRBA recognizer <https://github.com/konstantinkozhin/manuscript-ocr/blob/v_0_1_13/notebooks/v_0_1_13/%D0%9E%D0%B1%D1%83%D1%87%D0%B5%D0%BD%D0%B8%D0%B5_%D1%80%D0%B0%D1%81%D0%BF%D0%BE%D0%B7%D0%BD%D0%B0%D0%B2%D0%B0%D1%82%D0%B5%D0%BB%D1%8F_TRBA.ipynb>`_.
- `Integrating a custom detector into the pipeline <https://github.com/konstantinkozhin/manuscript-ocr/blob/v_0_1_13/notebooks/v_0_1_13/%D0%98%D0%BD%D1%82%D0%B5%D0%B3%D1%80%D0%B0%D1%86%D0%B8%D1%8F_%D0%BF%D0%BE%D0%BB%D1%8C%D0%B7%D0%BE%D0%B2%D0%B0%D1%82%D0%B5%D0%BB%D1%8C%D1%81%D0%BA%D0%BE%D0%B3%D0%BE_%D0%B4%D0%B5%D1%82%D0%B5%D0%BA%D1%82%D0%BE%D1%80%D0%B0_%D0%B2_%D0%BF%D0%B0%D0%B9%D0%BF%D0%BB%D0%B0%D0%B9%D0%BD.ipynb>`_.
- `Integrating a custom recognizer into the pipeline <https://github.com/konstantinkozhin/manuscript-ocr/blob/v_0_1_13/notebooks/v_0_1_13/%D0%98%D0%BD%D1%82%D0%B5%D0%B3%D1%80%D0%B0%D1%86%D0%B8%D1%8F_%D0%BF%D0%BE%D0%BB%D1%8C%D0%B7%D0%BE%D0%B2%D0%B0%D1%82%D0%B5%D0%BB%D1%8C%D1%81%D0%BA%D0%BE%D0%B3%D0%BE_%D1%80%D0%B0%D1%81%D0%BF%D0%BE%D0%B7%D0%BD%D0%B0%D0%B2%D0%B0%D1%82%D0%B5%D0%BB%D1%8F_%D0%B2_%D0%BF%D0%B0%D0%B9%D0%BF%D0%BB%D0%B0%D0%B9%D0%BD.ipynb>`_.
- `Integrating a custom corrector into the pipeline <https://github.com/konstantinkozhin/manuscript-ocr/blob/v_0_1_13/notebooks/v_0_1_13/%D0%98%D0%BD%D1%82%D0%B5%D0%B3%D1%80%D0%B0%D1%86%D0%B8%D1%8F_%D0%BF%D0%BE%D0%BB%D1%8C%D0%B7%D0%BE%D0%B2%D0%B0%D1%82%D0%B5%D0%BB%D1%8C%D1%81%D0%BA%D0%BE%D0%B3%D0%BE_%D0%BA%D0%BE%D1%80%D1%80%D0%B5%D0%BA%D1%82%D0%BE%D1%80%D0%B0_%D0%B2_%D0%BF%D0%B0%D0%B9%D0%BF%D0%BB%D0%B0%D0%B9%D0%BD.ipynb>`_.
- `Combining models and building ensembles <https://github.com/konstantinkozhin/manuscript-ocr/blob/v_0_1_13/notebooks/v_0_1_13/%D0%9E%D0%B1%D1%8A%D0%B5%D0%B4%D0%B8%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5_%D0%B8_%D0%B0%D0%BD%D1%81%D0%B0%D0%BC%D0%B1%D0%BB%D0%B8%D1%80%D0%BE%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5_%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B5%D0%B9.ipynb>`_.
- `Exporting and using yolov8m-historical-line-segmenter <https://github.com/konstantinkozhin/manuscript-ocr/blob/v_0_1_13/notebooks/v_0_1_13/%D0%AD%D0%BA%D1%81%D0%BF%D0%BE%D1%80%D1%82_%D0%B8_%D0%BF%D0%BE%D0%B4%D0%BA%D0%BB%D1%8E%D1%87%D0%B5%D0%BD%D0%B8%D0%B5_yolov8m-historical-line-segmenter.ipynb>`_.
- `Exporting and using cyrillic PP-OCRv5 mobile recognition <https://github.com/konstantinkozhin/manuscript-ocr/blob/v_0_1_13/notebooks/v_0_1_13/%D0%AD%D0%BA%D1%81%D0%BF%D0%BE%D1%80%D1%82_%D0%B8_%D0%BF%D0%BE%D0%B4%D0%BA%D0%BB%D1%8E%D1%87%D0%B5%D0%BD%D0%B8%D0%B5_cyrillic_PP-OCRv5_mobile_rec.ipynb>`_.

Additional installation options
-------------------------------

Apple Silicon acceleration (CoreML)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install "manuscript-ocr[cpu]"

Official macOS ``onnxruntime`` wheels include the CoreML execution provider;
a separate ``onnxruntime-silicon`` package is not needed. Use
``device="coreml"`` for the relevant models or pipeline components. See the
`ONNX Runtime CoreML documentation <https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html>`_.

Installation with training support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    pip install "manuscript-ocr[cpu,dev]"

This installs additional dependencies for model training:

- PyTorch and TorchVision
- ONNX export tools
- Training utilities (albumentations, tensorboard, etc.)
- Development tools (pytest, black, flake8, etc.)

Minimum System Requirements
---------------------------

- Python 3.9.2 or newer
- Linux (Ubuntu 20.04+), Windows 10+, or macOS 11+
- at least 2 CPU cores
- at least 8 GB RAM
- 16 GB RAM or more is recommended for model training
- NVIDIA GPU with CUDA support is recommended for acceleration
- at least 4 GB VRAM is recommended for GPU execution

CPU-only execution is supported, but performance may be significantly lower
than on GPU. Model training currently targets CUDA-capable NVIDIA GPUs and
falls back to CPU when CUDA is unavailable.
