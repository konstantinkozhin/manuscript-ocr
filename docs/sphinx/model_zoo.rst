Model Zoo
=========

This page lists the presets and release artifacts documented for version |release|.

Detection
---------

.. list-table::
   :class: model-zoo-table
   :widths: 16 16 9 9 28 16 12
   :header-rows: 1

   * - Preset
     - Architecture
     - Params
     - Rotated
     - Artifacts
     - Origin
     - License
   * - ``east_50_g1``
     - EAST (ResNet-50)
     - 53.86M
     - Yes
     - | `east_50_g1.onnx <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/east_50_g1.onnx>`_
       | `east_50_g1.pth <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/east_50_g1.pth>`_
     - Manuscript
     - MIT
   * - ``yolo26s_obb_text_g1``
     - YOLO26-S OBB
     - 9.75M
     - Yes
     - | `yolo26s_obb_text_g1.raw.onnx <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/yolo26s_obb_text_g1.raw.onnx>`_
       | `yolo26s_obb_text_g1.pt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/yolo26s_obb_text_g1.pt>`_
       | `yolo26s_obb_text_g1.raw.yaml <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/yolo26s_obb_text_g1.raw.yaml>`_
     - Trained by the authors with Ultralytics YOLO
     - `Ultralytics license <https://github.com/ultralytics/ultralytics/blob/main/LICENSE>`_
   * - ``yolo26x_obb_text_g1``
     - YOLO26-X OBB
     - 57.61M
     - Yes
     - | `yolo26x_obb_text_g1.raw.onnx <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/yolo26x_obb_text_g1.raw.onnx>`_
       | `yolo26x_obb_text_g1.pt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/yolo26x_obb_text_g1.pt>`_
       | `yolo26x_obb_text_g1.raw.yaml <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/yolo26x_obb_text_g1.raw.yaml>`_
     - Trained by the authors with Ultralytics YOLO
     - `Ultralytics license <https://github.com/ultralytics/ultralytics/blob/main/LICENSE>`_

Layout
------

.. list-table::
   :class: model-zoo-table
   :widths: 16 20 8 22 12 12 10
   :header-rows: 1

   * - Preset
     - Architecture
     - Params
     - Supports
     - Artifacts
     - Origin
     - License
   * - ``SimpleSorting``
     - Algorithmic ordering
     - -
     - Left-to-right, multi-column
     - -
     - Manuscript
     - MIT

Recognition
-----------

.. list-table::
   :class: model-zoo-table
   :widths: 16 14 8 18 28 10 8
   :header-rows: 1

   * - Preset
     - Architecture
     - Params
     - Script
     - Artifacts
     - Origin
     - License
   * - ``trba_base_g1``
     - TRBA
     - 45.10M
     - Modern + pre-reform Russian
     - | `trba_base_g1.onnx <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_base_g1.onnx>`_
       | `trba_base_g1.pth <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_base_g1.pth>`_
       | `trba_base_g1.json <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_base_g1.json>`_
       | `trba_base_g1.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_base_g1.txt>`_
     - Manuscript
     - MIT
   * - ``trba_lite_g1``
     - TRBA-Lite
     - 9.46M
     - Modern + pre-reform Russian
     - | `trba_lite_g1.onnx <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g1.onnx>`_
       | `trba_lite_g1.pth <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g1.pth>`_
       | `trba_lite_g1.json <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g1.json>`_
       | `trba_lite_g1.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g1.txt>`_
     - Manuscript
     - MIT
   * - ``trba_lite_g2``
     - TRBA-Lite
     - 9.46M
     - Modern + pre-reform Russian
     - | `trba_lite_g2.onnx <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g2.onnx>`_
       | `trba_lite_g2.pth <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g2.pth>`_
       | `trba_lite_g2.json <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g2.json>`_
       | `trba_lite_g2.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g2.txt>`_
     - Manuscript
     - MIT

Correction
----------

.. list-table::
   :class: model-zoo-table
   :widths: 16 14 8 16 28 10 8
   :header-rows: 1

   * - Preset
     - Architecture
     - Params
     - Orthography
     - Artifacts
     - Origin
     - License
   * - ``modern_charlm_g1``
     - CharLM
     - 4.38M
     - Modern Russian
     - | `modern_charlm_g1.onnx <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/modern_charlm_g1.onnx>`_
       | `modern_charlm_g1.pt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/modern_charlm_g1.pt>`_
       | `modern_charlm_g1.json <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/modern_charlm_g1.json>`_
       | `modern_words.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/modern_words.txt>`_
     - Manuscript
     - MIT
   * - ``prereform_charlm_g1``
     - CharLM
     - 4.39M
     - Pre-reform Russian
     - | `prereform_charlm_g1.onnx <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/prereform_charlm_g1.onnx>`_
       | `prereform_charlm_g1.pt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/prereform_charlm_g1.pt>`_
       | `prereform_charlm_g1.json <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/prereform_charlm_g1.json>`_
       | `prereform_words.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/prereform_words.txt>`_
     - Manuscript
     - MIT

Architecture Sources
--------------------

- `EAST: An Efficient and Accurate Scene Text Detector <https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_EAST_An_Efficient_CVPR_2017_paper.pdf>`_
  is the academic source for the ``EAST`` detector family documented here.
- `What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis <https://arxiv.org/abs/1904.01906>`_
  is the architectural basis for the ``TRBA`` family
  (TPS-ResNet-BiLSTM-Attn). The recognizers in manuscript-ocr are adapted
  for this project and are not a 1:1 reproduction of the original
  implementation.
