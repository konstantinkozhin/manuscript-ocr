Model Zoo
=========

This page lists the 12 model presets documented for version |release|.
The model registry and model files are hosted separately from the library and
can be updated independently of the installed package. The official registry is
``registry.json`` on
`GitHub <https://github.com/konstantinkozhin/manuscript-ocr/blob/main/registry.json>`_,
with a `GitVerse mirror <https://gitverse.ru/api/repos/konstantin_kozhin/manuscript-ocr/raw/branch/main/registry.json>`_.
When you select a preset by its model key, the required files are downloaded
automatically.

Detection
---------

.. list-table::
   :class: model-zoo-table
   :header-rows: 1
   :widths: 25 12 18 45

   * - Model key
     - Class
     - License
     - Artifacts
   * - ``east_50_g1``
     - `EAST <https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_EAST_An_Efficient_CVPR_2017_paper.pdf>`__
     - MIT
     - | weights (required): `east_50_g1.onnx <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/east_50_g1.onnx>`_
       | checkpoint (optional): `east_50_g1.pth <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/east_50_g1.pth>`_
       | config (required): `east_50_g1.json <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/east_50_g1.json>`_
       | license (required): `east_50_g1.LICENSE.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/east_50_g1.LICENSE.txt>`__
   * - ``east_50_yenisei_gov_reports_g1``
     - `EAST <https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_EAST_An_Efficient_CVPR_2017_paper.pdf>`__
     - MIT
     - | weights (required): `east_50_yenisei_gov_reports_g1.onnx <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/east_50_yenisei_gov_reports_g1.onnx>`_
       | checkpoint (optional): `east_50_yenisei_gov_reports_g1.pth <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/east_50_yenisei_gov_reports_g1.pth>`_
       | config (required): `east_50_yenisei_gov_reports_g1.json <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/east_50_yenisei_gov_reports_g1.json>`_
       | license (required): `east_50_yenisei_gov_reports_g1.LICENSE.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/east_50_yenisei_gov_reports_g1.LICENSE.txt>`__
   * - ``mask2former_line_v0_prev``
     - `Mask2Former <https://arxiv.org/abs/2112.01527>`__
     - CC-BY-NC-3.0
     - | weights (required): `mask2former_line_v0_prev.onnx <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/mask2former_line_v0_prev.onnx>`_
       | external_data (required): `mask2former_line_v0_prev.onnx.data <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/mask2former_line_v0_prev.onnx.data>`_
       | config (required): `mask2former_line_v0_prev.json <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/mask2former_line_v0_prev.json>`_
       | license (required): `mask2former_line_v0_prev.LICENSE.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/mask2former_line_v0_prev.LICENSE.txt>`__
   * - ``yolo26s_obb_text_g1``
     - `YOLO26 OBB <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26-obb.yaml>`__
     - AGPL-3.0-only
     - | checkpoint (optional): `yolo26s_obb_text_g1.pt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/yolo26s_obb_text_g1.pt>`_
       | weights (required): `yolo26s_obb_text_g1.raw.onnx <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/yolo26s_obb_text_g1.raw.onnx>`_
       | config (required): `yolo26s_obb_text_g1.raw.json <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/yolo26s_obb_text_g1.raw.json>`_
       | license (required): `yolo26s_obb_text_g1.LICENSE.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/yolo26s_obb_text_g1.LICENSE.txt>`__
   * - ``yolo26x_obb_text_g1``
     - `YOLO26 OBB <https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/26/yolo26-obb.yaml>`__
     - AGPL-3.0-only
     - | checkpoint (optional): `yolo26x_obb_text_g1.pt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/yolo26x_obb_text_g1.pt>`_
       | weights (required): `yolo26x_obb_text_g1.raw.onnx <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/yolo26x_obb_text_g1.raw.onnx>`_
       | config (required): `yolo26x_obb_text_g1.raw.json <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/yolo26x_obb_text_g1.raw.json>`_
       | license (required): `yolo26x_obb_text_g1.LICENSE.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/yolo26x_obb_text_g1.LICENSE.txt>`__

.. code-block:: python

    from manuscript.detectors import EAST

    detector = EAST(weights="east_50_g1")

Layout
------

.. list-table::
   :class: model-zoo-table
   :header-rows: 1
   :widths: 25 12 18 45

   * - Model key
     - Class
     - License
     - Artifacts
   * - ``SimpleSorting``
     - Algorithmic
     - MIT
     - None
   * - ``ReadingSorting``
     - Algorithmic
     - MIT
     - None

Layouts are included with the library and are created directly, without weights.

.. code-block:: python

    from manuscript.layouts import SimpleSorting

    layout = SimpleSorting()

Recognition
-----------

.. list-table::
   :class: model-zoo-table
   :header-rows: 1
   :widths: 25 12 18 45

   * - Model key
     - Class
     - License
     - Artifacts
   * - ``trba_base_g0``
     - `TRBA <https://arxiv.org/abs/1904.01906>`__
     - MIT
     - | **Training checkpoint only (for fine-tuning; no ONNX inference).**
       | checkpoint (optional): `trba_base_g0.pth <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_base_g0.pth>`_
       | license (required): `trba_base_g0.LICENSE.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_base_g0.LICENSE.txt>`__
   * - ``trba_base_g1``
     - `TRBA <https://arxiv.org/abs/1904.01906>`__
     - MIT
     - | config (required): `trba_base_g1.json <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_base_g1.json>`_
       | weights (required): `trba_base_g1.onnx <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_base_g1.onnx>`_
       | checkpoint (optional): `trba_base_g1.pth <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_base_g1.pth>`_
       | charset (required): `trba_base_g1.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_base_g1.txt>`_
       | license (required): `trba_base_g1.LICENSE.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_base_g1.LICENSE.txt>`__
   * - ``trba_lite_g1``
     - `TRBA <https://arxiv.org/abs/1904.01906>`__
     - MIT
     - | config (required): `trba_lite_g1.json <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g1.json>`_
       | weights (required): `trba_lite_g1.onnx <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g1.onnx>`_
       | checkpoint (optional): `trba_lite_g1.pth <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g1.pth>`_
       | charset (required): `trba_lite_g1.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g1.txt>`_
       | license (required): `trba_lite_g1.LICENSE.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g1.LICENSE.txt>`__
   * - ``trba_lite_g2``
     - `TRBA <https://arxiv.org/abs/1904.01906>`__
     - MIT
     - | config (required): `trba_lite_g2.json <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g2.json>`_
       | weights (required): `trba_lite_g2.onnx <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g2.onnx>`_
       | checkpoint (optional): `trba_lite_g2.pth <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g2.pth>`_
       | charset (required): `trba_lite_g2.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g2.txt>`_
       | license (required): `trba_lite_g2.LICENSE.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g2.LICENSE.txt>`__
   * - ``trba_lite_g3_prt``
     - `TRBA <https://arxiv.org/abs/1904.01906>`__
     - MIT
     - | config (required): `trba_lite_g3_prt.json <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g3_prt.json>`_
       | weights (required): `trba_lite_g3_prt.onnx <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g3_prt.onnx>`_
       | checkpoint (optional): `trba_lite_g3_prt.pth <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g3_prt.pth>`_
       | charset (required): `trba_lite_g3_prt.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g3_prt.txt>`_
       | license (required): `trba_lite_g3_prt.LICENSE.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/trba_lite_g3_prt.LICENSE.txt>`__

.. code-block:: python

    from manuscript.recognizers import TRBA

    recognizer = TRBA(weights="trba_lite_g2")

Correction
----------

.. list-table::
   :class: model-zoo-table
   :header-rows: 1
   :widths: 25 12 18 45

   * - Model key
     - Class
     - License
     - Artifacts
   * - ``modern_charlm_g1``
     - CharLM
     - MIT
     - | vocab (required): `modern_charlm_g1.json <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/modern_charlm_g1.json>`_
       | weights (required): `modern_charlm_g1.onnx <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/modern_charlm_g1.onnx>`_
       | checkpoint (optional): `modern_charlm_g1.pt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/modern_charlm_g1.pt>`_
       | lexicon (optional): `modern_words.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/modern_words.txt>`_
       | license (required): `modern_charlm_g1.LICENSE.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/modern_charlm_g1.LICENSE.txt>`__
   * - ``prereform_charlm_g1``
     - CharLM
     - MIT
     - | vocab (required): `prereform_charlm_g1.json <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/prereform_charlm_g1.json>`_
       | weights (required): `prereform_charlm_g1.onnx <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/prereform_charlm_g1.onnx>`_
       | checkpoint (optional): `prereform_charlm_g1.pt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/prereform_charlm_g1.pt>`_
       | lexicon (optional): `prereform_words.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/prereform_words.txt>`_
       | license (required): `prereform_charlm_g1.LICENSE.txt <https://github.com/konstantinkozhin/manuscript-ocr/releases/download/v0.1.0/prereform_charlm_g1.LICENSE.txt>`__

.. code-block:: python

    from manuscript.correctors import CharLM

    corrector = CharLM(weights="prereform_charlm_g1")
