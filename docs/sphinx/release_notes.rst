Release Notes
=============

0.1.13
------

For changes required in existing integrations, see :doc:`migration_0_1_13`.

- Replaced fixed blocks, lines and text spans with an extensible document tree using ``Element``, ``Page`` and ``TextRegion``.
- Added table, cell and image elements with grid validation and orientation metadata.
- Added single-image COCO serialization with Manuscript extensions and plain-text output from the tree.
- Added ``Stack`` for composing processors and running independent detectors in parallel.
- Added majority-voting and medoid ensembles with strict validation of recognizer output trees.
- Added ``ReadingSorting`` for reading order and geometric containment with configurable overlap thresholds.
- Updated ``SimpleSorting`` to group tree regions while retaining original objects and table contents.
- Added configurable recognition levels, crop preparation and orientation compensation.
- Added compatible PPOCRRec and TrOCR ONNX recognition, including TrOCR KV-cache decoding.
- Added compatible PPOCR, RFDETR and Mask2Former detector integrations.
- Added explicit detector class mappings and optional raw results, masks and original ONNX outputs.
- Preserved native detection polygons and deprecated axis-aligned output options on EAST and YOLO.
- Added a mirrored remote model registry with local caching, artifact verification and model-management helpers.
- Added per-stage pipeline timings and retained independent intermediate Page snapshots.
- Disabled correction by default and added explicit correction levels for CharLM.
- Fixed partial ONNX batches and preserved configured recognizer batch sizes in Pipeline.
- Improved Windows CUDA runtime loading and kept training dependencies out of ordinary inference imports.
- Added PARSeq, SEResNet31LiteV2 and optional TPS rectification to TRBA training and export.
- Improved checkpoint saving, training resume and final gradient-accumulation updates.
- Updated tree visualization, grouping, text collapsing and image-input utilities.
- Moved model export tools to repository scripts and updated the end-to-end demonstration interface.
- Added notebooks for custom stages, model composition, training and export.
- Expanded installation and test CI matrices through Python 3.14.

0.1.12
------

- Added configurable TRBA batch sizes with a default of 128 and per-call overrides.
- Updated Pipeline to respect the recognizer's configured batch size.
- Improved automatic CUDA DLL loading on Windows before ONNX session initialization.
- Added ``create_page_from_image()`` for wrapping one or more images in a Page.
- Added automatic ONNX export attempts after CharLM training.
- Added ``onnxscript`` and ``Levenshtein`` to development dependencies and removed ``ultralytics``.

0.1.11
------

- Introduced a consistent ``Page -> Page`` stage contract and base classes and protocols for custom processors.
- Refactored Pipeline around explicit stages and intermediate Page snapshots.
- Added the YOLO ONNX detector with preset and custom weights and axis-aligned and rotated detections.
- Set the default pipeline to YOLO with ``yolo26x_obb_text_g1``, SimpleSorting and TRBA with ``trba_lite_g2``.
- Added the ``manuscript.layouts`` package and SimpleSorting, with configurable layout placement through ``layout_after``.
- Updated the data model around ``TextSpan``, ``Line``, ``Block`` and ``Page``, retaining legacy ``Word`` aliases and versioned export schemas.
- Added support for polygons with more than four points and helpers for merging and collapsing text spans.
- Updated TRBA and CharLM to process Page objects with optional source-image context.
- Added ``bbox``, ``polygon_mask`` and ``quad_warp`` recognition crop presets and moved rotation and size filtering into TRBA.
- Improved TRBA training, export, validation, metrics and debug tools, and added the ``trba_lite_g2`` preset.
- Improved EAST polygon parsing, training augmentations, previews, checkpoint selection and resume behavior.
- Added HTTP model download retries with backoff.
- Fixed Python 3.8 compatibility and training edge cases, including ``val_size=0`` and ``pretrain_weights=None``.
- Updated API guides and example notebooks and introduced versioned documentation with archived 0.1.10 pages.

0.1.10
------

Reconstructed from published source distributions and Git history, relative to 0.1.8, the previous release on PyPI.

- Added TRBA recognition and an end-to-end Pipeline combining detection, recognition and optional text correction.
- Added the CharLM corrector with training, ONNX inference and export support.
- Introduced structured OCR results using ``Word``, ``Line``, ``Block`` and ``Page``, with JSON serialization.
- Added shared model loading and caching for presets, local files, HTTP URLs, GitHub releases and Google Drive.
- Reworked EAST inference around ONNX Runtime and added CPU, CUDA and CoreML provider selection for inference models.
- Separated inference dependencies from optional PyTorch training and development dependencies.
- Added configurable automatic rotation of vertical text crops before recognition.
- Added optional EAST inference augmentation, contained-box filtering and area-anomaly filtering.
- Improved EAST training resume and checkpoint recovery.
- Added document sorting, reading-order visualization, OCR metrics and text-to-Page utilities.
- Added a Gradio demonstration and expanded English and Russian API documentation.
