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
