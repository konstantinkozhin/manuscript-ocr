Detectors
=========

Text detection models.

All detectors return a ``Page`` by default. Use ``predict(image, return_raw=True)``
for compact decoded results:

* ``page``: the standard OCR page result.
* ``detections``: regions in source-image coordinates, including available
  classes, oriented boxes, contours and contour hierarchy.
* ``image_size``: source image height and width.
* ``metadata``: model-specific preprocessing information and class names.

Heavy data is opt-in:

* ``return_masks=True`` includes available binary masks in source-image
  coordinates. Instance masks are attached to detections; PPOCR exposes its
  page-level text mask in ``masks["text"]``. EAST/YOLO do not generate masks.
* ``return_outputs=True`` adds ``outputs``: original named ONNX tensors in
  model coordinates, including mask logits if emitted by the graph.

Either additional flag enables dictionary output, even without ``return_raw``.
Masks and original tensors are independent options. To obtain the previous
full raw result, set both flags. NumPy arrays are returned by reference where
possible and are not directly JSON serializable. Native polygon extraction
still uses model masks regardless of the output flags.

Pipeline requests compact results and passes them to compatible layout,
recognizer and corrector stages as ``detector_result``. Stages without this
argument retain the existing Page interface. Detector details are local to one
pipeline call; ``last_*`` snapshots contain only Page objects.

EAST and YOLO preserve quadrilateral and oriented-box geometry by default.
Their ``axis_aligned_output`` argument is deprecated. Explicitly passing ``True``
still provides the legacy rectangle output and emits ``DeprecationWarning``;
raw detections retain native geometry. Recognition crop selection belongs to
the recognizer's region preparer.

.. autoclass:: manuscript.detectors.EAST
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: manuscript.detectors.YOLO
   :members:
   :undoc-members:
   :show-inheritance:

.. rubric:: EAST: Training Notes

The ``EAST`` detector in manuscript-ocr is based on the architecture proposed in
`EAST: An Efficient and Accurate Scene Text Detector <https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_EAST_An_Efficient_CVPR_2017_paper.pdf>`_
(Zhou et al., CVPR 2017). The training procedure has been significantly reworked
compared to the original: the loss weighting scheme, augmentation pipeline,
quadrilateral annotation handling, and support for mixed annotations have all
been modified. Pretrained weights were produced by the project authors.

.. rubric:: EAST Training Quads

EAST training expects quadrilateral targets. When loading COCO
``segmentation`` polygons, use ``augmentation_config["quad_source"]`` in
``EAST.train(...)`` to control how polygons are converted into 4-point
training quads:

- ``"auto"`` keeps existing 4-point polygons as-is and falls back to
  ``minAreaRect`` for longer polygons.
- ``"as_is"`` accepts only 4-point polygons and skips polygons with a
  different number of vertices.
- ``"min_area_rect"`` always fits the minimum-area rectangle and matches
  the legacy conversion path.

