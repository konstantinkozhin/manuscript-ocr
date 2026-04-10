Detectors
=========

Text detection models.

.. automodule:: manuscript.detectors
   :members:
   :undoc-members:
   :show-inheritance:

EAST Training Quads
-------------------

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

