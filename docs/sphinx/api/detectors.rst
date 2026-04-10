Detectors
=========

Text detection models.

.. automodule:: manuscript.detectors
   :members:
   :undoc-members:
   :show-inheritance:

EAST
----

Детектор ``EAST`` в manuscript-ocr основан на архитектуре, предложенной в
`EAST: An Efficient and Accurate Scene Text Detector <https://openaccess.thecvf.com/content_cvpr_2017/papers/Zhou_EAST_An_Efficient_CVPR_2017_paper.pdf>`_
(Zhou et al., CVPR 2017). Процедура обучения существенно переработана по сравнению с оригиналом:
изменены схема взвешивания потерь, пайплайн аугментаций, обработка четырёхугольных разметок и
поддержка смешанных аннотаций. Предобученные веса получены авторами проекта.

EAST Training Quads
~~~~~~~~~~~~~~~~~~~

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

