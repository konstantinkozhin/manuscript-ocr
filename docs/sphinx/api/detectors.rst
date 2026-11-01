Detectors
=========

Supported models:

- :class:`~manuscript.detectors.YOLO`
- :class:`~manuscript.detectors.EAST`
- :class:`~manuscript.detectors.ConsensusWBFDetector`
- :class:`~manuscript.detectors._ppocr.PPOCR`
- :class:`~manuscript.detectors._rfdetr.RFDETR`
- :class:`~manuscript.detectors._mask2former.Mask2Former`

ConsensusWBFDetector
--------------------

.. autoclass:: manuscript.detectors.ConsensusWBFDetector
   :members: predict, predict_batch, required_votes
   :show-inheritance:

YOLO
----

.. autoclass:: manuscript.detectors.YOLO
   :members: predict, create_config
   :show-inheritance:

EAST
----

.. autoclass:: manuscript.detectors.EAST
   :members: predict, train, export
   :show-inheritance:

PPOCR
-----

.. autoclass:: manuscript.detectors._ppocr.PPOCR
   :members: predict, create_config
   :show-inheritance:

RFDETR
------

.. autoclass:: manuscript.detectors._rfdetr.RFDETR
   :members: predict, create_config
   :show-inheritance:

Mask2Former
-----------

.. autoclass:: manuscript.detectors._mask2former.Mask2Former
   :members: predict, create_config
   :show-inheritance:
