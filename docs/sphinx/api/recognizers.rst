Recognizers
===========

Supported recognizers:

- :class:`~manuscript.recognizers.TRBA`
- :class:`~manuscript.recognizers.PPOCRRec`
- :class:`~manuscript.recognizers.TrOCR`
- :class:`~manuscript.recognizers.MajorityVotingRecognizer`
- :class:`~manuscript.recognizers.MedoidRecognizer`

.. code-block:: python

    from manuscript.recognizers import TRBA, PPOCRRec, TrOCR
    from manuscript.utils import create_page_from_image

    recognizer = TRBA(weights="trba_lite_g2")
    page, image = create_page_from_image("word.png", return_image=True)
    result = recognizer.predict(page, image=image)
    print(result.get_text())

TRBA
----

.. autoclass:: manuscript.recognizers.TRBA
   :members: train, export
   :show-inheritance:

.. automethod:: manuscript.recognizers.TRBA.predict

PPOCRRec
--------

.. autoclass:: manuscript.recognizers.PPOCRRec
   :members: create_config
   :show-inheritance:

.. automethod:: manuscript.recognizers.PPOCRRec.predict

TrOCR
-----

.. autoclass:: manuscript.recognizers.TrOCR
   :members: create_config
   :show-inheritance:

.. automethod:: manuscript.recognizers.TrOCR.predict

MajorityVotingRecognizer
------------------------

.. autoclass:: manuscript.recognizers.MajorityVotingRecognizer
   :members: predict
   :show-inheritance:

MedoidRecognizer
----------------

.. autoclass:: manuscript.recognizers.MedoidRecognizer
   :no-special-members:
   :show-inheritance:

.. automethod:: manuscript.recognizers.MedoidRecognizer.predict
