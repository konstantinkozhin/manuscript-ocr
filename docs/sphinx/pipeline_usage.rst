Pipeline Usage Guide
====================

A pipeline passes a document-tree ``Page`` between detection, layout,
recognition and optional correction. See :doc:`api/data` for the data model
and :doc:`api/pipeline` for method signatures.

Quick start
-----------

.. code-block:: python

    from manuscript import Pipeline

    pipeline = Pipeline()
    result = pipeline.predict("document.jpg")
    print(result["page"].get_text())

Choosing stages
---------------

Supply initialized processors to replace the defaults. Set an optional stage
to None to disable it; detection cannot be disabled.

.. code-block:: python

    from manuscript.detectors import EAST
    from manuscript.layouts import SimpleSorting
    from manuscript.recognizers import TRBA
    from manuscript.correctors import CharLM

    pipeline = Pipeline(
        detector=EAST(weights="east_50_g1"),
        layout=SimpleSorting(use_columns=False),
        recognizer=TRBA(weights="trba_lite_g2", recognition_levels=["word"]),
        corrector=CharLM(),
    )

For detection only, use ``Pipeline(layout=None, recognizer=None)``. For detection
and recognition, use ``Pipeline(layout=None)``. Device and model parameters are
configured on each processor rather than on Pipeline.

``layout_after`` selects the layout slot: detector (default), recognizer or
corrector. The slot executes even if its anchor stage is disabled.

.. code-block:: python

    pipeline = Pipeline(layout_after="recognizer", corrector=None)

Combining processors
--------------------

Stack
~~~~~

``Stack`` runs processors in list order. Detectors each receive the original
image, and their regions are appended to a common Page. A layout can then infer
containment. Duplicate regions are retained.

.. code-block:: python

    from manuscript import Stack
    from manuscript.layouts import ReadingSorting

    # block_detector and line_detector are initialized detector instances.
    pipeline = Pipeline(
        detector=Stack([block_detector, line_detector]),
        layout=ReadingSorting(mode="containment", containment_overlap=0.5),
        recognizer=None,
    )

Page processors receive the preceding Page. Later processors may overwrite text;
built-in recognizers skip regions already containing text. Sequential behavior
therefore depends on the participating processors.

Recognizer voting
~~~~~~~~~~~~~~~~~

Use an ensemble when recognizers should independently read the same regions:

.. code-block:: python

    from manuscript.recognizers import MajorityVotingRecognizer, MedoidRecognizer

    # model_a, model_b and model_c are initialized recognizers.
    recognizer = MajorityVotingRecognizer([model_a, model_b, model_c])
    pipeline = Pipeline(recognizer=recognizer)

    medoid = MedoidRecognizer([model_a, model_b, model_c], parallel=False)

Majority voting selects the most frequent normalized string; medoid selects
an existing answer by summed character edit distance.

Custom stages
-------------

Custom components implement a compatible ``predict`` method; inheritance from
Manuscript base classes is optional. The detector receives an image; other
stages receive a ``Page``. Each stage returns a valid ``Page``; the detector may
also return a dictionary containing it under ``page``.

See :ref:`example-notebooks` for integration examples.

Recognition and correction levels
---------------------------------

The document tree stores text in word, line and block regions. Recognizers and
correctors process selected levels of this tree: ``recognition_levels`` selects
regions to recognize, and ``correction_levels`` selects text to correct. Both
accept ``word``, ``line`` and ``block``. For example, to select words and lines
for recognition and words for correction:

.. code-block:: python

    from manuscript import Pipeline
    from manuscript.recognizers import TRBA
    from manuscript.correctors import CharLM

    pipeline = Pipeline(
        recognizer=TRBA(
            weights="trba_lite_g2", recognition_levels=["word", "line"],
        ),
        corrector=CharLM(correction_levels=["word"]),
    )

The selected regions must already exist in the Page. When both a line and its
words are selected, the words take precedence. Choose recognition levels that
match the model's training task. See :doc:`api/recognizers` and
:doc:`api/correctors` for details.

Results, visualization and timing
---------------------------------

``predict()`` returns ``{"page": page}``. With ``vis=True``, it returns
``(result, preview)``, where ``preview`` is a PIL image:

.. code-block:: python

    result, preview = pipeline.predict("document.jpg", vis=True, profile=True)
    print(result["page"].get_text())
    preview.save("preview.png")
    print(pipeline.last_timings)

``profile=True`` prints stage times; ``last_timings`` provides them in seconds.
Intermediate Pages are available through ``last_detection_page``,
``last_layout_page``, ``last_recognition_page`` and ``last_correction_page``.
See :doc:`api/pipeline` for details.
