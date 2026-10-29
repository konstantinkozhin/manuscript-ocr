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
Both compare text after NFC normalization, lowercasing and trimming outer
whitespace. The returned text keeps the most frequent case-sensitive spelling
among answers matching the winning normalized string; spelling ties prefer the
first model. For example, ``Москва``, ``Москва``, ``москва`` produce ``Москва``.
With ``return_raw=True``, ``winner`` contains this spelling, while
``normalized_winner`` identifies its key in ``votes`` (and ``distance_sums``
for medoid).

Detector consensus
~~~~~~~~~~~~~~~~~~

Use ``ConsensusWBFDetector`` to retain detections supported by multiple models:

.. code-block:: python

    from manuscript.detectors import ConsensusWBFDetector

    detector = ConsensusWBFDetector(
        detectors=[detector_a, detector_b, detector_c],
        weights=None,  # Equal coordinate weights; alternatively [1, 2, 1].
        cluster_iou=0.5,
        consensus_fraction=0.5,
        minimum_votes=2,
    )
    pipeline = Pipeline(detector=detector)
    raw = detector.predict("page.jpg", return_raw=True)
    votes = [item["votes"] for item in raw["detections"]]

Required votes are ``max(minimum_votes, ceil(len(detectors) * consensus_fraction))``.
Empty answers count toward the total number of participants; inference errors
propagate. Each participant contributes at most one vote per cluster. Regions
are matched separately by ``(type, label)``; configure matching semantic labels
on participants. Polygons become axis-aligned enclosing boxes. Participant Pages
must already use original-image coordinates and dimensions, as built-in
detectors do. No extra resize or NMS is performed by the ensemble.

Participants run in supplied order. Greedy one-to-one matching uses representatives
frozen before each participant; equal IoUs prefer higher object and cluster indices.
The algorithm is order-dependent. The representative and final rectangle use
arithmetic means by default, or weighted means with one finite positive weight
per participant. Weights affect geometry and matching, never vote thresholds.
Confidence scores are ignored, and fused ``detection_confidence`` is ``None``.

The output is a flat Page; geometric descendants of participant Pages are
treated as detections. Raw detections include ``votes`` and ``detector_indices``
referencing the supplied order. Fusion does not produce masks or model tensors.
``predict_batch(images)`` returns independent results in image order, including
an empty list for an empty batch. It calls single-image prediction sequentially.
Pickle preserves parameters and nested participants when those participants
support pickling; no JSON configuration protocol for arbitrary models is assumed.

Model memory
------------

Use ``memory_policy="sequential"`` to release each model's sessions after use
and reload them when needed. The default, ``"keep"``, retains loaded sessions.

.. code-block:: python

    recognizer = MedoidRecognizer([model_a, model_b, model_c])
    pipeline = Pipeline(recognizer=recognizer, memory_policy="sequential")

The policy applies inside nested Stacks and ensembles. It can also be set on
an individual Stack, MajorityVotingRecognizer or MedoidRecognizer. Sequential
mode releases previously loaded participant sessions before starting and
cleans up the active model even if prediction fails. ``close()`` releases all
participants explicitly; the composition can be used again afterwards.

This mode requires ``parallel=False`` throughout the controlled composition.
Custom processors must implement an idempotent ``close()`` and support another
prediction after closing; stateless processors can use a no-op. Participants
must not be used concurrently by other pipelines
or direct calls without external synchronization. Calls on the same sequential
composition are serialized.

Image data, voting results, dictionaries and runtime allocations can still
occupy memory. Session release does not guarantee an immediate drop in process
RSS or GPU memory reported by the driver. Repeated session initialization
trades inference speed for lower model residency.

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
