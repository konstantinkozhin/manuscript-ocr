Pipeline Usage Guide
====================

The pipeline is stage-based and passes one ``Page`` object through the stages:

``detector -> layout -> recognizer -> corrector``

By default, ``Pipeline()`` creates:

- detector: ``YOLO(weights="yolo26x_obb_text_g1")``
- layout: ``SimpleSorting()``
- recognizer: ``TRBA(weights="trba_lite_g2")``
- corrector: ``None``

Stage Contracts
---------------

Detector
~~~~~~~~

Detector must implement:

.. code-block:: python

    def predict(self, image) -> Page:
        ...

Recognizer
~~~~~~~~~~

Recognizer must implement:

.. code-block:: python

    def predict(self, page: Page, image: Optional[np.ndarray] = None) -> Page:
        ...

Layout
~~~~~~

Layout model must implement:

.. code-block:: python

    def predict(self, page: Page, image: Optional[np.ndarray] = None) -> Page:
        ...

Corrector
~~~~~~~~~

Corrector must implement:

.. code-block:: python

    def predict(self, page: Page, image: Optional[np.ndarray] = None) -> Page:
        ...

Basic Usage
-----------

.. code-block:: python

    from manuscript import Pipeline

    pipeline = Pipeline()
    result = pipeline.predict("document.jpg")
    text = pipeline.get_text(result["page"])
    print(text)

Compose Processors with Stack
-----------------------------

``Stack`` runs processors in order and can occupy any pipeline slot:

.. code-block:: python

    from manuscript import Pipeline, Stack

    pipeline = Pipeline(
        detector=Stack([block_detector, line_detector]),
        recognizer=Stack([word_recognizer, line_recognizer]),
        corrector=Stack([corrector_a, corrector_b]),
    )

Image processors implement ``predict(image, ...)``. Page processors declare
their first parameter as ``page`` (or annotate it as ``Page``), optionally
accepting ``image``. Stack selects the input automatically. Each page processor
receives the preceding Page result. Member defaults such as recognition batch
size are preserved unless explicitly overridden in ``Stack.predict``.

Detectors run on the whole image and their regions are appended to the current
Page. Stack does not deduplicate regions or infer parents; a layout processor
can build containment and reading order afterwards. Recognition is sequential:
later processors may overwrite text on the same regions; there is no voting.

Stacks can also be nested or called directly:

.. code-block:: python

    page = Stack([detector, layout, recognizer]).predict(image)
    page = Stack([corrector_a, corrector_b]).predict(page)

``Stack([detector_a, detector_b], parallel=True)`` runs independent image
detectors concurrently and adds results in list order. This option rejects
page processors, including those inside nested stacks. It defaults to False.
Recognition voting uses ``MajorityVotingRecognizer`` instead of sequential Stack:

.. code-block:: python

    from manuscript.recognizers import MajorityVotingRecognizer

    pipeline = Pipeline(
        detector=Stack([detector_a, detector_b], parallel=True),
        recognizer=MajorityVotingRecognizer([model_a, model_b], parallel=False),
    )

``return_raw=True`` returns the final ``page``, combined ``detections``,
``image_size``, and model-specific dictionaries under ``detector_results``.
Masks and original outputs are requested only with ``return_masks=True`` and
``return_outputs=True``. Compatible page processors receive the combined
dictionary as ``detector_result``.

Disable Stages
--------------

You can disable optional stages via ``None``:

.. code-block:: python

    from manuscript import Pipeline

    # Detection + layout only
    pipeline = Pipeline(recognizer=None, corrector=None)

    # Detection + recognition only
    pipeline = Pipeline(layout=None, corrector=None)

    # Detection + layout + recognition (no correction)
    pipeline = Pipeline(corrector=None)

Layout Placement
----------------

Use ``layout_after`` to choose where layout runs:

- ``"detector"`` (default)
- ``"recognizer"``
- ``"corrector"``

.. code-block:: python

    from manuscript import Pipeline
    from manuscript.layouts import SimpleSorting

    pipeline = Pipeline(
        layout=SimpleSorting(),
        layout_after="recognizer",
    )

If the anchor stage is disabled (for example, ``recognizer=None`` with
``layout_after="recognizer"``), layout still executes in that slot.

Built-in Components
-------------------

.. code-block:: python

    from manuscript.detectors import EAST
    from manuscript.layouts import SimpleSorting
    from manuscript.recognizers import TRBA
    from manuscript.correctors import CharLM
    from manuscript import Pipeline

    detector = EAST(weights="east_50_g1", score_thresh=0.8, iou_threshold=0.2)
    layout = SimpleSorting(max_splits=10, use_columns=True)
    recognizer = TRBA(weights="trba_lite_g1", device="cuda", min_text_size=5)
    corrector = CharLM()

    pipeline = Pipeline(
        detector=detector,
        layout=layout,
        recognizer=recognizer,
        corrector=corrector,
        layout_after="detector",
    )

TRBA Region Preparation
-----------------------

``TRBA`` supports configurable crop preparation before recognition.

Default settings:

- ``region_preparer="bbox"`` extracts axis-aligned bounding boxes
- ``rotate_threshold=1.5`` auto-rotates tall crops before recognition
- ``min_text_size=5`` skips tiny detections

Built-in preparer presets:

- ``"bbox"``: axis-aligned crop
- ``"polygon_mask"``: tight crop with pixels outside the polygon masked to white
- ``"quad_warp"``: perspective rectification for 4-point polygons, with bbox fallback

.. code-block:: python

    from manuscript.recognizers import TRBA

    recognizer = TRBA(region_preparer="bbox")
    recognizer = TRBA(region_preparer="polygon_mask")
    recognizer = TRBA(region_preparer="quad_warp")
    recognizer = TRBA(
        region_preparer="bbox",
        region_preparer_options={"pad": 2},
    )

``region_preparer_options`` is reserved for built-in preset configuration:

- ``"bbox"`` / ``"polygon_mask"``: ``pad``
- ``"polygon_mask"``: ``background``
- ``"quad_warp"``: ``output_size=(width, height)``, ``fallback_to_bbox``

For advanced cases, you can inject hooks into ``TRBA`` instead of writing a
full custom recognizer:

.. code-block:: python

    import numpy as np

    def my_preparer(page, image, recognizer=None, options=None):
        regions = []
        for block in page.blocks:
            for line in block.lines:
                for text_span in line.text_spans:
                    poly = np.asarray(text_span.polygon, dtype=np.float32)
                    crop = image[10:40, 10:80]
                    regions.append(
                        {"text_span": text_span, "image": crop, "polygon": poly}
                    )
        return regions

    recognizer = TRBA(region_preparer=my_preparer)

If you need complete control over recognition logic, the simplest route is
still to provide your own recognizer class with ``predict(page, image) -> Page``.

Collapsing Text Spans
---------------------

Some recognizers work on whole lines or blocks rather than on individual
text-span crops. Use ``collapse_page_text_spans`` to convert a narrow page
structure into a wider one before recognition.

.. code-block:: python

    from manuscript.utils import collapse_page_text_spans

    line_level_page = collapse_page_text_spans(
        page,
        level="line",
        method="bbox",
    )

    block_level_page = collapse_page_text_spans(
        page,
        level="block",
        method="convex_hull",
    )

``"line"`` keeps the same blocks and lines, but replaces each line with one
merged ``TextSpan``. ``"block"`` replaces each block with one line
containing one merged ``TextSpan``.

Lower-level helpers are also available:

- ``merge_text_spans(text_spans, method="bbox")``
- ``collapse_line_text_spans(line, method="bbox")``
- ``collapse_block_text_spans(block, method="bbox")``

Visualization and Profiling
---------------------------

.. code-block:: python

    result, vis_img = pipeline.predict("document.jpg", vis=True)
    vis_img.save("output_visualization.jpg")

    result = pipeline.predict("document.jpg", profile=True)

Intermediate Results
--------------------

After each run, the pipeline keeps snapshots:

- ``pipeline.last_detection_page``
- ``pipeline.last_layout_page``
- ``pipeline.last_recognition_page``
- ``pipeline.last_correction_page``

Skipped stages keep corresponding ``last_*`` value as ``None``.
