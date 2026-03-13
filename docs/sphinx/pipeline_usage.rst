Pipeline Usage Guide
====================

The pipeline is stage-based and passes one ``Page`` object through the stages:

``detector -> layout -> recognizer -> corrector``

By default, ``Pipeline()`` creates:

- detector: ``EAST()``
- layout: ``SimpleSorting()``
- recognizer: ``TRBA()``
- corrector: ``None``

Stage Contracts
---------------

Detector
~~~~~~~~

Detector must implement:

.. code-block:: python

    def predict(self, image) -> Dict[str, Any]:
        return {"page": page}

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
    page = result["page"]
    text = pipeline.get_text(page)
    print(text)

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

``TRBA`` keeps the current default crop behavior, but now allows crop
preparation customization when needed.

Default behavior is unchanged:

- ``region_preparer="bbox"`` extracts axis-aligned bounding boxes
- ``rotate_threshold=1.5`` auto-rotates tall crops before recognition
- ``min_text_size=5`` skips tiny detections

Built-in preparer presets:

- ``"bbox"``: legacy axis-aligned crop behavior
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
                for word in line.words:
                    poly = np.asarray(word.polygon, dtype=np.float32)
                    crop = image[10:40, 10:80]
                    regions.append({"word": word, "image": crop, "polygon": poly})
        return regions

    recognizer = TRBA(region_preparer=my_preparer)

If you need complete control over recognition logic, the simplest route is
still to provide your own recognizer class with ``predict(page, image) -> Page``.

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
