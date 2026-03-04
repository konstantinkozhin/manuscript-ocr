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
