Pipeline Usage Guide
=====================

The Pipeline class in ``manuscript-ocr`` is designed to work with **any** detectors, recognizers, and correctors that implement a simple interface.

Detector Requirements
---------------------

A detector class must implement a ``predict`` method that takes an image and returns a dictionary with a ``"page"`` key:

.. code-block:: python

    def predict(self, image) -> Dict[str, Any]:
        """
        Parameters:
        - image: file path (str) or numpy array (H, W, 3) in uint8
        
        Returns dictionary:
        {
            "page": Page  # Page object with detection results
        }
        """
        pass

Result Structure
~~~~~~~~~~~~~~~~

The result must contain a ``Page`` object with hierarchy:  
**Page** → **Block** → **Line** → **Word**

See ``src/manuscript/data/structures.py`` for detailed structure documentation.

**Minimal example of creating a Page:**

.. code-block:: python

    from manuscript.data import Word, Line, Block, Page

    # Create a word with coordinates and detection confidence
    word = Word(
        polygon=[(10, 20), (100, 20), (100, 40), (10, 40)],
        detection_confidence=0.95
    )

    # Group words into a line
    line = Line(words=[word])

    # Group lines into a block
    block = Block(lines=[line])

    # Create a page
    page = Page(blocks=[block])

Recognizer Requirements
-----------------------

A recognizer class must implement a ``predict`` method that takes a ``Page`` object and optionally the source image, then returns updated ``Page``:

.. code-block:: python

    def predict(self, page: Page, image: Optional[np.ndarray] = None) -> Page:
        """
        Parameters:
        - page: Page object with detected words
        - image: optional source image (for crop extraction / context)
        
        Returns:
        - Page: Page object where recognizer fills word.text and
          word.recognition_confidence
        """
        pass

**Example:**

.. code-block:: python

    from typing import Optional
    import numpy as np
    from manuscript.data import Page

    class MyRecognizer:
        def predict(self, page: Page, image: Optional[np.ndarray] = None) -> Page:
            result = page.model_copy(deep=True)
            for block in result.blocks:
                for line in block.lines:
                    for word in line.words:
                        # Your recognition logic
                        word.text = "recognized_text"
                        word.recognition_confidence = 0.92
            return result

Corrector Requirements
----------------------

A corrector class must implement a ``predict`` method that takes a ``Page`` object and optionally the source image, then returns a corrected ``Page``:

.. code-block:: python

    def predict(self, page: Page, image: Optional[np.ndarray] = None) -> Page:
        """
        Parameters:
        - page: Page object with recognized text
        - image: optional source image (for context-aware correction)
        
        Returns:
        - Page: Page object with corrected text
        """
        pass

**Example:**

.. code-block:: python

    from typing import Optional
    import numpy as np
    from manuscript.data import Page

    class MyCorrector:
        def predict(self, page: Page, image: Optional[np.ndarray] = None) -> Page:
            result = page.model_copy(deep=True)
            for block in result.blocks:
                for line in block.lines:
                    for word in line.words:
                        if word.text:
                            # Your correction logic
                            word.text = self._correct(word.text)
            return result
        
        def _correct(self, text: str) -> str:
            # Text correction logic
            return text

Built-in CharLM Corrector
~~~~~~~~~~~~~~~~~~~~~~~~~~

CharLM is a Transformer-based character-level language model for correcting OCR errors:

.. code-block:: python

    from manuscript.correctors import CharLM

    # With default settings
    corrector = CharLM()

    # With custom parameters
    corrector = CharLM(
        weights="prereform_charlm_g1",  # or "modern_charlm_g1"
        mask_threshold=0.05,            # confidence threshold for correction
        apply_threshold=0.95,           # minimum model confidence
        max_edits=2,                    # max edits per word
        min_word_len=4,                 # min word length for correction
        lexicon="prereform_words"       # lexicon of known words
    )

Compatible Implementation Examples
----------------------------------

Complete Detector Example
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from manuscript.data import Word, Line, Block, Page

    class MyDetector:
        def predict(self, image):
            # Your image detection logic
            # ...
            
            # Create result
            words = [
                Word(
                    polygon=[(10, 20), (100, 20), (100, 40), (10, 40)],
                    detection_confidence=0.95
                ),
                Word(
                    polygon=[(110, 20), (200, 20), (200, 40), (110, 40)],
                    detection_confidence=0.92
                ),
            ]
            
            line = Line(words=words)
            block = Block(lines=[line])
            page = Page(blocks=[block])
            
            return {"page": page}

Using Custom Components
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from manuscript import Pipeline
    from my_package import MyDetector, MyRecognizer, MyCorrector

    # Use custom detector and recognizer
    detector = MyDetector()
    recognizer = MyRecognizer()
    corrector = MyCorrector()
    
    pipeline = Pipeline(
        detector=detector,
        recognizer=recognizer,
        corrector=corrector
    )
    
    result = pipeline.predict("document.jpg")

Pipeline Usage Examples
-----------------------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from manuscript import Pipeline

    # Initialize with default models
    pipeline = Pipeline()

    # Process image
    result = pipeline.predict("document.jpg")
    page = result["page"]

    # Extract text
    text = pipeline.get_text(page)
    print(text)

Detection Only (Without Recognition)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    result = pipeline.predict("document.jpg", recognize_text=False)
    page = result["page"]

    # Words have polygon and detection_confidence, but no text
    for block in page.blocks:
        for line in block.lines:
            for word in line.words:
                print(f"Polygon: {word.polygon}, Confidence: {word.detection_confidence}")

With Visualization
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    result, vis_img = pipeline.predict("document.jpg", vis=True)
    vis_img.save("output_visualization.jpg")

Intermediate Results
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from manuscript.correctors import CharLM
    
    pipeline = Pipeline(corrector=CharLM())
    result = pipeline.predict("document.jpg")

    # Result after detection (before recognition)
    detection_page = pipeline.last_detection_page

    # Result after recognition (before correction)
    recognition_page = pipeline.last_recognition_page

    # Result after correction (None if corrector not used)
    correction_page = pipeline.last_correction_page

Export/Import Page to JSON
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    page = result["page"]

    # Save to file
    page.to_json("result.json")

    # Get as string
    json_str = page.to_json()

    # Load from file
    from manuscript.data import Page
    page = Page.from_json("result.json")

    # Load from string
    page = Page.from_json('{"blocks": [...]}')

With Profiling
~~~~~~~~~~~~~~

.. code-block:: python

    # Prints execution time for each stage
    result = pipeline.predict("document.jpg", profile=True)
    # Output:
    # Detection: 0.123s
    # Load image for crops: 0.005s
    # Extract 45 crops: 0.012s
    # Recognition: 0.234s
    # Pipeline total: 0.374s

Batch Processing
~~~~~~~~~~~~~~~~

.. code-block:: python

    images = ["page1.jpg", "page2.jpg", "page3.jpg"]
    results = pipeline.process_batch(images)

    for result in results:
        text = pipeline.get_text(result["page"])
        print(text)

Component Configuration
-----------------------

Replacing Detector or Recognizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from manuscript import Pipeline

    # Only custom detector, default recognizer
    from my_package import MyCustomDetector
    pipeline = Pipeline(detector=MyCustomDetector())

    # Only custom recognizer, default detector
    from my_package import MyCustomRecognizer
    pipeline = Pipeline(recognizer=MyCustomRecognizer())

    # Both components custom
    pipeline = Pipeline(detector=MyCustomDetector(), recognizer=MyCustomRecognizer())

Built-in Model Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from manuscript import Pipeline
    from manuscript.detectors import EAST
    from manuscript.recognizers import TRBA

    # EAST with settings
    detector = EAST(
        weights="east_50_g1",        # weight selection
        score_thresh=0.8,            # confidence threshold
        iou_threshold=0.2,           # IoU threshold for NMS
        device="cpu"                 # device (cpu/cuda)
    )

    # TRBA with settings
    recognizer = TRBA(
        weights="trba_lite_g1",      # weight selection
        device="cuda"                # GPU for acceleration
    )

    pipeline = Pipeline(detector, recognizer)

Size Filtering
~~~~~~~~~~~~~~

.. code-block:: python

    # Ignore text blocks smaller than 10 pixels
    pipeline = Pipeline(min_text_size=10)

Automatic Rotation Control (TRBA)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from manuscript.recognizers import TRBA

    # Enable automatic rotation of vertical text (default)
    recognizer = TRBA(rotate_threshold=1.5)
    pipeline = Pipeline(recognizer=recognizer)

    # Disable automatic rotation
    recognizer = TRBA(rotate_threshold=0)
    pipeline = Pipeline(recognizer=recognizer)
