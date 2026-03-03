Correctors
==========

Character-level text corrector for post-processing OCR results.

``CharLM``
----------

.. autoclass:: manuscript.correctors.CharLM
   :members:
   :undoc-members:
   :show-inheritance:

Overview
~~~~~~~~

``CharLM`` is a character-level masked language model corrector that uses a Transformer 
architecture to fix OCR errors. It analyzes character-level context and applies 
corrections based on learned patterns.

**Key features:**

- Character-level Transformer-based correction
- Configurable confidence thresholds
- Support for custom vocabularies and lexicons
- ONNX Runtime inference for fast correction
- Optional lexicon filtering to preserve known words

Available Presets
~~~~~~~~~~~~~~~~~

The following pretrained models are available:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Preset Name
     - Description
   * - ``prereform_charlm_g1``
     - Pre-reform Russian text (default)
   * - ``modern_charlm_g1``
     - Modern Russian text

Quick Example
~~~~~~~~~~~~~

Use ``create_page_from_text`` to quickly test correction on text:

.. code-block:: python

    from manuscript.utils import create_page_from_text
    from manuscript.correctors import CharLM

    # Create page from text with potential OCR errors
    page = create_page_from_text(["Привѣтъ міръ", "Тестовая строка"])

    # Apply correction (using default prereform model)
    corrector = CharLM()
    corrected = corrector.predict(page)

    # Extract corrected text
    for line in corrected.blocks[0].lines:
        text = " ".join(w.text for w in line.words)
        print(text)

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from manuscript import Pipeline
    from manuscript.correctors import CharLM

    # Create corrector with default preset
    corrector = CharLM()

    # Create corrector with specific preset
    corrector = CharLM(weights="modern_charlm_g1")

    # Use in pipeline
    pipeline = Pipeline(corrector=corrector)
    result = pipeline.predict("document.jpg")

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from manuscript.correctors import CharLM

    # Fine-tune correction behavior
    corrector = CharLM(
        weights="prereform_charlm_g1",
        mask_threshold=0.05,      # Characters with confidence below this are corrected
        apply_threshold=0.95,     # Model must be this confident to apply correction
        max_edits=2,              # Maximum edits per word
        min_word_len=4,           # Minimum word length to attempt correction
        device="cuda"             # Use GPU for inference
    )

Using Custom Lexicon
~~~~~~~~~~~~~~~~~~~~

You can provide a lexicon (word list) to prevent corrections of known words:

.. code-block:: python

    from manuscript.correctors import CharLM

    # From preset
    corrector = CharLM(
        weights="prereform_charlm_g1",
        lexicon="prereform_words"  # Use preset lexicon
    )

    # From file
    corrector = CharLM(
        weights="prereform_charlm_g1",
        lexicon="path/to/words.txt"
    )

    # From set
    my_words = {"слово1", "слово2", "слово3"}
    corrector = CharLM(
        weights="prereform_charlm_g1",
        lexicon=my_words
    )

Training Custom Model
~~~~~~~~~~~~~~~~~~~~~

You can train CharLM on your own data:

.. code-block:: python

    from manuscript.correctors import CharLM

    # Train with OCR pairs dataset
    checkpoint_path = CharLM.train(
        pairs_path="ocr_pairs.csv",      # CSV with incorrect,correct columns
        charset_path="charset.txt",       # Allowed characters
        exp_dir="my_charlm_exp",
        epochs=50,
        batch_size=256,
    )

    # Train with word list (self-supervised)
    checkpoint_path = CharLM.train(
        words_path="words.txt",           # One word per line
        charset_path="charset.txt",
        exp_dir="my_charlm_exp",
    )

Export to ONNX
~~~~~~~~~~~~~~

.. code-block:: python

    from manuscript.correctors import CharLM

    # Export trained model to ONNX
    CharLM.export(
        weights_path="exp/checkpoints/charlm_epoch_50.pt",
        vocab_path="exp/vocab.json",
        output_path="my_model.onnx",
    )

