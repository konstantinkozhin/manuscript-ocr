Recognizers
===========

Text recognition models.

Medoid ensemble
---------------

.. autoclass:: manuscript.recognizers.MedoidRecognizer
   :members:
   :show-inheritance:

.. code-block:: python

    from manuscript.recognizers import MedoidRecognizer

    recognizer = MedoidRecognizer([model_a, model_b, model_c], parallel=False)
    page = recognizer.predict(page, image=image)

The winner minimizes the sum of unit-cost character Levenshtein distances to
all available normalized predictions. Repeated predictions retain their weight.
Ties prefer the earliest model. No new text is generated and confidence is not
used. Normalization and strict tree validation match MajorityVotingRecognizer.
None abstains, empty strings participate, and regions without answers preserve
their input text. Output confidence is None for an aggregated region.

For every text region, predictions longer than max(512, 8 times median length)
are excluded if at least three answers remain, including words, lines and blocks.
Raw ``voting`` records retain all original candidates and add ``distance_sums``,
``winning_distance_sum``, ``used_models``, ``excluded_model_indices`` and
``length_limit``. Distances use RapidFuzz, included in the runtime dependencies.

Majority voting
---------------

.. autoclass:: manuscript.recognizers.MajorityVotingRecognizer
   :members:
   :show-inheritance:

.. code-block:: python

    from manuscript.recognizers import MajorityVotingRecognizer

    recognizer = MajorityVotingRecognizer([model_a, model_b, model_c], parallel=False)
    page = recognizer.predict(page, image=image)
    raw = recognizer.predict(page, image=image, return_raw=True)

Participants receive independent copies with cleared text/confidence. Their
returned trees must preserve every field, geometry, node class, child count
and child order, except TextRegion text/recognition_confidence. Violations
raise an error naming the participant, node path and field. Position in the
unchanged tree identifies a region; word/line/block votes are separate.

Whole strings are normalized with NFC, lowercasing and strip. The most frequent
string wins; ties use the first supporting model in list order. Confidence
does not weight votes. Empty strings vote; None abstains. Regions without votes
keep their input text/confidence. Winners receive normalized text and confidence
None. Raw output contains ``page`` and per-region ``voting`` records with paths,
candidates, normalized strings, counts and the winner, without model tensors.

``parallel=True`` runs participants in threads. It keeps the same tie priority;
users should choose sequential execution for models sharing constrained device
memory. Per-model batch sizes are preserved unless explicitly overridden.

Batch size
----------

``TRBA(batch_size=128)`` sets the default inference batch size. An explicit
``predict(..., batch_size=...)`` overrides it; Pipeline uses the configured
recognizer value. The last incomplete ONNX batch is padded internally and only
the original predictions are returned. Fixed-batch models may restrict the
effective batch size.

Training architecture
---------------------

Pass architecture options through ``TRBA.train(..., **extra_config)``:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Option
     - Default
     - Meaning
   * - ``decoder_type``
     - ``"attention"``
     - Choose ``"attention"`` or ``"parseq"``. ``parseq_linear`` is an experiment configuration name.
   * - ``cnn_backbone``
     - ``"seresnet31"``
     - ``"seresnetlite31v2"`` selects SEResNet31LiteV2.
   * - ``transformation``
     - ``"none"``
     - ``"tps"`` enables input rectification.
   * - ``tps_num_fiducial``
     - ``20``
     - Number of TPS control points; must be even and at least 4.
   * - ``decoder_layers``, ``decoder_heads``, ``decoder_ffn``, ``decoder_dropout``
     - ``2``, ``8``, ``1024``, ``0.1``
     - PARSeq decoder settings. Hidden size must be divisible by the number of heads.
   * - ``parseq_permutations``, ``parseq_refine_iters``
     - ``6``, ``1``
     - PARSeq training permutations and inference refinement iterations.

.. code-block:: python

   checkpoint = TRBA.train(
       train_csvs="data/train.csv",
       train_roots="data/images",
       exp_dir="runs/parseq_tps",
       cnn_backbone="seresnetlite31v2",
       decoder_type="parseq",
       transformation="tps",
       max_len=40,
       pretrain_weights=None,
   )
   TRBA.export(
       weights_path=checkpoint,
       config_path="runs/parseq_tps/config.json",
       charset_path="runs/parseq_tps/charset.txt",
       output_path="runs/parseq_tps/model.onnx",
   )

Training saves architecture settings for export. TPS export uses ONNX opset 16
or newer for GridSample; the exporter raises the requested opset when needed.
The default ``pretrain_weights="default"`` (or ``True``) downloads the
``trba_lite_g1`` checkpoint through the registry. ``None`` or ``False`` trains
from scratch. To use another preset, download its ``checkpoint`` artifact and
pass that path; local paths and custom weight URLs are also supported. Resuming
from a checkpoint skips pretraining. Compatibility loading may skip layers with
different shapes, so inspect its messages when changing architectures.

.. autoclass:: manuscript.recognizers.TRBA
   :members:
   :undoc-members:
   :show-inheritance:
