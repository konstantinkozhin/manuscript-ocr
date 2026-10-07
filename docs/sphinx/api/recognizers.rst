Recognizers
===========

Text recognition models.

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
