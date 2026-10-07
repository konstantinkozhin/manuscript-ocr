Model Registry
==============

Version 0.1.13 resolves model presets through a shared registry. Local paths,
HTTP/HTTPS URLs, GitHub release URLs and Google Drive identifiers remain supported.
Importing ``manuscript.models`` does not make network requests.

Sources and refresh
-------------------

The official registry is read from ``main`` on GitHub, with GitVerse as a mirror.
The package includes a fallback snapshot. The cached catalog is refreshed when
missing, when a requested key is unknown, or once after all artifact mirrors fail.
There is no time-based refresh or background polling. Invalid responses do not
replace a valid cached catalog.

.. code-block:: python

   from manuscript import models
   from manuscript.recognizers import TRBA

   recognizer = TRBA(weights="trba_lite_g2")
   metadata = models.info("trba_lite_g2")
   files = models.download("trba_lite_g2")
   checkpoint = models.download("trba_base_g0", artifact="checkpoint")

``trba_base_g0`` is a checkpoint-only preset; it has no published ONNX weights.
Download progress is displayed for artifacts that need downloading.

Custom catalogs supplement the official catalog. With ``priority=True``, their
entries override official keys: use this only for a source whose metadata and
artifact URLs you trust. ``persist=True`` retains the source for later processes.
Call ``refresh()`` to activate changes to the source configuration. Unavailable
custom sources retain their last valid cached catalog.

.. code-block:: python

   models.add_registry("https://example.org/registry.json", priority=True, persist=True)
   models.refresh()

Storage and integrity
---------------------

The default root is ``~/.manuscript``; override it with ``MANUSCRIPT_HOME``.
Each bundle lives in ``models/<model-key>/`` with its ``model.json`` manifest,
weights, supporting files and a shared ``LICENSE.txt`` when declared. Catalogs
and persistent source settings live under ``registries/``.

Required artifacts are downloaded together. Checkpoints and lexicons may be
optional; CharLM requests its lexicon when needed. A declared license is fetched
alongside requested artifacts. Files are verified for declared size and SHA-256
before atomic replacement. A ``null`` size or hash disables that particular check.
Old flat files under ``weights/`` are reused only when their SHA-256 matches;
the old files are retained.

``list_installed()`` includes partial bundles. ``verify()`` returns checks for
required files, a declared license and optional files already present.
``is_installed()`` is true only if that nonempty set of checks passes.
``remove()`` deletes the bundle identified by its key.

.. code-block:: python

   directory = models.path("trba_lite_g2")
   checks = models.verify("trba_lite_g2")
   ready = models.is_installed("trba_lite_g2")
   installed = models.list_installed()
   # Delete a downloaded bundle when you no longer need it:
   # models.remove("trba_lite_g2")

Compatibility and catalog metadata
----------------------------------

``model_classes`` restricts the public class that may load a preset. Compatibility
metadata accepts an exact ``library_version`` and inclusive
``library_version_min`` / ``library_version_max`` bounds. A mismatch emits a
``RuntimeWarning`` and execution continues. Older official models declare
0.1.10–0.1.13; legacy exact entries declaring 0.1.13 also accept that range.
This describes tested compatibility, not a guarantee for untested versions.

Catalog schema version is ``1``. Each entry includes ``description``, a shared
model ``license`` and ``artifacts``. An artifact specifies ``filename``, ``format``,
``size``, ``sha256``, ordered mirror ``urls`` and optionally ``required`` (true by
default). Mirrors must serve identical bytes. Unknown metadata uses JSON ``null``.
Publish a new model under a new key; update URLs to move existing artifacts.
Do not infer a model license from the library license.

See :doc:`api/models` for function signatures.
