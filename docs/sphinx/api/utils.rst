Utilities
=========

Utility functions for image processing, visualization, page organization,
text-span collapsing, and more.

The utilities module also includes helpers for creating lightweight
``Page`` objects without running the full pipeline:

- ``create_page_from_text`` for testing correctors and other text-processing
  components from plain text.
- ``create_page_from_image`` for wrapping a single image or crop into a
  ``Page`` with one ``TextSpan`` covering the full image. It can also accept
  a sequence of crops and build a synthetic page for direct recognizer
  inference with the ``0.1.11+`` stage API.

.. automodule:: manuscript.utils
   :members:
   :undoc-members:
   :show-inheritance:

