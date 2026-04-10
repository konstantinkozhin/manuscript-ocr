Data Structures
===============

Core data structures for representing OCR results.

.. rubric:: Data Model

The following diagram shows the relationships between data structures:

.. mermaid::

    graph LR

        %% Entities
        Page[Page]
        Block[Block]
        Line[Line]
        TextSpan[TextSpan]

        %% Relations
        Page -->|"blocks: List[Block]"| Block
        Block -->|"lines: List[Line]"| Line
        Line -->|"text_spans: List[TextSpan]"| TextSpan

        %% TextSpan fields
        TextSpan --> Tpoly["polygon: List[(x, y)]<br>≥ 4 points, clockwise"]
        TextSpan --> Tdet["detection_confidence: float (0–1)"]
        TextSpan --> Ttext["text: Optional[str]"]
        TextSpan --> Trec["recognition_confidence: Optional[float] (0–1)"]
        TextSpan --> Torder["order: Optional[int]<br>assigned after sorting"]

        %% Line fields
        Line --> LineOrder["order: Optional[int]<br>assigned after sorting"]

        %% Block fields
        Block --> BlockOrder["order: Optional[int]<br>assigned after sorting"]
        Block --> FlatInput["text_spans: List[TextSpan]<br>optional flat input"]

.. rubric:: Compatibility

The canonical names in ``v0_1_11`` are ``TextSpan`` and ``text_spans``. For
code and services that still target ``v0_1_10``, ``Word`` and ``words``
remain available as compatibility aliases on import, validation, and Python
attribute access.

When exporting OCR results, choose the schema explicitly:

.. code-block:: python

    page.to_dict(schema="v0_1_11")
    page.to_json("result.json", schema="v0_1_10")

Use ``"v0_1_10"`` only for legacy JSON consumers. New integrations should
prefer ``"v0_1_11"``.

.. rubric:: Module Reference

.. automodule:: manuscript.data
   :members:
   :undoc-members:
   :show-inheritance:
