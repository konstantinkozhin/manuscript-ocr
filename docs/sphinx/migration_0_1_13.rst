Migrating from 0.1.12 to 0.1.13
===============================

Version 0.1.13 replaces the fixed OCR hierarchy with a document tree. Update
code that constructs regions, traverses results or reads saved JSON. See
:doc:`api/data` for the full data contract and :doc:`release_notes` for other
changes.

Elements and reading order
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Previous API
     - 0.1.13
   * - ``Block``, ``Line``, ``TextSpan`` (or ``Word``)
     - ``TextRegion(type="block" / "line" / "word")``
   * - ``page.blocks``, ``block.lines``, ``line.text_spans``
     - Ordered ``children`` links; use ``descendants`` to traverse all depths.
   * - ``order`` fields
     - Position in the parent's ``children`` sequence.
   * - Child lists passed to constructors
     - Assign ``children`` or ``parent`` after constructing elements.

The tree can be flat or nested. Do not assume every word has a line and block
above it. To construct the former three-level hierarchy explicitly:

.. code-block:: python

    from manuscript.data import Page, TextRegion

    page = Page(image_width=200, image_height=80)
    block = TextRegion(type="block")
    line = TextRegion(type="line")
    word = TextRegion(
        type="word", text="Archive",
        polygon=[(10, 10), (100, 10), (100, 30), (10, 30)],
    )
    page.children = [block]
    block.children = [line]
    line.children = [word]
    page.validate_tree()
    assert word.parent is line
    assert page.get_text() == "Archive"

To collect words regardless of their depth:

.. code-block:: python

    words = [node for node in page.descendants
             if isinstance(node, TextRegion) and node.type == "word"]
    assert words == [word]

Use ``page.get_text()`` for text output. It respects child order and prefers
descendant text over parent text. Use ``SimpleSorting`` to group flat words
into lines and blocks, or ``ReadingSorting`` to order existing regions and
optionally infer containment.

Saved results
-------------

``page.to_dict()`` now returns single-image polygon COCO with Manuscript
extensions. The previous ``schema="v0_1_10"`` and ``schema="v0_1_11"`` options
are removed. To write a file, replace ``page.to_json(path)`` with
``page.save(path)``. Unlike ``to_json()``, ``save()`` requires a path and returns
``None`` instead of a JSON string:

.. code-block:: python

    page.save("page.json")
    restored = Page.load("page.json")
    assert restored.to_tree_dict() == page.to_tree_dict()

For JSON strings, use explicit encoding and decoding. ``Page.from_json()`` is
removed; ``Page.load()`` accepts file paths only:

.. code-block:: python

    import json
    from manuscript.data import from_coco

    payload = json.dumps(page.to_dict(), ensure_ascii=False)
    restored = from_coco(json.loads(payload))
    assert restored.to_tree_dict() == page.to_tree_dict()

``Page.load()`` accepts the new format and supported unversioned polygon COCO;
it does not convert legacy nested OCR JSON. Convert those files explicitly by
creating typed elements and rebuilding their links, or read them with the
matching older library version first. For a nested representation in new
integrations, use ``to_tree_dict()`` and ``Page.from_tree_dict()``. The
``manuscript-coco-v1`` marker identifies the serialization format independently
of the package version.
