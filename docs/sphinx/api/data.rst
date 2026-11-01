Data Structures
===============

Document tree
-------------

``Page`` represents one image. Its children may contain words, lines, blocks,
tables, images or custom elements. There is no required tree depth: a detector
can return flat regions, and a layout stage can organize them later.

Each element has one ``parent``. The sequence of ``children`` defines reading
order; there is no separate ``order`` field. Assign links through ``parent``
or ``children`` after constructing the objects.

.. mermaid::

    graph TD
        Element --> Page
        Element --> TextRegion
        Element --> Table
        Element --> Cell
        Element --> Image
        TextRegion --> Types["type: word / line / block"]

The diagram shows class inheritance. Tree containment is defined separately
by parent-child links.

.. code-block:: python

    from manuscript.data import Page, TextRegion

    page = Page(image_width=1000, image_height=1500)
    line = TextRegion(
        type="line",
        polygon=[(10, 10), (400, 10), (400, 50), (10, 50)],
    )
    line.parent = page
    line.children = [
        TextRegion(type="word", text="Hello"),
        TextRegion(type="word", text="world"),
    ]

    assert line.children[0].parent is line
    assert page.get_text() == "Hello world"

Assigning ``parent`` moves the element between containers. Cycles are not
allowed; ``Page`` must remain the root.

Elements and fields
-------------------

All classes inherit from ``Element`` and share these fields:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Meaning
   * - ``type``
     - Nonempty structural type; also used as the COCO category name.
   * - ``label``
     - Optional model-specific class name.
   * - ``polygon``
     - Optional list of at least four finite XY vertices in original-image pixels.
   * - ``detection_confidence``
     - Optional detection confidence in [0, 1].
   * - ``orientation``
     - Optional absolute clockwise angle in degrees. ``None`` means unknown;
       zero means upright. The angle must be finite.

Geometry is not calculated from children automatically. Changing a tree link
also does not change coordinates or orientation.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Class
     - Additional fields
   * - ``Page``
     - Positive ``image_width`` and ``image_height``, and optional ``file_name``
       (empty string by default). Its type is ``page``.
   * - ``TextRegion``
     - Type ``word``, ``line`` or ``block``; optional ``text`` and
       ``recognition_confidence`` in [0, 1]. Text can be stored at any of these levels.
   * - ``Table``
     - Type ``table``; optional positive ``rows`` and ``columns``.
   * - ``Cell``
     - Type ``cell``; zero-based ``row`` and ``column``; positive ``row_span``
       and ``column_span``, both 1 by default.
   * - ``Image``
     - Type ``image`` for an image region within a page.

Field validation also applies to assignment. Call ``page.validate_tree()``
after editing to check contextual rules, including overlapping table cells and
cells extending beyond declared grid dimensions. Incomplete grids are allowed.

Tables
------

Table cells are linked like other elements. Their indices describe the grid;
their children can contain text regions.

.. code-block:: python

    from manuscript.data import Table, Cell

    table = Table(rows=1, columns=2)
    cell = Cell(row=0, column=0, column_span=2)
    cell.children = [TextRegion(type="word", text="Heading")]
    table.children = [cell]
    table.parent = page
    page.validate_tree()

Copying and serialization
-------------------------

``to_tree_dict()`` produces a nested dictionary of fields and ordered children,
without parent references. ``Element.from_tree_dict()`` restores the registered
classes and parent links. ``Page.clone()`` creates independent objects:

.. code-block:: python

    copied = page.clone()
    assert copied is not page
    assert copied.to_tree_dict() == page.to_tree_dict()

    tree = page.to_tree_dict()
    restored_tree = Page.from_tree_dict(tree)
    assert restored_tree.to_tree_dict() == tree

COCO files
~~~~~~~~~~

``Page.to_dict()``, ``save()`` and ``load()`` use single-image polygon COCO
with Manuscript extensions. The public ``to_coco()`` and ``from_coco()``
functions provide the same conversion.

.. code-block:: python

    from manuscript.data import to_coco, from_coco

    coco = to_coco(page)
    restored = from_coco(coco)
    assert restored.to_tree_dict() == page.to_tree_dict()

    page.save("page.json")
    restored = Page.load("page.json")

The format preserves text, geometry and tree links. Import also supports
unversioned polygon COCO with one image. RLE masks and multiple polygons per
annotation are unsupported; a bbox alone does not supply polygon geometry.

Text output
-----------

``Page.get_text(format="plain")`` serializes text without changing the tree.
``Pipeline.get_text(page, ...)`` delegates to this method.

Words use spaces, lines use newlines, and blocks and tables use blank lines.
Between unlike siblings, the higher-priority boundary wins. Text from descendants
takes precedence over text stored on their parent, preventing duplicate output.
Flat words remain a single line: serialization does not infer lines from geometry.
Separators are inserted only between nonempty results.

.. code-block:: python

    text = page.get_text()
    compact = page.get_text(separators={"line": " / ", "block": "\n"})

Use ``separator_priorities`` to change boundary precedence; higher values win.
Defaults are word=0, line=1, block=2 and table=2.

Tables serialize as row-major TSV, preserving empty grid positions. Spanned
cells appear only at their origin; covered slots remain empty. Whitespace inside
cells becomes spaces, so cell text cannot introduce extra table rows.
Only ``plain`` output is currently supported.

Custom elements
---------------

A generic ``Element(type="custom_region")`` can retain a custom type and the
common fields. During import or cloning, unknown types produce a warning and
remain generic elements. Specialized pipeline stages may skip them.

Register a subclass when a type needs additional fields or behavior:

.. code-block:: python

    from typing import Literal
    from manuscript.data import Element

    class Stamp(Element):
        type: Literal["stamp"] = "stamp"
        inscription: str = ""

    Element.register("stamp", Stamp)
    stamp = Stamp(inscription="Archive")
    stamp.parent = page

Register custom classes before restoring their serialized data. Fields not
declared by the selected class are rejected.

.. automodule:: manuscript.data
   :members:
   :undoc-members:
   :exclude-members: finite_orientation, finite_polygon, nonempty_type, registry, model_config, model_fields, model_computed_fields
   :show-inheritance:
