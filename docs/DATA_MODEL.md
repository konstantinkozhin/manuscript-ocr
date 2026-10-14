# Document tree

`Page` represents one image. Its `children` may contain words, lines, blocks,
tables, images, or custom elements. No fixed depth is required. Each element has
one `parent`; child sequence determines reading order.

```python
from manuscript.data import Page, TextRegion

page = Page(image_width=1000, image_height=1500)
line = TextRegion(type="line", polygon=[(10, 10), (400, 10), (400, 50), (10, 50)])
line.parent = page
line.children = [TextRegion(type="word", text="Example")]
page.save("page.json")
restored = Page.load("page.json")
```

All elements share `type`, optional `label`, `polygon`, `detection_confidence`,
and `orientation`. Coordinates refer to the original image. Orientation is an
absolute clockwise angle in degrees; `None` means unknown. Geometry is not
computed from children automatically.

`TextRegion` supports types `word`, `line`, and `block`, and stores optional
`text` and `recognition_confidence`. `Table` has optional `rows` and `columns`.
`Cell` has zero-based `row` and `column`, plus `row_span` and `column_span`.
Grid overlaps and out-of-bounds cells are rejected by `validate_tree()`; gaps
are allowed. Custom types can use generic `Element` or an explicitly registered
subclass.

`to_tree_dict()` serializes a nested tree without parent references.
`Page.clone()` rebuilds independent objects and parent links.
`Page.to_dict()`, `save()` and `load()` use single-image COCO with Manuscript
extensions. `manuscript_schema="manuscript-coco-v1"` identifies the version of
these extensions, independently of the package version. Category names are
structural types; annotations carry `parent_id` and model fields in `manuscript`.
Annotation sequence preserves sibling order.

Unversioned polygon COCO can be imported using category names as types and
optional `parent_id` links. Unknown types are preserved with a warning. RLE masks,
multiple polygons per annotation, and multi-image datasets are unsupported.
Missing segmentation leaves geometry unknown; bbox alone does not create it.

`Page.get_text(format="plain")` serializes text without changing geometry or
hierarchy. `Pipeline.get_text(page, ...)` delegates to the same method. Word
boundaries use spaces, line boundaries newlines, and block/table boundaries
blank lines. Between unlike siblings, the higher-priority boundary wins. Text
in descendants takes precedence over text stored on their parent. Flat words
remain a single line; serialization does not infer structure from geometry.

Use `separators={"word": " ", "line": "\n", "block": "\n\n"}` to customize
boundaries and `separator_priorities` to override their precedence. Separators
are inserted only between nonempty results. Tables serialize as row-major TSV,
preserving empty grid positions and emitting spanned cells only at their origin.
Internal cell whitespace becomes spaces. Only `plain` is currently supported.
