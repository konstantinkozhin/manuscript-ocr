"""Document elements, ordered parent-child links, and single-page COCO I/O.

Coordinates use original-image pixels. Each element has one parent; child
sequence defines reading order. Tree links are assigned through parent/children,
not Pydantic constructor fields. Unknown types retain their data as Element.
"""
import json
import math
import warnings
from pathlib import Path
from typing import ClassVar, Dict, List, Literal, Optional, Tuple, Type

from anytree import NodeMixin, PreOrderIter
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Element(BaseModel, NodeMixin):
    """A positioned or geometry-free document node.

    type defines structural semantics and the exported COCO category. label
    stores an optional model-specific class name. Polygon and confidence are
    optional; geometry is not inferred from children. orientation is an absolute
    clockwise angle in degrees, with None meaning unknown and zero upright.
    """
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    type: str
    label: Optional[str] = None
    polygon: Optional[List[Tuple[float, float]]] = Field(None, min_length=4)
    detection_confidence: Optional[float] = Field(None, ge=0, le=1)
    orientation: Optional[float] = None

    registry: ClassVar[Dict[str, Type["Element"]]] = {}

    @field_validator("orientation")
    @classmethod
    def finite_orientation(cls, value):
        if value is not None and not math.isfinite(value):
            raise ValueError("Orientation must be finite degrees or None")
        return value

    @field_validator("type")
    @classmethod
    def nonempty_type(cls, value):
        if not value.strip():
            raise ValueError("Element type must be a nonempty string")
        return value

    @classmethod
    def resolve_type(cls, kind: str) -> Type["Element"]:
        """Use registered semantics or preserve an unfamiliar structural type."""
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("Element type must be a nonempty string")
        if kind in cls.registry:
            return cls.registry[kind]
        warnings.warn(
            f"Unknown element type {kind!r}; preserving it as Element. "
            "Specialized stages may skip it. Register a class with Element.register() "
            "if this type needs additional fields or behavior.",
            UserWarning, stacklevel=2,
        )
        return Element

    @field_validator("polygon")
    @classmethod
    def finite_polygon(cls, value):
        if value is not None and not all(math.isfinite(c) for point in value for c in point):
            raise ValueError("Polygon coordinates must be finite")
        return value

    def _pre_attach(self, parent):
        if not isinstance(parent, Element):
            raise TypeError("Parent must be an Element")
        if isinstance(self, Page):
            raise ValueError("Page must remain a root")

    def validate_tree(self) -> None:
        """Check context-dependent rules after construction or editing.

        Geometry is never inferred from children. Empty containers and detached
        elements are allowed. Table grid checks do not require a complete grid.
        """
        for node in PreOrderIter(self):
            for child in node.children:
                if not isinstance(child, Element):
                    raise TypeError("Children must be Elements")
            if isinstance(node, Table):
                cells = [child for child in node.children if isinstance(child, Cell)]
                for i, cell in enumerate(cells):
                    if node.rows is not None and cell.row + cell.row_span > node.rows:
                        raise ValueError("Cell extends beyond table rows")
                    if node.columns is not None and cell.column + cell.column_span > node.columns:
                        raise ValueError("Cell extends beyond table columns")
                    for other in cells[:i]:
                        if (cell.row < other.row + other.row_span
                                and other.row < cell.row + cell.row_span
                                and cell.column < other.column + other.column_span
                                and other.column < cell.column + cell.column_span):
                            raise ValueError("Table cells overlap in the grid")

    @classmethod
    def register(cls, kind: str, node_class: Type["Element"]) -> None:
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("Registered type must be a nonempty string")
        if kind in cls.registry and cls.registry[kind] is not node_class:
            raise ValueError("Element kind already registered: " + kind)
        if not issubclass(node_class, Element):
            raise TypeError("Registered class must inherit Element")
        cls.registry[kind] = node_class

    def to_tree_dict(self) -> dict:
        """Serialize node fields and ordered children without parent cycles."""
        result = self.model_dump(mode="json")
        result["children"] = [child.to_tree_dict() for child in self.children]
        return result

    @classmethod
    def from_tree_dict(cls, data: dict) -> "Element":
        """Restore typed nodes and parent links from a nested tree dictionary."""
        fields = dict(data)
        children = fields.pop("children", [])
        kind = fields.get("type")
        node = cls.resolve_type(kind)(**fields)
        node.children = [cls.from_tree_dict(child) for child in children]
        return node


class Page(Element):
    """Root for one image; children may be flat regions or nested structures."""
    type: Literal["page"] = "page"
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    file_name: str = ""

    def clone(self):
        """Copy data and rebuild parent links within the copied tree."""
        return self.from_tree_dict(self.to_tree_dict())

    def to_dict(self):
        return to_coco(self)

    def get_text(self, format="plain", *, separators=None, separator_priorities=None):
        """Serialize text with type boundaries and row-major plain-text tables."""
        from ._text import get_text
        return get_text(self, format=format, separators=separators,
                        separator_priorities=separator_priorities)

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return from_coco(json.loads(Path(path).read_text(encoding="utf-8")))


class TextRegion(Element):
    """Word, line or block, with independently stored recognition results."""
    type: Literal["word", "line", "block"]
    text: Optional[str] = None
    recognition_confidence: Optional[float] = Field(None, ge=0, le=1)


class Table(Element):
    """Table with optional grid dimensions and cell children."""
    type: Literal["table"] = "table"
    rows: Optional[int] = Field(None, gt=0)
    columns: Optional[int] = Field(None, gt=0)


class Cell(Element):
    """Cell at a zero-based grid position, optionally spanning rows or columns."""
    type: Literal["cell"] = "cell"
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    row_span: int = Field(1, ge=1)
    column_span: int = Field(1, ge=1)


class Image(Element):
    type: Literal["image"] = "image"


for kind, node_class in (
    ("page", Page), ("word", TextRegion), ("line", TextRegion), ("block", TextRegion), ("table", Table),
    ("cell", Cell), ("image", Image),
):
    Element.register(kind, node_class)


# Identifies the saved format of Manuscript-specific COCO fields.
COCO_SCHEMA = "manuscript-coco-v1"


def geometry_fields(polygon):
    if polygon is None:
        return {"segmentation": [], "bbox": [0, 0, 0, 0], "area": 0.0, "iscrowd": 0}
    xs, ys = zip(*polygon)
    area = abs(sum(
        polygon[i][0] * polygon[(i + 1) % len(polygon)][1]
        - polygon[(i + 1) % len(polygon)][0] * polygon[i][1]
        for i in range(len(polygon))
    )) / 2
    return {"segmentation": [[c for point in polygon for c in point]],
            "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
            "area": area, "iscrowd": 0}


def to_coco(page: Page) -> dict:
    """Single-page COCO with lossless Manuscript extensions.

    IDs belong to this export. Missing geometry remains an empty segmentation,
    not a fabricated polygon; such objects are containers, not training targets.
    """
    page.validate_tree()
    nodes = list(PreOrderIter(page))[1:]
    ids = {id(node): index for index, node in enumerate(nodes, 1)}
    categories, category_ids, annotations = [], {}, []
    for node in nodes:
        name = node.type
        if name not in category_ids:
            category_ids[name] = len(categories) + 1
            categories.append({"id": category_ids[name], "name": name})
        fields = node.model_dump(mode="json")
        fields.pop("polygon")
        annotations.append({
            "id": ids[id(node)], "image_id": 1, "category_id": category_ids[name],
            "parent_id": None if node.parent is page else ids[id(node.parent)],
            **geometry_fields(node.polygon), "manuscript": fields,
        })
    page_fields = page.model_dump(mode="json", exclude={"image_width", "image_height", "file_name"})
    return {"manuscript_schema": COCO_SCHEMA,
            "images": [{"id": 1, "width": page.image_width, "height": page.image_height,
                        "file_name": page.file_name, "manuscript": page_fields}],
            "categories": categories, "annotations": annotations}


def from_coco(data: dict) -> Page:
    """Read one page from Manuscript COCO or unversioned polygon COCO.

    Unversioned input uses category names as types, optional parent_id links
    and value as text. Empty values mean unknown text. Annotation sequence
    determines sibling order. Known category names select typed elements; unknown
    names create generic Elements with a warning. RLE masks and multiple
    polygons are unsupported. Bbox alone does not supply a polygon.
    """
    native = "manuscript_schema" in data
    if native and data["manuscript_schema"] != COCO_SCHEMA:
        raise ValueError("Unsupported Manuscript COCO schema")
    if len(data["images"]) != 1:
        raise ValueError("Expected exactly one image per Page")
    image = data["images"][0]
    page = Page(**(image["manuscript"] if native else {}), image_width=image["width"],
                image_height=image["height"], file_name=image.get("file_name", ""))
    categories = {}
    for category in data["categories"]:
        if category["id"] in categories:
            raise ValueError("Duplicate category ID")
        categories[category["id"]] = category["name"]
    nodes, parents = {}, {}
    for annotation in data["annotations"]:
        aid = annotation["id"]
        if type(aid) is not int or aid <= 0 or aid in nodes:
            raise ValueError("Invalid or duplicate annotation ID")
        if annotation["image_id"] != image["id"]:
            raise ValueError("Unknown image ID")
        if native:
            fields = dict(annotation["manuscript"])
        else:
            fields = {"type": categories.get(annotation["category_id"])}
            if fields["type"] in ("word", "line", "block"):
                fields["text"] = annotation.get("value") or None
            for key in ("label", "detection_confidence", "recognition_confidence", "orientation"):
                if key in annotation:
                    fields[key] = annotation[key]
        kind = fields["type"]
        if kind == "page":
            raise ValueError("Invalid annotation type: page")
        if categories.get(annotation["category_id"]) != kind:
            raise ValueError("Conflicting category metadata")
        segmentation = annotation.get("segmentation", [])
        if not isinstance(segmentation, list):
            raise ValueError("Only polygon segmentations are supported")
        if segmentation:
            if len(segmentation) != 1 or len(segmentation[0]) < 8 or len(segmentation[0]) % 2:
                raise ValueError("Expected one polygon with at least four vertices")
            coords = segmentation[0]
            fields["polygon"] = list(zip(coords[::2], coords[1::2]))
        else:
            fields["polygon"] = None
        node = Element.resolve_type(kind)(**fields)
        if native and geometry_fields(node.polygon) != {
                key: annotation[key] for key in ("segmentation", "bbox", "area", "iscrowd")}:
            raise ValueError("Inconsistent geometry fields")
        nodes[aid] = node
        parents[aid] = annotation.get("parent_id")
    # Validate references and cycles before mutating any links.
    done = set()
    for aid in nodes:
        trail, current = set(), aid
        while current is not None and current not in done:
            if type(current) is not int or current not in nodes:
                raise ValueError("Unknown parent ID")
            if current in trail:
                raise ValueError("Cyclic parent links")
            trail.add(current)
            current = parents[current]
        done.update(trail)
    for aid, node in nodes.items():
        node.parent = page if parents[aid] is None else nodes[parents[aid]]
    page.validate_tree()
    return page


def recognition_candidates(root: Element, levels=("word", "line")) -> List[TextRegion]:
    """Unrecognized regions at selected levels; deeper selected levels win."""
    unknown = set(levels) - {"word", "line", "block"}
    if unknown:
        raise ValueError("Unknown recognition levels: " + str(unknown))
    return [
        node for node in PreOrderIter(root)
        if isinstance(node, TextRegion)
        and node.type in levels and node.text is None and node.polygon is not None
        and not any(isinstance(child, TextRegion) and child.type in levels
                    and child.polygon is not None for child in node.descendants)
    ]
