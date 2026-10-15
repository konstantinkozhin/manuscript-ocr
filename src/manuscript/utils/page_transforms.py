"""Collapse text regions in the document tree without legacy Page structures."""
from typing import Optional, Sequence

from manuscript.data import Element, Page, TextRegion
from .geometry import merge_polygons


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _text(region):
    children = [_text(child) for child in region.children]
    children = [text for text in children if text]
    if children:
        return (" " if region.type == "line" else "\n").join(children)
    return region.text or ""


def merge_text_spans(text_spans: Sequence[TextRegion], method="bbox", *, type="word") -> Optional[TextRegion]:
    """Merge detached text regions into one region; inputs remain unchanged."""
    if not text_spans:
        return None
    if any(not isinstance(node, TextRegion) for region in text_spans
           for node in (region, *region.descendants)):
        raise TypeError("Only text regions can be collapsed")
    polygons = [node.polygon for region in text_spans for node in (region, *region.descendants)
                if node.polygon is not None]
    polygon = merge_polygons(polygons, method=method) if polygons else None
    if polygon is not None and len(polygon) == 3:
        # Preserve the triangular outline with four vertices for TextRegion.
        first, second = polygon[:2]
        polygon.insert(1, tuple((a + b) / 2 for a, b in zip(first, second)))
    elif polygon is not None and len(polygon) < 3:
        polygon = merge_polygons(polygons, method="bbox")
    texts = [_text(region) for region in text_spans]
    angles = {region.orientation for region in text_spans}
    return TextRegion(type=type, polygon=polygon,
        text=" ".join(text for text in texts if text) or None,
        detection_confidence=_mean([region.detection_confidence for region in text_spans]),
        recognition_confidence=_mean([region.recognition_confidence for region in text_spans]),
        orientation=next(iter(angles)) if len(angles) == 1 else None)


def _collapse(region, expected, method):
    if not isinstance(region, TextRegion) or region.type != expected:
        raise TypeError(f"Expected a {expected} TextRegion")
    merged = merge_text_spans([region], method=method, type=expected)
    # Keep the region's metadata and existing geometry; remove its text children.
    result = Element.from_tree_dict(region.to_tree_dict())
    result.children = []
    result.text = merged.text
    if result.polygon is None:
        result.polygon = merged.polygon
    return result


def collapse_line_text_spans(line, method="bbox"):
    return _collapse(line, "line", method)


def collapse_block_text_spans(block, method="bbox"):
    return _collapse(block, "block", method)


def collapse_page_text_spans(page: Page, level="line", method="bbox") -> Page:
    """Collapse matching text containers while preserving unrelated tree nodes."""
    if not isinstance(page, Page):
        raise TypeError("Expected the document-tree Page")
    if level not in ("line", "block"):
        raise ValueError("level must be line or block")
    result = page.clone()
    for region in list(result.descendants):
        if isinstance(region, TextRegion) and region.type == level and region.root is result:
            collapsed = _collapse(region, level, method)
            region.text = collapsed.text
            region.polygon = collapsed.polygon
            region.children = []
    return result


__all__ = ["merge_text_spans", "collapse_line_text_spans", "collapse_block_text_spans", "collapse_page_text_spans"]
