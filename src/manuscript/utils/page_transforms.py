from typing import Iterable, List, Literal, Optional, Sequence, Tuple

from manuscript.data import Block, Line, Page, TextSpan

from .geometry import merge_polygons


MergeMethod = Literal["bbox", "convex_hull"]
CollapseLevel = Literal["line", "block"]


def _iter_block_text_spans(block: Block) -> Iterable[TextSpan]:
    for line in block.lines:
        yield from line.text_spans


def _mean_or_none(values: Sequence[Optional[float]]) -> Optional[float]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def merge_text_spans(
    text_spans: Sequence[TextSpan],
    method: MergeMethod = "bbox",
) -> Optional[TextSpan]:
    """
    Merge multiple ``TextSpan`` objects into a single wider ``TextSpan``.

    Parameters
    ----------
    text_spans : sequence of TextSpan
        Input text spans to merge.
    method : {"bbox", "convex_hull"}, optional
        Polygon merge strategy. ``"bbox"`` creates an axis-aligned rectangle
        covering all span polygons. ``"convex_hull"`` creates a convex hull
        around all polygon vertices. Default is ``"bbox"``.

    Returns
    -------
    TextSpan or None
        Merged text span, or ``None`` when ``text_spans`` is empty.
    """
    if not text_spans:
        return None

    polygon = merge_polygons([text_span.polygon for text_span in text_spans], method=method)
    if polygon is None:
        return None

    texts = [text_span.text.strip() for text_span in text_spans if text_span.text]
    text = " ".join(texts) if texts else None

    return TextSpan(
        polygon=polygon,
        detection_confidence=sum(
            float(text_span.detection_confidence) for text_span in text_spans
        )
        / len(text_spans),
        text=text,
        recognition_confidence=_mean_or_none(
            [text_span.recognition_confidence for text_span in text_spans]
        ),
        order=0,
    )


def collapse_line_text_spans(
    line: Line,
    method: MergeMethod = "bbox",
) -> Line:
    """
    Collapse all text spans inside a line into a single text span.

    Parameters
    ----------
    line : Line
        Input line.
    method : {"bbox", "convex_hull"}, optional
        Polygon merge strategy. Default is ``"bbox"``.

    Returns
    -------
    Line
        New line containing one merged text span or an empty span list.
    """
    merged = merge_text_spans(line.text_spans, method=method)
    return Line(
        text_spans=[] if merged is None else [merged],
        order=line.order,
    )


def collapse_block_text_spans(
    block: Block,
    method: MergeMethod = "bbox",
) -> Block:
    """
    Collapse all text spans inside a block into a single line with one text span.

    Parameters
    ----------
    block : Block
        Input block.
    method : {"bbox", "convex_hull"}, optional
        Polygon merge strategy. Default is ``"bbox"``.

    Returns
    -------
    Block
        New block containing a single collapsed line.
    """
    merged = merge_text_spans(list(_iter_block_text_spans(block)), method=method)
    return Block(
        lines=[Line(text_spans=[] if merged is None else [merged], order=0)],
        order=block.order,
    )


def collapse_page_text_spans(
    page: Page,
    level: CollapseLevel = "line",
    method: MergeMethod = "bbox",
) -> Page:
    """
    Collapse narrow OCR structure into wider line-level or block-level spans.

    Parameters
    ----------
    page : Page
        Input page.
    level : {"line", "block"}, optional
        Collapse target. ``"line"`` keeps the same block/line structure and
        replaces each line with one merged text span. ``"block"`` replaces each
        block with one line containing one merged text span. Default is ``"line"``.
    method : {"bbox", "convex_hull"}, optional
        Polygon merge strategy. Default is ``"bbox"``.

    Returns
    -------
    Page
        Collapsed page.
    """
    if level not in {"line", "block"}:
        raise ValueError(f"level must be 'line' or 'block', got: {level}")

    if level == "line":
        return Page(
            blocks=[
                Block(
                    lines=[
                        collapse_line_text_spans(line, method=method)
                        for line in block.lines
                    ],
                    order=block.order,
                )
                for block in page.blocks
            ]
        )

    return Page(
        blocks=[
            collapse_block_text_spans(block, method=method)
            for block in page.blocks
        ]
    )


__all__ = [
    "collapse_block_text_spans",
    "collapse_line_text_spans",
    "collapse_page_text_spans",
    "merge_text_spans",
]
