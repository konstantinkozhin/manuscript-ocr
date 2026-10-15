from typing import List, Optional, Tuple

import numpy as np

from manuscript.api.layout import BaseLayout

from ...data import Page, TextRegion
from collections import defaultdict, deque
from ...utils.grouping import group_regions


class SimpleSorting(BaseLayout):
    """Group detected words into lines and arrange text in reading order.

    Parameters
    ----------
    max_splits : int, optional
        Maximum number of column splits. Default is 10; zero uses the
        horizontal extent of the regions as the iteration limit.
    use_columns : bool, optional
        Separate Page-level words into columns before grouping. Default is True.

    Notes
    -----
    Modifies and returns the supplied Page. Words with polygons are grouped
    into new lines unless their parent is already a line. At the Page level,
    each detected column receives a new block; with use_columns=False, all
    Page-level words are grouped into one block.

    Existing parent links are preserved apart from grouping these words.
    Words inside lines sort left to right; other regions generally sort top
    to bottom. A Page containing only blocks sorts them left to right.
    Regions without geometry sort last. Table and cell contents are untouched.

    Original regions keep their polygons, text, labels and confidence values.
    New lines and blocks receive bounding rectangles and no text or confidence.

    Examples
    --------
    >>> from manuscript.layouts import SimpleSorting
    >>> layout = SimpleSorting(use_columns=False)
    >>> page = layout.predict(page)

    See Also
    --------
    ReadingSorting : Order detected regions and optionally infer containment.
    """

    def __init__(self, max_splits: int = 10, use_columns: bool = True):
        if isinstance(max_splits, bool) or not isinstance(max_splits, int) or max_splits < 0:
            raise ValueError("max_splits must be a nonnegative integer")
        if not isinstance(use_columns, bool):
            raise TypeError("use_columns must be a boolean")
        self.max_splits = max_splits
        self.use_columns = use_columns

    def _initialize_session(self):
        """Layout stage has no backend session."""
        return None

    @staticmethod
    def _resolve_intersections(
        boxes: List[Tuple[float, float, float, float]],
    ) -> List[Tuple[float, float, float, float]]:
        def intersect(
            b1: Tuple[float, float, float, float],
            b2: Tuple[float, float, float, float],
        ) -> bool:
            return not (
                b1[2] <= b2[0] or b2[2] <= b1[0] or b1[3] <= b2[1] or b2[3] <= b1[1]
            )

        resolved = list(boxes)
        max_iterations = 50

        for _ in range(max_iterations):
            changed = False
            for i in range(len(resolved)):
                for j in range(i + 1, len(resolved)):
                    if intersect(resolved[i], resolved[j]):
                        x0, y0, x1, y1 = resolved[i]
                        x0b, y0b, x1b, y1b = resolved[j]

                        resolved[i] = (
                            x0,
                            y0,
                            int(x1 - (x1 - x0) * 0.1),
                            int(y1 - (y1 - y0) * 0.1),
                        )
                        resolved[j] = (
                            x0b,
                            y0b,
                            int(x1b - (x1b - x0b) * 0.1),
                            int(y1b - (y1b - y0b) * 0.1),
                        )
                        changed = True

            if not changed:
                break

        return resolved

    @staticmethod
    def _find_gaps(
        boxes: List[Tuple[int, int, int, int]], start: int, end: int
    ) -> List[int]:
        segs = [
            (max(b[0], start), min(b[2], end))
            for b in boxes
            if not (b[2] <= start or b[0] >= end)
        ]
        if not segs:
            return []
        segs.sort()
        merged = [segs[0]]
        for s, e in segs[1:]:
            ms, me = merged[-1]
            if s <= me:
                merged[-1] = (ms, max(me, e))
            else:
                merged.append((s, e))
        gaps = []
        prev_end = start
        for s, e in merged:
            if s > prev_end:
                gaps.append((prev_end, s))
            prev_end = e
        if prev_end < end:
            gaps.append((prev_end, end))
        return [(a + b) // 2 for a, b in gaps if b - a > 1]

    @staticmethod
    def _emptiness(
        boxes: List[Tuple[int, int, int, int]], start: int, end: int
    ) -> float:
        col = [b for b in boxes if b[0] >= start and b[2] <= end]
        if not col:
            return 1.0
        min_y, min_x = min(b[1] for b in col), max(b[3] for b in col)
        rect = (end - start) * (min_x - min_y)
        area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in col)
        return (rect - area) / rect if rect else 1.0

    def _segment_columns(
        self,
        boxes: List[Tuple[int, int, int, int]],
    ) -> List[List[Tuple[int, int, int, int]]]:
        if not boxes:
            return []

        img_width = max(b[2] for b in boxes)
        left_edge = min(0, min(b[0] for b in boxes))
        segments = [(left_edge, img_width)]
        separators: List[int] = []

        for _ in range(self.max_splits or max(1, int(np.ceil(img_width - left_edge)))):
            best = None
            for idx, (s, e) in enumerate(segments):
                for x in self._find_gaps(boxes, s, e):
                    if not (
                        any(b[2] <= x and b[0] >= s for b in boxes)
                        and any(b[0] >= x and b[2] <= e for b in boxes)
                    ):
                        continue
                    score = self._emptiness(boxes, s, x) + self._emptiness(
                        boxes, x, e
                    )
                    if best is None or score < best[0]:
                        best = (score, x, idx)
            if not best:
                break
            _, x_split, idx = best
            s, e = segments.pop(idx)
            separators.append(x_split)
            segments.insert(idx, (s, x_split))
            segments.insert(idx + 1, (x_split, e))
            segments.sort()

        parts = [(left_edge, img_width)]
        for x in separators:
            new_parts: List[Tuple[int, int]] = []
            for s, e in parts:
                if s < x < e:
                    new_parts += [(s, x), (x, e)]
                else:
                    new_parts.append((s, e))
            parts = new_parts

        cols: List[List[Tuple[int, int, int, int]]] = []
        for s, e in parts:
            col = [b for b in boxes if b[0] >= s and b[2] <= e]
            cols.append(col)

        cols = [c for c in cols if c]
        if not cols:
            return []

        return sorted(cols, key=lambda c: min(b[0] for b in c))

    @staticmethod
    def _sort_column_into_lines(
        compressed: List[Tuple[int, int, int, int]],
        mapping: dict,
        y_tol_ratio: float,
        x_gap_ratio: float,
    ) -> List[List[Tuple[float, float, float, float]]]:
        if not compressed:
            return []

        avg_h = np.mean([b[3] - b[1] for b in compressed])
        lines = []

        for b in sorted(compressed, key=lambda b: (b[1] + b[3]) / 2):
            cy = (b[1] + b[3]) / 2
            placed = False

            for ln in lines:
                line_cy = np.mean([(v[1] + v[3]) / 2 for v in ln])
                last_x1 = max(v[2] for v in ln)

                if (
                    abs(cy - line_cy) <= avg_h * y_tol_ratio
                    and (b[0] - last_x1) <= avg_h * x_gap_ratio
                ):
                    ln.append(b)
                    placed = True
                    break

            if not placed:
                lines.append([b])

        lines.sort(key=lambda ln: np.mean([(b[1] + b[3]) / 2 for b in ln]))

        result = []
        for ln in lines:
            ln.sort(key=lambda b: b[0])
            original_line = [mapping[b] for b in ln]
            result.append(original_line)

        return result

    def _sort_into_lines(
        self,
        boxes: List[Tuple[float, float, float, float]],
        y_tol_ratio: float = 0.6,
        x_gap_ratio: float = np.inf,
        use_columns: bool = True,
    ) -> List[List[Tuple[float, float, float, float]]]:
        if not boxes:
            return []

        compressed = self._resolve_intersections(boxes)
        mapping = {c: o for c, o in zip(compressed, boxes)}

        if use_columns:
            columns = self._segment_columns(compressed)
            all_lines = []
            for column_boxes in columns:
                column_lines = self._sort_column_into_lines(
                    column_boxes, mapping, y_tol_ratio, x_gap_ratio
                )
                all_lines.extend(column_lines)
            return all_lines

        return self._sort_column_into_lines(
            compressed, mapping, y_tol_ratio, x_gap_ratio
        )

    @staticmethod
    def _bounds(node):
        points = np.asarray(node.polygon, dtype=float)
        return (*points.min(axis=0), *points.max(axis=0))

    def predict(self, page: Page, image: Optional[np.ndarray] = None) -> Page:
        """Group and sort regions in place.

        Parameters
        ----------
        page : Page
            Document tree to organize.
        image : numpy.ndarray, optional
            Accepted for Pipeline compatibility; not used.

        Returns
        -------
        Page
            The supplied Page with updated grouping and child order.
        """
        if not isinstance(page, Page):
            raise TypeError("SimpleSorting requires the document-tree Page")
        page.validate_tree()
        self._group(page)
        page.validate_tree()
        return page

    def _group(self, parent):
        if parent.type in {"table", "cell"}:
            return
        for child in tuple(parent.children):
            self._group(child)
        words = [child for child in parent.children
                 if isinstance(child, TextRegion) and child.type == "word"
                 and child.polygon is not None]
        is_line = isinstance(parent, TextRegion) and parent.type == "line"
        if words and not is_line:
            boxes = [self._bounds(word) for word in words]
            lookup = defaultdict(deque)
            for word, box in zip(words, boxes):
                lookup[box].append(word)
            columns = (self._segment_columns(boxes)
                       if self.use_columns and isinstance(parent, Page) else [boxes])
            for column in columns:
                grouped_lines = []
                for line_boxes in self._sort_column_into_lines(
                        column, {box: box for box in column}, 0.6, np.inf):
                    members = [lookup[box].popleft() for box in line_boxes]
                    grouped_lines.append(group_regions(members, type="line"))
                if isinstance(parent, Page):
                    group_regions(grouped_lines, type="block")
        children = list(parent.children)
        if is_line:
            key = lambda node: (node.polygon is None,
                                self._bounds(node)[0] if node.polygon else 0)
        else:
            key = lambda node: (node.polygon is None,
                                self._bounds(node)[1] if node.polygon else 0,
                                self._bounds(node)[0] if node.polygon else 0)
        if isinstance(parent, Page) and children and all(
                isinstance(node, TextRegion) and node.type == "block" for node in children):
            key = lambda node: (node.polygon is None,
                                self._bounds(node)[0] if node.polygon else 0)
        parent.children = sorted(children, key=key)
