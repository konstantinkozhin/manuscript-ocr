from typing import List, Optional, Tuple

import numpy as np

from ...data import Block, Line, Page, Word


class SimpleSorting:
    """
    Layout model that groups detected words into columns and lines.

    Parameters
    ----------
    max_splits : int, optional
        Maximum number of column split attempts. Default is 10.
    use_columns : bool, optional
        If True, segment into columns before line grouping. Default is True.
    """

    def __init__(self, max_splits: int = 10, use_columns: bool = True):
        self.max_splits = max_splits
        self.use_columns = use_columns

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
        segments = [(0, img_width)]
        separators: List[int] = []

        for _ in range(self.max_splits or img_width):
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

        parts = [(0, img_width)]
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

    def predict(self, page: Page, image: Optional[np.ndarray] = None) -> Page:
        """
        Organize words in a page into blocks/lines and assign reading order.

        Parameters
        ----------
        page : Page
            Input page with detected words.
        image : numpy.ndarray, optional
            Optional source image (unused by this layout model).

        Returns
        -------
        Page
            Organized page.
        """
        _ = image

        all_words: List[Word] = []
        for block in page.blocks:
            for line in block.lines:
                all_words.extend(line.words)

        if not all_words:
            return Page(blocks=[Block(lines=[Line(words=[], order=0)], order=0)])

        word_to_box = {}
        boxes = []
        for word in all_words:
            poly = np.array(word.polygon, dtype=np.int32)
            x_min, y_min = np.min(poly, axis=0)
            x_max, y_max = np.max(poly, axis=0)
            box = (int(x_min), int(y_min), int(x_max), int(y_max))
            boxes.append(box)
            word_to_box[box] = word

        if self.use_columns:
            columns = self._segment_columns(boxes)
        else:
            columns = [boxes]

        blocks: List[Block] = []
        for block_idx, column_boxes in enumerate(columns):
            lines_in_column = self._sort_into_lines(column_boxes, use_columns=False)

            lines: List[Line] = []
            for line_idx, line_boxes in enumerate(lines_in_column):
                line_words = []
                for word_idx, box in enumerate(line_boxes):
                    word = word_to_box[box]
                    word.order = word_idx
                    line_words.append(word)

                line = Line(words=line_words, order=line_idx)
                lines.append(line)

            block = Block(lines=lines, order=block_idx)
            blocks.append(block)

        return Page(blocks=blocks)
