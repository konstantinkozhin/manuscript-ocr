"""Recursive reading order with optional geometric containment."""
import math

from manuscript.api.layout import BaseLayout
from manuscript.data import Page, Table, Cell, TextRegion
from shapely.geometry import Polygon


class ReadingSorting(BaseLayout):
    """Arrange detected regions in reading order and optionally nest them.

    Parameters
    ----------
    mode : {"sort", "containment"}, optional
        "sort" changes child order while retaining parent links. "containment"
        also assigns Page-level regions to detected lines and blocks.
        Default is "sort".
    containment_overlap : float, optional
        Minimum intersection area divided by the child's polygon area for
        assigning a parent. Must be in (0, 1]; default is 0.5.
    row_overlap : float, optional
        Minimum vertical overlap divided by the smaller bounding-box height
        for regions to share a reading row. Must be in (0, 1]; default is 0.5.
    infer_columns : bool, optional
        Use horizontal whitespace gaps to order columns left to right.
        Default is False. In containment mode, also create column blocks
        when the input has no Page-level text blocks.

    Examples
    --------
    >>> from manuscript.layouts import ReadingSorting
    >>> layout = ReadingSorting(mode="containment", containment_overlap=0.5)
    >>> page = layout.predict(page)

    See Also
    --------
    SimpleSorting : Create lines and blocks from detected words.
    """

    def __init__(self, mode="sort", containment_overlap=0.5,
                 row_overlap=0.5, infer_columns=False):
        if mode not in {"sort", "containment"}:
            raise ValueError("mode must be 'sort' or 'containment'")
        for name, value in (("row_overlap", row_overlap),
                            ("containment_overlap", containment_overlap)):
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        self.mode = mode
        self.containment_overlap = containment_overlap
        self.infer_columns = infer_columns
        self.row_overlap = row_overlap

    def _initialize_session(self):
        return None

    @staticmethod
    def _bounds(node):
        xs, ys = zip(*node.polygon)
        return min(xs), min(ys), max(xs), max(ys)

    def _columns(self, items):
        if not self.infer_columns or not items:
            return [items]
        intervals = sorted((box[0], box[2]) for _, box in items)
        right = intervals[0][1]
        cuts = []
        for left, end in intervals[1:]:
            if left > right:
                cuts.append((left + right) / 2)
            right = max(right, end)
        columns = [[] for _ in range(len(cuts) + 1)]
        for item in items:
            columns[sum(item[1][0] > cut for cut in cuts)].append(item)
        return columns

    def _same_row(self, first, second):
        overlap = min(first[3], second[3]) - max(first[1], second[1])
        height = min(first[3] - first[1], second[3] - second[1])
        return height > 0 and overlap / height >= self.row_overlap

    def _geometry_order(self, children):
        items = [(node, self._bounds(node)) for node in children if node.polygon]
        others = [node for node in children if not node.polygon]
        ordered = []
        # Existing blocks may themselves represent separate columns. Use the
        # same geometry rule before and after creating column containers.
        for column in self._columns(items):
            rows = []
            for item in sorted(column, key=lambda item: (item[1][1], item[1][0])):
                # Compare with every row member to avoid chaining unrelated rows
                # through an unusually tall region.
                row = next((row for row in rows
                            if all(self._same_row(item[1], member[1]) for member in row)), None)
                if row is None:
                    rows.append([item])
                else:
                    row.append(item)
            for row in rows:
                ordered.extend(node for node, _ in sorted(row, key=lambda item: item[1][0]))
        return ordered + others

    def _sort(self, parent):
        children = list(parent.children)
        if isinstance(parent, Table) and children and all(
                isinstance(node, Cell) and node.row is not None and node.column is not None
                for node in children):
            parent.children = sorted(children, key=lambda node: (node.row, node.column))
        else:
            parent.children = self._geometry_order(children)
        for child in parent.children:
            self._sort(child)

    def _containment(self, children):
        shapes = {}
        for node in children:
            if node.polygon:
                shape = Polygon(node.polygon)
                if not shape.is_valid:
                    shape = shape.buffer(0)
                if not shape.is_empty and shape.area > 0:
                    shapes[id(node)] = shape
        parents = {}
        for child in children:
            if child.type == "block":
                continue
            shape = shapes.get(id(child))
            if shape is None:
                continue
            candidates = []
            for index, parent in enumerate(children):
                if parent is child or not isinstance(parent, TextRegion):
                    continue
                if parent.type != "block" and not (
                        parent.type == "line" and isinstance(child, TextRegion)
                        and child.type == "word"):
                    continue
                container = shapes.get(id(parent))
                if container is None:
                    continue
                coverage = container.intersection(shape).area / shape.area
                if coverage >= self.containment_overlap:
                    # Prefer word -> line -> block even when a larger block
                    # covers more of a word than an eligible line does.
                    priority = 0 if parent.type == "line" else 1
                    candidates.append((priority, -coverage, container.area, index, parent))
            if candidates:
                parents[id(child)] = min(candidates, key=lambda item: item[:4])[4]
        return parents

    def predict(self, page, image=None):
        """Organize the document tree in place.

        Parameters
        ----------
        page : Page
            Document tree to organize.
        image : optional
            Accepted for Pipeline compatibility; not used.

        Returns
        -------
        Page
            The supplied Page with updated child order and, in containment
            mode, inferred parent links and optional column blocks.
        """
        if not isinstance(page, Page):
            raise TypeError("ReadingSorting requires the document-tree Page")
        page.validate_tree()
        children = list(page.children)
        parents = self._containment(children)
        virtual = {id(node): [] for node in children}
        roots = []
        for node in children:
            parent = parents.get(id(node))
            (roots if parent is None else virtual[id(parent)]).append(node)

        ordered = []
        def visit(nodes):
            for node in self._geometry_order(nodes):
                ordered.append(node)
                visit(virtual[id(node)])
        visit(roots)
        if self.mode == "containment":
            for node in children:
                if id(node) in parents:
                    node.parent = parents[id(node)]
            self._sort(page)
            if self.infer_columns and not any(
                    isinstance(node, TextRegion) and node.type == "block"
                    for node in children):
                self._create_column_blocks(page)
        else:
            page.children = ordered
            for child in page.children:
                self._sort(child)
        page.validate_tree()
        return page

    def _create_column_blocks(self, page):
        positioned = [(node, self._bounds(node)) for node in page.children if node.polygon]
        others = [node for node in page.children if not node.polygon]
        blocks = []
        for column in self._columns(positioned):
            if not column:
                continue
            x0 = min(box[0] for _, box in column)
            y0 = min(box[1] for _, box in column)
            x1 = max(box[2] for _, box in column)
            y1 = max(box[3] for _, box in column)
            block = TextRegion(type="block", polygon=[(x0,y0),(x1,y0),(x1,y1),(x0,y1)])
            block.children = self._geometry_order([node for node, _ in column])
            blocks.append(block)
        page.children = blocks + others
