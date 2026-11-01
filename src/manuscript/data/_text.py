"""Plain-text serialization of document trees."""
from collections.abc import Mapping

from .tree import Page, Table, Cell


DEFAULT_SEPARATORS = {'word': ' ', 'line': '\n', 'block': '\n\n', 'table': '\n\n'}
DEFAULT_PRIORITIES = {'word': 0, 'line': 1, 'block': 2, 'table': 2}


def get_text(page, *, format='plain', separators=None, separator_priorities=None):
    """Serialize text in child order without modifying the Page.

    The stronger of two neighboring element boundaries separates their texts.
    Custom separator priorities resolve mixed types (higher wins). Text from
    descendants takes precedence over parent text. Flat words remain one line;
    geometry is not used to infer missing line boundaries.
    Tables use row-major TSV with empty and spanned slots preserved. Whitespace
    inside cells becomes spaces, so internal newlines cannot break table rows.
    Only plain format is currently supported.
    """
    if not isinstance(page, Page):
        raise TypeError('get_text requires Page')
    if format != 'plain':
        raise ValueError("Only format='plain' is currently supported")
    boundaries = dict(DEFAULT_SEPARATORS)
    priorities = dict(DEFAULT_PRIORITIES)
    for values, target, kind in ((separators, boundaries, str),
                                 (separator_priorities, priorities, int)):
        if values is not None:
            if not isinstance(values, Mapping) or any(
                    not isinstance(key, str) or not isinstance(value, kind)
                    for key, value in values.items()):
                raise ValueError('Separators must map types to strings; priorities to integers')
            target.update(values)
    page.validate_tree()

    def join(parts):
        parts = [(text, kind) for text, kind in parts if text]
        if not parts:
            return ''
        output = [parts[0][0]]
        for previous, current in zip(parts, parts[1:]):
            kind = max((previous[1], current[1]), key=lambda kind: priorities.get(kind, 1))
            output.extend((boundaries.get(kind, '\n'), current[0]))
        return ''.join(output)

    def table_text(table):
        cells = [child for child in table.children if isinstance(child, Cell)]
        if not cells:
            return join([read(child) for child in table.children])
        rows = table.rows or max(cell.row + cell.row_span for cell in cells)
        columns = table.columns or max(cell.column + cell.column_span for cell in cells)
        values = {(cell.row, cell.column): ' '.join(read(cell)[0].split()) for cell in cells}
        if not any(values.values()):
            return ''
        return '\n'.join('\t'.join(values.get((row, column), '') for column in range(columns))
                         for row in range(rows))

    def read(node):
        if isinstance(node, Table):
            return table_text(node), node.type
        child_text = join([read(child) for child in node.children])
        own_text = getattr(node, 'text', None)
        return child_text or (own_text.strip() if own_text else ''), node.type

    return read(page)[0]
