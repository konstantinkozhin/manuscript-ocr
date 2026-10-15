"""Explicit grouping of sibling text regions without losing their data."""
from typing import Callable, Optional, Sequence, Union
import math

import numpy as np
from manuscript.data import TextRegion


def group_regions(
    regions: Sequence[TextRegion], *, type: str,
    geometry: Optional[Union[str, Callable]] = 'bbox', concavity: float = 0.3,
) -> TextRegion:
    """Create a higher-level text parent and move existing siblings into it.

    Defaults to a bounding rectangle. Other geometry modes are convex_hull,
    concave_hull and None. A callable receives copied polygon lists and returns
    a polygon or None. For concave_hull, concavity is Shapely's ratio: 0 gives
    the most concave outline, 1 the convex hull (requires Shapely 2 / GEOS 3.11).
    Child order follows the input. The group replaces the first selected sibling
    in its parent's list. Detached regions may also be grouped. All validation
    and geometry computation happen before any tree links change.
    """
    regions = list(regions)
    ranks = {'word': 0, 'line': 1, 'block': 2}
    if type not in ('line', 'block'):
        raise ValueError('Grouping target must be line or block')
    if not regions:
        raise ValueError('Cannot group an empty list')
    if len({id(node) for node in regions}) != len(regions):
        raise ValueError('Duplicate regions')
    if any(not isinstance(node, TextRegion) or ranks[node.type] >= ranks[type] for node in regions):
        raise ValueError('Every child must have a lower text level than its group')
    parent = regions[0].parent
    if any(node.parent is not parent for node in regions):
        raise ValueError('Grouped regions must share a parent')
    polygons = [[tuple(point) for point in node.polygon] for node in regions if node.polygon is not None]
    polygon = None
    if geometry is not None:
        if not polygons:
            raise ValueError('Geometry computation requires positioned regions')
        if callable(geometry):
            polygon = geometry(polygons)
        elif geometry == 'bbox':
            points = np.vstack(polygons)
            x0,y0 = points.min(axis=0); x1,y1 = points.max(axis=0)
            polygon = [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
        elif geometry in ('convex_hull', 'concave_hull'):
            from shapely.geometry import MultiPoint
            points = MultiPoint([point for poly in polygons for point in poly])
            if geometry == 'convex_hull':
                hull = points.convex_hull
            else:
                if not math.isfinite(concavity) or not 0 <= concavity <= 1:
                    raise ValueError('concavity must be between 0 and 1')
                import shapely
                if not hasattr(shapely, 'concave_hull'):
                    raise ImportError('concave_hull requires Shapely >=2 and GEOS >=3.11')
                hull = shapely.concave_hull(points, ratio=concavity, allow_holes=False)
            if hull.geom_type != 'Polygon':
                raise ValueError('Hull must be a nondegenerate polygon')
            polygon = list(hull.exterior.coords)[:-1]
            if len(polygon) == 3:
                a,b = polygon[:2]
                polygon.insert(1,((a[0]+b[0])/2,(a[1]+b[1])/2))
        else:
            raise ValueError('Unknown grouping geometry mode')
    group = TextRegion(type=type, polygon=polygon)
    selected = {id(node) for node in regions}
    siblings = list(parent.children) if parent is not None else []
    replacement = []
    inserted = False
    for sibling in siblings:
        if id(sibling) in selected:
            if not inserted:
                replacement.append(group); inserted = True
        else:
            replacement.append(sibling)
    group.children = regions
    if parent is not None:
        parent.children = replacement
    return group
