"""Builders for actual document-tree objects used by stage tests."""

from manuscript.data import Page, TextRegion


def make_region(type, children=(), **fields):
    region = TextRegion(type=type, **fields)
    region.children = children
    return region


def make_page(children=(), **fields):
    page = Page(
        image_width=fields.pop("image_width", 1024),
        image_height=fields.pop("image_height", 1280),
        **fields,
    )
    page.children = children
    return page


def sibling_index(region):
    return (
        next(
            (i for i, child in enumerate(region.parent.children) if child is region),
            None,
        )
        if region.parent
        else None
    )
