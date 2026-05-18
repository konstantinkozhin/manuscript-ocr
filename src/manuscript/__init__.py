from .utils import read_image, create_page_from_text, create_page_from_image
from .data import TextSpan, Line, Block, Page
from .layouts import SimpleSorting

__all__ = [
    "Pipeline",
    "visualize_page",
    "read_image",
    "create_page_from_text",
    "create_page_from_image",
    "TextSpan",
    "Line",
    "Block",
    "Page",
    "CharLM",
    "SimpleSorting",
]


def __getattr__(name):
    if name == "Pipeline":
        from ._pipeline import Pipeline

        return Pipeline

    if name == "CharLM":
        from .correctors import CharLM

        return CharLM

    if name == "visualize_page":
        from .utils import visualize_page

        return visualize_page

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
