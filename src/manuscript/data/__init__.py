"""
Data structures for manuscript OCR.

This package contains the core data structures used to represent OCR results
throughout the manuscript-ocr library.
"""

from .structures import TextSpan, Line, Block, Page

__all__ = ["TextSpan", "Line", "Block", "Page"]


def __getattr__(name: str):
    if name == "Word":
        return TextSpan
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
