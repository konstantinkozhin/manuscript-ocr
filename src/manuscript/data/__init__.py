"""
Data structures for manuscript OCR.

This package contains the core data structures used to represent OCR results
throughout the manuscript-ocr library.
"""

from .structures import TextSpan, Line, Block, Page

__all__ = ["TextSpan", "Line", "Block", "Page"]
