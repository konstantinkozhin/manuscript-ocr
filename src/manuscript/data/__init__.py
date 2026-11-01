"""
Data structures for manuscript OCR.

This package contains the core data structures used to represent OCR results
throughout the manuscript-ocr library.
"""

from .tree import Element, Page, TextRegion, Table, Cell, Image, to_coco, from_coco

__all__ = ["Element", "Page", "TextRegion", "Table", "Cell", "Image", "to_coco", "from_coco"]
