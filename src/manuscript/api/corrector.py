from abc import abstractmethod
from typing import Any, Optional

from manuscript.data import Page, TextRegion
from anytree import PreOrderIter

from .base import BaseArtifactModel


class BaseCorrector(BaseArtifactModel):
    """Base class for text correction in document-tree Pages.

    correction_levels selects word, line or block text; the default is word.
    _copy_page preserves the input tree, and _text_regions yields nonempty
    TextRegions at those levels. Subclasses implement predict() and return a Page.
    """

    correction_levels = ("word",)

    def __init__(self, *args, correction_levels=("word",), **kwargs):
        self._configure_correction_levels(correction_levels)
        super().__init__(*args, **kwargs)

    def _configure_correction_levels(self, levels):
        if (isinstance(levels, str) or not isinstance(levels, (list, tuple))
                or not levels or any(level not in ("word", "line", "block") for level in levels)):
            raise ValueError("correction_levels must be a nonempty list of word, line or block")
        self.correction_levels = tuple(dict.fromkeys(levels))

    @staticmethod
    def _copy_page(page):
        if not isinstance(page, Page):
            raise TypeError("Corrector requires the document-tree Page")
        return page.clone()

    def _text_regions(self, page):
        """Visit nonempty text at explicitly selected structural levels."""
        return (node for node in PreOrderIter(page)
                if isinstance(node, TextRegion) and node.type in self.correction_levels and node.text)

    @abstractmethod
    def predict(
        self,
        page: Page,
        image: Optional[Any] = None,
        **kwargs: Any,
    ) -> Page: ...


__all__ = ["BaseCorrector"]
