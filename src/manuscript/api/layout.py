from abc import abstractmethod
from typing import Any, Optional

from manuscript.data import Page

from .base import BaseArtifactModel


class BaseLayout(BaseArtifactModel):
    """Base class for organizing document-tree Pages.

    Subclasses implement predict(page, image=None, **kwargs) and return a Page
    with the desired containment and reading order. Child sequence defines order.
    Layouts may accept detector_result to reuse detection geometry and classes
    supplied by Pipeline without running the detector again.
    """

    @abstractmethod
    def predict(
        self,
        page: Page,
        image: Optional[Any] = None,
        **kwargs: Any,
    ) -> Page: ...


__all__ = ["BaseLayout"]
