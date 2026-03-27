from abc import abstractmethod
from typing import Any, Optional

from manuscript.data import Page

from .base import BaseArtifactModel


class BaseLayout(BaseArtifactModel):
    """Public layout base with ``predict(page, image=None, **kwargs) -> Page``."""

    @abstractmethod
    def predict(
        self,
        page: Page,
        image: Optional[Any] = None,
        **kwargs: Any,
    ) -> Page: ...


__all__ = ["BaseLayout"]
