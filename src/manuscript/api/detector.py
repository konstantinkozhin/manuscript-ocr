from abc import abstractmethod
from typing import Any

from manuscript.data import Page

from .base import BaseArtifactModel


class BaseDetector(BaseArtifactModel):
    """Public detector base with ``predict(image, **kwargs) -> Page`` contract."""

    @abstractmethod
    def predict(self, image: Any, **kwargs: Any) -> Page: ...


__all__ = ["BaseDetector"]
