from typing import Any, Dict, Optional, Protocol, Union, runtime_checkable

from manuscript.data import Page


@runtime_checkable
class DetectorProtocol(Protocol):
    def predict(self, image: Any, **kwargs: Any) -> Union[Page, Dict[str, Any]]: ...


@runtime_checkable
class RecognizerProtocol(Protocol):
    def predict(
        self,
        page: Page,
        image: Optional[Any] = None,
        **kwargs: Any,
    ) -> Page: ...


@runtime_checkable
class LayoutProtocol(Protocol):
    def predict(
        self,
        page: Page,
        image: Optional[Any] = None,
        **kwargs: Any,
    ) -> Page: ...


@runtime_checkable
class CorrectorProtocol(Protocol):
    def predict(
        self,
        page: Page,
        image: Optional[Any] = None,
        **kwargs: Any,
    ) -> Page: ...


__all__ = [
    "CorrectorProtocol",
    "DetectorProtocol",
    "LayoutProtocol",
    "RecognizerProtocol",
]
