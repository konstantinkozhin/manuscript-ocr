import inspect
from typing import Any, Dict

from manuscript.data import Page


def filter_callable_kwargs(func: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return only kwargs accepted by ``func`` unless it has ``**kwargs``."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return {}

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs

    return {k: v for k, v in kwargs.items() if k in params}


def filter_predict_kwargs(stage: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return kwargs accepted by ``stage.predict``."""
    return filter_callable_kwargs(stage.predict, kwargs)


def accepts_predict_kwarg(stage: Any, name: str) -> bool:
    """Check whether ``stage.predict`` accepts a named kwarg."""
    return name in filter_predict_kwargs(stage, {name: object()})


def call_page_stage(stage: Any, page: Page, **kwargs: Any) -> Page:
    """Call ``stage.predict`` with kwargs filtered by signature."""
    return stage.predict(page, **filter_predict_kwargs(stage, kwargs))


__all__ = [
    "accepts_predict_kwarg",
    "call_page_stage",
    "filter_callable_kwargs",
    "filter_predict_kwargs",
]
