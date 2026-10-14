import pytest
from manuscript.api._page_helpers import filter_callable_kwargs, call_page_stage
from manuscript.data import Page


def test_signature_filters_unknown_options_and_keeps_kwargs():
    def strict(page, image=None): pass
    def flexible(page, **kwargs): pass
    options = {"image": object(), "batch_size": 2}
    assert filter_callable_kwargs(strict, options) == {"image": options["image"]}
    assert filter_callable_kwargs(flexible, options) == options


def test_uninspectable_signature_has_no_optional_arguments():
    class Callable:
        __signature__ = "invalid"
        def __call__(self): pass
    assert filter_callable_kwargs(Callable(), {"image": 1}) == {}


def test_page_stage_filters_arguments_and_rejects_non_page():
    class Stage:
        def predict(self, page): return page
    page = Page(image_width=10, image_height=10)
    assert call_page_stage(Stage(), page, image=object()) is page
    class InvalidStage:
        def predict(self, page): return {"page": page}
    with pytest.raises(TypeError, match="must return"):
        call_page_stage(InvalidStage(), page)
