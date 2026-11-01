"""Per-call timing state, including failed and visualized calls."""

from unittest.mock import Mock

import numpy as np
import pytest

from manuscript import Pipeline
from manuscript.data import Page, TextRegion
import manuscript._pipeline as module


class Detector:
    def predict(self, image):
        page = Page(image_width=20, image_height=10)
        page.children = [TextRegion(type="word")]
        return page


class Recognizer:
    def predict(self, page, image=None):
        page.children[0].text = "test"
        return page


def test_timings_reset_between_calls_and_return_a_copy():
    pipeline = Pipeline(detector=Detector(), layout=None, recognizer=Recognizer())
    image = np.zeros((10, 20, 3), np.uint8)
    pipeline.predict(image)
    timings = pipeline.last_timings
    assert set(timings) == {"detector", "recognizer", "load_image", "total"}
    assert all(value >= 0 for value in timings.values())
    assert timings["total"] >= sum(
        timings[key] for key in ("detector", "recognizer", "load_image")
    )
    timings.clear()
    assert pipeline.last_timings
    pipeline.recognizer = None
    pipeline.predict(image)
    assert set(pipeline.last_timings) == {"detector", "total"}
    assert pipeline.last_recognition_page is None


def test_visualization_failure_has_no_completed_total(monkeypatch):
    pipeline = Pipeline(detector=Detector(), layout=None, recognizer=None)
    image = np.zeros((10, 20, 3), np.uint8)
    pipeline.predict(image)
    monkeypatch.setattr(
        module, "visualize_page", Mock(side_effect=RuntimeError("render failed"))
    )
    with pytest.raises(RuntimeError, match="render failed"):
        pipeline.predict(image, vis=True)
    assert set(pipeline.last_timings) == {"detector", "load_image"}
    assert pipeline.last_detection_page is not None


def test_invalid_layout_position_fails_before_default_models_are_created(monkeypatch):
    detector = Mock(side_effect=AssertionError("must not initialize default detector"))
    recognizer = Mock(
        side_effect=AssertionError("must not initialize default recognizer")
    )
    monkeypatch.setattr(module, "YOLO", detector)
    monkeypatch.setattr(module, "TRBA", recognizer)
    with pytest.raises(ValueError, match="layout_after"):
        Pipeline(layout_after="invalid")
    detector.assert_not_called()
    recognizer.assert_not_called()


def test_profile_reports_completed_visualization_and_final_total(capsys):
    pipeline = Pipeline(detector=Detector(), layout=None, recognizer=None)
    pipeline.predict(np.zeros((10, 20, 3), np.uint8), vis=True, profile=True)
    lines = capsys.readouterr().out.splitlines()
    assert f"Visualization: {pipeline.last_timings['visualization']:.3f}s" in lines
    assert lines[-1] == f"Pipeline total: {pipeline.last_timings['total']:.3f}s"
