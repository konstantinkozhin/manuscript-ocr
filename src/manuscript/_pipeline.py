import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
from PIL import Image

from .api._page_helpers import (
    accepts_predict_kwarg,
    call_page_stage,
    filter_predict_kwargs,
)
from .api.protocols import (
    CorrectorProtocol,
    DetectorProtocol,
    LayoutProtocol,
    RecognizerProtocol,
)
from .data import Page
from .detectors import YOLO
from .layouts import SimpleSorting
from .recognizers import TRBA
from .utils import read_image, visualize_page

_DEFAULT = object()
_VALID_LAYOUT_AFTER = {"detector", "recognizer", "corrector"}


class Pipeline:
    """
    High-level OCR pipeline with configurable stage ordering.

    Default pipeline:
    ``detector -> layout -> recognizer``.
    ``corrector`` is optional and disabled by default.
    """

    def __init__(
        self,
        detector: DetectorProtocol = _DEFAULT,
        layout: Optional[LayoutProtocol] = _DEFAULT,
        recognizer: Optional[RecognizerProtocol] = _DEFAULT,
        corrector: Optional[CorrectorProtocol] = None,
        layout_after: str = "detector",
    ):
        """
        Initialize OCR pipeline.

        Parameters
        ----------
        detector : object, optional
            Detector instance with ``predict(image) -> Page``.
            If omitted, default ``YOLO(weights="yolo26x_obb_text_g1")`` is used.
            Detector cannot be disabled.
        layout : object or None, optional
            Layout model instance with ``predict(page, image=None) -> Page``.
            If omitted, default ``SimpleSorting()`` is used.
            Pass ``None`` to disable layout stage.
        recognizer : object or None, optional
            Recognizer instance with ``predict(page, image=None, ...) -> Page``.
            If omitted, default ``TRBA(weights="trba_lite_g2")`` is used.
            Pass ``None`` to disable recognition stage.
        corrector : object or None, optional
            Corrector instance with ``predict(page, image=None) -> Page``.
            Default is ``None`` (disabled).
        layout_after : {"detector", "recognizer", "corrector"}, optional
            Slot where layout stage is executed. Default is ``"detector"``.
        """
        if detector is _DEFAULT:
            self.detector = YOLO(weights="yolo26x_obb_text_g1")
        elif detector is None:
            raise ValueError("detector cannot be None")
        else:
            self.detector = detector

        if layout is _DEFAULT:
            self.layout = SimpleSorting()
        else:
            self.layout = layout

        if recognizer is _DEFAULT:
            self.recognizer = TRBA(weights="trba_lite_g2")
        else:
            self.recognizer = recognizer

        self.corrector = corrector

        if layout_after not in _VALID_LAYOUT_AFTER:
            raise ValueError(
                f"layout_after must be one of {_VALID_LAYOUT_AFTER}, got: {layout_after}"
            )
        self.layout_after = layout_after

        self._last_detection_page: Optional[Page] = None
        self._last_layout_page: Optional[Page] = None
        self._last_recognition_page: Optional[Page] = None
        self._last_correction_page: Optional[Page] = None

    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        vis: bool = False,
        profile: bool = False,
    ) -> Union[Dict[str, Page], tuple]:
        """
        Run pipeline on a single image.

        Parameters
        ----------
        image : str, Path, numpy.ndarray, or PIL.Image
            Input image.
        vis : bool, optional
            If True, returns visualization image together with result.
        profile : bool, optional
            If True, prints timing per stage.
        """
        start_time = time.time()

        self._last_detection_page = None
        self._last_layout_page = None
        self._last_recognition_page = None
        self._last_correction_page = None

        image_array_cache: Optional[np.ndarray] = None

        def get_image_array() -> np.ndarray:
            nonlocal image_array_cache
            if image_array_cache is None:
                t0 = time.time()
                image_array_cache = read_image(image)
                if profile:
                    print(f"Load image: {time.time() - t0:.3f}s")
            return image_array_cache

        def run_layout_slot(anchor: str, page: Page) -> Page:
            if self.layout is None or self.layout_after != anchor:
                return page

            t0 = time.time()
            layout_kwargs: Dict[str, Any] = {}
            if accepts_predict_kwarg(self.layout, "image"):
                layout_kwargs["image"] = get_image_array()
            page = call_page_stage(self.layout, page, **layout_kwargs)
            self._last_layout_page = page.model_copy(deep=True)
            if profile:
                print(f"Layout ({anchor}): {time.time() - t0:.3f}s")
            return page

        # Detection
        t0 = time.time()
        page = self.detector.predict(image)
        self._last_detection_page = page.model_copy(deep=True)
        if profile:
            print(f"Detection: {time.time() - t0:.3f}s")

        # After detector slot
        page = run_layout_slot("detector", page)

        # Recognition slot
        if self.recognizer is not None:
            t0 = time.time()
            recognizer_kwargs = filter_predict_kwargs(
                self.recognizer,
                {
                    "batch_size": 32,
                },
            )
            if accepts_predict_kwarg(self.recognizer, "image"):
                recognizer_kwargs["image"] = get_image_array()
            page = call_page_stage(self.recognizer, page, **recognizer_kwargs)
            self._last_recognition_page = page.model_copy(deep=True)
            if profile:
                print(f"Recognition: {time.time() - t0:.3f}s")

        # After recognizer slot
        page = run_layout_slot("recognizer", page)

        # Correction slot
        if self.corrector is not None:
            t0 = time.time()
            corrector_kwargs: Dict[str, Any] = {}
            if accepts_predict_kwarg(self.corrector, "image"):
                corrector_kwargs["image"] = get_image_array()
            page = call_page_stage(self.corrector, page, **corrector_kwargs)
            self._last_correction_page = page.model_copy(deep=True)
            if profile:
                print(f"Correction: {time.time() - t0:.3f}s")

        # After corrector slot
        page = run_layout_slot("corrector", page)

        if profile:
            print(f"Pipeline total: {time.time() - start_time:.3f}s")

        result: Dict[str, Page] = {"page": page}

        if not vis:
            return result

        image_array = get_image_array()
        pil_img = image if isinstance(image, Image.Image) else Image.fromarray(image_array)
        vis_img = visualize_page(pil_img, page, show_order=True)
        return result, vis_img

    def get_text(self, page: Page) -> str:
        """
        Extract plain text from ``Page`` object.
        """
        lines = []
        for block in page.blocks:
            for line in block.lines:
                texts = [span.text for span in line.text_spans if span.text]
                if texts:
                    lines.append(" ".join(texts))
        return "\n".join(lines)

    @property
    def last_detection_page(self) -> Optional[Page]:
        return self._last_detection_page

    @property
    def last_layout_page(self) -> Optional[Page]:
        return self._last_layout_page

    @property
    def last_recognition_page(self) -> Optional[Page]:
        return self._last_recognition_page

    @property
    def last_correction_page(self) -> Optional[Page]:
        return self._last_correction_page
