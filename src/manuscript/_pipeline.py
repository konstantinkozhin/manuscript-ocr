import time
import warnings
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
from .data import Page, TextRegion
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
        if layout_after not in _VALID_LAYOUT_AFTER:
            raise ValueError(
                f"layout_after must be one of {_VALID_LAYOUT_AFTER}, got: {layout_after}"
            )
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

        self.layout_after = layout_after

        self._last_detection_page: Optional[Page] = None
        self._last_layout_page: Optional[Page] = None
        self._last_recognition_page: Optional[Page] = None
        self._last_correction_page: Optional[Page] = None
        self._last_timings: Dict[str, float] = {}

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

        Compatible detectors are asked for compact raw results without masks
        or original tensors. Layout/recognizer/corrector stages accepting
        detector_result receive that same dictionary for this call only.
        """
        start_time = time.perf_counter()
        self._last_timings = {}

        self._last_detection_page = None
        self._last_layout_page = None
        self._last_recognition_page = None
        self._last_correction_page = None

        image_array_cache: Optional[np.ndarray] = None

        def get_image_array() -> np.ndarray:
            nonlocal image_array_cache
            if image_array_cache is None:
                t0 = time.perf_counter()
                image_array_cache = read_image(image)
                self._last_timings["load_image"] = time.perf_counter() - t0
                if profile:
                    print(f"Load image: {time.perf_counter() - t0:.3f}s")
            return image_array_cache

        def run_layout_slot(anchor: str, page: Page) -> Page:
            if self.layout is None or self.layout_after != anchor:
                return page

            t0 = time.perf_counter()
            load_before = self._last_timings.get("load_image", 0.0)
            layout_kwargs: Dict[str, Any] = {}
            if detector_result is not None:
                layout_kwargs["detector_result"] = detector_result
            if accepts_predict_kwarg(self.layout, "image"):
                layout_kwargs["image"] = get_image_array()
            page = call_page_stage(self.layout, page, **layout_kwargs)
            self._last_layout_page = page.clone()
            self._last_timings["layout"] = (time.perf_counter() - t0
                - (self._last_timings.get("load_image", 0.0) - load_before))
            if profile:
                print(f"Layout ({anchor}): {self._last_timings['layout']:.3f}s")
            return page

        # Detection
        t0 = time.perf_counter()
        detection = self.detector.predict(image, **filter_predict_kwargs(self.detector, {
            "return_raw": True, "return_masks": False, "return_outputs": False,
        }))
        detector_result = detection if isinstance(detection, dict) else None
        page = detection.get("page") if detector_result is not None else detection
        if not isinstance(page, Page):
            raise TypeError("Detector must return Page or a result dictionary containing Page")
        page.validate_tree()
        if isinstance(image, (str, Path)) and not page.file_name:
            page.file_name = Path(image).name
        self._last_detection_page = page.clone()
        self._last_timings["detector"] = time.perf_counter() - t0
        if profile:
            print(f"Detection: {time.perf_counter() - t0:.3f}s")

        # After detector slot
        page = run_layout_slot("detector", page)

        # Recognition slot
        if self.recognizer is not None:
            t0 = time.perf_counter()
            load_before = self._last_timings.get("load_image", 0.0)
            recognizer_kwargs = filter_predict_kwargs(
                self.recognizer,
                {
                    "batch_size": getattr(self.recognizer, "batch_size", 32),
                    **({"detector_result": detector_result} if detector_result is not None else {}),
                },
            )
            if accepts_predict_kwarg(self.recognizer, "image"):
                recognizer_kwargs["image"] = get_image_array()
            page = call_page_stage(self.recognizer, page, **recognizer_kwargs)
            self._last_recognition_page = page.clone()
            self._last_timings["recognizer"] = (time.perf_counter() - t0
                - (self._last_timings.get("load_image", 0.0) - load_before))
            if profile:
                print(f"Recognition: {self._last_timings['recognizer']:.3f}s")

        # After recognizer slot
        page = run_layout_slot("recognizer", page)

        # Correction slot
        if self.corrector is not None:
            t0 = time.perf_counter()
            load_before = self._last_timings.get("load_image", 0.0)
            corrector_kwargs: Dict[str, Any] = {}
            if detector_result is not None:
                corrector_kwargs["detector_result"] = detector_result
            if accepts_predict_kwarg(self.corrector, "image"):
                corrector_kwargs["image"] = get_image_array()
            page = call_page_stage(self.corrector, page, **corrector_kwargs)
            self._last_correction_page = page.clone()
            self._last_timings["corrector"] = (time.perf_counter() - t0
                - (self._last_timings.get("load_image", 0.0) - load_before))
            if profile:
                print(f"Correction: {self._last_timings['corrector']:.3f}s")

        # After corrector slot
        page = run_layout_slot("corrector", page)
        # Only Page snapshots persist between calls; no detector arrays are cached.
        detector_result = detection = None

        if any(isinstance(node, TextRegion) and node.text
               and any(isinstance(parent, TextRegion) and parent.text
                       for parent in node.ancestors)
               for node in page.descendants):
            warnings.warn(
                "Recognized text is present on both parent and child regions; "
                "multiple text levels may contain duplicate recognition results.",
                UserWarning, stacklevel=2,
            )

        result: Dict[str, Page] = {"page": page}

        if not vis:
            self._last_timings["total"] = time.perf_counter() - start_time
            if profile:
                print(f"Pipeline total: {self._last_timings['total']:.3f}s")
            return result

        image_array = get_image_array()
        pil_img = image if isinstance(image, Image.Image) else Image.fromarray(image_array)
        t0 = time.perf_counter()
        vis_img = visualize_page(pil_img, page)
        self._last_timings["visualization"] = time.perf_counter() - t0
        self._last_timings["total"] = time.perf_counter() - start_time
        if profile:
            print(f"Visualization: {self._last_timings['visualization']:.3f}s")
            print(f"Pipeline total: {self._last_timings['total']:.3f}s")
        return result, vis_img

    def get_text(self, page: Page, format="plain", *, separators=None, separator_priorities=None) -> str:
        """Delegate text serialization to Page.get_text."""
        return page.get_text(format=format, separators=separators,
                             separator_priorities=separator_priorities)

    @property
    def last_timings(self) -> Dict[str, float]:
        """Return elapsed seconds for the last predict call as a dictionary copy.

        Keys: detector, layout, recognizer, corrector, load_image, visualization
        and total. Only executed stages are present. Stage times include tree
        validation/snapshots and lazy model initialization. Explicit image loading
        by Pipeline is recorded separately and subtracted from page-stage times.
        Loading performed inside a model remains part of that stage's time.
        Total includes orchestration and optional visualization. On failure,
        completed stage timings remain available without a total entry.
        """
        return dict(self._last_timings)

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
