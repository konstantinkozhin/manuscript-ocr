import time
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Union, TYPE_CHECKING

import numpy as np
from PIL import Image

from .data import Page
from .detectors import EAST
from .recognizers import TRBA
from .utils import read_image, visualize_page

if TYPE_CHECKING:
    from .api.base import BaseModel as BaseCorrector


class Pipeline:
    """
    High-level OCR pipeline combining text detection, recognition, and correction.

    The Pipeline class orchestrates EAST detector, TRBA recognizer, and optional
    text corrector to perform complete OCR workflow:
    detection -> recognition -> correction -> result merging.

    Attributes
    ----------
    detector : EAST
        Text detector instance
    recognizer : TRBA
        Text recognizer instance
    corrector : BaseCorrector, optional
        Text corrector instance (None to skip correction)
    min_text_size : int
        Minimum text box size in pixels (width and height)

    Examples
    --------
    Create pipeline with default models:

    >>> from manuscript import Pipeline
    >>> pipeline = Pipeline()
    >>> result = pipeline.predict("document.jpg")
    >>> text = pipeline.get_text(result["page"])
    >>> print(text)

    Create pipeline with custom models:

    >>> from manuscript import Pipeline
    >>> from manuscript.detectors import EAST
    >>> from manuscript.recognizers import TRBA
    >>> detector = EAST(weights="east_50_g1", score_thresh=0.8)
    >>> recognizer = TRBA(weights="trba_lite_g1", device="cuda")
    >>> pipeline = Pipeline(detector=detector, recognizer=recognizer)

    Create pipeline with text correction:

    >>> from manuscript import Pipeline
    >>> from manuscript.correctors import CharLM
    >>> corrector = CharLM()
    >>> pipeline = Pipeline(corrector=corrector)
    """

    def __init__(
        self,
        detector: Optional[EAST] = None,
        recognizer: Optional[TRBA] = None,
        corrector: Optional["BaseCorrector"] = None,
        min_text_size: int = 5,
    ):
        """
        Initialize OCR pipeline.

        Parameters
        ----------
        detector : EAST, optional
            Text detector instance. If None, creates default EAST detector.
        recognizer : TRBA, optional
            Text recognizer instance. If None, creates default TRBA recognizer.
        corrector : BaseCorrector, optional
            Text corrector instance. If None, no text correction is applied.
            The corrector receives a Page object after recognition and returns
            a corrected Page object.
        min_text_size : int, optional
            Minimum text size in pixels. Boxes smaller than this will be
            filtered out before recognition. Default is 5.
        """
        self.detector = detector if detector is not None else EAST()
        self.recognizer = recognizer if recognizer is not None else TRBA()
        self.corrector = corrector
        self.min_text_size = min_text_size

        self._last_detection_page: Optional[Page] = None
        self._last_recognition_page: Optional[Page] = None
        self._last_correction_page: Optional[Page] = None

    def predict(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        recognize_text: bool = True,
        vis: bool = False,
        profile: bool = False,
    ) -> Union[Dict, tuple]:
        """
        Run OCR pipeline on a single image.

        Parameters
        ----------
        image : str, Path, numpy.ndarray, or PIL.Image
            Input image. Can be:
            - Path to image file (str or Path)
            - RGB numpy array with shape (H, W, 3) in uint8
            - PIL Image object
        recognize_text : bool, optional
            If True, performs both detection and recognition.
            If False, performs only detection. Default is True.
        vis : bool, optional
            If True, returns visualization image along with results.
            Default is False.
        profile : bool, optional
            If True, prints timing information for each pipeline stage.
            Default is False.

        Returns
        -------
        dict or tuple
            If vis=False:
                dict with keys:
                - "page" : Page object with detection/recognition results

            If vis=True:
                tuple of (result_dict, vis_image)

        Examples
        --------
        Basic usage:

        >>> pipeline = Pipeline()
        >>> result = pipeline.predict("document.jpg")
        >>> page = result["page"]
        >>> print(page.blocks[0].lines[0].words[0].text)

        Detection only:

        >>> result = pipeline.predict("document.jpg", recognize_text=False)
        >>> # Words will have polygon and detection_confidence but no text

        With visualization:

        >>> result, vis_img = pipeline.predict("document.jpg", vis=True)
        >>> vis_img.show()

        With profiling:

        >>> result = pipeline.predict("document.jpg", profile=True)
        # Prints timing for each stage
        """
        start_time = time.time()

        # ---- DETECTION ----
        t0 = time.time()
        detection_result = self.detector.predict(
            image, return_maps=False, sort_reading_order=True
        )
        page: Page = detection_result["page"]
        self._last_detection_page = page.model_copy(deep=True)

        if profile:
            print(f"Detection: {time.time() - t0:.3f}s")

        # ---- If recognition not needed ----
        if not recognize_text:
            result = {"page": page}

            if vis:
                img_array = read_image(image)
                pil_img = (
                    image
                    if isinstance(image, Image.Image)
                    else Image.fromarray(img_array)
                )
                vis_img = visualize_page(pil_img, page, show_order=False)
                return result, vis_img

            return result

        # ---- LOAD IMAGE FOR RECOGNITION ----
        t0 = time.time()
        image_array = read_image(image)
        if profile:
            print(f"Load image for recognition: {time.time() - t0:.3f}s")

        # ---- RECOGNITION ----
        t0 = time.time()
        recognizer_kwargs = {"image": image_array}
        try:
            predict_params = inspect.signature(self.recognizer.predict).parameters
            if "batch_size" in predict_params:
                recognizer_kwargs["batch_size"] = 32
            if "min_text_size" in predict_params:
                recognizer_kwargs["min_text_size"] = self.min_text_size
        except (TypeError, ValueError):
            pass

        page = self.recognizer.predict(page, **recognizer_kwargs)
        if profile:
            print(f"Recognition: {time.time() - t0:.3f}s")

        self._last_recognition_page = page.model_copy(deep=True)

        # ---- CORRECTION ----
        if self.corrector is not None:
            t0 = time.time()
            page = self.corrector.predict(page)
            self._last_correction_page = page.model_copy(deep=True)
            if profile:
                print(f"Correction: {time.time() - t0:.3f}s")
        else:
            self._last_correction_page = None

        if profile:
            print(f"Pipeline total: {time.time() - start_time:.3f}s")

        result = {"page": page}

        if vis:
            pil_img = (
                image
                if isinstance(image, Image.Image)
                else Image.fromarray(image_array)
            )
            vis_img = visualize_page(pil_img, page, show_order=True)
            return result, vis_img

        return result

    def get_text(self, page: Page) -> str:
        """
        Extract plain text from Page object.

        Parameters
        ----------
        page : Page
            Page object with recognition results.

        Returns
        -------
        str
            Extracted text with lines separated by newlines.

        Examples
        --------
        >>> pipeline = Pipeline()
        >>> result = pipeline.predict("document.jpg")
        >>> text = pipeline.get_text(result["page"])
        >>> print(text)
        """
        lines = []
        for block in page.blocks:
            for line in block.lines:
                # Extract text from words in the line
                texts = [w.text for w in line.words if w.text]
                if texts:
                    lines.append(" ".join(texts))
        return "\n".join(lines)

    @property
    def last_detection_page(self) -> Optional[Page]:
        return self._last_detection_page

    @property
    def last_recognition_page(self) -> Optional[Page]:
        return self._last_recognition_page

    @property
    def last_correction_page(self) -> Optional[Page]:
        return self._last_correction_page

