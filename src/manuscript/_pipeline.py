import numpy as np
from PIL import Image
from typing import Union, Optional
import cv2
from pathlib import Path
import time
from typing import List, Tuple
from .detectors import (
    EAST,
    visualize_page,
    read_image,
    sort_boxes_reading_order_with_resolutions,
)
from .recognizers import TRBA


class Pipeline:
    def __init__(
        self,
        detector: Optional[EAST] = None,
        recognizer: Optional[TRBA] = None,
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
        min_text_size : int, optional
            Minimum text size in pixels. Default is 5.

        Examples
        --------
        Create pipeline with default models:

        >>> from manuscript import Pipeline
        >>> pipeline = Pipeline()

        Create pipeline with custom models:

        >>> from manuscript import Pipeline
        >>> from manuscript.detectors import EAST
        >>> from manuscript.recognizers import TRBA
        >>> detector = EAST(score_thresh=0.8)
        >>> recognizer = TRBA(device="cuda")
        >>> pipeline = Pipeline(detector=detector, recognizer=recognizer)
        """
        self.detector = detector if detector is not None else EAST()
        self.recognizer = recognizer if recognizer is not None else TRBA()
        self.min_text_size = min_text_size

    def predict(
        self,
        image: Union[str, np.ndarray, Image.Image],
        recognize_text: bool = True,
        vis: bool = False,
        profile: bool = False,
    ):
        start_time = time.time()

        # ---- DETECTION ----
        t0 = time.time()
        det_out = self.detector.predict(image, vis=False, profile=profile)

        if isinstance(det_out, dict):
            detection_result = det_out.get("page")
        elif isinstance(det_out, tuple):
            detection_result = det_out[0]
        else:
            detection_result = det_out

        if detection_result is None:
            raise RuntimeError("Detector did not return a Page result.")

        if profile:
            print(f"Detection: {time.time() - t0:.3f}s")

        # ---- If recognition not needed ----
        if not recognize_text:
            if vis:
                arr = read_image(image)
                pil = image if isinstance(image, Image.Image) else Image.fromarray(arr)
                vis_img = visualize_page(pil, detection_result, show_order=False)
                return detection_result, vis_img
            return detection_result

        # ---- LOAD IMAGE ----
        t0 = time.time()
        image_array = read_image(image)
        if profile:
            print(f"Load image for crops: {time.time() - t0:.3f}s")

        # ---- SORT + EXTRACT ----
        t0 = time.time()
        all_words = []
        word_images = []

        for block in detection_result.blocks:

            boxes = []
            for w in block.words:
                poly = np.array(w.polygon, dtype=np.int32)
                x_min, y_min = np.min(poly, axis=0)
                x_max, y_max = np.max(poly, axis=0)
                boxes.append((x_min, y_min, x_max, y_max))

            sorted_boxes = sort_boxes_reading_order_with_resolutions(boxes)

            new_order = []
            for bx in sorted_boxes:
                for w in block.words:
                    poly = np.array(w.polygon, dtype=np.int32)
                    x_min, y_min = np.min(poly, axis=0)
                    x_max, y_max = np.max(poly, axis=0)
                    if (x_min, y_min, x_max, y_max) == bx:
                        new_order.append(w)
                        break

            block.words = new_order

            for word in block.words:
                poly = np.array(word.polygon, dtype=np.int32)
                x_min, y_min = np.min(poly, axis=0)
                x_max, y_max = np.max(poly, axis=0)

                width = x_max - x_min
                height = y_max - y_min

                if width >= self.min_text_size and height >= self.min_text_size:
                    region_image = self._extract_word_image(image_array, poly)
                    if region_image is not None and region_image.size > 0:
                        all_words.append(word)
                        word_images.append(region_image)

        if profile:
            print(f"Extract {len(word_images)} crops: {time.time() - t0:.3f}s")

        # ---- RECOGNITION ----
        if word_images:
            t0 = time.time()
            recognition_results = self.recognizer.predict(word_images)
            if profile:
                print(f"Recognition: {time.time() - t0:.3f}s")

            for idx, word in enumerate(all_words):
                result = recognition_results[idx]

                if isinstance(result, dict):
                    text = result.get("text", "")
                    confidence = result.get("confidence", None)
                elif isinstance(result, tuple) and len(result) == 2:
                    text, confidence = result
                else:
                    text = str(result) if result is not None else ""
                    confidence = None

                word.text = text
                word.recognition_confidence = confidence

        if profile:
            print(f"Pipeline total: {time.time() - start_time:.3f}s")

        if vis:
            pil = (
                image
                if isinstance(image, Image.Image)
                else Image.fromarray(image_array)
            )
            vis_img = visualize_page(pil, detection_result, show_order=True)
            return detection_result, vis_img

        return detection_result

    def process_batch(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        recognize_text: bool = True,
        vis: bool = False,
        profile: bool = False,
    ):
        results = []
        for img in images:
            res = self.process(
                img, recognize_text=recognize_text, vis=vis, profile=profile
            )
            results.append(res[0] if vis else res)
        return results

    def get_text(self, page) -> str:
        lines = []
        for block in page.blocks:
            sorted_words = sorted(
                block.words, key=lambda w: min(p[0] for p in w.polygon)
            )
            texts = [w.text for w in sorted_words if getattr(w, "text", None)]
            if texts:
                lines.append(" ".join(texts))
        return "\n".join(lines)

    def _extract_word_image(
        self, image: np.ndarray, polygon: np.ndarray
    ) -> Optional[np.ndarray]:
        try:
            x_min, y_min = np.min(polygon, axis=0)
            x_max, y_max = np.max(polygon, axis=0)

            h, w = image.shape[:2]
            x1 = max(0, int(x_min))
            y1 = max(0, int(y_min))
            x2 = min(w, int(x_max))
            y2 = min(h, int(y_max))

            region_image = image[y1:y2, x1:x2]

            return region_image if region_image.size > 0 else None
        except Exception:
            return None
