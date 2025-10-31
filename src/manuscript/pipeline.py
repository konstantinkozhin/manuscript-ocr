import numpy as np
from PIL import Image
from typing import Union, Optional
import cv2
from pathlib import Path
import time

from .detectors import EASTInfer
from .recognizers import TRBAInfer


class OCRPipeline:
    def __init__(
        self, detector: EASTInfer, recognizer: TRBAInfer, min_text_size: int = 5
    ):
        self.detector = detector
        self.recognizer = recognizer
        self.min_text_size = min_text_size

    def process(
        self,
        image: Union[str, np.ndarray, Image.Image],
        recognize_text: bool = True,
        vis: bool = False,
        profile: bool = False,
    ):
        start_time = time.time()

        # Detection
        t0 = time.time()
        detection_result = self.detector.predict(image, vis=vis, profile=profile)
        if profile:
            print(f"Detection: {time.time() - t0:.3f}s")

        # Detector may return (Page, vis_img) even if vis=False
        if isinstance(detection_result, tuple):
            detection_result, vis_image = detection_result
        else:
            vis_image = None

        if not recognize_text:
            if profile:
                print(f"Pipeline total: {time.time() - start_time:.3f}s")
            return (detection_result, vis_image) if vis else detection_result

        # Load image for cropping
        t0 = time.time()
        image_array = self._load_image_as_array(image)
        if profile:
            print(f"Load image for crops: {time.time() - t0:.3f}s")

        # Extract word regions
        t0 = time.time()
        all_words = []
        word_images = []

        for block in detection_result.blocks:
            for word in block.words:
                polygon = np.array(word.polygon, dtype=np.int32)
                x_min, y_min = np.min(polygon, axis=0)
                x_max, y_max = np.max(polygon, axis=0)

                width = x_max - x_min
                height = y_max - y_min

                if width >= self.min_text_size and height >= self.min_text_size:
                    region_image = self._extract_word_image(image_array, polygon)
                    if region_image is not None and region_image.size > 0:
                        all_words.append(word)
                        word_images.append(region_image)
        if profile:
            print(f"Extract {len(word_images)} crops: {time.time() - t0:.3f}s")

        # Recognition
        if word_images:
            t0 = time.time()
            # TRBA API returns List[Tuple[text, confidence]] for list input
            recognition_results = self.recognizer.predict(word_images)
            if profile:
                print(f"Recognition: {time.time() - t0:.3f}s")

            # Backward-compat: if recognizer returns list of strings
            for idx, word in enumerate(all_words):
                res = recognition_results[idx]
                if isinstance(res, tuple) and len(res) == 2:
                    text, confidence = res
                else:
                    text, confidence = str(res), None
                word.text = text
                word.recognition_confidence = confidence

        if profile:
            print(f"Pipeline total: {time.time() - start_time:.3f}s")
        return (detection_result, vis_image) if vis else detection_result

    def process_batch(
        self,
        images: list[Union[str, np.ndarray, Image.Image]],
        recognize_text: bool = True,
        vis: bool = False,
        profile: bool = False,
    ):
        results = []
        for img in images:
            res = self.process(img, recognize_text=recognize_text, vis=vis, profile=profile)
            # For batch, return only Page objects for simplicity
            results.append(res[0] if vis else res)
        return results

    def get_text(self, page) -> str:
        try:
            blocks = getattr(page, "blocks", []) or []
        except Exception:
            return ""
        lines = []
        for block in blocks:
            words = getattr(block, "words", []) or []
            texts = [getattr(w, "text", None) for w in words]
            texts = [t for t in texts if t]
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

    def _load_image_as_array(
        self, image: Union[str, np.ndarray, Image.Image]
    ) -> np.ndarray:
        if isinstance(image, str):
            # Сначала пробуем cv2
            img_array = cv2.imread(image)
            if img_array is None:
                # Пробуем через PIL если cv2 не смог открыть (проблемы с кодировкой путей)
                try:
                    pil_img = Image.open(image)
                    return np.array(pil_img.convert("RGB"))
                except Exception:
                    # Последняя попытка - через numpy с обходом кодировки
                    try:
                        # Читаем файл как байты, затем декодируем
                        img_path = Path(image)
                        if not img_path.exists():
                            raise FileNotFoundError(f"Файл не найден: {image}")

                        # Используем cv2.imdecode для обхода проблем с путями
                        with open(img_path, "rb") as f:
                            img_bytes = f.read()
                        img_np = np.frombuffer(img_bytes, np.uint8)
                        img_array = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

                        if img_array is None:
                            raise ValueError("Не удалось декодировать изображение")

                        return cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                    except Exception as e:
                        raise FileNotFoundError(
                            f"Не удалось загрузить изображение всеми способами: {image}. Последняя ошибка: {e}"
                        )
            return cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            return np.array(image.convert("RGB"))
        elif isinstance(image, np.ndarray):
            return image.copy()
        else:
            raise TypeError(f"Неподдерживаемый тип изображения: {type(image)}")
