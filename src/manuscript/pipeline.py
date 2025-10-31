import numpy as np
from PIL import Image
from typing import Union, Optional
import cv2
from pathlib import Path
import time
from typing import List, Tuple
from .detectors import EASTInfer
from .recognizers import TRBAInfer
from PIL import ImageDraw
from manuscript.detectors._east.utils import draw_quads


def visualize(
    pil_img: Image.Image,
    page,
    *,
    connect_words=True,
    show_numbers=True,
    line_color=(0, 255, 0),
    number_color=(255, 255, 255),
    number_bg=(0, 0, 0),
) -> Image.Image:

    img = np.array(pil_img.convert("RGB"))

    # --- Build quads from words (already sorted in page.blocks[x].words) ---
    quads = []
    words_in_order = []  # preserve order for lines & numbers

    for block in page.blocks:
        for w in block.words:
            poly = np.array(w.polygon).reshape(-1)
            quads.append(poly)
            words_in_order.append(w)

    if len(quads) == 0:
        return pil_img

    quads = np.stack(quads, axis=0)

    # draw polygons
    out = draw_quads(
        image=img,
        quads=quads,
        color=(0, 0, 255),
        thickness=2,
        dark_alpha=0.3,
        blur_ksize=11,
    )

    draw = ImageDraw.Draw(out)

    # ---- Draw lines and/or numbers strictly in existing order ----
    if connect_words or show_numbers:

        centers = []
        for w in words_in_order:
            xs = [p[0] for p in w.polygon]
            ys = [p[1] for p in w.polygon]
            centers.append((sum(xs) / 4, sum(ys) / 4))

        # Lines: between neighbors
        if connect_words and len(centers) > 1:
            for p, c in zip(centers, centers[1:]):
                draw.line([p, c], fill=line_color, width=3)

        # Numbers: on centers
        if show_numbers:
            for idx, c in enumerate(centers, start=1):
                cx, cy = c
                draw.rectangle(
                    [cx - 12, cy - 12, cx + 12, cy + 12],
                    fill=number_bg,
                )
                draw.text((cx - 6, cy - 8), str(idx), fill=number_color)

    return out


def resolve_intersections(boxes):
    def intersect(b1, b2):
        return not (
            b1[2] <= b2[0] or b2[2] <= b1[0] or b1[3] <= b2[1] or b2[3] <= b1[1]
        )

    resolved = list(boxes)
    max_iterations = 50

    for _ in range(max_iterations):
        changed = False
        for i in range(len(resolved)):
            for j in range(i + 1, len(resolved)):
                if intersect(resolved[i], resolved[j]):
                    x0, y0, x1, y1 = resolved[i]
                    x0b, y0b, x1b, y1b = resolved[j]

                    resolved[i] = (
                        x0,
                        y0,
                        int(x1 - (x1 - x0) * 0.1),
                        int(y1 - (y1 - y0) * 0.1),
                    )
                    resolved[j] = (
                        x0b,
                        y0b,
                        int(x1b - (x1b - x0b) * 0.1),
                        int(y1b - (y1b - y0b) * 0.1),
                    )
                    changed = True
        if not changed:
            break

    return resolved


def sort_boxes_reading_order(
    boxes: List[Tuple[int, int, int, int]],
    y_tol_ratio: float = 0.6,
    x_gap_ratio: float = np.inf,
) -> List[Tuple[int, int, int, int]]:
    if not boxes:
        return []

    avg_h = np.mean([b[3] - b[1] for b in boxes])
    lines = []

    for b in sorted(boxes, key=lambda b: (b[1] + b[3]) / 2):
        cy = (b[1] + b[3]) / 2
        placed = False

        for ln in lines:
            line_cy = np.mean([(v[1] + v[3]) / 2 for v in ln])
            last_x1 = max(v[2] for v in ln)

            if (
                abs(cy - line_cy) <= avg_h * y_tol_ratio
                and (b[0] - last_x1) <= avg_h * x_gap_ratio
            ):
                ln.append(b)
                placed = True
                break

        if not placed:
            lines.append([b])

    lines.sort(key=lambda ln: np.mean([(b[1] + b[3]) / 2 for b in ln]))
    for ln in lines:
        ln.sort(key=lambda b: b[0])

    return [b for ln in lines for b in ln]


def sort_boxes_reading_order_with_resolutions(
    boxes, y_tol_ratio=0.6, x_gap_ratio=np.inf
):
    compressed = resolve_intersections(boxes)
    mapping = {c: o for c, o in zip(compressed, boxes)}

    sorted_compressed = sort_boxes_reading_order(
        compressed, y_tol_ratio=y_tol_ratio, x_gap_ratio=x_gap_ratio
    )
    return [mapping[b] for b in sorted_compressed]


class OCRPipeline:
    def __init__(
        self, detector: EASTInfer, recognizer: TRBAInfer, min_text_size: int = 5
    ):
        self.detector = detector
        self.recognizer = recognizer
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
                arr = self._load_image_as_array(image)
                pil = image if isinstance(image, Image.Image) else Image.fromarray(arr)
                vis_img = visualize(pil, detection_result)
                return detection_result, vis_img
            return detection_result

        # ---- LOAD IMAGE ----
        t0 = time.time()
        image_array = self._load_image_as_array(image)
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
                res = recognition_results[idx]
                if isinstance(res, tuple) and len(res) == 2:
                    text, confidence = res
                else:
                    text, confidence = str(res), None

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
            vis_img = visualize(pil, detection_result)
            return detection_result, vis_img

        return detection_result

    def process_batch(
        self,
        images: list[Union[str, np.ndarray, Image.Image]],
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
            sorted_words = sorted(block.words, key=lambda w: w.x1)
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
