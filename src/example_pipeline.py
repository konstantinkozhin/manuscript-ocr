from manuscript.detectors._east import EASTInfer
from manuscript.recognizers.trba import TRBAInfer
from manuscript.pipeline import OCRPipeline


# Путь к модели распознавателя (укажите свой путь)
recognizer_model_path = r"C:\Users\USER\Desktop\OCR_MODELS\exp1_model_64\best_acc_weights.pth"
config_path = r"C:\Users\USER\Desktop\OCR_MODELS\exp1_model_64\config.json"

# Путь к изображению (укажите свой файл)
image_path = r"C:\Users\USER\Desktop\77686370-e02a-45ce-bb4d-1738030fad46 (2).jpg"

# Инициализация детектора и распознавателя
detector = EASTInfer()
recognizer = TRBAInfer(model_path=recognizer_model_path, config_path=config_path)

# Создание OCR-пайплайна
pipeline = OCRPipeline(detector=detector, recognizer=recognizer)

# Обработка изображения и получение результата
result, img = pipeline.process(image_path, recognize_text=True, profile=True, vis=True)
# Сбор распознанного текста в одну строку и вывод координат
recognized_text = ""
for b_idx, block in enumerate(result.blocks, start=1):
    for w_idx, word in enumerate(block.words, start=1):
        coords = getattr(word, "polygon", None)
        if coords:
            coords_str = ", ".join(f"({x:.2f}, {y:.2f})" for x, y in coords)
            minx = min(x for x, _ in coords)
            miny = min(y for _, y in coords)
            maxx = max(x for x, _ in coords)
            maxy = max(y for _, y in coords)
            bbox_str = f"bbox=({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f})"
        else:
            coords_str = ""
            bbox_str = ""

        rec_conf = word.recognition_confidence if word.recognition_confidence is not None else 0.0
        print(
            f"Блок {b_idx} Слово {w_idx}: "
            f"'{word.text or ''}' "
            f"(det_conf={word.detection_confidence:.4f}, rec_conf={rec_conf:.4f})"
        )
        print(f"  Координаты: [{coords_str}] {bbox_str}")

        if word.text:
            recognized_text += word.text + " "

# Вывод результата
print("Распознанный текст:")
print(recognized_text.strip())

from PIL import Image

img.show()