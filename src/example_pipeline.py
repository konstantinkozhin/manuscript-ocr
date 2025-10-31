from manuscript.detectors._east import EASTInfer
from manuscript.recognizers.trba import TRBAInfer
from manuscript.pipeline import OCRPipeline


# Путь к изображению (укажите свой файл)
image_path = r"C:\Users\pasha\OneDrive\Рабочий стол\i (1).jpg"

# Инициализация детектора и распознавателя
detector = EASTInfer()
recognizer = TRBAInfer()

# Создание OCR-пайплайна
pipeline = OCRPipeline(detector=detector, recognizer=recognizer)

# Обработка изображения и получение результата
result, img = pipeline.predict(image_path, vis=True)

# Визуализация результата
img.show()
