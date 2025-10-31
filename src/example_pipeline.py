from manuscript.detectors import EAST
from manuscript.recognizers import TRBAInfer
from manuscript import OCRPipeline


# Путь к изображению (укажите свой файл)
image_path = r"C:\Users\pasha\OneDrive\Рабочий стол\i (1).jpg"

# Инициализация детектора и распознавателя
detector = EAST()
recognizer = TRBAInfer()

# Создание OCR-пайплайна
pipeline = OCRPipeline(detector=detector, recognizer=recognizer)

# Обработка изображения и получение результата
result, img = pipeline.predict(image_path, vis=True)

# Визуализация результата
img.show()
