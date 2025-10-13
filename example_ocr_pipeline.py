from manuscript.detectors import EASTInfer
from manuscript import TRBAInfer  
from manuscript.pipeline import OCRPipeline

# Инициализация компонентов
detector = EASTInfer(score_thresh=0.9)
recognizer = TRBAInfer(
    model_path=r"C:\Users\USER\Desktop\OCR_MODELS\exp_4_model_32\best_acc_weights.pth",
    config_path=r"src\manuscript\recognizers\configs\config.json",
    charset_path=r"src\manuscript\recognizers\configs\charset.txt"
)
pipeline = OCRPipeline(detector, recognizer)

# Полная обработка изображения
result = pipeline.process("example/ocr_example_image.jpg")

# Получение распознанного текста
text = pipeline.get_text(result)
print("Распознанный текст:", text)

# Подробная информация о каждом слове
for block in result.blocks:
    for word in block.words:
        print(f"Текст: '{word.text}' | "
              f"Детекция: {word.detection_confidence:.3f} | "
              f"Распознавание: {word.recognition_confidence:.3f}")