<img width="2028" height="496" alt="Frame 8" src="https://github.com/user-attachments/assets/cdba0d4c-4cab-4f77-a056-6d3c20192566" />

## Installation

### Для пользователей
```bash
pip install manuscript-ocr
```

### Для разработчиков
```bash
pip install -r requirements-dev.txt
```

> **Примечание**: `requirements-dev.txt` включает GPU версию PyTorch, инструменты тестирования, форматирования и сборки.

### GPU поддержка
Если вы пользователь и хотите GPU поддержку:
```bash
pip install manuscript-ocr
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
```

**Проверка GPU:**
```python
import torch
print(f"CUDA доступна: {torch.cuda.is_available()}")
```

## Usage Examples

### OCR Pipeline

```python
from manuscript.detectors import EASTInfer
from manuscript import TRBAInfer  
from manuscript.pipeline import OCRPipeline

# Инициализация компонентов
detector = EASTInfer(score_thresh=0.7)

# TRBAInfer использует config.json для параметров модели
# img_h, img_w, hidden_size берутся из конфига
recognizer = TRBAInfer(
    model_path="path/to/model.pth",
    config_path="path/to/config.json", 
    charset_path="path/to/charset.txt"  # Опционально, можно в конфиге
)

# Альтернативно: charset_path в конфиге
recognizer = TRBAInfer(
    model_path="path/to/model.pth",
    config_path="path/to/config.json"  # charset_path внутри конфига
)

pipeline = OCRPipeline(detector, recognizer)

# Полная обработка изображения
result = pipeline.process("path/to/image.jpg")

# Получение распознанного текста
text = pipeline.get_text(result)
print("Распознанный текст:", text)

# Подробная информация о каждом слове
for block in result.blocks:
    for word in block.words:
        print(f"Текст: '{word.text}' | "
              f"Детекция: {word.detection_confidence:.3f} | "
              f"Распознавание: {word.recognition_confidence:.3f}")
```

### Детекция текста 

```python
from PIL import Image
from manuscript.detectors import EASTInfer

# Инициализация детектора
detector = EASTInfer(score_thresh=0.9)

# Детекция с визуализацией
page, vis_image = detector.predict("example/image.jpg", vis=True)

# Показать результат
Image.fromarray(vis_image).show()

# Информация о найденных областях
for block in page.blocks:
    for word in block.words:
        print(f"Область: {word.polygon} | Уверенность: {word.detection_confidence:.3f}")
```

### Пакетная обработка

```python
# Обработка нескольких изображений
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = pipeline.process_batch(image_paths)

for i, page in enumerate(results):
    text = pipeline.get_text(page)
    print(f"Изображение {i+1}: {text}")
```

### Конфигурация

TRBAInfer uses JSON configuration for model parameters:

```json
{
    "img_h": 32,
    "img_w": 128, 
    "hidden_size": 256,
    "max_len": 40,
    "encoding": "utf-8"
}
```

**Обязательные параметры в config.json:** `img_h`, `img_w`, `hidden_size`  
**Отдельные параметры:** `charset_path` (передается в конструктор)

### Запуск примеров

Полные рабочие примеры доступны в файлах:
- `example_ocr_pipeline.py` - подробная демонстрация всех возможностей
- `example_config_usage.py` - пример использования конфигурации
- `simple_ocr_example.py` - простые примеры для быстрого старта

```bash
python example_ocr_pipeline.py
python simple_ocr_example.py
```

