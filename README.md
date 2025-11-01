
<img width="2028" height="496" alt="Frame 8" src="docs\logo.png" />

# Manuscript OCR

Модуль для детекции и распознавания текста на исторических, архивных и рукописных документах.  
Включает:
- EAST для детекции
- TRBA для распознавания слов
- Pipeline — удобный интерфейс для полной обработки изображений

---

## Installation

### Для пользователей
```bash
pip install manuscript-ocr
````

### Для разработчиков

```bash
pip install -r requirements-dev.txt
```

> **Примечание:** `requirements-dev.txt` включает GPU-версию PyTorch, инструменты тестирования, форматирования и сборки.

### GPU поддержка

```bash
pip install manuscript-ocr
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --force-reinstall
```

**Проверка GPU:**

```python
import torch
print(f"CUDA доступна: {torch.cuda.is_available()}")
```

---

## Usage Examples

### OCR Pipeline (полная обработка)

```python
from manuscript.detectors import EAST
from manuscript.recognizers import TRBA
from manuscript import Pipeline

# Инициализация компонентов
detector = EAST(score_thresh=0.7)

recognizer = TRBA(
    model_path="path/to/model.pth",
    config_path="path/to/config.json",
    charset_path="path/to/charset.txt"
)

pipeline = Pipeline(detector, recognizer)

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

➡ **Подробные примеры для детектора (`EAST`) вынесены в отдельный файл:**  
📄 **[DETECTOR.md](./DETECTOR.md)**

➡ **Подробные примеры для распознавателя (`TRBA`) вынесены в отдельный файл:**  
📄 **[RECOGNIZERS.md](./RECOGNIZERS.md)**

---
