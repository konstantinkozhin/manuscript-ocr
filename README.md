
<img width="2028" height="496" alt="Frame 8" src="docs\logo.png" />

# Manuscript OCR

Библиотека для детекции и распознавания текста на исторических, архивных, рукописных и печатных документах.

---

## Быстрый пример

```python
from manuscript import Pipeline
from manuscript.utils import visualize_page

# Инициализация пайплайна
pipeline = Pipeline()

image_path = "example/image.jpg"

# Запуск пайплайна
result = pipeline.predict(image_path)

# Весь объект результата
print(result)

# Текст
text = pipeline.get_text(result["page"])
print(text)

# Визуализация
visualize_page(image_path, result["page"], show_lines=False, show_numbers=True)
```

<img src="example/image.jpg" alt="Результат визуализации" />

**Распознанный текст:**
```
мѣстахъ губернiи могли бы подъ
руководствомъ врача, подавать
помощь заболѣвающему скоту
принимать мѣры предупрежден
развитiя эпизотiи и наблюда
за исполненiемъ означенныхъ мѣ
Сильный падежъ скота необхо
мо имѣлъ влiянiе на взысканiе
податей и повинностей съ недо
ками, которое и было менѣе усп.
но нежели въ 1870 году.
За вычетомъ...
```

---

## Документация

**[Полная документация](https://konstantinkozhin.github.io/manuscript-ocr)** (English & Русский)

Подробные примеры и руководства:
- [Pipeline API](./docs/PIPELINE_API.md) - интеграция и создание кастомных компонентов

---

## Установка

### Режим 1: Пользовательская установка (только inference)

Минимальная установка для использования готовых моделей через ONNX Runtime (CPU):

```bash
pip install manuscript-ocr
```

**Для ускорения на GPU (NVIDIA CUDA):**
```bash
# Удалите CPU версию ONNX Runtime
pip uninstall onnxruntime

# Установите GPU версию
pip install onnxruntime-gpu
```

**Для Apple Silicon (M1/M2/M3) с CoreML:**
```bash
# Удалите стандартную версию
pip uninstall onnxruntime

# Установите версию для Apple Silicon (оптимизация через CoreML)
pip install onnxruntime-silicon
```

### Режим 2: Dev установка (с обучением моделей)

Полная установка с PyTorch для обучения собственных моделей:

```bash
pip install manuscript-ocr[dev]
```

**Для обучения на GPU (NVIDIA CUDA):**
```bash
# Сначала установите manuscript-ocr[dev]
pip install manuscript-ocr[dev]

# Затем обновите PyTorch на GPU версию
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> **Примечание:** GPU версии (ONNX Runtime GPU и PyTorch CUDA) пользователь устанавливает вручную по необходимости.

---

## Быстрый старт

```python
from manuscript import Pipeline

# Инициализация с моделями по умолчанию (CPU)
pipeline = Pipeline()

# Обработка изображения
result = pipeline.predict("document.jpg")

# Извлечение текста
text = pipeline.get_text(result["page"])
print(text)
```

### Использование GPU/CoreML

```python
# NVIDIA CUDA
detector = EAST(device="cuda")
recognizer = TRBA(device="cuda")
pipeline = Pipeline(detector=detector, recognizer=recognizer)

# Apple Silicon (M1/M2/M3)
detector = EAST(device="coreml")
recognizer = TRBA(device="coreml")
pipeline = Pipeline(detector=detector, recognizer=recognizer)
```
