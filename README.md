
<img width="2028" height="496" alt="Frame 8" src="docs\logo.png" />

# Manuscript OCR

Библиотека для детекции и распознавания текста на исторических, архивных и рукописных документах.

---

## Установка

```bash
pip install manuscript-ocr
```

Для GPU поддержки:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## Быстрый старт

```python
from manuscript import Pipeline
from manuscript.detectors import EAST
from manuscript.recognizers import TRBA

# Инициализация
detector = EAST()
recognizer = TRBA()
pipeline = Pipeline(detector, recognizer)

# Обработка изображения
result = pipeline.predict("document.jpg")

# Извлечение текста
text = pipeline.get_text(result)
print(text)
```

---

## Документация

Подробные примеры и руководства:

- [Детектор (EAST)](./DETECTOR.md) - настройка и использование детектора текста
- [Распознаватель (TRBA)](./RECOGNIZERS.md) - распознавание и обучение моделей
- [Pipeline API](./docs/PIPELINE_API.md) - интеграция и создание кастомных компонентов

---

## Основные возможности

**Детекция текста (EAST)**
- Высокая точность на рукописных документах
- Поддержка произвольных четырехугольников
- Настраиваемые параметры детекции

**Распознавание (TRBA)**
- Оптимизировано для исторических документов
- Поддержка beam search и greedy декодирования
- Возможность дообучения на своих данных

**Pipeline**
- Единый интерфейс для полного OCR
- Автоматическая сортировка в порядке чтения
- Гибкая настройка компонентов

---

## Примеры

Детекция с сортировкой:
```python
detector = EAST()
result = detector.predict("image.jpg", sort_reading_order=True)
```

Распознавание с beam search:
```python
recognizer = TRBA()
results = recognizer.predict(images, mode="beam")
for r in results:
    print(f"{r['text']} (confidence: {r['confidence']:.2f})")
```

Визуализация:
```python
from manuscript import visualize_page, read_image

img = read_image("document.jpg")
vis = visualize_page(img, result["page"], show_order=True)
vis.save("output.jpg")
```

---

## Лицензия

MIT

---

## Разработка

```bash
git clone https://github.com/konstantinkozhin/manuscript-ocr
cd manuscript-ocr
pip install -r requirements-dev.txt
pytest tests/
```
