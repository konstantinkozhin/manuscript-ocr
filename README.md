
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

# Инициализация с моделями по умолчанию
pipeline = Pipeline()

# Обработка изображения
result = pipeline.predict("document.jpg")

# Извлечение текста
text = pipeline.get_text(result)
print(text)
```

---

## Документация

Подробные примеры и руководства:

- [Детектор (EAST)](./docs/DETECTOR.md) - настройка и использование детектора текста
- [Распознаватель (TRBA)](./docs/RECOGNIZERS.md) - распознавание и обучение моделей
- [Pipeline API](./docs/PIPELINE_API.md) - интеграция и создание кастомных компонентов

