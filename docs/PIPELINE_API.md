# API Совместимость для Pipeline

Pipeline в `manuscript-ocr` спроектирован для работы с **любыми** детекторами и распознавателями, реализующими простой интерфейс.

---

## Требования к Детектору

Класс детектора должен реализовать метод `predict`:

```python
def predict(
    self, 
    image: Union[str, np.ndarray, Image.Image],
    vis: bool = False,
    profile: bool = False
) -> Union[Dict[str, Any], Tuple[Page, ...], Page]:
    """
    Параметры:
    - image: путь к файлу, numpy массив или PIL Image
    - vis: флаг визуализации (опционально)
    - profile: флаг профилирования (опционально)
    
    Возвращает один из форматов:
    1. dict с ключом "page": {"page": Page, ...}
    2. tuple где первый элемент Page: (Page, ...)
    3. напрямую объект Page
    """
    pass
```

### Структура результата

`Page` должна иметь:
- `blocks`: список объектов `Block`

`Block` должен иметь:
- `words`: список объектов `Word`

`Word` должен иметь:
- `polygon`: список из 4 точек `[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]`
- `detection_confidence`: float (уверенность детекции)

---

## Требования к Распознавателю

Класс распознавателя должен реализовать метод `predict`:

```python
def predict(
    self, 
    images: List[np.ndarray]
) -> List[Dict[str, Any]]:
    """
    Параметры:
    - images: список numpy массивов (RGB изображения слов)
    
    Возвращает список словарей, где каждый элемент:
    {
        "text": str,        # распознанный текст
        "confidence": float # уверенность распознавания [0, 1]
    }
    """
    pass
```

**Важно:** Формат возврата - список словарей для согласованности с API детектора.

---

## Примеры совместимых реализаций

### Минимальный детектор

```python
from manuscript.detectors._types import Word, Block, Page

class MyDetector:
    def predict(self, image, vis=False, profile=False):
        # Ваша логика детекции
        words = [
            Word(
                polygon=[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
                detection_confidence=0.95
            ),
            # ... другие слова
        ]
        page = Page(blocks=[Block(words=words)])
        return {"page": page}
```

### Минимальный распознаватель

```python
class MyRecognizer:
    def predict(self, images):
        results = []
        for img in images:
            # Ваша логика распознавания
            text = "распознанный текст"
            confidence = 0.92
            results.append({"text": text, "confidence": confidence})
        return results
```

### Использование с Pipeline

```python
from manuscript import Pipeline

# С пользовательскими компонентами
detector = MyDetector()
recognizer = MyRecognizer()
pipeline = Pipeline(detector, recognizer)

# Или с моделями по умолчанию
pipeline = Pipeline()

result = pipeline.predict("image.jpg")
text = pipeline.get_text(result)
```

---

## Тестирование совместимости

Файл `tests/test_pipeline_api_compatibility.py` содержит:

- `DummyDetector` - минимальная реализация детектора
- `DummyRecognizer` - минимальная реализация распознавателя
- Тесты проверяющие все варианты возвращаемых форматов

Вы можете использовать эти классы как reference implementation или для unit-тестов своих компонентов.

---

## Замена компонентов

Pipeline позволяет легко заменять детектор или распознаватель, или использовать модели по умолчанию:

```python
from manuscript import Pipeline

# Модели по умолчанию (автоматическая инициализация)
pipeline = Pipeline()

# Только детектор кастомный, распознаватель по умолчанию
from my_package import MyCustomDetector
pipeline = Pipeline(detector=MyCustomDetector())

# Только распознаватель кастомный, детектор по умолчанию
from my_package import MyCustomRecognizer
pipeline = Pipeline(recognizer=MyCustomRecognizer())

# Оба компонента кастомные
pipeline = Pipeline(
    detector=MyCustomDetector(),
    recognizer=MyCustomRecognizer()
)
```

Или с использованием встроенных моделей с параметрами:

```python
from manuscript import Pipeline
from manuscript.detectors import EAST
from manuscript.recognizers import TRBA

# EAST с повышенным порогом уверенности
detector = EAST(score_thresh=0.8)
# TRBA с GPU
recognizer = TRBA(device="cuda")

pipeline = Pipeline(detector, recognizer)
```

---

## Дополнительные параметры

Pipeline поддерживает дополнительные параметры:

```python
from manuscript import Pipeline

# С параметрами по умолчанию
pipeline = Pipeline()

# С минимальным размером текста
pipeline = Pipeline(min_text_size=10)

# С кастомными моделями и параметрами
from manuscript.detectors import EAST
from manuscript.recognizers import TRBA

pipeline = Pipeline(
    detector=EAST(score_thresh=0.7),
    recognizer=TRBA(device="cuda"),
    min_text_size=5  # Минимальный размер бокса (в пикселях)
)
```

Боксы меньше `min_text_size` будут отфильтрованы и не будут отправлены на распознавание.
