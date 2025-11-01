## EAST — примеры использования

### **1. Обработка одного изображения**

```python
from manuscript.detectors import EAST

# Создаём детектор с настройками по умолчанию.
detector = EAST()

# Запускаем предикт. Если vis=True — вернётся изображение с визуализацией боксов.
result = detector.predict("data/samples/page_001.jpg", vis=True)

# Результат содержит объект Page (разметка слов и блоков),
# а также опциональную визуализацию.
page = result["page"]
vis_image = result["vis_image"]

# Покажем изображение с боксами, если оно возвращено.
if vis_image is not None:
    vis_image.show()

# Перебираем все блоки и слова и выводим текст + confidence
for block in page.blocks:
    for word in block.words:
        print(word.text, word.detection_confidence)
```

### **2. Загрузка собственных весов и изменение параметров предобработки**

```python
from pathlib import Path
from manuscript.detectors import EAST

# Путь к вашим кастомным весам модели
custom_weights = Path("artifacts/east_quad_23_05.pth")

# Меняем размер входа, пороги и коэффициенты расширения боксов
detector = EAST(
    weights_path=custom_weights,
    target_size=1024,       # изображение будет масштабировано в 1024x1024
    score_thresh=0.5,       # порог отсева слабых предсказаний
    expand_ratio_w=0.85,    # горизонтальное расширение боксов после NMS
    expand_ratio_h=0.85,    # вертикальное расширение боксов
)

# Можно запросить raw карты сети: score_map (с вероятностями) и geo_map (геометрия боксов)
result = detector.predict("data/custom/page.png", vis=False, return_maps=True)

score_map = result["score_map"]
geo_map = result["geo_map"]

print("Score map shape:", score_map.shape if score_map is not None else "нет данных")
print("Geo map shape:", geo_map.shape if geo_map is not None else "нет данных")
```

### **3. Запуск обучения на своих датасетах**

```python
from manuscript.detectors import EAST

# Пути к несколько тренировочным датасетам:
# Каждый содержит изображения и COCO JSON разметку
train_images = [
    "data/train_main/images",      # Основной датасет
    "data/train_extra/images",     # Дополнительный датасет
]
train_anns = [
    "data/train_main/annotations.json",
    "data/train_extra/annotations.json",
]

# Валидационные данные
val_images = ["data/val/images"]
val_anns = ["data/val/annotations.json"]

# Запуск обучения
best_model = EAST.train(
    train_images=train_images,     # пути к директориям с тренировочными изображениями
    train_anns=train_anns,         # соответствующие COCO JSON
    val_images=val_images,         # директория с валидацией
    val_anns=val_anns,             # COCO JSON для валидации
    experiment_root="./experiments", # куда сохранять логи и чекпоинты
    model_name="east_doc_example",   # имя папки внутри experiment_root
    target_size=1024,                # размер входного изображения (H=W)
    epochs=20,                       # количество эпох
    batch_size=4,                    # размер батча
    score_geo_scale=None,            # если None — будет использоваться значение модели
)

# Возвращается путь к лучшей сохранённой версии модели
print("Best checkpoint path:", best_model)
```
