## TRBA — примеры использования

### **1. Распознавание одного изображения**

```python
from manuscript.recognizers import TRBA

# Создаём распознаватель с настройками по умолчанию (автоматическая загрузка весов).
recognizer = TRBA()

# Распознаём одно изображение слова
results = recognizer.predict("data/word_images/word_001.jpg")

# Результат — список кортежей (текст, уверенность)
text, confidence = results[0]
print(f"Распознанный текст: '{text}'")
print(f"Уверенность: {confidence:.3f}")
```

### **2. Пакетное распознавание с beam search**

```python
from manuscript.recognizers import TRBA

# Инициализация с кастомными весами
recognizer = TRBA(
    model_path="path/to/custom_model.pth",
    config_path="path/to/custom_config.json",
    device="cuda"  # или "cpu"
)

# Список изображений для распознавания
image_paths = [
    "data/words/word_001.jpg",
    "data/words/word_002.jpg",
    "data/words/word_003.jpg",
]

# Пакетное распознавание с beam search (медленнее, но точнее)
results = recognizer.predict(
    images=image_paths,
    batch_size=16,
    mode="beam",          # beam search декодирование
    beam_size=10,         # ширина луча
    temperature=1.5,      # температура для разнообразия
    alpha=0.9,            # штраф за длину последовательности
)

# Выводим результаты
for img_path, (text, conf) in zip(image_paths, results):
    print(f"{img_path}: '{text}' (confidence: {conf:.3f})")
```

### **3. Быстрое распознавание с greedy декодированием**

```python
from manuscript.recognizers import TRBA
import cv2
import numpy as np

recognizer = TRBA(device="auto")

# Можно передавать numpy массивы напрямую
img = cv2.imread("word.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Greedy декодирование (быстрее, но менее точное)
results = recognizer.predict(
    images=img_rgb,
    mode="greedy",
    batch_size=1
)

text, confidence = results[0]
print(f"Быстрое распознавание: '{text}' ({confidence:.3f})")
```

### **4. Обучение на своих данных**

```python
from manuscript.recognizers import TRBA

# Обучение на одном датасете
best_model = TRBA.train(
    train_csvs="data/train.csv",        # CSV с колонками: image_path, text
    train_roots="data/train_images",    # Корневая папка с изображениями
    val_csvs="data/val.csv",            # Валидационный CSV
    val_roots="data/val_images",        # Валидационные изображения
    exp_dir="./experiments/trba_exp1",  # Папка для сохранения
    epochs=50,
    batch_size=64,
    img_h=64,                           # Высота входного изображения
    img_w=256,                          # Ширина входного изображения
    max_len=25,                         # Максимальная длина текста
    lr=1e-3,
    optimizer="Adam",
)

print(f"Лучшая модель сохранена: {best_model}")
```

### **5. Обучение на нескольких датасетах с пропорциями**

```python
from manuscript.recognizers import TRBA

# Пути к нескольким датасетам
train_csvs = [
    "data/dataset1/train.csv",
    "data/dataset2/train.csv",
    "data/dataset3/train.csv",
]
train_roots = [
    "data/dataset1/images",
    "data/dataset2/images",
    "data/dataset3/images",
]

# Пропорции сэмплирования (должны суммироваться в 1.0)
train_proportions = [0.5, 0.3, 0.2]  # 50% из dataset1, 30% из dataset2, 20% из dataset3

best_model = TRBA.train(
    train_csvs=train_csvs,
    train_roots=train_roots,
    train_proportions=train_proportions,
    val_csvs="data/val.csv",
    val_roots="data/val_images",
    exp_dir="./experiments/multi_dataset",
    epochs=100,
    batch_size=32,
    lr=5e-4,
    optimizer="AdamW",
    weight_decay=1e-4,
    scheduler="CosineAnnealingLR",
)
```

### **6. Fine-tuning с замороженными слоями**

```python
from manuscript.recognizers import TRBA

# Fine-tuning с замороженным CNN и использованием предобученных весов
best_model = TRBA.train(
    train_csvs="data/finetune.csv",
    train_roots="data/finetune_images",
    val_csvs="data/val.csv",
    val_roots="data/val_images",
    exp_dir="./experiments/finetune_frozen_cnn",
    
    # Загрузка предобученных весов
    pretrain_weights="default",    # Или путь к своим весам
    
    # Заморозка слоёв
    freeze_cnn="all",              # Заморозить весь CNN
    freeze_enc_rnn="none",         # Не замораживать RNN энкодер
    freeze_attention="none",       # Не замораживать attention
    
    epochs=20,
    batch_size=64,
    lr=1e-4,                       # Меньший learning rate для fine-tuning
    optimizer="Adam",
)
```

### **7. Возобновление обучения**

```python
from manuscript.recognizers import TRBA

# Продолжить обучение с сохранённого чекпоинта
best_model = TRBA.train(
    train_csvs="data/train.csv",
    train_roots="data/train_images",
    val_csvs="data/val.csv",
    val_roots="data/val_images",
    
    # Путь к чекпоинту для продолжения
    resume_path="experiments/trba_exp1/checkpoints/last.pth",
    
    epochs=100,
    batch_size=32,
    save_every=5,  # Сохранять чекпоинт каждые 5 эпох
)
```

### **8. Двойная валидация (greedy + beam)**

```python
from manuscript.recognizers import TRBA

# Валидация с двумя стратегиями декодирования
best_model = TRBA.train(
    train_csvs="data/train.csv",
    train_roots="data/train_images",
    val_csvs="data/val.csv",
    val_roots="data/val_images",
    exp_dir="./experiments/dual_val",
    
    # Двойная валидация
    dual_validate=True,            # Валидировать и greedy, и beam
    beam_size=8,
    beam_alpha=0.9,
    beam_temperature=1.7,
    
    epochs=50,
    batch_size=64,
    eval_every=1,                  # Валидировать каждую эпоху
)
```

### **9. Настройка размеров изображения и архитектуры**

```python
from manuscript.recognizers import TRBA

# Кастомные параметры архитектуры
best_model = TRBA.train(
    train_csvs="data/train.csv",
    train_roots="data/train_images",
    val_csvs="data/val.csv",
    val_roots="data/val_images",
    exp_dir="./experiments/custom_arch",
    
    # Размеры изображения
    img_h=128,                     # Высота (больше для высоких символов)
    img_w=512,                     # Ширина (больше для длинных слов)
    
    # Архитектура
    max_len=40,                    # Максимальная длина последовательности
    hidden_size=512,               # Размер скрытых слоёв RNN
    
    epochs=100,
    batch_size=16,                 # Меньший batch из-за большего размера
    lr=1e-3,
)
```

### **10. Использование кастомного charset**

```python
from manuscript.recognizers import TRBA

# Обучение с собственным набором символов
best_model = TRBA.train(
    train_csvs="data/train.csv",
    train_roots="data/train_images",
    val_csvs="data/val.csv",
    val_roots="data/val_images",
    exp_dir="./experiments/custom_charset",
    
    # Путь к файлу с набором символов
    charset_path="data/custom_charset.txt",
    encoding="utf-8",
    
    epochs=50,
    batch_size=64,
)

# Затем использовать для инференса
recognizer = TRBA(
    model_path="experiments/custom_charset/checkpoints/best_acc_weights.pth",
    config_path="experiments/custom_charset/checkpoints/config.json",
    charset_path="data/custom_charset.txt"
)
```
