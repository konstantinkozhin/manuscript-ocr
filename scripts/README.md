# Scripts

Утилиты для работы с моделями EAST: экспорт в ONNX, бенчмарки производительности и тестирование.

## Установка зависимостей

```bash
# Для бенчмарков
pip install psutil

# Для ONNX экспорта/инференса
pip install onnx onnxruntime

# Опционально для оптимизации графа:
pip install onnx-simplifier
```

## 1. Бенчмарк производительности EAST

### east_infer_speed_test.py

Измеряет производительность EAST детектора на CPU и GPU.

**Что измеряет:**
- Среднее время инференса на изображение
- Throughput (FPS - изображений в секунду)
- Использование RAM и VRAM
- Сравнение CPU vs GPU

**Использование:**

```bash
# Базовый бенчмарк на папке с изображениями
python scripts/east_infer_speed_test.py --folder path/to/images

# С настройками
python scripts/east_infer_speed_test.py \
    --folder path/to/images \
    --target-size 1280 \
    --score-thresh 0.6 \
    --warmup 5

# Только CPU
python scripts/east_infer_speed_test.py --folder path/to/images --cpu-only

# Только GPU
python scripts/east_infer_speed_test.py --folder path/to/images --gpu-only

# Сохранить результаты в JSON
python scripts/east_infer_speed_test.py \
    --folder path/to/images \
    --output benchmark_results.json
```

**Параметры:**
- `--folder`: Папка с изображениями (обязательный)
- `--target-size`: Размер входа (default: 1280)
- `--score-thresh`: Порог уверенности (default: 0.6)
- `--warmup`: Количество прогревочных запусков (default: 3)
- `--cpu-only`: Бенчмарк только на CPU
- `--gpu-only`: Бенчмарк только на GPU
- `--output`: Сохранить результаты в JSON

**Пример вывода:**

```
============================================================
Results for CPU
============================================================

Dataset:
  Images: 4
  Target size: 640x640
  Total detections: 818
  Avg detections/image: 204.5

Inference Time:
  Mean: 530.07 ms
  Median: 524.61 ms
  Throughput: 1.89 FPS

Memory Usage (RAM):
  After load: 832.00 MB
  Peak: 939.15 MB
  Delta: 107.14 MB

============================================================
CPU vs GPU Comparison
============================================================

Speed:
  CPU mean time: 530.07 ms
  GPU mean time: 237.69 ms
  Speedup: 2.23x

Recommendation:
  GPU is 2.2x faster - strongly recommended for production
```

## 2. Экспорт модели в ONNX

### export_east_to_onnx.py

Экспортирует PyTorch модель EAST в формат ONNX для deployment.

**Использование:**

```bash
# Экспорт с параметрами по умолчанию (использует ~/.manuscript/east/east_quad_23_05.pth)
python scripts/export_east_to_onnx.py

# Указать свои веса и выходной файл
python scripts/export_east_to_onnx.py --weights path/to/weights.pth --output model.onnx

# Использовать другой размер входа
python scripts/export_east_to_onnx.py --input-size 1280 --output east_1280.onnx
```

**Параметры:**
- `--weights`: Путь к весам PyTorch (.pth файл). По умолчанию использует `~/.manuscript/east/east_quad_23_05.pth`
- `--output`: Путь для сохранения ONNX модели (default: `east_model.onnx`)
- `--input-size`: Размер входного изображения в пикселях (default: 1280)
- `--opset`: Версия ONNX opset (default: 14)
- `--no-simplify`: Отключить оптимизацию графа через onnx-simplifier

## 3. Тестирование ONNX инференса

### test_east_onnx_inference.py

Тестирует ONNX модель и сравнивает с PyTorch.

```bash
# Простой тест инференса
python scripts/test_east_onnx_inference.py \
    --model east_model.onnx \
    --image path/to/test/image.jpg

# Сравнение с PyTorch моделью
python scripts/test_east_onnx_inference.py \
    --model east_model.onnx \
    --image path/to/test/image.jpg \
    --compare
```

### Параметры

- `--model`: Путь к ONNX модели (обязательный)
- `--image`: Путь к тестовому изображению (обязательный)
- `--compare`: Сравнить с PyTorch моделью
- `--weights`: Путь к PyTorch весам для сравнения
- `--output`: Путь для сохранения визуализации (default: `onnx_detection_result.jpg`)
- `--target-size`: Размер входа (должен совпадать с размером при экспорте)
- `--score-thresh`: Порог уверенности для детекций (default: 0.6)
- `--use-cuda`: Использовать CUDA provider для ONNX Runtime

### Пример полного workflow

```bash
# 1. Экспорт модели
python scripts/export_east_to_onnx.py --output east_640.onnx --input-size 640

# 2. Тест инференса
python scripts/test_east_onnx_inference.py \
    --model east_640.onnx \
    --image example/test_image.jpg \
    --target-size 640 \
    --output result.jpg

# 3. Сравнение с PyTorch
python scripts/test_east_onnx_inference.py \
    --model east_640.onnx \
    --image example/test_image.jpg \
    --target-size 640 \
    --compare
```

## Что делают скрипты

### export_east_to_onnx.py

1. Загружает PyTorch модель EAST
2. Оборачивает её в EASTWrapper (конвертирует dict output в tuple для ONNX)
3. Экспортирует в ONNX с поддержкой dynamic axes
4. Проверяет валидность ONNX модели
5. Опционально упрощает граф через onnx-simplifier

**Выходы ONNX модели:**
- `score_map`: [B, 1, H/4, W/4] - карта уверенности детекций
- `geo_map`: [B, 8, H/4, W/4] - карта геометрии (координаты четырехугольников)

### test_east_onnx_inference.py

1. Загружает ONNX модель через onnxruntime
2. Предобрабатывает входное изображение
3. Запускает инференс и замеряет время
4. Опционально сравнивает с PyTorch моделью
5. Декодирует детекции из карт
6. Сохраняет визуализацию результата

## Производительность

ONNX модель обычно работает быстрее PyTorch на CPU:
- PyTorch CPU: ~200-500ms на изображение 1280x1280
- ONNX CPU: ~100-300ms на изображение 1280x1280
- Speedup: 1.5-2x

Для GPU инференса рекомендуется:
```bash
pip install onnxruntime-gpu
python scripts/test_east_onnx_inference.py --model model.onnx --image img.jpg --use-cuda
```

## Интеграция

После экспорта ONNX модель можно использовать:
- В production сервисах (FastAPI, Flask)
- На мобильных устройствах (через ONNX Runtime Mobile)
- В браузере (через ONNX.js)
- На edge устройствах

## Troubleshooting

**Ошибка "onnxruntime not installed":**
```bash
pip install onnxruntime
```

**Разные результаты PyTorch vs ONNX:**
- Это нормально если разница < 1e-3
- Может быть вызвано разными версиями операторов
- Проверьте что используете одинаковый input_size

**Модель слишком большая:**
- Используйте onnx-simplifier для оптимизации
- Попробуйте квантизацию (int8)
- Используйте меньший backbone (но нужно переобучить)
