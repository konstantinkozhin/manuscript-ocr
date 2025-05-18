import os
import json
import easyocr
from PIL import Image
import numpy as np
import time

# ----------------------------------------
# Укажите в Jupyter свои пути к данным:
# ----------------------------------------
INPUT_JSON = r"C:\data0205\Archives020525\test.json"
IMAGES_DIR = r"C:\data0205\Archives020525\test_images"
OUTPUT_JSON = r"C:\Users\pasha\OneDrive\Рабочий стол\Archives020525_test_craft.json"

# ---------------------------
# 1) Загрузка шаблона/GT-разметки
# ---------------------------
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

id2fname = {int(img["id"]): img["file_name"] for img in data.get("images", [])}

# ---------------------------
# 2) Инициализация easyOCR и сбор предсказаний
# ---------------------------
reader = easyocr.Reader(["en", "ru"], gpu=True)
all_preds = []
infer_times = []  # список для хранения времени инференса

for img_id, fname in id2fname.items():
    img_path = os.path.join(IMAGES_DIR, fname)
    if not os.path.exists(img_path):
        print(f"[WARN] не найден файл: {img_path}")
        continue

    # пробуем открыть через PIL
    try:
        pil_img = Image.open(img_path).convert("RGB")
        img_array = np.array(pil_img)
    except Exception as e:
        print(f"[WARN] не удалось открыть изображение {img_path}: {e}")
        continue

    # замер времени инференса EasyOCR (readtext на массиве)
    t0 = time.perf_counter()
    try:
        results = reader.readtext(img_array, detail=1)
    except Exception as e:
        print(f"[ERROR] при OCR на {img_path}: {e}")
        continue
    t1 = time.perf_counter()
    infer_times.append(t1 - t0)

    # формируем предсказания
    for bbox, text, confidence in results:
        segmentation = []
        xs, ys = [], []
        for x_pt, y_pt in bbox:
            x, y = float(x_pt), float(y_pt)
            segmentation.extend([x, y])
            xs.append(x)
            ys.append(y)

        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(xs), max(ys)
        w, h = x_max - x_min, y_max - y_min

        all_preds.append(
            {
                "image_id": img_id,
                "category_id": 0,
                "bbox": [x_min, y_min, w, h],
                "segmentation": [segmentation],
                "score": float(confidence),
                "text": text,
            }
        )

print(f"Собрано предсказаний: {len(all_preds)}")

# ---------------------------
# 3) Среднее время инференса
# ---------------------------
if infer_times:
    avg_time = sum(infer_times) / len(infer_times)
    print(f"Обработано изображений: {len(infer_times)}")
    print(f"Среднее время инференса: {avg_time:.3f} сек. на изображение")

# ---------------------------
# 4) Вставка предсказаний и сохранение
# ---------------------------
try:
    data["annotations"] = all_preds
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Результаты сохранены в {OUTPUT_JSON}")
except Exception as e:
    print(f"[ERROR] при сохранении JSON: {e}")
