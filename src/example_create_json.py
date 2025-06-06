import os
import json
import time
import numpy as np
from PIL import Image
from manuscript.detectors import EASTInfer

# ----------------------------------------
# Укажите в Jupyter свои пути к данным:
# ----------------------------------------
INPUT_JSON = r"C:\data0205\Archives020525\train.json"
IMAGES_DIR = r"C:\data0205\Archives020525\train_images"
OUTPUT_JSON = r"C:\Users\pasha\OneDrive\Рабочий стол\Archives020525_train.json"

# ---------------------------
# 1) Загрузка шаблона/GT-разметки
# ---------------------------
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

id2fname = {img["id"]: img["file_name"] for img in data.get("images", [])}

# ---------------------------
# 2) Инференс и сбор предсказаний
# ---------------------------
det = EASTInfer(shrink_ratio=0.0, score_thresh=0.9)
all_preds = []

# список для хранения времён инференса
infer_times = []

for img_id, fname in id2fname.items():
    img_path = os.path.join(IMAGES_DIR, fname)
    if not os.path.exists(img_path):
        print(f"[WARN] не найден файл: {img_path}")
        continue

    # замер времени инференса
    t0 = time.perf_counter()
    page, vis = det.infer(img_path, vis=True)
    t1 = time.perf_counter()
    infer_times.append(t1 - t0)

    orig_w, orig_h = Image.open(img_path).size
    vis_w, vis_h = (
        (vis.shape[1], vis.shape[0])
        if isinstance(vis, np.ndarray)
        else (vis.width, vis.height)
    )
    sx, sy = orig_w / vis_w, orig_h / vis_h

    for block in page.blocks:
        for word in block.words:
            xs = [p[0] * sx for p in word.polygon]
            ys = [p[1] * sy for p in word.polygon]
            x, y = min(xs), min(ys)
            w, h = max(xs) - x, max(ys) - y

            segmentation = []
            for px, py in word.polygon:
                segmentation.extend([px * sx, py * sy])

            all_preds.append(
                {
                    "image_id": img_id,
                    "category_id": 0,
                    "bbox": [x, y, w, h],
                    "segmentation": [segmentation],
                    "score": getattr(word, "score", 1.0),
                }
            )

print(f"Собрано предсказаний: {len(all_preds)}")

# ---------------------------
# 3) Среднее время инференса
# ---------------------------
if infer_times:
    avg_time = sum(infer_times) / len(infer_times)
    print(f"Обработано изображений: {len(infer_times)}")
    print(f"Среднее время инференса: {avg_time:.3f} сек.")

# ---------------------------
# 4) Вставка предсказаний и сохранение
# ---------------------------
data["annotations"] = all_preds
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print(f"Результаты сохранены в {OUTPUT_JSON}")
