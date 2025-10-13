# для rehand 

import json
import sqlite3
import pandas as pd

# === Загрузка images из частично доступного JSON ===
with open(r"C:\Users\USER\Desktop\data02065\ICDAR2015\test.json", "r", encoding="utf-8") as f:
    raw_data = f.read(100000)

import re
images_match = re.search(r'"images"\s*:\s*(\[[^\]]+\])', raw_data, re.DOTALL)
images = json.loads(images_match.group(1)) if images_match else []

# === Маппинг file_name -> image metadata ===
image_meta = {
    img["file_name"]: {"id": img["id"], "width": img["width"], "height": img["height"]}
    for img in images
}

# === Загрузка аннотаций из SQLite ===
conn = sqlite3.connect(r"C:\Users\USER\Desktop\data02065\ICDAR2015\icdar.db")
df_ann = pd.read_sql_query("SELECT * FROM annotations", conn)
conn.close()

# === Формирование аннотаций в формате COCO ===
annotations_out = []
ann_id = 1
for _, row in df_ann.iterrows():
    fname = row["image_name"]
    if fname not in image_meta:
        continue

    meta = image_meta[fname]
    img_id = meta["id"]
    img_w, img_h = meta["width"], meta["height"]

    # Декодирование координат и размеров
    cx, cy = row["x_center"] * img_w, row["y_center"] * img_h
    w, h = row["width"] * img_w, row["height"] * img_h
    x0, y0 = cx - w / 2, cy - h / 2

    segmentation = [
        x0, y0,
        x0 + w, y0,
        x0 + w, y0 + h,
        x0, y0 + h
    ]

    annotations_out.append({
        "id": ann_id,
        "image_id": img_id,
        "category_id": 0,
        "bbox": [x0, y0, w, h],
        "area": w * h,
        "iscrowd": 0,
        "attributes": {
            "transcription": row["decoding"]
        },
        "segmentation": [segmentation]
    })
    ann_id += 1

# === Финальный JSON ===
output_data = {
    "images": images,
    "annotations": annotations_out
}

# === Сохранение ===
with open(r"C:\Users\USER\Desktop\data02065\ICDAR2015\test_rehand.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=4)
