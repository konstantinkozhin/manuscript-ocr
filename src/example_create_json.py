import os
import json
import time
import numpy as np
from PIL import Image
from manuscript.detectors import EASTInfer
from manuscript.detectors.east.utils import (compute_f1_metrics, load_gt, load_preds)
from tqdm import tqdm


IMAGES_DIRS = [
    r"C:\Users\USER\Desktop\data02065\School\test_images",
    r"C:\Users\USER\Desktop\data02065\ICDAR2015\test_images"
]

INPUT_JSONS = [
    r"C:\Users\USER\Desktop\data02065\School\test.json",
    r"C:\Users\USER\Desktop\data02065\ICDAR2015\test.json"
]

# ---------------------------
# Основной цикл по папкам
# ---------------------------

for INPUT_JSON, IMAGES_DIR in zip(INPUT_JSONS, IMAGES_DIRS):
    print(f"\n=== Обработка: {IMAGES_DIR} ===")
    print(f"Загрузка разметки: {INPUT_JSON}")

    # 1) Загрузка разметки
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    id2fname = {img["id"]: img["file_name"] for img in data.get("images", [])}
    print(f"Изображений: {len(id2fname)}")

    # 2) Инициализация модели
    det = EASTInfer()
    all_preds = []
    infer_times = []

    # 3) Цикл по изображениям
    for img_id, fname in tqdm(id2fname.items(), desc="Инференс", unit="изобр"):
        img_path = os.path.join(IMAGES_DIR, fname)
        if not os.path.exists(img_path):
            print(f"[WARN] Не найден файл: {img_path}")
            continue

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
                segmentation = []
                for px, py in word.polygon:
                    segmentation.extend([px * sx, py * sy])

                all_preds.append(
                    {
                        "image_id": img_id,
                        "segmentation": [segmentation],
                        "score": getattr(word, "score", 1.0),
                    }
                )

    print(f"Собрано предсказаний: {len(all_preds)}")

    # 4) Среднее время
    if infer_times:
        avg_time = sum(infer_times) / len(infer_times)
        print(f"Обработано изображений: {len(infer_times)}")
        print(f"Среднее время инференса: {avg_time:.3f} сек.")

    # 5) Сохранение результатов
    OUTPUT_JSON = os.path.join(os.path.dirname(IMAGES_DIR), "test_east.json")
    data["annotations"] = all_preds

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"Результаты сохранены в {OUTPUT_JSON}")

    gt_segs = load_gt(INPUT_JSON)
    preds = load_preds(OUTPUT_JSON)
    processed_ids = list(id2fname.keys())

    f1_05, f1_avg = compute_f1_metrics(preds, gt_segs, processed_ids)

    print(f"[METRICS] F1@0.5: {f1_05:.3f}")
    print(f"[METRICS] F1@[0.5:0.95]: {f1_avg:.3f}")
