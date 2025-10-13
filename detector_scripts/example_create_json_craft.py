import os
import json
import time
import numpy as np
import psutil
from PIL import Image
import easyocr
from tqdm import tqdm
from manuscript.detectors.east.utils import compute_f1_metrics, load_gt, load_preds

# --- Пути к архивам и json ---
IMAGES_DIRS = [
    r"C:\Users\USER\Desktop\data02065\Archives020525\test_images",
    r"C:\Users\USER\Desktop\data02065\TotalText\test_images",
    r"C:\Users\USER\Desktop\data02065\IAM\test_images",
    r"C:\Users\USER\Desktop\data02065\School\test_images",
    r"C:\Users\USER\Desktop\data02065\ICDAR2015\test_images"
]

INPUT_JSONS = [
    r"C:\Users\USER\Desktop\data02065\Archives020525\test.json",
    r"C:\Users\USER\Desktop\data02065\TotalText\test_cleaned.json",
    r"C:\Users\USER\Desktop\data02065\IAM\test_corrected.json",
    r"C:\Users\USER\Desktop\data02065\School\test.json",
    r"C:\Users\USER\Desktop\data02065\ICDAR2015\test.json"
]

# --- OCR инициализация ---
reader = easyocr.Reader(["en", "ru"], gpu=True)

# --- Основной цикл ---
for INPUT_JSON, IMAGES_DIR in zip(INPUT_JSONS, IMAGES_DIRS):
    print(f"\n=== Обработка: {IMAGES_DIR} ===")
    print(f"Загрузка разметки: {INPUT_JSON}")

    process = psutil.Process(os.getpid())
    peak_mem_mb = 0

    # 1. Загрузка аннотаций
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    id2fname = {int(img["id"]): img["file_name"] for img in data.get("images", [])}
    print(f"Изображений: {len(id2fname)}")

    all_preds = []
    infer_times = []

    # 2. Инференс по изображениям
    for img_id, fname in tqdm(id2fname.items(), desc="Инференс", unit="изобр"):
        img_path = os.path.join(IMAGES_DIR, fname)
        if not os.path.exists(img_path):
            print(f"[WARN] Не найден файл: {img_path}")
            continue

        try:
            pil_img = Image.open(img_path).convert("RGB")
            img_array = np.array(pil_img)
        except Exception as e:
            print(f"[WARN] Ошибка открытия изображения: {e}")
            continue

        t0 = time.perf_counter()
        try:
            DETECT_PARAMS = {'text_threshold': 0.5213592572934042, 'low_text': 0.36159995700141034, 'link_threshold': 0.3211728225955032, 'canvas_size': 2560, 'mag_ratio': 2.457119400680926, 'slope_ths': 0.4065218089501456, 'ycenter_ths': 0.0, 'height_ths': 0.7308397742828456, 'width_ths': 0.6589587170590814, 'add_margin': 0.07212504180872058}
            results = reader.readtext(img_array, detail=1, **DETECT_PARAMS)
        except Exception as e:
            print(f"[ERROR] Ошибка OCR: {e}")
            continue
        t1 = time.perf_counter()
        infer_times.append(t1 - t0)

        mem = process.memory_info().rss / 1024 / 1024
        if mem > peak_mem_mb:
            peak_mem_mb = mem

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

    if infer_times:
        avg_time = sum(infer_times) / len(infer_times)
        print(f"Обработано изображений: {len(infer_times)}")
        print(f"Среднее время инференса: {avg_time:.3f} сек.")

    print(f"[INFO] Максимальное потребление памяти: {peak_mem_mb:.1f} МБ")

    # 3. Сохранение результатов
    OUTPUT_JSON = os.path.join(os.path.dirname(IMAGES_DIR), "test_easyocr_opt.json")
    data["annotations"] = all_preds
    try:
        with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Результаты сохранены в {OUTPUT_JSON}")
    except Exception as e:
        print(f"[ERROR] При сохранении: {e}")
        continue

    # 4. Метрики
    gt_segs = load_gt(INPUT_JSON)
    preds = load_preds(OUTPUT_JSON)
    processed_ids = list(id2fname.keys())

    f1_05, f1_avg = compute_f1_metrics(preds, gt_segs, processed_ids)
    print(f"[METRICS] F1@0.5: {f1_05:.3f}")
    print(f"[METRICS] F1@[0.5:0.95]: {f1_avg:.3f}")
