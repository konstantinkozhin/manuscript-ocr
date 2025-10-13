# использую это для рехенда но подходит для любых json файлов
# gt_path - эталонная, ручная разметка
# pred_path - предсказания

from manuscript.detectors.east.utils import (compute_f1_metrics, load_gt, load_preds)
import os

# Пути к JSON-файлам
gt_path = r"C:\Users\USER\Desktop\data02065\Archives020525\test_clean.json"
pred_path = r"C:\Users\USER\Desktop\data02065\Archives020525\test_rehand.json"

# Проверка существования
if not os.path.exists(gt_path) or not os.path.exists(pred_path):
    print("GT или предсказания не найдены")
    exit()

# Загрузка данных
gt_segs = load_gt(gt_path)
preds = load_preds(pred_path)
processed_ids = list(gt_segs.keys())

# Вычисление F1
f1_05, f1_avg = compute_f1_metrics(preds, gt_segs, processed_ids)

# Вывод
print(f"[METRICS] F1@0.5:        {f1_05:.3f}")
print(f"[METRICS] F1@[0.5:0.95]: {f1_avg:.3f}")
