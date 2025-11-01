import os
import time
from manuscript.recognizers import TRBA
from manuscript.recognizers._trba.training.metrics import (
    character_error_rate,
    word_error_rate,
    compute_accuracy,
)

# === Пути ===
image_dir = r"C:\shared\orig_cyrillic\test"
gt_path = r"C:\shared\orig_cyrillic\test.tsv"
model_path = (r"C:\Users\USER\Desktop\OCR_MODELS\exp1_model_64\best_acc_weights.pth",)
config_path = r"C:\Users\USER\Desktop\OCR_MODELS\exp1_model_64\config.json"

batch_size = 16

# === Читаем GT-файл ===
gt_data = {}
with open(gt_path, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("\t", 1)
        if len(parts) == 2:
            fname, text = parts
            gt_data[fname.strip()] = text.strip()

print(f"📄 Загрузили {len(gt_data)} строк из {os.path.basename(gt_path)}")

# === Сканируем изображения ===
valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
images = [
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    if os.path.splitext(f)[1].lower() in valid_ext
]

if not images:
    raise RuntimeError(f"❌ В папке {image_dir} не найдено изображений!")

print(f"📁 Найдено {len(images)} изображений для распознавания")

# === Инициализация модели ===
recognizer = TRBA(model_path=model_path, config_path=config_path)

# === Распознаём ===
start_time = time.perf_counter()
results = recognizer.predict(images=images, batch_size=batch_size, mode="greedy")
end_time = time.perf_counter()

total_time = end_time - start_time
avg_time = total_time / len(images)
fps = 1.0 / avg_time if avg_time > 0 else float("inf")

# === Сопоставляем с ground truth ===
refs, hyps = [], []
total_cer, total_wer = 0.0, 0.0
cer_count, wer_count = 0, 0

print("\n=== Результаты распознавания ===")
for path, (pred_text, score) in zip(images, results):
    fname = os.path.basename(path)
    ref_text = gt_data.get(fname)
    if ref_text is None:
        print(f"{fname:40s} → {pred_text:20s} (no GT)")
        continue

    refs.append(ref_text)
    hyps.append(pred_text)

    cer = character_error_rate(ref_text, pred_text)
    wer = word_error_rate(ref_text, pred_text)

    total_cer += cer
    total_wer += wer
    cer_count += 1
    wer_count += 1

    print(
        f"{fname:40s} → {pred_text:20s} | GT: {ref_text:20s} | CER={cer:.3f} | WER={wer:.3f}"
    )

# === Метрики ===
acc = compute_accuracy(refs, hyps)
avg_cer = total_cer / max(cer_count, 1)
avg_wer = total_wer / max(wer_count, 1)

print("\n=== Сводка ===")
print(f"Accuracy: {acc*100:.2f}%")
print(f"Avg CER:  {avg_cer:.4f}")
print(f"Avg WER:  {avg_wer:.4f}")
print(f"Processed {len(images)} images in {total_time:.3f} sec")
print(f"Average per image: {avg_time:.3f} sec ({fps:.1f} FPS)")
