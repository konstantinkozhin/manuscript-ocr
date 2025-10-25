import time
import os
from manuscript.recognizers import TRBAInfer

# === Настройки ===
image_dir = (
    r"C:\shared\Archive_19_04\data_cyrillic\test"  # 🟢 путь к папке с изображениями
)
batch_size = 16

# === Сканируем папку ===
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
recognizer = TRBAInfer(
    model_path=r"C:\shared\exp1_model_64\best_acc_ckpt.pth",
    config_path=r"C:\shared\exp1_model_64\config.json",
)

# === Измеряем время ===
start_time = time.perf_counter()
res = recognizer.predict(images=images, batch_size=batch_size)
end_time = time.perf_counter()

# === Вывод результатов ===
total_time = end_time - start_time
avg_time = total_time / len(images)
fps = 1.0 / avg_time if avg_time > 0 else float("inf")

print("\n=== Результаты распознавания ===")
for path, (text, score) in zip(images, res):
    print(f"{os.path.basename(path):40s} → {text:20s}  ({score:.4f})")

print(f"\nProcessed {len(images)} images in {total_time:.3f} sec")
print(f"Average per image: {avg_time:.3f} sec ({fps:.1f} FPS)")
