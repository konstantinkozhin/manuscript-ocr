import time
from manuscript.recognizers import TRBAInfer

# === Инициализация модели ===
recognizer = TRBAInfer(
    model_path=r"C:\shared\exp1_model_64\best_acc_ckpt.pth",
    config_path=r"C:\shared\exp1_model_64\config.json",
)

# === Список изображений ===
images = [
    r"C:\Screenshot 2025-10-25 180736.png",
    r"C:\Screenshot 2025-10-25 180134.png",
    r"C:\shared\Archive_19_04\data_hkr\test\hrk_23.png",
    r"C:\shared\Archive_19_04\data_cyrillic\test\cyrillic_52.png",
    r"C:\shared\Archive_19_04\data_archive\test\archive_288.png",
    r"C:\shared\Archive_19_04\data_archive\test\archive_31898.png",
    r"C:\Screenshot 2025-10-25 210128.png",
]

# === Измеряем время ===
start_time = time.perf_counter()
res = recognizer.predict(images=images, batch_size=16)
end_time = time.perf_counter()

# === Вывод результатов ===
total_time = end_time - start_time
avg_time = total_time / len(images)
fps = 1.0 / avg_time if avg_time > 0 else float("inf")

print("\n=== Результаты распознавания ===")
for text, score in res:
    print(f"Recognized: {text}, confidence: {score:.4f}")

print(f"\nProcessed {len(images)} images in {total_time:.3f} sec")
print(f"Average per image: {avg_time:.3f} sec ({fps:.1f} FPS)")
