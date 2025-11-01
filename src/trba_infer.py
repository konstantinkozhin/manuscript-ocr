import time
from manuscript.recognizers import TRBA

# === Инициализация модели ===
recognizer = TRBA()

# === Список изображений ===
images = [
    #    r"C:\Users\USER\Desktop\archive_25_09\dataset\handwritten\val\img\images_group_39_632_10901.png",
    #    r"C:\Users\USER\Desktop\archive_25_09\dataset\handwritten\val\img\images_group_39_632_10903.png",
    #    r"C:\Users\USER\Desktop\archive_25_09\dataset\handwritten\val\img\images_group_39_635_11456.png",
    r"C:\shared\Archive_19_04\data_archive\test\archive_15.png",
]

# === Измеряем время ===
start_time = time.perf_counter()
import numpy as np

res = recognizer.predict(
    images=images, batch_size=16, mode="beam"  # 'greedy' or 'beam'
)
end_time = time.perf_counter()

# === Вывод результатов ===
total_time = end_time - start_time
avg_time = total_time / len(images)
fps = 1.0 / avg_time if avg_time > 0 else float("inf")

print("\n=== Результаты распознавания ===")
for result in res:
    text = result["text"]
    score = result["confidence"]
    print(f"Recognized: {text}, confidence: {score:.4f}")

print(f"\nProcessed {len(images)} images in {total_time:.3f} sec")
print(f"Average per image: {avg_time:.3f} sec ({fps:.1f} FPS)")


res = recognizer.predict(
    images=images, batch_size=16, mode="greedy"  # 'greedy' or 'beam'
)
end_time = time.perf_counter()

# === Вывод результатов ===
total_time = end_time - start_time
avg_time = total_time / len(images)
fps = 1.0 / avg_time if avg_time > 0 else float("inf")

print("\n=== Результаты распознавания ===")
for result in res:
    text = result["text"]
    score = result["confidence"]
    print(f"Recognized: {text}, confidence: {score:.4f}")

print(f"\nProcessed {len(images)} images in {total_time:.3f} sec")
print(f"Average per image: {avg_time:.3f} sec ({fps:.1f} FPS)")
