import cv2
import numpy as np
from PIL import Image
from manuscript.detectors import EASTInfer
import random
from show_histogram import show_histogram

# Инициализация модели
model = EASTInfer(score_thresh=0.9, 
                    iou_threshold=1.2,
                    use_tta=True,
                    tta_merge_mode="mean"
                   )

img_path = r"C:\Users\USER\Desktop\верифицированные архъивы\combined_images\48.jpg"

# Инференс с визуализацией (возвращает изображение с боксами и score map)
page, img_with_boxes, score_map_resized = model.infer(img_path, vis=True, profile=True)

print(f"Детектировано боксов: {len(page.blocks[0].words)}")
print(f"Изображение с боксами: {img_with_boxes.shape}")
print(f"Score map размер: {score_map_resized.shape}")
print(f"Score map min/max: {score_map_resized.min():.3f}/{score_map_resized.max():.3f}")

# Вычисляем средний score для каждого бокса по score map
box_scores = []
for word in page.blocks[0].words:
    polygon = np.array(word.polygon, dtype=np.int32)
    
    # Создаем маску для текущего бокса
    mask = np.zeros(score_map_resized.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    
    # Вычисляем средний score в области бокса
    masked_scores = score_map_resized[mask > 0]
    avg_score = masked_scores[masked_scores > np.quantile(masked_scores, 0.05)].mean() if len(masked_scores) > 0 else 0.0
    box_scores.append(avg_score)
    
    # Находим центр бокса для отображения текста
    center = polygon.mean(axis=0).astype(int)
    
    # Рисуем score на изображении
    text = f"{avg_score:.2f}"
    cv2.putText(img_with_boxes, text, tuple(center), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

print(f"Box scores (mean): min={min(box_scores):.3f}, max={max(box_scores):.3f}, avg={np.mean(box_scores):.3f}")

# Нормализуем score map в диапазон [0, 255]
score_map_normalized = (score_map_resized * 255).astype(np.uint8)

# Применяем цветовую карту (heatmap) для лучшей визуализации
score_map_colored = cv2.applyColorMap(score_map_normalized, cv2.COLORMAP_JET)
score_map_colored_rgb = cv2.cvtColor(score_map_colored, cv2.COLOR_BGR2RGB)

# Накладываем score map на изображение с боксами с альфа-каналом
alpha = 0.2  # прозрачность score map (0.0 - только боксы, 1.0 - только score map)
overlay = cv2.addWeighted(img_with_boxes, 1 - alpha, score_map_colored_rgb, alpha, 0)

# Показываем результат
Image.fromarray(overlay).show()

# Опционально: сохраняем результат
# Image.fromarray(overlay).save("boxes_and_scoremap_overlay.jpg")

print("Визуализация боксов и score map с альфа-наложением завершена!")

# Выводим гистограмму score map для случайного бокса
if len(page.blocks[0].words) > 0:
    idx = random.randint(0, len(page.blocks[0].words) - 1)
    polygon = np.array(page.blocks[0].words[idx].polygon, dtype=np.int32)
    mask = np.zeros(score_map_resized.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 1)
    masked_scores = score_map_resized[mask > 0]
    show_histogram(masked_scores, title=f"Score Histogram for Box #{idx}")
