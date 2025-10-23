import cv2
import numpy as np
from PIL import Image
from manuscript.detectors import EASTInfer

# Инициализация модели
model = EASTInfer(score_thresh=0.9, 
                    #iou_threshold=1.2,
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
