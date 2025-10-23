import cv2
import numpy as np
from PIL import Image
from manuscript.detectors import EASTInfer
import random
from show_histogram import show_histogram

# Инициализация модели
model = EASTInfer(
    score_thresh=0.9,
    iou_threshold=0.2,
    use_tta=False,
    tta_merge_mode="mean",
    quantization=2,
)

img_path = r"C:\Users\pasha\OneDrive\Рабочий стол\scale_1200.png"

# Инференс с визуализацией (возвращает изображение с боксами и score map)
page, img_with_boxes, score_map_resized = model.infer(img_path, vis=True, profile=True)

img_with_boxes_pil = Image.fromarray(img_with_boxes)
img_with_boxes_pil.show()
