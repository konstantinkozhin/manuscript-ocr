import cv2
import numpy as np
from PIL import Image
from manuscript.detectors import EASTInfer

# Инициализация модели
model = EASTInfer(
    weights_path=r"C:\east_quad_23_05.pth",
)

img_path = r"C:\shared\Archive_19_04\combined_images\911.jpg"

# Инференс с визуализацией (возвращает изображение с боксами и score map)
page, img_with_boxes, score_map_resized = model.infer(img_path, vis=True, profile=True)

img_with_boxes_pil = Image.fromarray(img_with_boxes)
img_with_boxes_pil.show()
