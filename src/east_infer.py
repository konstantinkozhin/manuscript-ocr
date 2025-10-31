import cv2
import numpy as np
from PIL import Image
from manuscript.detectors import EASTInfer

# Инициализация модели
model = EASTInfer()

img_path = r"C:\Users\USER\Desktop\Для отчета\2907.jpg"

# Инференс с визуализацией (возвращает изображение с боксами и score map)
page, img_with_boxes = model.predict(img_path, vis=True, profile=True)

img_with_boxes_pil = Image.fromarray(img_with_boxes)
img_with_boxes_pil.show()
