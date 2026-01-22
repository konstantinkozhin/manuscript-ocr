import os
import time
import numpy as np
from manuscript import Pipeline
from manuscript.utils import visualize_page


image_path = "example/ocr_example_image.jpg"

# Инициализация детектора
pipeline = Pipeline()

result = pipeline.predict(image_path)

img = visualize_page(
    image_path,
    result,
)

img.show()

