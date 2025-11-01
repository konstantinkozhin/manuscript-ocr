from manuscript import Pipeline


# Путь к изображению (укажите свой файл)
image_path = r"C:\Users\pasha\OneDrive\Рабочий стол\i (1).jpg"

# Создание OCR-пайплайна с моделями по умолчанию
pipeline = Pipeline()

# Обработка изображения и получение результата
result, img = pipeline.predict(image_path, vis=True)

# Визуализация результата
img.show()

print(result)
