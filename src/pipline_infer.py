from manuscript import Pipeline

# Инициализация с моделями по умолчанию
pipeline = Pipeline()

# Обработка изображения
result = pipeline.predict("document.jpg")

# Извлечение текста
text = pipeline.get_text(result)
print(text)
