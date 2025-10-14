"""
Пример использования изолированного TRBAInfer из trba модуля.
Простой API с тремя параметрами: model_path, charset_path, config_path.
"""

import os
from src.manuscript.recognizers import TRBAInfer

def main():
    # Пути к файлам модели
    model_path = "C:\\Users\\USER\\Desktop\\OCR_MODELS\\exp_4_model_32\\best_acc_weights.pth"  # путь к весам модели
    charset_path = "src/manuscript/recognizers/trba/configs/charset.txt"          # путь к файлу с символами
    config_path = "C:\\Users\\USER\\Desktop\\OCR_MODELS\\exp_4_model_32\\config.json"          # путь к конфигурации
    
    # Проверяем существование файлов
    if not os.path.exists(model_path):
        print(f"❌ Файл модели не найден: {model_path}")
        return
    
    if not os.path.exists(charset_path):
        print(f"❌ Файл charset не найден: {charset_path}")
        return
        
    if not os.path.exists(config_path):
        print(f"❌ Файл конфига не найден: {config_path}")
        return
    
    try:
        # Инициализация модели - только 3 параметра!
        recognizer = TRBAInfer(
            model_path=model_path,
            charset_path=charset_path,
            config_path=config_path
        )
        
        print("✅ TRBAInfer успешно инициализирован")
        
        # Пример распознавания изображения
        image_path = "C:\\Users\\USER\\Desktop\\archive_25_09\\dataset\\printed\\val\\img\\images_group_1_1017_7866.png"
        
        if os.path.exists(image_path):
            # Простое распознавание - всегда возвращает список, берем [0]
            results = recognizer.predict(image_path)
            text, confidence = results[0]
            print(f"📝 Распознанный текст: '{text}' (уверенность: {confidence:.3f})")
            
            # Пакетное распознавание
            images = [image_path, image_path]  # пример с двумя изображениями
            results = recognizer.predict(images)
            print(f"📦 Пакетное распознавание:")
            for i, (text, conf) in enumerate(results):
                print(f"   {i+1}: '{text}' (уверенность: {conf:.3f})")
            
            print(f"⚙️  Используемые параметры из конфига:")
            print(f"   - max_length: {recognizer.max_length}")
            print(f"   - hidden_size: {recognizer.hidden_size}")
            print(f"   - img_size: {recognizer.img_h}x{recognizer.img_w}")
            
        else:
            print(f"⚠️  Тестовое изображение не найдено: {image_path}")
            print("💡 Создайте тестовое изображение или укажите правильный путь")
            
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")

if __name__ == "__main__":
    main()