#!/usr/bin/env python3
"""
Простой пример использования TRBAInfer с конфигурацией
Демонстрирует новый API с config.json
"""

import json
import os
from pathlib import Path

def create_example_config():
    """Создает пример конфигурации для демонстрации"""
    config = {
        "img_h": 32,
        "img_w": 128,
        "hidden_size": 256,
        "max_len": 40,
        "encoding": "utf-8"
    }
    
    os.makedirs("example/configs", exist_ok=True)
    
    config_path = "example/configs/demo_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print(f"✅ Создан пример конфигурации: {config_path}")
    return config_path

def example_with_config():
    """Пример использования TRBAInfer с конфигурацией"""
    
    print("🚀 Демонстрация TRBAInfer с config.json")
    print("=" * 50)
    
    # Создаем пример конфигурации
    config_path = create_example_config()
    
    print(f"\n📋 Содержимое конфигурации:")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print(f"\n🔧 Инициализация TRBAInfer:")
    print(f"Теперь достаточно передать только пути к модели и конфигу!")
    
    # Пример кода (закомментированный, так как нет реальной модели)
    example_code = '''
from manuscript import TRBAInfer

# Новый API - параметры модели из config.json, charset_path отдельно
recognizer = TRBAInfer(
    model_path="path/to/your/model.pth",
    config_path="example/configs/demo_config.json",
    charset_path="path/to/charset.txt"  # Отдельно, т.к. может быть где угодно
)

# Параметры img_h, img_w, hidden_size автоматически загружаются из конфига
# charset_path передается явно для большей гибкости

# Использование как обычно
text = recognizer.predict(image)
print(f"Результат: {text}")
'''
    
    print(example_code)
    
    print("✨ Преимущества нового API:")
    print("  • Параметры модели централизованы в config.json")  
    print("  • Не нужно дублировать img_h, img_w, hidden_size в коде")
    print("  • Легко переключаться между разными конфигурациями")
    print("  • charset_path передается явно для максимальной гибкости")
    print("  • Проверка обязательных параметров при загрузке")

if __name__ == "__main__":
    example_with_config()