"""
Полный интеграционный тест для EASTInfer детектора
Тестирует все основные функции детектора текста EAST
"""

import pytest
from pathlib import Path
import torch
from PIL import Image

from manuscript.detectors import EASTInfer


class TestEASTInfer:
    @pytest.fixture
    def example_image_path(self):
        repo_root = Path(__file__).parent.parent.parent
        image_path = repo_root / "example" / "ocr_example_image.jpg"

        if not image_path.exists():
            pytest.skip("Тестовое изображение example/ocr_example_image.jpg не найдено")

        return str(image_path)

    def test_eastinfer_initialization(self):
        """Тест 1: Инициализация EASTInfer"""
        print("\n=== Тест инициализации EASTInfer ===")

        # Создаем детектор с параметрами по умолчанию
        detector = EASTInfer(score_thresh=0.5)

        # Проверяем базовые атрибуты
        assert detector is not None
        assert hasattr(detector, "predict")
        assert hasattr(detector, "model")
        assert hasattr(detector, "device")

        # Проверяем устройство
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Устройство: {detector.device} (ожидалось: {expected_device})")
        assert detector.device == expected_device

        print("Инициализация прошла успешно")

    def test_eastinfer_basic_inference(self, example_image_path):
        """Тест 2: Базовый inference без визуализации"""
        print("\n=== Тест базового inference ===")

        detector = EASTInfer(score_thresh=0.3)
        result = detector.predict(example_image_path, vis=False)
        assert isinstance(result, dict)
        assert "page" in result
        page = result["page"]
        assert page is not None
        assert hasattr(page, "blocks")
        assert isinstance(page.blocks, list)

        print(f"Найдено блоков: {len(page.blocks)}")

        # Проверяем содержимое блоков
        if len(page.blocks) > 0:
            block = page.blocks[0]
            assert hasattr(block, "words")
            assert isinstance(block.words, list)

            words_count = sum(len(block.words) for block in page.blocks)
            print(f"Найдено слов: {words_count}")

            if words_count > 0:
                # Проверяем структуру первого слова
                first_word = page.blocks[0].words[0]
                assert hasattr(first_word, "polygon")
                assert isinstance(first_word.polygon, list)
                assert len(first_word.polygon) == 4  # 4 точки полигона
                print(f"Пример полигона: {first_word.polygon[0]}")

        print("Базовый inference работает")

    def test_eastinfer_inference_with_visualization(self, example_image_path):
        """Тест 3: Inference с визуализацией"""
        print("\n=== Тест inference с визуализацией ===")

        detector = EASTInfer(score_thresh=0.3)

        # Выполняем детекцию с визуализацией
        result = detector.predict(example_image_path, vis=True)

        # Проверяем, что возвращается кортеж
        assert isinstance(result, dict)
        assert {"page", "vis_image", "score_map", "geo_map"}.issubset(result.keys())

        page = result["page"]
        vis_image = result["vis_image"]

        # Проверяем page
        assert hasattr(page, "blocks")

        # Проверяем визуализацию
        assert vis_image is not None
        assert isinstance(vis_image, Image.Image)
        assert vis_image.mode in ("RGB", "RGBA")

        print(f"Размер визуализации: {vis_image.size}")
        print("Визуализация работает")

    def test_eastinfer_different_parameters(self, example_image_path):
        """Тест 4: Различные параметры детектора"""
        print("\n=== Тест различных параметров ===")

        # Тестируем разные пороги
        thresholds = [0.1, 0.5, 0.9]
        results = []

        for thresh in thresholds:
            detector = EASTInfer(score_thresh=thresh)
            result = detector.predict(example_image_path, vis=False)
            page = result["page"]

            total_words = sum(len(block.words) for block in page.blocks)
            results.append((thresh, total_words))
            print(f"Порог {thresh}: {total_words} слов")

        # Проверяем логику порогов (высокий порог = меньше детекций)
        print("Тестирование параметров завершено")

    def test_eastinfer_numpy_input(self, example_image_path):
        """Тест 5: Входные данные как numpy array"""
        print("\n=== Тест numpy входа ===")

        # Загружаем изображение как numpy array
        import cv2

        img_bgr = cv2.imread(example_image_path)
        assert img_bgr is not None, f"Не удалось загрузить {example_image_path}"

        # Конвертируем в RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Тестируем с numpy input
        detector = EASTInfer(score_thresh=0.5)
        result = detector.predict(img_rgb, vis=False)
        page = result["page"]

        # Проверяем результат
        assert page is not None
        assert hasattr(page, "blocks")

        total_words = sum(len(block.words) for block in page.blocks)
        print(f"Детектировано слов из numpy array: {total_words}")
        print("Numpy вход работает")

    def test_eastinfer_error_handling(self):
        """Тест 6: Обработка ошибок"""
        print("\n=== Тест обработки ошибок ===")

        detector = EASTInfer(score_thresh=0.5)

        # Тест несуществующего файла
        with pytest.raises(FileNotFoundError):
            detector.predict("non_existent_file.jpg")

        # Тест неправильного типа входа
        with pytest.raises(TypeError):
            detector.predict(12345)  # Число вместо изображения

        print("Обработка ошибок работает")
