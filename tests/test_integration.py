"""
Интеграционные тесты для manuscript-ocr
Тестируют работу детектора на реальном примере
"""
import os
import pytest
from pathlib import Path
from PIL import Image
import numpy as np

from manuscript.detectors import EASTInfer


class TestIntegration:
    """Интеграционные тесты для полного pipeline"""
    
    @pytest.fixture
    def example_image_path(self):
        """Путь к тестовому изображению"""
        repo_root = Path(__file__).parent.parent.parent
        image_path = repo_root / "example" / "ocr_example_image.jpg"
        
        if not image_path.exists():
            pytest.skip("Тестовое изображение example/ocr_example_image.jpg не найдено")
        
        return str(image_path)
    
    @pytest.fixture
    def detector(self):
        """Инициализация детектора"""
        # Используем низкий порог для теста
        return EASTInfer(score_thresh=0.5)
    
    def test_detector_initialization(self, detector):
        """Тест инициализации детектора"""
        assert detector is not None
        assert hasattr(detector, 'infer')
        assert hasattr(detector, 'device')
        
    def test_detector_inference_basic(self, detector, example_image_path):
        """Базовый тест inference без визуализации"""
        # Проверяем, что inference работает без ошибок
        page = detector.infer(example_image_path, vis=False)
        
        # Проверяем структуру результата
        assert hasattr(page, 'blocks')
        assert isinstance(page.blocks, list)
        
        # Логируем количество найденных блоков
        print(f"Найдено текстовых блоков: {len(page.blocks)}")
        
        # Проверяем, что найден хотя бы один блок (может быть 0 при очень строгом пороге)
        if len(page.blocks) > 0:
            block = page.blocks[0]
            assert hasattr(block, 'words')
            assert isinstance(block.words, list)
            
            if len(block.words) > 0:
                word = block.words[0]
                assert hasattr(word, 'polygon')
                assert isinstance(word.polygon, list)
                assert len(word.polygon) == 4  # 4 угла прямоугольника
                
    def test_detector_inference_with_visualization(self, detector, example_image_path):
        """Тест inference с визуализацией"""
        page, vis_image = detector.infer(example_image_path, vis=True)
        
        # Проверяем структуру результата
        assert hasattr(page, 'blocks')
        assert vis_image is not None
        assert isinstance(vis_image, np.ndarray)
        
        # Проверяем размерность изображения (высота, ширина, каналы)
        assert len(vis_image.shape) == 3
        assert vis_image.shape[2] == 3  # RGB
        
        print(f"Размер визуализации: {vis_image.shape}")
        
    def test_different_score_thresholds(self, example_image_path):
        """Тест различных порогов детекции"""
        thresholds = [0.3, 0.5, 0.7, 0.9]
        results = []
        
        for thresh in thresholds:
            detector = EASTInfer(score_thresh=thresh)
            page = detector.infer(example_image_path, vis=False)
            block_count = len(page.blocks)
            results.append((thresh, block_count))
            print(f"Порог {thresh}: найдено {block_count} блоков")
        
        # Проверяем, что с увеличением порога количество блоков не растет
        # (более строгий порог = меньше или равно блоков)
        for i in range(1, len(results)):
            curr_thresh, curr_count = results[i]
            prev_thresh, prev_count = results[i-1]
            assert curr_thresh > prev_thresh
            assert curr_count <= prev_count, f"При пороге {curr_thresh} найдено {curr_count} блоков, при {prev_thresh} - {prev_count}"


class TestInstallation:
    """Тесты корректности установки пакета"""
    
    def test_import_main_module(self):
        """Тест импорта основного модуля"""
        from manuscript.detectors import EASTInfer
        assert EASTInfer is not None
        
    def test_import_types(self):
        """Тест импорта типов данных"""
        from manuscript.detectors._types import Word, Block, Page
        assert Word is not None
        assert Block is not None 
        assert Page is not None
        
    def test_torch_availability(self):
        """Тест доступности PyTorch"""
        import torch
        assert torch is not None
        
        # Проверяем версию (должна быть >= 1.11.0)
        version = torch.__version__.split('+')[0]  # убираем +cu118 если есть
        major, minor = map(int, version.split('.')[:2])
        assert major >= 1
        if major == 1:
            assert minor >= 11
            
        print(f"PyTorch версия: {torch.__version__}")
        print(f"CUDA доступна: {torch.cuda.is_available()}")
        
    def test_other_dependencies(self):
        """Тест остальных зависимостей"""
        import numpy as np
        import cv2
        from PIL import Image
        import shapely
        import numba
        import pydantic
        import gdown
        
        # Все импорты прошли успешно
        print("Все основные зависимости импортированы успешно")


if __name__ == "__main__":
    # Для локального запуска
    pytest.main([__file__, "-v"])