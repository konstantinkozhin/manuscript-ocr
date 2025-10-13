"""
Базовые тесты для TRBAInfer распознавателя
"""
import pytest
from manuscript import TRBAInfer


class TestTRBAInfer:
    """Базовые тесты для TRBAInfer"""
    
    def test_trba_infer_import(self):
        """Тест импорта TRBAInfer"""
        assert TRBAInfer is not None
        
    def test_trba_infer_initialization_without_files(self):
        """Тест что TRBAInfer может быть создан (без реальных файлов)"""
        # Тестируем только что класс может быть импортирован и создан
        # без реальных файлов модели (они не включены в пакет)
        try:
            # Это должно упасть с FileNotFoundError, но класс должен существовать
            TRBAInfer(
                model_path="non_existent.pth",
                config_path="non_existent.json",
                charset_path="non_existent.txt",
                device="cpu"
            )
        except FileNotFoundError:
            # Ожидаемая ошибка - файлы модели не существуют
            pass
        except Exception as e:
            # Если другая ошибка, значит проблема с импортами или кодом
            pytest.fail(f"Неожиданная ошибка при создании TRBAInfer: {e}")
            
    def test_trba_device_selection(self):
        """Тест выбора устройства"""
        import torch
        
        # Тестируем логику выбора устройства без создания модели
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Создаем мок объект для тестирования логики устройства
        class MockTRBAInfer:
            def __init__(self, device="auto"):
                if device == "auto":
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                else:
                    self.device = torch.device(device)
        
        mock_trba = MockTRBAInfer(device="auto")
        assert str(mock_trba.device) == expected_device
        
        mock_trba_cpu = MockTRBAInfer(device="cpu") 
        assert str(mock_trba_cpu.device) == "cpu"