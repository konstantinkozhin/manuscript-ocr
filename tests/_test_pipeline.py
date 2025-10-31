"""
Тесты для OCRPipeline - end-to-end пайплайна детекции и распознавания
"""

import pytest
import numpy as np
from PIL import Image
from unittest.mock import Mock, MagicMock

from manuscript import OCRPipeline
from manuscript.detectors import EASTInfer
from manuscript.detectors._types import Page, Block, Word
from manuscript.recognizers import TRBAInfer


class TestOCRPipeline:
    """Тесты для основного OCRPipeline класса"""

    @pytest.fixture
    def mock_detector(self):
        """Создает mock объект EASTInfer"""
        detector = Mock(spec=EASTInfer)

        # Создаем mock результат детекции
        mock_word = Mock()
        mock_word.polygon = [
            [10, 10],
            [110, 10],
            [110, 50],
            [10, 50],
        ]  # Прямоугольник 100x40

        mock_block = Mock()
        mock_block.words = [mock_word]

        mock_page = Mock()
        mock_page.blocks = [mock_block]

        detector.predict.return_value = {
            "page": mock_page,
            "vis_image": None,
            "score_map": None,
            "geo_map": None,
        }
        return detector

    @pytest.fixture
    def mock_recognizer(self):
        """Создает mock объект TRBAInfer"""
        recognizer = Mock(spec=TRBAInfer)
        recognizer.predict.return_value = "Тестовый текст"
        return recognizer

    @pytest.fixture
    def pipeline(self, mock_detector, mock_recognizer):
        """Создает OCRPipeline с mock объектами"""
        return OCRPipeline(
            detector=mock_detector, recognizer=mock_recognizer, min_text_size=5
        )

    @pytest.fixture
    def test_image(self):
        """Создает тестовое изображение"""
        # Создаем простое RGB изображение 200x100
        return np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)

    def test_pipeline_initialization(self, mock_detector, mock_recognizer):
        """Тест инициализации OCRPipeline"""
        pipeline = OCRPipeline(
            detector=mock_detector, recognizer=mock_recognizer, min_text_size=10
        )

        assert pipeline.detector == mock_detector
        assert pipeline.recognizer == mock_recognizer
        assert pipeline.min_text_size == 10

    def test_process_basic_page_output(self, pipeline, test_image):
        """Тест базового процесса с выводом Page объекта"""
        # Настраиваем recognizer для batch режима
        pipeline.recognizer.predict.return_value = [("Тестовый текст", 0.9)]

        result = pipeline.process(test_image)

        assert hasattr(result, "blocks")
        assert len(result.blocks) == 1
        assert len(result.blocks[0].words) == 1

        word = result.blocks[0].words[0]
        assert word.text == "Тестовый текст"
        assert word.recognition_confidence == 0.9

        # Проверяем что детектор и распознаватель были вызваны
        pipeline.detector.predict.assert_called_once()
        pipeline.recognizer.predict.assert_called_once()

    def test_process_detection_only(self, pipeline, test_image):
        """Тест процесса только с детекцией (без распознавания)"""
        result = pipeline.process(test_image, recognize_text=False)

        assert hasattr(result, "blocks")
        assert len(result.blocks) == 1
        assert len(result.blocks[0].words) == 1

        word = result.blocks[0].words[0]
        assert word.text is None  # Нет распознавания
        assert word.recognition_confidence is None

        # Только детектор должен быть вызван
        pipeline.detector.predict.assert_called_once()
        pipeline.recognizer.predict.assert_not_called()

    def test_get_text_from_page(self, pipeline, test_image):
        """Тест извлечения текста из Page объекта"""
        pipeline.recognizer.predict.return_value = [("Тестовый текст", 0.9)]

        page_result = pipeline.process(test_image)
        text_result = pipeline.get_text(page_result)

        assert isinstance(text_result, str)
        assert text_result == "Тестовый текст"

    def test_process_full_output(self, pipeline, test_image):
        """Тест процесса с полным выводом - теперь всегда возвращает Page"""
        pipeline.recognizer.predict.return_value = [("Тестовый текст", 0.88)]

        result = pipeline.process(test_image)

        assert hasattr(result, "blocks")
        assert len(result.blocks) == 1
        assert len(result.blocks[0].words) == 1

        word = result.blocks[0].words[0]
        assert word.text == "Тестовый текст"
        assert word.recognition_confidence == 0.88

    def test_process_no_text_detected(self, pipeline, test_image):
        """Тест случая когда текст не найден"""
        # Настраиваем детектор для возврата пустого результата
        from manuscript.detectors._types import Page

        empty_page = Page(blocks=[])
        pipeline.detector.predict.return_value = {
            "page": empty_page,
            "vis_image": None,
            "score_map": None,
            "geo_map": None,
        }

        result = pipeline.process(test_image)
        assert hasattr(result, "blocks")
        assert len(result.blocks) == 0

        # Recognizer не должен вызываться если нет детекций
        pipeline.recognizer.predict.assert_not_called()

    def test_get_text_from_empty_page(self, pipeline, test_image):
        """Тест извлечения текста из пустой страницы"""
        from manuscript.detectors._types import Page

        empty_page = Page(blocks=[])

        text_result = pipeline.get_text(empty_page)

        assert text_result == ""

    def test_process_small_text_regions_filtered(self, pipeline, test_image):
        """Тест фильтрации слишком маленьких областей текста"""
        # Создаем область меньше min_text_size
        mock_word = Mock()
        mock_word.polygon = [[10, 10], [12, 10], [12, 12], [10, 12]]  # 2x2 пикселя

        mock_block = Mock()
        mock_block.words = [mock_word]

        mock_page = Mock()
        mock_page.blocks = [mock_block]

        pipeline.detector.predict.return_value = {
            "page": mock_page,
            "vis_image": None,
            "score_map": None,
            "geo_map": None,
        }

        result = pipeline.process(test_image)
        assert result == ""  # Область слишком маленькая, должна быть отфильтрована

    def test_process_batch(self, pipeline):
        """Тест пакетной обработки"""
        pipeline.recognizer.predict.return_value = [("Тестовый текст", 0.9)]

        test_images = [
            np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8),
            np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8),
        ]

        results = pipeline.process_batch(test_images)

        assert len(results) == 2
        assert all(hasattr(r, "blocks") for r in results)

    def test_load_image_as_array_numpy(self, pipeline):
        """Тест загрузки numpy array"""
        test_array = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        result = pipeline._load_image_as_array(test_array)

        assert isinstance(result, np.ndarray)
        assert result.shape == test_array.shape
        np.testing.assert_array_equal(result, test_array)

    def test_load_image_as_array_pil(self, pipeline):
        """Тест загрузки PIL Image"""
        pil_image = Image.new("RGB", (200, 100), color="red")
        result = pipeline._load_image_as_array(pil_image)

        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 200, 3)

    def test_load_image_as_array_invalid_type(self, pipeline):
        """Тест обработки неподдерживаемого типа изображения"""
        with pytest.raises(TypeError, match="Неподдерживаемый тип изображения"):
            pipeline._load_image_as_array(12345)

    def test_multiple_text_regions(self, pipeline, test_image):
        """Тест обработки нескольких областей текста"""
        # Создаем несколько областей текста
        mock_word1 = Mock()
        mock_word1.polygon = [[10, 10], [110, 10], [110, 30], [10, 30]]

        mock_word2 = Mock()
        mock_word2.polygon = [[10, 40], [110, 40], [110, 60], [10, 60]]

        mock_block = Mock()
        mock_block.words = [mock_word1, mock_word2]

        mock_page = Mock()
        mock_page.blocks = [mock_block]

        pipeline.detector.predict.return_value = {
            "page": mock_page,
            "vis_image": None,
            "score_map": None,
            "geo_map": None,
        }

        result = pipeline.process(test_image)

        # Должен быть вызван recognizer для каждой области
        assert pipeline.recognizer.predict.call_count == 2
        assert "Тестовый текст" in result


class TestOCRPipelineImports:
    """Тесты импортов OCRPipeline"""

    def test_pipeline_import_from_manuscript(self):
        """Тест импорта OCRPipeline из основного пакета"""
        from manuscript import OCRPipeline

        assert OCRPipeline is not None

    def test_pipeline_import_direct(self):
        """Тест прямого импорта OCRPipeline"""
        from manuscript.pipeline import OCRPipeline

        assert OCRPipeline is not None
