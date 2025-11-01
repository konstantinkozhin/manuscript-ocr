"""
Тесты совместимости Pipeline API с различными реализациями
детекторов и распознавателей
"""

import pytest
import numpy as np
from PIL import Image
from typing import List, Tuple, Union, Dict, Any

from manuscript import Pipeline
from manuscript.detectors._types import Word, Block, Page


class DummyDetector:
    """
    Минимальная реализация детектора для проверки совместимости API.

    Реализует только необходимый интерфейс:
    - метод predict(image, vis=False, profile=False)
    - возвращает dict с ключом "page" содержащим объект Page
    """

    def __init__(self, return_type: str = "dict"):
        """
        Parameters
        ----------
        return_type : {"dict", "tuple", "page"}
            Формат возвращаемого значения для проверки всех вариантов
        """
        self.return_type = return_type

    def predict(
        self,
        image: Union[str, np.ndarray, Image.Image],
        vis: bool = False,
        profile: bool = False,
    ) -> Union[Dict[str, Any], Tuple[Page, Any], Page]:
        """
        Возвращает фиксированный результат детекции с 3 словами.
        """
        # Создаем 3 фиксированных слова
        words = [
            Word(
                polygon=[[10.0, 10.0], [100.0, 10.0], [100.0, 50.0], [10.0, 50.0]],
                detection_confidence=0.95,
            ),
            Word(
                polygon=[[110.0, 10.0], [200.0, 10.0], [200.0, 50.0], [110.0, 50.0]],
                detection_confidence=0.92,
            ),
            Word(
                polygon=[[210.0, 10.0], [300.0, 10.0], [300.0, 50.0], [210.0, 50.0]],
                detection_confidence=0.88,
            ),
        ]

        block = Block(words=words)
        page = Page(blocks=[block])

        # Возвращаем в разных форматах в зависимости от настройки
        if self.return_type == "dict":
            return {"page": page, "vis_image": None, "score_map": None, "geo_map": None}
        elif self.return_type == "tuple":
            return (page, None)
        else:  # "page"
            return page


class DummyRecognizer:
    """
    Минимальная реализация распознавателя для проверки совместимости API.

    Реализует только необходимый интерфейс:
    - метод predict(images)
    - возвращает список словарей {"text": str, "confidence": float}
    """

    def __init__(self):
        """Инициализация dummy распознавателя."""
        self.call_count = 0

    def predict(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        Возвращает фиксированные тексты для каждого изображения.
        """
        self.call_count += 1

        results = []
        for i, img in enumerate(images):
            results.append({"text": f"word{i + 1}", "confidence": 0.9 - i * 0.05})

        return results


class TestPipelineAPICompatibility:
    """Тесты совместимости Pipeline с различными реализациями"""

    def test_pipeline_with_dict_detector(self):
        """Тест Pipeline с детектором возвращающим dict"""
        detector = DummyDetector(return_type="dict")
        recognizer = DummyRecognizer()

        pipeline = Pipeline(detector=detector, recognizer=recognizer)

        # Создаем тестовое изображение
        img = np.zeros((100, 400, 3), dtype=np.uint8)

        result = pipeline.predict(img, recognize_text=True, vis=False)

        # Проверяем результат
        assert isinstance(result, Page)
        assert len(result.blocks) == 1
        assert len(result.blocks[0].words) == 3

        # Проверяем что распознавание выполнено
        assert result.blocks[0].words[0].text == "word1"
        assert result.blocks[0].words[1].text == "word2"
        assert result.blocks[0].words[2].text == "word3"

    def test_pipeline_with_tuple_detector(self):
        """Тест Pipeline с детектором возвращающим tuple"""
        detector = DummyDetector(return_type="tuple")
        recognizer = DummyRecognizer()

        pipeline = Pipeline(detector=detector, recognizer=recognizer)

        img = np.zeros((100, 400, 3), dtype=np.uint8)
        result = pipeline.predict(img, recognize_text=True, vis=False)

        assert isinstance(result, Page)
        assert len(result.blocks[0].words) == 3

    def test_pipeline_with_page_detector(self):
        """Тест Pipeline с детектором возвращающим напрямую Page"""
        detector = DummyDetector(return_type="page")
        recognizer = DummyRecognizer()

        pipeline = Pipeline(detector=detector, recognizer=recognizer)

        img = np.zeros((100, 400, 3), dtype=np.uint8)
        result = pipeline.predict(img, recognize_text=True, vis=False)

        assert isinstance(result, Page)
        assert len(result.blocks[0].words) == 3

    def test_pipeline_with_non_tuple_recognizer(self):
        """Тест Pipeline с распознавателем возвращающим dict (новый формат)"""
        detector = DummyDetector()
        recognizer = DummyRecognizer()

        pipeline = Pipeline(detector=detector, recognizer=recognizer)

        img = np.zeros((100, 400, 3), dtype=np.uint8)
        result = pipeline.predict(img, recognize_text=True, vis=False)

        # Проверяем что новый формат работает
        assert result.blocks[0].words[0].text == "word1"
        assert result.blocks[0].words[0].recognition_confidence == 0.9

    def test_pipeline_without_recognition(self):
        """Тест Pipeline без распознавания"""
        detector = DummyDetector()
        recognizer = DummyRecognizer()

        pipeline = Pipeline(detector=detector, recognizer=recognizer)

        img = np.zeros((100, 400, 3), dtype=np.uint8)
        result = pipeline.predict(img, recognize_text=False, vis=False)

        # Распознаватель не должен вызываться
        assert recognizer.call_count == 0

        # Слова должны быть без текста
        assert result.blocks[0].words[0].text is None

    def test_pipeline_with_visualization(self):
        """Тест Pipeline с визуализацией"""
        detector = DummyDetector()
        recognizer = DummyRecognizer()

        pipeline = Pipeline(detector=detector, recognizer=recognizer)

        img = np.zeros((100, 400, 3), dtype=np.uint8)
        result, vis_img = pipeline.predict(img, recognize_text=True, vis=True)

        assert isinstance(result, Page)
        assert isinstance(vis_img, Image.Image)

    def test_pipeline_get_text(self):
        """Тест метода get_text"""
        detector = DummyDetector()
        recognizer = DummyRecognizer()

        pipeline = Pipeline(detector=detector, recognizer=recognizer)

        img = np.zeros((100, 400, 3), dtype=np.uint8)
        result = pipeline.predict(img, recognize_text=True, vis=False)

        text = pipeline.get_text(result)

        # Должен вернуть объединенный текст
        assert "word1" in text
        assert "word2" in text
        assert "word3" in text

    def test_pipeline_min_text_size_filtering(self):
        """Тест фильтрации по минимальному размеру"""

        class SmallBoxDetector(DummyDetector):
            """Детектор с очень маленькими боксами"""

            def predict(self, image, vis=False, profile=False):
                # Создаем слова с маленькими боксами (меньше min_text_size)
                words = [
                    Word(
                        polygon=[
                            [10.0, 10.0],
                            [12.0, 10.0],
                            [12.0, 12.0],
                            [10.0, 12.0],
                        ],
                        detection_confidence=0.95,
                    ),
                ]
                return {"page": Page(blocks=[Block(words=words)])}

        detector = SmallBoxDetector()
        recognizer = DummyRecognizer()

        # min_text_size = 5 (по умолчанию)
        pipeline = Pipeline(detector=detector, recognizer=recognizer, min_text_size=5)

        img = np.zeros((100, 400, 3), dtype=np.uint8)
        result = pipeline.predict(img, recognize_text=True, vis=False)

        # Распознаватель не должен вызываться т.к. все боксы отфильтрованы
        assert recognizer.call_count == 0


class TestDummyImplementations:
    """Тесты самих dummy реализаций"""

    def test_dummy_detector_dict_format(self):
        """Тест DummyDetector возвращает корректный dict"""
        detector = DummyDetector(return_type="dict")
        result = detector.predict(np.zeros((100, 100, 3), dtype=np.uint8))

        assert isinstance(result, dict)
        assert "page" in result
        assert isinstance(result["page"], Page)

    def test_dummy_detector_tuple_format(self):
        """Тест DummyDetector возвращает корректный tuple"""
        detector = DummyDetector(return_type="tuple")
        result = detector.predict(np.zeros((100, 100, 3), dtype=np.uint8))

        assert isinstance(result, tuple)
        assert isinstance(result[0], Page)

    def test_dummy_detector_page_format(self):
        """Тест DummyDetector возвращает корректный Page"""
        detector = DummyDetector(return_type="page")
        result = detector.predict(np.zeros((100, 100, 3), dtype=np.uint8))

        assert isinstance(result, Page)

    def test_dummy_recognizer_tuple_format(self):
        """Тест DummyRecognizer возвращает словари"""
        recognizer = DummyRecognizer()
        images = [np.zeros((64, 256, 3), dtype=np.uint8) for _ in range(3)]

        results = recognizer.predict(images)

        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)
        assert all("text" in r and "confidence" in r for r in results)

    def test_dummy_recognizer_text_only_format(self):
        """Тест DummyRecognizer структура данных"""
        recognizer = DummyRecognizer()
        images = [np.zeros((64, 256, 3), dtype=np.uint8) for _ in range(3)]

        results = recognizer.predict(images)

        assert len(results) == 3
        assert results[0]["text"] == "word1"
        assert results[0]["confidence"] == 0.9


def test_readme_example_works_with_dummy():
    """
    Проверка что пример из README работает с dummy реализациями.
    Это гарантирует что API действительно универсален.
    """
    # Используем dummy реализации вместо EAST и TRBA
    detector = DummyDetector()
    recognizer = DummyRecognizer()

    # Пример из README
    pipeline = Pipeline(detector, recognizer)

    # Создаем тестовое изображение
    img = np.zeros((100, 400, 3), dtype=np.uint8)

    # Полная обработка изображения
    result = pipeline.predict(img)

    # Получение распознанного текста
    text = pipeline.get_text(result)

    assert text is not None
    assert len(text) > 0

    # Подробная информация о каждом слове
    for block in result.blocks:
        for word in block.words:
            assert word.text is not None
            assert word.detection_confidence is not None
            # recognition_confidence может быть None если распознаватель не вернул его
