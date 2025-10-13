import numpy as np
from PIL import Image
from typing import List, Union, Optional, Tuple, TYPE_CHECKING
import cv2

from .detectors import EASTInfer
from .detectors._types import Page

if TYPE_CHECKING:
    from . import TRBAInfer


class OCRPipeline:
    
    def __init__(
        self,
        detector: EASTInfer,
        recognizer: "TRBAInfer",
        min_text_size: int = 10
    ):
        """
        Инициализация OCR пайплайна
        
        Args:
            detector: Инстанс EASTInfer для детекции текста
            recognizer: Инстанс TRBAInfer для распознавания текста
            min_text_size: Минимальный размер текстовой области (пиксели)
        """
        self.detector = detector
        self.recognizer = recognizer
        self.min_text_size = min_text_size
    
    def process(
        self,
        image: Union[str, np.ndarray, Image.Image],
        recognize_text: bool = True,
        batch_size: int = 32
    ) -> Page:
        """
        Основной метод обработки: детекция + распознавание
        
        Args:
            image: Входное изображение (путь, numpy array или PIL Image)
            recognize_text: Выполнять ли распознавание текста (иначе только детекция)
            batch_size: Размер батча для распознавания (для оптимизации)
            
        Returns:
            Page объект с заполненными полями text и recognition_confidence для Word'ов
        """
        # 1. Детекция текстовых областей
        detection_result = self.detector.infer(image, vis=False)
        
        if not recognize_text:
            # Возвращаем результат детекции как есть
            return detection_result
        
        # 2. Извлечение изображений для распознавания
        image_array = self._load_image_as_array(image)
        
        # 3. Собираем все слова из всех блоков для батчевого распознавания  
        all_words = []
        word_images = []
        
        for block in detection_result.blocks:
            for word in block.words:
                # Проверяем размер области
                polygon = np.array(word.polygon, dtype=np.int32)
                x_min, y_min = np.min(polygon, axis=0)
                x_max, y_max = np.max(polygon, axis=0)
                
                width = x_max - x_min
                height = y_max - y_min
                
                if width >= self.min_text_size and height >= self.min_text_size:
                    # Извлекаем изображение области
                    region_image = self._extract_word_image(image_array, polygon)
                    if region_image is not None and region_image.size > 0:
                        all_words.append(word)
                        word_images.append(region_image)
        
        # 4. Батчевое распознавание для оптимизации скорости
        if word_images:
            recognition_results = self._batch_recognize(word_images, batch_size)
            
            # Заполняем результаты распознавания
            for word, (text, confidence) in zip(all_words, recognition_results):
                word.text = text
                word.recognition_confidence = confidence
        
        return detection_result
    
    def process_batch(
        self,
        images: List[Union[str, np.ndarray, Image.Image]],
        **kwargs
    ) -> List[Page]:
        """
        Пакетная обработка изображений
        
        Args:
            images: Список изображений для обработки
            **kwargs: Параметры для метода process()
            
        Returns:
            Список Page объектов
        """
        results = []
        for image in images:
            try:
                result = self.process(image, **kwargs)
                results.append(result)
            except Exception as e:
                # В случае ошибки создаем пустую страницу
                error_page = Page(blocks=[])
                results.append(error_page)
        
        return results
    
    def get_text(self, page: Page, separator: str = ' ') -> str:
        all_words = []
        for block in page.blocks:
            for word in block.words:
                if word.text:  # Только если есть распознанный текст
                    all_words.append({
                        'text': word.text,
                        'center_y': sum(y for _, y in word.polygon) / len(word.polygon)
                    })
        
        if not all_words:
            return ""
        
        all_words.sort(key=lambda x: x['center_y'])
        
        return separator.join([w['text'] for w in all_words])
    
    def _batch_recognize(self, word_images: List[np.ndarray], batch_size: int) -> List[Tuple[str, float]]:
        results = []
        
        for i in range(0, len(word_images), batch_size):
            batch_images = word_images[i:i + batch_size]
            
            # Распознавание батча
            batch_results = self.recognizer.predict(batch_images, return_confidence=True)
            
            # Обрабатываем результат (может быть список или одиночный результат)
            if isinstance(batch_results, list):
                results.extend(batch_results)
            else:
                # Если батч размером 1, результат может быть не списком
                results.append(batch_results)
        
        return results
    
    def _extract_word_image(self, image: np.ndarray, polygon: np.ndarray) -> Optional[np.ndarray]:
        """
        Извлекает изображение слова по полигону
        """
        try:
            # Вычисляем bounding box
            x_min, y_min = np.min(polygon, axis=0)
            x_max, y_max = np.max(polygon, axis=0)
            
            # Обрезаем по границам изображения
            h, w = image.shape[:2]
            x1 = max(0, int(x_min))
            y1 = max(0, int(y_min))
            x2 = min(w, int(x_max))
            y2 = min(h, int(y_max))
            
            # Вырезаем область
            region_image = image[y1:y2, x1:x2]
            
            return region_image if region_image.size > 0 else None
        except Exception:
            return None
    
    def _load_image_as_array(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Загружает изображение как numpy array"""
        if isinstance(image, str):
            # Путь к файлу
            img_array = cv2.imread(image)
            if img_array is None:
                raise FileNotFoundError(f"Не удалось загрузить изображение: {image}")
            return cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            # PIL Image
            return np.array(image.convert('RGB'))
        elif isinstance(image, np.ndarray):
            # Numpy array
            return image.copy()
        else:
            raise TypeError(f"Неподдерживаемый тип изображения: {type(image)}")
    
