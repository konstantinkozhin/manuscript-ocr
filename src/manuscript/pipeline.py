import numpy as np
from PIL import Image
from typing import Union, Optional 
import cv2

from .detectors import EASTInfer
from .detectors._types import Page
from .recognizers import TRBAInfer


class OCRPipeline:
    
    def __init__(
        self,
        detector: EASTInfer,
        recognizer: TRBAInfer,
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
        recognize_text: bool = True
    ) -> Page:
        """
        Основной метод обработки: детекция + распознавание
        
        Args:
            image: Входное изображение (путь, numpy array или PIL Image)
            recognize_text: Выполнять ли распознавание текста (иначе только детекция)
            
        Returns:
            Page объект с заполненными полями text и recognition_confidence для Word'ов
        """
        detection_result = self.detector.infer(image, vis=False)
        
        if not recognize_text:
            return detection_result
        
        image_array = self._load_image_as_array(image)
        
        all_words = []
        word_images = []
        
        for block in detection_result.blocks:
            for word in block.words:
                polygon = np.array(word.polygon, dtype=np.int32)
                x_min, y_min = np.min(polygon, axis=0)
                x_max, y_max = np.max(polygon, axis=0)
                
                width = x_max - x_min
                height = y_max - y_min
                
                if width >= self.min_text_size and height >= self.min_text_size:
                    region_image = self._extract_word_image(image_array, polygon)
                    if region_image is not None and region_image.size > 0:
                        all_words.append(word)
                        word_images.append(region_image)
        
        if word_images:
            recognition_results = self.recognizer.predict(word_images)
            
            for word, (text, confidence) in zip(all_words, recognition_results):
                word.text = text
                word.recognition_confidence = confidence
        
        return detection_result
    
    def _extract_word_image(self, image: np.ndarray, polygon: np.ndarray) -> Optional[np.ndarray]:
        """
        Извлекает изображение слова по полигону
        """
        try:
            x_min, y_min = np.min(polygon, axis=0)
            x_max, y_max = np.max(polygon, axis=0)
            
            h, w = image.shape[:2]
            x1 = max(0, int(x_min))
            y1 = max(0, int(y_min))
            x2 = min(w, int(x_max))
            y2 = min(h, int(y_max))
            
            region_image = image[y1:y2, x1:x2]
            
            return region_image if region_image.size > 0 else None
        except Exception:
            return None
    
    def _load_image_as_array(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Загружает изображение как numpy array"""
        if isinstance(image, str):
            img_array = cv2.imread(image)
            if img_array is None:
                raise FileNotFoundError(f"Не удалось загрузить изображение: {image}")
            return cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        elif isinstance(image, Image.Image):
            return np.array(image.convert('RGB'))
        elif isinstance(image, np.ndarray):
            return image.copy()
        else:
            raise TypeError(f"Неподдерживаемый тип изображения: {type(image)}")
    
