from pathlib import Path
from typing import Sequence, Tuple, Union, List

import cv2
import numpy as np
from PIL import Image

from manuscript.data import TextSpan, Line, Block, Page


def read_image(img_or_path: Union[str, Path, bytes, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Универсальное чтение изображения с поддержкой нескольких форматов входных данных.

    Параметры
    ----------
    img_or_path : str, Path, bytes, np.ndarray, or PIL.Image
        Источник изображения в одном из следующих форматов:
        - Путь к файлу (str или Path) — поддерживает пути с Unicode (например, кириллица)
        - Байтовый буфер (например, из HTTP-ответа)
        - Массив NumPy (уже загруженное изображение)
        - Объект PIL Image

    Возвращает
    -------
    np.ndarray
        RGB-изображение в виде массива numpy с формой (H, W, 3) и типом данных uint8.

    Raises
    ------
    FileNotFoundError
        Если файл изображения не удаётся прочитать ни через OpenCV, ни через PIL.
    TypeError
        Если тип входных данных не поддерживается.
    ValueError
        Если байты не удаётся декодировать в изображение.

    Examples
    --------
    >>> # Чтение из пути к файлу (с поддержкой Unicode)
    >>> img = read_image("путь/к/изображению.jpg")
    >>> img.shape
    (480, 640, 3)

    >>> # Чтение из байтов
    >>> with open("image.jpg", "rb") as f:
    ...     img = read_image(f.read())

    >>> # Чтение из PIL Image
    >>> pil_img = Image.open("image.jpg")
    >>> img = read_image(pil_img)

    >>> # Передача готового массива numpy
    >>> img = read_image(existing_array)
    """
    # File path (str or Path) - TRBA method with Unicode support
    if isinstance(img_or_path, (str, Path)):
        # Use np.fromfile to handle Unicode paths on Windows
        data = np.fromfile(str(img_or_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        
        # Fallback to PIL for non-standard formats (TIFF, etc.)
        if img is None:
            try:
                with Image.open(str(img_or_path)) as pil_img:
                    img = np.array(pil_img.convert("RGB"))
            except Exception as e:
                raise FileNotFoundError(
                    f"Cannot read image with cv2 or PIL: {img_or_path}. Error: {e}"
                )
        else:
            # OpenCV reads as BGR, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Bytes buffer (e.g., from HTTP response or file.read())
    elif isinstance(img_or_path, bytes):
        arr = np.frombuffer(img_or_path, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from bytes")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # NumPy array (already loaded image)
    elif isinstance(img_or_path, np.ndarray):
        img = img_or_path
    
    # PIL Image object (duck typing check)
    elif hasattr(img_or_path, 'convert'):
        img = np.array(img_or_path.convert("RGB"))
    
    else:
        raise TypeError(
            f"Unsupported type for image input: {type(img_or_path)}. "
            f"Expected str, Path, bytes, numpy.ndarray, or PIL.Image"
        )
    
    return img


def _tensor_to_image(
    tensor: "torch.Tensor",  # type: ignore
    denormalize: dict = None,
    to_uint8: bool = True,
) -> np.ndarray:
    """
    Конвертирует тензор PyTorch в массив numpy с изображением.

    Параметры
    ----------
    tensor : torch.Tensor
        Входной тензор с формой (C, H, W) для одного изображения или (N, C, H, W) для батча.
        Значения должны быть в диапазоне [0, 1] или нормализованы с mean/std.
    denormalize : dict, optional
        Словарь с ключами ``'mean'`` и ``'std'`` для денормализации.
        Пример: ``{'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}``
    to_uint8 : bool, default=True
        Если ``True``, конвертирует в uint8 в диапазоне [0, 255].
        Если ``False``, возвращает float32 в диапазоне [0, 1].

    Возвращает
    -------
    np.ndarray
        Изображение(я) в виде массива numpy:
        - Одно изображение: форма (H, W, C)
        - Батч: форма (N, H, W, C)
        - dtype: uint8 если to_uint8=True, иначе float32

    Examples
    --------
    >>> import torch
    >>> # Одно изображение
    >>> tensor = torch.rand(3, 224, 224)
    >>> img = tensor_to_image(tensor)
    >>> img.shape
    (224, 224, 3)

    >>> # Батч изображений
    >>> batch = torch.rand(8, 3, 224, 224)
    >>> imgs = tensor_to_image(batch)
    >>> imgs.shape
    (8, 224, 224, 3)

    >>> # С денормализацией
    >>> normalized = torch.rand(3, 224, 224)
    >>> img = tensor_to_image(
    ...     normalized,
    ...     denormalize={'mean': [0.5, 0.5, 0.5], 'std': [0.5, 0.5, 0.5]}
    ... )
    """    
    # Detach from computation graph and move to CPU
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # Convert to numpy
    arr = tensor.numpy()
    
    # Handle batch vs single image
    is_batch = arr.ndim == 4  # (N, C, H, W)
    
    if not is_batch:
        # Add batch dimension for uniform processing
        arr = arr[np.newaxis, ...]  # (1, C, H, W)
    
    # Denormalize if parameters provided
    if denormalize is not None:
        mean = np.array(denormalize['mean']).reshape(1, -1, 1, 1)
        std = np.array(denormalize['std']).reshape(1, -1, 1, 1)
        arr = arr * std + mean
    
    # Transpose from (N, C, H, W) to (N, H, W, C)
    arr = np.transpose(arr, (0, 2, 3, 1))
    
    # Clip to valid range
    arr = np.clip(arr, 0, 1)
    
    # Convert to uint8 if requested
    if to_uint8:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.astype(np.float32)
    
    # Remove batch dimension if input was single image
    if not is_batch:
        arr = arr[0]
    
    return arr


def create_page_from_text(
    lines: List[str],
    confidence: float = 1.0,
) -> Page:
    """
    Создаёт объект ``Page`` из списка текстовых строк.

    Вспомогательная функция создаёт простую структуру ``Page`` из «сырого» текста,
    полезна для тестирования корректоров и других компонентов обработки текста
    без необходимости реального OCR (детекции/распознавания).

    Каждая строка становится объектом ``Line`` с текстовыми областями, разбитыми
    по пробелам. Текстовым областям назначаются фиктивные координаты полигона для
    совместимости со структурами данных.

    Параметры
    ----------
    lines : List[str]
        Список текстовых строк. Каждая строка будет разбита на текстовые области.
    confidence : float, optional
        Оценка уверенности, назначаемая всем текстовым областям. По умолчанию ``1.0``.

    Возвращает
    -------
    Page
        Объект ``Page`` с одним блоком, содержащим переданные строки.

    Examples
    --------
    >>> from manuscript.utils import create_page_from_text
    >>> page = create_page_from_text(["Hello world", "This is a test"])
    >>> page.blocks[0].lines[0].text_spans[0].text
    'Hello'
    >>> len(page.blocks[0].lines)
    2

    Использование с корректором:

    >>> from manuscript.correctors import CharLM
    >>> from manuscript.utils import create_page_from_text
    >>>
    >>> # Создаём страницу из текста с возможными ошибками OCR
    >>> page = create_page_from_text(["Привѣтъ міръ"])
    >>>
    >>> # Применяем корректор
    >>> corrector = CharLM()
    >>> corrected = corrector.predict(page)
    >>>
    >>> # Получаем исправленный текст
    >>> for line in corrected.blocks[0].lines:
    ...     print(" ".join(span.text for span in line.text_spans))
    """
    result_lines = []
    y_offset = 0
    line_height = 30
    
    for line_text in lines:
        words_text = line_text.split()
        if not words_text:
            y_offset += line_height
            continue
            
        text_spans = []
        x_offset = 0
        
        for word_text in words_text:
            word_width = len(word_text) * 10
            polygon = [
                (float(x_offset), float(y_offset)),
                (float(x_offset + word_width), float(y_offset)),
                (float(x_offset + word_width), float(y_offset + line_height)),
                (float(x_offset), float(y_offset + line_height)),
            ]
            text_span = TextSpan(
                polygon=polygon,
                detection_confidence=confidence,
                text=word_text,
                recognition_confidence=confidence,
            )
            text_spans.append(text_span)
            x_offset += word_width + 10
        
        result_lines.append(Line(text_spans=text_spans))
        y_offset += line_height + 5
    
    block = Block(lines=result_lines)
    return Page(blocks=[block])


def create_page_from_image(
    image: Union[
        str,
        Path,
        bytes,
        np.ndarray,
        Image.Image,
        Sequence[Union[str, Path, bytes, np.ndarray, Image.Image]],
    ],
    confidence: float = 1.0,
    gap: int = 8,
    return_image: bool = False,
) -> Union[Page, Tuple[Page, np.ndarray]]:
    """
    Создаёт объект ``Page``, оборачивающий одно или несколько изображений или текстовых
    кропов.

    Вспомогательная функция полезна, когда распознаватель ожидает API стадии ``0.1.11+``
    (``predict(page, image=...) -> Page``), но вы хотите запустить инференс на одном или
    нескольких заранее вырезанных изображениях без детектора. Для одного изображения
    функция создаёт один блок, одну строку и одну ``TextSpan``, охватывающую всё
    изображение. Для нескольких изображений кропы складываются вертикально в
    синтетическую страницу, и каждый кроп становится отдельной строкой с одной
    ``TextSpan``.

    Параметры
    ----------
    image : str, Path, bytes, numpy.ndarray, PIL.Image, or sequence thereof
        Источник изображения, принимаемый :func:`read_image`, или последовательность
        таких источников.
    confidence : float, optional
        Уверенность детекции, назначаемая созданной текстовой области.
        По умолчанию ``1.0``.
    gap : int, optional
        Вертикальный промежуток в пикселях между кропами, когда передаётся
        последовательность изображений. По умолчанию ``8``.
    return_image : bool, optional
        Если ``True``, дополнительно возвращает нормализованное RGB-изображение,
        соответствующее созданному ``Page``. Особенно полезно, когда ``image`` является
        последовательностью и строится синтетическое изображение страницы.
        По умолчанию ``False``.

    Возвращает
    -------
    Page or tuple of (Page, numpy.ndarray)
        Объект ``Page``, описывающий входное изображение (или изображения). Если
        ``return_image=True``, дополнительно возвращается RGB-изображение, использованное
        для создания страницы.

    Examples
    --------
    >>> from manuscript.utils import create_page_from_image
    >>> page = create_page_from_image("crop1.png")
    >>> span = page.blocks[0].lines[0].text_spans[0]
    >>> span.polygon
    [(0.0, 0.0), (120.0, 0.0), (120.0, 32.0), (0.0, 32.0)]

    Использование с распознавателем:

    >>> from manuscript.recognizers import TRBA
    >>> page = create_page_from_image("crop1.png")
    >>> recognizer = TRBA()
    >>> result_page = recognizer.predict(page, image="crop1.png")
    >>> result_page.blocks[0].lines[0].text_spans[0].text
    'example'

    Использование с несколькими кропами:

    >>> page, composed_image = create_page_from_image(
    ...     ["crop1.png", "crop2.png"],
    ...     return_image=True,
    ... )
    >>> recognizer = TRBA()
    >>> result_page = recognizer.predict(page, image=composed_image)
    """
    def _is_sequence_input(value) -> bool:
        return isinstance(value, Sequence) and not isinstance(
            value, (str, Path, bytes, np.ndarray, Image.Image)
        )

    def _full_image_span(width: int, height: int, order: int) -> TextSpan:
        return TextSpan(
            polygon=[
                (0.0, 0.0),
                (float(width), 0.0),
                (float(width), float(height)),
                (0.0, float(height)),
            ],
            detection_confidence=confidence,
            order=order,
        )

    if not _is_sequence_input(image):
        img = read_image(image)
        h, w = img.shape[:2]
        page = Page(
            blocks=[
                Block(
                    lines=[
                        Line(
                            text_spans=[_full_image_span(w, h, order=0)],
                            order=0,
                        )
                    ],
                    order=0,
                )
            ]
        )
        if return_image:
            return page, img
        return page

    images = [read_image(item) for item in image]
    if not images:
        raise ValueError("image sequence must not be empty")

    max_width = max(img.shape[1] for img in images)
    total_height = sum(img.shape[0] for img in images) + gap * (len(images) - 1)
    canvas = np.full((total_height, max_width, 3), 255, dtype=np.uint8)

    lines = []
    y_offset = 0
    for order, img in enumerate(images):
        h, w = img.shape[:2]
        canvas[y_offset : y_offset + h, 0:w] = img
        text_span = TextSpan(
            polygon=[
                (0.0, float(y_offset)),
                (float(w), float(y_offset)),
                (float(w), float(y_offset + h)),
                (0.0, float(y_offset + h)),
            ],
            detection_confidence=confidence,
            order=0,
        )
        lines.append(Line(text_spans=[text_span], order=order))
        y_offset += h + gap

    page = Page(blocks=[Block(lines=lines, order=0)])
    if return_image:
        return page, canvas
    return page
