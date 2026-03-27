from pathlib import Path
from typing import Union, List

import cv2
import numpy as np
from PIL import Image

from manuscript.data import TextSpan, Line, Block, Page


def read_image(img_or_path: Union[str, Path, bytes, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Universal image reading with support for multiple input types.

    Parameters
    ----------
    img_or_path : str, Path, bytes, np.ndarray, or PIL.Image
        Image source in one of the following formats:
        - File path (str or Path) - supports Unicode paths (e.g., Cyrillic)
        - Bytes buffer (e.g., from HTTP response)
        - NumPy array (already loaded image)
        - PIL Image object
        
    Returns
    -------
    np.ndarray
        RGB image as numpy array with shape (H, W, 3) and dtype uint8.
        
    Raises
    ------
    FileNotFoundError
        If the image file cannot be read with either OpenCV or PIL.
    TypeError
        If the input type is not supported.
    ValueError
        If bytes cannot be decoded into an image.
        
    Examples
    --------
    >>> # Read from file path (with Unicode support)
    >>> img = read_image("путь/к/изображению.jpg")
    >>> img.shape
    (480, 640, 3)
    
    >>> # Read from bytes
    >>> with open("image.jpg", "rb") as f:
    ...     img = read_image(f.read())
    
    >>> # Read from PIL Image
    >>> pil_img = Image.open("image.jpg")
    >>> img = read_image(pil_img)
    
    >>> # Pass through numpy array
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
    Convert PyTorch tensor to numpy image array.
    
    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor with shape (C, H, W) for single image or (N, C, H, W) for batch.
        Values should be in range [0, 1] or normalized with mean/std.
    denormalize : dict, optional
        Dictionary with 'mean' and 'std' keys for denormalization.
        Example: {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    to_uint8 : bool, default=True
        If True, convert to uint8 in range [0, 255].
        If False, return float32 in range [0, 1].
        
    Returns
    -------
    np.ndarray
        Image(s) as numpy array:
        - Single image: shape (H, W, C)
        - Batch: shape (N, H, W, C)
        - dtype: uint8 if to_uint8=True, else float32
        
    Examples
    --------
    >>> import torch
    >>> # Single image
    >>> tensor = torch.rand(3, 224, 224)
    >>> img = tensor_to_image(tensor)
    >>> img.shape
    (224, 224, 3)
    
    >>> # Batch of images
    >>> batch = torch.rand(8, 3, 224, 224)
    >>> imgs = tensor_to_image(batch)
    >>> imgs.shape
    (8, 224, 224, 3)
    
    >>> # With denormalization
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
    Create a Page object from a list of text lines.

    This utility function creates a simple Page structure from raw text,
    useful for testing correctors or other text processing components
    without requiring actual OCR detection/recognition.

    Each line becomes a Line object with text spans split by whitespace.
    Text spans are assigned dummy polygon coordinates for compatibility
    with the data structures.

    Parameters
    ----------
    lines : List[str]
        List of text lines. Each line will be split into text spans.
    confidence : float, optional
        Confidence score to assign to all text spans (default 1.0).

    Returns
    -------
    Page
        Page object with one Block containing the provided lines.

    Examples
    --------
    >>> from manuscript.utils import create_page_from_text
    >>> page = create_page_from_text(["Hello world", "This is a test"])
    >>> page.blocks[0].lines[0].text_spans[0].text
    'Hello'
    >>> len(page.blocks[0].lines)
    2

    Use with corrector:

    >>> from manuscript.correctors import CharLM
    >>> from manuscript.utils import create_page_from_text
    >>> 
    >>> # Create page from text with potential OCR errors
    >>> page = create_page_from_text(["Привѣтъ міръ"])
    >>> 
    >>> # Apply correction
    >>> corrector = CharLM()
    >>> corrected = corrector.predict(page)
    >>> 
    >>> # Get corrected text
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
