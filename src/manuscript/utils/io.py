from pathlib import Path
from typing import TYPE_CHECKING, Sequence, Tuple, Union, List

import cv2
import numpy as np
from PIL import Image

from manuscript.data import Page, TextRegion

if TYPE_CHECKING:
    import torch


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
    >>> img = read_image("path/to/image.jpg")
    >>> img.shape
    (480, 640, 3)

    >>> with open("image.jpg", "rb") as f:
    ...     img = read_image(f.read())

    >>> pil_img = Image.open("image.jpg")
    >>> img = read_image(pil_img)

    >>> img = read_image(existing_array)
    """
    if isinstance(img_or_path, (str, Path)):
        # Use np.fromfile to handle Unicode paths on Windows
        data = np.fromfile(str(img_or_path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        
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
        if not img_or_path:
            raise ValueError("Failed to decode image from empty bytes")
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
    
    if not img.size:
        raise ValueError("Image arrays must not be empty")
    if img.dtype != np.uint8:
        raise ValueError("Image arrays must have dtype uint8")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Image arrays must have shape [H, W], [H, W, 1], [H, W, 3] or [H, W, 4]")
    return img


def _tensor_to_image(
    tensor: "torch.Tensor",  # type: ignore
    denormalize: dict = None,
    to_uint8: bool = True,
) -> np.ndarray:
    """
    Convert a PyTorch tensor to a numpy image array.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor with shape (C, H, W) for a single image or (N, C, H, W)
        for a batch. Values should be in [0, 1] or normalised with mean/std.
    denormalize : dict, optional
        Dict with keys ``'mean'`` and ``'std'`` for de-normalisation.
        Example: ``{'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}``
    to_uint8 : bool, default=True
        If ``True``, convert to uint8 in range [0, 255].
        If ``False``, return float32 in range [0, 1].

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
    >>> tensor = torch.rand(3, 224, 224)
    >>> img = _tensor_to_image(tensor)
    >>> img.shape
    (224, 224, 3)

    >>> batch = torch.rand(8, 3, 224, 224)
    >>> imgs = _tensor_to_image(batch)
    >>> imgs.shape
    (8, 224, 224, 3)

    >>> normalized = torch.rand(3, 224, 224)
    >>> img = _tensor_to_image(
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
    lines: Union[str, Sequence[str]],
    confidence: float = 1.0,
) -> Page:
    """Create a block → lines → words tree with synthetic pixel coordinates.

    Accepts a list of lines or a multiline string. Whitespace splits words;
    empty lines reserve vertical space without creating empty text regions.
    Geometry is illustrative, useful for testing text processing without OCR.

    >>> page = create_page_from_text(["Hello world"])
    >>> page.children[0].children[0].children[0].text
    'Hello'
    """
    # Validate confidence even when the input contains no words.
    TextRegion(type="word", detection_confidence=confidence)
    if isinstance(lines, str):
        lines = lines.splitlines()
    if not isinstance(lines, Sequence) or any(not isinstance(line, str) for line in lines):
        raise TypeError("lines must be a string or a sequence of strings")
    result_lines = []
    width = 1
    y = 0
    for text in lines:
        words = []
        x = 0
        for word in text.split():
            right = x + len(word) * 10
            words.append(TextRegion(type="word", text=word,
                polygon=[(x, y), (right, y), (right, y + 30), (x, y + 30)],
                detection_confidence=confidence, recognition_confidence=confidence))
            x = right + 10
        if words:
            right = x - 10
            line = TextRegion(type="line",
                polygon=[(0, y), (right, y), (right, y + 30), (0, y + 30)])
            line.children = words
            result_lines.append(line)
            width = max(width, right)
        y += 35
    page = Page(image_width=width, image_height=max(1, y - 5))
    if result_lines:
        top = result_lines[0].polygon[0][1]
        bottom = result_lines[-1].polygon[2][1]
        block = TextRegion(type="block", polygon=[(0, top), (width, top), (width, bottom), (0, bottom)])
        block.children = result_lines
        page.children = [block]
    return page


def create_page_from_image(
    image: Union[str, Path, bytes, np.ndarray, Image.Image,
                 Sequence[Union[str, Path, bytes, np.ndarray, Image.Image]]],
    confidence: float = 1.0,
    gap: int = 8,
    return_image: bool = False,
    *,
    region_type: str = "word",
) -> Union[Page, Tuple[Page, np.ndarray]]:
    """Wrap each crop as a flat text region in the document-tree Page.

    Multiple images are stacked vertically in input order on a white RGB
    canvas. region_type defaults to word (TRBA); use line for PPOCRRec/TrOCR
    line crops, or block for a cropped text block. Page dimensions and crop
    polygons refer to the returned canvas. Single file inputs retain file_name.

    >>> page, pixels = create_page_from_image(np.zeros((32, 100, 3), np.uint8),
    ...                                      region_type="line", return_image=True)
    >>> page.children[0].type
    'line'
    >>> (page.image_width, page.image_height)
    (100, 32)
    """
    TextRegion(type=region_type, detection_confidence=confidence)
    if isinstance(gap, bool) or not isinstance(gap, int) or gap < 0:
        raise ValueError("gap must be a nonnegative integer")
    multiple = isinstance(image, Sequence) and not isinstance(
        image, (str, Path, bytes, np.ndarray, Image.Image))
    images = [read_image(item) for item in image] if multiple else [read_image(image)]
    if not images:
        raise ValueError("image sequence must not be empty")
    width = max(img.shape[1] for img in images)
    height = sum(img.shape[0] for img in images) + gap * (len(images) - 1)
    canvas = np.full((height, width, 3), 255, dtype=np.uint8) if multiple else images[0]
    page = Page(image_width=width, image_height=height,
                file_name=str(image) if not multiple and isinstance(image, (str, Path)) else "")
    regions = []
    y = 0
    for img in images:
        h, w = img.shape[:2]
        if multiple:
            canvas[y:y + h, :w] = img
        regions.append(TextRegion(type=region_type, detection_confidence=confidence,
            polygon=[(0, y), (w, y), (w, y + h), (0, y + h)]))
        y += h + gap
    page.children = regions
    return (page, canvas) if return_image else page
