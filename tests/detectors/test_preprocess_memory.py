import cv2
import numpy as np
import pytest

from manuscript.detectors._rfdetr import RFDETR
from manuscript.detectors._ppocr import PPOCR
from manuscript.detectors._mask2former import Mask2Former


@pytest.mark.parametrize('dtype', [np.float16, np.float32])
@pytest.mark.parametrize('kind', ['rfdetr', 'ppocr', 'mask2former'])
def test_in_place_normalization_matches_original_arithmetic(kind, dtype):
    image = np.random.default_rng(42).integers(0, 256, (137, 311, 3), np.uint8)[:, ::-1]
    mean = np.array([.485, .456, .406], np.float32)
    std = np.array([.229, .224, .225], np.float32)
    if kind == 'rfdetr':
        model = RFDETR.__new__(RFDETR)
        model._input_hw = (64, 96)
        model.mean, model.std, model._input_dtype = mean, std, dtype
        pixels = cv2.resize(image.astype(np.float32) / 255., (96, 64), interpolation=cv2.INTER_LINEAR)
        expected = np.ascontiguousarray(((pixels - mean) / std).transpose(2, 0, 1)[None], dtype=dtype)
        actual = model._preprocess(image)
    elif kind == 'ppocr':
        model = PPOCR.__new__(PPOCR)
        model._resize_shape = lambda *args: (64, 96)
        model._input_dtype = dtype
        model.preprocess = {'color_order': 'BGR', 'scale': 1 / 255., 'mean': mean, 'std': std}
        pixels = cv2.resize(image.astype(np.float32), (96, 64), interpolation=cv2.INTER_LINEAR)[:, :, ::-1]
        expected = np.ascontiguousarray(((pixels * (1 / 255.) - mean) / std).transpose(2, 0, 1)[None], dtype=dtype)
        actual = model._preprocess(image)
    else:
        model = Mask2Former.__new__(Mask2Former)
        model.image_size, model.pixel_dtype = 96, dtype
        model.image_mean, model.image_std = mean, std
        actual, left, top, size = model._preprocess(image)
        canvas = np.full((96, 96, 3), 255, np.uint8)
        canvas[top:top + size[1], left:left + size[0]] = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        pixels = canvas.astype(np.float32) / 255.
        expected = ((pixels - mean) / std).transpose(2, 0, 1)[None].astype(dtype)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == dtype
