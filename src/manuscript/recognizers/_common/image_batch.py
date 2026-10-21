"""Assemble equal-size preprocessed images without intermediate batch copies."""
import numpy as np


def prepare_image_batch(regions, preprocess, size):
    """Keep one prepared image alongside the final buffer; repeat the last if needed."""
    if not regions or size < len(regions):
        raise ValueError("Image batch size must accommodate all regions")
    first = preprocess(regions[0].image)
    if first.ndim != 4 or first.shape[0] != 1:
        raise ValueError("Expected a preprocessed image with shape [1, C, H, W]")
    shape = first.shape
    batch = np.empty((size, *shape[1:]), dtype=first.dtype)
    batch[0] = first[0]
    del first
    for index in range(1, len(regions)):
        tensor = preprocess(regions[index].image)
        if tensor.shape != shape:
            raise ValueError("Preprocessed images must have matching shapes")
        batch[index] = tensor[0]
        del tensor
    if size > len(regions):
        batch[len(regions):] = batch[len(regions) - 1]
    return batch
