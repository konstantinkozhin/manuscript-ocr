"""RT-DETR ONNX detection with the original model class names."""
import numpy as np
from PIL import Image

from .._rfdetr import RFDETR


class RTDETR(RFDETR):
    """RT-DETR/RT-DETRv2 with normalized boxes and per-class query logits.

    Uses the shared DETR decoder, preserving config region_classes verbatim.
    Images use PIL bilinear resizing before rescaling and optional normalization,
    matching Hugging Face's RTDetrImageProcessor for the exported Heron preset.
    Runtime inference requires ONNX Runtime; Torch/Transformers are export-only.
    """

    registry_model_class = 'RTDETR'

    def _preprocess(self, image):
        height, width = self._input_hw
        resized = Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = (array - self.mean) / self.std
        return np.ascontiguousarray(tensor.transpose(2, 0, 1)[None], dtype=self._input_dtype)


__all__ = ['RTDETR']
