"""Doc-UFCN semantic segmentation through ONNX Runtime."""
from pathlib import Path

import cv2
import numpy as np

from manuscript.api.detector import BaseDetector
from manuscript.utils import read_image
from manuscript.utils.geometry import repair_detection_polygon


class DocUFCN(BaseDetector):
    """Extract foreground contours from Doc-UFCN class-probability maps.

    The longest image side is resized to target_size, retaining aspect ratio;
    mean-colored padding makes both sides divisible by eight. Classes use
    argmax, as in the original implementation. Background channel zero is
    excluded; min_cc is measured before scaling back to source coordinates.
    region_classes maps foreground IDs to document elements while class_names
    defines output class names (generic-page is configured as block).
    """

    registry_model_class = 'DocUFCN'

    def __init__(self, weights, config=None, device=None, *, score_thresh=None,
                 target_size=None, min_cc=None, force_download=False):
        super().__init__(weights=str(weights), device=device, force_download=force_download)
        values, self.config_path = self._load_detector_config(config)
        if values.get('schema_version') != 1 or values.get('task') != 'semantic_segmentation':
            raise ValueError('DocUFCN requires a semantic_segmentation config, schema_version=1')
        self._configure_region_types(values)
        self.class_names = {int(key): name for key, name in values['class_names'].items()}
        if (set(self.class_names) != set(range(len(self.class_names)))
                or len(self.class_names) < 2 or 0 in self.region_classes
                or not set(self.region_classes) <= self.class_names.keys()
                or any(not isinstance(name, str) or not name for name in self.class_names.values())):
            raise ValueError('DocUFCN requires contiguous class IDs and foreground-only region_classes')
        self.mean = np.asarray(values['mean'], dtype=np.float32)
        self.std = np.asarray(values['std'], dtype=np.float32)
        if (self.mean.shape != (3,) or self.std.shape != (3,)
                or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all()
                or np.any(self.std <= 0)):
            raise ValueError('DocUFCN mean/std must have three finite values, with positive std')
        self.target_size = values['target_size'] if target_size is None else target_size
        self.min_cc = values.get('min_cc', 50) if min_cc is None else min_cc
        for name in ('target_size', 'min_cc'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.target_size < 8:
            raise ValueError('target_size must be at least eight')
        self.score_thresh = float(values.get('score_thresh', 0) if score_thresh is None else score_thresh)
        if not np.isfinite(self.score_thresh) or not 0 <= self.score_thresh <= 1:
            raise ValueError('score_thresh must be between zero and one')
        self.input_name = values.get('input_name', 'images')
        self.output_name = values.get('output_name', 'probabilities')

    def _initialize_session(self):
        if self.session is None:
            session = self._create_onnx_session()
            inputs = session.get_inputs()
            if len(inputs) != 1 or inputs[0].name != self.input_name or len(inputs[0].shape) != 4:
                raise ValueError('DocUFCN requires one NCHW image input')
            if inputs[0].type != 'tensor(float)':
                raise ValueError('DocUFCN requires float32 images')
            shape = inputs[0].shape
            if shape[0] != 1 or shape[1] != 3 or any(isinstance(size, int) for size in shape[2:]):
                raise ValueError('DocUFCN requires batch one, RGB and dynamic image height/width')
            if self.output_name not in {item.name for item in session.get_outputs()}:
                raise ValueError('Configured DocUFCN output is absent')
            self.session = session

    def _preprocess(self, image):
        height, width = image.shape[:2]
        ratio = self.target_size / max(height, width)
        resized_hw = (max(1, int(height * ratio)), max(1, int(width * ratio)))
        resized = cv2.resize(image, (resized_hw[1], resized_hw[0]))
        delta_h, delta_w = (-resized_hw[0]) % 8, (-resized_hw[1]) % 8
        top, left = delta_h // 2, delta_w // 2
        padded = cv2.copyMakeBorder(resized, top, delta_h-top, left, delta_w-left,
                                    cv2.BORDER_CONSTANT, value=self.mean.tolist())
        normalized = (padded.astype(np.float32) - self.mean) / self.std
        return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None]), resized_hw, (top, left)

    def predict(self, image, return_raw=False, *, return_masks=False, return_outputs=False):
        self._initialize_session()
        image = read_image(image)
        tensor, resized_hw, padding = self._preprocess(image)
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        expected_shape = (1, len(self.class_names), *tensor.shape[2:])
        if (output.shape != expected_shape or not np.isfinite(output).all()
                or output.min() < -1e-5 or output.max() > 1+1e-5
                or not np.allclose(output.sum(axis=1), 1, atol=1e-3)):
            raise ValueError('DocUFCN must return normalized NCHW class probabilities at input resolution')
        probabilities = np.clip(output[0], 0, 1)
        labels = probabilities.argmax(axis=0)
        height, width = image.shape[:2]
        top, left = padding
        detections, polygons, scores, ids = [], [], [], []
        for class_id in self.region_classes:
            probability = np.where(labels == class_id, probabilities[class_id], 0)
            binary = (labels == class_id).astype(np.uint8)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            class_detections = []
            for contour in contours:
                if cv2.contourArea(contour) <= self.min_cc:
                    continue
                mask = np.zeros(labels.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], 0, 1, -1)
                confidence = float(round(float(probability[mask.astype(bool)].astype(np.float64).mean()), 2))
                if confidence < self.score_thresh:
                    continue
                points = contour[:, 0].astype(np.float64)
                points[:, 0] = np.clip(np.trunc((points[:, 0]-left) * width/resized_hw[1]), 0, width)
                points[:, 1] = np.clip(np.trunc((points[:, 1]-top) * height/resized_hw[0]), 0, height)
                polygon = repair_detection_polygon(points)
                if polygon is None:
                    continue
                class_detections.append({'class_id': class_id, 'class_name': self.class_names[class_id],
                                         'confidence': confidence, 'polygon': polygon.astype(float).tolist()})
            class_detections.sort(key=lambda item: (item['polygon'][0][1], item['polygon'][0][0]))
            for detail in class_detections:
                detections.append(detail)
                polygons.append(detail['polygon']); scores.append(detail['confidence']); ids.append(class_id)
        page = self._tree_from_regions(polygons, scores, ids, (height, width))
        if return_raw or return_masks or return_outputs:
            result = self._raw_result(page, {self.output_name: output} if return_outputs else None,
                                      detections, (height, width), class_names=self.class_names.copy())
            if return_masks:
                foreground = labels[top:top+resized_hw[0], left:left+resized_hw[1]]
                result['masks'] = {self.class_names[key]: cv2.resize((foreground == key).astype(np.uint8),
                                  (width, height), interpolation=cv2.INTER_NEAREST)
                                  for key in self.region_classes}
            return result
        return page


__all__ = ['DocUFCN']
