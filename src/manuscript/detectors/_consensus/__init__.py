"""Detector consensus with optional participant weights for geometry only."""
import math
from numbers import Real
from pathlib import Path

import numpy as np

from manuscript._internal.model_lifecycle import (
    ModelComposition, managed_prediction, processor_scope, validate_composition,
)
from manuscript.api._page_helpers import filter_predict_kwargs
from manuscript.api.detector import BaseDetector
from manuscript.data import Page
from manuscript.utils import read_image


def _iou_matrix(first, second):
    a = np.asarray(first, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(second, dtype=np.float64).reshape(-1, 4)
    left = np.maximum(a[:, None, :2], b[None, :, :2])
    right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(0, right - left).prod(axis=2)
    union = (np.maximum(0, a[:, 2:] - a[:, :2]).prod(axis=1)[:, None]
             + np.maximum(0, b[:, 2:] - b[:, :2]).prod(axis=1)[None, :] - intersection)
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


class ConsensusWBFDetector(ModelComposition):
    """Fuse detections supported by enough independent participants.

    ``detectors`` is a nonempty ordered iterable of objects implementing
    ``predict(image) -> Page`` (or a dictionary containing ``page``). Their
    Pages must use original-image coordinates and dimensions. All geometric
    descendants are treated as flat detections, grouped by ``(type, label)``.
    Configure equivalent labels on participants to let their objects match.

    ``weights=None`` uses equal arithmetic means, reproducing consensus box
    fusion. Otherwise supply one finite positive weight per detector. Weights
    affect cluster representatives and fused coordinates, never vote counts.
    Detection confidence is not used. Output confidence remains None.

    Detectors run in supplied order. At each step, cluster representatives are
    frozen and IoU pairs are greedily matched one-to-one. Ties sort by descending
    (IoU, box index, cluster index). Each detector supplies at most one vote per
    cluster. Empty answers do not reduce the required vote count; errors abort.
    No additional NMS is applied. The algorithm depends on participant order.

    ``cluster_iou`` is in [0, 1], ``consensus_fraction`` in [0, 1], and
    ``minimum_votes`` is a positive integer. Required votes are the maximum of
    minimum_votes and ceil(number of detectors * consensus_fraction). A threshold
    above the number of detectors is allowed and yields an empty Page.

    ``memory_policy`` follows Pipeline/recognizer compositions: "keep" (default)
    or "sequential"; the latter requires close() on every participant.
    Pickle support preserves nested participants, order, weights and parameters,
    provided participants themselves support pickling.
    """

    def __init__(self, detectors, *, weights=None, cluster_iou=0.5,
                 consensus_fraction=0.5, minimum_votes=2, memory_policy="keep"):
        self._init_memory_policy(memory_policy)
        self.detectors = tuple(detectors)
        if not self.detectors:
            raise ValueError('ConsensusWBFDetector requires at least one detector')
        for detector in self.detectors:
            if not callable(getattr(detector, 'predict', None)):
                raise TypeError('Participants must implement predict')
        for name, value in [('cluster_iou', cluster_iou), ('consensus_fraction', consensus_fraction)]:
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be a finite number in [0, 1]')
        if isinstance(minimum_votes, bool) or not isinstance(minimum_votes, int) or minimum_votes < 1:
            raise ValueError('minimum_votes must be a positive integer')
        self.cluster_iou = float(cluster_iou)
        self.consensus_fraction = float(consensus_fraction)
        self.minimum_votes = minimum_votes
        self.weights = None if weights is None else tuple(weights)
        if self.weights is not None:
            if len(self.weights) != len(self.detectors):
                raise ValueError('weights must contain one value per detector')
            if any(isinstance(w, bool) or not isinstance(w, Real) or not math.isfinite(w) or w <= 0
                   for w in self.weights):
                raise ValueError('weights must be finite positive numbers')
            self.weights = tuple(float(w) for w in self.weights)
        validate_composition(self)

    @property
    def required_votes(self):
        return max(self.minimum_votes, math.ceil(len(self.detectors) * self.consensus_fraction))

    def _memory_children(self):
        return self.detectors

    def _mean(self, cluster):
        boxes = [entry['box'] for entry in cluster]
        if self.weights is None:
            return np.mean(boxes, axis=0)
        weights = np.asarray([self.weights[entry['detector_index']] for entry in cluster])
        # Avoid overflow for large, otherwise valid participant weights.
        return np.average(boxes, axis=0, weights=weights / weights.max())

    @managed_prediction
    def predict(self, image, return_raw=False, *, return_masks=False, return_outputs=False):
        """Return a flat Page, or standard raw detection data with vote metadata.

        Any return flag enables dictionary output. Fusion produces no masks or
        model tensors. Raw detections include box, polygon, type, label, votes,
        and detector_indices; indices refer to the supplied participant order.
        """
        original = read_image(image)
        height, width = original.shape[:2]
        clusters = []
        for detector_index, detector in enumerate(self.detectors):
            with processor_scope(detector):
                prediction = detector.predict(original, **filter_predict_kwargs(detector, {
                    'return_raw': False, 'return_masks': False, 'return_outputs': False,
                }))
            page = prediction.get('page') if isinstance(prediction, dict) else prediction
            if not isinstance(page, Page):
                raise TypeError(f'detector[{detector_index}].predict must return Page or a dictionary containing Page')
            page.validate_tree()
            if (page.image_height, page.image_width) != (height, width):
                raise ValueError(f'detector[{detector_index}] must return original-image dimensions and coordinates')
            objects = []
            for node in page.descendants:
                if node.polygon is None:
                    continue
                polygon = np.asarray(node.polygon, dtype=np.float64)
                low, high = polygon.min(axis=0), polygon.max(axis=0)
                if np.any(high <= low):
                    continue
                objects.append({'box': np.concatenate((low, high)),
                                'key': (node.type, node.label), 'detector_index': detector_index,
                                'node': node})
            representatives = [self._mean(cluster) for cluster in clusters]
            overlap = _iou_matrix([obj['box'] for obj in objects], representatives)
            pairs = sorted(((float(overlap[i, j]), int(i), int(j))
                            for i, j in zip(*np.nonzero(overlap >= self.cluster_iou))
                            if objects[i]['key'] == clusters[j][0]['key']), reverse=True)
            assigned_boxes, assigned_clusters = set(), set()
            for _, i, j in pairs:
                if i in assigned_boxes or j in assigned_clusters:
                    continue
                clusters[j].append(objects[i])
                assigned_boxes.add(i)
                assigned_clusters.add(j)
            clusters.extend([obj] for i, obj in enumerate(objects) if i not in assigned_boxes)

        result = Page(image_height=height, image_width=width,
                      file_name=Path(image).name if isinstance(image, (str, Path)) else '')
        nodes, detections = [], []
        for cluster in clusters:
            if len(cluster) < self.required_votes:
                continue
            box = self._mean(cluster).tolist()
            x1, y1, x2, y2 = box
            polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            kind, label = cluster[0]['key']
            prototype = cluster[0]['node']
            fields = prototype.model_dump()
            fields.update(polygon=polygon, detection_confidence=None, orientation=None)
            for field in ('text', 'recognition_confidence'):
                if field in fields:
                    fields[field] = None
            nodes.append(type(prototype)(**fields))
            detections.append({'box': box, 'polygon': polygon, 'type': kind, 'label': label,
                               'votes': len(cluster),
                               'detector_indices': [obj['detector_index'] for obj in cluster]})
        result.children = nodes
        if return_raw or return_masks or return_outputs:
            return BaseDetector._raw_result(
                result, None, detections, (height, width), aggregation='consensus_wbf',
                required_votes=self.required_votes, total_detectors=len(self.detectors),
                weights=list(self.weights) if self.weights is not None else None,
                cluster_iou=self.cluster_iou, consensus_fraction=self.consensus_fraction,
                minimum_votes=self.minimum_votes)
        return result

    def predict_batch(self, images, **kwargs):
        """Fuse each image independently, in input order; an empty iterable returns []."""
        return [self.predict(image, **kwargs) for image in images]


__all__ = ['ConsensusWBFDetector']
