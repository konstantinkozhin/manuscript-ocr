"""Consensus geometry and composition contracts without model artifacts."""
import math
import pickle

import numpy as np
import pytest

from manuscript import Pipeline
from manuscript.data import Page, TextRegion
from manuscript.detectors import ConsensusWBFDetector


class Detector:
    def __init__(self, boxes=(), *, kind='line', label=None, confidence=.9):
        self.boxes, self.kind, self.label = boxes, kind, label
        self.confidence = confidence
        self.closed = 0

    def predict(self, image):
        h, w = image.shape[:2]
        page = Page(image_height=h, image_width=w)
        nodes = []
        for x1, y1, x2, y2 in self.boxes:
            nodes.append(TextRegion(type=self.kind, label=self.label,
                polygon=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                detection_confidence=self.confidence))
        page.children = nodes
        return page

    def close(self):
        self.closed += 1


IMAGE = np.zeros((100, 200, 3), dtype=np.uint8)
A = (0, 0, 10, 10)
B = (2, 0, 12, 10)


def run(detectors, **kwargs):
    return ConsensusWBFDetector(detectors, **kwargs).predict(IMAGE, return_raw=True)


def test_equal_and_weighted_coordinates():
    models = [Detector([A], confidence=.01), Detector([B], confidence=.99), Detector()]
    raw = run(models)
    assert raw['detections'][0]['box'] == [1, 0, 11, 10]
    assert raw['detections'][0]['votes'] == 2
    assert raw['detections'][0]['detector_indices'] == [0, 1]
    assert raw['page'].children[0].detection_confidence is None
    weighted = run(models, weights=[1, 3, 1])
    assert weighted['detections'][0]['box'] == [1.5, 0, 11.5, 10]
    assert run(models, weights=[1e308, 1e308, 1e308])['detections'] == raw['detections']
    assert run([Detector([A]), Detector(), Detector()])['detections'] == []


@pytest.mark.parametrize('count, required', [(3, 2), (4, 2), (5, 3)])
def test_vote_threshold_and_empty_answers(count, required):
    ensemble = ConsensusWBFDetector([Detector([A])] * count)
    assert ensemble.required_votes == required
    assert len(run([Detector([A])] * required + [Detector()] * (count-required))['detections']) == 1
    assert run([Detector([A])] * (required-1) + [Detector()] * (count-required+1))['detections'] == []


def test_one_vote_per_participant_and_descending_ties():
    raw = run([Detector([A]), Detector([A, B])], cluster_iou=.5)
    assert raw['detections'][0]['votes'] == 2
    assert len(raw['detections']) == 1
    # Equal IoU: highest box index and highest cluster index are matched first.
    raw = run([Detector([A, (4, 0, 14, 10)]), Detector([B])])
    assert raw['detections'][0]['detector_indices'] == [0, 1]
    assert len(raw['detections']) == 1
    assert raw['detections'][0]['box'] == [3, 0, 13, 10]
    raw = run([Detector([B]), Detector([A, (4, 0, 14, 10)])])
    assert raw['detections'][0]['box'] == [3, 0, 13, 10]
    raw = run([Detector([A]), Detector([A, B])], cluster_iou=0, minimum_votes=1)
    assert raw['detections'][0]['box'] == [0, 0, 10, 10]
    assert raw['detections'][1]['box'] == list(B)


def test_nontransitive_and_weighted_representatives():
    # A~B and B~C, but mean(A,B) is too far from C.
    models = [Detector([A]), Detector([B]), Detector([(4, 0, 14, 10)])]
    equal = run(models, cluster_iou=.6, minimum_votes=3)
    assert equal['detections'] == []
    weighted = run(models, cluster_iou=.6, minimum_votes=3, weights=[1, 9, 1])
    assert weighted['detections'][0]['votes'] == 3


def test_semantic_classes_and_batches():
    assert run([Detector([A], kind='word'), Detector([A], kind='line')])['detections'] == []
    assert run([Detector([A], label='a'), Detector([A], label='b')])['detections'] == []
    ensemble = ConsensusWBFDetector([Detector([A]), Detector([A])])
    assert ensemble.predict_batch([]) == []
    pages = ensemble.predict_batch([IMAGE, np.zeros((50, 60, 3), np.uint8)])
    assert [(p.image_height, p.image_width) for p in pages] == [(100, 200), (50, 60)]
    assert pages[0] is not pages[1]
    assert run([Detector(), Detector()])['page'].children == ()


def test_coordinate_contract_and_inference_error():
    class WrongSize:
        def predict(self, image):
            return Page(image_height=10, image_width=20)
    with pytest.raises(ValueError, match='original-image'):
        run([WrongSize(), Detector()])
    class Broken:
        def predict(self, image):
            raise RuntimeError('inference failed')
    with pytest.raises(RuntimeError, match='inference failed'):
        run([Broken(), Detector()])
    class Invalid:
        def predict(self, image):
            return []
    with pytest.raises(TypeError, match=r'detector\[0\]'):
        run([Invalid(), Detector()])


def test_polygon_envelope_and_restored_coordinates():
    class ResizingDetector:
        def predict(self, image):
            # Simulated participant restores coordinates from its half-size input.
            page = Page(image_height=image.shape[0], image_width=image.shape[1])
            poly = np.array([(5, 0), (10, 5), (5, 10), (0, 5)]) * 2
            page.children = [TextRegion(type='line', polygon=poly.tolist())]
            return {'page': page}
    raw = run([ResizingDetector(), Detector([(0, 0, 20, 20)])])
    assert raw['detections'][0]['box'] == [0, 0, 20, 20]
    assert raw['page'].children[0].polygon == [(0, 0), (20, 0), (20, 20), (0, 20)]


def test_pipeline_pickle_memory_and_repeatability():
    detectors = [Detector([A]), Detector([B])]
    ensemble = ConsensusWBFDetector(detectors, weights=[1, 2], memory_policy='sequential')
    restored = pickle.loads(pickle.dumps(ensemble))
    assert restored.weights == (1, 2)
    assert restored.predict(IMAGE).to_tree_dict() == ensemble.predict(IMAGE).to_tree_dict()
    first = ensemble.predict(IMAGE).to_tree_dict()
    assert ensemble.predict(IMAGE).to_tree_dict() == first
    page = Pipeline(detector=ensemble, layout=None, recognizer=None).predict(IMAGE)['page']
    assert page.to_tree_dict() == first
    assert all(d.closed > 0 for d in detectors)


@pytest.mark.parametrize('kwargs', [
    {'weights': [1]}, {'weights': [0, 1]}, {'weights': [-1, 1]},
    {'weights': [math.inf, 1]}, {'weights': [True, 1]},
    {'cluster_iou': math.nan}, {'cluster_iou': 1.1},
    {'consensus_fraction': -1}, {'minimum_votes': 0}, {'minimum_votes': True},
])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        ConsensusWBFDetector([Detector(), Detector()], **kwargs)


def test_equal_weights_match_reference_on_seeded_pages():
    def reference(by_model):
        clusters = []
        for boxes in by_model:
            representatives = [np.mean(c, axis=0) for c in clusters]
            pairs = []
            for i, box in enumerate(boxes):
                a = np.asarray(box)
                for j, b in enumerate(representatives):
                    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
                    union = (a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - intersection
                    iou = intersection / union if union > 0 else 0
                    if iou >= .5:
                        pairs.append((float(iou), i, j))
            used_boxes, used_clusters = set(), set()
            for _, i, j in sorted(pairs, reverse=True):
                if i not in used_boxes and j not in used_clusters:
                    clusters[j].append(boxes[i])
                    used_boxes.add(i)
                    used_clusters.add(j)
            clusters.extend([box] for i, box in enumerate(boxes) if i not in used_boxes)
        required = max(2, math.ceil(len(by_model) * .5))
        return [np.mean(c, axis=0).tolist() for c in clusters if len(c) >= required]

    rng = np.random.default_rng(42)
    for _ in range(24):
        by_model = []
        for _ in range(5):
            origins = rng.integers(0, 20, size=(rng.integers(0, 12), 2))
            by_model.append([tuple(np.concatenate((p, p + 15))) for p in origins])
        actual = run([Detector(boxes) for boxes in by_model])
        assert [item['box'] for item in actual['detections']] == reference(by_model)
