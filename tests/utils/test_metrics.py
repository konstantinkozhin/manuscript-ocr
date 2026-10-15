"""Reliability tests for detection metrics helpers."""

import multiprocessing as mp

import manuscript.utils.metrics as metrics_module


class FakePool:
    def __init__(self, processes=None):
        self.processes = processes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def map(self, func, iterable):
        return [func(item) for item in iterable]

    def imap(self, func, iterable):
        for item in iterable:
            yield func(item)


class FakeContext:
    def Pool(self, processes=None):
        return FakePool(processes=processes)


class TestMatchBoxes:
    def test_match_boxes_handles_empty_inputs(self):
        assert metrics_module._match_boxes([], []) == (0, 0, 0)
        assert metrics_module._match_boxes([(0, 0, 10, 10)], []) == (0, 1, 0)
        assert metrics_module._match_boxes([], [(0, 0, 10, 10)]) == (0, 0, 1)

    def test_match_boxes_uses_greedy_one_to_one_matching(self):
        pred_boxes = [
            (0, 0, 10, 10),
            (0, 0, 10, 10),
            (20, 20, 30, 30),
        ]
        gt_boxes = [
            (0, 0, 10, 10),
            (20, 20, 30, 30),
        ]

        tp, fp, fn = metrics_module._match_boxes(
            pred_boxes,
            gt_boxes,
            iou_threshold=0.5,
        )

        assert (tp, fp, fn) == (2, 1, 0)


class TestF1Helpers:
    def test_compute_f1_score_handles_zero_cases(self):
        assert metrics_module._compute_f1_score(0, 0, 5) == (0.0, 0.0, 0.0)
        assert metrics_module._compute_f1_score(0, 3, 0) == (0.0, 0.0, 0.0)

    def test_compute_f1_score_returns_expected_values(self):
        f1, precision, recall = metrics_module._compute_f1_score(2, 1, 1)

        assert precision == 2 / 3
        assert recall == 2 / 3
        assert round(f1, 6) == round(2 * precision * recall / (precision + recall), 6)

    def test_evaluate_image_worker_returns_threshold_metrics(self):
        result = metrics_module._evaluate_image_worker(
            (
                "img-1",
                [(0, 0, 10, 10)],
                [(0, 0, 10, 10)],
                [0.5, 0.75],
            )
        )

        assert result[0.5] == (1, 0, 0)
        assert result[0.75] == (1, 0, 0)


class TestDatasetEvaluation:
    def test_evaluate_dataset_sequential_aggregates_summary_metrics(self):
        predictions = {
            "img1": [(0, 0, 10, 10), (30, 30, 40, 40)],
            "img2": [],
        }
        ground_truths = {
            "img1": [(0, 0, 10, 10)],
            "img3": [(5, 5, 15, 15)],
        }

        result = metrics_module._evaluate_dataset(
            predictions,
            ground_truths,
            iou_thresholds=[0.5],
            verbose=False,
            n_jobs=1,
        )

        assert result["tp@0.50"] == 1
        assert result["fp@0.50"] == 1
        assert result["fn@0.50"] == 1
        assert result["f1@0.5"] == result["f1@0.50"]
        assert result["precision@0.5"] == result["precision@0.50"]
        assert result["recall@0.5"] == result["recall@0.50"]
        assert result["num_images"] == 3
        assert result["num_predictions"] == 2
        assert result["num_ground_truths"] == 2

    def test_evaluate_dataset_parallel_path_uses_pool(self, monkeypatch):
        monkeypatch.setattr(mp, "cpu_count", lambda: 2)
        monkeypatch.setattr(mp, "get_context", lambda _name: FakeContext())
        monkeypatch.setattr(metrics_module, "tqdm", lambda iterable, **_: iterable)

        predictions = {"img1": [(0, 0, 10, 10)]}
        ground_truths = {"img1": [(0, 0, 10, 10)]}

        result = metrics_module._evaluate_dataset(
            predictions,
            ground_truths,
            iou_thresholds=[0.5, 0.75],
            verbose=True,
            n_jobs=2,
        )

        assert result["tp@0.50"] == 1
        assert result["tp@0.75"] == 1
        assert result["f1@0.5:0.95"] == 1.0
