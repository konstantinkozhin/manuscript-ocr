import json
from types import SimpleNamespace

import numpy as np
import pytest

from manuscript import Pipeline
from manuscript.data import Page
from manuscript.detectors._rfdetr import RFDETR


@pytest.fixture
def class_config():
    return {
        "text_class_ids": [2, 3, 4, 5],
        "class_names": {
            "1": "text_region", "2": "text_line_upright",
            "3": "text_line_upside_down", "4": "text_line_rotated_cw",
            "5": "text_line_rotated_ccw",
        },
    }


@pytest.fixture
def weights(tmp_path, class_config):
    path = tmp_path / "model.onnx"
    path.write_bytes(b"fake-session")
    path.with_suffix(".json").write_text(json.dumps(class_config))
    return path


def outputs():
    boxes = np.array(
        [
            [
                [0.5, 0.5, 0.8, 0.8],
                [0.5, 0.25, 0.6, 0.1],
                [0.5, 0.45, 0.6, 0.1],
                [0.3, 0.5, 0.1, 0.6],
                [0.7, 0.5, 0.1, 0.6],
            ]
        ],
        np.float32,
    )
    logits = np.full((1, 5, 6), -20, np.float32)
    logits[0, np.arange(5), np.arange(1, 6)] = [6, 5, 4, 3, 2]
    return boxes, logits


def test_four_orientations_become_text_and_full_mode_keeps_region(weights):
    model = RFDETR(weights)
    page = model._postprocess(*outputs(), (100, 200))
    assert isinstance(page, Page)
    assert len(page.blocks[0].lines[0].text_spans) == 4
    result = model._postprocess(*outputs(), (100, 200), return_raw=True)
    assert [row["class_id"] for row in result["detections"]] == [1, 2, 3, 4, 5]
    assert [row["class_name"] for row in result["detections"]] == list(
        model.class_names.values()
    )
    np.testing.assert_allclose(
        result["detections"][1]["bbox"], [40, 20, 160, 30], atol=1e-5
    )
    assert len(result["page"].blocks[0].lines[0].text_spans) == 4
    assert result["image_size"] == {"height": 100, "width": 200}


def test_custom_classes_do_not_change_full_result(weights):
    model = RFDETR(weights, class_ids=[1], score_thresh=0.95)
    result = model._postprocess(*outputs(), (100, 200), return_raw=True)
    assert [row["class_id"] for row in result["detections"]] == [1, 2, 3, 4]
    assert len(result["page"].blocks[0].lines[0].text_spans) == 1


def test_topk_is_global_stable_and_no_nms(weights, class_config):
    class_config["num_select"] = 2
    weights.with_suffix(".json").write_text(json.dumps(class_config))
    model = RFDETR(weights)
    boxes = np.array([[[0.5, 0.5, 0.5, 0.5]]], np.float32)
    logits = np.full((1, 1, 6), -100, np.float32)
    logits[:, :, 2:4] = 5
    result = model._postprocess(boxes, logits, (100, 100), return_raw=True)
    assert [row["class_id"] for row in result["detections"]] == [2, 3]
    assert result["detections"][0]["bbox"] == result["detections"][1]["bbox"]


def test_boxes_are_clipped_invalid_and_empty_are_removed(weights):
    model = RFDETR(weights)
    boxes = np.array([[[0, 0, 1, 1], [0.5, 0.5, -1, 1], [3, 3, 1, 1]]], np.float32)
    logits = np.full((1, 3, 6), -100, np.float32)
    logits[:, :, 2] = 10
    result = model._postprocess(boxes, logits, (80, 120), return_raw=True)
    assert len(result["detections"]) == 1
    assert result["detections"][0]["bbox"] == [0, 0, 60, 40]
    assert (
        not model._postprocess(np.empty((1, 0, 4)), np.empty((1, 0, 6)), (80, 120))
        .blocks[0]
        .lines[0]
        .text_spans
    )


@pytest.mark.parametrize("dtype", ["tensor(float)", "tensor(float16)"])
def test_public_predict_and_pipeline_use_fixed_graph_shape(weights, monkeypatch, dtype):
    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name="images", shape=[1, 3, 64, 64], type=dtype)]

        def get_outputs(self):
            return [
                SimpleNamespace(name="pred_logits"),
                SimpleNamespace(name="pred_boxes"),
            ]

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def run(self, names, feed):
            assert names == ["pred_boxes", "pred_logits"]
            assert feed["images"].shape == (1, 3, 64, 64)
            assert feed["images"].dtype == (
                np.float16 if "float16" in dtype else np.float32
            )
            assert feed["images"].flags.c_contiguous
            return outputs()

    monkeypatch.setattr(
        "manuscript.detectors._rfdetr.ort.InferenceSession", lambda *a, **kw: Session()
    )
    model = RFDETR(weights)
    image = np.zeros((100, 200, 3), np.uint8)
    assert len(model.predict(image, return_raw=True)["detections"]) == 5
    page = Pipeline(detector=model, recognizer=None, layout=None).predict(image)["page"]
    assert len(page.blocks[0].lines[0].text_spans) == 4


def test_future_registry_and_explicit_config_priority(weights, tmp_path, monkeypatch):
    config = tmp_path / "registry.json"
    config.write_text(
        json.dumps(
            {"score_thresh": 0.7, "text_class_ids": [7], "class_names": {"7": "line"}}
        )
    )
    calls = []

    def resolve(name, model_class, force_download=False):
        calls.append((name, model_class, force_download))
        return {"weights": weights, "config": config}

    monkeypatch.setattr("manuscript.models.resolve", resolve)
    model = RFDETR("future_rf_model", force_download=True)
    assert calls == [("future_rf_model", "RFDETR", True)]
    assert model.class_ids == (7,)
    assert model.class_names == {7: "line"}
    assert RFDETR("future_rf_model", config={"text_class_ids": [9], "class_names": {"9": "custom"}}).class_ids == (9,)


@pytest.mark.parametrize(
    "config",
    [
        {"score_thresh": -1},
        {"std": [0, 1, 1]},
        {"text_class_ids": [-1]},
        {"target_size": 0},
        {"schema_version": 2},
        {"activation": "invalid"},
    ],
)
def test_invalid_configuration(weights, config, class_config):
    class_config.update(config)
    with pytest.raises(ValueError):
        RFDETR(weights, config=class_config)


def test_invalid_outputs_fail_explicitly(weights):
    model = RFDETR(weights)
    with pytest.raises(ValueError, match="non-finite"):
        model._postprocess(np.ones((1, 2, 4)), np.full((1, 2, 6), np.nan), (100, 100))
    with pytest.raises(ValueError, match="boxes"):
        model._postprocess(np.ones((2, 2, 4)), np.ones((2, 2, 6)), (100, 100))


def test_segmentation_preserves_contours_holes_and_masks(weights):
    import cv2

    boxes, logits = outputs()
    masks = np.full((1, 5, 32, 32), -10, np.float32)
    masks[:, :, 5:25, 5:25] = 10
    masks[:, :, 10:15, 10:15] = -10  # retain holes in the full result
    masks[:, :, 27:30, 27:30] = 10  # retain disconnected components
    model = RFDETR(weights)
    result = model._postprocess(
        boxes, logits, (32, 32), True, masks=masks, return_masks=True, return_outputs=True
    )
    compact = model._postprocess(boxes, logits, (32, 32), return_raw=True, masks=masks)
    assert "outputs" not in compact and all("mask" not in row for row in compact["detections"])
    assert compact["page"] == result["page"]
    assert [row["polygon"] for row in compact["detections"]] == [row["polygon"] for row in result["detections"]]
    only_masks = model._postprocess(boxes, logits, (32, 32), masks=masks, return_masks=True)
    assert "outputs" not in only_masks and "mask" in only_masks["detections"][0]
    only_outputs = model._postprocess(boxes, logits, (32, 32), masks=masks, return_outputs=True)
    assert "outputs" in only_outputs and "mask" not in only_outputs["detections"][0]
    row = result["detections"][1]
    assert row["geometry_source"] == "mask_polygon"
    assert len(row["contours"]) == 3
    assert any(item[3] >= 0 for item in row["contour_hierarchy"])
    np.testing.assert_array_equal(row["mask"], masks[0, 1] > 0)
    assert len(result["page"].blocks[0].lines[0].text_spans) == 4
    assert result["outputs"]["pred_masks"] is masks
    with pytest.raises(ValueError, match="masks"):
        model._postprocess(boxes, logits, (32, 32), masks=np.zeros((1, 4, 32, 32)))


def test_missing_config_fails(weights):
    weights.with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="requires a configuration"):
        RFDETR(weights)


@pytest.mark.parametrize("config,match", [
    ({"text_class_ids": [0]}, "class_names"),
    ({"class_names": {"0": "arbitrary"}}, "text_class_ids"),
    ({"class_names": {"0": ""}, "text_class_ids": [0]}, "nonempty strings"),
    ({"class_names": {"-1": "bad"}, "text_class_ids": []}, "nonnegative"),
    ({"class_names": {"0": "arbitrary"}, "text_class_ids": [1]}, "selected class_id"),
])
def test_invalid_class_metadata(weights, config, match):
    with pytest.raises(ValueError, match=match):
        RFDETR(weights, config=config)


def test_arbitrary_class_metadata_and_explicit_selection(weights):
    config = {"class_names": {"1": "table", "2": "formula"}, "text_class_ids": [1]}
    model = RFDETR(weights, config=config, class_ids=[2])
    result = model._postprocess(*outputs(), (100, 200), return_raw=True)
    assert result["detections"][0]["class_name"] == "table"
    assert result["detections"][1]["class_name"] == "formula"
    assert len(result["page"].blocks[0].lines[0].text_spans) == 1
    assert config["text_class_ids"] == [1]
