import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import onnx


def save_constant_graph(path, input_name, input_shape, output_name, value):
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "Constant", [], [output_name], value=onnx.numpy_helper.from_array(value)
            )
        ],
        path.stem,
        [
            onnx.helper.make_tensor_value_info(
                input_name, onnx.TensorProto.FLOAT, input_shape
            )
        ],
        [
            onnx.helper.make_tensor_value_info(
                output_name, onnx.TensorProto.FLOAT, value.shape
            )
        ],
    )
    onnx.save(
        onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9
        ),
        path,
    )


def run_pipeline(directory):
    from manuscript import Pipeline, Page
    from manuscript.correctors import CharLM
    from manuscript.detectors import YOLO
    from manuscript.layouts import SimpleSorting
    from manuscript.recognizers import PPOCRRec

    detector_path = directory / "detector.onnx"
    save_constant_graph(
        detector_path,
        "images",
        [1, 3, 80, 80],
        "boxes",
        np.array([[[5, 25, 60, 45, 0.95, 0]]], np.float32),
    )
    recognizer_path = directory / "recognizer.onnx"
    logits = np.full((1, 7, 3), -20, np.float32)
    logits[0, np.arange(7), [2, 0, 2, 0, 2, 0, 2]] = 20
    save_constant_graph(recognizer_path, "images", [1, 3, 32, 64], "logits", logits)
    rec_config = PPOCRRec.create_config(
        {
            "rec_image_shape": [3, 32, 64],
            "PostProcess": {"character_dict": ["a", "b"]},
            "postprocess": {"output_is_logits": True},
        },
        recognition_levels=["word"],
        output_path=directory / "recognizer.json",
    )
    assert rec_config["recognition_levels"] == ["word"]
    vocab = ["<PAD>", "<MASK>", "<UNK>", "a", "b", " "]
    (directory / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    correction_logits = np.full(len(vocab), -20, np.float32)
    correction_logits[3] = 20
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Shape", ["input"], ["shape"]),
            onnx.helper.make_node(
                "Concat", ["shape", "vocab_size"], ["output_shape"], axis=0
            ),
            onnx.helper.make_node("Expand", ["values", "output_shape"], ["logits"]),
        ],
        "corrector",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.INT64, ["B", 8])],
        [
            onnx.helper.make_tensor_value_info(
                "logits", onnx.TensorProto.FLOAT, ["B", 8, len(vocab)]
            )
        ],
        [
            onnx.numpy_helper.from_array(correction_logits, "values"),
            onnx.numpy_helper.from_array(
                np.array([len(vocab)], np.int64), "vocab_size"
            ),
        ],
    )
    onnx.save(
        onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9
        ),
        directory / "corrector.onnx",
    )
    pipeline = Pipeline(
        detector=YOLO(
            weights=detector_path,
            config={"imgsz": 80, "region_classes": {0: {"type": "word"}}},
        ),
        layout=SimpleSorting(use_columns=False),
        recognizer=PPOCRRec(
            weights=recognizer_path, config=directory / "recognizer.json"
        ),
        corrector=CharLM(
            weights=directory / "corrector.onnx",
            vocab=directory / "vocab.json",
            max_len=8,
            max_edits=4,
        ),
    )
    page = pipeline.predict(np.full((40, 80, 3), 255, np.uint8))["page"]
    assert pipeline.get_text(pipeline.last_recognition_page) == "bbbb"
    assert page.get_text() == "aaaa"
    page.save(directory / "page.json")
    assert Page.load(directory / "page.json").to_tree_dict() == page.to_tree_dict()
    assert all(
        child.parent is node
        for node in (page, *page.descendants)
        for child in node.children
    )


def check_installed_package():
    import manuscript
    import manuscript.models.registry as registry
    import manuscript.recognizers._trba.configs as configs

    source_root = Path(__file__).resolve().parents[2] / "src"
    assert not Path(manuscript.__file__).resolve().is_relative_to(source_root)
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
    assert not Path(registry.__file__).with_name("registry.json").exists()
    assert Path(configs.__file__).with_name("charset.txt").read_text(encoding="utf-8")
    assert json.loads(
        Path(configs.__file__).with_name("config.json").read_text(encoding="utf-8")
    )
    with tempfile.TemporaryDirectory() as directory:
        run_pipeline(Path(directory))
    assert "torch" not in sys.modules
    print("Installed package resources and ONNX pipeline passed")


if __name__ == "__main__":
    check_installed_package()
