import numpy as np
import pytest

onnx = pytest.importorskip("onnx")

from manuscript.utils._onnx_export import half_model


@pytest.mark.parametrize("keep_io_types", [True, False])
@pytest.mark.parametrize("filename", ["model.fp32.onnx", "model.onnx"])
def test_half_model_preserves_source_and_runs_dynamic_batches(
    tmp_path, keep_io_types, filename
):
    import onnxruntime as ort

    path = tmp_path / filename
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Mul", ["images", "scale"], ["scaled"]),
            onnx.helper.make_node("Add", ["scaled", "bias"], ["output"]),
        ],
        "scale",
        [
            onnx.helper.make_tensor_value_info(
                "images", onnx.TensorProto.FLOAT, ["B", 3]
            )
        ],
        [
            onnx.helper.make_tensor_value_info(
                "output", onnx.TensorProto.FLOAT, ["B", 3]
            )
        ],
        [
            onnx.numpy_helper.from_array(np.array([0.5], np.float32), "scale"),
            onnx.numpy_helper.from_array(np.array([0.25], np.float32), "bias"),
        ],
    )
    onnx.save(
        onnx.helper.make_model(
            graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9
        ),
        path,
    )
    original = path.read_bytes()
    converted = half_model(str(path), keep_io_types=keep_io_types)
    assert path.read_bytes() == original
    assert converted != path and converted.name == "model.fp16.onnx"
    onnx.checker.check_model(str(converted))
    session = ort.InferenceSession(str(converted), providers=["CPUExecutionProvider"])
    dtype = np.float32 if keep_io_types else np.float16
    for batch in (1, 3):
        values = np.arange(batch * 3).reshape(batch, 3).astype(dtype)
        output = session.run(None, {"images": values})[0]
        assert output.dtype == dtype
        np.testing.assert_array_equal(output, values * 0.5 + 0.25)
