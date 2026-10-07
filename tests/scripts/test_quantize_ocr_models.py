"""Check quantized weight axes, zero channels and shared float consumers."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

spec = importlib.util.spec_from_file_location(
    "quantize_ocr_models", Path(__file__).resolve().parents[2] / "scripts/quantize_ocr_models.py"
)
quantizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quantizer)


@pytest.mark.parametrize("op,transposed", [("MatMul", False), ("Gemm", False), ("Gemm", True)])
def test_weights_and_shared_constant(tmp_path, op, transposed):
    weights = np.array([[0, 1, -2], [0, .5, 3]], dtype=np.float32)
    if transposed:
        weights = weights.T.copy()
    tensor = onnx.numpy_helper.from_array(weights, "value")
    nodes = [onnx.helper.make_node("Constant", [], ["w"], value=tensor),
             onnx.helper.make_node(op, ["x", "w"], ["y"],
                                   **({"transB": int(transposed)} if op == "Gemm" else {})),
             onnx.helper.make_node("Identity", ["w"], ["original_weights"])]
    info = lambda name, shape: onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, shape)
    graph = onnx.helper.make_graph(nodes, "quantization", [info("x", [1, 2])],
                                   [info("y", [1, 3]), info("original_weights", list(weights.shape))])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)], ir_version=9)
    source, destination = tmp_path / "source.onnx", tmp_path / "int8.onnx"
    onnx.save(model, source)
    result = quantizer.quantize_weights(source, destination)
    assert result["quantized_weight_tensors"] == 1
    assert result["lifted_constants"] == 1
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    output, retained = session.run(None, {"x": np.array([[1, 2]], np.float32)})
    np.testing.assert_array_equal(retained, weights)
    np.testing.assert_allclose(output, [[0, 2, 4]], atol=.04)
    assert np.isfinite(output).all()
