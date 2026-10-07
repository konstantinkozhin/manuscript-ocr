"""Helpers loaded only by model exporters."""

def half_model(path, *, keep_io_types=True):
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    graph = convert_float_to_float16(onnx.load(path), keep_io_types=keep_io_types)
    pending = list(graph.graph.node)
    available = {item.name for item in graph.graph.input} | {item.name for item in graph.graph.initializer}
    ordered = []
    while pending:
        ready = [node for node in pending if all(not name or name in available for name in node.input)]
        if not ready:
            raise ValueError("Unresolved dependencies in FP16 graph")
        for node in ready:
            ordered.append(node)
            available.update(node.output)
            pending.remove(node)
    del graph.graph.node[:]
    graph.graph.node.extend(ordered)
    onnx.checker.check_model(graph)
    output = path.with_name(path.name.replace(".fp32.", ".fp16."))
    onnx.save(graph, output)
    return output
