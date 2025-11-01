from manuscript.detectors import EAST


EAST.export_to_onnx(
    weights_path=r"C:\Users\pasha\.manuscript\east\east_quad_23_05.pth",
    output_path="east_model.onnx",
)
