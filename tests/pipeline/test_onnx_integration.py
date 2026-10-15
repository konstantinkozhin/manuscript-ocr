import pytest

pytest.importorskip("onnx")

from tests.package.wheel_smoke import run_pipeline


def test_onnx_stages_preserve_tree_and_correction(tmp_path):
    run_pipeline(tmp_path)
