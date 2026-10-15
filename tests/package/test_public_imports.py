"""Public package imports remain usable without training dependencies."""
import subprocess
import sys


def test_public_runtime_api_imports_without_loading_training_libraries():
    source = '''
import sys
import manuscript
from manuscript import Pipeline, Page, CharLM, SimpleSorting, visualize_page
from manuscript.detectors import EAST, YOLO
from manuscript.recognizers import TRBA, PPOCRRec, TrOCR
from manuscript.layouts import ReadingSorting
assert all(getattr(manuscript, name) is not None for name in manuscript.__all__)
assert "torch" not in sys.modules
assert "transformers" not in sys.modules
assert "ultralytics" not in sys.modules
assert "paddle" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
