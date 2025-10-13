"""
Manuscript OCR - библиотека для детекции и распознавания текста
"""

from . import detectors
from . import recognizers

__version__ = "0.1.8"
__all__ = ["detectors", "recognizers"]