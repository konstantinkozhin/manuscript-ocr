"""
Text recognition metrics using JiWER directly, without model-hub dependencies.

This module provides standard metrics for evaluating OCR/text recognition:
- CER (Character Error Rate)
- WER (Word Error Rate)  
- Accuracy (exact match)
"""

import jiwer
from typing import List


# Load metrics once at module level for efficiency
_cer_metric = None
_wer_metric = None


class _JiwerMetric:
    """The same CER/WER backend used by Evaluate, without its heavy imports."""

    def __init__(self, name):
        self.function = jiwer.cer if name == "cer" else jiwer.wer

    def compute(self, *, predictions, references):
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have equal lengths")
        return self.function(references, predictions)


def get_cer_metric():
    """Lazy load CER metric."""
    global _cer_metric
    if _cer_metric is None:
        _cer_metric = _JiwerMetric("cer")
    return _cer_metric


def get_wer_metric():
    """Lazy load WER metric."""
    global _wer_metric
    if _wer_metric is None:
        _wer_metric = _JiwerMetric("wer")
    return _wer_metric


def compute_cer(references: List[str], predictions: List[str]) -> float:
    """
    Compute corpus Character Error Rate using JiWER.
    
    Args:
        references: List of ground truth strings
        predictions: List of predicted strings
        
    Returns:
        CER value (lower is better)
    """
    if len(references) == 0:
        return 0.0
    # CER требует непустые строки; заменяем пустые на пробел
    refs = [r if r else " " for r in references]
    preds = [p if p else " " for p in predictions]
    metric = get_cer_metric()
    return metric.compute(predictions=preds, references=refs)


def compute_wer(references: List[str], predictions: List[str]) -> float:
    """
    Compute corpus Word Error Rate using JiWER.
    
    Args:
        references: List of ground truth strings
        predictions: List of predicted strings
        
    Returns:
        WER value (lower is better)
    """
    if len(references) == 0:
        return 0.0
    # WER требует хотя бы одно слово в каждой строке; заменяем пустые на пробел
    refs = [r if r else " " for r in references]
    preds = [p if p else " " for p in predictions]
    metric = get_wer_metric()
    return metric.compute(predictions=preds, references=refs)


def compute_accuracy(references: List[str], predictions: List[str]) -> float:
    """
    Compute exact match accuracy.
    
    Args:
        references: List of ground truth strings
        predictions: List of predicted strings
        
    Returns:
        Accuracy in range [0, 1] (higher is better)
    """
    if len(references) == 0:
        return 0.0
    return sum(r == p for r, p in zip(references, predictions)) / len(references)

