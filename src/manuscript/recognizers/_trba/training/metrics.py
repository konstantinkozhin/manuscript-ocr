"""
Text recognition metrics using HuggingFace Evaluate library.

This module provides standard metrics for evaluating OCR/text recognition:
- CER (Character Error Rate)
- WER (Word Error Rate)  
- Accuracy (exact match)
"""

import evaluate
from typing import List


# Load metrics once at module level for efficiency
_cer_metric = None
_wer_metric = None


def get_cer_metric():
    """Lazy load CER metric."""
    global _cer_metric
    if _cer_metric is None:
        _cer_metric = evaluate.load("cer")
    return _cer_metric


def get_wer_metric():
    """Lazy load WER metric."""
    global _wer_metric
    if _wer_metric is None:
        _wer_metric = evaluate.load("wer")
    return _wer_metric


def compute_cer(references: List[str], predictions: List[str]) -> float:
    """
    Compute Character Error Rate using HuggingFace evaluate.
    
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
    Compute Word Error Rate using HuggingFace evaluate.
    
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

