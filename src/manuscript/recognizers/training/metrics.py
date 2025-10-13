import Levenshtein as lev
from jiwer import wer as jiwer_wer
from typing import List


def character_error_rate(reference: str, hypothesis: str) -> float:
    """
    Вычисляет Character Error Rate (CER):
    CER = edit_distance_chars / len(reference)
    """
    if len(reference) == 0:
        return float("inf") if len(hypothesis) > 0 else 0.0
    dist = lev.distance(reference, hypothesis)
    return dist / len(reference)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """
    Вычисляет Word Error Rate (WER) с помощью библиотеки jiwer.
    WER = количество ошибок на уровне слов / число слов в reference
    """
    return jiwer_wer(reference, hypothesis)


def compute_accuracy(references: List[str], hypotheses: List[str]) -> float:
    """
    Простая точность: доля точных совпадений (prediction == reference).
    """
    total = len(references)
    if total == 0:
        return 0.0
    hits = sum(1 for r, h in zip(references, hypotheses) if r == h)
    return hits / total
