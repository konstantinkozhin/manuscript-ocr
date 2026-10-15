from collections import Counter
from statistics import median

from rapidfuzz.distance import Levenshtein

from .._majority_voting import MajorityVotingRecognizer


def _distance(first, second):
    """Unit-cost character Levenshtein using RapidFuzz."""
    return Levenshtein.distance(first, second)


class MedoidRecognizer(MajorityVotingRecognizer):
    """Select the existing text hypothesis closest to all model responses.

    Parameters
    ----------
    models : iterable of recognizers
        Nonempty ordered collection implementing predict(page, image=None).
        Each model receives an independent Page copy. Only TextRegion text
        and recognition_confidence may change; other changes raise an error.
    parallel : bool, optional
        Run models concurrently in threads if True; default is False.
        List order determines tie priority regardless of completion order.

    Notes
    -----
    Responses are NFC-normalized, lowercased and stripped. The winner minimizes
    the sum of unit-cost character Levenshtein distances to all responses,
    including duplicates. Ties prefer the first supporting model. Confidence
    is unused; the normalized winner receives recognition_confidence=None.
    None abstains, empty strings participate, and regions without responses
    retain their input values. No new string is generated.

    For every text region, discard answers longer than max(512, 8 * median
    length) if at least three answers remain, regardless of the region type.
    Distances are computed using RapidFuzz.

    The inherited predict method accepts image, batch_size and return_raw.
    Raw voting records add distance_sums, winning_distance_sum, used_models,
    excluded_model_indices and length_limit. Candidates remain available before
    filtering; votes count exact occurrences after filtering.

    Examples
    --------
    >>> recognizer = MedoidRecognizer([model_a, model_b, model_c])
    >>> result = recognizer.predict(page, image=image, return_raw=True)
    >>> scores = result["voting"][0]["distance_sums"]

    See Also
    --------
    MajorityVotingRecognizer : Vote by exact whole-string agreement.
    """

    def _aggregate(self, candidates, region_type):
        included = candidates
        limit = None
        if candidates:
            limit = max(512, int(median(len(c['normalized_text']) for c in candidates) * 8))
            filtered = [c for c in candidates if len(c['normalized_text']) <= limit]
            if len(filtered) >= 3:
                included = filtered
        counts = Counter(candidate['normalized_text'] for candidate in included)
        scores = dict.fromkeys(counts, 0)
        unique = list(counts)
        for index, first in enumerate(unique):
            for second in unique[index + 1:]:
                distance = _distance(first, second)
                scores[first] += counts[second] * distance
                scores[second] += counts[first] * distance
        winner = min(scores, key=scores.get) if scores else None
        kept = {c['model_index'] for c in included}
        details = {
            'aggregation': 'medoid', 'distance_sums': scores,
            'winning_distance_sum': scores[winner] if winner is not None else None,
            'used_models': len(included),
            'excluded_model_indices': [c['model_index'] for c in candidates if c['model_index'] not in kept],
            'length_limit': limit,
        }
        return winner, counts, details


__all__ = ['MedoidRecognizer']
