"""Exact unit-cost Levenshtein distance over Unicode code points."""


def levenshtein_distance(first: str, second: str) -> int:
    """Count insertions, deletions and substitutions; do not normalize text.

    A bit-vector dynamic program keeps one bit per code point in the shorter
    string. Python's arbitrary-precision integers handle lengths beyond a
    machine word, without a quadratic Python table or a native dependency.
    """
    if first == second:
        return 0
    if not first or not second:
        return len(first) + len(second)

    # Equal ends cannot affect edit distance. Trim them before allocating masks.
    start = 0
    end_first, end_second = len(first), len(second)
    while start < min(end_first, end_second) and first[start] == second[start]:
        start += 1
    while end_first > start and end_second > start and first[end_first - 1] == second[end_second - 1]:
        end_first -= 1
        end_second -= 1
    first, second = first[start:end_first], second[start:end_second]
    if len(first) > len(second):
        first, second = second, first
    size = len(first)
    if not size:
        return len(second)

    matches = {}
    for index, character in enumerate(first):
        matches[character] = matches.get(character, 0) | (1 << index)
    mask = (1 << size) - 1
    highest = 1 << (size - 1)
    positive, negative, distance = mask, 0, size
    for character in second:
        equal = matches.get(character, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        positive_horizontal = negative | ~(horizontal | positive)
        negative_horizontal = positive & horizontal
        distance += bool(positive_horizontal & highest) - bool(negative_horizontal & highest)
        positive_horizontal = (positive_horizontal << 1) | 1
        negative_horizontal <<= 1
        positive = (negative_horizontal | ~(vertical | positive_horizontal)) & mask
        negative = positive_horizontal & vertical & mask
    return distance
