"""Check exact distances against an independent scalar dynamic program."""
import itertools
import random

import pytest

from manuscript._internal.edit_distance import levenshtein_distance


def reference(first, second):
    row = list(range(len(second) + 1))
    for i, a in enumerate(first, 1):
        new = [i]
        for j, b in enumerate(second, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = new
    return row[-1]


def test_exhaustive_short_unicode_strings():
    alphabet = ['a', 'я', '\u0301', '🐠']
    words = [''.join(chars) for length in range(4) for chars in itertools.product(alphabet, repeat=length)]
    for first in words:
        for second in words:
            expected = reference(first, second)
            assert levenshtein_distance(first, second) == expected, (first, second)


@pytest.mark.parametrize('size', [63, 64, 65, 127, 128, 129, 513])
def test_distances_beyond_machine_word(size):
    rng = random.Random(size)
    first = ''.join(rng.choices('абвг🐠', k=size))
    second = ''.join(rng.choices('абвг🐠', k=size + 11))
    assert levenshtein_distance(first, second) == reference(first, second)
    assert levenshtein_distance(second, first) == reference(first, second)


def test_long_common_ends_and_distinct_unicode_forms():
    first = 'a' * 10000 + 'ѣ' + 'б' * 10000
    second = 'a' * 10000 + 'е' + 'б' * 10000
    assert levenshtein_distance(first, second) == 1
    assert levenshtein_distance('й', 'и\u0306') == 2
    assert levenshtein_distance('', '🐠') == 1
