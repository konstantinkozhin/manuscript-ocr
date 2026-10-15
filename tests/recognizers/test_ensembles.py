"""Voting behavior and recognizer tree contracts without model artifacts."""
import threading

import pytest

from manuscript.data import Page, TextRegion
from manuscript.recognizers import MajorityVotingRecognizer, MedoidRecognizer


class Recognizer:
    batch_size = 7

    def __init__(self, text, *, kind='word', barrier=None):
        self.text = text
        self.kind = kind
        self.barrier = barrier
        self.calls = 0

    def predict(self, page, image=None, batch_size=None, return_raw=False):
        self.calls += 1
        self.received_batch_size = batch_size
        assert not return_raw
        if self.barrier is not None:
            self.barrier.wait(timeout=5)
        for node in page.descendants:
            if isinstance(node, TextRegion) and node.type == self.kind:
                assert node.text is None and node.recognition_confidence is None
                node.text = self.text
                node.recognition_confidence = .9 if self.text is not None else None
        return page


@pytest.fixture
def page():
    result = Page(image_width=100, image_height=100)
    block = TextRegion(type='block')
    line = TextRegion(type='line')
    line.children = [TextRegion(type='word', text='old', recognition_confidence=.8,
        polygon=[(x, 0), (x + 8, 0), (x + 8, 8), (x, 8)]) for x in (0, 20)]
    block.children = [line]
    result.children = [block]
    return result


@pytest.fixture(params=[MajorityVotingRecognizer, MedoidRecognizer])
def ensemble(request):
    return request.param


def first_word(page):
    return page.children[0].children[0].children[0]


@pytest.mark.parametrize('parallel', [False, True])
def test_normalized_votes_preserve_input_and_model_batch_defaults(page, ensemble, parallel):
    before = page.to_tree_dict()
    models = [Recognizer(' Е\u0308Ж '), Recognizer('ёж'), Recognizer('еж')]
    raw = ensemble(models, parallel=parallel).predict(page, return_raw=True)
    assert first_word(raw['page']).text == 'ёж'
    assert first_word(raw['page']).recognition_confidence is None
    assert page.to_tree_dict() == before
    assert all(model.received_batch_size == 7 for model in models)
    record = next(record for record in raw['voting'] if record['type'] == 'word')
    assert record['votes'] == {'ёж': 2, 'еж': 1}
    assert record['winning_votes'] == 2
    ensemble(models).predict(page, batch_size=3)
    assert all(model.received_batch_size == 3 for model in models)


def test_missing_answers_abstain_but_empty_strings_vote(page, ensemble):
    result = ensemble([Recognizer(None)]).predict(page)
    assert first_word(result).text == 'old'
    assert first_word(result).recognition_confidence == .8
    result = ensemble([Recognizer(None), Recognizer('')]).predict(page)
    assert first_word(result).text == ''


def test_ties_prefer_first_model_even_in_parallel(page, ensemble):
    barrier = threading.Barrier(3)
    models = [Recognizer(value, barrier=barrier) for value in ('a', 'b', 'c')]
    result = ensemble(models, parallel=True).predict(page)
    assert first_word(result).text == 'a'


@pytest.mark.parametrize('field', ['file_name', 'polygon', 'order', 'children', 'orientation'])
def test_tree_changes_abort_without_mutating_input(page, ensemble, field):
    class Invalid:
        def predict(self, page):
            line = page.children[0].children[0]
            if field == 'file_name':
                page.file_name = 'changed'
            elif field == 'polygon':
                first_word(page).polygon = [(0, 0), (10, 0), (10, 10), (0, 10)]
            elif field == 'order':
                line.children = line.children[::-1]
            elif field == 'children':
                line.children = line.children[:1]
            else:
                line.orientation = 90
            return page
    before = page.to_tree_dict()
    later = Recognizer('later')
    with pytest.raises(ValueError, match=r'model\[0\].*page'):
        ensemble([Invalid(), later]).predict(page)
    assert page.to_tree_dict() == before
    assert later.calls == 0


def test_region_levels_are_aggregated_independently(page, ensemble):
    result = ensemble([Recognizer('WORD'), Recognizer('LINE', kind='line')]).predict(page)
    assert first_word(result).text == 'word'
    assert result.children[0].children[0].text == 'line'


def test_plurality_and_medoid_make_different_selections(page):
    models = [Recognizer(value) for value in ('aab', 'aba', 'baa', 'aaa', 'zzz', 'zzz')]
    assert first_word(MajorityVotingRecognizer(models).predict(page)).text == 'zzz'
    raw = MedoidRecognizer(models).predict(page, return_raw=True)
    assert first_word(raw['page']).text == 'aaa'
    record = next(record for record in raw['voting'] if record['type'] == 'word')
    assert record['winning_distance_sum'] == 9


@pytest.mark.parametrize('kind', ['word', 'line', 'block'])
def test_medoid_length_guard_for_every_text_level(page, kind):
    models = [Recognizer(value, kind=kind) for value in ('text', 'text', 'texts', 'x' * 600)]
    raw = MedoidRecognizer(models).predict(page, return_raw=True)
    record = next(record for record in raw['voting'] if record['type'] == kind)
    assert record['winner'] == 'text'
    assert record['excluded_model_indices'] == [3]
    assert record['used_models'] == 3 and record['length_limit'] == 512
    raw = MedoidRecognizer(models[1:]).predict(page, return_raw=True)
    record = next(record for record in raw['voting'] if record['type'] == kind)
    assert record['excluded_model_indices'] == []


def test_invalid_participant_result_is_rejected(page, ensemble):
    class Invalid:
        def predict(self, page):
            return {'page': page}
    with pytest.raises(TypeError, match=r'model\[0\].*return Page'):
        ensemble([Invalid()]).predict(page)
    with pytest.raises(ValueError, match='at least one'):
        ensemble([])
    with pytest.raises(TypeError, match='parallel'):
        ensemble([Recognizer('text')], parallel=1)
