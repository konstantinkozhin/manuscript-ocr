import sys
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from manuscript.api._page_helpers import accepts_predict_kwarg, filter_predict_kwargs
from manuscript.data import Page, TextRegion
from manuscript._internal.model_lifecycle import (
    ModelComposition, managed_prediction, processor_scope, validate_composition,
)
from manuscript.utils import read_image


class MajorityVotingRecognizer(ModelComposition):
    """Combine independent recognizers by whole-region plurality voting.

    Parameters
    ----------
    models : iterable of recognizers
        Nonempty ordered collection implementing predict(page, image=None).
        Each model receives an independent Page copy. Only TextRegion text
        and recognition_confidence may change; tree structure, child order,
        geometry and other fields must remain unchanged.
    parallel : bool, optional
        Run models concurrently in threads if True; default is False.
        Completion order does not change tie priority. Participants share
        the image array and must not modify it.
    memory_policy : {"keep", "sequential"}, optional
        Keep sessions loaded by default. "sequential" releases each member's
        sessions after its call, including on failure. Nested compositions
        inherit this policy. Requires close() on every participant and
        disallows parallel=True. Shared participants need external locking.

    Examples
    --------
    >>> from manuscript import Pipeline
    >>> recognizer = MajorityVotingRecognizer([model_a, model_b, model_c])
    >>> pipeline = Pipeline(recognizer=recognizer)
    >>> result = pipeline.predict("document.jpg")

    See Also
    --------
    manuscript.recognizers.MedoidRecognizer : Select by summed edit distance.
    manuscript.Stack : Apply models sequentially to preceding results.
    """

    batch_size = None

    def __init__(self, models, *, parallel=False, memory_policy="keep"):
        self._init_memory_policy(memory_policy)
        self.models = tuple(models)
        if not self.models:
            raise ValueError(f'{type(self).__name__} requires at least one recognizer')
        if not isinstance(parallel, bool):
            raise TypeError('parallel must be a boolean')
        for model in self.models:
            if not callable(getattr(model, 'predict', None)):
                raise TypeError('Voting participants must implement predict')
        if parallel and sys.platform == "emscripten":
            raise ValueError("Python thread parallelism is unavailable in Pyodide; use parallel=False.")
        self.parallel = parallel
        validate_composition(self)

    def _memory_children(self):
        return self.models

    @staticmethod
    def _nodes(page):
        def walk(node, path):
            yield path, node
            for index, child in enumerate(node.children):
                yield from walk(child, f'{path}.children[{index}]')
        return list(walk(page, 'page'))

    @classmethod
    def _check_tree(cls, reference, predicted, model_name):
        def fail(path, field):
            raise ValueError(f'{model_name}: recognizer changed {path}.{field}; '
                             'only TextRegion text and recognition_confidence may change')
        def compare(first, second, path):
            if type(first) is not type(second):
                fail(path, 'node_class')
            before, after = first.model_dump(mode='json'), second.model_dump(mode='json')
            allowed = {'text', 'recognition_confidence'} if isinstance(first, TextRegion) else set()
            for field in sorted(before.keys() | after.keys()):
                if field not in allowed and (field not in before or field not in after
                                             or before[field] != after[field]):
                    fail(path, field)
            if len(first.children) != len(second.children):
                fail(path, 'children')
            for index, (a, b) in enumerate(zip(first.children, second.children)):
                if b.parent is not second:
                    fail(f'{path}.children[{index}]', 'parent')
                compare(a, b, f'{path}.children[{index}]')
        if predicted.parent is not None:
            fail('page', 'parent')
        compare(reference, predicted, 'page')

    @managed_prediction
    def predict(self, page: Page, image=None, batch_size=None, *, return_raw=False, **kwargs):
        """Return a Page copy with aggregated recognition results.

        Parameters
        ----------
        page : Page
            Document tree to recognize; the supplied object is not modified.
        image : str, pathlib.Path, numpy.ndarray or PIL.Image.Image, optional
            Original image, loaded once for models accepting image.
        batch_size : int or None, optional
            Override participant batch sizes. None preserves model defaults.
        return_raw : bool, optional
            Return a dictionary with page and per-region voting records if True.
        **kwargs
            Forward supported predict options; participants use return_raw=False.

        Returns
        -------
        Page or dict
            Aggregated Page, or {"page": Page, "voting": records}. Records contain
            path, type, candidates, votes, winner, winning_votes,
            participating_models and total_models. Candidates contain model_index,
            text, normalized_text and confidence. Missing answers are omitted;
            winner is None without votes. Model tensors are not included.

        Raises
        ------
        TypeError
            If the input or a participant result is not a Page.
        ValueError
            If a tree is invalid or a participant changes forbidden fields.
            Contract errors identify the model, node path and changed field.

        Notes
        -----
        Participant exceptions abort aggregation. Models may target different
        levels; parent and descendant texts are aggregated independently.
        """
        if not isinstance(page, Page):
            raise TypeError(f'{type(self).__name__} requires a document-tree Page')
        page.validate_tree()
        reference = page.clone()
        image_array = (read_image(image) if image is not None
                       and any(accepts_predict_kwarg(model, 'image') for model in self.models) else None)

        def recognize(index):
            model = self.models[index]
            name = f'model[{index}] ({type(model).__name__})'
            candidate = reference.clone()
            for _, node in self._nodes(candidate):
                if isinstance(node, TextRegion):
                    node.text = None
                    node.recognition_confidence = None
            options = {**kwargs, 'image': image_array, 'return_raw': False}
            selected_batch = batch_size if batch_size is not None else getattr(model, 'batch_size', None)
            if selected_batch is not None:
                options['batch_size'] = selected_batch
            with processor_scope(model):
                prediction = model.predict(candidate, **filter_predict_kwargs(model, options))
            if not isinstance(prediction, Page):
                raise TypeError(f'{name}.predict must return Page')
            self._check_tree(reference, prediction, name)
            try:
                prediction.validate_tree()
            except (ValueError, TypeError) as exc:
                raise ValueError(f'{name}: invalid returned tree: {exc}') from exc
            return [(node.text, node.recognition_confidence)
                    for _, node in self._nodes(prediction) if isinstance(node, TextRegion)]

        if self.parallel:
            with ThreadPoolExecutor(max_workers=len(self.models)) as executor:
                predictions = list(executor.map(recognize, range(len(self.models))))
        else:
            predictions = [recognize(index) for index in range(len(self.models))]
        result = reference.clone()
        result_regions = [(path, node) for path, node in self._nodes(result) if isinstance(node, TextRegion)]
        records = []
        for region_index, (path, region) in enumerate(result_regions):
            candidates = []
            for model_index, prediction in enumerate(predictions):
                text, confidence = prediction[region_index]
                if text is None:
                    continue
                normalized = unicodedata.normalize('NFC', text).lower().strip()
                candidates.append({'model_index': model_index, 'text': text,
                                   'normalized_text': normalized, 'confidence': confidence})
            winner, counts, details = self._aggregate(candidates, region.type)
            if winner is not None:
                region.text = winner
                region.recognition_confidence = None
            if return_raw:
                records.append({'path': path, 'type': region.type, 'candidates': candidates,
                                'votes': dict(counts), 'winner': winner,
                                'winning_votes': counts[winner] if winner is not None else 0,
                                'participating_models': len(candidates), 'total_models': len(self.models),
                                **details})
        return {'page': result, 'voting': records} if return_raw else result

    def _aggregate(self, candidates, region_type):
        counts = Counter(candidate['normalized_text'] for candidate in candidates)
        # Counter insertion order is model order, so max resolves ties by priority.
        winner = max(counts, key=counts.get) if counts else None
        return winner, counts, {}


__all__ = ['MajorityVotingRecognizer']
