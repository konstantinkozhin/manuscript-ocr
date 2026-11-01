"""Session residency, cleanup and propagation through processor compositions."""
from concurrent.futures import ThreadPoolExecutor
import pickle
from threading import Event

import numpy as np
import pytest

from manuscript import Pipeline, Stack
from manuscript.api.base import BaseArtifactModel
from manuscript.data import Page, TextRegion
from manuscript.recognizers import MajorityVotingRecognizer, MedoidRecognizer, PPOCRRec


class Participant:
    def __init__(self, active, name='model', fail=None):
        self.active, self.name, self.fail = active, name, fail
        self.loads = 0
        self.busy = False

    def run(self):
        assert not (self.active - {self.name}), 'More than one model resident'
        if self.name not in self.active:
            self.loads += 1
        self.active.add(self.name)
        if self.fail:
            raise self.fail

    def close(self):
        assert not self.busy, 'Closed during inference'
        self.active.discard(self.name)


class Detector(Participant):
    def predict(self, image):
        self.run()
        page = Page(image_width=20, image_height=20)
        page.children = [TextRegion(type='word', polygon=[(0,0),(10,0),(10,10),(0,10)])]
        return page


class Recognizer(Participant):
    def predict(self, page, image=None):
        self.run()
        result = page.clone()
        for region in result.descendants:
            if isinstance(region, TextRegion):
                region.text = 'answer'
        return result


def page():
    return Detector(set()).predict(None)


@pytest.mark.parametrize('ensemble', [MajorityVotingRecognizer, MedoidRecognizer])
def test_ensembles_release_each_member_and_reload_on_next_call(ensemble):
    active = set()
    models = [Recognizer(active, str(i)) for i in range(3)]
    # Warming multiple models beforehand must not defeat the policy.
    active.update(str(i) for i in range(3))
    recognizer = ensemble(models, memory_policy='sequential')
    for _ in range(2):
        result = recognizer.predict(page(), return_raw=True)
        assert result['page'].get_text() == 'answer'
        assert result['voting'][0]['total_models'] == 3
        assert not active
    assert [m.loads for m in models] == [2,2,2]


def test_pipeline_propagates_to_nested_stacks_and_ensembles_without_mutation():
    active = set()
    detectors = Stack([Stack([Detector(active, 'a')]), Detector(active, 'b')])
    recognition = Stack([MedoidRecognizer([Recognizer(active, 'c'), Recognizer(active, 'd')])])
    pipeline = Pipeline(detector=detectors, layout=None, recognizer=recognition,
                        corrector=None, memory_policy='sequential')
    image = np.zeros((20,20,3), np.uint8)
    for _ in range(2):
        result = pipeline.predict(image)
        assert result['page'].get_text() == 'answer answer'
        assert not active
        assert pipeline.last_detection_page is not None
    assert detectors.memory_policy == recognition.memory_policy == 'keep'
    # Policy propagation is scoped to a call, not a permanent mode switch.
    models = recognition.stages[0].models
    Stack([models[0]]).predict(page())
    assert active == {'c'}
    recognition.close()
    assert not active


@pytest.mark.parametrize('kind', ['stack', 'pipeline', 'voting', 'medoid'])
@pytest.mark.parametrize('failure', [ValueError('inference failed'), KeyboardInterrupt()])
def test_errors_release_current_model_and_restore_context(kind, failure):
    active = set()
    model = Recognizer(active, fail=failure)
    if kind == 'stack':
        composition = Stack([model], memory_policy='sequential')
        predict = lambda: composition.predict(page())
    elif kind == 'pipeline':
        composition = Pipeline(detector=Detector(active, 'det'), layout=None, recognizer=model,
                               memory_policy='sequential')
        predict = lambda: composition.predict(np.zeros((20,20,3), np.uint8))
    else:
        cls = MajorityVotingRecognizer if kind == 'voting' else MedoidRecognizer
        composition = cls([model], memory_policy='sequential')
        predict = lambda: composition.predict(page())
    with pytest.raises(type(failure)):
        predict()
    assert not active
    model.fail = None
    Stack([model]).predict(page())
    assert active == {'model'}
    model.close()


@pytest.mark.parametrize('cls', [Stack, MajorityVotingRecognizer, MedoidRecognizer])
def test_missing_close_only_rejected_in_sequential_mode(cls):
    class Custom:
        def predict(self, page):
            return page
    cls([Custom()])
    with pytest.raises(TypeError, match='close'):
        cls([Custom()], memory_policy='sequential')


@pytest.mark.parametrize('policy', [None, True, [], 'auto'])
def test_invalid_policies_fail_before_default_model_resolution(policy, monkeypatch):
    monkeypatch.setattr(BaseArtifactModel, '_resolve_weights', lambda *args: pytest.fail('weights resolved'))
    with pytest.raises(ValueError, match='memory_policy'):
        Pipeline(memory_policy=policy)


def test_parallel_conflicts_are_rejected_in_both_nested_directions():
    a = Detector(set())
    with pytest.raises(ValueError, match='parallel'):
        Stack([a], parallel=True, memory_policy='sequential')
    with pytest.raises(ValueError, match='parallel'):
        Stack([Stack([a], parallel=True)], memory_policy='sequential')
    with pytest.raises(ValueError, match='parallel'):
        Stack([Stack([a], memory_policy='sequential')], parallel=True)
    with pytest.raises(ValueError, match='parallel'):
        MajorityVotingRecognizer([Recognizer(set())], parallel=True, memory_policy='sequential')


def test_close_visits_all_unique_participants_even_after_failure():
    active = {'first','second'}
    first, second = Recognizer(active, 'first'), Recognizer(active, 'second')
    def fail():
        raise RuntimeError('cannot release')
    first.close = fail
    composition = Stack([Stack([first, second]), second])
    with pytest.raises(RuntimeError, match='cannot release'):
        composition.close()
    assert active == {'first'}


def test_release_failure_stops_before_next_model_and_preserves_inference_error(caplog):
    active = set()
    first, second = Recognizer(active, 'first'), Recognizer(active, 'second')
    close = first.close
    def fail_if_resident():
        if 'first' in active:
            raise RuntimeError('cannot release')
        close()
    first.close = fail_if_resident
    composition = Stack([first, second], memory_policy='sequential')
    with pytest.raises(RuntimeError, match='cannot release'):
        composition.predict(page())
    assert second.loads == 0
    active.clear()
    first.fail = ValueError('original failure')
    with pytest.raises(ValueError, match='original failure'):
        composition.predict(page())
    assert 'Failed to close' in caplog.text


def test_context_manager_and_pickle_preserve_default_keep_behavior():
    composition = pickle.loads(pickle.dumps(Stack([Recognizer(set())])))
    model = composition.stages[0]
    with composition:
        composition.predict(page())
        assert model.active == {'model'}
    assert not model.active
    composition.close()


def test_sequential_calls_on_same_composition_do_not_close_active_inference():
    active = set()
    entered, finish = Event(), Event()
    class Blocking(Recognizer):
        def predict(self, page):
            self.run()
            self.busy = True
            entered.set()
            try:
                assert finish.wait(timeout=5)
                return page
            finally:
                self.busy = False
    model = Blocking(active)
    composition = Stack([model], memory_policy='sequential')
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(composition.predict, page())
        assert entered.wait(timeout=5)
        second = executor.submit(composition.predict, page())
        finish.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert not active and model.loads == 2


def test_real_native_onnx_session_is_released_and_recreated(tmp_path):
    from tests.package.wheel_smoke import save_constant_graph
    graph = tmp_path / 'ctc.onnx'
    save_constant_graph(graph, 'images', [1,3,32,64], 'logits', np.array([[[0,10],[10,0]]], np.float32))
    config = PPOCRRec.create_config({'rec_image_shape':[3,32,64],
        'PostProcess': {'character_dict':['a']}, 'postprocess':{'output_is_logits':True}}, recognition_levels=['word'], output_path=tmp_path / 'ctc.json')
    model = PPOCRRec(weights=graph, config=tmp_path / 'ctc.json', device='cpu')
    composition = MedoidRecognizer([model], memory_policy='sequential')
    image = np.zeros((20,20,3), np.uint8)
    for _ in range(2):
        assert composition.predict(page(), image=image).get_text() == 'a'
        assert model.onnx_session is None and model._runtime_sessions == []


def test_trocr_kv_cache_releases_all_three_native_sessions(tmp_path, monkeypatch):
    import json
    import weakref
    onnx = pytest.importorskip('onnx')
    tokenizers = pytest.importorskip('tokenizers')
    Tokenizer, models = tokenizers.Tokenizer, tokenizers.models
    from manuscript.recognizers import TrOCR
    from tests.package.wheel_smoke import save_constant_graph

    info = onnx.helper.make_tensor_value_info
    save_constant_graph(tmp_path / 'encoder.onnx', 'pixels', [1,3,8,16],
                        'hidden', np.zeros((1,1,2), np.float32))
    weights = np.full((5,5), -20, np.float32)
    weights[1,3] = weights[3,2] = 20
    for cached, name in [(False, 'decoder'), (True, 'cached')]:
        nodes = [onnx.helper.make_node('Gather', ['weights','input_ids'], ['logits'], axis=0)]
        inputs = [info('input_ids', onnx.TensorProto.INT64, [1,'T']),
                  info('encoder_hidden_states', onnx.TensorProto.FLOAT, [1,1,2])]
        if cached:
            inputs.append(info('past', onnx.TensorProto.FLOAT, [1,1,1,2]))
            nodes.append(onnx.helper.make_node('Identity', ['past'], ['present']))
        else:
            nodes.append(onnx.helper.make_node('Constant', [], ['present'],
                value=onnx.numpy_helper.from_array(np.zeros((1,1,1,2), np.float32))))
        graph = onnx.helper.make_graph(nodes, name, inputs,
            [info('logits', onnx.TensorProto.FLOAT, [1,'T',5]),
             info('present', onnx.TensorProto.FLOAT, [1,1,1,2])],
            [onnx.numpy_helper.from_array(weights, 'weights')])
        onnx.save(onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid('',17)],
                                        ir_version=9), tmp_path / (name + '.onnx'))
    tokenizer = Tokenizer(models.WordLevel({'<pad>':0,'<s>':1,'</s>':2,'text':3,'<unk>':4}, unk_token='<unk>'))
    tokenizer.add_special_tokens(['<pad>','<s>','</s>'])
    tokenizer.save(str(tmp_path / 'tokenizer.json'))
    config = {'schema_version':1, 'recognition_levels':['word'], 'algorithm':'autoregressive',
              'decoder':'decoder.onnx', 'decoder_with_past':'cached.onnx',
              'decoder_format':'kv_cache', 'cache_mapping':[{'input':'past','output':'present'}],
              'tokenizer':'tokenizer.json', 'preprocess':{'height':8,'width':16,'mean':[0,0,0],'std':[1,1,1]},
              'generation':{'decoder_start_token_id':1,'eos_token_id':2,'num_beams':1,'max_length':8}}
    (tmp_path / 'encoder.json').write_text(json.dumps(config))
    model = TrOCR(tmp_path / 'encoder.onnx', config=tmp_path / 'encoder.json', device='cpu')
    references = []
    create = model._create_onnx_session
    def track(*args, **kwargs):
        native = create(*args, **kwargs)
        references.append(weakref.ref(native))
        return native
    monkeypatch.setattr(model, '_create_onnx_session', track)
    ensemble = MajorityVotingRecognizer([model], memory_policy='sequential')
    for count in (3,6):
        assert ensemble.predict(page(), image=np.zeros((20,20,3), np.uint8)).get_text() == 'text'
        assert len(references) == count
        assert all(ref() is None for ref in references)
        assert model.onnx_session is model.decoder_session is model.decoder_with_past_session is None
        assert model._runtime_sessions == []
