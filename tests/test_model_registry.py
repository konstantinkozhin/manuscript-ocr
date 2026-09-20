import hashlib
import io
import json
from pathlib import Path

import pytest

from manuscript.models import registry as m


def artifact(payload=b'weights', filename='weights.onnx', urls=None):
    return dict(filename=filename, sha256=hashlib.sha256(payload).hexdigest(),
                size=len(payload), urls=urls or ['https://files/weights'])


def catalog(name='demo', **extras):
    return {'schema_version': 1, 'models': {name: {
        'model_version': '1', 'model_classes': ['TRBA'], 'library_version': None,
        'artifacts': {'weights': artifact(), **extras}}}}


def network(monkeypatch, responses):
    calls = []
    def open_url(url, **kwargs):
        calls.append(url)
        value = responses[url]
        if isinstance(value, Exception):
            raise value
        return io.BytesIO(json.dumps(value).encode() if isinstance(value, dict) else value)
    monkeypatch.setattr(m.urllib.request, 'urlopen', open_url)
    return calls


def cached(tmp_path, data=None):
    r = m.Registry(tmp_path, sources=['https://registry'])
    m._atomic_json(r.cache, data or catalog())
    return r


def test_fallback_and_offline_cache(tmp_path, monkeypatch):
    calls = network(monkeypatch, {'https://github': OSError('offline'),
                                 'https://gitverse': catalog(), 'https://files/weights': b'weights'})
    r = m.Registry(tmp_path, sources=['https://github', 'https://gitverse'])
    result = r.resolve('demo', 'TRBA')
    assert calls == ['https://github', 'https://gitverse', 'https://files/weights']
    assert result['weights'] == tmp_path / 'models/demo/1/weights.onnx'
    assert (result['weights'].parent / 'model.json').is_file()
    calls.clear()
    m.Registry(tmp_path, sources=r.sources).resolve('demo', 'TRBA')
    assert calls == []


def test_unknown_key_refreshes_once(tmp_path, monkeypatch):
    r = cached(tmp_path)
    calls = network(monkeypatch, {'https://registry': catalog('new'), 'https://files/weights': b'weights'})
    r.resolve('new', 'TRBA')
    assert calls.count('https://registry') == 1
    with pytest.raises(ValueError, match='Unknown model'):
        r.info('absent')
    assert calls.count('https://registry') == 2


def test_wrong_class_before_download(tmp_path, monkeypatch):
    calls = network(monkeypatch, {})
    with pytest.raises(ValueError, match='not YOLO'):
        cached(tmp_path).resolve('demo', 'YOLO')
    assert not calls


def test_broken_mirror_and_hash_mismatch(tmp_path, monkeypatch):
    data = catalog()
    data['models']['demo']['artifacts']['weights']['urls'] = ['https://bad', 'https://wrong', 'https://good']
    r = cached(tmp_path, data)
    network(monkeypatch, {'https://bad': OSError('404'), 'https://wrong': b'invalid', 'https://good': b'weights'})
    result = r.resolve('demo')
    assert result['weights'].read_bytes() == b'weights'
    assert not list(tmp_path.rglob('*.part'))


def test_refresh_after_all_artifact_mirrors_fail(tmp_path, monkeypatch):
    new = catalog()
    new['models']['demo']['artifacts']['weights']['urls'] = ['https://new']
    calls = network(monkeypatch, {'https://files/weights': OSError('404'), 'https://registry': new, 'https://new': b'weights'})
    cached(tmp_path).resolve('demo')
    assert calls == ['https://files/weights', 'https://registry', 'https://new']


def test_failed_update_preserves_registry_and_old_files(tmp_path, monkeypatch):
    r = cached(tmp_path)
    original = r.cache.read_bytes()
    calls = network(monkeypatch, {'https://files/weights': OSError('offline'), 'https://registry': b'invalid JSON'})
    with pytest.raises(OSError):
        r.resolve('demo')
    assert r.cache.read_bytes() == original
    assert calls.count('https://registry') == 1
    assert not list(tmp_path.rglob('*.part'))


def test_license_and_optional_checkpoint(tmp_path, monkeypatch):
    r = cached(tmp_path, catalog(license=artifact(b'license', 'LICENSE.txt', ['https://license']),
                                 checkpoint={**artifact(filename='weights.pth'), 'required': False}))
    calls = network(monkeypatch, {'https://files/weights': b'weights', 'https://license': b'license'})
    paths = r.resolve('demo')
    assert set(paths) == {'weights', 'license'}
    assert calls == ['https://files/weights', 'https://license']


def test_old_cache_copied_only_when_hash_matches(tmp_path, monkeypatch):
    r = cached(tmp_path)
    old = tmp_path / 'weights/weights.onnx'
    old.parent.mkdir()
    old.write_bytes(b'weights')
    calls = network(monkeypatch, {})
    assert r.resolve('demo')['weights'].read_bytes() == old.read_bytes()
    assert old.exists() and not calls


def test_corrupt_cache_repaired(tmp_path, monkeypatch):
    r = cached(tmp_path)
    calls = network(monkeypatch, {'https://files/weights': b'weights'})
    paths = r.resolve('demo')
    paths['weights'].write_bytes(b'corrupt')
    assert r.resolve('demo')['weights'].read_bytes() == b'weights'
    assert len(calls) == 2


@pytest.mark.parametrize('name', ['../demo', 'CON', 'C:/demo', 'demo/../bad'])
def test_unsafe_names_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        cached(tmp_path).info(name)


def test_version_compatibility(tmp_path, monkeypatch):
    data = catalog()
    data['models']['demo']['library_version'] = '>=99'
    monkeypatch.setattr(m, 'version', lambda _: '0.1.13')
    with pytest.raises(ValueError, match='requires manuscript'):
        cached(tmp_path, data).info('demo')


def test_unknown_version_ignores_mirror_changes(tmp_path):
    r = cached(tmp_path)
    entry = catalog()['models']['demo']
    entry['model_version'] = None
    path = r.directory('demo', entry)
    entry['artifacts']['weights']['urls'] = ['https://new']
    assert r.directory('demo', entry) == path
    entry['artifacts']['weights']['sha256'] = '0' * 64
    assert r.directory('demo', entry) != path


def test_custom_catalog_priority_and_persistence(tmp_path, monkeypatch):
    monkeypatch.setenv('MANUSCRIPT_HOME', str(tmp_path))
    monkeypatch.setattr(m, '_extra_sources', [])
    monkeypatch.setattr(m, 'SOURCES', ('https://official', 'https://mirror'))
    official = catalog()
    custom = catalog('private')
    custom['models']['demo'] = {**official['models']['demo'], 'description': 'custom'}
    network(monkeypatch, {'https://official': official, 'https://custom': custom})
    m.add_registry('https://custom', priority=True, persist=True)
    monkeypatch.setattr(m, '_extra_sources', [])
    r = m.Registry()
    assert r.info('demo')['description'] == 'custom'
    assert 'private' in r.load()['models']


def test_new_alias_integrates_with_base(tmp_path, monkeypatch):
    from manuscript.api.base import BaseArtifactModel
    from manuscript import models
    r = cached(tmp_path)
    monkeypatch.setattr(models, 'resolve', r.resolve)
    network(monkeypatch, {'https://files/weights': b'weights'})
    obj = object.__new__(type('Fake', (BaseArtifactModel,), {
        'registry_model_class': 'TRBA', 'predict': lambda *a: None,
        'train': lambda *a: None, 'export': lambda *a: None,
        '_initialize_session': lambda *a: None}))
    obj.force_download = False
    assert Path(obj._resolve_weights('demo')).read_bytes() == b'weights'
    with pytest.raises(ValueError, match='not EAST'):
        obj.registry_model_class = 'EAST'
        obj._resolve_weights('demo')


def test_bundled_registry_valid():
    data = m._validate(json.loads(Path(m.__file__).with_name('registry.json').read_text()))
    assert len(data['models']) == 10
    assert sum(a is not None for e in data['models'].values() for a in e['artifacts'].values()) == 31


@pytest.mark.parametrize('class_name', ['TRBA', 'YOLO', 'EAST', 'CharLM', 'PPOCRv5Rec'])
def test_public_classes_resolve_new_names_and_companions(tmp_path, monkeypatch, class_name):
    from manuscript import models
    from manuscript.recognizers import TRBA, PPOCRv5Rec
    from manuscript.detectors import YOLO, EAST
    from manuscript.correctors import CharLM

    classes = {c.__name__: c for c in (TRBA, YOLO, EAST, CharLM, PPOCRv5Rec)}
    config = b'{"max_len":25,"hidden_size":256,"img_h":64,"img_w":256,"imgsz":1280}'
    charset = b'<PAD>\n<SOS>\n<EOS>\na\nb\nc'
    vocab = b'["a","b","c"]'
    data = catalog(config=artifact(config, 'config.json', ['https://config']),
                   charset=artifact(charset, 'charset.txt', ['https://charset']),
                   vocab=artifact(vocab, 'vocab.json', ['https://vocab']))
    data['models']['demo']['model_classes'] = [class_name]
    r = cached(tmp_path, data)
    monkeypatch.setattr(models, 'resolve', r.resolve)
    monkeypatch.setattr(models, 'info', r.info)
    network(monkeypatch, {'https://files/weights': b'weights', 'https://config': config,
                          'https://charset': charset, 'https://vocab': vocab})
    if class_name == 'PPOCRv5Rec':
        # PPOCR's format-specific startup is tested separately; exercise its resolver here.
        instance = object.__new__(PPOCRv5Rec)
        instance.force_download = False
        assert Path(instance._resolve_weights('demo')).read_bytes() == b'weights'
    else:
        instance = classes[class_name](weights='demo', device='cpu')
        assert Path(instance.weights).parent == tmp_path / 'models/demo/1'
        if class_name == 'TRBA':
            assert Path(instance.config_path).name == 'config.json'
            assert Path(instance.charset_path).name == 'charset.txt'
        if class_name == 'CharLM':
            assert instance.c2i == {'a': 0, 'b': 1, 'c': 2}


def test_local_management(tmp_path, monkeypatch):
    monkeypatch.setenv('MANUSCRIPT_HOME', str(tmp_path))
    r = cached(tmp_path)
    network(monkeypatch, {'https://files/weights': b'weights'})
    r.resolve('demo')
    assert m.is_installed('demo')
    assert m.verify('demo') == {'weights': True}
    assert m.list_installed()[0]['id'] == 'demo'
    m.remove('demo')
    assert not m.is_installed('demo')
