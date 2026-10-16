"""Run with: env/bin/python -m pytest scripts/browser_probe/test_app.py -q."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('browser_pipeline', Path(__file__).with_name('full_pipeline.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExampleDetector(module.BaseDetector):
    registry_model_class = 'ExampleDetector'
    failure = None
    instances = []

    def __init__(self, weights, device=None, **kwargs):
        super().__init__(weights=weights, device=device, **kwargs)
        self.closed = False
        self.instances.append(self)

    def _initialize_session(self):
        if self.failure == 'dependency':
            raise ImportError('missing_example_dependency')
        if self.failure == self.device:
            raise RuntimeError('backend cannot execute graph')

    def predict(self, image):
        if self.failure == 'prediction':
            raise RuntimeError('unsupported operator on first execution')
        return module.Page(image_width=320, image_height=80)

    def close(self):
        self.closed = True
        super().close()


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path / 'model'
    root.mkdir()
    (root / 'model.onnx').write_bytes(b'test graph')
    (root / 'model.json').write_text(json.dumps({'model_classes': ['ExampleDetector'],
        'artifacts': {'weights': {'filename': 'model.onnx'}}}))
    return root


@pytest.fixture
def app(monkeypatch, tmp_path):
    ExampleDetector.failure = None
    ExampleDetector.instances = []
    implementations = {'detector': {'ExampleDetector': ExampleDetector}, 'recognizer': {}, 'corrector': {}}
    monkeypatch.setattr(module, 'discover_models', lambda: (implementations, []))
    result = module.BrowserPipelineApp({'schema_version': 1, 'models': {}},
                                      runtime='native', registry_root=tmp_path / 'cache')
    yield result
    result.close()


def test_unknown_family_uses_real_probe_instead_of_support_allowlist(app, bundle):
    response = app.validate_folder(bundle, 'detector', 'cpu', owned=True)
    assert response['ok']
    assert response['model']['class'] == 'ExampleDetector'
    assert len(app.loaded) == 1


@pytest.mark.parametrize('failure', ['dependency', 'prediction', 'cpu'])
def test_failure_is_soft_releases_model_and_discards_owned_files(app, bundle, failure):
    ExampleDetector.failure = failure
    response = app.validate_folder(bundle, 'detector', 'cpu', owned=True)
    assert not response['ok']
    assert response['message'] and response['detail']
    assert not app.items and not app.loaded
    assert ExampleDetector.instances[-1].closed
    assert not bundle.exists()


def test_device_recheck_keeps_bundle_for_retry(app, bundle):
    first = app.validate_folder(bundle, 'detector', 'cpu', owned=True)
    identifier = first['model']['id']
    ExampleDetector.failure = 'gpu'
    assert not app.recheck('gpu')[0]['ok']
    assert bundle.is_dir()
    assert identifier in app.items
    assert app.recheck('cpu')[0]['ok']
    assert identifier in app.loaded


def test_remove_releases_sessions_and_owned_files(app, bundle):
    response = app.validate_folder(bundle, 'detector', 'cpu', owned=True)
    model = app.loaded[response['model']['id']][2]
    app.remove(response['model']['id'])
    assert model.closed
    assert not bundle.exists()
    assert not app.items and not app.loaded


def test_preset_failure_keeps_original_files(app, bundle):
    ExampleDetector.failure = 'dependency'
    assert not app.validate_folder(bundle, 'detector', 'cpu', owned=False)['ok']
    assert bundle.is_dir()


def test_companion_cannot_escape_model_directory(app, bundle, tmp_path):
    (tmp_path / 'outside.onnx').write_bytes(b'outside')
    manifest = json.loads((bundle / 'model.json').read_text())
    manifest['artifacts']['weights']['filename'] = '../outside.onnx'
    (bundle / 'model.json').write_text(json.dumps(manifest))
    result = app.validate_folder(bundle, 'detector', 'cpu', owned=True)
    assert not result['ok']
    assert (tmp_path / 'outside.onnx').is_file()
    assert not ExampleDetector.instances


def test_invalid_checksum_fails_before_creating_session(app, bundle):
    manifest = json.loads((bundle / 'model.json').read_text())
    manifest['artifacts']['weights']['sha256'] = '0' * 64
    (bundle / 'model.json').write_text(json.dumps(manifest))
    result = app.validate_folder(bundle, 'detector', 'cpu', owned=True)
    assert not result['ok']
    assert not ExampleDetector.instances


def test_close_keeps_files_and_allows_model_reinitialization(app, bundle):
    response = app.validate_folder(bundle, 'detector', 'cpu', owned=True)
    app.close()
    assert bundle.is_dir()
    assert not app.loaded
    assert app._get_model(response['model']['id'], 'detector', 'cpu', {}) is not None


@pytest.mark.parametrize('skip', [False, True])
def test_recognizer_probe_uses_effective_levels_and_rejects_skipped_inference(skip):
    class LineRecognizer:
        levels = ('line',)

        def _initialize_session(self):
            pass

        def predict(self, page, image=None):
            assert page.children[0].type == 'line'
            assert image.shape[1] >= 8 * image.shape[0]
            if not skip:
                page.children[0].text = ''  # An empty decoded string is still a real result.
            return page

    if skip:
        with pytest.raises(RuntimeError, match='пропустила пробный регион'):
            module.BrowserPipelineApp._smoke(LineRecognizer(), 'recognizer')
    else:
        module.BrowserPipelineApp._smoke(LineRecognizer(), 'recognizer')


def test_download_failure_suggests_local_folder():
    response = module.friendly_error(module.DownloadError('Failed to fetch'))
    assert 'скачать' in response['message']
    assert 'папку' in response['message']


@pytest.mark.parametrize('stage', ['detector', 'recognizer', 'corrector', 'layout'])
def test_pipeline_accepts_only_one_processor_per_stage(app, bundle, stage):
    response = app.validate_folder(bundle, 'detector', 'cpu')
    settings = {'detector': [response['model']['id']]}
    settings[stage] = ['first', 'second']
    with pytest.raises(ValueError, match='Выбери од'):
        app.create_pipeline(settings, 'cpu')


def test_folder_selection_does_not_initialize_or_probe_model(app, bundle):
    ExampleDetector.failure = 'dependency'
    response = app.validate_folder(bundle, 'detector', 'cpu', probe=False)
    assert response['ok']
    assert not ExampleDetector.instances and not app.loaded
    with pytest.raises(ImportError, match='missing_example_dependency'):
        app._get_model(response['model']['id'], 'detector', 'cpu', {})
    assert ExampleDetector.instances[-1].closed


def test_preset_download_is_deferred_until_run(app, bundle, monkeypatch):
    downloads = []
    monkeypatch.setattr(app.registry, 'info', lambda *a, **k: {'model_classes': ['ExampleDetector']})
    monkeypatch.setattr(app.registry, 'directory', lambda *a: bundle)
    monkeypatch.setattr(app.registry, 'resolve', lambda name: downloads.append(name))
    response = app.validate_preset('example', 'detector', 'cpu', probe=False)
    assert response['ok']
    assert not downloads and not ExampleDetector.instances
    app._get_model(response['model']['id'], 'detector', 'cpu', {})
    assert downloads == ['example']


def test_preset_downloads_optional_artifact_used_by_constructor(app, bundle, monkeypatch):
    class WithLexicon(ExampleDetector):
        def __init__(self, weights, lexicon=None, **kwargs):
            super().__init__(weights, **kwargs)
            self.lexicon = lexicon

    manifest = json.loads((bundle / 'model.json').read_text())
    manifest['artifacts']['lexicon'] = {'filename': 'lexicon.txt', 'required': False}
    (bundle / 'model.json').write_text(json.dumps(manifest))
    (bundle / 'lexicon.txt').write_text('example')
    app.implementations['detector']['ExampleDetector'] = WithLexicon
    calls = []
    monkeypatch.setattr(app.registry, 'info', lambda *a, **k: manifest)
    monkeypatch.setattr(app.registry, 'directory', lambda *a: bundle)
    monkeypatch.setattr(app.registry, 'resolve', lambda name, artifact=None: calls.append(artifact))
    response = app.validate_preset('example', 'detector', 'cpu', probe=False)
    model = app._get_model(response['model']['id'], 'detector', 'cpu', {})
    assert calls == [None, 'lexicon']
    assert model.lexicon == str(bundle / 'lexicon.txt')


def test_download_retry_refreshes_host_catalog_instead_of_official_sources(app, tmp_path, monkeypatch):
    import io
    import manuscript.models.registry as registry_module

    catalog_url = 'http://127.0.0.1:8014/catalog.json'
    catalog = {'schema_version': 1, 'models': {'example': {
        'model_classes': ['ExampleDetector'],
        'artifacts': {'weights': {'filename': 'model.onnx',
                                 'urls': ['http://127.0.0.1:8014/_artifacts/example']}}}}}
    browser = module.BrowserPipelineApp(catalog, runtime='native',
                                       registry_root=tmp_path / 'retry', catalog_url=catalog_url)
    requests, attempts = [], []

    def fetch(url, **kwargs):
        requests.append(url)
        return io.BytesIO(json.dumps(catalog).encode())

    def artifact(directory, spec, force=False):
        attempts.append(spec['urls'])
        if len(attempts) == 1:
            raise registry_module.DownloadError('temporary connection failure')
        return tmp_path / 'model.onnx'

    monkeypatch.setattr(registry_module, 'urlopen', fetch)
    monkeypatch.setattr(browser.registry, '_artifact', artifact)
    browser.registry.resolve('example')
    assert requests == [catalog_url]
    assert attempts == [['http://127.0.0.1:8014/_artifacts/example']] * 2
