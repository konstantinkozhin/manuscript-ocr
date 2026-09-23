"""Small, on-demand registry with ordered mirrors and verified artifact caching."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import urllib.request
import warnings
from importlib.metadata import version

from packaging.version import Version

SOURCES = (
    'https://raw.githubusercontent.com/konstantinkozhin/manuscript-ocr/main/registry.json',
    'https://gitverse.ru/api/repos/konstantin_kozhin/manuscript-ocr/raw/branch/main/registry.json',
)
_extra_sources = []


class DownloadError(OSError):
    """All mirrors of an artifact failed."""


def _segment(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value) or value.endswith('.'):
        raise ValueError(f'Invalid registry path component: {value!r}')
    if value.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}:
        raise ValueError(f'Reserved filename: {value!r}')
    return value


def _url(value):
    if not isinstance(value, str) or not value.startswith(('https://', 'http://')):
        raise ValueError('Registry sources and mirrors must be HTTP(S) URLs')
    return value


def _validate(data):
    if not isinstance(data, dict) or data.get('schema_version') != 1 or not isinstance(data.get('models'), dict):
        raise ValueError('Unsupported or malformed registry schema')
    for name, entry in data['models'].items():
        _segment(name)
        if not isinstance(entry, dict) or not isinstance(entry.get('model_classes'), list) or not entry['model_classes']:
            raise ValueError(f'Missing model_classes for {name}')
        if not all(isinstance(c, str) for c in entry['model_classes']):
            raise ValueError(f'Invalid model_classes for {name}')
        if entry.get('library_version') is not None:
            Version(entry['library_version'])
        if entry.get('library_version_min') is not None:
            Version(entry['library_version_min'])
        if entry.get('library_version_max') is not None:
            Version(entry['library_version_max'])
        if not isinstance(entry.get('artifacts'), dict):
            raise ValueError(f'Missing artifacts for {name}')
        filenames = set()
        for role, artifact in entry['artifacts'].items():
            _segment(role)
            if artifact is None:
                continue
            filename = _segment(artifact['filename'])
            if filename.casefold() in filenames or filename.casefold() == 'model.json':
                raise ValueError(f'Duplicate or reserved artifact filename: {filename}')
            filenames.add(filename.casefold())
            digest = artifact.get('sha256')
            if digest is not None and not re.fullmatch('[0-9a-f]{64}', digest):
                raise ValueError(f'Invalid SHA-256 for {name}/{role}')
            size = artifact.get('size')
            if size is not None and (type(size) is not int or size < 0):
                raise ValueError(f'Invalid size for {name}/{role}')
            if not isinstance(artifact.get('urls'), list):
                raise ValueError(f'Missing URLs for {name}/{role}')
            for url in artifact['urls']:
                _url(url)
    return data


def _atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix='.part')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _matches(path, artifact):
    if not path.is_file():
        return False
    if artifact.get('size') is not None and path.stat().st_size != artifact['size']:
        return False
    if artifact.get('sha256'):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != artifact['sha256']:
            return False
    return True


class Registry:
    def __init__(self, root=None, sources=None, timeout=10):
        self.root = Path(root or os.environ.get('MANUSCRIPT_HOME', Path.home() / '.manuscript')).expanduser()
        self.cache = self.root / 'registries' / 'registry.json'
        self.timeout = timeout
        self.groups = [list(sources)] if sources is not None else self._source_groups()
        self.sources = [url for group in self.groups for url in group]
        self._data = None

    def _source_groups(self):
        path = self.root / 'registries' / 'sources.json'
        saved = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
        entries = _extra_sources + saved
        unique = {x['url']: x for x in reversed(entries)}
        return ([[x['url']] for x in unique.values() if x['priority']] +
                [list(SOURCES)] + [[x['url']] for x in unique.values() if not x['priority']])

    def refresh(self):
        errors = []
        catalogs = []
        fresh = False
        for group in self.groups:
            group_cache = self.cache.parent / (hashlib.sha256(json.dumps(group).encode()).hexdigest() + '.json')
            for url in group:
                try:
                    with urllib.request.urlopen(_url(url), timeout=self.timeout) as response:
                        raw = response.read(8 * 1024 * 1024 + 1)
                    if len(raw) > 8 * 1024 * 1024:
                        raise ValueError('Registry exceeds 8 MiB')
                    data = _validate(json.loads(raw))
                    _atomic_json(group_cache, data)
                    catalogs.append(data)
                    fresh = True
                    break
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append(f'{url}: {exc}')
            else:
                try:
                    catalogs.append(_validate(json.loads(group_cache.read_text(encoding='utf-8'))))
                except (OSError, ValueError, KeyError, TypeError):
                    if group == list(SOURCES):
                        catalogs.append(_validate(json.loads(Path(__file__).with_name('registry.json').read_text(encoding='utf-8'))))
        if not fresh:
            raise OSError('Cannot update model registry: ' + '; '.join(errors))
        data = {'schema_version': 1, 'models': {}, '_sources': self.sources}
        for catalog in catalogs:
            for name, entry in catalog['models'].items():
                data['models'].setdefault(name, entry)
        _atomic_json(self.cache, data)
        self._data = data
        return data

    def load(self):
        if self._data is not None:
            return self._data
        try:
            self._data = _validate(json.loads(self.cache.read_text(encoding='utf-8')))
            if self._data.get('_sources', self.sources) != self.sources:
                try:
                    return self.refresh()
                except OSError:
                    pass
        except (OSError, ValueError, KeyError, TypeError):
            try:
                return self.refresh()
            except OSError:
                self._data = _validate(json.loads(Path(__file__).with_name('registry.json').read_text(encoding='utf-8')))
                self._data['_sources'] = self.sources
                _atomic_json(self.cache, self._data)
        return self._data

    def info(self, name, model_class=None, *, update_missing=True):
        _segment(name)
        data = self.load()
        if name not in data['models'] and update_missing:
            try:
                data = self.refresh()
            except OSError as exc:
                raise ValueError(f"Unknown model {name!r}; registry update failed: {exc}") from exc
        if name not in data['models']:
            raise ValueError(f'Unknown model {name!r}')
        entry = data['models'][name]
        if model_class and model_class not in entry['model_classes']:
            raise ValueError(f"Model {name!r} belongs to {entry['model_classes']}, not {model_class}")
        installed = Version(version('manuscript-ocr'))
        minimum = entry.get('library_version_min')
        maximum = entry.get('library_version_max')
        required = entry.get('library_version')
        outside_range = ((minimum and installed < Version(minimum)) or
                         (maximum and installed > Version(maximum)))
        exact_mismatch = required and installed != Version(required)
        if outside_range or exact_mismatch:
            if minimum or maximum:
                expected = f"{minimum or '*'}..{maximum or '*'}"
            else:
                expected = f'=={required}'
            warnings.warn(
                f"Model {name!r} was tested with manuscript-ocr {expected}, "
                f"but {installed} is installed; attempting to run it anyway",
                RuntimeWarning,
                stacklevel=2,
            )
        return entry

    def directory(self, name, entry):
        path = self.root / 'models' / _segment(name)
        try:
            path.resolve().relative_to(self.root.resolve())
        except ValueError:
            raise ValueError('Model directory escapes cache root')
        return path

    def _artifact(self, directory, artifact, force=False):
        path = directory / artifact['filename']
        if path.is_symlink() or path.name.casefold() == 'model.json':
            raise ValueError('Artifact path must not be a symlink')
        if not force and _matches(path, artifact):
            return path
        directory.mkdir(parents=True, exist_ok=True)
        legacy = self.root / 'weights' / artifact['filename']
        mirrors = artifact['urls']
        # Old flat cache files are reused only when their identity is verifiable.
        reuse = not force and artifact.get('sha256') and _matches(legacy, artifact)
        errors = []
        for url in ([None] if reuse else mirrors):
            fd, tmp = tempfile.mkstemp(dir=directory, suffix='.part')
            try:
                with os.fdopen(fd, 'wb') as output:
                    if reuse:
                        with legacy.open('rb') as source:
                            shutil.copyfileobj(source, output)
                    else:
                        with urllib.request.urlopen(url, timeout=self.timeout) as source:
                            shutil.copyfileobj(source, output, length=1024 * 1024)
                if not _matches(Path(tmp), artifact):
                    raise ValueError('Downloaded artifact failed size/SHA-256 verification')
                os.replace(tmp, path)
                return path
            except (OSError, ValueError) as exc:
                errors.append(f'{url}: {exc}')
            finally:
                Path(tmp).unlink(missing_ok=True)
        raise DownloadError(f"Cannot download {artifact['filename']}: " + '; '.join(errors))

    def resolve(self, name, model_class=None, artifact=None, force_download=False):
        entry = self.info(name, model_class)
        for attempt in range(2):
            directory = self.directory(name, entry)
            roles = [artifact] if artifact else [k for k, a in entry['artifacts'].items() if a is not None and a.get('required', True)]
            if entry['artifacts'].get('license') and 'license' not in roles:
                roles.append('license')
            try:
                paths = {}
                for role in roles:
                    spec = entry['artifacts'].get(role)
                    if spec is None:
                        raise ValueError(f'Model {name!r} has no {role!r} artifact')
                    paths[role] = self._artifact(directory, spec, force_download)
                _atomic_json(directory / 'model.json', {'id': name, **entry})
                return paths
            except DownloadError as exc:
                if attempt:
                    raise
                try:
                    self.refresh()
                except OSError as update_error:
                    raise DownloadError(f'{exc}; registry update also failed: {update_error}') from exc
                entry = self.info(name, model_class, update_missing=False)


def add_registry(url, priority=False, persist=False):
    """Add an extra catalog; priority=True lets its entries override official ones."""
    item = {'url': _url(url), 'priority': bool(priority)}
    _extra_sources[:] = [x for x in _extra_sources if x['url'] != url]
    _extra_sources.insert(0, item)
    if persist:
        path = Registry().root / 'registries' / 'sources.json'
        saved = json.loads(path.read_text(encoding='utf-8')) if path.exists() else []
        _atomic_json(path, [item] + [x for x in saved if x['url'] != url])


def info(name, model_class=None):
    return Registry().info(name, model_class)


def refresh():
    return Registry().refresh()


def resolve(name, model_class=None, artifact=None, force_download=False):
    return Registry().resolve(name, model_class, artifact, force_download)


download = resolve


def path(name):
    """Return the model directory without downloading artifacts."""
    registry = Registry()
    return registry.directory(name, registry.info(name))


def list_installed():
    """List local model manifests, including partially downloaded bundles."""
    entries = []
    for manifest in (Registry().root / 'models').glob('*/model.json'):
        try:
            entry = json.loads(manifest.read_text(encoding='utf-8'))
            _validate({'schema_version': 1, 'models': {entry['id']: entry}})
            entries.append({**entry, 'directory': str(manifest.parent)})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return entries


def verify(name):
    """Verify required and already downloaded optional files against the registry."""
    registry = Registry()
    entry = registry.info(name)
    directory = registry.directory(name, entry)
    return {role: _matches(directory / spec['filename'], spec)
            for role, spec in entry['artifacts'].items() if spec is not None and
            (spec.get('required', True) or role == 'license' or (directory / spec['filename']).exists())}


def is_installed(name):
    results = verify(name)
    return bool(results) and all(results.values())


def remove(name):
    """Remove the bundle identified by its registry key."""
    directory = path(name)
    if directory.is_symlink():
        raise ValueError('Model directory must not be a symlink')
    if directory.exists():
        shutil.rmtree(directory)
