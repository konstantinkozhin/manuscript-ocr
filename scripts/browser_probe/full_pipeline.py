"""Browser counterpart of end_to_end_app.py; all processing stays in Python."""
import base64
import gc
import hashlib
import importlib
import inspect
import io
import json
from pathlib import Path
import pkgutil
import shutil
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from manuscript import Element, Pipeline
from manuscript.api.corrector import BaseCorrector
from manuscript.api.detector import BaseDetector
from manuscript.api.recognizer import BaseRecognizer
from manuscript.data import Page
from manuscript.models.registry import DownloadError, Registry, _validate
from manuscript.utils import create_page_from_image, visualize_page


STAGES = {
    'detector': ('manuscript.detectors', BaseDetector),
    'recognizer': ('manuscript.recognizers', BaseRecognizer),
    'corrector': ('manuscript.correctors', BaseCorrector),
}


def discover_models():
    """Find actual library implementations, not a browser support allowlist."""
    result, errors = {}, []
    for stage, (package_name, base) in STAGES.items():
        result[stage] = {}
        package = importlib.import_module(package_name)
        for module in pkgutil.iter_modules(package.__path__, package_name + '.'):
            try:
                loaded = importlib.import_module(module.name)
                for _, cls in inspect.getmembers(loaded, inspect.isclass):
                    key = getattr(cls, 'registry_model_class', None)
                    if key and cls.__module__ == loaded.__name__ and issubclass(cls, base) and not inspect.isabstract(cls):
                        result[stage][key] = cls
            except Exception as exc:
                errors.append(f'{module.name}: {exc}')
    return result, errors


def child_file(root, filename):
    if not isinstance(filename, str) or not filename or '\\' in filename:
        raise ValueError('Некорректное имя файла в model.json')
    path = (root / filename).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError:
        raise ValueError('Файлы модели должны находиться внутри выбранной папки') from None
    if not path.is_file():
        raise FileNotFoundError(f'В папке нет файла {filename}')
    return path


def folder_options(root, stage, implementations):
    """Use the same model.json convention as the native application."""
    root = Path(root).resolve()
    manifest = json.loads(child_file(root, 'model.json').read_text(encoding='utf-8'))
    classes = [name for name in manifest.get('model_classes', []) if name in implementations[stage]]
    if len(classes) != 1:
        raise ValueError('model.json должен указывать одну модель для выбранного этапа')
    cls = implementations[stage][classes[0]]
    artifacts = manifest.get('artifacts')
    if not isinstance(artifacts, dict):
        raise ValueError('В model.json отсутствует раздел artifacts')
    signature = inspect.signature(cls.__init__)
    accepted = set(signature.parameters)
    options = {}
    for role, artifact in artifacts.items():
        if artifact is None:
            continue
        path = root / artifact['filename']
        if artifact.get('required', True) is False and not path.exists():
            continue
        path = child_file(root, artifact['filename'])
        if artifact.get('size') is not None and path.stat().st_size != artifact['size']:
            raise ValueError(f'Размер {path.name} отличается от указанного в model.json')
        if artifact.get('sha256'):
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != artifact['sha256']:
                raise ValueError(f'Контрольная сумма {path.name} не совпадает')
        if role in accepted and role not in {'self', 'kwargs'}:
            options[role] = str(path)
        if role == 'external_data':
            options['web_external_data'] = {artifact['filename']: str(path)}
    if 'weights' not in options:
        raise ValueError('В комплекте нет весов для инференса (артефакт weights)')
    return cls, options


def friendly_error(exc):
    """Explain observed failures without naming unsupported model families."""
    detail = f'{type(exc).__name__}: {exc}'
    if isinstance(exc, DownloadError):
        message = 'Не удалось скачать комплект модели. Можно скачать его отдельно и выбрать папку; браузеру может мешать сеть или CORS.'
    elif isinstance(exc, ImportError):
        message = 'Для этой модели не удалось загрузить нужную зависимость в браузере.'
    elif isinstance(exc, MemoryError) or any(s in detail.lower() for s in ('out of memory', 'allocation failed', 'memory access out of bounds')):
        message = 'Для этой модели сейчас не хватает памяти. Попробуй меньшую модель или освободи память.'
    elif isinstance(exc, (FileNotFoundError, KeyError, json.JSONDecodeError)):
        message = 'В папке не хватает файлов модели или её описание не удалось прочитать.'
    else:
        message = 'Не удалось запустить модель на выбранном устройстве. Можно попробовать CPU или другой комплект.'
    return {'message': message, 'detail': detail}


class BrowserPipelineApp:
    def __init__(self, catalog, runtime='web', registry_root='/tmp/browser-catalog', catalog_url=None):
        self.implementations, self.discovery_errors = discover_models()
        self.catalog = _validate(catalog)
        self.registry = Registry(root=registry_root, sources=[catalog_url] if catalog_url else None, timeout=30)
        self.registry._data = self.catalog
        self.runtime = runtime
        self.items = {}
        self.loaded = {}

    def description(self):
        presets = {stage: [] for stage in STAGES}
        for key, entry in self.catalog['models'].items():
            for stage, implementations in self.implementations.items():
                if any(cls in implementations for cls in entry['model_classes']):
                    presets[stage].append({'id': key, 'label': entry.get('display_name') or key,
                                           'demo': bool(entry.get('demo'))})
        return {'presets': presets, 'implementations': {s: list(v) for s, v in self.implementations.items()},
                'warnings': self.discovery_errors}

    def _instantiate(self, item, device, overrides=None):
        cls, options = item['_class'], item['_options']
        return cls(**{**options, 'runtime': self.runtime, 'device': device, **(overrides or {})})

    @staticmethod
    def _smoke(model, stage):
        # Session creation also catches missing dependencies and unsupported graphs.
        model._initialize_session()
        sample = Image.new('RGB', (320, 80), 'white')
        ImageDraw.Draw(sample).text((12, 24), 'Manuscript example 123', fill='black')
        if stage == 'detector':
            result = model.predict(sample)
            page = result['page'] if isinstance(result, dict) else result
        else:
            levels = getattr(model, 'levels' if stage == 'recognizer' else 'correction_levels', ['word'])
            level = next(iter(levels))
            if stage == 'recognizer' and level == 'line':
                # Dynamic line models must also handle a wide crop, not just a word.
                sample = Image.new('RGB', (960, 96), 'white')
                ImageDraw.Draw(sample).text((12, 24), 'Manuscript line recognition example 123 ' * 3, fill='black')
            page, image = create_page_from_image(sample, region_type=level, return_image=True)
            if stage == 'corrector':
                alphabet = ''.join(c for c in getattr(model, 'c2i', {}) if len(c) == 1 and c.isalpha())
                page.children[0].text = alphabet[:10] or 'example'
            page = model.predict(page, image=image)
            if stage == 'recognizer' and isinstance(page, Page) and not any(
                node.type == level and getattr(node, 'text', None) is not None for node in page.descendants
            ):
                raise RuntimeError('Модель пропустила пробный регион: распознавание не было проверено')
        if not isinstance(page, Page):
            raise TypeError('Модель вернула результат без Page')
        page.validate_tree()

    def validate_folder(self, root, stage, device, label=None, identifier=None, owned=False, remove_on_failure=True, probe=True):
        if stage not in STAGES:
            raise ValueError('Неизвестный этап пайплайна')
        identifier = identifier or str(Path(root).resolve())
        item = {'id': identifier, 'root': str(Path(root).resolve()), 'stage': stage,
                'label': label or Path(root).name, 'owned': owned}
        model = None
        try:
            print(f'Читаем описание {item["label"]}…', flush=True)
            self._release(identifier)
            item['_class'], item['_options'] = folder_options(root, stage, self.implementations)
            if probe:
                model = self._instantiate(item, device)
                self._smoke(model, stage)
            item['class'] = item['_class'].registry_model_class
            item['device'] = device
            item['levels'] = list(getattr(model, 'levels', []))
            item['parameters'] = list(inspect.signature(item['_class'].__init__).parameters)
            self.items[identifier] = item
            if model is not None:
                self.loaded[identifier] = (device, '{}', model)
            print(f'✓ {item["label"]}: модель выбрана', flush=True)
            return {'ok': True, 'model': {k: v for k, v in item.items() if k not in {'root', 'owned'} and not k.startswith('_')}}
        except Exception as exc:
            if model is not None:
                model.close()
            if remove_on_failure:
                self.items.pop(identifier, None)
            gc.collect()
            if owned and remove_on_failure:
                shutil.rmtree(root, ignore_errors=True)
            return {'ok': False, **friendly_error(exc)}

    def validate_preset(self, name, stage, device, probe=True):
        try:
            entry = self.registry.info(name, update_missing=False)
            if not probe:
                implementations = self.implementations[stage]
                classes = [implementations[key] for key in entry['model_classes'] if key in implementations]
                if len(classes) != 1:
                    raise ValueError('Пресет не соответствует выбранному этапу')
                identifier = 'preset:' + name
                self._release(identifier)
                item = {'id': identifier, 'stage': stage, 'root': str(self.registry.directory(name, entry)),
                        'owned': False, 'label': entry.get('display_name') or name,
                        'class': classes[0].registry_model_class, 'device': device,
                        '_class': classes[0], '_options': {}, '_preset': name}
                self.items[identifier] = item
                return {'ok': True, 'model': {k: v for k, v in item.items()
                                            if not k.startswith('_') and k not in ('root', 'owned')}}
            self.registry.resolve(name)
            root = self.registry.directory(name, entry)
            return self.validate_folder(root, stage, device,
                                        label=entry.get('display_name') or name,
                                        identifier='preset:' + name, owned=False)
        except Exception as exc:
            return {'ok': False, **friendly_error(exc)}

    def _release(self, identifier):
        previous = self.loaded.pop(identifier, None)
        if previous:
            previous[2].close()

    def remove(self, identifier):
        self._release(identifier)
        item = self.items.pop(identifier, None)
        if item and item['owned']:
            shutil.rmtree(item['root'], ignore_errors=True)
        gc.collect()

    def recheck(self, device):
        results = []
        for item in list(self.items.values()):
            result = self.validate_folder(item['root'], item['stage'], device,
                                          item['label'], item['id'], item['owned'], remove_on_failure=False)
            results.append({'id': item['id'], 'stage': item['stage'], **result})
        return results

    def _get_model(self, identifier, stage, device, settings):
        item = self.items[identifier]
        if item['stage'] != stage:
            raise ValueError('Модель относится к другому этапу')
        if item.get('_preset') and not item['_options']:
            self.registry.resolve(item['_preset'])
            entry = self.registry.info(item['_preset'], update_missing=False)
            accepted = inspect.signature(item['_class'].__init__).parameters
            for role, spec in entry.get('artifacts', {}).items():
                if spec and spec.get('required', True) is False and role in accepted:
                    self.registry.resolve(item['_preset'], artifact=role)
            item['_class'], item['_options'] = folder_options(item['root'], stage, self.implementations)
        cls = item['_class']
        parameters = inspect.signature(cls.__init__).parameters
        overrides = {}
        if stage == 'detector' and settings.get('detector_override'):
            for parameter in ('score_thresh', 'score_threshold', 'box_threshold'):
                if parameter in parameters:
                    overrides[parameter] = settings['score_thresh']
                    break
            for parameter in ('target_size', 'expand_ratio_w', 'expand_ratio_h'):
                if parameter in parameters:
                    overrides[parameter] = settings[parameter]
        if stage == 'recognizer':
            overrides['region_preparer'] = settings.get('region_preparer', 'bbox')
            if settings.get('recognition_override'):
                levels = settings.get('recognition_levels', [])
                if not levels:
                    raise ValueError('Выбери хотя бы один уровень распознавания')
                overrides['recognition_levels'] = levels
            if settings.get('recognition_batch', 0):
                overrides['batch_size'] = int(settings['recognition_batch'])
        if stage == 'corrector':
            for parameter in ('mask_threshold', 'apply_threshold', 'max_edits'):
                if parameter in parameters:
                    overrides[parameter] = settings[parameter]
        fingerprint = json.dumps(overrides, sort_keys=True)
        previous = self.loaded.get(identifier)
        if previous and previous[:2] == (device, fingerprint):
            return previous[2]
        self._release(identifier)
        model = self._instantiate(item, device, overrides)
        try:
            model._initialize_session()
        except Exception:
            model.close()
            raise
        self.loaded[identifier] = (device, fingerprint, model)
        return model

    def create_pipeline(self, settings, device):
        processors = {}
        for stage in STAGES:
            selected = settings.get(stage, [])
            if len(selected) > 1:
                raise ValueError('Выбери одну модель на этап')
            processors[stage] = self._get_model(selected[0], stage, device, settings) if selected else None
        if processors['detector'] is None:
            raise ValueError('Выбери хотя бы один детектор')
        from manuscript.layouts import ReadingSorting, SimpleSorting
        layouts = []
        selected = settings.get('layout', [])
        if len(selected) > 1:
            raise ValueError('Выбери один алгоритм лейаута')
        for name in selected:
            if name == 'ReadingSorting':
                layouts.append(ReadingSorting(mode=settings.get('layout_mode', 'containment'),
                    infer_columns=settings.get('infer_columns', True),
                    containment_overlap=settings.get('containment_overlap', .5),
                    row_overlap=settings.get('row_overlap', .5)))
            elif name == 'SimpleSorting':
                layouts.append(SimpleSorting(use_columns=settings.get('simple_columns', True),
                                              max_splits=int(settings.get('max_splits', 10))))
            else:
                raise ValueError('Неизвестный лейаут')
        return Pipeline(**processors, layout=layouts[0] if layouts else None)

    def recognize(self, photo_bytes, device, settings):
        started = time.perf_counter()
        with Image.open(io.BytesIO(bytes(photo_bytes))) as source:
            if source.width * source.height > 32_000_000:
                raise ValueError('Выбери изображение не больше 32 мегапикселей')
            image = ImageOps.exif_transpose(source).convert('RGB')
            pixels = np.asarray(image).copy()
        setup_started = time.perf_counter()
        pipeline = self.create_pipeline(settings, device)
        setup_seconds = time.perf_counter() - setup_started
        result = pipeline.predict(pixels, profile=True)
        page = result['page']
        visualization_started = time.perf_counter()
        preview = visualize_page(pixels, page, max_size=1440, upscale=True,
            show_labels=any(node.label or node.type not in Element.registry for node in page.descendants))
        if isinstance(preview, np.ndarray):
            preview = Image.fromarray(preview)
        output = io.BytesIO()
        preview.save(output, format='PNG')
        counts = {}
        for node in page.descendants:
            counts[node.type] = counts.get(node.type, 0) + 1
        words = sum(node.type == 'word' and bool(getattr(node, 'text', None)) for node in page.descendants)
        total = pipeline.last_timings['total']
        models_used = {}
        for stage in STAGES:
            selected = settings.get(stage, [])
            models_used[stage] = [{k: self.items[key][k] for k in ('id', 'label', 'class')} for key in selected]
        layouts = settings.get('layout', [])
        models_used['layout'] = [{'class': name, 'label': name} for name in layouts]
        return {'text': pipeline.get_text(page), 'width': image.width, 'height': image.height,
                'models': models_used, 'device': device,
                'seconds': time.perf_counter() - started, 'setup_seconds': setup_seconds,
                'visualization_seconds': time.perf_counter() - visualization_started,
                'timings': pipeline.last_timings, 'counts': counts,
                'pages_per_second': 1 / total if total else 0, 'words_per_second': words / total if total else 0,
                'preview': 'data:image/png;base64,' + base64.b64encode(output.getvalue()).decode('ascii'),
                'page': page.to_dict()}

    def close(self):
        for identifier in list(self.loaded):
            self._release(identifier)
        gc.collect()
