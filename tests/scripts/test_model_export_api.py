import sys
from types import ModuleType

import pytest

from manuscript.detectors._ppocr import PPOCR
from manuscript.detectors._rfdetr import RFDETR
from manuscript.recognizers import PPOCRRec, TrOCR


@pytest.mark.parametrize('model,args,kwargs', [
    (PPOCR, ('source', 'output'), {'model_id': 'example', 'revision': 'revision'}),
    (RFDETR, ('source', 'output', ['image.jpg']), {}),
    (PPOCRRec, ('source', 'output'), {'crops': ['crop.png']}),
    (TrOCR, ('source', 'output'), {}),
])
def test_static_export_passes_options_without_cli_or_model_initialization(monkeypatch, model, args, kwargs):
    assert isinstance(model.__dict__['export'], staticmethod)
    module = ModuleType(model.__module__ + '.export')
    calls = []

    def export(*received, **options):
        calls.append((received, options))
        return 'exported'

    module.export = export
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert model.export(*args, **kwargs) == 'exported'
    positional, options = calls[0]
    assert positional[:2] == args[:2]
    if model is RFDETR:
        assert options['images'] == args[2]
    elif model is TrOCR:
        assert options == {'use_cache': True, 'fp16': True, **kwargs}
    else:
        assert options == kwargs
