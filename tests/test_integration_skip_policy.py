"""Optional runtime capabilities must not mask other integration failures."""
from types import SimpleNamespace

import pytest

from tests.conftest import pytest_sessionfinish


@pytest.mark.parametrize('reason,initial,expected', [
    ('ONNX Runtime DLPack export support is required', 0, 0),
    ('ONNX Runtime DLPack export support is required', 1, 1),
    ('could not import onnx', 0, 1),
])
def test_integration_skip_policy(reason, initial, expected):
    messages = []
    reporter = SimpleNamespace(
        stats={'skipped': [SimpleNamespace(longrepr=('test_bridge.py', 1, 'Skipped: ' + reason),
                                          nodeid='test_bridge')]},
        write_sep=lambda *args: messages.append(args),
    )
    session = SimpleNamespace(
        config=SimpleNamespace(
            getoption=lambda option: True,
            pluginmanager=SimpleNamespace(get_plugin=lambda name: reporter),
        ),
        exitstatus=initial,
    )
    pytest_sessionfinish(session, initial)
    assert session.exitstatus == expected
    assert bool(messages) == (reason == 'could not import onnx')
