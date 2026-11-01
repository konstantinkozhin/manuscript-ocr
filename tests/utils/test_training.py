import random

import pytest

torch = pytest.importorskip("torch")

from manuscript.utils import training


def test_seed_replays_python_and_torch_generators():
    training.set_seed(42)
    expected = random.random(), torch.rand(3)
    training.set_seed(42)
    assert random.random() == expected[0]
    torch.testing.assert_close(torch.rand(3), expected[1])


def test_seed_reports_missing_torch(monkeypatch):
    monkeypatch.setattr(training, "torch", None)
    with pytest.raises(ImportError, match="PyTorch"):
        training.set_seed(42)
