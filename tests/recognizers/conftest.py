"""Unit tests use local recognizer artifacts or explicitly mocked registry calls."""
import pytest
import manuscript.models


@pytest.fixture(autouse=True)
def forbid_registry_downloads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Provide local artifacts or mock the recognizer registry")
    monkeypatch.setattr(manuscript.models, "resolve", forbidden)
