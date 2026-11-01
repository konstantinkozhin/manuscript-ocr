"""Corrector unit tests use local artifacts and explicitly mocked presets."""
import pytest
import manuscript.models


@pytest.fixture(autouse=True)
def forbid_registry_downloads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Mock the registry or provide local CharLM artifacts")
    monkeypatch.setattr(manuscript.models, "resolve", forbidden)
