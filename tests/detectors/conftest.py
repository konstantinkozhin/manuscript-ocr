"""Detector unit tests must supply local or explicitly mocked model artifacts."""
import pytest
import manuscript.models


@pytest.fixture(autouse=True)
def forbid_registry_downloads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Provide local artifacts or mock the detector registry")
    monkeypatch.setattr(manuscript.models, "resolve", forbidden)
