import pytest
from manuscript.data import Element


@pytest.fixture(autouse=True)
def isolated_element_registry(monkeypatch):
    """Custom registrations remain local to each test."""
    monkeypatch.setattr(Element, "registry", dict(Element.registry))
