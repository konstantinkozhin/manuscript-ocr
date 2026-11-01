"""Keep API tests independent of user files and network access."""
from pathlib import Path
import pytest
import manuscript.api.base as base_module


@pytest.fixture(autouse=True)
def isolated_artifacts(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    original_expanduser = Path.expanduser

    def expanduser(path):
        value = str(path)
        if value == "~" or value.startswith(("~/", "~\\")):
            return home / value[2:] if value != "~" else home
        return original_expanduser(path)

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(Path, "expanduser", expanduser)

    def forbid_network(*args, **kwargs):
        raise AssertionError("API unit tests must mock network access")

    monkeypatch.setattr(base_module.urllib.request, "urlopen", forbid_network)
    monkeypatch.setattr(base_module.urllib.request, "urlretrieve", forbid_network)
