import os
from pathlib import Path
import shutil
import subprocess
import sys
from tarfile import open as open_tar
from zipfile import ZipFile

import pytest


def test_wheel_excludes_caches_and_runs_installed_pipeline(tmp_path):
    pytest.importorskip("build")
    pytest.importorskip("onnx")
    root = Path(__file__).resolve().parents[2]
    project = tmp_path / "project"
    project.mkdir()
    for name in ("pyproject.toml", "MANIFEST.in", "README.md", "LICENSE", "registry.json"):
        shutil.copy2(root / name, project / name)
    shutil.copytree(root / "src/manuscript", project / "src/manuscript")
    cache = project / "src/manuscript/__pycache__"
    cache.mkdir(exist_ok=True)
    for name in ("local.pyc", "local.nbc", "local.nbi"):
        (cache / name).write_bytes(b"cache")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--sdist",
            "--no-isolation",
            str(project),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with open_tar(next((project / "dist").glob("*.tar.gz"))) as archive:
        assert not any(name.endswith("/registry.json") for name in archive.getnames())
    wheel = next((project / "dist").glob("*.whl"))
    with ZipFile(wheel) as archive:
        names = archive.namelist()
        assert not any(
            "__pycache__" in name or name.endswith((".pyc", ".nbc", ".nbi"))
            for name in names
        )
        assert not any(name.endswith("registry.json") for name in names)
        assert "manuscript/recognizers/_trba/configs/charset.txt" in names
    target = tmp_path / "installed"
    with ZipFile(wheel) as archive:
        archive.extractall(target)
    source = "import runpy, sys; sys.path.insert(0, sys.argv[1]); runpy.run_path(sys.argv[2], run_name='__main__')"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            source,
            str(target),
            str(root / "tests/package/wheel_smoke.py"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": ""},
    )
    assert result.returncode == 0, result.stdout + result.stderr
