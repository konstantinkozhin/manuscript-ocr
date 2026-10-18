import importlib.util
import os
import pytest

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")


def pytest_addoption(parser):
    parser.addoption("--require-integration-deps", action="store_true")


def pytest_sessionstart(session):
    if session.config.getoption("--require-integration-deps"):
        required = (
            "torch",
            "torchvision",
            "onnx",
            "onnxsim",
            "transformers",
            "tokenizers",
            "albumentations",
            "build",
        )
        missing = [name for name in required if importlib.util.find_spec(name) is None]
        if missing:
            raise pytest.UsageError(
                "Missing integration dependencies: " + ", ".join(missing)
            )


def pytest_sessionfinish(session, exitstatus):
    if session.config.getoption("--require-integration-deps"):
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is None:
            return
        unexpected = [
            report.nodeid
            for report in reporter.stats.get("skipped", [])
            if "CUDA" not in str(report.longrepr)
        ]
        if unexpected:
            reporter.write_sep(
                "=", "Unexpected integration skips: " + ", ".join(unexpected)
            )
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
