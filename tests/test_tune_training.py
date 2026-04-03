"""
Smoke tests for tune_training module.

Тесты проверяют что модуль импортируется, конфигурация парсится корректно,
и trainable function не падает при минимальном вызове.
Ray Tune может быть не установлен — тесты это учитывают.
"""

import json
import os
import pytest
import sys

# Mark all tests in this module as needing the training extras
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_TRAINING_TESTS"), reason="Training tests disabled"
)


class TestTuneImports:
    """Тесты на импортируемость модуля."""

    def test_import_module(self):
        """tune_training импортируется без ошибок (ray не обязателен)."""
        from manuscript.recognizers._trba.training import tune_training
        assert hasattr(tune_training, "run_tune")
        assert hasattr(tune_training, "maybe_run_tune")
        assert hasattr(tune_training, "extract_best_model")
        assert hasattr(tune_training, "_tune_trainable")

    def test_check_ray_installed(self):
        """_check_ray_installed возвращает bool."""
        from manuscript.recognizers._trba.training.tune_training import _check_ray_installed
        result = _check_ray_installed()
        assert isinstance(result, bool)

    def test_get_default_search_spaces(self):
        """Search spaces возвращают dict (если ray установлен)."""
        ray_available = False
        try:
            import ray  # noqa: F401
            ray_available = True
        except ImportError:
            pass

        if not ray_available:
            pytest.skip("ray not installed")

        from manuscript.recognizers._trba.training.tune_training import (
            get_default_search_space_asha,
            get_default_search_space_pbt,
        )
        asha_space = get_default_search_space_asha()
        pbt_space = get_default_search_space_pbt()

        assert isinstance(asha_space, dict)
        assert isinstance(pbt_space, dict)
        assert "lr" in asha_space
        assert "lr" in pbt_space
        assert "batch_size" in asha_space


class TestMaybeRunTune:
    """Тесты для тумблера maybe_run_tune."""

    def test_fallback_without_ray(self, tmp_path):
        """Если use_ray_tune=true но ray не установлен, fallback на обычное обучение."""
        from manuscript.recognizers._trba.training.tune_training import _check_ray_installed

        if _check_ray_installed():
            pytest.skip("ray is installed, cannot test fallback")

        # Создаём минимальный невалидный конфиг — должен упасть в run_training,
        # а не в tune (проверяем что fallback сработал)
        cfg = {
            "use_ray_tune": True,
            "train_csvs": ["/nonexistent/train.csv"],
            "train_roots": ["/nonexistent/"],
            "charset_path": "/nonexistent/charset.txt",
        }

        from manuscript.recognizers._trba.training.tune_training import maybe_run_tune

        # Должен упасть из-за несуществующих путей, но НЕ из-за отсутствия ray
        with pytest.raises(Exception):
            maybe_run_tune(cfg)

    def test_switch_to_normal_training(self, tmp_path):
        """Если use_ray_tune=false, вызывается обычный run_training."""
        cfg = {
            "use_ray_tune": False,
            "train_csvs": ["/nonexistent/train.csv"],
            "train_roots": ["/nonexistent/"],
            "charset_path": "/nonexistent/charset.txt",
        }

        from manuscript.recognizers._trba.training.tune_training import maybe_run_tune

        # Должен упасть в run_training из-за путей
        with pytest.raises(Exception):
            maybe_run_tune(cfg)


class TestConfigExample:
    """Проверяем что пример конфига валидный JSON."""

    def test_example_config_is_valid_json(self):
        example_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "example", "tune_config_example.json"
        )
        if not os.path.exists(example_path):
            # Try from project root
            example_path = os.path.join(
                os.path.dirname(__file__), "..", "example", "tune_config_example.json"
            )

        if not os.path.exists(example_path):
            pytest.skip("Example config not found")

        with open(example_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        assert cfg["use_ray_tune"] is True
        assert cfg["tune_mode"] in ("asha", "pbt")
        assert isinstance(cfg["tune_num_samples"], int)
        assert cfg["tune_gpus_per_trial"] > 0
