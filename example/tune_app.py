#!/usr/bin/env python3
"""
Пример запуска Ray Tune для OCR.

Три способа использования:

1) Через тумблер в config.json (use_ray_tune: true):
     python tune_app.py --config config.json

2) Напрямую через run_tune():
     python tune_app.py --config config.json --mode asha --trials 8

3) Извлечение лучшей модели после tune:
     python tune_app.py --extract ~/ray_results/ocr_tune_asha --config config.json --output best_model/
"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="Ray Tune OCR Hyperparameter Search")
    parser.add_argument("--config", required=True, help="Path to config.json")
    parser.add_argument(
        "--mode", default=None, choices=["asha", "pbt"],
        help="Tune mode. Overrides config. (default: from config or 'asha')"
    )
    parser.add_argument("--trials", type=int, default=None, help="Number of trials")
    parser.add_argument("--epochs", type=int, default=None, help="Max epochs per trial")
    parser.add_argument("--gpus", type=float, default=None, help="GPUs per trial (e.g. 0.5)")
    parser.add_argument(
        "--extract", default=None,
        help="Path to Ray results dir to extract best model from"
    )
    parser.add_argument("--output", default="best_tune_model", help="Output dir for extracted model")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted experiment")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Extract mode: just extract the best model from existing results
    if args.extract:
        from manuscript.recognizers._trba.training.tune_training import extract_best_model
        # Load results from directory
        try:
            from ray import tune
            tuner = tune.Tuner.restore(args.extract, trainable=lambda config: None)
            results = tuner.get_results()
        except Exception as e:
            print(f"Error loading results from {args.extract}: {e}")
            print("Make sure the experiment directory exists and ray[tune] is installed.")
            sys.exit(1)

        extract_best_model(results, output_dir=args.output, base_config=cfg)
        return

    # Apply CLI overrides
    if args.mode:
        cfg["tune_mode"] = args.mode
    if args.trials:
        cfg["tune_num_samples"] = args.trials
    if args.epochs:
        cfg["tune_max_epochs"] = args.epochs
        cfg["epochs"] = args.epochs
    if args.gpus is not None:
        cfg["tune_gpus_per_trial"] = args.gpus

    # Если use_ray_tune не задан, включаем (раз скрипт вызван)
    cfg.setdefault("use_ray_tune", True)

    # Запуск через тумблер
    from manuscript.recognizers._trba.training.tune_training import maybe_run_tune

    results = maybe_run_tune(cfg)

    # Если это были tune результаты — предлагаем извлечь модель
    if hasattr(results, "get_best_result"):
        print("\nTo extract the best model, run:")
        print(f"  python {sys.argv[0]} --extract ~/ray_results/ocr_tune_{cfg.get('tune_mode', 'asha')} "
              f"--config {args.config} --output best_tune_model/")


if __name__ == "__main__":
    main()
