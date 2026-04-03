#!/usr/bin/env python3
"""
Скрипт запуска Ray Tune для OCR-модели.

Использование:
    # Быстрый ASHA поиск (8 trial-ов, 2 trial на GPU)
    python run_tune.py config.json

    # PBT (Population Based Training) — мутация гиперпараметров на лету
    python run_tune.py config.json --mode pbt

    # Тумблер — решает по use_ray_tune в конфиге
    python run_tune.py config.json --auto

    # Кастомные параметры
    python run_tune.py config.json --mode asha --num-samples 16 --gpus 0.25 --epochs 30

    # Продолжить прерванный эксперимент
    python run_tune.py config.json --resume --storage ~/ray_results

    # Извлечь лучшую модель после tune
    python run_tune.py config.json --extract-best /path/to/output_dir
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Ray Tune hyperparameter search для OCR-модели TRBA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # ASHA: 8 trial-ов, 0.5 GPU каждый (= 2 параллельно на 1 GPU)
  python run_tune.py tune_config_example.json

  # PBT: 4 trial-а, мутация lr/ctc_weight на лету
  python run_tune.py tune_config_example.json --mode pbt --num-samples 4

  # Использовать тумблер из конфига (use_ray_tune: true/false)
  python run_tune.py tune_config_example.json --auto

  # Продолжить прерванный эксперимент
  python run_tune.py tune_config_example.json --resume
""",
    )

    parser.add_argument("config", help="Путь к JSON-конфигу обучения")
    parser.add_argument(
        "--mode",
        choices=["asha", "pbt", "auto"],
        default=None,
        help="Режим: asha (быстрый поиск), pbt (мутация на лету), auto (по конфигу). "
             "Default: берётся из конфига или asha",
    )
    parser.add_argument("--auto", action="store_true",
                        help="Использовать maybe_run_tune (тумблер use_ray_tune в конфиге)")
    parser.add_argument("--num-samples", "-n", type=int, default=None,
                        help="Количество trial-ов (default: из конфига или 8)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Макс. эпох на trial (default: из конфига)")
    parser.add_argument("--gpus", type=float, default=None,
                        help="GPU на trial (0.5 = 2 trial-а на 1 GPU)")
    parser.add_argument("--cpus", type=int, default=None,
                        help="CPU на trial (default: 2)")
    parser.add_argument("--storage", type=str, default=None,
                        help="Путь для хранения результатов Ray Tune")
    parser.add_argument("--name", type=str, default=None,
                        help="Имя эксперимента")
    parser.add_argument("--resume", action="store_true",
                        help="Продолжить прерванный эксперимент")
    parser.add_argument("--extract-best", type=str, default=None, metavar="DIR",
                        help="Извлечь лучшую модель в указанную директорию")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Устройство для обычного обучения (--auto режим)")

    args = parser.parse_args()

    if not os.path.isfile(args.config):
        print(f"Ошибка: конфиг не найден: {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # ── Автоматический режим (тумблер) ────────────────────────────────────────
    if args.auto:
        from manuscript.recognizers._trba.training.tune_training import maybe_run_tune
        result = maybe_run_tune(cfg, device=args.device)
        print("\nГотово!")
        if isinstance(result, dict):
            print(f"  val_acc: {result.get('val_acc', 'N/A')}")
            print(f"  val_loss: {result.get('val_loss', 'N/A')}")
        return

    # ── Ray Tune режим ────────────────────────────────────────────────────────
    from manuscript.recognizers._trba.training.tune_training import (
        run_tune,
        extract_best_model,
    )

    mode = args.mode or cfg.get("tune_mode", "asha")
    num_samples = args.num_samples or cfg.get("tune_num_samples", 8)
    max_epochs = args.epochs or cfg.get("tune_max_epochs", cfg.get("epochs", 20))
    gpus_per_trial = args.gpus if args.gpus is not None else cfg.get("tune_gpus_per_trial", 1.0)
    cpus_per_trial = args.cpus if args.cpus is not None else cfg.get("tune_cpus_per_trial", 2)
    storage_path = args.storage or cfg.get("tune_storage_path", None)
    experiment_name = args.name or cfg.get("tune_experiment_name", None)

    print("=" * 60)
    print(f"  Ray Tune OCR Hyperparameter Search")
    print("=" * 60)
    print(f"  Mode:           {mode}")
    print(f"  Num samples:    {num_samples}")
    print(f"  Max epochs:     {max_epochs}")
    print(f"  GPU/trial:      {gpus_per_trial}")
    print(f"  CPU/trial:      {cpus_per_trial}")
    print(f"  Storage:        {storage_path or '~/ray_results'}")
    print(f"  Resume:         {args.resume}")
    print("=" * 60)

    results = run_tune(
        base_config=cfg,
        mode=mode,
        num_samples=num_samples,
        max_epochs=max_epochs,
        gpus_per_trial=gpus_per_trial,
        cpus_per_trial=cpus_per_trial,
        storage_path=storage_path,
        experiment_name=experiment_name,
        resume=args.resume,
    )

    # ── Извлечение лучшей модели ──────────────────────────────────────────────
    if args.extract_best:
        extract_best_model(results, args.extract_best, cfg)
    else:
        print("\nЧтобы извлечь лучшую модель:")
        print(f"  python run_tune.py {args.config} --extract-best ./best_tune_model")


if __name__ == "__main__":
    main()
