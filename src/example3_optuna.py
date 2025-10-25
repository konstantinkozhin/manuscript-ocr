import os
import threading
import time
import webbrowser
from typing import Dict, List, Optional

import optuna

try:
    import optuna_dashboard
except ImportError:  # pragma: no cover
    optuna_dashboard = None  # type: ignore

from manuscript.recognizers import TRBAInfer
from manuscript.recognizers.trba.training.metrics import (
    character_error_rate,
    compute_accuracy,
)

# === User-configurable paths & parameters ===
IMAGE_DIR = r"C:\shared\orig_cyrillic\test"
GT_FILE = r"C:\shared\orig_cyrillic\test.tsv"
MODEL_PATH = r"C:\shared\exp1_model_64\best_acc_ckpt.pth"
CONFIG_PATH = r"C:\shared\exp1_model_64\config.json"
CHARSET_PATH: Optional[str] = (
    None  # set to override default charset, otherwise keep None
)
BATCH_SIZE = 16
TRIALS = 100
STUDY_NAME = "trba-decode-search"
DEFAULT_STORAGE_FILE = os.path.join(os.path.dirname(__file__), "optuna_trba.db")
STORAGE: Optional[str] = f"sqlite:///{DEFAULT_STORAGE_FILE}"
SEED = 42

# Dashboard settings (requires optuna-dashboard and persistent storage)
ENABLE_DASHBOARD = True
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8080


def load_ground_truth(gt_path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                fname, text = parts
                mapping[fname.strip()] = text.strip()
    if not mapping:
        raise RuntimeError(f"No ground-truth entries found in {gt_path}")
    return mapping


def collect_images(image_dir: str) -> List[str]:
    valid_ext = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images: List[str] = []
    for fname in os.listdir(image_dir):
        path = os.path.join(image_dir, fname)
        if not os.path.isfile(path):
            continue
        if os.path.splitext(fname)[1].lower() in valid_ext:
            images.append(path)
    images.sort()
    if not images:
        raise RuntimeError(f"No images with valid extensions found in {image_dir}")
    return images


def evaluate_avg_cer(
    recognizer: TRBAInfer,
    images: List[str],
    gt_map: Dict[str, str],
    batch_size: int,
    *,
    mode: str,
    beam_size: int,
    temperature: float,
    length_penalty: float,
    normalize_by_length: bool,
    diverse_groups: int,
    diversity_strength: float,
) -> float:
    predictions = recognizer.predict(
        images=images,
        batch_size=batch_size,
        mode=mode,
        beam_size=beam_size,
        temperature=temperature,
        length_penalty=length_penalty,
        normalize_by_length=normalize_by_length,
        diverse_groups=diverse_groups,
        diversity_strength=diversity_strength,
    )

    total_cer = 0.0
    count = 0
    for path, (pred_text, _) in zip(images, predictions):
        ref = gt_map.get(os.path.basename(path))
        if ref is None:
            continue
        total_cer += character_error_rate(ref, pred_text)
        count += 1

    if count == 0:
        raise RuntimeError("No predictions matched the ground-truth keys.")
    return total_cer / count


def evaluate_metrics(
    recognizer: TRBAInfer,
    images: List[str],
    gt_map: Dict[str, str],
    batch_size: int,
    *,
    mode: str,
    beam_size: int,
    temperature: float,
    length_penalty: float,
    normalize_by_length: bool,
    diverse_groups: int,
    diversity_strength: float,
) -> tuple[float, float]:
    """
    Возвращает (avg_cer, accuracy).
    accuracy - доля точных совпадений, вычисленная через compute_accuracy.
    """
    predictions = recognizer.predict(
        images=images,
        batch_size=batch_size,
        mode=mode,
        beam_size=beam_size,
        temperature=temperature,
        length_penalty=length_penalty,
        normalize_by_length=normalize_by_length,
        diverse_groups=diverse_groups,
        diversity_strength=diversity_strength,
    )

    refs: List[str] = []
    hyps: List[str] = []
    total_cer = 0.0

    for path, (pred_text, _) in zip(images, predictions):
        ref = gt_map.get(os.path.basename(path))
        if ref is None:
            continue
        refs.append(ref)
        hyps.append(pred_text)
        total_cer += character_error_rate(ref, pred_text)

    if len(refs) == 0:
        raise RuntimeError("No predictions matched the ground-truth keys.")

    avg_cer = total_cer / len(refs)
    accuracy = compute_accuracy(refs, hyps)
    return avg_cer, accuracy


def maybe_launch_dashboard(study: optuna.Study) -> None:
    """
    Автоматически запускает optuna-dashboard при старте оптимизации.
    """
    global optuna_dashboard  # ✅ указываем, что используем глобальную переменную

    if optuna_dashboard is None:
        try:
            import optuna_dashboard  # type: ignore
        except ImportError:
            print(
                "[Dashboard] Не установлен optuna-dashboard. "
                "Установи: pip install optuna-dashboard"
            )
            return

    if STORAGE is None:
        print(
            "[Dashboard] STORAGE не задан. "
            "Задай путь, например: STORAGE = 'sqlite:///trba_decode.db'"
        )
        return

    def _run_server():
        try:
            # optuna-dashboard требует storage URL, а не объект study
            optuna_dashboard.run_server(
                STORAGE,
                host=DASHBOARD_HOST,
                port=DASHBOARD_PORT,
            )
        except Exception as e:
            print(f"[Dashboard] Ошибка запуска сервера: {e}")
            import traceback

            traceback.print_exc()

    # 🔥 Автоматический старт Dashboard в отдельном потоке
    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()
    print(f"🚀 Dashboard запускается на http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    print(f"📦 Storage: {STORAGE}")

    # Ждём немного, чтобы сервер успел запуститься
    print("⏳ Ожидание запуска сервера...")
    time.sleep(3)

    # Автоматическое открытие дашборда в браузере
    dashboard_url = f"http://{DASHBOARD_HOST}:{DASHBOARD_PORT}"
    try:
        webbrowser.open(dashboard_url)
        print(f"🌐 Браузер открыт на {dashboard_url}")
    except Exception as e:
        print(f"⚠️ Не удалось открыть браузер: {e}")
        print(f"   Откройте вручную: {dashboard_url}")


def main():
    gt_map = load_ground_truth(GT_FILE)
    images = collect_images(IMAGE_DIR)

    recognizer = TRBAInfer(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        charset_path=CHARSET_PATH,
    )

    baseline_params = {
        "mode": "greedy",
        "beam_size": 1,
        "temperature": 1.0,
        "length_penalty": 0.0,
        "normalize_by_length": True,
        "diverse_groups": 1,
        "diversity_strength": 0.0,
    }
    baseline_cer = evaluate_avg_cer(
        recognizer,
        images,
        gt_map,
        BATCH_SIZE,
        **baseline_params,
    )
    print(f"[Baseline] mode=greedy Avg CER={baseline_cer:.4f}")

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study_kwargs = {
        "study_name": STUDY_NAME,
        "direction": "maximize",
        "sampler": sampler,
    }
    if STORAGE:
        study_kwargs["storage"] = STORAGE
        study_kwargs["load_if_exists"] = True

    study = optuna.create_study(**study_kwargs)
    study.enqueue_trial(baseline_params)
    maybe_launch_dashboard(study)

    def objective(trial: optuna.Trial) -> float:
        mode = trial.suggest_categorical("mode", ["greedy", "beam"])

        if mode == "greedy":
            # Для greedy режима остальные параметры фиксированы
            beam_size = 1
            temperature = 1.0
            length_penalty = 0.0
            normalize_by_length = True
            diverse_groups = 1
            diversity_strength = 0.0
        else:
            # Для beam режима оптимизируем все параметры
            beam_size = trial.suggest_int("beam_size", 2, 12)
            temperature = trial.suggest_float("temperature", 0.7, 2.0)
            length_penalty = trial.suggest_float("length_penalty", 0.0, 2.5)
            normalize_by_length = True
            diverse_groups = trial.suggest_int("diverse_groups", 1, 4)
            diversity_strength = trial.suggest_float("diversity_strength", 0.0, 2.0)

        avg_cer, accuracy = evaluate_metrics(
            recognizer,
            images,
            gt_map,
            BATCH_SIZE,
            mode=mode,
            beam_size=beam_size,
            temperature=temperature,
            length_penalty=length_penalty,
            normalize_by_length=normalize_by_length,
            diverse_groups=diverse_groups,
            diversity_strength=diversity_strength,
        )

        if mode == "greedy":
            print(
                f"[Trial {trial.number}] mode=greedy -> CER={avg_cer:.4f} Acc={accuracy:.4f}"
            )
        else:
            print(
                f"[Trial {trial.number}] mode=beam beam={beam_size} temp={temperature:.2f} "
                f"len_pen={length_penalty:.2f} norm={normalize_by_length} "
                f"groups={diverse_groups} div={diversity_strength:.2f} -> CER={avg_cer:.4f} Acc={accuracy:.4f}"
            )
        return accuracy

    study.optimize(objective, n_trials=TRIALS, show_progress_bar=True)

    best_params = study.best_params
    best_value = study.best_value
    print("\n=== Optuna Results ===")
    print(f"Best accuracy: {best_value:.4f}")
    for key, value in best_params.items():
        print(f"{key}: {value}")

    print("\nRe-evaluating best parameters...")
    best_cer, best_accuracy = evaluate_metrics(
        recognizer,
        images,
        gt_map,
        BATCH_SIZE,
        mode=best_params.get("mode", "beam"),
        beam_size=best_params.get("beam_size", 5),
        temperature=best_params.get("temperature", 1.0),
        length_penalty=best_params.get("length_penalty", 0.0),
        normalize_by_length=best_params.get("normalize_by_length", True),
        diverse_groups=best_params.get("diverse_groups", 1),
        diversity_strength=best_params.get("diversity_strength", 0.0),
    )
    print(f"Confirmed CER: {best_cer:.4f}    Confirmed accuracy: {best_accuracy:.4f}")

    if ENABLE_DASHBOARD and optuna_dashboard is not None and STORAGE is not None:
        print(
            f"[Dashboard] Keep monitoring at http://{DASHBOARD_HOST}:{DASHBOARD_PORT} "
            "or stop the script to close it."
        )


if __name__ == "__main__":
    main()
