import os
import threading
import time
import webbrowser
from typing import Dict, List, Optional

import optuna
from tqdm import tqdm

try:
    import optuna_dashboard
except ImportError:  # pragma: no cover
    optuna_dashboard = None  # type: ignore

from manuscript.recognizers import TRBA
from manuscript.recognizers._trba.training.metrics import (
    character_error_rate,
    compute_accuracy,
)

# === User-configurable paths & parameters ===
IMAGE_DIR = r"C:\shared\Archive_19_04\data_archive\test"
GT_FILE = r"C:\shared\Archive_19_04\data_archive\gt_test - Copy.txt"
MODEL_PATH = r"C:\shared\exp1_model_64\best_acc_ckpt.pth"
CONFIG_PATH = r"C:\shared\exp1_model_64\config.json"
CHARSET_PATH: Optional[str] = (
    None  # set to override default charset, otherwise keep None
)
BATCH_SIZE = 128
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


def make_image_list_from_gt(gt_map: Dict[str, str], image_dir: str) -> List[str]:
    """
    Строит список путей к изображениям строго по порядку ключей в gt_map,
    игнорируя содержимое директории. Предупреждает об отсутствующих файлах.
    """
    images: List[str] = []
    missing: List[str] = []

    for fname in gt_map.keys():  # порядок соответствует строкам в GT_FILE (Py3.7+)
        path = os.path.join(image_dir, fname)
        if os.path.isfile(path):
            images.append(path)
        else:
            missing.append(fname)

    if missing:
        print(
            f"[Warn] В GT указано {len(missing)} файлов, которых нет в папке '{image_dir}':"
        )
        for m in missing[:20]:
            print("   -", m)
        if len(missing) > 20:
            print(f"   ... и ещё {len(missing) - 20}")

    if not images:
        raise RuntimeError("По GT не найдено ни одного существующего изображения.")

    return images


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
    recognizer: TRBA,
    images: List[str],
    gt_map: Dict[str, str],
    batch_size: int,
    *,
    mode: str,
    beam_size: int,
    alpha: float,
    temperature: float = 1.0,
) -> float:
    num_batches = (len(images) + batch_size - 1) // batch_size

    print(
        f"  🔄 Запуск предсказания: {len(images)} изображений, {num_batches} батчей "
        f"(batch_size={batch_size}, mode={mode}, beam_size={beam_size})"
    )

    # Создаём прогресс-бар для батчей
    with tqdm(
        total=num_batches, desc="  Обработка батчей", unit="batch", leave=False
    ) as pbar:
        # Разбиваем на батчи и обрабатываем с обновлением прогресс-бара
        all_predictions = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            batch_predictions = recognizer.predict(
                images=batch_images,
                batch_size=batch_size,
                mode=mode,
                beam_size=beam_size,
                alpha=alpha,
                temperature=temperature,
            )
            all_predictions.extend(batch_predictions)
            pbar.update(1)

    predictions = all_predictions
    print(f"  ✅ Предсказания получены, вычисляем метрики...")

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
    recognizer: TRBA,
    images: List[str],
    gt_map: Dict[str, str],
    batch_size: int,
    *,
    mode: str,
    beam_size: int,
    alpha: float,
    temperature: float,
) -> tuple[float, float]:
    """
    Возвращает (avg_cer, accuracy).
    accuracy - доля точных совпадений, вычисленная через compute_accuracy.
    """
    num_batches = (len(images) + batch_size - 1) // batch_size

    print(
        f"  🔄 Запуск предсказания: {len(images)} изображений, {num_batches} батчей "
        f"(batch_size={batch_size}, mode={mode}, beam_size={beam_size})"
    )

    # Создаём прогресс-бар для батчей
    with tqdm(
        total=num_batches, desc="  Обработка батчей", unit="batch", leave=False
    ) as pbar:
        # Разбиваем на батчи и обрабатываем с обновлением прогресс-бара
        all_predictions = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            batch_predictions = recognizer.predict(
                images=batch_images,
                batch_size=batch_size,
                mode=mode,
                beam_size=beam_size,
                alpha=alpha,
                temperature=temperature,
            )
            all_predictions.extend(batch_predictions)
            pbar.update(1)

    predictions = all_predictions
    print(f"  ✅ Предсказания получены, вычисляем метрики...")

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
    print("📂 Загрузка ground truth...")
    gt_map = load_ground_truth(GT_FILE)
    print(f"   Загружено {len(gt_map)} записей ground truth")

    print("🖼️  Сборка списка изображений...")
    images = make_image_list_from_gt(gt_map, IMAGE_DIR)
    print(f"   Найдено {len(images)} изображений\n")

    print("🤖 Загрузка модели TRBA...")
    recognizer = TRBA(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        charset_path=CHARSET_PATH,
    )
    print("   Модель загружена\n")

    baseline_params = {
        "mode": "greedy",
        "beam_size": 1,
        "alpha": 0.0,
    }
    print(f"\n[Baseline evaluation] Запуск с параметрами: {baseline_params}")
    baseline_cer = evaluate_avg_cer(
        recognizer,
        images,
        gt_map,
        BATCH_SIZE,
        **baseline_params,
    )
    print(f"[Baseline] mode=greedy Avg CER={baseline_cer:.4f}\n")

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
            # Для greedy режима beam-специфичные параметры фиксированы
            beam_size = 1
            alpha = 0.0
            temperature = 1.0
            # normalize_by_length = True
            # diverse_groups = 1
            # diversity_strength = 0.0
            # noise_level = 0.3
            # topk_sampling_steps = 3
            # topk = 5
            # coverage_penalty_weight = 0.1
            # expand_beam_steps = 3
        else:
            # Для beam режима оптимизируем параметры
            beam_size = trial.suggest_int("beam_size", 2, 12)
            alpha = trial.suggest_float("alpha", 0.0, 1.0)
            temperature = trial.suggest_float("temperature", 0.7, 2.0)
            # normalize_by_length = True
            # diverse_groups = trial.suggest_int("diverse_groups", 1, 4)
            # diversity_strength = trial.suggest_float("diversity_strength", 0.0, 2.0)
            # noise_level = trial.suggest_float("noise_level", 0.0, 1.0)
            # topk_sampling_steps = trial.suggest_int("topk_sampling_steps", 0, 10)
            # topk = trial.suggest_int("topk", 1, 20)
            # coverage_penalty_weight = trial.suggest_float(
            #     "coverage_penalty_weight", 0.0, 1.0
            # )
            # expand_beam_steps = trial.suggest_int("expand_beam_steps", 0, 10)

        avg_cer, accuracy = evaluate_metrics(
            recognizer,
            images,
            gt_map,
            BATCH_SIZE,
            mode=mode,
            beam_size=beam_size,
            alpha=alpha,
            temperature=temperature,
        )

        if mode == "greedy":
            print(
                f"[Trial {trial.number}] mode=greedy -> CER={avg_cer:.4f} Acc={accuracy:.4f}"
            )
        else:
            print(
                f"[Trial {trial.number}] mode=beam beam={beam_size} alpha={alpha:.2f} "
                f"-> CER={avg_cer:.4f} Acc={accuracy:.4f}"
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
        alpha=best_params.get("alpha", 0.0),
        temperature=best_params.get("temperature", 1.0),
    )
    print(f"Confirmed CER: {best_cer:.4f}    Confirmed accuracy: {best_accuracy:.4f}")

    if ENABLE_DASHBOARD and optuna_dashboard is not None and STORAGE is not None:
        print(
            f"[Dashboard] Keep monitoring at http://{DASHBOARD_HOST}:{DASHBOARD_PORT} "
            "or stop the script to close it."
        )


if __name__ == "__main__":
    main()
