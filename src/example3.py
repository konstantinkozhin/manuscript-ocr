from pathlib import Path
import time
from statistics import mean

from tqdm import tqdm

from manuscript.detectors import EASTInfer

# Укажите путь к папке с изображениями
FOLDER = Path(r"C:\shared\Archive_19_04\combined_images")

# Настройки
RECURSIVE = False  # если нужно обрабатывать подпапки, выставьте True
KEEP_ROTATED = (
    False  # True — оставить исходные боксы, False — привести к осеориентированным
)
DISABLE_ANOMALY_FILTER = False  # True — отключить фильтр аномальных по площади боксов


def collect_images(folder: Path, recursive: bool) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if recursive:
        images = [
            p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts
        ]
    else:
        images = [
            p
            for p in sorted(folder.iterdir())
            if p.is_file() and p.suffix.lower() in exts
        ]
    return images


def main() -> None:
    if not FOLDER.exists() or not FOLDER.is_dir():
        raise SystemExit(f"Папка {FOLDER} не найдена или не является директорией.")

    image_paths = collect_images(FOLDER, RECURSIVE)
    if not image_paths:
        raise SystemExit("В папке нет изображений поддерживаемых форматов.")

    print(f"Найдено изображений: {len(image_paths)}")

    model = EASTInfer(
        weights_path=r"C:\east_quad_23_05.pth",
        axis_aligned_output=not KEEP_ROTATED,
        remove_area_anomalies=not DISABLE_ANOMALY_FILTER,
    )

    all_times: list[float] = []
    total_start = time.perf_counter()

    for image_path in tqdm(image_paths, desc="Inference", unit="img"):
        t0 = time.perf_counter()
        model.infer(str(image_path), vis=False, profile=False)
        all_times.append(time.perf_counter() - t0)

    total_elapsed = time.perf_counter() - total_start

    print("\n--- Результаты ---")
    print(f"Всего изображений: {len(image_paths)}")
    print(f"Суммарное время: {total_elapsed:.3f} с")
    print(f"Среднее на изображение: {mean(all_times):.3f} с")
    print(
        f"Минимальное / Максимальное: {min(all_times):.3f} с / {max(all_times):.3f} с"
    )


if __name__ == "__main__":
    main()
