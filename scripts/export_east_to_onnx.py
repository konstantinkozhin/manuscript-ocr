"""
Скрипт для экспорта модели EAST в формат ONNX.

Usage:
    python scripts/export_east_to_onnx.py
    python scripts/export_east_to_onnx.py --weights path/to/weights.pth --output model.onnx
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from manuscript.detectors._east.east import EAST as EASTModel


class EASTWrapper(torch.nn.Module):
    """Wrapper для EAST модели, который возвращает tuple вместо dict для ONNX."""

    def __init__(self, east_model):
        super().__init__()
        self.east = east_model

    def forward(self, x):
        output = self.east(x)
        return output["score"], output["geometry"]


def export_east_to_onnx(
    weights_path: str,
    output_path: str,
    input_size: int = 1280,
    opset_version: int = 14,
    simplify: bool = True,
):
    """
    Экспортирует модель EAST в формат ONNX.

    Parameters
    ----------
    weights_path : str
        Путь к весам модели PyTorch (.pth файл)
    output_path : str
        Путь для сохранения ONNX модели
    input_size : int
        Размер входного изображения (квадратное)
    opset_version : int
        Версия ONNX opset (рекомендуется 14+)
    simplify : bool
        Использовать onnx-simplifier для оптимизации графа
    """
    print(f"Loading EAST model from: {weights_path}")

    # Создаём модель
    east_model = EASTModel(
        pretrained_backbone=False,
        pretrained_model_path=weights_path,
    )
    east_model.eval()

    # Оборачиваем для ONNX экспорта
    model = EASTWrapper(east_model)
    model.eval()

    # Создаём dummy input
    # EAST принимает [B, 3, H, W] тензор (нормализованные RGB изображения)
    dummy_input = torch.randn(1, 3, input_size, input_size)

    print(f"Model architecture: {model.__class__.__name__}")
    print(f"Input shape: {dummy_input.shape}")

    # Проверяем что модель работает
    with torch.no_grad():
        score_map, geo_map = model(dummy_input)

    print(f"Output shapes:")
    print(f"  - score_map: {score_map.shape}")
    print(f"  - geo_map: {geo_map.shape}")

    # Экспортируем в ONNX
    print(f"\nExporting to ONNX (opset {opset_version})...")

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["score_map", "geo_map"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "score_map": {0: "batch_size", 2: "height", 3: "width"},
            "geo_map": {0: "batch_size", 2: "height", 3: "width"},
        },
        verbose=False,
    )

    print(f"[OK] ONNX model saved to: {output_path}")

    # Проверяем ONNX модель
    print("\nVerifying ONNX model...")
    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("[OK] ONNX model is valid")

    # Опционально: упрощаем граф
    if simplify:
        try:
            import onnxsim

            print("\nSimplifying ONNX model...")
            model_simplified, check = onnxsim.simplify(onnx_model)

            if check:
                onnx.save(model_simplified, output_path)
                print("[OK] ONNX model simplified and saved")
            else:
                print("[WARNING] Simplification check failed, keeping original model")
        except ImportError:
            print("[WARNING] onnx-simplifier not installed, skipping simplification")
            print("  Install with: pip install onnx-simplifier")

    # Выводим информацию о модели
    print(f"\n=== ONNX Model Info ===")
    print(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Opset version: {opset_version}")
    print(f"Inputs: {[inp.name for inp in onnx_model.graph.input]}")
    print(f"Outputs: {[out.name for out in onnx_model.graph.output]}")


def main():
    parser = argparse.ArgumentParser(description="Export EAST model to ONNX format")
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to EAST weights (.pth file). If not specified, uses default weights from ~/.manuscript/east/",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="east_model.onnx",
        help="Output path for ONNX model (default: east_model.onnx)",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=1280,
        help="Input image size (default: 1280)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable ONNX graph simplification",
    )

    args = parser.parse_args()

    # Если weights не указан, используем дефолтный путь
    if args.weights is None:
        weights_path = Path.home() / ".manuscript" / "east" / "east_quad_23_05.pth"
        if not weights_path.exists():
            print(f"Error: Default weights not found at {weights_path}")
            print("Please download weights first or specify --weights argument")
            sys.exit(1)
        args.weights = str(weights_path)

    # Проверяем что файл весов существует
    if not Path(args.weights).exists():
        print(f"Error: Weights file not found: {args.weights}")
        sys.exit(1)

    # Экспортируем
    export_east_to_onnx(
        weights_path=args.weights,
        output_path=args.output,
        input_size=args.input_size,
        opset_version=args.opset,
        simplify=not args.no_simplify,
    )

    print(f"\n[OK] Export completed successfully!")
    print(f"\nNext steps:")
    print(
        f"  1. Test ONNX inference: python scripts/test_east_onnx_inference.py --model {args.output}"
    )
    print(f"  2. Benchmark performance: python scripts/benchmark_east_onnx.py")


if __name__ == "__main__":
    main()
