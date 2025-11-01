"""
Скрипт для тестирования инференса ONNX модели EAST и сравнения с PyTorch.

Usage:
    python scripts/test_east_onnx_inference.py --model east_test.onnx --image "C:\shared\Archive_19_04\combined_images\16.jpg"
    python scripts/test_east_onnx_inference.py --model east_model.onnx --image path/to/image.jpg --compare
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from manuscript.detectors._east.east import EAST as EASTModel
from manuscript.detectors._east.utils import decode_quads_from_maps, visualize_page


def preprocess_image(image_path: str, target_size: int = 1280) -> tuple:
    """
    Загружает и предобрабатывает изображение для EAST.

    Returns
    -------
    tuple
        (preprocessed_tensor, original_image, original_size)
    """
    # Загружаем изображение
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img.shape[:2]  # (H, W)

    # Ресайзим до target_size
    img_resized = cv2.resize(img, (target_size, target_size))

    # Нормализуем в [-1, 1]
    img_norm = (img_resized.astype(np.float32) / 255.0 - 0.5) / 0.5

    # Конвертируем в tensor [1, 3, H, W]
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)

    return img_tensor, img, original_size


def run_onnx_inference(
    onnx_path: str,
    image_tensor: np.ndarray,
    providers: list = None,
) -> tuple:
    """
    Запускает инференс ONNX модели.

    Parameters
    ----------
    onnx_path : str
        Путь к ONNX модели
    image_tensor : np.ndarray
        Предобработанное изображение [1, 3, H, W]
    providers : list, optional
        ONNX Runtime провайдеры (default: ["CPUExecutionProvider"])

    Returns
    -------
    tuple
        (score_map, geo_map, inference_time)
    """
    import onnxruntime as ort

    if providers is None:
        providers = ["CPUExecutionProvider"]

    # Создаём сессию
    session = ort.InferenceSession(onnx_path, providers=providers)

    # Получаем имена входов/выходов
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]

    print(f"ONNX Runtime providers: {session.get_providers()}")
    print(f"Input: {input_name}")
    print(f"Outputs: {output_names}")

    # Конвертируем в numpy если это torch.Tensor
    if isinstance(image_tensor, torch.Tensor):
        image_np = image_tensor.numpy()
    else:
        image_np = image_tensor

    # Запускаем инференс
    start_time = time.time()
    outputs = session.run(output_names, {input_name: image_np})
    inference_time = time.time() - start_time

    score_map, geo_map = outputs

    return score_map, geo_map, inference_time


def run_pytorch_inference(
    weights_path: str,
    image_tensor: torch.Tensor,
    device: str = "cpu",
) -> tuple:
    """
    Запускает инференс PyTorch модели для сравнения.

    Returns
    -------
    tuple
        (score_map, geo_map, inference_time)
    """
    # Загружаем модель
    model = EASTModel(
        pretrained_backbone=False,
        pretrained_model_path=weights_path,
    ).to(device)
    model.eval()

    # Запускаем инференс
    start_time = time.time()
    with torch.no_grad():
        score_map, geo_map = model(image_tensor.to(device))
    inference_time = time.time() - start_time

    # Конвертируем в numpy
    score_map = score_map.cpu().numpy()
    geo_map = geo_map.cpu().numpy()

    return score_map, geo_map, inference_time


def compare_outputs(
    pytorch_score: np.ndarray,
    pytorch_geo: np.ndarray,
    onnx_score: np.ndarray,
    onnx_geo: np.ndarray,
    rtol: float = 1e-3,
    atol: float = 1e-5,
):
    """
    Сравнивает выходы PyTorch и ONNX моделей.
    """
    print("\n=== Output Comparison ===")

    # Score map
    score_diff = np.abs(pytorch_score - onnx_score)
    print(f"Score map:")
    print(f"  Max difference: {score_diff.max():.6f}")
    print(f"  Mean difference: {score_diff.mean():.6f}")
    print(f"  Median difference: {np.median(score_diff):.6f}")

    score_close = np.allclose(pytorch_score, onnx_score, rtol=rtol, atol=atol)
    print(f"  All close (rtol={rtol}, atol={atol}): {score_close}")

    # Geo map
    geo_diff = np.abs(pytorch_geo - onnx_geo)
    print(f"\nGeo map:")
    print(f"  Max difference: {geo_diff.max():.6f}")
    print(f"  Mean difference: {geo_diff.mean():.6f}")
    print(f"  Median difference: {np.median(geo_diff):.6f}")

    geo_close = np.allclose(pytorch_geo, onnx_geo, rtol=rtol, atol=atol)
    print(f"  All close (rtol={rtol}, atol={atol}): {geo_close}")

    if score_close and geo_close:
        print("\n[OK] ONNX and PyTorch outputs match!")
    else:
        print("\n[WARNING] ONNX and PyTorch outputs differ")
        print("  This might be acceptable if differences are small")


def decode_and_visualize(
    score_map: np.ndarray,
    geo_map: np.ndarray,
    original_image: np.ndarray,
    original_size: tuple,
    target_size: int,
    output_path: str,
    score_thresh: float = 0.6,
):
    """
    Декодирует детекции и сохраняет визуализацию.
    """
    # Декодируем боксы
    boxes = decode_quads_from_maps(
        score_map[0, 0],  # [H, W]
        geo_map[0],  # [5, H, W]
        score_thresh=score_thresh,
        scale=0.25,
        quantization=2,
    )

    print(f"\n=== Detection Results ===")
    print(f"Detected {len(boxes)} boxes")

    if len(boxes) > 0:
        # Масштабируем боксы к оригинальному размеру
        orig_h, orig_w = original_size
        scale_x = orig_w / target_size
        scale_y = orig_h / target_size

        boxes_scaled = boxes.copy()
        boxes_scaled[:, 0:8:2] *= scale_x
        boxes_scaled[:, 1:8:2] *= scale_y

        # Конвертируем в список полигонов
        polygons = []
        for box in boxes_scaled:
            poly = box[:8].reshape(4, 2)
            polygons.append(poly)

        # Визуализируем
        vis_image = visualize_page(
            original_image,
            polygons,
            show_order=True,
        )

        # Сохраняем
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, vis_image_bgr)
        print(f"[OK] Visualization saved to: {output_path}")
    else:
        print("[WARNING] No boxes detected")


def main():
    parser = argparse.ArgumentParser(description="Test EAST ONNX model inference")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to test image",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with PyTorch model",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to PyTorch weights for comparison",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="onnx_detection_result.jpg",
        help="Output path for visualization",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=1280,
        help="Target image size (default: 1280)",
    )
    parser.add_argument(
        "--score-thresh",
        type=float,
        default=0.6,
        help="Score threshold for detections (default: 0.6)",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Use CUDA provider for ONNX Runtime (if available)",
    )

    args = parser.parse_args()

    # Проверяем файлы
    if not Path(args.model).exists():
        print(f"Error: ONNX model not found: {args.model}")
        sys.exit(1)

    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        sys.exit(1)

    print(f"=== ONNX Inference Test ===")
    print(f"Model: {args.model}")
    print(f"Image: {args.image}")
    print(f"Target size: {args.target_size}")

    # Предобрабатываем изображение
    print("\nPreprocessing image...")
    image_tensor, original_image, original_size = preprocess_image(
        args.image, args.target_size
    )
    print(f"Original size: {original_size}")
    print(f"Preprocessed tensor shape: {image_tensor.shape}")

    # ONNX инференс
    print("\n=== Running ONNX Inference ===")
    try:
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if args.use_cuda:
            providers.insert(0, "CUDAExecutionProvider")

        onnx_score, onnx_geo, onnx_time = run_onnx_inference(
            args.model, image_tensor, providers
        )
        print(f"[OK] ONNX inference completed in {onnx_time*1000:.2f} ms")
        print(f"  Score map shape: {onnx_score.shape}")
        print(f"  Geo map shape: {onnx_geo.shape}")

    except ImportError:
        print("Error: onnxruntime not installed")
        print("Install with: pip install onnxruntime")
        sys.exit(1)
    except Exception as e:
        print(f"Error during ONNX inference: {e}")
        sys.exit(1)

    # PyTorch инференс для сравнения
    if args.compare:
        print("\n=== Running PyTorch Inference (for comparison) ===")

        if args.weights is None:
            weights_path = Path.home() / ".manuscript" / "east" / "east_quad_23_05.pth"
            if not weights_path.exists():
                print(f"Error: Default weights not found at {weights_path}")
                sys.exit(1)
            args.weights = str(weights_path)

        device = "cuda" if torch.cuda.is_available() and args.use_cuda else "cpu"
        print(f"Using device: {device}")

        pytorch_score, pytorch_geo, pytorch_time = run_pytorch_inference(
            args.weights, image_tensor, device
        )
        print(f"[OK] PyTorch inference completed in {pytorch_time*1000:.2f} ms")

        # Сравниваем
        compare_outputs(pytorch_score, pytorch_geo, onnx_score, onnx_geo)

        print(f"\n=== Performance Comparison ===")
        print(f"PyTorch: {pytorch_time*1000:.2f} ms")
        print(f"ONNX: {onnx_time*1000:.2f} ms")
        speedup = pytorch_time / onnx_time
        print(f"Speedup: {speedup:.2f}x")

    # Декодируем и визуализируем результат
    decode_and_visualize(
        onnx_score,
        onnx_geo,
        original_image,
        original_size,
        args.target_size,
        args.output,
        args.score_thresh,
    )

    print(f"\n[OK] Test completed successfully!")


if __name__ == "__main__":
    main()
