"""Local experimental models demo.

Run: python scripts/gradio_demo_local.py (http://127.0.0.1:7861).
Optional: --port 7862, --share. Model roots can be overridden with
MANUSCRIPT_DEMO_MODELS and MANUSCRIPT_DEMO_PPOCR_DET environment variables.
"""

import time
import tempfile
import os
import sys
import html
from pathlib import Path
from functools import lru_cache

# Run directly from the checkout, including unreleased model classes.
_checkout = Path(__file__).resolve().parents[1] / "src"
if _checkout.is_dir():
    sys.path.insert(0, str(_checkout))

import gradio as gr
import numpy as np
from PIL import Image

from manuscript import CharLM, Pipeline
from manuscript.detectors import EAST, YOLO
from manuscript.detectors._mask2former import Mask2Former
from manuscript.detectors._ppocr import PPOCR
from manuscript.detectors._rfdetr import RFDETR
from manuscript.recognizers import TRBA, PPOCRRec, TrOCR


# Override roots when moving local bundles to another machine.
MODEL_ROOT = Path(
    os.environ.get("MANUSCRIPT_DEMO_MODELS", Path.home() / "Desktop" / "модельки")
).expanduser()
PPOCR_DET_ROOT = Path(
    os.environ.get(
        "MANUSCRIPT_DEMO_PPOCR_DET", Path.home() / "Desktop" / "PP-OCRv6_tiny_det"
    )
).expanduser()
LOCAL_DETECTORS = {}
LOCAL_RECOGNIZERS = {}
DETECTOR_MODELS = [
    "mask2former_line_v0_prev",
    "yolo26x_obb_text_g1",
    "yolo26s_obb_text_g1",
    "east_50_g1",
]
RECOGNIZER_MODELS = ["trba_lite_g2", "trba_lite_g1", "trba_base_g1"]
CORRECTOR_MODELS = ["Без корректора", "prereform_charlm_g1", "modern_charlm_g1"]

for precision in ("fp16", "fp32"):
    key = f"PP-OCRv6 tiny det · {precision.upper()}"
    LOCAL_DETECTORS[key] = (
        "ppocr",
        PPOCR_DET_ROOT / f"ppocrv6_tiny_det.{precision}.onnx",
    )
    key = f"RF-DETR textline/textregion 2XL · {precision.upper()}"
    name = "rfdetr-textline-textregion-detection-2xl"
    LOCAL_DETECTORS[key] = (
        "rfdetr",
        MODEL_ROOT / name / "segmentation" / f"{name}.{precision}.onnx",
    )
    for name in (
        "cyrillic_PP-OCRv5_mobile_rec",
        "eslav_PP-OCRv5_mobile_rec",
        "kraken-ppocrv6-tiny",
        "kraken-ppocrv6-small",
        "kraken-ppocrv6-medium",
    ):
        LOCAL_RECOGNIZERS[f"{name} · {precision.upper()}"] = (
            "ppocr",
            MODEL_ROOT / name / f"{name}.{precision}.onnx",
        )
    LOCAL_RECOGNIZERS[f"cyrillic-large-handwritten (TrOCR) · {precision.upper()}"] = (
        "trocr",
        MODEL_ROOT / "cyrillic-large-handwritten" / f"encoder.{precision}.onnx",
    )
    LOCAL_RECOGNIZERS[f"cyrillic-large-handwritten (TrOCR KV-cache) · {precision.upper()}"] = (
        "trocr",
        MODEL_ROOT / "cyrillic-large-handwritten-kv-cache" / f"encoder.{precision}.onnx",
    )
for name in ("kraken-ppocrv6-small", "kraken-ppocrv6-medium"):
    LOCAL_RECOGNIZERS[f"{name} · INT8 веса (эксперимент)"] = (
        "ppocr",
        MODEL_ROOT / "квантование" / name / f"{name}.int8w.onnx",
    )
DETECTOR_MODELS += list(LOCAL_DETECTORS)
RECOGNIZER_MODELS += list(LOCAL_RECOGNIZERS)


def require_local_bundle(path, algorithm):
    required = [path, path.with_suffix(".json")]
    if algorithm == "trocr":
        import json

        if path.with_suffix(".json").is_file():
            config = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            required += [
                path.parent / config[role] for role in ("decoder", "tokenizer")
            ]
            if config.get("decoder_format") == "kv_cache":
                required.append(path.parent / config["decoder_with_past"])
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(
            "Не найдены файлы локальной модели: " + ", ".join(missing)
        )
    return str(path)


DETECTOR_DEFAULTS = {
    "mask2former_line_v0_prev": {
        "target_size": 1024,
        "score_thresh": 0.15,
        "expand_ratio_w": 1.0,
        "expand_ratio_h": 1.0,
        "is_east": False,
        "is_mask2former": True,
    },
    "yolo26x_obb_text_g1": {
        "target_size": 1408,
        "score_thresh": 0.1,
        "expand_ratio_w": 1.4,
        "expand_ratio_h": 1.5,
        "is_east": False,
        "is_mask2former": False,
    },
    "yolo26s_obb_text_g1": {
        "target_size": 1408,
        "score_thresh": 0.1,
        "expand_ratio_w": 1.4,
        "expand_ratio_h": 1.5,
        "is_east": False,
        "is_mask2former": False,
    },
    "east_50_g1": {
        "target_size": 1280,
        "score_thresh": 0.6,
        "expand_ratio_w": 1.4,
        "expand_ratio_h": 1.5,
        "is_east": True,
        "is_mask2former": False,
    },
}

for key, (algorithm, path) in LOCAL_DETECTORS.items():
    DETECTOR_DEFAULTS[key] = {
        "target_size": 736 if algorithm == "ppocr" else 768,
        "score_thresh": 0.4 if algorithm == "ppocr" else 0.5,
        "expand_ratio_w": 1.0,
        "expand_ratio_h": 1.0,
        "is_east": False,
        "is_mask2former": False,
    }

last_recognition_page = None
last_correction_page = None


@lru_cache(maxsize=1)
def create_pipeline(
    detector_model,
    recognizer_model,
    corrector_model,
    target_size,
    score_thresh,
    expand_ratio_w,
    expand_ratio_h,
    mask_threshold,
    apply_threshold,
    max_edits,
    region_preparer="bbox",
    batch_size=2,
):
    if detector_model in LOCAL_DETECTORS:
        algorithm, path = LOCAL_DETECTORS[detector_model]
        weights = require_local_bundle(path, algorithm)
        if algorithm == "ppocr":
            detector = PPOCR(
                weights=weights, box_threshold=score_thresh
            )
        else:
            detector = RFDETR(weights=weights, score_thresh=score_thresh)
    elif detector_model == "mask2former_line_v0_prev":
        detector = Mask2Former(
            weights=detector_model,
            score_threshold=score_thresh,
        )
    elif detector_model.startswith("east_"):
        detector = EAST(
            weights=detector_model,
            target_size=int(target_size),
            score_thresh=score_thresh,
            expand_ratio_w=expand_ratio_w,
            expand_ratio_h=expand_ratio_h,
        )
    else:
        detector = YOLO(
            weights=detector_model,
            target_size=int(target_size),
            score_thresh=score_thresh,
        )

    if recognizer_model in LOCAL_RECOGNIZERS:
        algorithm, path = LOCAL_RECOGNIZERS[recognizer_model]
        weights = require_local_bundle(path, algorithm)
        cls = TrOCR if algorithm == "trocr" else PPOCRRec
        recognizer = cls(weights=weights, region_preparer=region_preparer, batch_size=int(batch_size))
    else:
        recognizer = TRBA(weights=recognizer_model, region_preparer=region_preparer, batch_size=int(batch_size))
    corrector = (
        None
        if corrector_model == "Без корректора"
        else CharLM(
            weights=corrector_model,
            mask_threshold=mask_threshold,
            apply_threshold=apply_threshold,
            max_edits=int(max_edits),
        )
    )

    return Pipeline(
        detector=detector,
        recognizer=recognizer,
        corrector=corrector,
    )


def update_detector_controls(detector_model):
    cfg = DETECTOR_DEFAULTS[detector_model]
    is_east = cfg["is_east"]
    is_mask2former = cfg["is_mask2former"]

    return (
        gr.update(
            value=cfg["target_size"],
            interactive=False if not is_east else True,
            label=(
                "Размер изображения"
                if is_east
                else (
                    "Размер изображения (из конфигурации модели)"
                    if is_mask2former
                    else "Размер изображения (по модели)"
                )
            ),
        ),
        gr.update(value=cfg["score_thresh"]),
        gr.update(
            value=cfg["expand_ratio_w"],
            visible=is_east,
            interactive=is_east,
        ),
        gr.update(
            value=cfg["expand_ratio_h"],
            visible=is_east,
            interactive=is_east,
        ),
    )


def count_words_in_page(page):
    if page is None:
        return 0

    count = 0
    for block in page.blocks:
        for line in block.lines:
            count += sum(1 for span in line.text_spans if span.text)
    return count


html_escape = html.escape


def highlight_differences(original, corrected):
    html = []
    i, j = 0, 0

    while i < len(original) or j < len(corrected):
        if i < len(original) and j < len(corrected):
            if original[i] == corrected[j]:
                if original[i] == "\n":
                    html.append("<br>")
                else:
                    html.append(html_escape(corrected[j]))
                i += 1
                j += 1
            else:
                html.append(
                    f'<span style="background-color: #90EE90; font-weight: bold;">{html_escape(corrected[j])}</span>'
                )
                i += 1
                j += 1
        elif i < len(original):
            i += 1
        else:
            html.append(
                f'<span style="background-color: #90EE90; font-weight: bold;">{html_escape(corrected[j])}</span>'
            )
            j += 1

    return f'<div style="white-space: pre-wrap; font-family: monospace;">{"".join(html)}</div>'


def process_image(
    image,
    detector_model,
    recognizer_model,
    corrector_model,
    target_size,
    score_thresh,
    expand_ratio_w,
    expand_ratio_h,
    mask_threshold,
    apply_threshold,
    max_edits,
    region_preparer="bbox",
    batch_size=2,
):
    global last_recognition_page, last_correction_page

    last_recognition_page = last_correction_page = None

    if image is None:
        return None, "", "", ""

    try:
        pipeline = create_pipeline(
            detector_model,
            recognizer_model,
            corrector_model,
            target_size,
            score_thresh,
            expand_ratio_w,
            expand_ratio_h,
            mask_threshold,
            apply_threshold,
            max_edits,
            region_preparer,
            batch_size,
        )

        start_time = time.time()
        _, vis_image = pipeline.predict(image, vis=True)
        elapsed_time = time.time() - start_time

        last_recognition_page = pipeline.last_recognition_page
        last_correction_page = pipeline.last_correction_page or last_recognition_page

        text_before = (
            pipeline.get_text(last_recognition_page) if last_recognition_page else ""
        )
        text_after = (
            pipeline.get_text(last_correction_page) if last_correction_page else ""
        )

        word_count = count_words_in_page(last_correction_page)
        pages_per_sec = 1.0 / elapsed_time if elapsed_time > 0 else 0.0
        words_per_sec = word_count / elapsed_time if elapsed_time > 0 else 0.0

        stats_text = (
            f"Время: {elapsed_time:.2f} сек | "
            f"{pages_per_sec:.2f} стр/сек | "
            f"{words_per_sec:.1f} слов/сек"
        )

        if isinstance(vis_image, np.ndarray):
            vis_image = Image.fromarray(vis_image)

        highlighted = highlight_differences(text_before, text_after)

        return vis_image, text_before, highlighted, stats_text

    except Exception as e:
        error_msg = f"Ошибка: {e}"
        return None, error_msg, "", error_msg


def save_recognition_json():
    global last_recognition_page

    if last_recognition_page is None:
        return None

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(last_recognition_page.to_json())
        return f.name


def save_correction_json():
    global last_correction_page

    if last_correction_page is None:
        return None

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(last_correction_page.to_json())
        return f.name


default_detector = "PP-OCRv6 tiny det · FP16"
default_cfg = DETECTOR_DEFAULTS[default_detector]

_gradio6 = int(gr.__version__.split(".")[0]) >= 6
_theme_options = {"theme": gr.themes.Soft()}
with gr.Blocks(
    title="OCR Pipeline — local models", **({} if _gradio6 else _theme_options)
) as demo:
    gr.Markdown("# Manuscript Demo — локальные модели")
    gr.Markdown(
        "Выберите детектор и распознаватель. TrOCR обрабатывает выделенные строки и может работать медленно на CPU. INT8 веса — экспериментальные варианты с небольшими изменениями текста. RF-DETR возвращает контуры текстовых строк классов 2–5. Способ вырезания области выбирается отдельно."
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Изображение", type="pil")

            with gr.Row():
                detector_selector = gr.Dropdown(
                    choices=DETECTOR_MODELS,
                    value=default_detector,
                    label="Детектор",
                )
                recognizer_selector = gr.Dropdown(
                    choices=RECOGNIZER_MODELS,
                    value="kraken-ppocrv6-small · FP16",
                    label="Распознаватель",
                )
                corrector_selector = gr.Dropdown(
                    choices=CORRECTOR_MODELS,
                    value="Без корректора",
                    label="Корректор",
                )

            with gr.Accordion("Параметры детектора", open=False):
                target_size = gr.Slider(
                    640,
                    2560,
                    value=default_cfg["target_size"],
                    step=64,
                    label="Размер изображения (по модели)",
                    interactive=False,
                )
                score_thresh = gr.Slider(
                    0.1,
                    0.9,
                    value=default_cfg["score_thresh"],
                    step=0.05,
                    label="Порог уверенности",
                )
                expand_ratio_w = gr.Slider(
                    0.5,
                    3.0,
                    value=default_cfg["expand_ratio_w"],
                    step=0.1,
                    label="Расширение по ширине (EAST)",
                    visible=False,
                    interactive=False,
                )
                expand_ratio_h = gr.Slider(
                    0.5,
                    3.0,
                    value=default_cfg["expand_ratio_h"],
                    step=0.1,
                    label="Расширение по высоте (EAST)",
                    visible=False,
                    interactive=False,
                )

            region_preparer_selector = gr.Dropdown(
                choices=[
                    ("Прямоугольная вырезка", "bbox"),
                    ("Маска по полигону", "polygon_mask"),
                    ("Выпрямить четырехугольник", "quad_warp"),
                ],
                value="bbox",
                label="Подготовка области для распознавателя",
            )
            recognition_batch_size = gr.Slider(
                1, 32, value=2, step=1,
                label="Областей в партии распознавания",
                info="Большие партии используют больше памяти. Размер ограничен возможностями модели.",
            )

            with gr.Accordion("Параметры корректора", open=False):
                mask_threshold = gr.Slider(
                    0.0, 0.5, value=0.05, step=0.01, label="Порог маскирования"
                )
                apply_threshold = gr.Slider(
                    0.5, 1.0, value=0.95, step=0.01, label="Порог применения"
                )
                max_edits = gr.Slider(1, 10, value=1, step=1, label="Максимум правок")

            btn = gr.Button("Распознать", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Визуализация", type="pil")
            stats_display = gr.Textbox(label="Статистика", interactive=False)

    with gr.Row():
        with gr.Column():
            text_before = gr.Textbox(label="Текст без корректора", lines=10)
            btn_save_recognition = gr.Button("Сохранить в JSON")
            file_recognition = gr.File(label="Результат распознавания")

        with gr.Column():
            text_after = gr.HTML(label="Текст с корректором")
            btn_save_correction = gr.Button("Сохранить в JSON")
            file_correction = gr.File(label="Результат коррекции")

    detector_selector.change(
        update_detector_controls,
        inputs=[detector_selector],
        outputs=[target_size, score_thresh, expand_ratio_w, expand_ratio_h],
    )

    btn.click(
        process_image,
        inputs=[
            input_image,
            detector_selector,
            recognizer_selector,
            corrector_selector,
            target_size,
            score_thresh,
            expand_ratio_w,
            expand_ratio_h,
            mask_threshold,
            apply_threshold,
            max_edits,
            region_preparer_selector,
            recognition_batch_size,
        ],
        outputs=[output_image, text_before, text_after, stats_display],
    )

    btn_save_recognition.click(
        save_recognition_json,
        inputs=[],
        outputs=[file_recognition],
    )

    btn_save_correction.click(
        save_correction_json,
        inputs=[],
        outputs=[file_correction],
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gradio demo with local OCR bundles")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    demo.queue(default_concurrency_limit=1).launch(
        share=args.share,
        server_port=args.port,
        **(_theme_options if _gradio6 else {}),
    )
