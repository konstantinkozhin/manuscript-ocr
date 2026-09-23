import time
import tempfile

import gradio as gr
import numpy as np
from PIL import Image

from manuscript import CharLM, Pipeline
from manuscript.detectors import EAST, Mask2Former, YOLO
from manuscript.recognizers import TRBA


DETECTOR_MODELS = ["mask2former_line_v0_prev", "yolo26x_obb_text_g1", "yolo26s_obb_text_g1", "east_50_g1"]
CORRECTOR_MODELS = ["prereform_charlm_g1", "modern_charlm_g1"]
RECOGNIZER_MODELS = ["trba_lite_g2", "trba_lite_g1", "trba_base_g1"]

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

last_recognition_page = None
last_correction_page = None


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
):
    if detector_model == "mask2former_line_v0_prev":
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

    recognizer = TRBA(weights=recognizer_model)
    corrector = CharLM(
        weights=corrector_model,
        mask_threshold=mask_threshold,
        apply_threshold=apply_threshold,
        max_edits=int(max_edits),
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
            label="Размер изображения" if is_east else "Размер изображения (из конфигурации модели)" if is_mask2former else "Размер изображения (по модели)",
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


def highlight_differences(original, corrected):
    html = []
    i, j = 0, 0

    while i < len(original) or j < len(corrected):
        if i < len(original) and j < len(corrected):
            if original[i] == corrected[j]:
                if original[i] == "\n":
                    html.append("<br>")
                else:
                    html.append(corrected[j])
                i += 1
                j += 1
            else:
                html.append(
                    f'<span style="background-color: #90EE90; font-weight: bold;">{corrected[j]}</span>'
                )
                i += 1
                j += 1
        elif i < len(original):
            i += 1
        else:
            html.append(
                f'<span style="background-color: #90EE90; font-weight: bold;">{corrected[j]}</span>'
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
):
    global last_recognition_page, last_correction_page

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
        )

        start_time = time.time()
        _, vis_image = pipeline.predict(image, vis=True)
        elapsed_time = time.time() - start_time

        last_recognition_page = pipeline.last_recognition_page
        last_correction_page = pipeline.last_correction_page

        text_before = pipeline.get_text(last_recognition_page) if last_recognition_page else ""
        text_after = pipeline.get_text(last_correction_page) if last_correction_page else ""

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


default_detector = "yolo26x_obb_text_g1"
default_cfg = DETECTOR_DEFAULTS[default_detector]

with gr.Blocks(title="OCR Pipeline", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Manuscript Demo")

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
                    value="trba_lite_g2",
                    label="Распознаватель",
                )
                corrector_selector = gr.Dropdown(
                    choices=CORRECTOR_MODELS,
                    value="prereform_charlm_g1",
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

            with gr.Accordion("Параметры корректора", open=False):
                mask_threshold = gr.Slider(
                    0.0, 0.5, value=0.05, step=0.01, label="Порог маскирования"
                )
                apply_threshold = gr.Slider(
                    0.5, 1.0, value=0.95, step=0.01, label="Порог применения"
                )
                max_edits = gr.Slider(
                    1, 10, value=1, step=1, label="Максимум правок"
                )

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
    demo.launch(share=True)
