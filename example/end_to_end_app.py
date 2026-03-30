import time
import tempfile
import gradio as gr
import numpy as np
from PIL import Image
from manuscript import Pipeline, CharLM
from manuscript.detectors import EAST
from manuscript.recognizers import TRBA


CORRECTOR_MODELS = ["prereform_charlm_g1", "modern_charlm_g1"]
RECOGNIZER_MODELS = ["trba_lite_g1", "trba_lite_g2", "trba_base_g1"]

last_recognition_page = None
last_correction_page = None


def create_pipeline(
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
    detector = EAST(
        target_size=int(target_size),
        score_thresh=score_thresh,
        expand_ratio_w=expand_ratio_w,
        expand_ratio_h=expand_ratio_h,
    )
    recognizer = TRBA(weights=recognizer_model)
    corrector = CharLM(
        weights=corrector_model,
        mask_threshold=mask_threshold,
        apply_threshold=apply_threshold,
        max_edits=int(max_edits),
    )
    return Pipeline(detector=detector, recognizer=recognizer, corrector=corrector)


def get_text_from_page(page):
    lines = []
    for block in page.blocks:
        for line in block.lines:
            words = [w.text for w in line.words if w.text]
            if words:
                lines.append(" ".join(words))
    return "\n".join(lines)


def count_words_in_page(page):
    count = 0
    for block in page.blocks:
        for line in block.lines:
            count += len([w for w in line.words if w.text])
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

        text_before = get_text_from_page(last_recognition_page)
        text_after = get_text_from_page(last_correction_page)
        word_count = count_words_in_page(last_correction_page)

        pages_per_sec = 1.0 / elapsed_time if elapsed_time > 0 else 0
        words_per_sec = word_count / elapsed_time if elapsed_time > 0 else 0

        stats_text = f"Время: {elapsed_time:.2f} сек | {pages_per_sec:.2f} стр/сек | {words_per_sec:.1f} слов/сек"

        if isinstance(vis_image, np.ndarray):
            vis_image = Image.fromarray(vis_image)

        highlighted = highlight_differences(text_before, text_after)

        return vis_image, text_before, highlighted, stats_text

    except Exception as e:
        error_msg = f"Ошибка: {str(e)}"
        return None, error_msg, "", error_msg


def save_recognition_json():
    global last_recognition_page
    if last_recognition_page is None:
        return None
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        f.write(last_recognition_page.to_json())
        return f.name


def save_correction_json():
    global last_correction_page
    if last_correction_page is None:
        return None
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        f.write(last_correction_page.to_json())
        return f.name


with gr.Blocks(title="OCR Pipeline", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Manuscript Demo")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Изображение", type="pil")

            with gr.Row():
                recognizer_selector = gr.Dropdown(
                    choices=RECOGNIZER_MODELS,
                    value="trba_lite_g1",
                    label="Распознаватель",
                )
                corrector_selector = gr.Dropdown(
                    choices=CORRECTOR_MODELS,
                    value="prereform_charlm_g1",
                    label="Корректор",
                )

            with gr.Accordion("Параметры детектора", open=False):
                target_size = gr.Slider(
                    640, 2560, value=1280, step=64, label="Размер изображения"
                )
                score_thresh = gr.Slider(
                    0.1, 0.9, value=0.6, step=0.05, label="Порог уверенности"
                )
                expand_ratio_w = gr.Slider(
                    0.5, 3.0, value=1.4, step=0.1, label="Расширение по ширине"
                )
                expand_ratio_h = gr.Slider(
                    0.5, 3.0, value=1.5, step=0.1, label="Расширение по высоте"
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

    btn.click(
        process_image,
        inputs=[
            input_image,
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
    demo.launch()
