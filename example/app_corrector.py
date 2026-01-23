import time
import gradio as gr
import numpy as np
import torch
from PIL import Image
from manuscript import Pipeline, CharLM, visualize_page
from manuscript.detectors import EAST
from manuscript.recognizers import TRBA


def get_device_info():
    """Получить информацию об устройстве."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        return f"GPU: {device_name}"
    else:
        return "CPU"


# Доступные пресеты
CORRECTOR_MODELS = ["prereform_charlm_g1", "modern_charlm_g1"]
RECOGNIZER_MODELS = ["trba_lite_g1", "trba_base_g1"]


def create_pipeline(recognizer_model, corrector_model, 
                   target_size, score_thresh, expand_ratio_w, expand_ratio_h,
                   mask_threshold, apply_threshold, max_edits,
                   use_corrector=True):
    """Создать пайплайн с заданными параметрами."""
    detector = EAST(
        target_size=int(target_size),
        score_thresh=score_thresh,
        expand_ratio_w=expand_ratio_w,
        expand_ratio_h=expand_ratio_h,
    )
    
    recognizer = TRBA(weights=recognizer_model)
    
    corrector = None
    if use_corrector:
        corrector = CharLM(
            weights=corrector_model,
            mask_threshold=mask_threshold,
            apply_threshold=apply_threshold,
            max_edits=int(max_edits),
        )
    
    return Pipeline(detector=detector, recognizer=recognizer, corrector=corrector)


def get_text_from_page(page):
    """Извлечь текст из Page."""
    lines = []
    for block in page.blocks:
        for line in block.lines:
            words = [w.text for w in line.words if w.text]
            if words:
                lines.append(" ".join(words))
    return "\n".join(lines)


def count_words_in_page(page):
    """Подсчитать количество слов в Page."""
    count = 0
    for block in page.blocks:
        for line in block.lines:
            count += len([w for w in line.words if w.text])
    return count


def highlight_differences(original, corrected):
    """Подсветить исправления в тексте с корректором."""
    html = []
    i, j = 0, 0
    
    while i < len(original) or j < len(corrected):
        if i < len(original) and j < len(corrected):
            if original[i] == corrected[j]:
                if original[i] == '\n':
                    html.append('<br>')
                else:
                    html.append(corrected[j])
                i += 1
                j += 1
            else:
                # Показываем только исправленный символ, подсвеченный зелёным
                html.append(f'<span style="background-color: #90EE90; font-weight: bold;">{corrected[j]}</span>')
                i += 1
                j += 1
        elif i < len(original):
            # Символ удалён — пропускаем
            i += 1
        else:
            # Символ добавлен
            html.append(f'<span style="background-color: #90EE90; font-weight: bold;">{corrected[j]}</span>')
            j += 1
    
    return f'<div style="white-space: pre-wrap; font-family: monospace;">{"".join(html)}</div>'


def process_image(image, recognizer_model, corrector_model,
                  target_size, score_thresh, expand_ratio_w, expand_ratio_h,
                  mask_threshold, apply_threshold, max_edits):
    """Обработать изображение через пайплайн."""
    if image is None:
        return None, "", "", ""
    
    try:
        # Пайплайн без корректора (для текста до коррекции)
        pipeline_base = create_pipeline(
            recognizer_model, corrector_model,
            target_size, score_thresh, expand_ratio_w, expand_ratio_h,
            mask_threshold, apply_threshold, max_edits,
            use_corrector=False
        )
        result_base, _ = pipeline_base.predict(image, vis=True)
        page_base = result_base["page"]
        text_base = get_text_from_page(page_base)
        
        # Пайплайн с корректором — замер скорости
        pipeline_corrected = create_pipeline(
            recognizer_model, corrector_model,
            target_size, score_thresh, expand_ratio_w, expand_ratio_h,
            mask_threshold, apply_threshold, max_edits,
            use_corrector=True
        )
        
        # Замер времени полного пайплайна (с корректором)
        start_time = time.time()
        result_corrected, vis_image = pipeline_corrected.predict(image, vis=True)
        elapsed_time = time.time() - start_time
        
        page_corrected = result_corrected["page"]
        text_corrected = get_text_from_page(page_corrected)
        word_count = count_words_in_page(page_corrected)
        
        # Статистика скорости
        pages_per_sec = 1.0 / elapsed_time if elapsed_time > 0 else 0
        words_per_sec = word_count / elapsed_time if elapsed_time > 0 else 0
        device_info = get_device_info()
        
        stats_text = f"⏱ Время: {elapsed_time:.2f} сек | 📄 {pages_per_sec:.2f} стр/сек | 📝 {words_per_sec:.1f} слов/сек | 💻 {device_info}"
        
        # Конвертируем в PIL если numpy
        if isinstance(vis_image, np.ndarray):
            vis_image = Image.fromarray(vis_image)
        
        # Подсветка различий
        highlighted = highlight_differences(text_base, text_corrected)
        
        return vis_image, text_base, highlighted, stats_text
    
    except Exception as e:
        error_msg = f"Ошибка: {str(e)}"
        return None, error_msg, "", error_msg


with gr.Blocks(title="OCR Pipeline", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# OCR Pipeline с CharLM корректором")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Изображение", type="pil")
            
            with gr.Row():
                recognizer_selector = gr.Dropdown(
                    choices=RECOGNIZER_MODELS,
                    value="trba_lite_g1",
                    label="Распознаватель"
                )
                corrector_selector = gr.Dropdown(
                    choices=CORRECTOR_MODELS,
                    value="prereform_charlm_g1",
                    label="Корректор"
                )
            
            with gr.Accordion("Параметры детектора", open=False):
                target_size = gr.Slider(640, 2560, value=1280, step=64, label="Размер изображения")
                score_thresh = gr.Slider(0.1, 0.9, value=0.6, step=0.05, label="Порог уверенности")
                expand_ratio_w = gr.Slider(0.5, 3.0, value=1.4, step=0.1, label="Расширение по ширине")
                expand_ratio_h = gr.Slider(0.5, 3.0, value=1.5, step=0.1, label="Расширение по высоте")
            
            with gr.Accordion("Параметры корректора", open=False):
                mask_threshold = gr.Slider(0.0, 0.5, value=0.05, step=0.01, label="Порог маскирования")
                apply_threshold = gr.Slider(0.5, 1.0, value=0.95, step=0.01, label="Порог применения")
                max_edits = gr.Slider(1, 10, value=1, step=1, label="Максимум правок")
            
            btn = gr.Button("Распознать", variant="primary")
        
        with gr.Column():
            output_image = gr.Image(label="Визуализация", type="pil")
            stats_display = gr.Textbox(label="Статистика", interactive=False)
    
    with gr.Row():
        with gr.Column():
            text_before = gr.Textbox(label="Текст без корректора", lines=10)
        
        with gr.Column():
            text_after = gr.HTML(label="Текст с корректором (зелёный — исправлено)")
    
    btn.click(
        process_image,
        inputs=[input_image, recognizer_selector, corrector_selector, target_size, score_thresh, expand_ratio_w, expand_ratio_h, mask_threshold, apply_threshold, max_edits],
        outputs=[output_image, text_before, text_after, stats_display]
    )

if __name__ == "__main__":
    demo.launch()
