import gradio as gr
from manuscript import CharLM, create_page_from_text


CORRECTOR_MODELS = ["prereform_charlm_g1", "modern_charlm_g1"]


def get_text_from_page(page):
    """Извлекает текст из Page объекта."""
    lines = []
    for block in page.blocks:
        for line in block.lines:
            words = [w.text for w in line.words if w.text]
            if words:
                lines.append(" ".join(words))
    return "\n".join(lines)


def create_diff_html(original_text, corrected_text):
    original_words = original_text.split()
    corrected_words = corrected_text.split()

    html_parts = []
    html_parts.append(
        '<div style="font-family: Arial, sans-serif; font-size: 16px; line-height: 1.8;">'
    )

    i, j = 0, 0

    while i < len(original_words) or j < len(corrected_words):
        if i < len(original_words) and j < len(corrected_words):
            if original_words[i] == corrected_words[j]:
                html_parts.append(f"{corrected_words[j]} ")
                i += 1
                j += 1
            else:
                html_parts.append(
                    f'<span style="background-color: #ffcccb; text-decoration: line-through; padding: 2px 4px; margin: 0 2px;">'
                    f"{original_words[i]}</span> "
                )
                html_parts.append(
                    f'<span style="background-color: #90EE90; padding: 2px 4px; margin: 0 2px; font-weight: bold;">'
                    f"{corrected_words[j]}</span> "
                )
                i += 1
                j += 1
        elif i < len(original_words):
            html_parts.append(
                f'<span style="background-color: #ffcccb; text-decoration: line-through; padding: 2px 4px; margin: 0 2px;">'
                f"{original_words[i]}</span> "
            )
            i += 1
        else:
            html_parts.append(
                f'<span style="background-color: #90EE90; padding: 2px 4px; margin: 0 2px; font-weight: bold;">'
                f"{corrected_words[j]}</span> "
            )
            j += 1

    html_parts.append("</div>")
    return "".join(html_parts)


def create_changes_list(original_text, corrected_text):
    original_words = original_text.split()
    corrected_words = corrected_text.split()

    changes = []
    i, j = 0, 0
    change_num = 0

    while i < len(original_words) or j < len(corrected_words):
        if i < len(original_words) and j < len(corrected_words):
            if original_words[i] != corrected_words[j]:
                change_num += 1
                changes.append(
                    f"{change_num}. '{original_words[i]}' → '{corrected_words[j]}'"
                )
            i += 1
            j += 1
        elif i < len(original_words):
            change_num += 1
            changes.append(f"{change_num}. '{original_words[i]}' → [удалено]")
            i += 1
        else:
            change_num += 1
            changes.append(f"{change_num}. [добавлено] → '{corrected_words[j]}'")
            j += 1

    if not changes:
        return "Изменений не обнаружено"

    return "\n".join(changes)


def correct_text(
    input_text,
    corrector_model,
    mask_threshold,
    apply_threshold,
    max_edits,
):
    """Исправляет текст с помощью корректора."""
    if not input_text or not input_text.strip():
        return "", "", "Введите текст для коррекции"

    try:
        # Создаем корректор
        corrector = CharLM(
            weights=corrector_model,
            mask_threshold=mask_threshold,
            apply_threshold=apply_threshold,
            max_edits=int(max_edits),
        )

        lines = input_text.strip().split("\n")
        page = create_page_from_text(lines)

        corrected_page = corrector.predict(page)

        corrected_text = get_text_from_page(corrected_page)

        diff_html = create_diff_html(input_text, corrected_text)

        changes_list = create_changes_list(input_text, corrected_text)

        return corrected_text, diff_html, changes_list

    except Exception as e:
        error_msg = f"Ошибка: {str(e)}"
        return "", error_msg, error_msg


with gr.Blocks(title="Корректор текста", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Корректор текста для рукописей")
    gr.Markdown(
        "Введите текст слева, а справа увидите исправленный вариант с выделением изменений"
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Исходный текст")
            input_text = gr.Textbox(
                label="Введите текст для коррекции",
                placeholder="Впишите текст здесь...",
                lines=10,
            )

            corrector_selector = gr.Dropdown(
                choices=CORRECTOR_MODELS,
                value="prereform_charlm_g1",
                label="Модель корректора",
            )

            with gr.Accordion("Параметры корректора", open=False):
                mask_threshold = gr.Slider(
                    0.0,
                    0.5,
                    value=0.05,
                    step=0.01,
                    label="Порог маскирования",
                    info="Минимальная вероятность для маскирования символа",
                )
                apply_threshold = gr.Slider(
                    0.5,
                    1.0,
                    value=0.95,
                    step=0.01,
                    label="Порог применения",
                    info="Минимальная вероятность для применения исправления",
                )
                max_edits = gr.Slider(
                    1,
                    10,
                    value=1,
                    step=1,
                    label="Максимум правок",
                    info="Максимальное количество правок на одно слово",
                )

            btn_correct = gr.Button("Исправить текст", variant="primary", size="lg")

        with gr.Column():
            gr.Markdown("### Исправленный текст")
            output_text = gr.Textbox(
                label="Исправленный текст",
                lines=10,
                interactive=False,
            )

            gr.Markdown("### Визуализация изменений")
            gr.Markdown("🔴 Красный = удалено | 🟢 Зеленый = добавлено/исправлено")
            diff_display = gr.HTML(
                label="Визуализация изменений",
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Список изменений")
            changes_list = gr.Textbox(
                label="Детали изменений",
                lines=8,
                interactive=False,
            )

    btn_correct.click(
        correct_text,
        inputs=[
            input_text,
            corrector_selector,
            mask_threshold,
            apply_threshold,
            max_edits,
        ],
        outputs=[output_text, diff_display, changes_list],
    )

    gr.Examples(
        examples=[
            ["Въ старыя времена жилъ одинъ царь"],
            ["Онъ былъ очень мудръ и справедливъ"],
            ["Народъ его любилъ и почиталъ"],
        ],
        inputs=[input_text],
        label="Примеры текстов",
    )


if __name__ == "__main__":
    demo.launch()
