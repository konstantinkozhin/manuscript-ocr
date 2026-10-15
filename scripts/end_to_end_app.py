"""Run with env/bin/python scripts/end_to_end_app.py."""

import time
import json
from functools import partial
from pathlib import Path

import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image

from manuscript import CharLM, Pipeline, Stack, Element
from manuscript.detectors import EAST, YOLO, RTDETR, DocUFCN
from manuscript.detectors._mask2former import Mask2Former
from manuscript.recognizers import TRBA, PPOCRRec, TrOCR, MajorityVotingRecognizer, MedoidRecognizer
from manuscript.detectors._ppocr import PPOCR
from manuscript.detectors._rfdetr import RFDETR
from manuscript.layouts import SimpleSorting, ReadingSorting
from manuscript.utils import visualize_page

ort.set_default_logger_severity(3)  # Show ONNX Runtime errors, suppress warnings.


DETECTOR_MODELS = ["mask2former_line_v0_prev", "yolo26x_obb_text_g1", "yolo26s_obb_text_g1", "east_50_g1", "east_50_yenisei_gov_reports_g1"]
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

DETECTOR_DEFAULTS["east_50_yenisei_gov_reports_g1"] = dict(DETECTOR_DEFAULTS["east_50_g1"])

LOCAL_MODEL = "Выбрать папку…"
NO_RECOGNIZER = "Без распознавателя"
NO_CORRECTOR = "Без корректора"
NO_LAYOUT = "Без лейаута"
RECOGNIZER_MODELS.insert(0, NO_RECOGNIZER)
CORRECTOR_MODELS.insert(0, NO_CORRECTOR)
LAYOUT_MODELS = {NO_LAYOUT: None, "ReadingSorting": ReadingSorting, "SimpleSorting": SimpleSorting}
MODEL_CLASSES = {
    "detector": {"EAST": EAST, "YOLO": YOLO, "Mask2Former": Mask2Former, "PPOCR": PPOCR, "RFDETR": RFDETR, "RTDETR": RTDETR, "DocUFCN": DocUFCN},
    "recognizer": {"TRBA": TRBA, "PPOCRRec": PPOCRRec, "TrOCR": TrOCR},
    "corrector": {"CharLM": CharLM},
}


def load_model_folder(folder, stage):
    root = Path(folder).expanduser().resolve()
    manifest = json.loads((root / "model.json").read_text(encoding="utf-8"))
    classes = [name for name in manifest["model_classes"] if name in MODEL_CLASSES[stage]]
    if len(classes) != 1:
        raise ValueError(f"Папка не содержит однозначную модель для этапа {stage}")
    allowed = {"weights", "config", "charset", "vocab", "lexicon", "decoder", "decoder_with_past", "tokenizer"}
    kwargs = {}
    for role, artifact in manifest["artifacts"].items():
        if role not in allowed or artifact is None:
            continue
        path = (root / artifact["filename"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Файл модели должен находиться внутри выбранной папки")
        if not path.is_file():
            raise FileNotFoundError(path)
        kwargs[role] = str(path)
    required = {"weights", "vocab"} if stage == "corrector" else {"weights", "config"}
    if classes[0] == "TRBA":
        required.add("charset")
    if required - kwargs.keys():
        raise ValueError(f"Не указаны артефакты: {sorted(required - kwargs.keys())}")
    return MODEL_CLASSES[stage][classes[0]], kwargs


def choose_model_folder(current):
    import sys
    if sys.platform == "darwin":
        import subprocess
        result = subprocess.run(
            ["osascript", "-e", 'POSIX path of (choose folder with prompt "Выберите папку модели")'],
            capture_output=True, text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else current
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    try:
        return filedialog.askdirectory(initialdir=current or str(Path.home() / "Desktop" / "модели_инференса")) or current
    finally:
        root.destroy()


def selection_values(selection):
    if isinstance(selection, str):
        return [selection]
    return list(selection or [])


def selected_models(selection):
    """Normalize scalar selections and ordered multiselect values."""
    return list(dict.fromkeys(model for model in selection_values(selection)
                              if model not in (NO_RECOGNIZER, NO_CORRECTOR, NO_LAYOUT, LOCAL_MODEL)))


def normalize_model_selection(current, previous, disabled):
    current = selection_values(current)
    if disabled in current and disabled not in selection_values(previous):
        return [disabled]
    if selected_models(current) or LOCAL_MODEL in current:
        return [model for model in current if model != disabled]
    return current


def choose_folder_and_model(current, previous, available, stage):
    multiple = isinstance(current, (list, tuple))
    current, previous = selection_values(current), selection_values(previous)
    original = current.copy()
    disabled = {"recognizer": NO_RECOGNIZER, "corrector": NO_CORRECTOR}.get(stage)
    # Selecting "without" clears the models; selecting a model enables the stage.
    current = normalize_model_selection(current, previous, disabled)
    if LOCAL_MODEL in current:
        current.remove(LOCAL_MODEL)
        if not multiple:
            current = previous
        presets = {"detector": DETECTOR_MODELS, "recognizer": RECOGNIZER_MODELS,
                   "corrector": CORRECTOR_MODELS}[stage]
        last_folder = next((model for model in reversed(available)
                            if model not in presets and model != LOCAL_MODEL), "")
        path = choose_model_folder(last_folder)
        if path:
            try:
                load_model_folder(path, stage)
            except (ValueError, KeyError, OSError, TypeError) as exc:
                gr.Warning(f"Не удалось загрузить папку модели: {exc}")
            else:
                available = list(dict.fromkeys([*available[:-1], path, LOCAL_MODEL]))
                current = list(dict.fromkeys([*current, path])) if multiple else [path]
        if not current and disabled:
            current = [disabled]
    value = current if multiple else (current[0] if current else None)
    # Ordinary selection is already reflected in the browser. Writing it back
    # can overwrite a newer choice made while the callback was running.
    update = (gr.update(choices=available, value=value, multiselect=multiple)
              if current != original or LOCAL_MODEL in original else gr.update())
    return update, current, available


def pack_models(models):
    if len(models) > 1:
        return Stack(models)
    return models[0] if models else None


def models_for_mode(selection, mode):
    if mode not in (None, "single", "stack", "voting", "medoid"):
        raise ValueError(f"Неизвестный режим моделей: {mode}")
    models = selected_models(selection)
    return models[:1] if mode == "single" else models


def create_detector(model, target_size, score_thresh, expand_ratio_w, expand_ratio_h):
    if model not in DETECTOR_MODELS:
        detector_class, options = load_model_folder(model, "detector")
        if detector_class is Mask2Former:
            options["score_threshold"] = score_thresh
        elif detector_class is PPOCR:
            options["box_threshold"] = score_thresh
        else:
            options["score_thresh"] = score_thresh
        if detector_class is EAST:
            options.update(target_size=int(target_size), expand_ratio_w=expand_ratio_w,
                           expand_ratio_h=expand_ratio_h)
        return detector_class(**options)
    if model == "mask2former_line_v0_prev":
        return Mask2Former(weights=model, score_threshold=score_thresh)
    if model.startswith("east_"):
        return EAST(weights=model, target_size=int(target_size), score_thresh=score_thresh,
                    expand_ratio_w=expand_ratio_w, expand_ratio_h=expand_ratio_h)
    # Each YOLO keeps its configured image size, including in a mixed Stack.
    return YOLO(weights=model, target_size=DETECTOR_DEFAULTS[model]["target_size"],
                score_thresh=score_thresh)


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
    layout_model="ReadingSorting",
    layout_mode="containment", infer_columns=True, containment_overlap=0.5,
    row_overlap=0.5, simple_columns=True, max_splits=10,
    recognition_override=False, recognition_levels=("word",), recognition_batch=0,
    region_preparer="bbox",
    detector_mode=None, recognizer_mode=None, layout_execution_mode=None, corrector_mode=None,
):
    detector_names = models_for_mode(detector_model, detector_mode)
    recognizer_names = models_for_mode(recognizer_model, recognizer_mode)
    corrector_names = models_for_mode(corrector_model, corrector_mode)
    if not detector_names:
        raise ValueError("Выберите хотя бы один детектор")
    detector = pack_models([
        create_detector(model, target_size, score_thresh, expand_ratio_w, expand_ratio_h)
        for model in detector_names
    ])

    recognition_options = {"region_preparer": region_preparer}
    if recognition_override:
        if not recognition_levels:
            raise ValueError("Выберите хотя бы один уровень распознавания")
        recognition_options["recognition_levels"] = list(recognition_levels)
    if recognition_batch:
        recognition_options["batch_size"] = int(recognition_batch)
    recognizers = []
    for model in recognizer_names:
        if model in RECOGNIZER_MODELS:
            recognizer_class, options = TRBA, {"weights": model}
        else:
            recognizer_class, options = load_model_folder(model, "recognizer")
        recognizers.append(recognizer_class(**{**options, **recognition_options}))
    ensembles = {"voting": MajorityVotingRecognizer, "medoid": MedoidRecognizer}
    recognizer = (ensembles[recognizer_mode](recognizers)
                  if recognizer_mode in ensembles and recognizers else pack_models(recognizers))
    correctors = []
    for model in corrector_names:
        if model in CORRECTOR_MODELS:
            corrector_class, options = CharLM, {"weights": model}
        else:
            corrector_class, options = load_model_folder(model, "corrector")
        correctors.append(corrector_class(
            **options, mask_threshold=mask_threshold,
            apply_threshold=apply_threshold, max_edits=int(max_edits),
        ))
    corrector = pack_models(correctors)

    layouts = []
    for model in models_for_mode(layout_model, layout_execution_mode):
        if model == "ReadingSorting":
            layouts.append(ReadingSorting(mode=layout_mode, infer_columns=infer_columns,
                                          containment_overlap=containment_overlap, row_overlap=row_overlap))
        elif model == "SimpleSorting":
            layouts.append(SimpleSorting(use_columns=simple_columns, max_splits=int(max_splits)))
        else:
            raise ValueError(f"Неизвестный лейаут: {model}")
    layout = pack_models(layouts)

    return Pipeline(
        detector=detector,
        layout=layout,
        recognizer=recognizer,
        corrector=corrector,
    )


def detector_settings(selection):
    if selection in DETECTOR_DEFAULTS:
        return dict(DETECTOR_DEFAULTS[selection])
    detector_class, artifacts = load_model_folder(selection, "detector")
    config = json.loads(Path(artifacts["config"]).read_text(encoding="utf-8"))
    settings = {"target_size": None, "expand_ratio_w": 1.0, "expand_ratio_h": 1.0,
                "is_east": detector_class is EAST, "is_mask2former": detector_class is Mask2Former}
    if detector_class is EAST:
        settings.update(DETECTOR_DEFAULTS["east_50_g1"])
    elif detector_class is YOLO:
        settings.update(target_size=YOLO._parse_imgsz(config.get("imgsz", YOLO.default_target_size)),
                        score_thresh=0.1)
    elif detector_class is Mask2Former:
        settings.update(target_size=config["image_size"], score_thresh=config["score_threshold"])
    elif detector_class is PPOCR:
        settings.update(target_size=config["preprocess"]["limit_side_len"],
                        score_thresh=config["postprocess"]["box_threshold"])
    elif detector_class in (RFDETR, RTDETR):
        settings.update(target_size=config.get("target_size"), score_thresh=config.get("score_thresh", 0.5))
    elif detector_class is DocUFCN:
        settings.update(target_size=config["target_size"], score_thresh=config.get("score_thresh", 0.1))
    return settings


def update_detector_controls(detector_model):
    models = selected_models(detector_model)
    if not models:
        return (gr.update(),) * 4
    configs = [detector_settings(model) for model in models]
    cfg = configs[0]
    east_cfg = next((item for item in configs if item["is_east"]), None)
    size_cfg = east_cfg or cfg
    is_east = east_cfg is not None
    size = gr.update(visible=size_cfg["target_size"] is not None, interactive=is_east,
                     label="Размер изображения" if is_east else "Размер изображения (из конфигурации модели)")
    if size_cfg["target_size"] is not None:
        size["value"] = size_cfg["target_size"]
    return (
        size,
        gr.update(value=cfg["score_thresh"]),
        gr.update(value=size_cfg["expand_ratio_w"], visible=is_east, interactive=is_east),
        gr.update(value=size_cfg["expand_ratio_h"], visible=is_east, interactive=is_east),
    )


def count_words_in_page(page):
    if page is None:
        return 0

    return sum(
        1 for region in page.descendants
        if region.type == "word" and getattr(region, "text", None)
    )


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
    layout_model="ReadingSorting",
    layout_mode="containment", infer_columns=True, containment_overlap=0.5,
    row_overlap=0.5, simple_columns=True, max_splits=10,
    recognition_override=False, recognition_levels=("word",), recognition_batch=0,
    region_preparer="bbox",
    detector_mode=None, recognizer_mode=None, layout_execution_mode=None, corrector_mode=None,
):
    if image is None:
        return None, "", ""

    try:
        request_start = time.perf_counter()
        setup_start = request_start
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
            layout_model, layout_mode, infer_columns, containment_overlap,
            row_overlap, simple_columns, max_splits,
            recognition_override, recognition_levels, recognition_batch, region_preparer,
            detector_mode, recognizer_mode, layout_execution_mode, corrector_mode,
        )

        setup_time = time.perf_counter() - setup_start
        result = pipeline.predict(image)
        visualization_start = time.perf_counter()
        vis_image = visualize_page(
            image, result["page"], max_size=1440, upscale=True,
            show_labels=any(node.label or node.type not in Element.registry
                            for node in result["page"].descendants),
        )
        visualization_time = time.perf_counter() - visualization_start

        text = pipeline.get_text(result["page"])
        elapsed_time = time.perf_counter() - request_start
        pipeline_time = pipeline.last_timings["total"]
        word_count = count_words_in_page(result["page"])
        pages_per_sec = 1.0 / pipeline_time if pipeline_time > 0 else 0.0
        words_per_sec = word_count / pipeline_time if pipeline_time > 0 else 0.0

        stats_text = (
            f"Всего: {elapsed_time:.2f} сек | Пайплайн: {pipeline_time:.2f} сек\n"
            f"Подготовка моделей: {setup_time:.2f} сек | Визуализация: {visualization_time:.2f} сек\n"
        )
        timings = pipeline.last_timings
        stage_names = [("load_image", "Загрузка изображения"), ("detector", "Детекция"),
                       ("layout", "Лейаут"), ("recognizer", "Распознавание"),
                       ("corrector", "Коррекция")]
        stats_text += "\n".join(
            f"{label}: {timings[key]:.3f} сек" if key in timings else f"{label}: выключено"
            for key, label in stage_names if key != "load_image" or key in timings
        )
        counts = {}
        for node in result["page"].descendants:
            counts[node.type] = counts.get(node.type, 0) + 1
        stats_text += "\nОбъекты: " + (", ".join(f"{kind}: {count}" for kind, count in counts.items()) or "0")
        stats_text += (
            f"\nСкорость пайплайна: "
            f"{pages_per_sec:.2f} стр/сек | "
            f"{words_per_sec:.1f} слов/сек"
        )

        if isinstance(vis_image, np.ndarray):
            vis_image = Image.fromarray(vis_image)

        return vis_image, text, stats_text

    except Exception as e:
        error_msg = f"Ошибка: {e}"
        return None, "", error_msg


def update_layout_parameters(selection):
    models = selected_models(selection)
    enabled = bool(models)
    return (
        parameter_section(enabled),
        *[gr.update(visible="ReadingSorting" in models, interactive=enabled) for _ in range(4)],
        *[gr.update(visible="SimpleSorting" in models, interactive=enabled) for _ in range(2)],
    )


def choose_layouts(current, previous):
    multiple = isinstance(current, (list, tuple))
    original = selection_values(current)
    current = normalize_model_selection(current, previous, NO_LAYOUT)
    value = current if multiple else (current[0] if current else None)
    update = (gr.update(value=value, multiselect=multiple)
              if current != original else gr.update())
    return update, current


def toggle_model_mode(mode, selection):
    multiple = mode != "single"
    values = selection_values(selection)
    if not multiple:
        values = values[:1]
    return gr.update(multiselect=multiple,
                     value=values if multiple else (values[0] if values else None)), values


def parameter_section(enabled):
    return gr.update(open=False, elem_classes=[] if enabled else ["disabled-parameters"])


default_detector = "yolo26x_obb_text_g1"
default_cfg = DETECTOR_DEFAULTS[default_detector]

with gr.Blocks(title="OCR Pipeline", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Manuscript Demo")
    selector_states = {}

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Изображение", type="pil")

            def model_selector(stage, label, choices, default):
                available_choices = [*choices, LOCAL_MODEL]
                selector = gr.Dropdown(choices=available_choices, value=default, label=label,
                                       multiselect=False)
                previous = gr.State([default])
                selector_states[stage] = previous
                available = gr.State(available_choices)
                selector.input(partial(choose_folder_and_model, stage=stage),
                               inputs=[selector, previous, available],
                               outputs=[selector, previous, available],
                               trigger_mode="always_last")
                return selector

            with gr.Group():
                detector_mode = gr.Radio(choices=[("Одна модель", "single"), ("Stack", "stack")],
                                         value="single", label="Режим детектора")
                detector_selector = model_selector("detector", "Детектор", DETECTOR_MODELS, default_detector)
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

            with gr.Group():
                recognizer_mode = gr.Radio(
                    choices=[("Одна модель", "single"), ("Stack", "stack"),
                             ("MajorityVotingRecognizer", "voting"), ("MedoidRecognizer", "medoid")],
                    value="single", label="Режим распознавателя",
                )
                recognizer_selector = model_selector("recognizer", "Распознаватель", RECOGNIZER_MODELS, "trba_lite_g2")
                with gr.Accordion("Параметры распознавателя", open=False) as recognizer_parameters:
                    recognition_override = gr.Checkbox(value=False, label="Переопределить уровни из конфига")
                    recognition_levels = gr.Dropdown(
                        choices=[("Слова", "word"), ("Строки", "line"), ("Блоки", "block")],
                        value=["word"], multiselect=True, label="Уровни распознавания", visible=False,
                    )
                    region_preparer = gr.Dropdown(
                        choices=[("Прямоугольная обрезка", "bbox"),
                                 ("Полигон с маской", "polygon_mask"),
                                 ("Выпрямление четырёхугольника", "quad_warp")],
                        value="bbox", label="Подготовка области",
                    )
                    recognition_batch = gr.Number(value=0, minimum=0, precision=0,
                                                   label="Размер пакета (0 — настройка модели)")
                    recognition_override.change(lambda enabled: gr.update(visible=enabled),
                                                inputs=[recognition_override], outputs=[recognition_levels])
                recognizer_selector.change(
                    lambda models: [parameter_section(bool(selected_models(models))),
                                    *[gr.update(interactive=bool(selected_models(models)))] * 4],
                    inputs=[recognizer_selector],
                    outputs=[recognizer_parameters, recognition_override, recognition_levels, recognition_batch, region_preparer],
                )

            with gr.Group():
                layout_execution_mode = gr.Radio(
                    choices=[("Одна модель", "single"), ("Stack", "stack")],
                    value="single", label="Режим лейаута",
                )
                layout_selector = gr.Dropdown(
                    choices=list(LAYOUT_MODELS),
                    value="ReadingSorting", multiselect=False,
                    label="Лейаут",
                )
                previous_layouts = gr.State(["ReadingSorting"])
                selector_states["layout"] = previous_layouts
                layout_selector.input(choose_layouts, inputs=[layout_selector, previous_layouts],
                                      outputs=[layout_selector, previous_layouts],
                                      trigger_mode="always_last")

                with gr.Accordion("Параметры лейаута", open=False) as layout_parameters:
                    with gr.Column() as reading_parameters:
                        layout_mode = gr.Radio(choices=[("Только порядок", "sort"),
                                                        ("Вложенность и порядок", "containment")],
                                               value="containment", label="Режим")
                        infer_columns = gr.Checkbox(
                            value=True, label="Учитывать колонки",
                            info="В режиме вложенности создаёт блоки колонок, если исходных блоков нет.",
                        )
                        containment_overlap = gr.Slider(0.1, 1, value=0.5, step=0.05,
                                                        label="Доля площади внутри родителя")
                        row_overlap = gr.Slider(0.1, 1, value=0.5, step=0.05,
                                                label="Вертикальное перекрытие для ряда")
                    with gr.Column() as simple_parameters:
                        simple_columns = gr.Checkbox(value=True, label="Разделять на колонки", visible=False)
                        max_splits = gr.Slider(1, 30, value=10, step=1, label="Максимум разделений", visible=False)
                layout_selector.change(
                    update_layout_parameters,
                    inputs=[layout_selector],
                    outputs=[layout_parameters, layout_mode, infer_columns,
                             containment_overlap, row_overlap, simple_columns, max_splits],
                )

            with gr.Group():
                corrector_mode = gr.Radio(choices=[("Одна модель", "single"), ("Stack", "stack")],
                                          value="single", label="Режим корректора")
                corrector_selector = model_selector("corrector", "Корректор", CORRECTOR_MODELS, NO_CORRECTOR)
                with gr.Accordion("Параметры корректора", open=False,
                                  elem_classes=["disabled-parameters"]) as corrector_parameters:
                    mask_threshold = gr.Slider(
                        0.0, 0.5, value=0.05, step=0.01, label="Порог маскирования", interactive=False
                    )
                    apply_threshold = gr.Slider(
                        0.5, 1.0, value=0.95, step=0.01, label="Порог применения", interactive=False
                    )
                    max_edits = gr.Slider(
                        1, 10, value=1, step=1, label="Максимум правок", interactive=False
                    )

                corrector_selector.change(
                    lambda models: [parameter_section(bool(selected_models(models))),
                                    *[gr.update(interactive=bool(selected_models(models)))] * 3],
                    inputs=[corrector_selector], outputs=[corrector_parameters, mask_threshold, apply_threshold, max_edits],
                )

            btn = gr.Button("Распознать", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Визуализация", type="pil")
            final_text = gr.Textbox(label="Распознанный текст", lines=15, interactive=False)
            stats_display = gr.Textbox(label="Статистика", interactive=False)

    for stage, mode, selector in (
        ("detector", detector_mode, detector_selector),
        ("recognizer", recognizer_mode, recognizer_selector),
        ("layout", layout_execution_mode, layout_selector),
        ("corrector", corrector_mode, corrector_selector),
    ):
        mode.change(toggle_model_mode, inputs=[mode, selector],
                    outputs=[selector, selector_states[stage]])

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
            layout_selector,
            layout_mode, infer_columns, containment_overlap, row_overlap, simple_columns, max_splits,
            recognition_override, recognition_levels, recognition_batch, region_preparer,
            detector_mode, recognizer_mode, layout_execution_mode, corrector_mode,
        ],
        outputs=[output_image, final_text, stats_display],
    )

if __name__ == "__main__":
    demo.launch(
        css="""
            .disabled-parameters { pointer-events: none; opacity: 0.5; }
        """,
        js="""() => {
            const update = () => document.querySelectorAll('.disabled-parameters, [data-disabled-section]')
                .forEach(element => {
                    const disabled = element.classList.contains('disabled-parameters');
                    element.inert = disabled;
                    if (disabled) element.dataset.disabledSection = '';
                    else delete element.dataset.disabledSection;
                });
            new MutationObserver(update).observe(document.body,
                {subtree: true, childList: true, attributes: true, attributeFilter: ['class']});
            update();
        }""",
    )
