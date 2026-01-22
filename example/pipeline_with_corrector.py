from manuscript import Pipeline, CharLM


def main():
    # Пути к модели CharLM (должны существовать)
    onnx_path = "C:\\Users\\USER\\c\\exp_stage_a5\\charlm.onnx"
    vocab_path = "C:\\Users\\USER\\c\\exp_stage_a5\\vocab.json"
    subs_path = "C:\\Users\\USER\\c\\exp_stage_a5\\substitutions.json"

    # Путь к изображению
    image_path = "example/ocr_example_image.jpg"

    # Инициализация CharLM
    corrector = CharLM(
        weights=onnx_path,
        vocab=vocab_path,
        substitutions=subs_path,
        mask_threshold=0.05,
        apply_threshold=0.95,
        max_edits=1,
        sub_threshold=50,
    )

    # Один pipeline
    pipeline = Pipeline(corrector=corrector)

    # Запуск OCR
    result = pipeline.predict(image_path)
    page = result["page"]
    text = pipeline.get_text(page)

    print(text)


if __name__ == "__main__":
    main()
