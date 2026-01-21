"""
Example: Pipeline with CharLM corrector.

This script demonstrates how to use the manuscript OCR pipeline
with the CharLM corrector for OCR post-correction.
"""

from pathlib import Path
from manuscript import Pipeline, CharLM, DummyCorrector


def main():
    # ==========================================================================
    # Example 1: Pipeline without corrector (detection + recognition only)
    # ==========================================================================
    print("=" * 60)
    print("Example 1: Pipeline without corrector")
    print("=" * 60)
    
    pipeline = Pipeline()
    
    # Run on test image
    image_path = "example/ocr_example_image.jpg"
    if Path(image_path).exists():
        result = pipeline.predict(image_path)
        page = result["page"]
        text = pipeline.get_text(page)
        print(f"Detected text:\n{text}\n")
    else:
        print(f"Test image not found: {image_path}")
        print("Creating dummy page for demonstration...\n")

    # ==========================================================================
    # Example 2: Pipeline with DummyCorrector (no-op correction)
    # ==========================================================================
    print("=" * 60)
    print("Example 2: Pipeline with DummyCorrector")
    print("=" * 60)
    
    corrector = DummyCorrector()
    pipeline_with_dummy = Pipeline(corrector=corrector)
    
    print(f"Pipeline configured with: {type(corrector).__name__}")
    print("DummyCorrector returns page unchanged (useful for testing)\n")

    # ==========================================================================
    # Example 3: Pipeline with CharLM corrector (requires trained model)
    # ==========================================================================
    print("=" * 60)
    print("Example 3: Pipeline with CharLM corrector")
    print("=" * 60)
    
    # CharLM without weights (will act as pass-through)
    charlm = CharLM()
    pipeline_with_charlm = Pipeline(corrector=charlm)
    
    print(f"Pipeline configured with: {type(charlm).__name__}")
    print("Note: Without trained weights, CharLM returns page unchanged\n")

    # ==========================================================================
    # Example 4: CharLM with trained model (if available)
    # ==========================================================================
    print("=" * 60)
    print("Example 4: CharLM with trained model")
    print("=" * 60)
    
    # Path to trained model (adjust as needed)
    model_dir = Path("exp_stage_a5")
    onnx_path = model_dir / "charlm.onnx"
    vocab_path = model_dir / "vocab.json"
    subs_path = model_dir / "substitutions.json"
    
    if onnx_path.exists() and vocab_path.exists():
        print(f"Loading CharLM from: {onnx_path}")
        charlm_trained = CharLM(
            weights=str(onnx_path),
            vocab=str(vocab_path),
            substitutions=str(subs_path) if subs_path.exists() else None,
            mask_threshold=0.05,
            apply_threshold=0.95,
            max_edits=1,
            sub_threshold=50,
        )
        
        pipeline_full = Pipeline(corrector=charlm_trained)
        print("Pipeline with trained CharLM ready!\n")
        
        # Run on test image if available
        if Path(image_path).exists():
            result = pipeline_full.predict(image_path)
            page = result["page"]
            text = pipeline_full.get_text(page)
            print(f"Corrected text:\n{text}\n")
    else:
        print(f"Trained model not found at: {onnx_path}")
        print("To train CharLM, use:")
        print("""
    from manuscript.correctors import CharLM
    
    CharLM.train(
        words_path="data/words.txt",
        text_path="data/texts.txt",
        pairs_path="data/pairs.csv",
        charset_path="data/charset.txt",
        exp_dir="exp_charlm",
        epochs=100,
    )
""")
        print("Then export to ONNX:")
        print("""
    CharLM.export(
        weights_path="exp_charlm/checkpoints/charlm_epoch_100.pt",
        vocab_path="exp_charlm/vocab.json",
        output_path="exp_charlm/charlm.onnx",
    )
""")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("=" * 60)
    print("Summary: Pipeline components")
    print("=" * 60)
    print("""
Pipeline(
    detector=EAST(),        # Text detection (returns Page with word polygons)
    recognizer=TRBA(),      # Text recognition (fills word.text)
    corrector=CharLM(),     # Text correction (corrects word.text)
)

Flow: image → detection → recognition → correction → Page

Each stage:
1. EAST: image → Page with empty words (polygon, detection_confidence)
2. TRBA: Page → Page with recognized text (text, recognition_confidence)  
3. CharLM: Page → Page with corrected text (corrected text)
""")


if __name__ == "__main__":
    main()
