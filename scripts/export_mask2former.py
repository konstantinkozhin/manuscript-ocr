"""Adapted from sherstpasha/mask2former_train_text_line/scripts/export_mask2former_onnx.py.

Training dependencies are imported only when exporting a local checkpoint.
"""
from pathlib import Path


def export(source, output, *, image_size=1024, opset=20, fp16=False,
           region_classes=None, **config_options):
    from manuscript.detectors._mask2former import Mask2Former
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if output.suffix.lower() != '.onnx':
        raise ValueError('output must be an ONNX file path')
    if isinstance(opset, bool) or not isinstance(opset, int) or opset <= 0:
        raise ValueError('opset must be a positive integer')
    # Validate config options before loading a checkpoint or producing any files.
    config = Mask2Former.create_config(image_size=image_size, region_classes=region_classes,
                                      **config_options)
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation
    except ImportError as exc:
        raise ImportError('Mask2Former export requires torch and transformers; '
                          'install them in the export environment') from exc
    processor = AutoImageProcessor.from_pretrained(source, local_files_only=True)
    if (not getattr(processor, 'do_rescale', False)
            or abs(float(processor.rescale_factor) - 1 / 255) > 1e-8
            or not getattr(processor, 'do_normalize', False)):
        raise ValueError('Mask2Former inference requires rescale=1/255 and normalized RGB inputs')
    config_options.setdefault('image_mean', processor.image_mean)
    config_options.setdefault('image_std', processor.image_std)
    config = Mask2Former.create_config(image_size=image_size, region_classes=region_classes,
                                      **config_options)
    dtype = torch.float16 if fp16 else torch.float32
    model = AutoModelForUniversalSegmentation.from_pretrained(
        source, local_files_only=True).cpu().eval().to(dtype=dtype)
    # Default mapping is deliberately single-class. Multiclass checkpoints must
    # describe every foreground class explicitly rather than guessing semantics.
    expected_classes = set(range(model.config.num_labels))
    if set(map(int, config['region_classes'])) != expected_classes:
        raise ValueError('region_classes must map every foreground class in the checkpoint')

    class Wrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values, pixel_mask):
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            return outputs.class_queries_logits, outputs.masks_queries_logits

    wrapper = Wrapper(model).eval()
    pixels = torch.randn(1, 3, image_size, image_size, dtype=dtype)
    pixel_mask = torch.ones(1, image_size, image_size, dtype=torch.long)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper, (pixels, pixel_mask), str(output),
        input_names=['pixel_values', 'pixel_mask'],
        output_names=['class_queries_logits', 'masks_queries_logits'],
        dynamic_axes={name: {0: 'batch'} for name in
                      ('pixel_values', 'pixel_mask', 'class_queries_logits', 'masks_queries_logits')},
        opset_version=opset, dynamo=True,
    )
    import onnx
    # Path-based checking resolves external data relative to the ONNX file.
    onnx.checker.check_model(str(output))
    Mask2Former.create_config(image_size=image_size, region_classes=region_classes,
                             output_path=output.with_suffix('.json'), **config_options)
    return output


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--region-classes", type=Path, help="JSON file mapping class IDs to Page element fields")
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    import json
    mapping = json.loads(args.region_classes.read_text(encoding="utf-8")) if args.region_classes else None
    export(args.source, args.output, image_size=args.image_size, fp16=args.fp16, region_classes=mapping)


if __name__ == "__main__":
    main()
