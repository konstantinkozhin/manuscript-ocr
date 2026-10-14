"""Export local Ultralytics detection/OBB weights and a Manuscript JSON config.

Requires ultralytics and onnx only for export. Runtime inference uses Manuscript
and ONNX Runtime. NMS is embedded to produce ready-to-use detection rows.
"""
import json
from pathlib import Path
import shutil


def export(source, output, *, imgsz=640, region_type="word", region_classes=None,
           opset=17, conf=0.05, iou=0.5, max_det=300):
    """Export batch-one FP32 ONNX without optional graph simplification.

    Single-class models default to words; pass region_type='line' for line
    detectors. Multiclass models require an explicit region_classes mapping.
    conf and iou are baked into export-time NMS. Returns weights/config paths.
    """
    from manuscript.detectors import YOLO

    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.suffix.lower() != ".onnx":
        raise ValueError("output must be an ONNX file path")
    for name, value in (("opset", opset), ("max_det", max_det)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not 0 <= conf <= 1 or not 0 <= iou <= 1:
        raise ValueError("conf and iou must be in [0, 1]")
    # Validate geometry and element types before loading weights.
    mapping = region_classes if region_classes is not None else {0: {"type": region_type}}
    config = YOLO.create_config(imgsz=imgsz, region_classes=mapping)
    try:
        import onnx
        from ultralytics import YOLO as UltralyticsYOLO
    except ImportError as exc:
        raise ImportError("YOLO export requires ultralytics and onnx") from exc

    model = UltralyticsYOLO(str(source))
    if model.task not in {"detect", "obb"}:
        raise ValueError("Manuscript YOLO supports detection and OBB exports")
    if region_classes is None:
        if len(model.names) != 1:
            raise ValueError("Multiclass models require region_classes")
        class_id = next(iter(model.names))
        config = YOLO.create_config(imgsz=config["imgsz"],
                                    region_classes={class_id: {"type": region_type}})
    if any(int(key) not in model.names for key in config["region_classes"]):
        raise ValueError("region_classes contains IDs absent from the source model")

    output.parent.mkdir(parents=True, exist_ok=True)
    exported = Path(model.export(format="onnx", imgsz=config["imgsz"], batch=1,
                                 device="cpu", opset=opset, nms=True, conf=conf,
                                 iou=iou, max_det=max_det, dynamic=False, simplify=False))
    if exported.resolve() != output:
        shutil.copy2(exported, output)
    onnx.checker.check_model(str(output))
    config_path = output.with_suffix(".json")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"weights": output, "config": config_path}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local Ultralytics checkpoint (.pt)")
    parser.add_argument("output", type=Path, help="Output ONNX file; JSON is saved alongside it")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--region-type", default="word", help="Page type for a single-class model")
    parser.add_argument("--region-classes", type=Path, help="JSON object mapping class IDs to element fields")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--conf", type=float, default=0.05, help="Minimum confidence baked into NMS")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()
    mapping = json.loads(args.region_classes.read_text(encoding="utf-8")) if args.region_classes else None
    paths = export(args.source, args.output, imgsz=args.imgsz, region_type=args.region_type,
                   region_classes=mapping, opset=args.opset, conf=args.conf,
                   iou=args.iou, max_det=args.max_det)
    for role, path in paths.items():
        print(f"{role}: {path}")


if __name__ == "__main__":
    main()
