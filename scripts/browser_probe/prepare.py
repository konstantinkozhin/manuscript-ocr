"""Build a static browser app; serving it requires no Python inference server."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def add_folder(source, dest, catalog, base_url, name=None, manifest=None, demo=False):
    manifest = manifest or json.loads((source / 'model.json').read_text(encoding='utf-8'))
    identifier = name or f'local_{len(catalog["models"])}'
    target = dest / 'models' / identifier
    target.mkdir(parents=True, exist_ok=True)
    entry = {**manifest, 'display_name': source.name, 'demo': demo, 'artifacts': {}}
    for role, artifact in manifest['artifacts'].items():
        if artifact is None:
            entry['artifacts'][role] = None
            continue
        filename = artifact['filename']
        path = (source / filename).resolve()
        path.relative_to(source.resolve())
        if not path.is_file() and artifact.get('required', True) is False:
            continue
        # Registry artifacts have flat filenames, matching the library catalog.
        if Path(filename).name != filename:
            raise ValueError('For served presets, artifact filenames must be flat')
        shutil.copyfile(path, target / filename)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        entry['artifacts'][role] = {**artifact, 'size': path.stat().st_size,
            'sha256': digest.hexdigest(),
            'urls': [base_url.rstrip('/') + f'/models/{identifier}/{filename}']}
    (target / 'model.json').write_text(json.dumps(entry, ensure_ascii=False, indent=2))
    catalog['models'][identifier] = entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/private/tmp/manuscript-pyodide-check'))
    parser.add_argument('--ort-dist', type=Path, required=True, help='onnxruntime-web 1.30.0 dist directory')
    parser.add_argument('--base-url', default='http://127.0.0.1:8014/')
    parser.add_argument('--model-folder', type=Path, action='append', default=[], help='Optional model.json bundle to serve as a preset')
    parser.add_argument('--detector-dir', type=Path, help='Optional PP-OCRv6 Tiny sample bundle')
    parser.add_argument('--recognizer-dir', type=Path, help='Optional Kraken Tiny sample bundle')
    parser.add_argument('--checks', action='store_true', help='Generate tiny adapter regression graphs (requires onnx)')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    dest = args.output.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='manuscript-browser-wheel-') as build_dir:
        subprocess.run([sys.executable, '-m', 'pip', 'wheel', str(root), '--no-deps', '--no-build-isolation', '-w', build_dir], check=True)
        built_wheel = next(Path(build_dir).glob('manuscript_ocr-*.whl'))
        wheel = dest / built_wheel.name
        shutil.copyfile(built_wheel, wheel)
    for name in ['index.html', 'style.css', 'app.mjs', 'full_pipeline.py', 'app.worker.mjs', 'browser_checks.py']:
        shutil.copyfile(Path(__file__).with_name(name), dest / name)
    shutil.copyfile(root / 'src/manuscript/runtimes/bridge.mjs', dest / 'bridge.mjs')
    ort_dest = dest / 'ort'
    ort_dest.mkdir(exist_ok=True)
    for name in ['ort.webgpu.min.mjs', 'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm', 'ort-wasm-simd-threaded.asyncify.mjs', 'ort-wasm-simd-threaded.asyncify.wasm']:
        shutil.copyfile(args.ort_dist / name, ort_dest / name)
    catalog = json.loads((root / 'registry.json').read_text())
    for folder in args.model_folder:
        add_folder(folder, dest, catalog, args.base_url)
    for folder, cls, filename, name in [
        (args.detector_dir, 'PPOCR', 'ppocrv6_tiny_det', 'demo_detector'),
        (args.recognizer_dir, 'PPOCRRec', 'kraken-ppocrv6-tiny', 'demo_recognizer'),
    ]:
        if folder:
            artifacts = {'weights': {'filename': filename + '.fp32.onnx'},
                         'config': {'filename': filename + '.fp32.json'}}
            for notice in ['LICENSE.txt', 'NOTICE.txt']:
                if (folder / notice).is_file():
                    artifacts['license' if notice == 'LICENSE.txt' else 'notice'] = {'filename': notice}
            add_folder(folder, dest, catalog, args.base_url, name=name, demo=True,
                       manifest={'model_classes': [cls], 'artifacts': artifacts})
    (dest / 'catalog.json').write_text(json.dumps(catalog, ensure_ascii=False))
    from PIL import Image, ImageDraw, ImageFont
    image = Image.new('RGB', (950, 340), 'white')
    font_path = Path('/System/Library/Fonts/Supplemental/Arial.ttf')
    font = ImageFont.truetype(str(font_path), 40) if font_path.exists() else ImageFont.load_default()
    draw = ImageDraw.Draw(image)
    for y, line in zip([50, 130, 210], ['Это пример документа.', 'Распознавание текста в браузере.', 'Проверка полного пайплайна 123.']):
        draw.text((40, y), line, fill='black', font=font)
    image.save(dest / 'sample.png')
    if args.checks:
        prepare_checks(dest, args.base_url)
    package_metadata = args.ort_dist.parent / 'package.json'
    ort_version = json.loads(package_metadata.read_text()).get('version', 'unknown') if package_metadata.is_file() else 'unknown'
    (dest / 'probe-config.json').write_text(json.dumps({'wheel': wheel.name + '?v=' + hashlib.sha256(wheel.read_bytes()).hexdigest()[:16],
                                                   'ort_version': ort_version}))
    print(f'Browser app ready: {dest}\nCatalog: {len(catalog["models"])} models\nServe at {args.base_url}')


def prepare_checks(dest, base_url):
    import onnx
    import numpy as np
    tensor = onnx.numpy_helper.from_array(np.array([[2,0],[0,3]], dtype=np.float32), name='weight')
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('MatMul', ['x','weight'], ['y'])], 'external-test',
        [onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1,2])],
        [onnx.helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [1,2])], [tensor],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid('',17)])
    model.ir_version = 9
    (dest / 'external.onnx.data').unlink(missing_ok=True)
    onnx.save_model(model, str(dest / 'external.onnx'), save_as_external_data=True, all_tensors_to_one_file=True, location='external.onnx.data', size_threshold=0)
    artifacts = {}
    for role, filename in [('weights','external.onnx'), ('external_data','external.onnx.data')]:
        value = (dest / filename).read_bytes()
        artifacts[role] = {'filename':filename, 'size':len(value), 'sha256':hashlib.sha256(value).hexdigest(), 'urls':[base_url.rstrip('/') + '/' + filename]}
    (dest / 'selfcheck-registry.json').write_text(json.dumps({'schema_version':1, 'models':{'external_test':{'model_classes':['Test'], 'artifacts':artifacts}}}))
    tensor_cases = [
        (onnx.TensorProto.FLOAT, []), (onnx.TensorProto.FLOAT, [0, 2]),
        (onnx.TensorProto.INT64, [2]), (onnx.TensorProto.BOOL, [2]),
        (onnx.TensorProto.FLOAT16, [4]), (onnx.TensorProto.FLOAT, [3, 2]),
    ]
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node('Identity', [f'x{i}'], [f'y{i}']) for i in range(len(tensor_cases))],
        'tensor-conversion-test',
        [onnx.helper.make_tensor_value_info(f'x{i}', dtype, shape) for i, (dtype, shape) in enumerate(tensor_cases)],
        [onnx.helper.make_tensor_value_info(f'y{i}', dtype, shape) for i, (dtype, shape) in enumerate(tensor_cases)],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid('', 17)])
    model.ir_version = 9
    onnx.save_model(model, str(dest / 'tensor-check.onnx'))

if __name__ == '__main__':
    main()
