import * as ort from './ort/ort.webgpu.min.mjs';
import {registerOrtWeb} from './bridge.mjs';

let py, busy = false;
const uploads = new Map();
const log = message => self.postMessage({log: String(message)});
const python = async source => JSON.parse(await py.runPythonAsync(`json.dumps(${source}, ensure_ascii=False)`));
const safePath = value => {
  if (!value || value.startsWith('/') || value.includes('\\') || value.split('/').some(s => !s || s === '.' || s === '..')) {
    throw Error('Некорректный путь файла в папке модели.');
  }
  return value;
};
function closeUpload(upload) {
  if (upload.stream) py.FS.close(upload.stream);
  upload.stream = null;
}
self.onmessage = async ({data}) => {
  const {id, action} = data;
  if (busy) { self.postMessage({id, error: 'Дождись завершения текущей операции.'}); return; }
  busy = true;
  try {
    let result;
    if (action === 'load') {
      const {loadPyodide} = await import('https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs');
      py = await loadPyodide({stdout: log, stderr: log});
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.proxy = false;
      ort.env.wasm.wasmPaths = new URL('./ort/', self.location.href).href;
      registerOrtWeb(py, ort);
      await py.loadPackage('micropip');
      const config = await (await fetch('./probe-config.json', {cache: 'no-store'})).json();
      py.globals.set('wheel_url', new URL(config.wheel, self.location.href).href);
      await py.runPythonAsync('import micropip\nawait micropip.install(f"manuscript-ocr[web] @ {wheel_url}")');
      await py.runPythonAsync(await (await fetch('./full_pipeline.py', {cache: 'no-store'})).text());
      const catalog = await (await fetch('./catalog.json', {cache: 'no-store'})).json();
      py.globals.set('catalog_json', JSON.stringify(catalog));
      py.globals.set('catalog_url', new URL('./catalog.json', self.location.href).href);
      await py.runPythonAsync('app = BrowserPipelineApp(json.loads(catalog_json), catalog_url=catalog_url)');
      result = await python('app.description()');
    } else {
      if (!py) throw Error('Сначала загрузи библиотеку.');
      if (action === 'preset') {
        py.globals.set('preset_name', data.name);
        py.globals.set('stage', data.stage);
        py.globals.set('device', data.device);
        result = await python('app.validate_preset(preset_name, stage, device, probe=False)');
      } else if (action === 'upload_begin') {
        const token = crypto.randomUUID();
        const root = `/tmp/user-models/${token}`;
        py.FS.mkdirTree(root);
        uploads.set(token, {root, stage: data.stage, device: data.device, label: data.label, stream: null, written: 0});
        result = {token};
      } else if (action === 'upload_chunk') {
        const upload = uploads.get(data.token);
        if (!upload) throw Error('Передача папки уже завершена.');
        const path = `${upload.root}/${safePath(data.path)}`;
        if (upload.path !== path) {
          closeUpload(upload);
          py.FS.mkdirTree(path.slice(0, path.lastIndexOf('/')));
          upload.stream = py.FS.open(path, 'w');
          upload.path = path;
          upload.written = 0;
        }
        if (data.offset !== upload.written) throw Error('Нарушен порядок чтения файла. Выбери папку ещё раз.');
        const bytes = new Uint8Array(data.bytes);
        py.FS.write(upload.stream, bytes, 0, bytes.length, upload.written);
        upload.written += bytes.length;
        result = {written: upload.written};
      } else if (action === 'upload_finish' || action === 'upload_abort') {
        const upload = uploads.get(data.token);
        if (!upload) throw Error('Не найдена передаваемая папка.');
        closeUpload(upload);
        uploads.delete(data.token);
        py.globals.set('folder_root', upload.root);
        if (action === 'upload_abort') {
          await py.runPythonAsync('shutil.rmtree(folder_root, ignore_errors=True)');
          result = {};
        } else {
          py.globals.set('stage', upload.stage);
          py.globals.set('device', upload.device);
          py.globals.set('folder_label', upload.label);
          result = await python('app.validate_folder(folder_root, stage, device, folder_label, owned=True, probe=False)');
        }
      } else if (action === 'remove') {
        py.globals.set('model_id', data.model);
        await py.runPythonAsync('app.remove(model_id)');
        result = {};
      } else if (action === 'recheck') {
        py.globals.set('device', data.device);
        result = await python('app.recheck(device)');
      } else if (action === 'recognize') {
        py.FS.writeFile('/tmp/browser-photo', new Uint8Array(data.bytes));
        py.globals.set('settings_json', JSON.stringify(data.settings));
        py.globals.set('device', data.device);
        try {
          result = await python("app.recognize(Path('/tmp/browser-photo').read_bytes(), device, json.loads(settings_json))");
        } finally { py.FS.unlink('/tmp/browser-photo'); }
      } else if (action === 'checks') {
        py.globals.set('browser_base_url', new URL('./', self.location.href).href);
        await py.runPythonAsync(await (await fetch('./browser_checks.py')).text());
        await py.runPythonAsync('check_browser_features(browser_base_url)');
        result = {};
      } else throw Error('Неизвестная операция.');
    }
    self.postMessage({id, result});
  } catch (error) {
    log(String(error));
    let friendly;
    if (py) {
      try {
        py.globals.set('browser_error', String(error));
        friendly = await python('friendly_error(RuntimeError(browser_error))');
      } catch { /* The interpreter may be unavailable after memory exhaustion. */ }
    }
    self.postMessage({id, error: friendly?.message || 'Операция не завершилась. Попробуй ещё раз или перезапусти обработчик.', detail: String(error)});
  } finally { busy = false; }
};
