const el = id => document.getElementById(id);
const stages = {detector: 'Детектор', recognizer: 'Распознаватель', layout: 'Лейаут', corrector: 'Корректор'};
const state = {ready: false, busy: false, items: {detector: [], recognizer: [], corrector: []}, catalog: null, result: null, photo: null};
let worker, nextId = 0, folderStage, photoURL;
const pending = new Map();
window.browserAppState = state;
window.ocrResult = null;
function log(value) { el('log').textContent += `${value}\n`; el('log').scrollTop = el('log').scrollHeight; }
function notice(message = '') { el('notice').hidden = !message; el('notice').textContent = message; }
function update() {
  document.body.classList.toggle('busy', state.busy);
  el('stop').disabled = !worker;
  el('device').disabled = state.busy;
  el('run').disabled = state.busy || !state.ready || !state.photo || !state.items.detector.some(i => i.selected && !i.failed);
  el('saveText').disabled = el('saveJson').disabled = !state.result;
  for (const button of document.querySelectorAll('[data-requires-runtime]')) button.disabled = !state.ready || state.busy;
  const levels = el('levels');
  if (levels) levels.hidden = !el('recognition_override').checked;
  for (const stage of ['detector', 'recognizer', 'corrector']) {
    const section = document.querySelector(`[data-stage=${stage}] .parameters`);
    if (!section) continue;
    const enabled = state.items[stage].some(i => i.selected && !i.failed);
    for (const input of section.querySelectorAll('input, select')) input.disabled = state.busy || !enabled;
  }
}
function resetResult() {
  state.result = window.ocrResult = null;
  el('text').value = ''; el('stats').textContent = ''; el('previewModels').textContent = '';
  el('resultBadge').textContent = 'Ожидает запуска';
  el('preview').hidden = !photoURL; el('emptyPreview').hidden = !!photoURL;
  if (photoURL) el('preview').src = photoURL;
  else el('preview').removeAttribute('src');
  update();
}
function rpc(action, payload = {}, transfer = []) {
  if (!worker) return Promise.reject(Error('Обработчик остановлен. Обнови страницу.'));
  const id = ++nextId;
  return new Promise((resolve, reject) => { pending.set(id, {resolve, reject}); worker.postMessage({id, action, ...payload}, transfer); });
}
function stop() {
  worker?.terminate(); worker = null;
  for (const request of pending.values()) request.reject(Error('Обработка остановлена.'));
  pending.clear(); state.ready = false; state.busy = false;
  for (const stage of Object.keys(state.items)) { state.items[stage] = []; renderModels(stage); }
  resetResult(); window.probeDone = true;
  el('status').textContent = 'Обработка остановлена. Обнови страницу, чтобы начать снова.'; update();
}
async function operation(status, action, logOnly = false) {
  if (state.busy) return;
  state.busy = true; window.probeDone = false; notice(); if (logOnly) log(status); else el('status').textContent = status; update();
  try { await action(); }
  catch (error) { log(error.detail || String(error)); if (!logOnly) { notice(error.message); el('status').textContent = 'Не получилось завершить операцию. Настройки можно изменить и попробовать снова.'; } }
  finally { state.busy = false; window.probeDone = true; update(); }
}
function workerResult(data) {
  if (data.log !== undefined) { log(data.log); return; }
  const request = pending.get(data.id);
  if (!request) return;
  pending.delete(data.id);
  if (data.error) request.reject(Object.assign(Error(data.error), {detail: data.detail}));
  else request.resolve(data.result);
}
function field(id, label, value, min, max, step = 1) {
  return `<label>${label}<input id="${id}" type="number" value="${value}" min="${min}" max="${max}" step="${step}"></label>`;
}
function check(id, label, value = false) { return `<label>${label}<input id="${id}" type="checkbox" ${value ? 'checked' : ''}></label>`; }
function select(id, label, options) { return `<label>${label}<select id="${id}">${options.map(([v, t]) => `<option value="${v}">${t}</option>`).join('')}</select></label>`; }
const parameters = {
  detector: check('detector_override', 'Переопределить параметры') + field('target_size', 'Размер изображения', 1280, 32, 4096, 32) + field('score_thresh', 'Порог уверенности', .3, 0, 1, .05) + field('expand_ratio_w', 'Расширение по ширине', 1.4, .1, 5, .1) + field('expand_ratio_h', 'Расширение по высоте', 1.5, .1, 5, .1),
  recognizer: check('recognition_override', 'Переопределить уровни') + '<div id="levels" class="levels">' + ['word', 'line', 'block'].map(v => `<label><input type="checkbox" name="level" value="${v}" ${v === 'word' ? 'checked' : ''}>${v}</label>`).join('') + '</div>' + select('region_preparer', 'Подготовка области', [['bbox', 'Прямоугольная обрезка'], ['polygon_mask', 'Полигон с маской'], ['quad_warp', 'Выпрямление четырёхугольника']]) + field('recognition_batch', 'Батч (0 — по модели)', 0, 0, 128),
  layout: select('layout_mode', 'Режим ReadingSorting', [['containment', 'Вложенность и порядок'], ['sort', 'Только порядок']]) + check('infer_columns', 'Учитывать колонки', true) + field('containment_overlap', 'Площадь внутри родителя', .5, 0, 1, .05) + field('row_overlap', 'Перекрытие для ряда', .5, 0, 1, .05) + check('simple_columns', 'SimpleSorting: колонки', true) + field('max_splits', 'Максимум разделений', 10, 1, 30),
  corrector: field('mask_threshold', 'Порог маскирования', .05, 0, 1, .01) + field('apply_threshold', 'Порог применения', .95, 0, 1, .01) + field('max_edits', 'Максимум правок', 1, 1, 10),
};
for (const [index, [stage, title]] of Object.entries(stages).entries()) {
  const section = document.createElement('section'); section.className = 'card'; section.dataset.stage = stage;
  section.innerHTML = `<div class="card-title"><span class="number">0${index + 2}</span><h2>${title}</h2></div>`;
  if (stage === 'layout') {
    section.innerHTML += select('layout_choice', 'Алгоритм', [['ReadingSorting', 'ReadingSorting'], ['SimpleSorting', 'SimpleSorting'], ['', 'Выключен']]);
  } else {
    section.innerHTML += `<ul id="${stage}_list" class="model-list"></ul><div class="add-model"><select id="${stage}_preset" data-requires-runtime aria-label="Пресет: ${title}"><option value="">—</option></select><button id="${stage}_folder" data-requires-runtime>Выбрать папку…</button></div><div class="stage-tools"><button id="${stage}_off" class="link-button">Выключить этап</button></div>`;
  }
  section.innerHTML += `<details class="parameters"><summary>Параметры</summary><div class="parameter-grid">${parameters[stage]}</div></details>`;
  el('stages').append(section);
  if (stage !== 'layout') {
    el(`${stage}_folder`).onclick = () => { folderStage = stage; el('folder').value = ''; el('folder').click(); };
    el(`${stage}_preset`).onchange = () => operation('Выбираем модель…', async () => {
      resetResult();
      const name = el(`${stage}_preset`).value;
      if (!name) { await clearModel(stage); return; }
      await addPreset(name, stage);
    });
    el(`${stage}_off`).onclick = () => operation('Освобождаем модель…', () => clearModel(stage));
    renderModels(stage);
  }
}
async function clearModel(stage) {
  for (const item of state.items[stage]) await rpc('remove', {model: item.id});
  state.items[stage] = []; el(`${stage}_preset`).value = ''; renderModels(stage); resetResult();
}
async function addValidated(stage, model) {
  for (const item of state.items[stage]) {
    if (item.id !== model.id) await rpc('remove', {model: item.id});
  }
  state.items[stage] = [{...model, selected: true, failed: false}];
  el(`${stage}_preset`).value = model.id.startsWith('preset:') ? model.id.slice(7) : '';
  renderModels(stage); resetResult(); el('status').textContent = `${model.label}: модель выбрана.`;
}
async function checkResponse(response, stage) {
  if (!response.ok) { const error = Error(response.message); error.detail = response.detail; throw error; }
  await addValidated(stage, response.model);
}
async function addPreset(name, stage) { await checkResponse(await rpc('preset', {name, stage, device: el('device').value}), stage); }
function renderModels(stage) {
  const list = el(`${stage}_list`); list.replaceChildren();
  if (!state.items[stage].length) { const empty = document.createElement('li'); empty.className = 'stage-empty'; empty.textContent = stage === 'detector' ? 'Добавь детектор' : 'Этап выключен'; list.append(empty); }
  for (const item of state.items[stage]) {
    const row = document.createElement('li'); row.className = `model-row${item.failed ? ' failed' : ''}`; row.dataset.model = item.id;
    const label = document.createElement('label'); label.textContent = item.label; row.append(label);
    const info = document.createElement('small'); info.textContent = item.failed ? 'Проверка не пройдена. Можно сменить устройство.' : `${item.class} · ${item.device === 'cpu' ? 'WASM' : 'WebGPU'} · выбрана`; label.append(info); row.append(label);
    const remove = document.createElement('button'); remove.textContent = '×'; remove.title = 'Удалить модель';
    remove.onclick = () => operation('Освобождаем модель…', () => clearModel(stage)); row.append(remove); list.append(row);
  }
  update();
}
function initialize() { return operation('Загружаем Python и библиотеку…', async () => {
  worker?.terminate();
  worker = new Worker(new URL('./app.worker.mjs?v=registry-routing-2', import.meta.url), {type: 'module'});
  worker.onmessage = ({data}) => workerResult(data);
  worker.onerror = event => { const wasReady = state.ready; log(event.message || 'Ошибка обработчика.'); stop(); if (wasReady) notice('Обработчик завершился. Обнови страницу, чтобы начать снова.'); else el('status').textContent = ''; };
  state.catalog = await rpc('load'); state.ready = true;
  for (const stage of Object.keys(state.items)) {
    const selector = el(`${stage}_preset`); selector.replaceChildren(new Option('Выбрать модель…', ''));
    for (const preset of state.catalog.presets[stage]) { const option = document.createElement('option'); option.value = preset.id; option.textContent = preset.label; selector.append(option); }
  }
  for (const warning of state.catalog.warnings) log(warning);
  log('Библиотека готова.');
}, true); }
el('stop').onclick = stop;
el('device').onchange = () => {
  for (const stage of Object.keys(state.items)) {
    for (const item of state.items[stage]) item.device = el('device').value;
    renderModels(stage);
  }
  resetResult();
};
el('folder').onchange = () => {
  const files = Array.from(el('folder').files), stage = folderStage;
  if (!files.length) return;
  operation('Читаем папку модели на твоём устройстве…', async () => {
    resetResult();
    const folder = files[0].webkitRelativePath.split('/')[0];
    const relative = new Map(files.map(file => [file.webkitRelativePath.slice(folder.length + 1), file]));
    if (!relative.has('model.json')) throw Error('Выбери папку модели с файлом model.json в её корне.');
    let manifest;
    try { manifest = JSON.parse(await relative.get('model.json').text()); } catch { throw Error('Не удалось прочитать model.json. Проверь формат файла.'); }
    if (!manifest.artifacts || typeof manifest.artifacts !== 'object') throw Error('В model.json нет описания файлов artifacts.');
    const needed = new Set(['model.json']);
    for (const artifact of Object.values(manifest.artifacts)) {
      if (!artifact) continue;
      if (typeof artifact.filename !== 'string' || artifact.filename.startsWith('/') || artifact.filename.includes('\\') || artifact.filename.split('/').some(s => !s || s === '.' || s === '..')) throw Error('В model.json указан некорректный путь файла.');
      if (!relative.has(artifact.filename)) { if (artifact.required === false) continue; throw Error(`В выбранной папке нет файла ${artifact.filename}.`); }
      needed.add(artifact.filename);
    }
    // Preserve unlisted external ONNX companions as well as declared artifacts.
    for (const name of relative.keys()) if (name.endsWith('.data')) needed.add(name);
    const total = [...needed].reduce((n, name) => n + relative.get(name).size, 0);
    let done = 0, token;
    try {
      ({token} = await rpc('upload_begin', {stage, device: el('device').value, label: folder}));
      for (const name of needed) {
        const file = relative.get(name);
        for (let offset = 0; offset < file.size || offset === 0; offset += 1024 * 1024) {
          const bytes = await file.slice(offset, offset + 1024 * 1024).arrayBuffer();
          await rpc('upload_chunk', {token, path: name, offset, bytes}, [bytes]);
          done += Math.min(1024 * 1024, file.size - offset);
          el('status').textContent = `Читаем ${folder}: ${(done / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} МБ`;
        }
      }
      el('status').textContent = `${folder}: читаем описание модели…`;
      const response = await rpc('upload_finish', {token}); token = null; await checkResponse(response, stage);
    } finally { if (token && worker) await rpc('upload_abort', {token}).catch(() => {}); }
  });
};
function setPhoto(file) {
  state.photo = file; resetResult();
  if (photoURL) URL.revokeObjectURL(photoURL);
  photoURL = file ? URL.createObjectURL(file) : null;
  el('photoName').textContent = file?.name || 'Пример изображения';
  el('preview').hidden = !file; el('emptyPreview').hidden = !!file;
  if (file) el('preview').src = photoURL;
  update();
}
el('photo').onchange = () => setPhoto(el('photo').files[0] || null);
el('sample').onclick = async () => { try { const response = await fetch('./sample.png'); if (!response.ok) throw Error('Пример изображения недоступен. Выбери свой файл.'); setPhoto(new File([await response.blob()], 'sample.png', {type: 'image/png'})); } catch (error) { notice(error.message); } };
function settings() {
  const values = {};
  for (const stage of Object.keys(state.items)) { values[stage] = state.items[stage].filter(i => i.selected && !i.failed).map(i => i.id);  }
  values.layout = el('layout_choice').value ? [el('layout_choice').value] : [];
  for (const id of ['detector_override', 'recognition_override', 'infer_columns', 'simple_columns']) values[id] = el(id).checked;
  for (const id of ['target_size', 'score_thresh', 'expand_ratio_w', 'expand_ratio_h', 'recognition_batch', 'containment_overlap', 'row_overlap', 'max_splits', 'mask_threshold', 'apply_threshold', 'max_edits']) {
    const input = el(id), value = Number(input.value);
    if (!input.value.trim() || !Number.isFinite(value) || value < Number(input.min) || value > Number(input.max)) throw Error(`Проверь параметр «${input.parentElement.textContent}».`);
    values[id] = value;
  }
  for (const id of ['region_preparer', 'layout_mode']) values[id] = el(id).value;
  values.recognition_levels = [...document.querySelectorAll('[name=level]:checked')].map(input => input.value);
  return values;
}
el('run').onclick = () => operation('Обрабатываем изображение на твоём устройстве…', async () => {
  const options = settings(); resetResult();
  if (!state.photo || state.photo.size > 40 * 1024 * 1024) throw Error('Выбери изображение размером до 40 МБ.');
  const bytes = await state.photo.arrayBuffer();
  const result = await rpc('recognize', {bytes, device: el('device').value, settings: options}, [bytes]);
  state.result = window.ocrResult = result;
  el('text').value = result.text; el('preview').src = result.preview; el('preview').hidden = false; el('emptyPreview').hidden = true;
  el('resultBadge').textContent = `${result.seconds.toFixed(2)} с`;
  el('previewModels').textContent = Object.entries(result.models).filter(([, models]) => models.length).map(([stage, models]) => `${stages[stage]}: ${models.map(model => `${model.class} — ${model.label}`).join(', ')}`).join(' · ') + ` · ${result.device === 'cpu' ? 'CPU / WASM' : 'WebGPU'}`;
  const names = {load_image: 'Загрузка изображения', detector: 'Детекция', layout: 'Лейаут', recognizer: 'Распознавание', corrector: 'Коррекция'};
  el('stats').textContent = `Всего: ${result.seconds.toFixed(2)} с · Пайплайн: ${result.timings.total.toFixed(2)} с\nПодготовка моделей: ${result.setup_seconds.toFixed(2)} с · Визуализация: ${result.visualization_seconds.toFixed(2)} с\n` + Object.entries(names).filter(([key]) => key in result.timings).map(([key, name]) => `${name}: ${result.timings[key].toFixed(3)} с`).join('\n') + '\nОбъекты: ' + (Object.entries(result.counts).map(([type, count]) => `${type}: ${count}`).join(', ') || '0') + `\nСкорость: ${result.pages_per_second.toFixed(2)} стр/с · ${result.words_per_second.toFixed(1)} слов/с`;
  el('status').textContent = Object.keys(result.counts).length ? 'Готово.' : 'Обработка завершена. Регионы не найдены — попробуй другое изображение или настройки.';
});
function save(data, type, name) { const url = URL.createObjectURL(new Blob([data], {type})); const link = document.createElement('a'); link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }
el('saveText').onclick = () => save(el('text').value, 'text/plain;charset=utf-8', 'manuscript.txt');
el('saveJson').onclick = () => save(JSON.stringify(state.result.page, null, 2), 'application/json', 'manuscript.json');
if (!navigator.gpu) el('device').querySelector('[value=gpu]').disabled = true;
for (const input of document.querySelectorAll('.configuration input, .configuration select')) input.addEventListener('change', resetResult);
update();

initialize();
