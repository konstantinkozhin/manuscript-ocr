/** Register once per Pyodide interpreter. Pass the imported ORT Web module. */
export function registerOrtWeb(pyodide, ort) {
  if (!ort.InferenceSession?.create || !ort.Tensor) {
    throw new Error('A compatible ONNX Runtime Web module is required (tested: 1.23.2).');
  }
  const typed = {
    float32: Float32Array, float64: Float64Array,
    int8: Int8Array, uint8: Uint8Array, int16: Int16Array, uint16: Uint16Array,
    int32: Int32Array, uint32: Uint32Array, int64: BigInt64Array, uint64: BigUint64Array,
    bool: Uint8Array, float16: Uint16Array,
  };
  const metadata = (entries) => {
    if (!Array.isArray(entries)) throw new Error('ONNX Runtime Web must expose inputMetadata/outputMetadata (tested: 1.23.2).');
    return entries.map(item => ({
    name: item.name, type: item.type, shape: [...item.shape],
  }));
  };
  pyodide.registerJsModule('manuscript_ort_web', {
    async create_session(bytes, provider, externalData = [], logSeverityLevel = null) {
      const session = await ort.InferenceSession.create(new Uint8Array(bytes), {
        executionProviders: [provider],
        ...(logSeverityLevel == null ? {} : {logSeverityLevel}),
        externalData: externalData.map(item => ({path: item.path, data: new Uint8Array(item.data)})),
      });
      try {
        return { session, inputs: metadata(session.inputMetadata), outputs: metadata(session.outputMetadata) };
      } catch (error) { await session.release(); throw error; }
    },
    async run_session(handle, descriptors, names) {
      const feeds = {};
      let outputs;
      try {
        for (const item of descriptors) {
          const ArrayType = typed[item.type];
          if (!ArrayType) throw new Error(`Unsupported tensor type: ${item.type}`);
          // Own the input buffers across await; never retain a Python memory view.
          const values = new ArrayType(item.data);
          feeds[item.name] = new ort.Tensor(item.type, values, item.shape);
        }
        outputs = await handle.session.run(feeds, names);
        return names.map(name => {
          const tensor = outputs[name];
          // Pyodide cannot convert Float16Array yet; transfer its raw bits.
          const data = tensor.type === 'float16'
            ? new Uint16Array(tensor.data.buffer, tensor.data.byteOffset, tensor.data.length).slice()
            : tensor.data.slice();
          return { type: tensor.type, shape: [...tensor.dims], data };
        });
      } finally {
        for (const tensor of Object.values(feeds)) tensor.dispose();
        if (outputs) for (const tensor of Object.values(outputs)) tensor.dispose();
      }
    },
    async release_session(handle) { await handle.session.release(); },
  });
}
