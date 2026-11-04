// The Altlando dance model in the browser: a small promise API over js/altlando-worker.js.
//
//   const model = createAltlando({ onProgress: (loaded, total) => … });
//   const { backend, manifest } = await model.load({ base: 'models/altlando/' });   // or { files }
//   const frames = await model.generate(inputFrames, 28); // 224 Float32Array(72) in, 448 out

/** The models bundled with the site. Altlando is the default. */
export const MODELS = {
  altlando: { url: 'models/altlando/', label: 'Altlando', title: 'Altlando (Zhou 2025): Bailando retrained without music on AIST++, FineDance and CMU motion capture' },
  bailando: { url: 'models/bailando/', label: 'Bailando', title: 'Bailando (Siyao et al. 2022): the original AIST++ weights, with the music input removed' },
};
export const DEFAULT_MODEL = 'altlando';
export const DEFAULT_MODEL_URL = MODELS[DEFAULT_MODEL].url;

export function createAltlando({ onProgress = () => {} } = {}) {
  let worker = null, nextId = 1, loaded = null;
  const pending = new Map();
  let loadWaiter = null;

  function start() {
    if (worker) return worker;
    worker = new Worker(new URL('./altlando-worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = e => {
      const m = e.data;
      if (m.type === 'progress') onProgress(m.loaded, m.total, m.file);
      else if (m.type === 'ready') { loaded = { backend: m.backend, manifest: m.manifest }; loadWaiter?.resolve(loaded); loadWaiter = null; }
      else if (m.type === 'result') {
        const p = pending.get(m.id); pending.delete(m.id);
        if (!p) return;
        const frames = [];
        for (let f = 0; f < m.frames; f++) frames.push(m.pose.subarray(f * 72, (f + 1) * 72));
        p.resolve(Object.assign(frames, { codesUp: m.codesUp, codesDown: m.codesDown, ms: m.ms, raw: m.raw }));
      } else if (m.type === 'error') {
        if (m.id && pending.has(m.id)) { pending.get(m.id).reject(new Error(m.message)); pending.delete(m.id); }
        else if (loadWaiter) { loadWaiter.reject(new Error(m.message)); loadWaiter = null; }
      }
    };
    worker.onerror = e => {
      const err = new Error(e.message || 'the model worker failed to start');
      loadWaiter?.reject(err); loadWaiter = null;
      for (const p of pending.values()) p.reject(err);
      pending.clear();
    };
    return worker;
  }

  return {
    /** Load the model from a folder URL (altlando.json + the .onnx files), or from chosen files { name: ArrayBuffer }. */
    load({ base = DEFAULT_MODEL_URL, files = null } = {}) {
      start();
      loaded = null;
      return new Promise((resolve, reject) => {
        loadWaiter = { resolve, reject };
        const transfer = files ? Object.values(files).filter(v => v instanceof ArrayBuffer) : [];
        worker.postMessage({ type: 'load', base: files ? null : new URL(base, location.href).href, files }, transfer);
      });
    },
    get loaded() { return loaded; },
    /** Altlando's reply to a stretch of movement: frames of 24 SMPL joints (metres, y up, 60 fps). */
    generate(frames, length = 28) {
      if (!loaded) return Promise.reject(new Error('the model is not loaded'));
      const n = frames.length, payload = new Float32Array(n * 72);
      frames.forEach((f, i) => payload.set(f, i * 72));
      const id = nextId++;
      return new Promise((resolve, reject) => {
        pending.set(id, { resolve, reject });
        worker.postMessage({ type: 'generate', id, payload, frames: n, length }, [payload.buffer]);
      });
    },
    terminate() {
      worker?.terminate(); worker = null; loaded = null;
      for (const p of pending.values()) p.reject(new Error('stopped'));
      pending.clear();
    },
  };
}
