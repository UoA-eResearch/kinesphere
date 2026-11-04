// Runs the Altlando dance model in the browser (a module Web Worker, so the page stays smooth),
// with ONNX Runtime Web: WebGPU when the browser has it, WebAssembly otherwise.
//
// The model files come from ai-partner/export_onnx.py (altlando.json, encoder.onnx, gpt.onnx,
// decoder.onnx). Generation follows Altlando's server (dance-ai-research-project,
// src/bailandopp/app/server_v1.py and models_v1/bailando_v1.py, eval_single_epoch):
//   1. joints relative to the root (root kept absolute), padded to a multiple of 5 frames
//   2. encode to pose codes (upper and lower body), keep the first `shift`
//   3. greedily add `length` codes with the GPT, 29 codes of context at a time
//   4. decode; the root channels are its velocity: integrate them, starting at the origin
//   5. joints back to absolute positions
//
// Messages in:  { type: 'load', base?: url of the model folder, files?: { name: ArrayBuffer | text } }
//               { type: 'generate', id, payload: Float32Array(frames * 72), frames, length }
// Messages out: { type: 'progress', loaded, total, file } | { type: 'ready', backend, manifest }
//               { type: 'result', id, pose: Float32Array, frames, codesUp, codesDown, ms } | { type: 'error', id?, message }

const ORT_VERSION = '1.22.0';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
const CACHE = 'kinesphere-models';
const JOINTS = 24;

let ort = null, sessions = null, manifest = null, backend = null;

const post = (msg, transfer) => self.postMessage(msg, transfer ?? []);

/**
 * Fetch a model file, reporting progress, from the browser's cache when we have it. `version`
 * (the file's hash from the manifest) is part of the cache key, so a model replaced at the same
 * address is downloaded again instead of mixed with the old one.
 */
async function fetchFile(url, onProgress, version = '', { useCache = true } = {}) {
  let cache = null;
  try { cache = await caches.open(CACHE); } catch { /* no Cache Storage (e.g. private window) */ }
  const key = version ? `${url}?v=${version}` : url;
  let res = cache && useCache ? await cache.match(key) : null;
  const fromCache = Boolean(res);
  if (!res) {
    res = await fetch(url);
    if (!res.ok) throw new Error(`${url.split('/').pop()}: ${res.status === 404 ? 'not found' : `HTTP ${res.status}`}`);
  }
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.clone().body?.getReader();
  if (!reader) return res.arrayBuffer();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }
  const buf = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  if (cache && !fromCache && version) cache.put(key, new Response(buf, { headers: { 'content-type': 'application/octet-stream', 'content-length': String(loaded) } })).catch(() => {});
  return buf.buffer;
}

async function load({ base, files }) {
  ort ??= await import(`${ORT_BASE}ort.webgpu.min.mjs`);
  ort.env.wasm.wasmPaths = ORT_BASE;
  ort.env.logLevel = 'error'; // not the warnings about which ops run where
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const root = base ? new URL(base.endsWith('/') ? base : `${base}/`, self.location.href).href : null;
  const text = async name => {
    if (files?.[name] != null) return typeof files[name] === 'string' ? files[name] : new TextDecoder().decode(files[name]);
    const res = await fetch(new URL(name, root), { cache: 'no-cache' });
    if (!res.ok) throw new Error(`no model at ${root} (${name}: ${res.status === 404 ? 'not found' : `HTTP ${res.status}`})`);
    return res.text();
  };
  const m = JSON.parse(await text('altlando.json'));
  if (m.format !== 'kinesphere-altlando-onnx') throw new Error('altlando.json is not a Kinesphere Altlando model manifest');
  const names = ['encoder', 'gpt', 'decoder'];
  const totals = names.map(n => m.bytes?.[n] || 0), loaded = names.map(() => 0);
  const buffers = [];
  for (const [i, n] of names.entries()) {
    // a file, or a list of parts to join (the export splits big files: GitHub refuses files over 100 MB)
    const parts = [].concat(m.files[n]);
    // cache only what the manifest identifies (its hash, or failing that its size)
    const version = m.sha256?.[n] ?? (m.bytes?.[n] ? `${m.bytes[n]}b` : '');
    const fetchAll = async useCache => {
      const got = [];
      let before = 0;
      for (const file of parts) {
        const buf = files?.[file] ?? await fetchFile(new URL(file, root).href, l => {
          loaded[i] = before + l;
          post({ type: 'progress', loaded: loaded.reduce((a, b) => a + b, 0), total: totals.reduce((a, b) => a + b, 0) || 0, file });
        }, version, { useCache });
        if (!buf) throw new Error(`${file} is missing`);
        got.push(new Uint8Array(buf));
        before += buf.byteLength;
      }
      return { got, before };
    };
    let { got, before } = await fetchAll(true);
    if (m.bytes?.[n] && before !== m.bytes[n] && !files) ({ got, before } = await fetchAll(false)); // a stale or cut-off copy: fetch afresh
    if (m.bytes?.[n] && before !== m.bytes[n]) throw new Error(`${n} is ${before} bytes, the manifest says ${m.bytes[n]}`);
    const whole = new Uint8Array(before);
    let o = 0;
    for (const g of got) { whole.set(g, o); o += g.length; }
    buffers.push(whole.buffer);
  }
  // The GPT does nearly all the work (it re-reads its 29-code window for every new code): it runs on
  // WebGPU when the browser has it. The encoder and decoder run once per reply and use ScatterND,
  // which ONNX Runtime's WebGPU backend gets wrong, so they stay on WebAssembly.
  const create = (b, eps) => ort.InferenceSession.create(new Uint8Array(b), { executionProviders: eps, graphOptimizationLevel: 'all' });
  const s = [];
  s[0] = await create(buffers[0], ['wasm']); // one at a time: session creation isn't re-entrant
  backend = 'WebAssembly';
  if (self.navigator.gpu && !self.__forceWasm) {
    try { s[1] = await create(buffers[1], ['webgpu']); backend = 'WebGPU'; } catch { s[1] = null; }
  }
  s[1] ??= await create(buffers[1], ['wasm']);
  s[2] = await create(buffers[2], ['wasm']);
  sessions = { encoder: s[0], gpt: s[1], decoder: s[2] };
  manifest = m;
  post({ type: 'ready', backend, manifest: m });
}

const int64 = arr => new ort.Tensor('int64', BigInt64Array.from(arr, v => BigInt(v)), [1, arr.length]);

/** Altlando's generation for one stretch of movement (see the steps at the top). */
async function generate({ payload, frames, length }) {
  const { blockSize, codeFrames, shift, padTo = 5 } = manifest;
  const t0 = performance.now();
  // 1. root-relative joints, root absolute; pad as Altlando's server does
  const padded = (Math.floor(frames / padTo) + 1) * padTo;
  const x = new Float32Array(padded * 72);
  for (let f = 0; f < padded; f++) {
    const src = Math.min(f, frames - 1) * 72;
    const rx = payload[src], ry = payload[src + 1], rz = payload[src + 2];
    for (let j = 0; j < JOINTS; j++) {
      x[f * 72 + j * 3] = payload[src + j * 3] - (j ? rx : 0);
      x[f * 72 + j * 3 + 1] = payload[src + j * 3 + 1] - (j ? ry : 0);
      x[f * 72 + j * 3 + 2] = payload[src + j * 3 + 2] - (j ? rz : 0);
    }
  }
  // 2. encode
  const enc = await sessions.encoder.run({ pose: new ort.Tensor('float32', x, [1, padded, 72]) });
  const up = Array.from(enc.codes_up.data, Number).slice(0, shift), down = Array.from(enc.codes_down.data, Number).slice(0, shift);
  // 3. continue, greedily, 29 codes of context at a time (padded at the end: the attention is causal)
  for (let k = 0; k < length; k++) {
    const wu = up.slice(-blockSize), wd = down.slice(-blockSize), n = wu.length;
    while (wu.length < blockSize) { wu.push(0); wd.push(0); }
    const out = await sessions.gpt.run({ codes_up: int64(wu), codes_down: int64(wd) });
    up.push(Number(out.next_up.data[n - 1]));
    down.push(Number(out.next_down.data[n - 1]));
  }
  // 4. decode, integrate the root velocity
  const dec = await sessions.decoder.run({ codes_up: int64(up), codes_down: int64(down) });
  const raw = dec.pose.data, total = raw.length / 72;
  const pose = new Float32Array(raw.length);
  let px = 0, py = 0, pz = 0;
  for (let f = 0; f < total; f++) {
    if (f > 0) { px += raw[(f - 1) * 72]; py += raw[(f - 1) * 72 + 1]; pz += raw[(f - 1) * 72 + 2]; }
    // 5. absolute joints
    pose[f * 72] = px; pose[f * 72 + 1] = py; pose[f * 72 + 2] = pz;
    for (let j = 1; j < JOINTS; j++) {
      pose[f * 72 + j * 3] = raw[f * 72 + j * 3] + px;
      pose[f * 72 + j * 3 + 1] = raw[f * 72 + j * 3 + 1] + py;
      pose[f * 72 + j * 3 + 2] = raw[f * 72 + j * 3 + 2] + pz;
    }
  }
  return { pose, frames: total, codesUp: up, codesDown: down, ms: performance.now() - t0, raw: Float32Array.from(raw) };
}

let queue = Promise.resolve();
self.onmessage = e => {
  const msg = e.data;
  // one job at a time: the sessions are not re-entrant
  queue = queue.then(async () => {
    try {
      if (msg.type === 'load') await load(msg);
      else if (msg.type === 'generate') {
        if (!sessions) throw new Error('the model is not loaded');
        const r = await generate(msg);
        post({ type: 'result', id: msg.id, ...r }, [r.pose.buffer, r.raw.buffer]);
      }
    } catch (err) {
      post({ type: 'error', id: msg.id, message: err?.message ?? String(err) });
    }
  });
};
