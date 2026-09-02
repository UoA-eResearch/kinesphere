// Session data model, compact serialisation, localStorage persistence and file export.
//
// In memory a session is:
//   { id, name, createdAt, durationMs, width, height, mirrored, model, trigger,
//     frameCount, times: Float64Array (ms since start), lm: Float32Array (frameCount*33*4) }
//
// On disk / in localStorage it is JSON (format "kinesphere-session", version 1) with the
// landmark positions packed as base64 int16 (x,y,z / 10000) and visibility as base64 uint8 (/255).
// Only pose landmarks are ever stored - never video.

import { LANDMARK_NAMES, NUM_LANDMARKS, STRIDE, FRAME_SIZE } from './pose.js';

export const FORMAT = 'kinesphere-session';
export const VERSION = 1;
const POS_SCALE = 10000;
const INDEX_KEY = 'kinesphere:sessions';
const keyFor = id => `kinesphere:session:${id}`;

export class StorageFullError extends Error {
  constructor(msg) { super(msg); this.name = 'StorageFullError'; }
}

export function makeId() {
  return crypto.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export function defaultName(d = new Date()) {
  const date = d.toLocaleDateString(undefined, { year: 'numeric', month: 'short', day: 'numeric' });
  const time = d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
  return `Session ${date} ${time}`;
}

/** Build a session object from recorded frames [{ t, lm: Float32Array(132) }]. */
export function buildSession({ frames, width, height, model, trigger, startedAt, mirrored = true }) {
  const n = frames.length;
  const times = new Float64Array(n);
  const lm = new Float32Array(n * FRAME_SIZE);
  frames.forEach((f, i) => {
    times[i] = f.t;
    lm.set(f.lm, i * FRAME_SIZE);
  });
  const started = new Date(startedAt ?? Date.now());
  return {
    id: makeId(),
    name: defaultName(started),
    createdAt: started.toISOString(),
    durationMs: n ? Math.round(times[n - 1]) : 0,
    width, height,
    mirrored: mirrored !== false,
    model: model ?? 'lite',
    trigger: trigger ?? 'button',
    frameCount: n,
    times, lm,
  };
}

// ---- base64 helpers ------------------------------------------------------------------

function bytesToB64(bytes) {
  let s = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(s);
}

function b64ToBytes(b64) {
  const s = atob(b64);
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}

// ---- (de)serialisation ---------------------------------------------------------------

export function serialize(session) {
  const n = session.frameCount;
  const pos = new DataView(new ArrayBuffer(n * NUM_LANDMARKS * 3 * 2));
  const vis = new Uint8Array(n * NUM_LANDMARKS);
  const { lm } = session;
  let j = 0;
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < NUM_LANDMARKS; k++) {
      const o = (i * NUM_LANDMARKS + k) * STRIDE;
      for (let c = 0; c < 3; c++) {
        const q = Math.max(-32768, Math.min(32767, Math.round(lm[o + c] * POS_SCALE)));
        pos.setInt16(j * 2, q, true);
        j++;
      }
      vis[i * NUM_LANDMARKS + k] = Math.round(Math.max(0, Math.min(1, lm[o + 3])) * 255);
    }
  }
  return JSON.stringify({
    format: FORMAT,
    version: VERSION,
    id: session.id,
    name: session.name,
    createdAt: session.createdAt,
    durationMs: session.durationMs,
    width: session.width,
    height: session.height,
    mirrored: session.mirrored,
    model: session.model,
    trigger: session.trigger,
    frameCount: n,
    landmarkNames: LANDMARK_NAMES,
    encoding: {
      times: 'milliseconds since the start of the recording, one per frame',
      positions: 'base64 of little-endian int16 triples [x, y, z] per landmark per frame (frame-major); divide by 10000 to get MediaPipe normalised image coordinates',
      visibility: 'base64 of uint8, one per landmark per frame; divide by 255',
    },
    times: Array.from(session.times, t => Math.round(t)),
    positions: bytesToB64(new Uint8Array(pos.buffer)),
    visibility: bytesToB64(vis),
  });
}

export function deserialize(text) {
  const j = typeof text === 'string' ? JSON.parse(text) : text;
  if (!j || j.format !== FORMAT) throw new Error('Not a Kinesphere session file');
  if (j.version !== VERSION) throw new Error(`Unsupported session version ${j.version}`);
  const n = j.frameCount | 0;
  if (!Array.isArray(j.times) || j.times.length !== n) throw new Error('Corrupt session: times');
  const posBytes = b64ToBytes(j.positions);
  const visBytes = b64ToBytes(j.visibility);
  if (posBytes.length !== n * NUM_LANDMARKS * 6 || visBytes.length !== n * NUM_LANDMARKS) {
    throw new Error('Corrupt session: landmark data length mismatch');
  }
  const pos = new DataView(posBytes.buffer, posBytes.byteOffset, posBytes.byteLength);
  const lm = new Float32Array(n * FRAME_SIZE);
  let p = 0;
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < NUM_LANDMARKS; k++) {
      const o = (i * NUM_LANDMARKS + k) * STRIDE;
      lm[o] = pos.getInt16(p, true) / POS_SCALE; p += 2;
      lm[o + 1] = pos.getInt16(p, true) / POS_SCALE; p += 2;
      lm[o + 2] = pos.getInt16(p, true) / POS_SCALE; p += 2;
      lm[o + 3] = visBytes[i * NUM_LANDMARKS + k] / 255;
    }
  }
  return {
    id: String(j.id || makeId()),
    name: String(j.name || defaultName(new Date(j.createdAt || Date.now()))),
    createdAt: j.createdAt || new Date().toISOString(),
    durationMs: Number(j.durationMs) || (n ? j.times[n - 1] : 0),
    width: Number(j.width) || 1280,
    height: Number(j.height) || 720,
    mirrored: j.mirrored !== false,
    model: j.model || 'unknown',
    trigger: j.trigger || 'unknown',
    frameCount: n,
    times: Float64Array.from(j.times),
    lm,
  };
}

// ---- localStorage --------------------------------------------------------------------

function readIndex() {
  try {
    const list = JSON.parse(localStorage.getItem(INDEX_KEY));
    return Array.isArray(list) ? list : [];
  } catch {
    return [];
  }
}
function writeIndex(list) {
  localStorage.setItem(INDEX_KEY, JSON.stringify(list));
}
function metaOf(session, bytes) {
  return {
    id: session.id, name: session.name, createdAt: session.createdAt,
    durationMs: session.durationMs, frameCount: session.frameCount, bytes,
  };
}

export const store = {
  /** Session metadata, newest first. */
  list() {
    return readIndex().sort((a, b) => String(b.createdAt).localeCompare(String(a.createdAt)));
  },
  load(id) {
    const text = localStorage.getItem(keyFor(id));
    if (!text) return null;
    try {
      return deserialize(text);
    } catch (err) {
      console.warn('Could not read stored session', id, err);
      return null;
    }
  },
  /** Persist a session. Throws StorageFullError when the browser quota is exhausted. */
  save(session) {
    const text = serialize(session);
    try {
      localStorage.setItem(keyFor(session.id), text);
    } catch (err) {
      throw new StorageFullError(
        `Browser storage is full (this session needs ${(text.length * 2 / 1024 / 1024).toFixed(1)} MB). Export it to a file, or delete older sessions.`,
      );
    }
    const index = readIndex().filter(m => m.id !== session.id);
    index.push(metaOf(session, text.length * 2));
    writeIndex(index);
  },
  remove(id) {
    localStorage.removeItem(keyFor(id));
    writeIndex(readIndex().filter(m => m.id !== id));
  },
  rename(id, name) {
    const session = this.load(id);
    if (!session) return;
    session.name = name;
    this.save(session);
  },
  /** Approximate bytes used by Kinesphere in localStorage (UTF-16 => 2 bytes per char). */
  usage() {
    let bytes = 0;
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (k && k.startsWith('kinesphere:')) bytes += (localStorage.getItem(k)?.length ?? 0) * 2;
    }
    return bytes;
  },
};

// ---- export ----------------------------------------------------------------------------

export function toCSV(session) {
  const header = ['t_ms'];
  for (const name of LANDMARK_NAMES) header.push(`${name}_x`, `${name}_y`, `${name}_z`, `${name}_visibility`);
  const rows = [header.join(',')];
  const { lm, times, frameCount } = session;
  for (let i = 0; i < frameCount; i++) {
    const cells = [Math.round(times[i])];
    const base = i * FRAME_SIZE;
    for (let j = 0; j < FRAME_SIZE; j++) cells.push(lm[base + j].toFixed(4));
    rows.push(cells.join(','));
  }
  return rows.join('\n');
}

export function safeFilename(name) {
  return String(name).replace(/[^a-z0-9_-]+/gi, '_').replace(/^_+|_+$/g, '').slice(0, 60) || 'session';
}

export function download(filename, text, mime = 'application/json') {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
