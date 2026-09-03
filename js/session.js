// Session data model, compact serialisation, localStorage persistence and file export.
//
// In memory a session is:
//   { id, name, createdAt, durationMs, width, height, mirrored, model, engine, people, trigger,
//     frameCount, times: Float64Array (ms since start),
//     lm: Float32Array (frameCount * people * 33 * 4) }
// Landmarks for frame i and person p start at (i * people + p) * FRAME_SIZE. A person who was
// not detected in a frame has visibility 0 for every landmark.
//
// Holistic sessions also carry `hands`: Float32Array (frameCount * people * 42 * 4), left hand
// (21 points) then right hand per person per frame, in the same [x, y, z, visibility] layout.
//
// On disk / in localStorage it is JSON (format "kinesphere-session", version 3) with the
// landmark positions packed as base64 int16 (x,y,z / 10000) and visibility as base64 uint8 (/255),
// and optional handPositions / handVisibility packed the same way.
// Version 1 (single person, no engine) and 2 (no hands) files are still read.
// Only pose landmarks are ever stored - never video.

import { LANDMARK_NAMES, NUM_LANDMARKS, STRIDE, FRAME_SIZE, HAND_LANDMARKS, HAND_SIZE } from './pose.js';

export const FORMAT = 'kinesphere-session';
export const VERSION = 3;
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

/** Build a session from recorded frames [{ t, lm: Float32Array(people * 132), hands?: Float32Array(people * 168) }]. */
export function buildSession({ frames, width, height, model, engine, people = 1, trigger, startedAt, mirrored = true }) {
  const n = frames.length;
  const stride = people * FRAME_SIZE;
  const times = new Float64Array(n);
  const lm = new Float32Array(n * stride);
  const withHands = frames.some(f => f.hands);
  const handStride = people * HAND_SIZE;
  const hands = withHands ? new Float32Array(n * handStride) : null;
  frames.forEach((f, i) => {
    times[i] = f.t;
    lm.set(f.lm.subarray(0, stride), i * stride);
    if (hands && f.hands) hands.set(f.hands.subarray(0, handStride), i * handStride);
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
    engine: engine ?? 'mediapipe',
    people,
    trigger: trigger ?? 'button',
    frameCount: n,
    times, lm, hands,
  };
}

/** Offset into session.hands of frame `i`, person `p`. */
export function handOffset(session, i, p = 0) {
  return (i * (session.people || 1) + p) * HAND_SIZE;
}

/** Offset into session.lm of frame `i`, person `p`. */
export function frameOffset(session, i, p = 0) {
  return (i * (session.people || 1) + p) * FRAME_SIZE;
}

/** True when the person has any visible landmark in that frame. */
export function personPresent(lm, offset) {
  for (let k = 0; k < NUM_LANDMARKS; k++) if (lm[offset + k * STRIDE + 3] > 0) return true;
  return false;
}

/** A single-person view of one tracked person, suitable for analyze(). */
export function personView(session, slot) {
  const people = session.people || 1;
  if (people === 1 && slot === 0) return session;
  const n = session.frameCount;
  const lm = new Float32Array(n * FRAME_SIZE);
  const hands = session.hands ? new Float32Array(n * HAND_SIZE) : null;
  for (let i = 0; i < n; i++) {
    const o = (i * people + slot) * FRAME_SIZE;
    lm.set(session.lm.subarray(o, o + FRAME_SIZE), i * FRAME_SIZE);
    if (hands) {
      const ho = (i * people + slot) * HAND_SIZE;
      hands.set(session.hands.subarray(ho, ho + HAND_SIZE), i * HAND_SIZE);
    }
  }
  return { ...session, people: 1, lm, hands };
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

/** Pack a [x,y,z,visibility]* track into { positions, visibility } base64 strings. */
function packTrack(arr) {
  const total = arr.length / STRIDE;
  const pos = new DataView(new ArrayBuffer(total * 3 * 2));
  const vis = new Uint8Array(total);
  for (let idx = 0; idx < total; idx++) {
    const o = idx * STRIDE;
    for (let c = 0; c < 3; c++) {
      const q = Math.max(-32768, Math.min(32767, Math.round(arr[o + c] * POS_SCALE)));
      pos.setInt16((idx * 3 + c) * 2, q, true);
    }
    vis[idx] = Math.round(Math.max(0, Math.min(1, arr[o + 3])) * 255);
  }
  return { positions: bytesToB64(new Uint8Array(pos.buffer)), visibility: bytesToB64(vis) };
}

function unpackTrack(positionsB64, visibilityB64, total, what) {
  const posBytes = b64ToBytes(positionsB64);
  const visBytes = b64ToBytes(visibilityB64);
  if (posBytes.length !== total * 6 || visBytes.length !== total) {
    throw new Error(`Corrupt session: ${what} data length mismatch`);
  }
  const pos = new DataView(posBytes.buffer, posBytes.byteOffset, posBytes.byteLength);
  const out = new Float32Array(total * STRIDE);
  for (let idx = 0; idx < total; idx++) {
    const o = idx * STRIDE;
    out[o] = pos.getInt16(idx * 6, true) / POS_SCALE;
    out[o + 1] = pos.getInt16(idx * 6 + 2, true) / POS_SCALE;
    out[o + 2] = pos.getInt16(idx * 6 + 4, true) / POS_SCALE;
    out[o + 3] = visBytes[idx] / 255;
  }
  return out;
}

export function serialize(session) {
  const n = session.frameCount;
  const people = session.people || 1;
  const body = packTrack(session.lm);
  const hands = session.hands ? packTrack(session.hands) : null;
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
    engine: session.engine ?? 'mediapipe',
    people,
    trigger: session.trigger,
    frameCount: n,
    landmarkNames: LANDMARK_NAMES,
    handLandmarks: hands ? HAND_LANDMARKS : 0,
    encoding: {
      times: 'milliseconds since the start of the recording, one per frame',
      positions: 'base64 of little-endian int16 triples [x, y, z] per landmark, ordered frame -> person -> landmark; divide by 10000 to get MediaPipe normalised image coordinates',
      visibility: 'base64 of uint8, one per landmark in the same order; divide by 255. A person with visibility 0 everywhere was not detected in that frame',
      movenet: 'when engine is "movenet" only the 17 COCO joints are filled (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles); z is always 0',
      hands: 'when present (MediaPipe Holistic), handPositions/handVisibility hold 42 landmarks per person per frame: the left hand (21 points, MediaPipe hand order) then the right hand, packed like positions/visibility; a hand with visibility 0 was not detected',
    },
    times: Array.from(session.times, t => Math.round(t)),
    positions: body.positions,
    visibility: body.visibility,
    ...(hands ? { handPositions: hands.positions, handVisibility: hands.visibility } : {}),
  });
}

export function deserialize(text) {
  const j = typeof text === 'string' ? JSON.parse(text) : text;
  if (!j || j.format !== FORMAT) throw new Error('Not a Kinesphere session file');
  if (![1, 2, VERSION].includes(j.version)) throw new Error(`Unsupported session version ${j.version}`);
  const n = j.frameCount | 0;
  const people = Math.max(1, j.people | 0 || 1);
  if (!Array.isArray(j.times) || j.times.length !== n) throw new Error('Corrupt session: times');
  const lm = unpackTrack(j.positions, j.visibility, n * people * NUM_LANDMARKS, 'landmark');
  const hands = j.handPositions && j.handVisibility
    ? unpackTrack(j.handPositions, j.handVisibility, n * people * 2 * HAND_LANDMARKS, 'hand')
    : null;
  return {
    id: String(j.id || makeId()),
    name: String(j.name || defaultName(new Date(j.createdAt || Date.now()))),
    createdAt: j.createdAt || new Date().toISOString(),
    durationMs: Number(j.durationMs) || (n ? j.times[n - 1] : 0),
    width: Number(j.width) || 1280,
    height: Number(j.height) || 720,
    mirrored: j.mirrored !== false,
    model: j.model || 'unknown',
    engine: j.engine || 'mediapipe',
    people,
    trigger: j.trigger || 'unknown',
    frameCount: n,
    times: Float64Array.from(j.times),
    lm, hands,
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
    people: session.people || 1, engine: session.engine || 'mediapipe', model: session.model, hands: Boolean(session.hands),
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

/** One row per frame (and per detected person when the session tracked several). */
export function toCSV(session) {
  const people = session.people || 1;
  const header = people > 1 ? ['person', 't_ms'] : ['t_ms'];
  for (const name of LANDMARK_NAMES) header.push(`${name}_x`, `${name}_y`, `${name}_z`, `${name}_visibility`);
  if (session.hands) {
    for (const side of ['left_hand', 'right_hand']) {
      for (let k = 0; k < HAND_LANDMARKS; k++) header.push(`${side}_${k}_x`, `${side}_${k}_y`, `${side}_${k}_z`, `${side}_${k}_visibility`);
    }
  }
  const rows = [header.join(',')];
  const { lm, hands, times, frameCount } = session;
  for (let i = 0; i < frameCount; i++) {
    for (let p = 0; p < people; p++) {
      const base = frameOffset(session, i, p);
      if (people > 1 && !personPresent(lm, base)) continue;
      const cells = people > 1 ? [p + 1, Math.round(times[i])] : [Math.round(times[i])];
      for (let j = 0; j < FRAME_SIZE; j++) cells.push(lm[base + j].toFixed(4));
      if (hands) {
        const hb = handOffset(session, i, p);
        for (let j = 0; j < HAND_SIZE; j++) cells.push(hands[hb + j].toFixed(4));
      }
      rows.push(cells.join(','));
    }
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
