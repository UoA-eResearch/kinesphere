// AI dance partner: your movement goes to the Altlando dance model, running in the browser
// (js/altlando.js), as 3D SMPL joints, and the partner dances back what the model generates.
//
// The model (Zhou 2025, "Interactive AI Dancer", a fork of Bailando with the music condition
// removed: github.com/jzho987/Bailando, branch feat/remove-music-condition) works on AIST++-style
// motion: 24 SMPL joint positions per frame (metres, y up, root absolute and the other joints
// absolute too), at 60 frames a second. Its VQ-VAE turns every 8 frames into one pose code for
// the upper and one for the lower body; the GPT continues a sequence of codes. Given shift * 8
// frames of your movement it returns (shift + length) * 8 frames: first its reading of your
// movement (its "mimicry" phase), then its own continuation.
//
// Conversions here: MediaPipe world landmarks (or 2D landmarks, lifted) -> SMPL-24 joints, and
// SMPL-24 joints -> MediaPipe-style landmarks so every overlay style can draw the partner.

import { WORLD_JOINTS } from './pose.js';

export const SMPL_JOINTS = ['pelvis', 'l_hip', 'r_hip', 'spine1', 'l_knee', 'r_knee', 'spine2', 'l_ankle', 'r_ankle', 'spine3',
  'l_foot', 'r_foot', 'neck', 'l_collar', 'r_collar', 'head', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist', 'l_hand', 'r_hand'];
export const SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21];
export const FPS = 60;
export const CODE_FRAMES = 8; // frames per VQ-VAE pose code
export const DEFAULTS = { shift: 28, length: 28, mimic: false };
const HFOV_TAN = Math.tan((65 / 2) * Math.PI / 180); // the analysis' webcam field of view

// ---- small vector helpers (arrays of 3) --------------------------------------------------------
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b, k = 1) => [a[0] + b[0] * k, a[1] + b[1] * k, a[2] + b[2] * k];
const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
const len = a => Math.hypot(a[0], a[1], a[2]);
const norm = a => { const l = len(a) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const at = (J, i) => [J[i * 3], J[i * 3 + 1], J[i * 3 + 2]];

const W = Object.fromEntries(['nose', 'shL', 'shR', 'elL', 'elR', 'wrL', 'wrR', 'hipL', 'hipR', 'knL', 'knR', 'anL', 'anR'].map((k, j) => [k, j]));
if (WORLD_JOINTS.length !== 13) throw new Error('ai-partner expects 13 world joints');

/**
 * Your body from MediaPipe world landmarks (13 joints, metres, hip-centred, y down, z away from the
 * camera), in SMPL axes: x to your left, y up, z towards the camera (the way you face).
 */
export function bodyFromWorld(world) {
  if (!world) return null;
  const P = {};
  for (const [k, j] of Object.entries(W)) P[k] = [world[j * 3], -world[j * 3 + 1], -world[j * 3 + 2]];
  if (![P.shL, P.shR, P.hipL, P.hipR].every(p => p.every(Number.isFinite)) || len(sub(P.shL, P.hipL)) < 0.05) return null;
  return P;
}

/**
 * Your body from 2D landmarks only (MoveNet, or MediaPipe without world landmarks): lifted onto a
 * flat plane facing the camera, scaled so the torso is `torsoM` metres.
 */
export function bodyFromLandmarks(lm, aspect = 16 / 9, minVis = 0.3, torsoM = 0.5) {
  const ok = i => lm[i * 4 + 3] >= minVis;
  if (![11, 12, 23, 24].every(ok)) return null;
  const hx = (lm[92] + lm[96]) / 2, hy = (lm[93] + lm[97]) / 2, sx = (lm[44] + lm[48]) / 2, sy = (lm[45] + lm[49]) / 2;
  const torso = Math.hypot((sx - hx) * aspect, sy - hy);
  if (torso < 0.02) return null;
  const m = torsoM / torso;
  const pt = i => [(lm[i * 4] - hx) * aspect * m, -(lm[i * 4 + 1] - hy) * m, 0];
  const P = {};
  const src = { nose: 0, shL: 11, shR: 12, elL: 13, elR: 14, wrL: 15, wrR: 16, hipL: 23, hipR: 24, knL: 25, knR: 26, anL: 27, anR: 28 };
  for (const [k, i] of Object.entries(src)) P[k] = ok(i) ? pt(i) : null;
  // limbs out of view hang straight down from the joint above
  const fill = (k, from, drop) => { P[k] ??= P[from] ? add(P[from], [0, -drop, 0]) : null; };
  fill('elL', 'shL', 0.28); fill('elR', 'shR', 0.28); fill('wrL', 'elL', 0.26); fill('wrR', 'elR', 0.26);
  fill('knL', 'hipL', 0.42); fill('knR', 'hipR', 0.42); fill('anL', 'knL', 0.42); fill('anR', 'knR', 0.42);
  P.nose ??= add(mix(P.shL, P.shR, 0.5), [0, 0.25, 0]);
  return P;
}

/**
 * SMPL-24 joint positions (Float32Array(72), metres, y up) from a body (bodyFromWorld /
 * bodyFromLandmarks), with the pelvis at `root`. The spine, neck, collars, head, hands and feet,
 * which the camera doesn't track, are placed from the tracked joints in SMPL's proportions.
 */
export function smplFromBody(P, root = [0, 0.95, 0]) {
  const pel = mix(P.hipL, P.hipR, 0.5), neckBase = mix(P.shL, P.shR, 0.5);
  const up = norm(sub(neckBase, pel));
  let across = sub(P.hipL, P.hipR);
  if (len(across) < 0.05) across = sub(P.shL, P.shR);
  const fwd = norm(cross(norm(across), up));
  const flatFwd = norm([fwd[0], 0, fwd[2]].every(v => Math.abs(v) < 1e-6) ? [0, 0, 1] : [fwd[0], 0, fwd[2]]);
  const J = new Array(24);
  const o = sub(root, pel); // move the pelvis to the root
  const put = (i, p) => { J[i] = add(p, o); };
  put(0, pel);
  put(1, P.hipL); put(2, P.hipR);
  put(3, mix(pel, neckBase, 0.22)); put(6, mix(pel, neckBase, 0.45)); put(9, mix(pel, neckBase, 0.66));
  put(4, P.knL); put(5, P.knR); put(7, P.anL); put(8, P.anR);
  put(10, add(add(P.anL, flatFwd, 0.12), [0, -0.05, 0])); put(11, add(add(P.anR, flatFwd, 0.12), [0, -0.05, 0]));
  const neck = add(neckBase, up, 0.04);
  put(12, neck);
  put(13, mix(neckBase, P.shL, 0.4)); put(14, mix(neckBase, P.shR, 0.4));
  const head = P.nose ? add(add(P.nose, fwd, -0.08), up, -0.01) : add(neck, up, 0.2);
  put(15, len(sub(head, neck)) > 0.35 ? add(neck, norm(sub(head, neck)), 0.22) : head);
  put(16, P.shL); put(17, P.shR); put(18, P.elL); put(19, P.elR); put(20, P.wrL); put(21, P.wrR);
  put(22, add(P.wrL, sub(P.wrL, P.elL), 0.35)); put(23, add(P.wrR, sub(P.wrR, P.elR), 0.35));
  const out = new Float32Array(72);
  J.forEach((p, i) => { out[i * 3] = p[0]; out[i * 3 + 1] = p[1]; out[i * 3 + 2] = p[2]; });
  return out;
}

/**
 * Turns a stream of detections into SMPL frames with a believable root: the pelvis height from the
 * legs, and the position on the floor from where your hips are in the picture and how big your
 * torso looks (a pinhole camera with the analysis' 65° field of view), smoothed.
 */
export function createUserConverter({ smoothing = 0.25 } = {}) {
  let root = null;
  return {
    /** pose: { lm, world } from the detector; aspect: the camera picture's width / height. */
    convert(pose, aspect = 16 / 9) {
      const P = bodyFromWorld(pose.world) ?? bodyFromLandmarks(pose.lm, aspect);
      if (!P) return null;
      const lm = pose.lm;
      const hx = (lm[92] + lm[96]) / 2, hy = (lm[93] + lm[97]) / 2, sx = (lm[44] + lm[48]) / 2, sy = (lm[45] + lm[49]) / 2;
      const torsoImg = Math.hypot((sx - hx) * aspect, sy - hy);
      const torsoM = Math.min(0.7, Math.max(0.3, len(sub(mix(P.shL, P.shR, 0.5), mix(P.hipL, P.hipR, 0.5)))));
      const tanV = HFOV_TAN / aspect;
      const depth = torsoImg > 0.02 ? Math.min(8, Math.max(0.8, torsoM / (torsoImg * 2 * tanV))) : 3;
      const x = (hx - 0.5) * 2 * depth * HFOV_TAN;
      const pel = mix(P.hipL, P.hipR, 0.5);
      const legs = Math.max(pel[1] - P.anL[1], pel[1] - P.anR[1]);
      const height = legs > 0.4 && legs < 1.2 ? legs + 0.08 : 0.95;
      const target = [x, height, -depth];
      root = root ? root.map((v, i) => v + (target[i] - v) * smoothing) : target;
      return smplFromBody(P, root);
    },
    reset() { root = null; },
  };
}

/** Linear resampling of an irregular stream of frames to a fixed rate. */
export function createResampler(fps = FPS, maxGapMs = 300) {
  const step = 1000 / fps;
  let last = null, next = 0;
  return {
    push(t, frame, emit) {
      if (!last || t <= last.t || t - last.t > maxGapMs) {
        last = { t, f: frame }; emit(frame); next = t + step;
        return;
      }
      while (next <= t) {
        const a = (next - last.t) / (t - last.t), f = new Float32Array(frame.length);
        for (let i = 0; i < f.length; i++) f[i] = last.f[i] + (frame[i] - last.f[i]) * a;
        emit(f);
        next += step;
      }
      last = { t, f: frame };
    },
    reset() { last = null; },
  };
}

/**
 * MediaPipe-style landmarks (Float32Array(132), x and y in 0..1 of a virtual camera picture) for
 * SMPL joints, so the overlay styles can draw the partner. The camera stands `camDist` metres in
 * front of `centre` (the partner faces it), 1 m up.
 */
export function landmarksFromSmpl(J, { aspect = 16 / 9, camDist = 4, centre = [0, 0, 0] } = {}) {
  const lm = new Float32Array(132);
  const tanV = HFOV_TAN / aspect;
  const put = (i, p, v = 0.95) => {
    const d = Math.max(0.5, camDist - (p[2] - centre[2]));
    lm[i * 4] = 0.5 + (p[0] - centre[0]) / (d * 2 * HFOV_TAN);
    lm[i * 4 + 1] = 0.5 - (p[1] - 1) / (d * 2 * tanV);
    lm[i * 4 + 2] = -(p[2] - centre[2]);
    lm[i * 4 + 3] = v;
  };
  const j = i => at(J, i);
  const across = norm(sub(j(1), j(2))), up = norm(sub(j(12), j(0)));
  const fwd = norm(cross(across, up));
  const head = j(15);
  const face = (f, a, u) => add(add(add(head, fwd, f), across, a), up, u);
  put(0, face(0.1, 0, 0));
  put(1, face(0.08, 0.02, 0.035)); put(2, face(0.075, 0.035, 0.035)); put(3, face(0.07, 0.05, 0.035));
  put(4, face(0.08, -0.02, 0.035)); put(5, face(0.075, -0.035, 0.035)); put(6, face(0.07, -0.05, 0.035));
  put(7, face(0, 0.08, 0.01)); put(8, face(0, -0.08, 0.01));
  put(9, face(0.085, 0.025, -0.04)); put(10, face(0.085, -0.025, -0.04));
  const pairs = [[11, 16], [12, 17], [13, 18], [14, 19], [15, 20], [16, 21], [23, 1], [24, 2], [25, 4], [26, 5], [27, 7], [28, 8], [31, 10], [32, 11]];
  for (const [m, s] of pairs) put(m, j(s));
  for (const [m, s] of [[17, 22], [19, 22], [21, 22], [18, 23], [20, 23], [22, 23]]) put(m, j(s), 0.8);
  put(29, add(j(7), sub(j(7), j(10)), 0.4)); put(30, add(j(8), sub(j(8), j(11)), 0.4)); // heels
  return lm;
}

/** MediaPipe-style world landmarks (13 joints, hip-centred, y down) for SMPL joints: for recording. */
export function worldFromSmpl(J) {
  const out = new Float32Array(WORLD_JOINTS.length * 3);
  const pel = at(J, 0);
  const src = [15, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8];
  src.forEach((s, k) => {
    const p = at(J, s);
    out[k * 3] = p[0] - pel[0]; out[k * 3 + 1] = -(p[1] - pel[1]); out[k * 3 + 2] = -(p[2] - pel[2]);
  });
  return out;
}

/** Bone lengths of an SMPL frame (metres), for sanity checks. */
export function boneLengths(J) {
  return SMPL_PARENTS.map((p, i) => (p < 0 ? 0 : len(sub(at(J, i), at(J, p)))));
}

/**
 * The partner: collects your movement (60 fps SMPL frames), asks the model for a reply when it has
 * enough (shift * 8 frames, 3.7 s by default) and plays the reply back, cross-fading from one
 * reply to the next and keeping the partner near the middle of its spot on the floor.
 *
 *   const partner = createPartner({ generate: (frames, { shift, length }) => model.generate(frames, length) });
 *   partner.start();
 *   partner.push(performance.now(), smplFrame);   // every detection (any rate)
 *   const J = partner.frameAt(performance.now()); // SMPL joints to show, or null
 */
export function createPartner({
  generate, shift = DEFAULTS.shift, length = DEFAULTS.length, mimic = DEFAULTS.mimic,
  onStatus = () => {}, fadeFrames = 18, retryMs = 3000,
} = {}) {
  const need = shift * CODE_FRAMES;
  const input = []; // recent 60 fps frames of you
  const resampler = createResampler(FPS);
  let active = false, inFlight = null, queue = [], head = 0, playAt = null, shown = null;
  let latency = 1500, requests = 0, failures = 0, retryAt = 0, error = null, status = 'off', lastFrameAt = 0;
  const setStatus = (s, extra = {}) => {
    if (s !== status || extra.error !== undefined) { status = s; onStatus({ status, error, latency, requests, ...extra }); }
  };

  async function request() {
    const frames = input.slice(-need);
    const t0 = Date.now();
    try {
      const result = await generate(frames, { shift, length });
      if (!Array.isArray(result) || !result.length || result[0].length !== 72) throw new Error('the model returned no dance');
      latency = latency * 0.5 + (Date.now() - t0) * 0.5;
      requests++; failures = 0; error = null;
      if (active) receive(result.map(f => Float32Array.from(f)));
    } catch (err) {
      failures++;
      error = err?.message ?? String(err);
      retryAt = Date.now() + Math.min(30000, retryMs * 2 ** (failures - 1));
      setStatus('error', { error });
    } finally {
      inFlight = null;
    }
  }

  /** A reply from the model: re-anchor it where the partner is now, and fade into it. */
  function receive(frames) {
    const clip = mimic ? frames : frames.slice(Math.min(frames.length - 1, need));
    const from = shown ? [shown[0], shown[2]] : [0, 0];
    // drift back towards the middle of the partner's spot between replies
    const target = [from[0] * 0.6, from[1] * 0.6];
    const dx = target[0] - clip[0][0], dz = target[1] - clip[0][2];
    for (const f of clip) for (let i = 0; i < 24; i++) { f[i * 3] += dx; f[i * 3 + 2] += dz; }
    if (shown) {
      const n = Math.min(fadeFrames, clip.length);
      for (let k = 0; k < n; k++) {
        const a = (k + 1) / (n + 1), w = a * a * (3 - 2 * a);
        const f = clip[k];
        for (let i = 0; i < 72; i++) f[i] = shown[i] + (f[i] - shown[i]) * w;
      }
    }
    queue = clip; head = 0; playAt = null;
    setStatus('dancing');
  }

  return {
    start() {
      if (active) return;
      active = true; error = null; failures = 0; retryAt = 0;
      setStatus('watching');
    },
    stop() {
      active = false; queue = []; shown = null; input.length = 0; resampler.reset();
      setStatus('off');
    },
    get active() { return active; },
    /** Your latest pose (SMPL joints, any frame rate). */
    push(t, frame) {
      if (!active || !frame) return;
      lastFrameAt = t;
      resampler.push(t, frame, f => {
        input.push(f);
        if (input.length > need * 2) input.splice(0, input.length - need * 2);
      });
    },
    /** The partner's pose at time t (SMPL joints), or null before its first reply. */
    frameAt(t) {
      if (!active) return null;
      const remaining = queue.length - head;
      const leadFrames = (latency / 1000) * FPS + 20;
      const haveInput = input.length >= need && t - lastFrameAt < 1000;
      if (!inFlight && haveInput && Date.now() >= retryAt && remaining < leadFrames) {
        if (status !== 'dancing' || remaining <= 0) setStatus('thinking');
        inFlight = request();
      } else if (!inFlight && !queue.length && status !== 'error') {
        setStatus('watching', { progress: Math.min(1, input.length / need) });
      }
      if (!queue.length) return shown;
      playAt ??= t;
      head = Math.min(queue.length - 1, Math.floor(((t - playAt) / 1000) * FPS));
      shown = queue[head];
      return shown;
    },
    info() {
      return { status, error, latency, requests, inFlight: Boolean(inFlight), buffered: input.length, need, queued: Math.max(0, queue.length - head) };
    },
    /** For waiting in tests: resolves when the request in flight (if any) has finished. */
    settle() { return inFlight ?? Promise.resolve(); },
  };
}

/** A recorded person's frames as Altlando input: SMPL-24 joints at 60 fps (for export). */
export function smplSequence(frames, aspect = 16 / 9) {
  const conv = createUserConverter({ smoothing: 0.25 });
  const rs = createResampler(FPS, 1000);
  const out = [];
  for (const { t, lm, world } of frames) {
    const J = lm ? conv.convert({ lm, world }, aspect) : null;
    if (J) rs.push(t, J, f => out.push(Array.from(f, v => Math.round(v * 1e4) / 1e4)));
  }
  return out;
}
