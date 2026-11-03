// Synthetic sessions with known ground truth, built in memory for the tests:
//   phasesSession(people)  ten 10 s phases with known kinesphere sizes and space levels (PHASES)
//   floorSession(secs)     a dancer walking a known square on the floor, with tracking glitches
// Kinesphere / space-level ground truth:
// Body units: torso lengths, waist (mid-hip) at the origin, y down, +x = the dancer's LEFT.
// The camera is not mirrored, so the dancer's left appears on the image's right.
import { buildSession, serialize } from '../../js/session.js';
import { FRAME_SIZE, STRIDE, WORLD_JOINTS, WORLD_SIZE } from '../../js/pose.js';
export { serialize };

const W = 1280, H = 720, aspect = W / H, fps = 30;
const UA = 0.6, FA = 0.55, TH = 0.9, SH = 0.85;
const rad = d => d * Math.PI / 180;
// Phases: [name, expected kinesphere, expected level]
// Expected classes (torso-centre spheres sized from the arms, nearest-sphere rule):
// [name, kinesphere measured in 3D, level, kinesphere measured in 2D (no world landmarks)]
export const PHASES = [
  ['arms down', 'small', 'mid', 'small'],
  ['arms 45 deg', 'medium', 'mid', 'medium'],
  ['arms just above horizontal', 'large', 'high', 'large'],
  ['arms up', 'large', 'high', 'large'],
  ['deep squat, hands low', 'small', 'low', 'small'],
  ['side kick (dancer right leg)', 'large', 'mid', 'large'],
  ['arms folded across chest', 'small', 'mid', 'small'],
  ['arms forward and up, to camera', 'large', 'high', 'medium'],
  ['close to camera, feet out of view', 'small', 'mid', 'small'],
  ['close to camera, arms up', 'large', 'high', 'large'],
];

function pose(phase, t) {
  const P = {};
  let drop = 0; // waist lowered (crouch)
  const wob = (amp, f, ph = 0) => amp * Math.sin(t * f + ph);
  // arm abduction (0 = down, 180 = up) per side; right arm moves more than left
  const armAngle = [8, 45, 92, 170, 0, 8, 0, 0, 8, 120][phase];
  const thL = armAngle + wob(3, 3), thR = armAngle + wob(phase === 2 ? 3 : 12, 4, 1);
  if (phase === 4) drop = 1.2;
  P.waist = [0, drop];
  P.hipL = [0.2, drop]; P.hipR = [-0.2, drop];
  P.shL = [0.4, drop - 1]; P.shR = [-0.4, drop - 1];
  P.nose = [0, drop - 1.45];
  const arm = (sh, side, th) => {
    const dir = [side * Math.sin(rad(th)), Math.cos(rad(th))];
    const el = [sh[0] + UA * dir[0], sh[1] + UA * dir[1]];
    return [el, [el[0] + FA * dir[0], el[1] + FA * dir[1]]];
  };
  if (phase === 4) { // deep squat, hands down by the shins
    P.elL = [0.45, 0.79]; P.wrL = [0.5, 1.34 + wob(0.03, 5)]; P.elR = [-0.45, 0.79]; P.wrR = [-0.5, 1.34 + wob(0.05, 6)];
  } else if (phase === 7) { // straight arms raised 45 deg forwards (towards the camera, -z) and up
    const e = rad(45 + wob(3, 3));
    for (const [side, sh, el, wr] of [[1, 'shL', 'elL', 'wrL'], [-1, 'shR', 'elR', 'wrR']]) {
      const d = [0, -Math.sin(e), -Math.cos(e)];
      P[el] = [P[sh][0], P[sh][1] + UA * d[1], UA * d[2]];
      P[wr] = [P[sh][0], P[sh][1] + (UA + FA) * d[1], (UA + FA) * d[2]];
    }
  } else if (phase === 6) { // forearms folded across the chest
    P.elL = [0.45, -0.41]; P.wrL = [-0.1 + wob(0.03, 3), -0.55]; P.elR = [-0.45, -0.41]; P.wrR = [0.1 + wob(0.05, 4), -0.5];
  } else {
    [P.elL, P.wrL] = arm(P.shL, 1, thL); [P.elR, P.wrR] = arm(P.shR, -1, thR);
  }
  // legs: feet stay on the floor at y = 1.75 unless kicking
  P.anL = [0.22, 1.75]; P.anR = [-0.22, 1.75];
  if (phase === 4) { P.knL = [1.06, 1.45]; P.knR = [-1.06, 1.45]; }
  else { P.knL = [0.22, 0.9]; P.knR = [-0.22, 0.9]; }
  if (phase === 5) {
    const a = rad(70 + wob(8, 3));
    const dir = [-Math.sin(a), Math.cos(a)];
    P.knR = [-0.2 + TH * dir[0], TH * dir[1]];
    P.anR = [P.knR[0] + SH * dir[0], P.knR[1] + SH * dir[1]];
  }
  return P;
}

function toLandmarks(P, cx, cy, T) {
  const lm = new Float32Array(FRAME_SIZE);
  const put = (i, [x, y], v = 1) => {
    const px = cx + x * T / aspect, py = cy + y * T;
    lm[i * STRIDE] = px; lm[i * STRIDE + 1] = py;
    lm[i * STRIDE + 3] = py > 1 || py < 0 || px < 0 || px > 1 ? 0.05 : v; // off-screen: MediaPipe still guesses, with low visibility
  };
  put(0, P.nose);
  for (let k = 1; k <= 10; k++) put(k, [P.nose[0] + (k <= 3 || k === 7 || k === 9 ? 0.1 : -0.1), P.nose[1] - 0.03]);
  put(11, P.shL); put(12, P.shR); put(13, P.elL); put(14, P.elR); put(15, P.wrL); put(16, P.wrR);
  for (const [k, w] of [[17, P.wrL], [19, P.wrL], [21, P.wrL], [18, P.wrR], [20, P.wrR], [22, P.wrR]]) put(k, [w[0], w[1] + 0.08], 0.9);
  put(23, P.hipL); put(24, P.hipR); put(25, P.knL); put(26, P.knR); put(27, P.anL); put(28, P.anR);
  put(29, [P.anL[0], P.anL[1] + 0.05]); put(30, [P.anR[0], P.anR[1] + 0.05]);
  put(31, [P.anL[0] + 0.05, P.anL[1] + 0.07]); put(32, [P.anR[0] - 0.05, P.anR[1] + 0.07]);
  return lm;
}

// World landmarks: metres (torso = 0.5 m), origin at the hip midpoint, z towards the camera negative.
function toWorld(P) {
  const map = { 0: 'nose', 11: 'shL', 12: 'shR', 13: 'elL', 14: 'elR', 15: 'wrL', 16: 'wrR', 23: 'hipL', 24: 'hipR', 25: 'knL', 26: 'knR', 27: 'anL', 28: 'anR' };
  const out = new Float32Array(WORLD_SIZE);
  WORLD_JOINTS.forEach((k, j) => {
    const p = P[map[k]];
    out[j * 3] = p[0] * 0.5; out[j * 3 + 1] = (p[1] - P.waist[1]) * 0.5; out[j * 3 + 2] = (p[2] || 0) * 0.5;
  });
  return out;
}

function make(name, people, phaseSeconds = 10) {
  const frames = [];
  const total = PHASES.length * phaseSeconds;
  for (let i = 0; i < total * fps; i++) {
    const t = i / fps, phase = Math.min(PHASES.length - 1, Math.floor(t / phaseSeconds));
    const lm = new Float32Array(people * FRAME_SIZE);
    const world = new Float32Array(people * WORLD_SIZE);
    for (let p = 0; p < people; p++) {
      if (p === 1 && t < 10) continue; // second dancer enters after 10 s
      const cx = people === 1 ? 0.5 + 0.1 * Math.sin(t / 6) : (p === 0 ? 0.3 : 0.7);
      const ph = p === 0 ? phase : (phase + 3) % PHASES.length;
      const P = pose(ph, t);
      const close = ph >= 8;
      lm.set(toLandmarks(P, cx, close ? 0.62 : 0.38, close ? 0.34 : 0.17), p * FRAME_SIZE);
      world.set(toWorld(P), p * WORLD_SIZE);
    }
    frames.push({ t: t * 1000, lm, world });
  }
  const s = buildSession({ frames, width: W, height: H, model: 'lite', engine: 'mediapipe', people, trigger: 'button', startedAt: Date.now() - total * 1000 });
  s.name = name;
  return s;
}

/** One dancer (or two, the second entering after 10 s) through PHASES, 10 s each. */
export function phasesSession(people = 1) {
  return make(people === 1 ? 'Synthetic phases' : 'Synthetic duet', people);
}

// ---- floor path ---------------------------------------------------------------------------------
// A dancer walking a square 2 m wide, 2.5 m to 4 m from a pinhole camera (65 deg HFOV, 16:9), one
// lap per 40 s, with 3 s too close to the camera (1 m, head out of frame) at 30% of the session and
// injected tracking glitches (a wrist teleporting for 2 frames, the whole skeleton jumping sideways).
const tanH = Math.tan(32.5 * Math.PI / 180), tanV = tanH / aspect, CAM_H = 1.0;
function walker(t) {
  const sw = Math.sin(t * 4) * 0.6;
  const B = {
    0: [0, 1.72], 11: [-0.2, 1.5], 12: [0.2, 1.5], 23: [-0.13, 1.0], 24: [0.13, 1.0],
    25: [-0.13, 0.55], 26: [0.13, 0.55], 27: [-0.13, 0.08], 28: [0.13, 0.08],
  };
  const arm = (sh, side, a) => { const el = [sh[0] + side * 0.28 * Math.sin(a), sh[1] - 0.28 * Math.cos(a)]; return [el, [el[0] + side * 0.26 * Math.sin(a), el[1] - 0.26 * Math.cos(a)]]; };
  [B[13], B[15]] = arm(B[11], -1, 0.3 + Math.abs(sw));
  [B[14], B[16]] = arm(B[12], 1, 0.3 + Math.abs(sw * 1.3));
  return B;
}
export function floorAt(t, secs) {
  if (t > 0.3 * secs && t < 0.3 * secs + 3) return { x: 0, z: 1.0 };
  const ph = (t % 40) / 40 * 4, s = ph % 1;
  const corners = [[-1, 2.5], [1, 2.5], [1, 4], [-1, 4], [-1, 2.5]];
  const a = corners[Math.floor(ph)], b = corners[Math.floor(ph) + 1];
  return { x: a[0] + (b[0] - a[0]) * s, z: a[1] + (b[1] - a[1]) * s };
}
export function floorSession(secs = 60, { glitches = true } = {}) {
  const frames = [];
  for (let i = 0; i < secs * fps; i++) {
    const t = i / fps, B = walker(t), f = floorAt(t, secs);
    const lm = new Float32Array(FRAME_SIZE);
    for (let k = 0; k < 33; k++) {
      const p = B[k] ?? B[0];
      const camX = -(f.x + p[0]);
      const u = 0.5 + camX / (2 * f.z * tanH), v = 0.5 + (CAM_H - p[1]) / (2 * f.z * tanV);
      const inside = u >= 0 && u <= 1 && v >= 0 && v <= 1;
      lm.set([u, v, 0, inside ? 0.95 : 0.1], k * STRIDE);
    }
    if (glitches && i % 211 === 100) lm[16 * STRIDE] += 0.35;
    if (glitches && i % 211 === 101) lm[16 * STRIDE] += 0.33;
    if (glitches && i % 509 === 300) for (let k = 0; k < 33; k++) lm[k * STRIDE] += 0.2;
    const world = new Float32Array(WORLD_SIZE);
    WORLD_JOINTS.forEach((k, j) => { const p = B[k] ?? B[0]; world.set([p[0], 1.0 - p[1], 0], j * 3); });
    frames.push({ t: t * 1000, lm, hands: null, world });
  }
  const s = buildSession({ frames, width: W, height: H, model: 'lite', engine: 'mediapipe', people: 1, trigger: 'button', startedAt: Date.now() - secs * 1000, mirrored: true });
  s.name = `Floor test ${secs}s`;
  return s;
}
