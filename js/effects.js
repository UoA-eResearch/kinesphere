// Alternative ways of drawing the detected pose: plain skeleton, stick figure, neon glow,
// light trails, sparks (particles), a constellation, sandman, fuzzy blur, origami and pencil.
// Every style exposes
//   draw(ctx, lm, offset, width, height, dt, { mirror, minVis, person, videoVisible, ... })
// and keeps its own state (particles, trails) so it animates smoothly between detections.

import { CONNECTIONS, NUM_LANDMARKS, STRIDE, drawSkeleton, drawHandSkeleton, drawFaceContours } from './pose.js';

/** Colour set per tracked person (slot). Person 1 keeps the classic orange/blue. */
export const PERSON_COLORS = [
  { left: '#eb6834', right: '#3987e5', mid: '#f4f4f2', body: '#ffffff', hue: 0 },
  { left: '#f2c14e', right: '#9b6cf0', mid: '#fff6d8', body: '#fff0b8', hue: 55 },
  { left: '#f06fb0', right: '#3ecf8e', mid: '#ffe3f0', body: '#ffd3e8', hue: 125 },
  { left: '#ff9f43', right: '#00c2d1', mid: '#ffefdc', body: '#ffe2bf', hue: 190 },
  { left: '#c0f050', right: '#5f7cff', mid: '#f1ffd8', body: '#e2ffb0', hue: 250 },
  { left: '#ff6b6b', right: '#4dd0e1', mid: '#ffe0e0', body: '#ffc4c4', hue: 310 },
];
export const personColors = p => PERSON_COLORS[(p ?? 0) % PERSON_COLORS.length];

export const STYLES = [
  { id: 'skeleton', label: 'Skeleton' },
  { id: 'stickman', label: 'Stick figure' },
  { id: 'neon', label: 'Neon glow' },
  { id: 'trails', label: 'Light trails' },
  { id: 'sparks', label: 'Sparks' },
  { id: 'constellation', label: 'Constellation' },
  { id: 'sandman', label: 'Sandman (falling particles)' },
  { id: 'fuzzy', label: 'Fuzzy blur' },
  { id: 'origami', label: 'Origami (paper)' },
  { id: 'pencil', label: 'Pencil sketch' },
];
export const DEFAULT_STYLE = 'skeleton';

const BODY = new Set([11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]);
const BODY_BONES = CONNECTIONS.filter(([a, b]) => BODY.has(a) && BODY.has(b));
const MAJOR = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
const EMITTERS = [0, 13, 14, 15, 16, 25, 26, 27, 28];
const isLeft = i => i >= 11 && i % 2 === 1;
const hsla = (h, s, l, a = 1) => `hsla(${h.toFixed(0)},${s}%,${l}%,${a.toFixed(3)})`;
const rand = (a, b) => a + Math.random() * (b - a);

/** Landmarks in pixel coordinates: { x, y, v } arrays of length 33. */
function project(lm, offset, w, h, mirror) {
  const x = new Float32Array(NUM_LANDMARKS), y = new Float32Array(NUM_LANDMARKS), v = new Float32Array(NUM_LANDMARKS);
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    const o = offset + i * STRIDE;
    const nx = mirror ? 1 - lm[o] : lm[o];
    x[i] = nx * w; y[i] = lm[o + 1] * h; v[i] = lm[o + 3];
  }
  return { x, y, v };
}

/**
 * Tracks joint velocity (px/s) and acceleration (px/s^2) over time, smoothed with exponential
 * filters (time constants in seconds) so the estimate is steady whatever the frame rate.
 */
function makeTracker(tauV = 0.06, tauA = 0.1) {
  let prev = null;
  const vx = new Float32Array(NUM_LANDMARKS), vy = new Float32Array(NUM_LANDMARKS);
  const ax = new Float32Array(NUM_LANDMARKS), ay = new Float32Array(NUM_LANDMARKS);
  return {
    vx, vy, ax, ay,
    update(p, dt) {
      if (prev && dt > 0 && dt < 0.5) {
        const kv = 1 - Math.exp(-dt / tauV), ka = 1 - Math.exp(-dt / tauA);
        for (let i = 0; i < NUM_LANDMARKS; i++) {
          const nvx = vx[i] + ((p.x[i] - prev.x[i]) / dt - vx[i]) * kv;
          const nvy = vy[i] + ((p.y[i] - prev.y[i]) / dt - vy[i]) * kv;
          ax[i] += ((nvx - vx[i]) / dt - ax[i]) * ka;
          ay[i] += ((nvy - vy[i]) / dt - ay[i]) * ka;
          vx[i] = nvx; vy[i] = nvy;
        }
      } else { vx.fill(0); vy.fill(0); ax.fill(0); ay.fill(0); }
      prev = p;
    },
    reset() { prev = null; vx.fill(0); vy.fill(0); ax.fill(0); ay.fill(0); },
  };
}

function drawBones(ctx, p, minVis, bones, strokeFor, width) {
  ctx.lineWidth = width;
  for (const [a, b] of bones) {
    if (p.v[a] < minVis || p.v[b] < minVis) continue;
    ctx.strokeStyle = strokeFor(a, b);
    ctx.beginPath();
    ctx.moveTo(p.x[a], p.y[a]);
    ctx.lineTo(p.x[b], p.y[b]);
    ctx.stroke();
  }
}

function neckAndHead(p, minVis) {
  if (p.v[11] < minVis || p.v[12] < minVis || p.v[0] < minVis) return null;
  const nx = (p.x[11] + p.x[12]) / 2, ny = (p.y[11] + p.y[12]) / 2;
  const ears = p.v[7] >= minVis && p.v[8] >= minVis ? Math.hypot(p.x[7] - p.x[8], p.y[7] - p.y[8]) : 0;
  const r = Math.max(ears * 0.75, Math.hypot(p.x[0] - nx, p.y[0] - ny) * 0.45, 6);
  return { nx, ny, hx: p.x[0], hy: p.y[0] - r * 0.15, r };
}

/**
 * Hands (recorded) and face contours (live only) that Holistic adds on top of the pose.
 * `mode` picks a look: 'full' (coloured), 'ghost' (faint white) or 'glow' (neon).
 */
function drawExtras(ctx, w, h, opts, mode, hue = 0) {
  const { hands = null, face = null, faceContours = null, mirror = false, minVis = 0.5, person = 0 } = opts;
  if (!hands && !face) return;
  const pc = personColors(person);
  const thin = Math.max(1.2, w / 640);
  if (mode === 'ghost') {
    drawHandSkeleton(ctx, hands, w, h, { mirror, minVis, lineWidth: thin, radius: thin, leftColor: '#fff', rightColor: '#fff', alpha: 0.35 });
    return;
  }
  if (mode === 'glow') {
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.shadowBlur = 10;
    ctx.shadowColor = hsla(hue, 100, 60);
    drawHandSkeleton(ctx, hands, w, h, { mirror, minVis, lineWidth: thin * 1.5, radius: thin, leftColor: hsla(hue + 70, 100, 70), rightColor: hsla(hue - 70, 100, 70) });
    drawFaceContours(ctx, face, faceContours, w, h, { mirror, color: hsla(hue, 100, 75), lineWidth: thin, alpha: 0.7 });
    ctx.restore();
    return;
  }
  drawHandSkeleton(ctx, hands, w, h, { mirror, minVis, lineWidth: thin * 1.5, radius: thin * 1.3, leftColor: pc.left, rightColor: pc.right });
  drawFaceContours(ctx, face, faceContours, w, h, { mirror, color: pc.mid, lineWidth: thin, alpha: 0.7 });
}

// ---- styles -------------------------------------------------------------------------

function skeletonStyle() {
  return {
    draw(ctx, lm, offset, w, h, dt, opts = {}) {
      const { mirror = false, minVis = 0.5, person = 0 } = opts;
      if (!lm) return;
      const pc = personColors(person);
      drawSkeleton(ctx, lm, offset, w, h, {
        mirror, minVis, lineWidth: Math.max(2, w / 320), radius: Math.max(3, w / 240),
        leftColor: pc.left, rightColor: pc.right, midColor: pc.mid,
      });
      drawExtras(ctx, w, h, opts, 'full');
    },
    reset() {},
  };
}

function stickmanStyle() {
  return {
    draw(ctx, lm, offset, w, h, dt, opts = {}) {
      const { mirror = false, minVis = 0.5, person = 0 } = opts;
      if (!lm) return;
      const p = project(lm, offset, w, h, mirror);
      const head = neckAndHead(p, minVis);
      const thick = Math.max(6, w / 45);
      const body = personColors(person).body;
      const bones = BODY_BONES.filter(([a, b]) => a < 29 && b < 29).concat([[27, 31], [28, 32]]);
      ctx.save();
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      for (const [color, width] of [['#111', thick + 6], [body, thick]]) {
        drawBones(ctx, p, minVis, bones, () => color, width);
        if (head) {
          ctx.strokeStyle = color;
          ctx.lineWidth = width;
          ctx.beginPath();
          ctx.moveTo(head.nx, head.ny);
          ctx.lineTo(head.hx, head.hy + head.r);
          ctx.stroke();
          ctx.beginPath();
          ctx.arc(head.hx, head.hy, head.r + (color === '#111' ? 3 : 0), 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
        }
      }
      if (head && p.v[2] >= minVis && p.v[5] >= minVis) {
        ctx.fillStyle = '#111';
        for (const e of [2, 5]) {
          ctx.beginPath();
          ctx.arc(p.x[e], p.y[e], Math.max(2, head.r * 0.12), 0, Math.PI * 2);
          ctx.fill();
        }
        // a smile whose curve grows with how high the hands are
        const lift = Math.max(0, Math.min(1, (p.y[0] - Math.min(p.y[15], p.y[16])) / (h * 0.3)));
        ctx.strokeStyle = '#111';
        ctx.lineWidth = Math.max(2, head.r * 0.1);
        ctx.beginPath();
        ctx.arc(head.hx, head.hy + head.r * 0.15, head.r * 0.5, Math.PI * (0.15 + 0.1 * lift), Math.PI * (0.85 - 0.1 * lift));
        ctx.stroke();
      }
      ctx.restore();
      if (opts.hands) {
        drawHandSkeleton(ctx, opts.hands, w, h, { mirror, minVis, lineWidth: thick * 0.45 + 4, radius: 1, leftColor: '#111', rightColor: '#111' });
        drawHandSkeleton(ctx, opts.hands, w, h, { mirror, minVis, lineWidth: thick * 0.45, radius: 1, leftColor: body, rightColor: body });
      }
    },
    reset() {},
  };
}

function neonStyle() {
  let t = 0;
  return {
    draw(ctx, lm, offset, w, h, dt, opts = {}) {
      const { mirror = false, minVis = 0.5, person = 0 } = opts;
      t += dt;
      if (!lm) return;
      const p = project(lm, offset, w, h, mirror);
      const hue = (t * 40 + personColors(person).hue) % 360;
      const colorFor = (a, b) => {
        const side = isLeft(a) && isLeft(b) ? 1 : !isLeft(a) && !isLeft(b) && a >= 11 && b >= 11 ? -1 : 0;
        return hsla(hue + side * 70, 100, 60);
      };
      const bones = BODY_BONES.concat([[11, 12], [23, 24]]);
      const head = neckAndHead(p, minVis);
      ctx.save();
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.globalCompositeOperation = 'lighter';
      for (const [blur, width, alpha] of [[28, w / 70, 0.5], [12, w / 140, 0.8], [0, Math.max(1.5, w / 400), 1]]) {
        ctx.shadowBlur = blur;
        ctx.globalAlpha = alpha;
        drawBones(ctx, p, minVis, bones, (a, b) => { const c = colorFor(a, b); ctx.shadowColor = c; return blur ? c : '#fff'; }, width);
        if (head) {
          const c = hsla(hue, 100, 60);
          ctx.shadowColor = c;
          ctx.strokeStyle = blur ? c : '#fff';
          ctx.lineWidth = width;
          ctx.beginPath();
          ctx.arc(head.hx, head.hy, head.r, 0, Math.PI * 2);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(head.nx, head.ny);
          ctx.lineTo(head.hx, head.hy + head.r);
          ctx.stroke();
        }
      }
      ctx.restore();
      drawExtras(ctx, w, h, opts, 'glow', hue);
    },
    reset() {},
  };
}

function trailsStyle() {
  const TRAIL_MS = 700;
  const JOINTS = [15, 16, 27, 28, 13, 14, 0];
  const trails = new Map(JOINTS.map(j => [j, []]));
  let clock = 0;
  return {
    draw(ctx, lm, offset, w, h, dt, opts = {}) {
      const { mirror = false, minVis = 0.5, person = 0 } = opts;
      clock += dt * 1000;
      const shift = personColors(person).hue;
      const p = lm ? project(lm, offset, w, h, mirror) : null;
      for (const j of JOINTS) {
        const tr = trails.get(j);
        if (p && p.v[j] >= minVis) tr.push({ x: p.x[j], y: p.y[j], t: clock });
        while (tr.length && clock - tr[0].t > TRAIL_MS) tr.shift();
      }
      ctx.save();
      if (p) {
        ctx.globalAlpha = 0.35;
        drawSkeleton(ctx, lm, offset, w, h, { mirror, minVis, lineWidth: Math.max(1.5, w / 500), radius: Math.max(2, w / 400), midColor: '#fff', leftColor: personColors(person).left, rightColor: personColors(person).right });
        ctx.globalAlpha = 1;
        drawExtras(ctx, w, h, opts, 'ghost');
      }
      ctx.globalCompositeOperation = 'lighter';
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      const maxW = Math.max(6, w / 60);
      for (const j of JOINTS) {
        const tr = trails.get(j);
        const hue = (j === 0 ? 55 : isLeft(j) ? 25 : 200) + shift;
        for (let i = 1; i < tr.length; i++) {
          const age = (clock - tr[i].t) / TRAIL_MS; // 0 new .. 1 old
          const speed = Math.hypot(tr[i].x - tr[i - 1].x, tr[i].y - tr[i - 1].y) / Math.max(1, tr[i].t - tr[i - 1].t);
          const glow = Math.min(1, speed * 1.5);
          ctx.strokeStyle = hsla(hue, 100, 55 + 20 * glow, (1 - age) * (0.35 + 0.65 * glow));
          ctx.lineWidth = maxW * (1 - age) * (0.4 + 0.6 * glow) + 1;
          ctx.beginPath();
          ctx.moveTo(tr[i - 1].x, tr[i - 1].y);
          ctx.lineTo(tr[i].x, tr[i].y);
          ctx.stroke();
        }
      }
      ctx.restore();
    },
    reset() { for (const tr of trails.values()) tr.length = 0; },
  };
}

function sparksStyle() {
  const MAX = 1600;
  const particles = [];
  const tracker = makeTracker();
  return {
    draw(ctx, lm, offset, w, h, dt, opts = {}) {
      const { mirror = false, minVis = 0.5, person = 0 } = opts;
      const g = h * 0.9; // gravity, px/s^2
      const pc = personColors(person);
      const p = lm ? project(lm, offset, w, h, mirror) : null;
      if (p) {
        tracker.update(p, dt);
        for (const j of EMITTERS) {
          if (p.v[j] < minVis) continue;
          const speed = Math.hypot(tracker.vx[j], tracker.vy[j]), acc = Math.hypot(tracker.ax[j], tracker.ay[j]);
          const n = Math.min(18, Math.floor((speed / (h * 0.9) + acc / (h * 8)) * 10 * Math.min(1, dt * 30)) + (Math.random() < dt * 4 ? 1 : 0));
          const hue = (j === 0 ? 50 : isLeft(j) ? rand(10, 45) : rand(190, 230)) + pc.hue;
          for (let k = 0; k < n && particles.length < MAX; k++) {
            const a = rand(0, Math.PI * 2), s = rand(0, h * 0.15);
            particles.push({
              x: p.x[j] + rand(-4, 4), y: p.y[j] + rand(-4, 4),
              // sparks carry the joint's momentum (and a little of its acceleration) off with them
              vx: (tracker.vx[j] + tracker.ax[j] * 0.04) * 0.8 + Math.cos(a) * s, vy: (tracker.vy[j] + tracker.ay[j] * 0.04) * 0.8 + Math.sin(a) * s - h * 0.05,
              life: rand(0.45, 1.1), age: 0, size: rand(1.5, Math.max(3, w / 320)), hue,
            });
          }
        }
      } else tracker.reset();
      ctx.save();
      if (p) {
        ctx.globalAlpha = 0.3;
        drawSkeleton(ctx, lm, offset, w, h, { mirror, minVis, lineWidth: Math.max(1.5, w / 500), radius: Math.max(2, w / 400), midColor: '#fff', leftColor: pc.left, rightColor: pc.right });
        ctx.globalAlpha = 1;
        drawExtras(ctx, w, h, opts, 'ghost');
      }
      ctx.globalCompositeOperation = 'lighter';
      for (let i = particles.length - 1; i >= 0; i--) {
        const q = particles[i];
        q.age += dt;
        if (q.age >= q.life) { particles[i] = particles[particles.length - 1]; particles.pop(); continue; }
        q.vy += g * dt;
        q.vx *= Math.pow(0.6, dt);
        q.x += q.vx * dt;
        q.y += q.vy * dt;
        const k = 1 - q.age / q.life;
        ctx.fillStyle = hsla(q.hue, 100, 55 + 35 * k, 0.9 * k);
        ctx.beginPath();
        ctx.arc(q.x, q.y, q.size * (0.4 + 0.6 * k), 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    },
    reset() { particles.length = 0; tracker.reset(); },
  };
}

function constellationStyle() {
  let t = 0;
  let stars = null;
  return {
    draw(ctx, lm, offset, w, h, dt, opts = {}) {
      const { mirror = false, minVis = 0.5, person = 0 } = opts;
      t += dt;
      const shift = personColors(person).hue;
      if (!stars || stars.w !== w || stars.h !== h) {
        stars = { w, h, list: Array.from({ length: 90 }, () => ({ x: Math.random() * w, y: Math.random() * h, s: rand(0.6, 1.8), ph: rand(0, 6.3), sp: rand(2, 8) })) };
      }
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      // only the first person's effect draws the shared starfield, so it is not stacked N times
      for (const s of person === 0 ? stars.list : []) {
        s.x = (s.x + s.sp * dt + w) % w;
        ctx.fillStyle = hsla(210, 60, 90, 0.25 + 0.35 * (0.5 + 0.5 * Math.sin(t * 1.7 + s.ph)));
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.s, 0, Math.PI * 2);
        ctx.fill();
      }
      if (lm) {
        const p = project(lm, offset, w, h, mirror);
        const reach = h * 0.32;
        ctx.lineCap = 'round';
        for (let i = 0; i < MAJOR.length; i++) {
          for (let j = i + 1; j < MAJOR.length; j++) {
            const a = MAJOR[i], b = MAJOR[j];
            if (p.v[a] < minVis || p.v[b] < minVis) continue;
            const d = Math.hypot(p.x[a] - p.x[b], p.y[a] - p.y[b]);
            if (d > reach) continue;
            ctx.strokeStyle = hsla(200 + shift, 80, 85, 0.75 * (1 - d / reach) + 0.05);
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(p.x[a], p.y[a]);
            ctx.lineTo(p.x[b], p.y[b]);
            ctx.stroke();
          }
        }
        for (const [a, b] of BODY_BONES) {
          if (p.v[a] < minVis || p.v[b] < minVis) continue;
          ctx.strokeStyle = hsla(200 + shift, 90, 80, 0.9);
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(p.x[a], p.y[a]);
          ctx.lineTo(p.x[b], p.y[b]);
          ctx.stroke();
        }
        MAJOR.forEach((i, k) => {
          if (p.v[i] < minVis) return;
          const tw = 0.7 + 0.3 * Math.sin(t * 3 + k * 1.3);
          const r = Math.max(3, w / 220) * tw;
          const grad = ctx.createRadialGradient(p.x[i], p.y[i], 0, p.x[i], p.y[i], r * 4);
          grad.addColorStop(0, hsla(50 + shift, 100, 95, 0.9));
          grad.addColorStop(0.3, hsla(45 + shift, 100, 75, 0.5));
          grad.addColorStop(1, hsla(45 + shift, 100, 70, 0));
          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.arc(p.x[i], p.y[i], r * 4, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = hsla(50 + shift, 100, 95, 0.9 * tw);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(p.x[i] - r * 3, p.y[i]); ctx.lineTo(p.x[i] + r * 3, p.y[i]);
          ctx.moveTo(p.x[i], p.y[i] - r * 3); ctx.lineTo(p.x[i], p.y[i] + r * 3);
          ctx.stroke();
        });
        drawExtras(ctx, w, h, opts, 'ghost');
      }
      ctx.restore();
    },
    reset() { stars = null; },
  };
}

/**
 * Numbered badge above a person's head, used when several people are tracked.
 * `mirrorText` un-flips the digits when the canvas itself is CSS-mirrored.
 */
export function drawPersonBadge(ctx, lm, offset, w, h, person, { mirror = false, mirrorText = false, minVis = 0.3 } = {}) {
  const p = project(lm, offset, w, h, mirror);
  let idx = 0;
  if (p.v[0] < minVis) {
    idx = p.v[11] >= minVis ? 11 : p.v[12] >= minVis ? 12 : -1;
    if (idx < 0) return;
  }
  const r = Math.max(10, w / 90);
  const x = p.x[idx], y = p.y[idx] - (idx === 0 ? r * 3.2 : r * 2.2);
  const pc = personColors(person);
  ctx.save();
  ctx.translate(x, y);
  if (mirrorText) ctx.scale(-1, 1);
  ctx.fillStyle = pc.left;
  ctx.strokeStyle = 'rgba(0,0,0,0.6)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(0, 0, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = '#111';
  ctx.font = `bold ${Math.round(r * 1.25)}px system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(String(person + 1), 0, 1);
  ctx.restore();
}

// ---- sandman: a body of sand that sheds grains, which fall, pile up on the floor and fade ----

/** Live-adjustable sandman settings (shared by every sandman effect, read on every frame). */
export const SAND_DEFAULTS = { hue: 38, gravity: 1, decay: 2.5, momentum: 1 };
export const sandSettings = { ...SAND_DEFAULTS };
export function setSandSettings(patch) {
  for (const k of Object.keys(SAND_DEFAULTS)) if (Number.isFinite(patch?.[k])) sandSettings[k] = patch[k];
}

function sandmanStyle() {
  const MAX = 6000;
  const grains = []; // { x, y, vx, vy, age, life, size, l }
  const tracker = makeTracker();
  const SOURCES = BODY_BONES.filter(([a, b]) => a <= 28 && b <= 28);
  return {
    draw(ctx, lm, offset, w, h, dt, opts = {}) {
      const { mirror = false, minVis = 0.5, person = 0 } = opts;
      const { hue: baseHue, gravity, decay, momentum } = sandSettings;
      const hue = baseHue + (person ? personColors(person).hue : 0);
      const g = h * 0.8 * gravity;              // px/s^2
      const grain = Math.max(1.5, w / 480);     // grain size, px
      const p = lm ? project(lm, offset, w, h, mirror) : null;
      ctx.save();
      if (p) {
        tracker.update(p, dt);
        const { vx, vy, ax, ay } = tracker;
        // Grains leave with the momentum of the body part that shed them (its velocity, nudged
        // along its acceleration), so a sweeping arm flings sand onwards and a sudden stop throws
        // it forwards. `momentum` scales how much of that motion they keep.
        const emit = (x, y, jvx, jvy, jax, jay, n) => {
          for (let k = 0; k < n && grains.length < MAX; k++) {
            const keep = momentum * rand(0.75, 1.05);
            grains.push({
              x: x + rand(-grain, grain), y: y + rand(-grain, grain),
              vx: (jvx + jax * 0.05) * keep + rand(-1, 1) * h * 0.03, vy: (jvy + jay * 0.05) * keep + rand(-1, 1) * h * 0.02,
              age: 0, life: decay * rand(0.6, 1.2), size: grain * rand(0.7, 1.3), l: rand(52, 78),
            });
          }
        };
        // the body itself: a dusting of grains along every bone, the torso and the head
        const body = [];
        const scatter = Math.max(2, w / 260);
        for (const [a, b] of SOURCES) {
          if (p.v[a] < minVis || p.v[b] < minVis) continue;
          const len = Math.hypot(p.x[b] - p.x[a], p.y[b] - p.y[a]);
          const n = Math.ceil(len / (grain * 2.2));
          for (let k = 0; k < n; k++) {
            const t = Math.random();
            body.push([p.x[a] + (p.x[b] - p.x[a]) * t + rand(-scatter, scatter), p.y[a] + (p.y[b] - p.y[a]) * t + rand(-scatter, scatter)]);
          }
          // shed grains: a trickle when still, a stream when the limb moves fast, a burst when it
          // speeds up or stops sharply
          const sp = (Math.hypot(vx[a], vy[a]) + Math.hypot(vx[b], vy[b])) / 2;
          const acc = (Math.hypot(ax[a], ay[a]) + Math.hypot(ax[b], ay[b])) / 2;
          const rate = (len / h) * 260 * (1 + 4 * Math.min(2, sp / (h * 0.8)) + 3 * Math.min(2, acc / (h * 6)));
          let n2 = Math.floor(rate * dt + Math.random());
          while (n2-- > 0) {
            const t = Math.random(), lerp = (arr) => arr[a] + (arr[b] - arr[a]) * t;
            emit(p.x[a] + (p.x[b] - p.x[a]) * t, p.y[a] + (p.y[b] - p.y[a]) * t, lerp(vx), lerp(vy), lerp(ax), lerp(ay), 1);
          }
        }
        if ([11, 12, 23, 24].every(i => p.v[i] >= minVis)) {
          const area = Math.abs((p.x[12] - p.x[11]) * (p.y[23] - p.y[11]));
          const n = Math.min(500, Math.ceil(area / (grain * grain * 9)));
          for (let k = 0; k < n; k++) {
            const u = Math.random(), v = Math.random();
            const top = [p.x[11] + (p.x[12] - p.x[11]) * u, p.y[11] + (p.y[12] - p.y[11]) * u];
            const bot = [p.x[23] + (p.x[24] - p.x[23]) * u, p.y[23] + (p.y[24] - p.y[23]) * u];
            body.push([top[0] + (bot[0] - top[0]) * v, top[1] + (bot[1] - top[1]) * v]);
          }
          const hx = (vx[23] + vx[24]) / 2, hy = (vy[23] + vy[24]) / 2;
          if (Math.random() < dt * 40) emit(body[body.length - 1][0], body[body.length - 1][1], hx, hy, (ax[23] + ax[24]) / 2, (ay[23] + ay[24]) / 2, 2);
        }
        const head = neckAndHead(p, minVis);
        if (head) {
          const n = Math.min(300, Math.ceil(head.r * head.r * 3 / (grain * grain * 4)));
          for (let k = 0; k < n; k++) {
            const a = rand(0, Math.PI * 2), r = head.r * Math.sqrt(Math.random());
            body.push([head.hx + Math.cos(a) * r, head.hy + Math.sin(a) * r]);
          }
          body.push(...Array.from({ length: 12 }, () => { const t = Math.random(); return [head.nx + (head.hx - head.nx) * t, head.ny + (head.hy - head.ny) * t]; }));
          if (Math.random() < dt * 30) emit(head.hx + rand(-head.r, head.r), head.hy + rand(-head.r, head.r) * 0.5, vx[0], vy[0], ax[0], ay[0], 2);
        }
        ctx.fillStyle = hsla(hue, 70, 72, 0.9);
        for (const [x, y] of body) ctx.fillRect(x - grain / 2, y - grain / 2, grain, grain);
        drawExtras(ctx, w, h, opts, 'ghost');
      } else tracker.reset();
      // falling grains: gravity, a little air drag, a soft landing on the floor (the frame's bottom)
      const floor = h - 1;
      for (let i = grains.length - 1; i >= 0; i--) {
        const q = grains[i];
        q.age += dt;
        if (q.age >= q.life) { grains[i] = grains[grains.length - 1]; grains.pop(); continue; }
        q.vy += g * dt;
        q.vx *= Math.pow(0.55, dt); // light air drag: momentum carries the grains a while
        q.vy *= Math.pow(0.8, dt);
        q.x += q.vx * dt;
        q.y += q.vy * dt;
        if (q.y > floor - q.size) {
          q.y = floor - q.size;
          q.vy = q.vy > h * 0.2 ? -q.vy * 0.15 : 0;
          q.vx *= 0.5;
        } else if (q.y < -h * 0.2 && g < 0) {
          q.age = q.life; // floated away
        }
        const fade = Math.min(1, 3 * (1 - q.age / q.life));
        ctx.fillStyle = hsla(hue, 65, q.l, 0.85 * fade);
        ctx.fillRect(q.x - q.size / 2, q.y - q.size / 2, q.size, q.size);
      }
      ctx.restore();
    },
    reset() { grains.length = 0; tracker.reset(); },
  };
}

// ---- shared helpers for the paper styles ----

/** Lighten (k > 0, towards white) or darken (k < 0, towards black) a #rrggbb colour. */
function shadeHex(hex, k) {
  const c = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16));
  const out = c.map(v => Math.round(k >= 0 ? v + (255 - v) * k : v * (1 + k)));
  return `rgb(${out.join(',')})`;
}
/** Small deterministic PRNG, so hand-drawn wobble is stable between frames. */
function seeded(seed) {
  let a = (seed * 2654435761) >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** Torso length in px, the size unit for the paper styles. */
function torsoPx(p) {
  return Math.hypot((p.x[11] + p.x[12] - p.x[23] - p.x[24]) / 2, (p.y[11] + p.y[12] - p.y[23] - p.y[24]) / 2);
}

// ---- origami: the body folded from paper, each limb a creased kite of two shaded facets ----

function origamiStyle() {
  const LIGHT = -Math.PI * 0.75; // light from the upper left
  return {
    draw(ctx, lm, offset, w, h, dt, opts = {}) {
      const { mirror = false, minVis = 0.5, person = 0 } = opts;
      if (!lm) return;
      const p = project(lm, offset, w, h, mirror);
      if (![11, 12, 23, 24].every(i => p.v[i] >= minVis)) return;
      const pc = personColors(person);
      const paper = { left: pc.left, right: pc.right, mid: '#efe6d2' };
      const T = torsoPx(p);
      const polys = []; // { pts, fill }
      const lit = (ang, side) => 0.5 + 0.5 * Math.cos(ang + side * Math.PI / 2 - LIGHT); // 0..1 facing the light
      const facet = (pts, base, light) => polys.push({ pts, fill: shadeHex(base, -0.35 + 0.55 * light) });
      const ok = (...ids) => ids.every(i => p.v[i] >= minVis);
      // a limb segment: a kite folded along the bone, widest a third of the way down
      const kite = (a, b, width, base, at = 0.33) => {
        if (!ok(a, b)) return;
        const dx = p.x[b] - p.x[a], dy = p.y[b] - p.y[a], len = Math.hypot(dx, dy) || 1;
        const nx = -dy / len, ny = dx / len, ang = Math.atan2(dy, dx);
        const mx = p.x[a] + dx * at, my = p.y[a] + dy * at;
        const A = [p.x[a] - dx / len * width * 0.4, p.y[a] - dy / len * width * 0.4], B = [p.x[b], p.y[b]];
        facet([A, [mx + nx * width, my + ny * width], B], base, lit(ang, 1));
        facet([A, B, [mx - nx * width, my - ny * width]], base, lit(ang, -1));
      };
      // a pointed tip (hand / foot) continuing a segment's direction
      const tip = (a, b, length, width, base) => {
        if (!ok(a, b)) return;
        const dx = p.x[b] - p.x[a], dy = p.y[b] - p.y[a], len = Math.hypot(dx, dy) || 1;
        const ux = dx / len, uy = dy / len, ang = Math.atan2(dy, dx);
        const end = [p.x[b] + ux * length, p.y[b] + uy * length];
        facet([[p.x[b] - uy * width, p.y[b] + ux * width], end, [p.x[b], p.y[b]]], base, lit(ang, 1));
        facet([[p.x[b], p.y[b]], end, [p.x[b] + uy * width, p.y[b] - ux * width]], base, lit(ang, -1));
      };
      // legs behind the body
      kite(23, 25, T * 0.2, paper.left); kite(25, 27, T * 0.16, paper.left); tip(25, 27, T * 0.22, T * 0.08, paper.left);
      kite(24, 26, T * 0.2, paper.right); kite(26, 28, T * 0.16, paper.right); tip(26, 28, T * 0.22, T * 0.08, paper.right);
      // torso: four facets folded towards a centre point
      const S = [[p.x[11], p.y[11]], [p.x[12], p.y[12]]], H = [[p.x[24], p.y[24]], [p.x[23], p.y[23]]];
      const c = [(S[0][0] + S[1][0] + H[0][0] + H[1][0]) / 4, (S[0][1] + S[1][1] + H[0][1] + H[1][1]) / 4 + T * 0.05];
      const ring = [S[0], S[1], H[0], H[1]];
      const sides = [-Math.PI / 2, 0, Math.PI / 2, Math.PI]; // top, right, bottom, left facets
      for (let i = 0; i < 4; i++) facet([ring[i], ring[(i + 1) % 4], c], paper.mid, 0.5 + 0.5 * Math.cos(sides[i] - LIGHT));
      // arms in front
      kite(11, 13, T * 0.17, paper.left); kite(13, 15, T * 0.14, paper.left); tip(13, 15, T * 0.2, T * 0.09, paper.left);
      kite(12, 14, T * 0.17, paper.right); kite(14, 16, T * 0.14, paper.right); tip(14, 16, T * 0.2, T * 0.09, paper.right);
      // head: a folded hexagon with a pointed chin
      const head = neckAndHead(p, minVis);
      if (head) {
        const r = head.r * 1.1, tilt = Math.atan2(head.hy - head.ny, head.hx - head.nx) + Math.PI / 2;
        const pts = Array.from({ length: 6 }, (_, i) => {
          const a = tilt + (i / 6) * Math.PI * 2 - Math.PI / 2;
          const rr = i === 3 ? r * 1.25 : r; // chin
          return [head.hx + Math.cos(a) * rr, head.hy + Math.sin(a) * rr];
        });
        facet([[head.nx, head.ny], [head.hx - r * 0.3, head.hy + r * 0.6], [head.hx + r * 0.3, head.hy + r * 0.6]], paper.mid, 0.3);
        for (let i = 0; i < 6; i++) {
          const a = tilt + ((i + 0.5) / 6) * Math.PI * 2 - Math.PI / 2;
          facet([pts[i], pts[(i + 1) % 6], [head.hx, head.hy]], paper.mid, 0.5 + 0.5 * Math.cos(a - LIGHT));
        }
      }
      const path = pts => { ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]); ctx.closePath(); };
      ctx.save();
      ctx.lineJoin = 'round';
      // soft drop shadow so the paper seems to stand off the background
      ctx.fillStyle = 'rgba(0,0,0,0.28)';
      const off = Math.max(2, T * 0.05);
      for (const q of polys) { path(q.pts.map(([x, y]) => [x + off * 0.6, y + off])); ctx.fill(); }
      // facets with thin crease lines
      ctx.lineWidth = Math.max(0.75, w / 1400);
      ctx.strokeStyle = 'rgba(60,45,30,0.35)';
      for (const q of polys) { path(q.pts); ctx.fillStyle = q.fill; ctx.fill(); ctx.stroke(); }
      ctx.restore();
      drawExtras(ctx, w, h, opts, 'full');
    },
    reset() {},
  };
}

// ---- pencil: a loose graphite sketch with doubled contours, hatching and a gentle line boil ----

const paperCache = new Map();
/** An off-white paper sheet with a little grain, cached per size. */
function paperSheet(w, h) {
  const key = `${w}x${h}`;
  let c = paperCache.get(key);
  if (c) return c;
  c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d');
  g.fillStyle = '#f4f0e6';
  g.fillRect(0, 0, w, h);
  const rnd = seeded(7);
  for (let i = 0; i < (w * h) / 60; i++) {
    g.fillStyle = `rgba(${rnd() < 0.5 ? '120,105,80' : '255,255,255'},${(rnd() * 0.12).toFixed(3)})`;
    g.fillRect(rnd() * w, rnd() * h, 1 + rnd() * 1.5, 1 + rnd() * 1.5);
  }
  const vig = g.createRadialGradient(w / 2, h / 2, Math.min(w, h) * 0.3, w / 2, h / 2, Math.max(w, h) * 0.75);
  vig.addColorStop(0, 'rgba(0,0,0,0)'); vig.addColorStop(1, 'rgba(90,70,40,0.12)');
  g.fillStyle = vig; g.fillRect(0, 0, w, h);
  paperCache.clear();
  paperCache.set(key, c);
  return c;
}

function pencilStyle() {
  const BOIL = 0.14; // s between redraws of the wobble, like hand-drawn animation
  let clock = 0;
  const LIMBS = [[11, 13, 0.11], [13, 15, 0.09], [12, 14, 0.11], [14, 16, 0.09], [23, 25, 0.14], [25, 27, 0.11], [24, 26, 0.14], [26, 28, 0.11]];
  return {
    draw(ctx, lm, offset, w, h, dt, opts = {}) {
      const { mirror = false, minVis = 0.5, person = 0, videoVisible = true } = opts;
      clock += dt;
      // without a camera picture behind, draw on paper (once, by the first person's effect)
      if (!videoVisible && person === 0) ctx.drawImage(paperSheet(w, h), 0, 0);
      if (!lm) return;
      const p = project(lm, offset, w, h, mirror);
      if (![11, 12, 23, 24].every(i => p.v[i] >= minVis)) return;
      const T = torsoPx(p);
      const frame = Math.floor(clock / BOIL);
      const rnd = seeded(frame * 97 + person * 13 + 1);
      const jit = a => (rnd() - 0.5) * 2 * a;
      const graphite = videoVisible ? 'rgba(25,25,28,' : 'rgba(45,42,40,';
      const lw = Math.max(1, T / 70);
      ctx.save();
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      // one pencil stroke: a slightly bowed line that overshoots its ends a little
      const stroke = (x1, y1, x2, y2, alpha = 0.75, width = lw) => {
        const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
        const ux = dx / len, uy = dy / len, wob = Math.min(T * 0.025, len * 0.04);
        const o1 = jit(T * 0.04), o2 = T * 0.03 + jit(T * 0.05);
        const sx = x1 - ux * o1 + jit(wob), sy = y1 - uy * o1 + jit(wob);
        const ex = x2 + ux * o2 + jit(wob), ey = y2 + uy * o2 + jit(wob);
        const bow = jit(len * 0.03);
        if (videoVisible) { // a pale under-stroke keeps graphite readable over dark clothes
          ctx.strokeStyle = 'rgba(255,255,255,0.25)'; ctx.lineWidth = width * 2.6;
          ctx.beginPath(); ctx.moveTo(sx, sy); ctx.quadraticCurveTo((sx + ex) / 2 - uy * bow, (sy + ey) / 2 + ux * bow, ex, ey); ctx.stroke();
        }
        ctx.strokeStyle = `${graphite}${(alpha * (0.75 + rnd() * 0.25)).toFixed(3)})`;
        ctx.lineWidth = width * (0.8 + rnd() * 0.4);
        ctx.beginPath(); ctx.moveTo(sx, sy); ctx.quadraticCurveTo((sx + ex) / 2 - uy * bow, (sy + ey) / 2 + ux * bow, ex, ey); ctx.stroke();
      };
      // a loose sketched ellipse, drawn twice
      const loop = (cx, cy, rx, ry, rot = 0) => {
        for (let k = 0; k < 2; k++) {
          ctx.strokeStyle = `${graphite}${(0.55 + rnd() * 0.25).toFixed(3)})`;
          ctx.lineWidth = lw * (0.8 + rnd() * 0.4);
          ctx.beginPath();
          const a0 = rnd() * Math.PI * 2, sweep = Math.PI * 2 * (1.04 + rnd() * 0.1);
          for (let i = 0; i <= 28; i++) {
            const a = a0 + sweep * i / 28, e = 1 + jit(0.04);
            const x = Math.cos(a) * rx * e, y = Math.sin(a) * ry * e;
            const px = cx + x * Math.cos(rot) - y * Math.sin(rot), py = cy + x * Math.sin(rot) + y * Math.cos(rot);
            if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
          }
          ctx.stroke();
        }
      };
      // torso: sketched outline with diagonal hatching
      const tor = [[p.x[11], p.y[11]], [p.x[12], p.y[12]], [p.x[24], p.y[24]], [p.x[23], p.y[23]]];
      ctx.save();
      ctx.beginPath(); tor.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath();
      ctx.clip();
      const xs = tor.map(q => q[0]), ys = tor.map(q => q[1]);
      const x0 = Math.min(...xs) - T, x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys) + T;
      const gap = Math.max(4, T / 9);
      for (let x = x0; x < x1 + T; x += gap) stroke(x, y0, x + (y1 - y0), y1, 0.28, lw * 0.7);
      ctx.restore();
      for (let i = 0; i < 4; i++) {
        const [a, b] = [tor[i], tor[(i + 1) % 4]];
        stroke(a[0], a[1], b[0], b[1], 0.8);
        if (rnd() < 0.6) stroke(a[0], a[1], b[0], b[1], 0.45);
      }
      // limbs: two contour lines either side of the bone, with sketched joints
      for (const [a, b, thick] of LIMBS) {
        if (p.v[a] < minVis || p.v[b] < minVis) continue;
        const dx = p.x[b] - p.x[a], dy = p.y[b] - p.y[a], len = Math.hypot(dx, dy) || 1;
        const nx = -dy / len, ny = dx / len, r0 = thick * T, r1 = thick * T * 0.8;
        stroke(p.x[a] + nx * r0, p.y[a] + ny * r0, p.x[b] + nx * r1, p.y[b] + ny * r1);
        stroke(p.x[a] - nx * r0, p.y[a] - ny * r0, p.x[b] - nx * r1, p.y[b] - ny * r1);
        if (rnd() < 0.5) stroke(p.x[a] + nx * r0 * 0.4, p.y[a] + ny * r0 * 0.4, p.x[b] + nx * r1 * 0.4, p.y[b] + ny * r1 * 0.4, 0.25, lw * 0.6); // shading
        if (b === 13 || b === 14 || b === 25 || b === 26) loop(p.x[b], p.y[b], r1 * 0.9, r1 * 0.9);
      }
      // hands and feet: small ovals / wedges
      for (const [a, b, s2] of [[13, 15, 0.1], [14, 16, 0.1]]) {
        if (p.v[a] < minVis || p.v[b] < minVis) continue;
        const ang = Math.atan2(p.y[b] - p.y[a], p.x[b] - p.x[a]);
        loop(p.x[b] + Math.cos(ang) * T * s2, p.y[b] + Math.sin(ang) * T * s2, T * 0.12, T * 0.08, ang);
      }
      for (const k of [27, 28]) if (p.v[k] >= minVis) { stroke(p.x[k], p.y[k], p.x[k] + (k === 27 ? -1 : 1) * T * 0.04, p.y[k] + T * 0.1, 0.8); stroke(p.x[k] - T * 0.08, p.y[k] + T * 0.1, p.x[k] + T * 0.12, p.y[k] + T * 0.1, 0.8); }
      // head and neck
      const head = neckAndHead(p, minVis);
      if (head) {
        stroke(head.nx - T * 0.04, head.ny, head.hx - T * 0.04, head.hy + head.r * 0.9, 0.7);
        stroke(head.nx + T * 0.04, head.ny, head.hx + T * 0.04, head.hy + head.r * 0.9, 0.7);
        loop(head.hx, head.hy, head.r * 0.95, head.r * 1.15, Math.atan2(head.hy - head.ny, head.hx - head.nx) + Math.PI / 2);
      }
      ctx.restore();
      drawExtras(ctx, w, h, opts, 'ghost');
    },
    reset() { clock = 0; },
  };
}

// ---- fuzzy blur: a soft, out-of-focus glow of the body that smears with motion ----

const blobSprites = new Map();
/** A soft round brush in one colour (cached): opaque centre fading smoothly to nothing. */
function blobSprite(color) {
  let c = blobSprites.get(color);
  if (c) return c;
  c = document.createElement('canvas');
  c.width = c.height = 64;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, color);
  grad.addColorStop(0.35, color.replace(/^#(..)(..)(..)$/, (_, r, gg, b) => `rgba(${parseInt(r, 16)},${parseInt(gg, 16)},${parseInt(b, 16)},0.6)`));
  grad.addColorStop(0.7, color.replace(/^#(..)(..)(..)$/, (_, r, gg, b) => `rgba(${parseInt(r, 16)},${parseInt(gg, 16)},${parseInt(b, 16)},0.15)`));
  grad.addColorStop(1, color.replace(/^#(..)(..)(..)$/, (_, r, gg, b) => `rgba(${parseInt(r, 16)},${parseInt(gg, 16)},${parseInt(b, 16)},0)`));
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  blobSprites.set(color, c);
  return c;
}

/** Average of two #rrggbb colours, lightened a little. */
function mixHex(a, b) {
  const ch = (h, i) => parseInt(h.slice(i, i + 2), 16);
  return `#${[1, 3, 5].map(i => Math.round(Math.min(255, (ch(a, i) + ch(b, i)) / 2 * 1.15)).toString(16).padStart(2, '0')).join('')}`;
}

function fuzzyStyle() {
  const PERSIST = 0.12; // s: how long the glow lingers, which smears fast movement into a blur
  let buf = null, bctx = null;
  const BONES = BODY_BONES.filter(([a, b]) => a <= 28 && b <= 28);
  return {
    draw(ctx, lm, offset, w, h, dt, opts = {}) {
      const { mirror = false, minVis = 0.5, person = 0 } = opts;
      if (!buf || buf.width !== w || buf.height !== h) {
        buf = document.createElement('canvas');
        buf.width = w; buf.height = h;
        bctx = buf.getContext('2d');
      }
      // fade what was drawn before, then add this frame's body (amounts scaled by dt so the glow
      // is equally bright at any frame rate)
      const k = 1 - Math.exp(-dt / PERSIST);
      bctx.globalCompositeOperation = 'destination-out';
      bctx.globalAlpha = 1;
      bctx.fillStyle = `rgba(0,0,0,${k.toFixed(3)})`;
      bctx.fillRect(0, 0, w, h);
      const p = lm ? project(lm, offset, w, h, mirror) : null;
      if (p && [11, 12, 23, 24].every(i => p.v[i] >= minVis)) {
        const pc = personColors(person);
        const T = Math.hypot((p.x[11] + p.x[12] - p.x[23] - p.x[24]) / 2, (p.y[11] + p.y[12] - p.y[23] - p.y[24]) / 2);
        bctx.globalCompositeOperation = 'lighter';
        bctx.globalAlpha = Math.min(1, 0.2 * k * 6);
        const blob = (x, y, r, color) => bctx.drawImage(blobSprite(color), x - r, y - r, 2 * r, 2 * r);
        for (const [a, b] of BONES) {
          if (p.v[a] < minVis || p.v[b] < minVis) continue;
          if ((a === 11 && b === 12) || (a === 23 && b === 24)) continue; // the torso blobs cover these
          const color = isLeft(a) && isLeft(b) ? pc.left : !isLeft(a) && !isLeft(b) ? pc.right : pc.mid;
          const r = T * (a >= 23 ? 0.32 : 0.26);
          const len = Math.hypot(p.x[b] - p.x[a], p.y[b] - p.y[a]);
          const n = Math.max(2, Math.ceil(len / (r * 0.5)));
          for (let i = 0; i <= n; i++) blob(p.x[a] + (p.x[b] - p.x[a]) * i / n, p.y[a] + (p.y[b] - p.y[a]) * i / n, r, color);
        }
        // torso and head in a blend of the two side colours, so they glow rather than burn white
        const core = mixHex(pc.left, pc.right);
        for (const [u, v] of [[0.3, 0.2], [0.7, 0.2], [0.5, 0.5], [0.3, 0.8], [0.7, 0.8]]) {
          const tx = p.x[11] + (p.x[12] - p.x[11]) * u, ty = p.y[11] + (p.y[12] - p.y[11]) * u;
          const bx = p.x[23] + (p.x[24] - p.x[23]) * u, by = p.y[23] + (p.y[24] - p.y[23]) * u;
          blob(tx + (bx - tx) * v, ty + (by - ty) * v, T * 0.38, core);
        }
        const head = neckAndHead(p, minVis);
        if (head) {
          blob(head.hx, head.hy, head.r * 1.6, core);
          blob((head.nx + head.hx) / 2, (head.ny + head.hy) / 2, head.r * 0.8, core);
        }
      }
      ctx.save();
      ctx.filter = `blur(${Math.max(2, w / 220).toFixed(1)}px)`; // extra softness where supported
      ctx.drawImage(buf, 0, 0);
      ctx.filter = 'none';
      ctx.restore();
      if (p) drawExtras(ctx, w, h, opts, 'ghost');
    },
    reset() { bctx?.clearRect(0, 0, buf.width, buf.height); },
  };
}

const FACTORIES = { skeleton: skeletonStyle, stickman: stickmanStyle, neon: neonStyle, trails: trailsStyle, sparks: sparksStyle, constellation: constellationStyle, sandman: sandmanStyle, fuzzy: fuzzyStyle, origami: origamiStyle, pencil: pencilStyle };

export function createEffect(id) {
  const factory = FACTORIES[id] ?? FACTORIES[DEFAULT_STYLE];
  return { id: FACTORIES[id] ? id : DEFAULT_STYLE, ...factory() };
}
