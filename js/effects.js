// Alternative ways of drawing the detected pose: plain skeleton, stick figure, neon glow,
// light trails, sparks (particles) and a constellation. Every style exposes
//   draw(ctx, lm, offset, width, height, dt, { mirror, minVis })
// and keeps its own state (particles, trails) so it animates smoothly between detections.

import { CONNECTIONS, NUM_LANDMARKS, STRIDE, drawSkeleton } from './pose.js';

export const STYLES = [
  { id: 'skeleton', label: 'Skeleton' },
  { id: 'stickman', label: 'Stick figure' },
  { id: 'neon', label: 'Neon glow' },
  { id: 'trails', label: 'Light trails' },
  { id: 'sparks', label: 'Sparks' },
  { id: 'constellation', label: 'Constellation' },
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

/** Tracks joint velocities (px/s) between frames with light smoothing. */
function makeTracker() {
  let prev = null;
  const vx = new Float32Array(NUM_LANDMARKS), vy = new Float32Array(NUM_LANDMARKS);
  return {
    vx, vy,
    update(p, dt) {
      if (prev && dt > 0 && dt < 0.5) {
        for (let i = 0; i < NUM_LANDMARKS; i++) {
          vx[i] = vx[i] * 0.4 + ((p.x[i] - prev.x[i]) / dt) * 0.6;
          vy[i] = vy[i] * 0.4 + ((p.y[i] - prev.y[i]) / dt) * 0.6;
        }
      } else { vx.fill(0); vy.fill(0); }
      prev = p;
    },
    reset() { prev = null; vx.fill(0); vy.fill(0); },
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

// ---- styles -------------------------------------------------------------------------

function skeletonStyle() {
  return {
    draw(ctx, lm, offset, w, h, dt, { mirror = false, minVis = 0.5 } = {}) {
      if (!lm) return;
      drawSkeleton(ctx, lm, offset, w, h, { mirror, minVis, lineWidth: Math.max(2, w / 320), radius: Math.max(3, w / 240) });
    },
    reset() {},
  };
}

function stickmanStyle() {
  return {
    draw(ctx, lm, offset, w, h, dt, { mirror = false, minVis = 0.5 } = {}) {
      if (!lm) return;
      const p = project(lm, offset, w, h, mirror);
      const head = neckAndHead(p, minVis);
      const thick = Math.max(6, w / 45);
      const bones = BODY_BONES.filter(([a, b]) => a < 29 && b < 29).concat([[27, 31], [28, 32]]);
      ctx.save();
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      for (const [color, width] of [['#111', thick + 6], ['#fff', thick]]) {
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
    },
    reset() {},
  };
}

function neonStyle() {
  let t = 0;
  return {
    draw(ctx, lm, offset, w, h, dt, { mirror = false, minVis = 0.5 } = {}) {
      t += dt;
      if (!lm) return;
      const p = project(lm, offset, w, h, mirror);
      const hue = (t * 40) % 360;
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
    draw(ctx, lm, offset, w, h, dt, { mirror = false, minVis = 0.5 } = {}) {
      clock += dt * 1000;
      const p = lm ? project(lm, offset, w, h, mirror) : null;
      for (const j of JOINTS) {
        const tr = trails.get(j);
        if (p && p.v[j] >= minVis) tr.push({ x: p.x[j], y: p.y[j], t: clock });
        while (tr.length && clock - tr[0].t > TRAIL_MS) tr.shift();
      }
      ctx.save();
      if (p) {
        ctx.globalAlpha = 0.35;
        drawSkeleton(ctx, lm, offset, w, h, { mirror, minVis, lineWidth: Math.max(1.5, w / 500), radius: Math.max(2, w / 400), midColor: '#fff' });
        ctx.globalAlpha = 1;
      }
      ctx.globalCompositeOperation = 'lighter';
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      const maxW = Math.max(6, w / 60);
      for (const j of JOINTS) {
        const tr = trails.get(j);
        const hue = j === 0 ? 55 : isLeft(j) ? 25 : 200;
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
    draw(ctx, lm, offset, w, h, dt, { mirror = false, minVis = 0.5 } = {}) {
      const g = h * 0.9; // gravity, px/s^2
      const p = lm ? project(lm, offset, w, h, mirror) : null;
      if (p) {
        tracker.update(p, dt);
        for (const j of EMITTERS) {
          if (p.v[j] < minVis) continue;
          const speed = Math.hypot(tracker.vx[j], tracker.vy[j]);
          const n = Math.min(14, Math.floor(speed / (h * 0.9) * 10 * Math.min(1, dt * 30)) + (Math.random() < dt * 4 ? 1 : 0));
          const hue = j === 0 ? 50 : isLeft(j) ? rand(10, 45) : rand(190, 230);
          for (let k = 0; k < n && particles.length < MAX; k++) {
            const a = rand(0, Math.PI * 2), s = rand(0, h * 0.15);
            particles.push({
              x: p.x[j] + rand(-4, 4), y: p.y[j] + rand(-4, 4),
              vx: tracker.vx[j] * 0.35 + Math.cos(a) * s, vy: tracker.vy[j] * 0.35 + Math.sin(a) * s - h * 0.05,
              life: rand(0.45, 1.1), age: 0, size: rand(1.5, Math.max(3, w / 320)), hue,
            });
          }
        }
      } else tracker.reset();
      ctx.save();
      if (p) {
        ctx.globalAlpha = 0.3;
        drawSkeleton(ctx, lm, offset, w, h, { mirror, minVis, lineWidth: Math.max(1.5, w / 500), radius: Math.max(2, w / 400), midColor: '#fff' });
        ctx.globalAlpha = 1;
      }
      ctx.globalCompositeOperation = 'lighter';
      for (let i = particles.length - 1; i >= 0; i--) {
        const q = particles[i];
        q.age += dt;
        if (q.age >= q.life) { particles[i] = particles[particles.length - 1]; particles.pop(); continue; }
        q.vy += g * dt;
        q.vx *= 0.985;
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
    draw(ctx, lm, offset, w, h, dt, { mirror = false, minVis = 0.5 } = {}) {
      t += dt;
      if (!stars || stars.w !== w || stars.h !== h) {
        stars = { w, h, list: Array.from({ length: 90 }, () => ({ x: Math.random() * w, y: Math.random() * h, s: rand(0.6, 1.8), ph: rand(0, 6.3), sp: rand(2, 8) })) };
      }
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      for (const s of stars.list) {
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
            ctx.strokeStyle = hsla(200, 80, 85, 0.75 * (1 - d / reach) + 0.05);
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(p.x[a], p.y[a]);
            ctx.lineTo(p.x[b], p.y[b]);
            ctx.stroke();
          }
        }
        for (const [a, b] of BODY_BONES) {
          if (p.v[a] < minVis || p.v[b] < minVis) continue;
          ctx.strokeStyle = hsla(200, 90, 80, 0.9);
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
          grad.addColorStop(0, hsla(50, 100, 95, 0.9));
          grad.addColorStop(0.3, hsla(45, 100, 75, 0.5));
          grad.addColorStop(1, hsla(45, 100, 70, 0));
          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.arc(p.x[i], p.y[i], r * 4, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = hsla(50, 100, 95, 0.9 * tw);
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(p.x[i] - r * 3, p.y[i]); ctx.lineTo(p.x[i] + r * 3, p.y[i]);
          ctx.moveTo(p.x[i], p.y[i] - r * 3); ctx.lineTo(p.x[i], p.y[i] + r * 3);
          ctx.stroke();
        });
      }
      ctx.restore();
    },
    reset() { stars = null; },
  };
}

const FACTORIES = { skeleton: skeletonStyle, stickman: stickmanStyle, neon: neonStyle, trails: trailsStyle, sparks: sparksStyle, constellation: constellationStyle };

export function createEffect(id) {
  const factory = FACTORIES[id] ?? FACTORIES[DEFAULT_STYLE];
  return { id: FACTORIES[id] ? id : DEFAULT_STYLE, ...factory() };
}
