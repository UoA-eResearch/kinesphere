// Movement analysis of a recorded session.
// All distances are made scale-invariant by dividing by the dancer's torso length
// (median distance between mid-shoulders and mid-hips over the recording), so a
// "speed of 1" means one torso length per second regardless of camera distance.

import { NUM_LANDMARKS, STRIDE } from './pose.js';
import { clamp, mean, median, percentile, rng } from './util.js';

/** The 13 "major" joints used for activity metrics, in this order. */
export const MAJOR = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
export const MAJOR_NAMES = ['Nose', 'L shoulder', 'R shoulder', 'L elbow', 'R elbow', 'L wrist', 'R wrist',
  'L hip', 'R hip', 'L knee', 'R knee', 'L ankle', 'R ankle'];
/** Connections between MAJOR indices (for stick figures). Nose->neck is drawn separately. */
export const MAJOR_CONNECTIONS = [[1, 2], [1, 3], [3, 5], [2, 4], [4, 6], [1, 7], [2, 8], [7, 8], [7, 9], [9, 11], [8, 10], [10, 12]];

const VIS_MIN = 0.3;
const MAX_GAP_S = 0.5; // gaps longer than this are not used for velocities

const ZONE_VERTICAL = { upper: [13, 14, 15, 16], mid: [11, 12, 23, 24], lower: [25, 26, 27, 28] };
const ZONE_LATERAL = { left: [11, 13, 15, 23, 25, 27], right: [12, 14, 16, 24, 26, 28] };
const GRID = [
  [[13, 15], [14, 16]], // upper: elbows + wrists
  [[11, 23], [12, 24]], // mid: shoulders + hips
  [[25, 27], [26, 28]], // lower: knees + ankles
];
export const GRID_ROWS = ['Upper (arms)', 'Mid (torso)', 'Lower (legs)'];
export const GRID_COLS = ['Left side', 'Right side'];

export const KINESPHERE_CLASSES = [
  { key: 'small', label: 'Small', max: 0.55 },
  { key: 'medium', label: 'Medium', max: 0.8 },
  { key: 'large', label: 'Large', max: Infinity },
];

const ANGLES = [
  { key: 'shoulderL', label: 'Shoulder', side: 'L', pts: [23, 11, 13] },
  { key: 'shoulderR', label: 'Shoulder', side: 'R', pts: [24, 12, 14] },
  { key: 'elbowL', label: 'Elbow', side: 'L', pts: [11, 13, 15] },
  { key: 'elbowR', label: 'Elbow', side: 'R', pts: [12, 14, 16] },
  { key: 'hipL', label: 'Hip', side: 'L', pts: [11, 23, 25] },
  { key: 'hipR', label: 'Hip', side: 'R', pts: [12, 24, 26] },
  { key: 'kneeL', label: 'Knee', side: 'L', pts: [23, 25, 27] },
  { key: 'kneeR', label: 'Knee', side: 'R', pts: [24, 26, 28] },
];

function angleDeg(ax, ay, bx, by, cx, cy) {
  const v1x = ax - bx, v1y = ay - by, v2x = cx - bx, v2y = cy - by;
  const d = Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y);
  if (!d) return NaN;
  return Math.acos(clamp((v1x * v2x + v1y * v2y) / d, -1, 1)) * 180 / Math.PI;
}

/**
 * Centered moving average (radius r frames) of a Float32Array. Frames where `mask` is false
 * (landmark not visible) are left out of the average, so a person entering the frame is not
 * smeared towards the zeros recorded while they were absent.
 */
function smoothFrames(arr, r, mask) {
  const n = arr.length, out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0, c = 0;
    for (let j = Math.max(0, i - r); j <= Math.min(n - 1, i + r); j++) {
      if (mask && !mask[j]) continue;
      s += arr[j]; c++;
    }
    out[i] = c ? s / c : arr[i];
  }
  return out;
}

/** Moving average over a time window (+-halfWindowMs), ignoring NaN. */
function smoothTime(times, values, halfWindowMs) {
  const n = values.length, out = new Float32Array(n);
  let lo = 0, hi = 0;
  let sum = 0, cnt = 0;
  const add = i => { if (Number.isFinite(values[i])) { sum += values[i]; cnt++; } };
  const sub = i => { if (Number.isFinite(values[i])) { sum -= values[i]; cnt--; } };
  for (let i = 0; i < n; i++) {
    while (hi < n && times[hi] <= times[i] + halfWindowMs) { add(hi); hi++; }
    while (lo < i && times[lo] < times[i] - halfWindowMs) { sub(lo); lo++; }
    out[i] = cnt ? sum / cnt : NaN;
  }
  return out;
}

/** Average values into ~nBins equal time bins => [{ t, v }]. Empty bins are skipped. */
function bucketize(times, values, nBins, durationMs) {
  const n = values.length;
  if (!n) return [];
  const bins = Math.max(1, Math.min(nBins, n));
  const w = Math.max(1, durationMs) / bins;
  const sum = new Float64Array(bins), cnt = new Uint32Array(bins);
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(values[i])) continue;
    const b = Math.min(bins - 1, Math.floor(times[i] / w));
    sum[b] += values[i]; cnt[b]++;
  }
  const out = [];
  for (let b = 0; b < bins; b++) if (cnt[b]) out.push({ t: (b + 0.5) * w, v: sum[b] / cnt[b] });
  return out;
}

function kmeans(vectors, k, iters, random) {
  const m = vectors.length, d = vectors[0].length;
  const dist2 = (a, b) => { let s = 0; for (let i = 0; i < d; i++) { const x = a[i] - b[i]; s += x * x; } return s; };
  // k-means++ initialisation
  const centroids = [vectors[Math.floor(random() * m)].slice()];
  const best = new Float64Array(m).fill(Infinity);
  while (centroids.length < k) {
    let total = 0;
    for (let i = 0; i < m; i++) { best[i] = Math.min(best[i], dist2(vectors[i], centroids[centroids.length - 1])); total += best[i]; }
    let r = random() * total, pick = m - 1;
    for (let i = 0; i < m; i++) { r -= best[i]; if (r <= 0) { pick = i; break; } }
    centroids.push(vectors[pick].slice());
  }
  const assign = new Int32Array(m);
  for (let it = 0; it < iters; it++) {
    let changed = false;
    for (let i = 0; i < m; i++) {
      let bi = 0, bd = Infinity;
      for (let c = 0; c < k; c++) { const dd = dist2(vectors[i], centroids[c]); if (dd < bd) { bd = dd; bi = c; } }
      if (assign[i] !== bi) { assign[i] = bi; changed = true; }
    }
    const sums = centroids.map(() => new Float64Array(d)), counts = new Uint32Array(k);
    for (let i = 0; i < m; i++) { const c = assign[i]; counts[c]++; const v = vectors[i]; for (let j = 0; j < d; j++) sums[c][j] += v[j]; }
    for (let c = 0; c < k; c++) if (counts[c]) for (let j = 0; j < d; j++) centroids[c][j] = sums[c][j] / counts[c];
    if (!changed) break;
  }
  const counts = new Uint32Array(k);
  for (let i = 0; i < m; i++) counts[assign[i]]++;
  return { centroids, counts };
}

/**
 * Analyse a session. Returns { ok: false, reason } when there is not enough usable pose data.
 */
export function analyze(session) {
  const { times, lm, width, height, frameCount: n, durationMs } = session;
  const aspect = (width || 16) / (height || 9);
  const flip = session.mirrored ? -1 : 1; // present everything as the dancer saw it (selfie view)
  if (n < 10) return { ok: false, reason: 'Fewer than 10 frames were recorded.' };

  // Per-landmark position arrays in "square" units (x scaled by aspect so distances are isotropic).
  const X = {}, Y = {}, V = {};
  for (const k of MAJOR) {
    const xs = new Float32Array(n), ys = new Float32Array(n), vs = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const o = (i * NUM_LANDMARKS + k) * STRIDE;
      xs[i] = lm[o] * aspect; ys[i] = lm[o + 1]; vs[i] = lm[o + 3];
    }
    const mask = new Uint8Array(n);
    for (let i = 0; i < n; i++) mask[i] = vs[i] >= VIS_MIN ? 1 : 0;
    X[k] = smoothFrames(xs, 2, mask); Y[k] = smoothFrames(ys, 2, mask); V[k] = vs;
  }
  const visible = (k, i) => V[k][i] >= VIS_MIN;
  const valid = new Uint8Array(n);
  let validCount = 0;
  for (let i = 0; i < n; i++) {
    valid[i] = visible(11, i) && visible(12, i) && visible(23, i) && visible(24, i) ? 1 : 0;
    validCount += valid[i];
  }
  if (validCount < 10) return { ok: false, reason: 'The torso (shoulders and hips) was rarely visible.' };

  // Body scale: median torso length (mid-shoulder to mid-hip).
  const torso = new Float32Array(n).fill(NaN);
  const cx = new Float32Array(n).fill(NaN), cy = new Float32Array(n).fill(NaN); // body centre (mid-hip)
  for (let i = 0; i < n; i++) {
    if (!valid[i]) continue;
    const msx = (X[11][i] + X[12][i]) / 2, msy = (Y[11][i] + Y[12][i]) / 2;
    const mhx = (X[23][i] + X[24][i]) / 2, mhy = (Y[23][i] + Y[24][i]) / 2;
    torso[i] = Math.hypot(msx - mhx, msy - mhy);
    cx[i] = mhx; cy[i] = mhy;
  }
  const S = Math.max(1e-3, median(torso));

  // Per-landmark speeds (torso lengths / s).
  const speed = {};
  const meanSpeed = {};
  for (const k of MAJOR) {
    const sp = new Float32Array(n).fill(NaN);
    for (let i = 1; i < n; i++) {
      const dt = (times[i] - times[i - 1]) / 1000;
      if (dt <= 0 || dt > MAX_GAP_S || !valid[i] || !valid[i - 1] || !visible(k, i) || !visible(k, i - 1)) continue;
      sp[i] = Math.hypot(X[k][i] - X[k][i - 1], Y[k][i] - Y[k][i - 1]) / dt / S;
    }
    speed[k] = sp;
    meanSpeed[k] = mean(sp);
  }
  const zoneScore = ids => mean(ids.map(k => meanSpeed[k]));
  const normalise = obj => {
    const total = Object.values(obj).reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0) || 1;
    return Object.fromEntries(Object.entries(obj).map(([k, v]) => [k, (Number.isFinite(v) ? v : 0) / total]));
  };
  const vertical = normalise(Object.fromEntries(Object.entries(ZONE_VERTICAL).map(([z, ids]) => [z, zoneScore(ids)])));
  const lateral = normalise(Object.fromEntries(Object.entries(ZONE_LATERAL).map(([z, ids]) => [z, zoneScore(ids)])));
  const gridRaw = GRID.map(row => row.map(ids => zoneScore(ids)));
  const gridTotal = gridRaw.flat().reduce((a, b) => a + (Number.isFinite(b) ? b : 0), 0) || 1;
  const grid = gridRaw.map(row => row.map(v => (Number.isFinite(v) ? v : 0) / gridTotal));
  const maxJoint = Math.max(...MAJOR.map(k => meanSpeed[k]).filter(Number.isFinite), 1e-6);
  const joints = MAJOR.map((k, j) => ({
    index: k, name: MAJOR_NAMES[j], meanSpeed: meanSpeed[k], relative: (meanSpeed[k] || 0) / maxJoint,
  }));

  // Overall speed over time.
  const frameSpeed = new Float32Array(n).fill(NaN);
  for (let i = 0; i < n; i++) {
    let s = 0, c = 0;
    for (const k of MAJOR) { const v = speed[k][i]; if (Number.isFinite(v)) { s += v; c++; } }
    if (c) frameSpeed[i] = s / c;
  }
  const smoothSpeed = smoothTime(times, frameSpeed, 500);
  let peak = -Infinity, peakT = 0;
  for (let i = 0; i < n; i++) if (smoothSpeed[i] > peak) { peak = smoothSpeed[i]; peakT = times[i]; }
  const speedSeries = bucketize(times, smoothSpeed, 400, durationMs);

  // Joint angles + kinesphere openness.
  const angleValues = Object.fromEntries(ANGLES.map(a => [a.key, new Float32Array(n).fill(NaN)]));
  const openness = new Float32Array(n).fill(NaN);
  for (let i = 0; i < n; i++) {
    if (!valid[i]) continue;
    for (const a of ANGLES) {
      const [p, q, r] = a.pts;
      if (!visible(p, i) || !visible(q, i) || !visible(r, i)) continue;
      angleValues[a.key][i] = angleDeg(X[p][i], Y[p][i], X[q][i], Y[q][i], X[r][i], Y[r][i]);
    }
    const arm = mean(['shoulderL', 'shoulderR', 'elbowL', 'elbowR'].map(k => angleValues[k][i] / 180));
    const leg = mean(['hipL', 'hipR', 'kneeL', 'kneeR'].map(k => angleValues[k][i] / 180));
    if (Number.isFinite(arm) && Number.isFinite(leg)) openness[i] = 0.6 * arm + 0.4 * leg;
    else if (Number.isFinite(arm)) openness[i] = arm;
    else if (Number.isFinite(leg)) openness[i] = leg;
  }
  const smoothOpen = smoothTime(times, openness, 250);
  const classCounts = { small: 0, medium: 0, large: 0 };
  let openCount = 0;
  for (let i = 0; i < n; i++) {
    const v = smoothOpen[i];
    if (!Number.isFinite(v)) continue;
    openCount++;
    classCounts[KINESPHERE_CLASSES.find(c => v < c.max).key]++;
  }
  const kinesphere = {
    shares: Object.fromEntries(Object.entries(classCounts).map(([k, c]) => [k, openCount ? c / openCount : 0])),
    meanOpenness: mean(smoothOpen),
    series: bucketize(times, smoothOpen, 400, durationMs),
  };
  const angles = ANGLES.map(a => {
    const vals = angleValues[a.key];
    return { ...a, mean: mean(vals), p5: percentile(vals, 0.05), p95: percentile(vals, 0.95) };
  });

  // Space: where the body centre was in the frame (selfie view, normalised 0..1).
  const px = new Float32Array(n).fill(NaN), py = new Float32Array(n).fill(NaN);
  for (let i = 0; i < n; i++) {
    if (!valid[i]) continue;
    const x = cx[i] / aspect;
    px[i] = session.mirrored ? 1 - x : x;
    py[i] = cy[i];
  }
  const pathX = bucketize(times, px, 600, durationMs), pathY = bucketize(times, py, 600, durationMs);
  const path = pathX.map((p, i) => ({ t: p.t, x: p.v, y: pathY[i]?.v ?? p.v }));
  const cols = 12, rows = Math.max(3, Math.round(cols / aspect));
  const cells = new Float32Array(cols * rows);
  let pathLen = 0;
  for (let i = 0; i < n; i++) {
    if (!Number.isFinite(px[i])) continue;
    const c = clamp(Math.floor(px[i] * cols), 0, cols - 1), r = clamp(Math.floor(py[i] * rows), 0, rows - 1);
    cells[r * cols + c] += 1 / validCount;
    if (i > 0 && Number.isFinite(px[i - 1]) && (times[i] - times[i - 1]) / 1000 <= MAX_GAP_S) {
      pathLen += Math.hypot(cx[i] - cx[i - 1], cy[i] - cy[i - 1]) / S;
    }
  }
  const space = {
    path,
    occupancy: { cols, rows, cells },
    xRange: [percentile(px, 0.02), percentile(px, 0.98)],
    yRange: [percentile(py, 0.02), percentile(py, 0.98)],
    pathLength: pathLen,
  };

  // Shapes: cluster body poses (relative to hips, in torso units, selfie view) with k-means.
  const vectors = [], vectorFrames = [];
  const step = Math.max(1, Math.floor(validCount / 2500));
  for (let i = 0, c = 0; i < n; i++) {
    if (!valid[i]) continue;
    if (c++ % step) continue;
    const v = new Float64Array(MAJOR.length * 2);
    MAJOR.forEach((k, j) => {
      v[j * 2] = flip * (X[k][i] - cx[i]) / S;
      v[j * 2 + 1] = (Y[k][i] - cy[i]) / S;
    });
    vectors.push(v); vectorFrames.push(i);
  }
  let shapes = [];
  if (vectors.length >= 30) {
    const k = Math.min(5, Math.floor(vectors.length / 15));
    const { centroids, counts } = kmeans(vectors, k, 40, rng(42));
    shapes = centroids.map((c, ci) => ({
      share: counts[ci] / vectors.length,
      pose: MAJOR.map((_, j) => [c[j * 2], c[j * 2 + 1]]),
    })).sort((a, b) => b.share - a.share).slice(0, 3);
  }

  const fps = n > 1 ? (n - 1) / ((times[n - 1] - times[0]) / 1000) : 0;
  return {
    ok: true,
    durationMs, frameCount: n, validFraction: validCount / n, fps, torsoLength: S,
    kinesphere, angles,
    zones: { vertical, lateral, grid, joints },
    speed: { series: speedSeries, mean: mean(frameSpeed), peak: Number.isFinite(peak) ? peak : NaN, peakT },
    space, shapes,
  };
}
