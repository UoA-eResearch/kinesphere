// Movement analysis of a recorded session.
// All distances are made scale-invariant by dividing by the dancer's torso length
// (median distance between mid-shoulders and mid-hips over the recording), so a
// "speed of 1" means one torso length per second regardless of camera distance.

import { NUM_LANDMARKS, STRIDE, WORLD_JOINTS, WORLD_SIZE } from './pose.js';
import { clamp, mean, median, percentile, rng } from './util.js';

/** The 13 "major" joints used for activity metrics, in this order (same as the stored world joints). */
export const MAJOR = WORLD_JOINTS;
/** Display names: 1 head, 2 shoulders, 2 elbows, 2 hands, 2 hips, 2 knees, 2 feet. */
export const MAJOR_NAMES = ['Head', 'L shoulder', 'R shoulder', 'L elbow', 'R elbow', 'L hand', 'R hand',
  'L hip', 'R hip', 'L knee', 'R knee', 'L foot', 'R foot'];
/** Connections between MAJOR indices (for stick figures). Nose->neck is drawn separately. */
export const MAJOR_CONNECTIONS = [[1, 2], [1, 3], [3, 5], [2, 4], [4, 6], [1, 7], [2, 8], [7, 8], [7, 9], [9, 11], [8, 10], [10, 12]];
/** MAJOR slots on the dancer's left and right (head excluded). */
export const LEFT_SLOTS = [1, 3, 5, 7, 9, 11];
export const RIGHT_SLOTS = [2, 4, 6, 8, 10, 12];

const VIS_MIN = 0.3;
const MAX_GAP_S = 0.5;        // gaps longer than this are not used for velocities
const ACTIVE_SPEED = 0.5;     // torso lengths / s: a body part faster than this counts as "active"
const SCALE_WINDOW_MS = 2000; // half-window for the per-frame body scale
const LEG_GESTURE = 0.5;      // a foot raised more than this many shin lengths is a gesture, not support

export const KINESPHERE_CLASSES = [
  { key: 'small', label: 'Small' },
  { key: 'medium', label: 'Medium' },
  { key: 'large', label: 'Large' },
];
export const SPACE_LEVELS = [
  { key: 'high', label: 'High' },
  { key: 'mid', label: 'Medium' },
  { key: 'low', label: 'Low' },
];
/** Limbs used for the kinesphere: [proximal, mid, distal] landmark indices. */
const LIMBS = [
  { key: 'armL', name: 'L arm', kind: 'arm', pts: [11, 13, 15] },
  { key: 'armR', name: 'R arm', kind: 'arm', pts: [12, 14, 16] },
  { key: 'legL', name: 'L leg', kind: 'leg', pts: [23, 25, 27] },
  { key: 'legR', name: 'R leg', kind: 'leg', pts: [24, 26, 28] },
];
const FLOOR_EXTRA = [29, 30, 31, 32]; // heels and toes, when the engine provides them

/** Percentile p over a +-halfMs time window around every frame (NaN where the value is NaN). */
function rollingPercentile(times, values, halfMs, p) {
  const n = values.length, out = new Float32Array(n).fill(NaN);
  let lo = 0, hi = 0;
  const buf = [];
  for (let i = 0; i < n; i++) {
    while (hi < n && times[hi] <= times[i] + halfMs) hi++;
    while (lo < n && times[lo] < times[i] - halfMs) lo++;
    if (!Number.isFinite(values[i])) continue;
    buf.length = 0;
    for (let j = lo; j < hi; j++) if (Number.isFinite(values[j])) buf.push(values[j]);
    buf.sort((a, b) => a - b);
    out[i] = buf[Math.min(buf.length - 1, Math.round(p * (buf.length - 1)))];
  }
  return out;
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

  // ---- Body part activity: share of movement and time spent moving, per marker ----
  const jointSpeed = MAJOR.map(k => smoothTime(times, speed[k], 150));
  const totalMean = MAJOR.reduce((acc, k) => acc + (Number.isFinite(meanSpeed[k]) ? meanSpeed[k] : 0), 0) || 1;
  const maxJoint = Math.max(...MAJOR.map(k => meanSpeed[k]).filter(Number.isFinite), 1e-6);
  const joints = MAJOR.map((k, j) => {
    let active = 0, seen = 0;
    for (let i = 0; i < n; i++) {
      const v = jointSpeed[j][i];
      if (!Number.isFinite(v)) continue;
      seen++;
      if (v > ACTIVE_SPEED) active++;
    }
    const m = Number.isFinite(meanSpeed[k]) ? meanSpeed[k] : 0;
    return { index: k, name: MAJOR_NAMES[j], meanSpeed: meanSpeed[k], share: m / totalMean, relative: m / maxJoint, activeTime: seen ? active / seen : NaN };
  });
  const sideSum = slots => slots.reduce((acc, j) => acc + (Number.isFinite(meanSpeed[MAJOR[j]]) ? meanSpeed[MAJOR[j]] : 0), 0);
  const leftSum = sideSum(LEFT_SLOTS), rightSum = sideSum(RIGHT_SLOTS);
  const lateral = { left: leftSum / ((leftSum + rightSum) || 1), right: rightSum / ((leftSum + rightSum) || 1) };

  // ---- Per-frame body scale: rolling 90th percentile of the torso length, so the scale follows
  // the dancer moving towards or away from the camera but ignores brief foreshortening.
  const frameScale = rollingPercentile(times, torso, SCALE_WINDOW_MS, 0.9);
  for (let i = 0; i < n; i++) {
    if (Number.isFinite(frameScale[i])) frameScale[i] = clamp(frameScale[i], S * 0.5, S * 2);
    else if (valid[i]) frameScale[i] = S;
  }
  const pt = (k, i) => [X[k][i], Y[k][i]];
  const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
  // Kinesphere centre: midway between the mid-hip and the mid-shoulder (centre of the torso).
  const centreAt = i => [(cx[i] + (X[11][i] + X[12][i]) / 2) / 2, (cy[i] + (Y[11][i] + Y[12][i]) / 2) / 2];

  // ---- Limb lengths from the skeleton (torso units). 2D projections only ever shorten a
  // segment, so the 90th percentile over the session approximates its true length.
  const segP90 = (a, b) => {
    const vals = [];
    for (let i = 0; i < n; i++) if (valid[i] && visible(a, i) && visible(b, i)) vals.push(dist(pt(a, i), pt(b, i)) / frameScale[i]);
    return percentile(vals, 0.9);
  };
  const fromCentreP90 = k => {
    const vals = [];
    for (let i = 0; i < n; i++) if (valid[i] && visible(k, i)) vals.push(dist(centreAt(i), pt(k, i)) / frameScale[i]);
    return percentile(vals, 0.9);
  };
  const pair = (a, b) => {
    const v = [a, b].filter(Number.isFinite);
    return v.length ? mean(v) : NaN;
  };
  const limbs = {
    upperArm: pair(segP90(11, 13), segP90(12, 14)),
    forearm: pair(segP90(13, 15), segP90(14, 16)),
    thigh: pair(segP90(23, 25), segP90(24, 26)),
    shin: pair(segP90(25, 27), segP90(26, 28)),
    centreShoulder: pair(fromCentreP90(11), fromCentreP90(12)),
    centreHip: pair(fromCentreP90(23), fromCentreP90(24)),
  };
  // ---- 3D (MediaPipe world landmarks): metric positions with depth, so limb lengths are true
  // lengths (median, no foreshortening) and reaches towards the camera are measured properly.
  const slotOf = k => MAJOR.indexOf(k);
  let world3d = null;
  if (session.world && session.world.length >= n * WORLD_SIZE) {
    const Wd = session.world;
    const present = new Uint8Array(n);
    let presentCount = 0;
    for (let i = 0; i < n; i++) {
      const o = i * WORLD_SIZE;
      let any = false;
      for (let d = 0; d < WORLD_SIZE; d++) if (Wd[o + d] !== 0) { any = true; break; }
      present[i] = valid[i] && any ? 1 : 0;
      presentCount += present[i];
    }
    if (presentCount >= Math.max(10, 0.2 * validCount)) {
      const WP = MAJOR.map((k, j) => [0, 1, 2].map(c => {
        const arr = new Float32Array(n);
        const mask = new Uint8Array(n);
        for (let i = 0; i < n; i++) { arr[i] = Wd[i * WORLD_SIZE + j * 3 + c]; mask[i] = present[i] && visible(k, i) ? 1 : 0; }
        return smoothFrames(arr, 2, mask);
      }));
      const p3 = (k, i) => { const j = slotOf(k); return [WP[j][0][i], WP[j][1][i], WP[j][2][i]]; };
      const d3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
      const mid3 = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2];
      const centre3 = i => mid3(mid3(p3(11, i), p3(12, i)), mid3(p3(23, i), p3(24, i)));
      const torsos = [];
      for (let i = 0; i < n; i++) if (present[i]) torsos.push(d3(mid3(p3(11, i), p3(12, i)), mid3(p3(23, i), p3(24, i))));
      const T3 = median(torsos);
      if (T3 > 0.05) {
        const seg3 = (a, b) => {
          const v = [];
          for (let i = 0; i < n; i++) if (present[i] && visible(a, i) && visible(b, i)) v.push(d3(p3(a, i), p3(b, i)));
          return median(v);
        };
        const fromCentre3 = k => {
          const v = [];
          for (let i = 0; i < n; i++) if (present[i] && visible(k, i)) v.push(d3(centre3(i), p3(k, i)));
          return median(v);
        };
        const metres = {
          upperArm: pair(seg3(11, 13), seg3(12, 14)),
          forearm: pair(seg3(13, 15), seg3(14, 16)),
          thigh: pair(seg3(23, 25), seg3(24, 26)),
          shin: pair(seg3(25, 27), seg3(26, 28)),
          centreShoulder: pair(fromCentre3(11), fromCentre3(12)),
          centreHip: pair(fromCentre3(23), fromCentre3(24)),
        };
        for (const key of Object.keys(limbs)) limbs[key] = metres[key] / T3;
        world3d = { present, p3, d3, centre3, T3, metres: { ...metres, torso: T3 } };
      }
    }
  }

  // Fall back to typical proportions (in torso lengths) when a segment was never visible, and keep
  // estimates within plausible human proportions so a run of bad detections cannot inflate them.
  const typical = { upperArm: 0.6, forearm: 0.55, thigh: 0.9, shin: 0.85, centreShoulder: 0.64, centreHip: 0.54 };
  const plausible = { upperArm: [0.4, 0.85], forearm: [0.35, 0.8], thigh: [0.6, 1.2], shin: [0.55, 1.15], centreShoulder: [0.5, 1.0], centreHip: [0.5, 0.9] };
  for (const key of Object.keys(limbs)) {
    limbs[key] = limbs[key] > 0 ? clamp(limbs[key], ...plausible[key]) : typical[key];
  }
  const arm = { small: limbs.centreShoulder, medium: limbs.centreShoulder + limbs.upperArm, large: limbs.centreShoulder + limbs.upperArm + limbs.forearm };
  const leg = { small: limbs.centreHip, medium: limbs.centreHip + limbs.thigh, large: limbs.centreHip + limbs.thigh + limbs.shin };
  // One set of concentric spheres for every limb: each level is the further of the arm and leg
  // distance (small = proximal joints, medium = elbows/knees, large = hands/feet outstretched).
  const shared = { small: Math.max(arm.small, leg.small), medium: Math.max(arm.medium, leg.medium), large: Math.max(arm.large, leg.large) };
  const radii = { arm, leg, shared };

  // ---- Floor and standing heights. The floor in each frame is the lowest visible joint (the body
  // always touches the floor somewhere, except mid-jump); heights are in torso units above it.
  const floorY = new Float32Array(n).fill(NaN);
  for (let i = 0; i < n; i++) {
    if (!valid[i]) continue;
    let low = -Infinity;
    for (const k of MAJOR) if (k !== 0 && visible(k, i) && Y[k][i] > low) low = Y[k][i];
    for (const k of FLOOR_EXTRA) {
      const o = (i * NUM_LANDMARKS + k) * STRIDE;
      if (lm[o + 3] >= VIS_MIN && lm[o + 1] > low) low = lm[o + 1];
    }
    if (Number.isFinite(low)) floorY[i] = low;
  }
  const heightOf = (y, i) => (floorY[i] - y) / frameScale[i];
  const shoulderHeights = [], kneeHeights = [];
  for (let i = 0; i < n; i++) {
    if (!valid[i] || !Number.isFinite(floorY[i])) continue;
    shoulderHeights.push(heightOf((Y[11][i] + Y[12][i]) / 2, i));
    // the lower knee belongs to the supporting leg, so a raised knee does not skew the estimate
    const ks = [25, 26].filter(k => visible(k, i)).map(k => heightOf(Y[k][i], i));
    if (ks.length) kneeHeights.push(Math.min(...ks));
  }
  // The dancer stands for most of a session, so the upper percentiles are standing heights.
  const standing = {
    shoulder: shoulderHeights.length >= 10 ? percentile(shoulderHeights, 0.8) : limbs.shin + limbs.thigh + 1,
    knee: kneeHeights.length >= 10 ? percentile(kneeHeights, 0.8) : limbs.shin,
  };

  // ---- Per-frame kinesphere class and space level.
  const kClass = new Int8Array(n).fill(-1), kLimb = new Int8Array(n).fill(-1), level = new Int8Array(n).fill(-1);
  const gesturing = (leg, i) => {
    const ankle = leg.pts[2];
    if (!visible(ankle, i) || !Number.isFinite(floorY[i])) return false;
    return heightOf(Y[ankle][i], i) > LEG_GESTURE * limbs.shin;
  };
  for (let i = 0; i < n; i++) {
    if (!valid[i]) continue;
    const w = centreAt(i), sc = frameScale[i];
    let best = -1, bestLimb = -1;
    LIMBS.forEach((limb, li) => {
      if (limb.kind === 'leg' && !gesturing(limb, i)) return; // supporting legs are not reach
      const R = radii.shared;
      const [, mid, end] = limb.pts;
      // distance from the torso centre in torso lengths: 3D from world landmarks when available
      const reach = k => (world3d?.present[i]
        ? world3d.d3(world3d.centre3(i), world3d.p3(k, i)) / world3d.T3
        : dist(w, pt(k, i)) / sc);
      let c = -1;
      if (visible(end, i)) {
        const d = reach(end);
        c = d <= R.small ? 0 : d <= R.medium ? 1 : 2;
      } else if (visible(mid, i)) {
        c = reach(mid) <= R.small ? 0 : 1;
      }
      if (c > best) { best = c; bestLimb = li; }
    });
    kClass[i] = best;
    kLimb[i] = bestLimb;
    if (Number.isFinite(floorY[i])) {
      let high = false, low = false;
      for (const limb of LIMBS) {
        if (limb.kind === 'leg' && !gesturing(limb, i)) continue;
        const [, mid, end] = limb.pts;
        for (const k of [mid, end]) if (visible(k, i) && heightOf(Y[k][i], i) > standing.shoulder) high = true;
        if (limb.kind === 'arm' && visible(end, i) && heightOf(Y[end][i], i) < standing.knee) low = true;
      }
      if (heightOf(cy[i], i) < standing.knee) low = true; // hips below standing knee height: crouching or on the floor
      level[i] = high ? 0 : low ? 2 : 1;
    }
  }
  const shareOf = (arr, k) => {
    let c = 0, t = 0;
    for (let i = 0; i < n; i++) { if (arr[i] < 0) continue; t++; if (arr[i] === k) c++; }
    return t ? c / t : 0;
  };
  // Typical pose for each kinesphere class: the real frame closest to the class's average pose
  // (relative to the mid-hip, torso units, selfie view), so the figure is a pose that happened.
  const poseVec = i => MAJOR.flatMap(k => [flip * (X[k][i] - cx[i]) / frameScale[i], (Y[k][i] - cy[i]) / frameScale[i]]);
  const classFrames = [[], [], []];
  for (let i = 0; i < n; i++) if (kClass[i] >= 0 && MAJOR.every(k => visible(k, i))) classFrames[kClass[i]].push(i);
  const typicalPose = frames => {
    if (!frames.length) return null;
    const stepF = Math.max(1, Math.floor(frames.length / 1500));
    const sample = frames.filter((_, j) => j % stepF === 0).map(poseVec);
    const avg = sample[0].map((_, d) => mean(sample.map(v => v[d])));
    let best = sample[0], bestD = Infinity;
    for (const v of sample) {
      let dd = 0;
      for (let d = 0; d < v.length; d++) dd += (v[d] - avg[d]) ** 2;
      if (dd < bestD) { bestD = dd; best = v; }
    }
    return MAJOR.map((_, j) => [best[j * 2], best[j * 2 + 1]]);
  };
  const kinesphere = {
    shares: Object.fromEntries(KINESPHERE_CLASSES.map((c, k) => [c.key, shareOf(kClass, k)])),
    limbs, radii,
    mode: world3d ? '3d' : '2d',
    metres: world3d ? world3d.metres : null,
    coverage3d: world3d ? world3d.present.reduce((a, b) => a + b, 0) / Math.max(1, validCount) : 0,
    poses: Object.fromEntries(KINESPHERE_CLASSES.map((c, k) => [c.key, typicalPose(classFrames[k])])),
    limbNames: LIMBS.map(l => l.name),
  };
  const levels = {
    shares: Object.fromEntries(SPACE_LEVELS.map((l, k) => [l.key, shareOf(level, k)])),
    standing,
  };

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
    kinesphere,
    body: { joints, lateral },
    levels,
    speed: { series: speedSeries, mean: mean(frameSpeed), peak: Number.isFinite(peak) ? peak : NaN, peakT },
    space, shapes,
    // Per-frame values for the replay's metrics overlay.
    frames: { kClass, kLimb, level, scale: frameScale, floor: floorY, jointSpeed },
  };
}
