// A simple 2D human body fitted to pose keypoints, so styles can draw a body with a natural shape
// instead of straight lines between joints. Works with MoveNet's 17 keypoints (hands and feet are
// inferred from the forearm and shin) and uses MediaPipe's extra hand / foot points when present.
//
//   const rig = createRig();                       // one per drawn person: smooths the motion
//   const J = rig.update(p, dt, minVis);           // p: projected landmarks { x, y, v } in pixels
//   const body = buildBody(J, opts);               // smooth outlines of every body part
//
// Limbs are tubes along a Catmull-Rom curve through their joints (so elbows and knees bend in a
// curve) with an anatomical width profile: shoulder, deltoid, bicep, elbow, forearm, wrist, hand;
// hip, thigh, knee, calf, ankle, foot. The torso is a smooth outline from the neck over the
// shoulders, chest and waist to the hips; the head is an egg shape with a narrower jaw.

const CORE = [0, 2, 5, 7, 8, 11, 12, 23, 24];
const LIMB = [13, 14, 25, 26];
const ENDS = [15, 16, 17, 18, 19, 20, 21, 22, 27, 28, 29, 30, 31, 32];
const HOLD_S = 0.4; // keep a joint that drops out for this long, so the body doesn't flicker

/**
 * Smooths joint positions with critically damped springs: the trunk follows quickly, the elbows and
 * knees a touch slower and the hands, feet and head slower still, which gives a little natural
 * follow-through. Returns { x, y, v } arrays (pixels, visibility) like the input.
 */
export function createRig() {
  const n = 33;
  const x = new Float32Array(n), y = new Float32Array(n), vx = new Float32Array(n), vy = new Float32Array(n);
  const v = new Float32Array(n), lost = new Float32Array(n).fill(Infinity);
  const omega = new Float32Array(n).fill(2 * Math.PI * 6);
  for (const i of LIMB) omega[i] = 2 * Math.PI * 5;
  for (const i of ENDS) omega[i] = 2 * Math.PI * 4.2;
  omega[0] = 2 * Math.PI * 4.5;
  for (const i of CORE.filter(i => i !== 0)) omega[i] = 2 * Math.PI * 7;
  return {
    update(p, dt, minVis = 0.5) {
      const step = Math.min(Math.max(dt, 0), 0.05);
      for (let i = 0; i < n; i++) {
        const seen = p.v[i] >= minVis && Number.isFinite(p.x[i]);
        if (!seen) {
          lost[i] += step;
          if (lost[i] > HOLD_S) v[i] = 0;
          continue;
        }
        if (lost[i] > HOLD_S) { x[i] = p.x[i]; y[i] = p.y[i]; vx[i] = 0; vy[i] = 0; } // (re)appeared: snap
        else {
          const w = omega[i];
          vx[i] += (w * w * (p.x[i] - x[i]) - 2 * w * vx[i]) * step;
          vy[i] += (w * w * (p.y[i] - y[i]) - 2 * w * vy[i]) * step;
          x[i] += vx[i] * step; y[i] += vy[i] * step;
        }
        lost[i] = 0;
        v[i] = p.v[i];
      }
      return { x, y, v };
    },
    reset() { lost.fill(Infinity); v.fill(0); },
  };
}

// ---- geometry helpers ------------------------------------------------------------------------

const add = (a, b, k = 1) => [a[0] + b[0] * k, a[1] + b[1] * k];
const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
const len = (a, b) => Math.hypot(b[0] - a[0], b[1] - a[1]);
const unit = (a, b) => { const l = len(a, b) || 1; return [(b[0] - a[0]) / l, (b[1] - a[1]) / l]; };
const perp = u => [-u[1], u[0]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1];

/** Centripetal-ish Catmull-Rom point between p1 and p2 at t (uniform parameterisation). */
function catmull(p0, p1, p2, p3, t) {
  const t2 = t * t, t3 = t2 * t;
  const f = (a, b, c, d) => 0.5 * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t2 + (-a + 3 * b - 3 * c + d) * t3);
  return [f(p0[0], p1[0], p2[0], p3[0]), f(p0[1], p1[1], p2[1], p3[1])];
}

/** Piecewise-linear width profile [[t, w], ...] at t in 0..1. */
function profileAt(profile, t) {
  for (let i = 1; i < profile.length; i++) {
    if (t <= profile[i][0]) {
      const [t0, w0] = profile[i - 1], [t1, w1] = profile[i];
      return w0 + (w1 - w0) * ((t - t0) / ((t1 - t0) || 1));
    }
  }
  return profile[profile.length - 1][1];
}

/**
 * A tube along a smooth curve through `pts`, with one width profile per segment (in pixels).
 * Returns { centre, a, b, seg } - the centreline and the two sides (a = left of travel), each a
 * list of points, and for each sample the segment index plus position within it.
 */
export function tube(pts, profiles, samples = 10) {
  const centre = [], a = [], b = [], seg = [];
  const P = i => pts[Math.max(0, Math.min(pts.length - 1, i))];
  for (let k = 0; k < pts.length - 1; k++) {
    // extrapolate past the ends so the curve leaves the first / last joint in a natural direction
    const p0 = k === 0 ? add(P(0), unit(P(1), P(0)), len(P(0), P(1))) : P(k - 1);
    const p3 = k === pts.length - 2 ? add(P(k + 1), unit(P(k), P(k + 1)), len(P(k), P(k + 1))) : P(k + 2);
    for (let s = 0; s < samples; s++) {
      if (k > 0 && s === 0) continue; // shared joint
      const t = s / (samples - 1);
      const c = catmull(p0, P(k), P(k + 1), p3, t);
      const c2 = catmull(p0, P(k), P(k + 1), p3, Math.min(1, t + 0.02)), c1 = catmull(p0, P(k), P(k + 1), p3, Math.max(0, t - 0.02));
      const n = perp(unit(c1, c2));
      const w = profileAt(profiles[k], t);
      centre.push(c); a.push(add(c, n, w)); b.push(add(c, n, -w)); seg.push([k, t]);
    }
  }
  return { centre, a, b, seg };
}

/** A closed outline from a tube: side a, a rounded cap at the end, side b back, a rounded start. */
export function tubeOutline(tb, capSteps = 5) {
  const out = [...tb.a];
  const cap = (centre, from, to) => {
    const r = len(centre, from), a0 = Math.atan2(from[1] - centre[1], from[0] - centre[0]);
    let a1 = Math.atan2(to[1] - centre[1], to[0] - centre[0]);
    const dir = tb.centre.length > 1 ? unit(tb.centre[tb.centre.length - 2], tb.centre[tb.centre.length - 1]) : [1, 0];
    // go round the outside of the end
    let sweep = a1 - a0;
    const mid = a0 + sweep / 2, midPt = [Math.cos(mid), Math.sin(mid)];
    if (dot(midPt, dir) < 0) sweep = sweep > 0 ? sweep - 2 * Math.PI : sweep + 2 * Math.PI;
    a1 = a0 + sweep;
    for (let i = 1; i < capSteps; i++) { const a = a0 + (a1 - a0) * i / capSteps; out.push([centre[0] + Math.cos(a) * r, centre[1] + Math.sin(a) * r]); }
  };
  const last = tb.centre.length - 1;
  cap(tb.centre[last], tb.a[last], tb.b[last]);
  for (let i = last; i >= 0; i--) out.push(tb.b[i]);
  return out;
}

// ---- the body --------------------------------------------------------------------------------

const ARM = [
  [[0, 0.125], [0.15, 0.135], [0.45, 0.105], [0.85, 0.08], [1, 0.075]],  // shoulder -> elbow
  [[0, 0.075], [0.25, 0.088], [0.7, 0.062], [1, 0.05]],                   // elbow -> wrist
  [[0, 0.05], [0.35, 0.066], [0.8, 0.05], [1, 0.022]],                    // wrist -> fingertips
];
const LEG = [
  [[0, 0.2], [0.2, 0.19], [0.6, 0.14], [1, 0.1]],                         // hip -> knee
  [[0, 0.1], [0.3, 0.118], [0.75, 0.072], [1, 0.058]],                    // knee -> ankle
  [[0, 0.06], [0.3, 0.068], [1, 0.032]],                                   // ankle -> toes
];
const NECK = [[[0, 0.1], [1, 0.085]]];

const scaleProfile = (prof, s) => prof.map(seg => seg.map(([t, w]) => [t, w * s]));

/**
 * Fit the body to smoothed joints J ({ x, y, v } in pixels). Options: `minVis`; `limbScale` and
 * `legScale` widen limbs (e.g. sleeves, trousers); `legFlare` widens the leg towards the hem.
 * Returns null without shoulders and hips, else
 *   { T, axis, side, shM, hpM, torso, head, neck, arms: [l, r], legs: [l, r], outOf, chest, waist }
 * where torso/head are closed outlines (point lists), arms / legs / neck are tubes (see tube()),
 * each limb also carrying { outline, outer } (outer: 'a' or 'b', the side away from the body).
 */
export function buildBody(J, { minVis = 0.5, limbScale = 1, legScale = 1, legFlare = 0, samples = 10 } = {}) {
  const ok = i => J.v[i] >= minVis;
  if (![11, 12, 23, 24].every(ok)) return null;
  const P = i => [J.x[i], J.y[i]];
  const shL = P(11), shR = P(12), hpL = P(23), hpR = P(24);
  const shM = mix(shL, shR, 0.5), hpM = mix(hpL, hpR, 0.5);
  const T = Math.max(8, len(shM, hpM));
  const axis = unit(shM, hpM), side = perp(axis);
  const outOf = pt => { const d = dot([pt[0] - shM[0], pt[1] - shM[1]], side); return d >= 0 ? side : [-side[0], -side[1]]; };

  // torso: neck root, trapezius, shoulder cap, armpit, chest, waist, hip crest, hip, crotch
  const half = (sh, hp) => {
    const o = outOf(sh);
    return [
      add(mix(shM, sh, 0.3), axis, -T * 0.07),
      add(mix(shM, sh, 0.72), axis, -T * 0.05),
      add(add(sh, o, T * 0.075), axis, T * 0.015), // rounded shoulder
      add(mix(sh, hp, 0.2), o, T * 0.035),
      add(mix(sh, hp, 0.42), o, T * 0.03),
      add(mix(sh, hp, 0.7), o, -T * 0.025),
      add(mix(sh, hp, 0.92), o, T * 0.05),
      add(add(hp, o, T * 0.1), axis, T * 0.1),
    ];
  };
  const left = half(shL, hpL), right = half(shR, hpR);
  const crotch = add(hpM, axis, T * 0.24);
  const torsoKnots = [...left, crotch, ...right.slice().reverse()];
  const torso = smoothClosed(torsoKnots, 6);

  // limbs
  const limb = (a, b, c, ends, profile, tipFn) => {
    if (!ok(a) || !ok(b)) return null;
    const A = P(a), B = P(b);
    const C = ok(c) ? P(c) : add(B, unit(A, B), len(A, B) * 0.9);
    const tip = tipFn(B, C, ends);
    const pts = [A, B, C, tip];
    const tb = tube(pts, profile, samples);
    const mid = Math.floor(tb.centre.length * 0.15);
    const outer = dot([tb.a[mid][0] - tb.centre[mid][0], tb.a[mid][1] - tb.centre[mid][1]], outOf(A)) > 0 ? 'a' : 'b';
    return { ...tb, outline: tubeOutline(tb), outer, joints: pts };
  };
  const handTip = (E, W, [idx, pinky, thumb]) => {
    if (ok(idx) && ok(pinky)) return add(mix(P(idx), P(pinky), 0.5), unit(W, mix(P(idx), P(pinky), 0.5)), T * 0.09);
    const d = unit(E, W);
    return add(W, d, T * 0.21);
  };
  const footTip = (K, A, [heel, toe]) => {
    if (ok(toe)) return add(P(toe), unit(A, P(toe)), T * 0.02);
    const o = outOf(A);
    return add(add(A, o, T * 0.2), axis, T * 0.05); // feet point outwards along the floor
  };
  const armProf = scaleProfile(ARM, T * limbScale);
  const legProf = scaleProfile(LEG, T * legScale).map((seg, k) => (legFlare && k === 1 ? seg.map(([t, w]) => [t, w + legFlare * T * t * t]) : seg));
  const arms = [limb(11, 13, 15, [19, 17, 21], armProf, handTip), limb(12, 14, 16, [20, 18, 22], armProf, handTip)];
  const legs = [limb(23, 25, 27, [29, 31], legProf, footTip), limb(24, 26, 28, [30, 32], legProf, footTip)];

  // head: an egg (narrower jaw) facing along the neck, sized from the ears or the torso
  let head = null, neck = null;
  const earsOk = ok(7) && ok(8);
  if (ok(0) || earsOk) {
    const centre = earsOk ? add(mix(P(7), P(8), 0.5), unit(mix(P(7), P(8), 0.5), ok(0) ? P(0) : mix(P(7), P(8), 0.5)), T * 0.02) : P(0);
    const hw = earsOk ? Math.max(T * 0.17, len(P(7), P(8)) * 0.62) : T * 0.2;
    const base = add(shM, axis, -T * 0.05);
    const up = unit(centre, base).map(v => -v);
    const hh = hw * 1.3;
    const c = add(centre, up, -hh * 0.05);
    const pts = [];
    for (let i = 0; i < 28; i++) {
      const th = (i / 28) * Math.PI * 2;
      const yy = -Math.cos(th), jaw = yy > 0 ? 1 - 0.3 * yy * yy : 1; // narrower towards the chin
      const xx = Math.sin(th) * hw * jaw, yv = yy * hh;
      pts.push(add(add(c, perp(up), xx), up, -yv));
    }
    head = { outline: pts, centre: c, up, hw, hh };
    const chin = add(c, up, -hh * 0.8);
    neck = tube([base, mix(base, chin, 0.5), chin], scaleProfile([...NECK, ...NECK], T), 6);
    neck.outline = tubeOutline(neck);
  }
  const chest = [add(left[4], outOf(shL), -T * 0.01), mix(mix(left[4], right[4], 0.5), add(shM, axis, T * 0.42), 0.4), add(right[4], outOf(shR), -T * 0.01)];
  const waist = { centre: add(mix(shM, hpM, 0.86), axis, T * 0.02), rx: len(left[6], right[6]) * 0.5, ry: T * 0.07, rot: Math.atan2(side[1], side[0]) };
  return { T, axis, side, shM, hpM, shL, shR, hpL, hpR, torso, torsoKnots, crotch, head, neck, arms, legs, outOf, chest, waist };
}

/** A smooth closed curve through knots (Catmull-Rom), `steps` points per span. */
export function smoothClosed(knots, steps = 6) {
  const n = knots.length, out = [];
  for (let i = 0; i < n; i++) {
    const p0 = knots[(i - 1 + n) % n], p1 = knots[i], p2 = knots[(i + 1) % n], p3 = knots[(i + 2) % n];
    for (let s = 0; s < steps; s++) out.push(catmull(p0, p1, p2, p3, s / steps));
  }
  return out;
}

/** A smooth open curve through points (Catmull-Rom), `steps` points per span. */
export function smoothOpen(pts, steps = 6) {
  if (pts.length < 3) return pts.slice();
  const out = [], n = pts.length;
  for (let i = 0; i < n - 1; i++) {
    const p0 = pts[Math.max(0, i - 1)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[Math.min(n - 1, i + 2)];
    for (let s = 0; s < steps; s++) out.push(catmull(p0, p1, p2, p3, s / steps));
  }
  out.push(pts[n - 1]);
  return out;
}

/** Trace a point list as a path on ctx (optionally closed). */
export function tracePath(ctx, pts, close = false) {
  ctx.beginPath();
  ctx.moveTo(pts[0][0], pts[0][1]);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
  if (close) ctx.closePath();
}

/**
 * A ribbon of varying width along a smooth curve through `pts`: width(u) gives the half-width at
 * u = 0 (first point) .. 1 (last point), by length along the curve. Returns { centre, a, b }.
 */
export function ribbon(pts, width, steps = 4) {
  const c = smoothOpen(pts, steps);
  const cum = [0];
  for (let i = 1; i < c.length; i++) cum.push(cum[i - 1] + len(c[i - 1], c[i]));
  const total = cum[cum.length - 1] || 1;
  const a = [], b = [];
  for (let i = 0; i < c.length; i++) {
    const p = c[Math.max(0, i - 1)], q = c[Math.min(c.length - 1, i + 1)];
    const n = perp(unit(p, q)), w = width(cum[i] / total);
    a.push(add(c[i], n, w)); b.push(add(c[i], n, -w));
  }
  return { centre: c, a, b, length: total };
}
