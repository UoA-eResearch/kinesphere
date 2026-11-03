// AI dance partner: pose conversions to and from SMPL-24 joints, resampling, and talking to an
// Altlando server (a fake one here).
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import {
  bodyFromWorld, bodyFromLandmarks, smplFromBody, createUserConverter, createResampler, landmarksFromSmpl,
  worldFromSmpl, boneLengths, createPartner, smplSequence, SMPL_JOINTS, CODE_FRAMES,
} from '../../js/ai-partner.js';
import { phasesSession } from '../helpers/synth.mjs';
import { personView } from '../../js/session.js';
import { FRAME_SIZE, WORLD_SIZE } from '../../js/pose.js';

const session = phasesSession(1);
const frame = i => ({ lm: session.lm.subarray(i * FRAME_SIZE, (i + 1) * FRAME_SIZE), world: session.world.subarray(i * WORLD_SIZE, (i + 1) * WORLD_SIZE) });
const J3 = (J, name) => { const i = SMPL_JOINTS.indexOf(name); return [J[i * 3], J[i * 3 + 1], J[i * 3 + 2]]; };
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

describe('your pose → SMPL-24 joints', () => {
  const f = frame(150); // standing, arms down
  const J = smplFromBody(bodyFromWorld(f.world), [0.3, 0.95, -2.5]);
  test('24 joints, pelvis at the root', () => {
    assert.equal(J.length, 72);
    assert.deepEqual(Array.from(J.slice(0, 3), v => +v.toFixed(3)), [0.3, 0.95, -2.5]);
    assert.ok(J.every(Number.isFinite));
  });
  test('y is up: head above the neck above the pelvis above the knees above the feet', () => {
    const y = n => J3(J, n)[1];
    assert.ok(y('head') > y('neck') && y('neck') > y('spine3') && y('spine3') > y('pelvis') && y('pelvis') > y('l_knee') && y('l_knee') > y('l_ankle') && y('l_ankle') > y('l_foot'));
  });
  test('the dancer\'s left joints are on +x', () => {
    for (const [l, r] of [['l_hip', 'r_hip'], ['l_shoulder', 'r_shoulder'], ['l_knee', 'r_knee'], ['l_wrist', 'r_wrist']]) assert.ok(J3(J, l)[0] > J3(J, r)[0], l);
  });
  test('bone lengths are human (metres)', () => {
    const b = boneLengths(J);
    const bone = name => b[SMPL_JOINTS.indexOf(name)];
    assert.ok(bone('l_knee') > 0.3 && bone('l_knee') < 0.6, `thigh ${bone('l_knee')}`);
    assert.ok(bone('l_elbow') > 0.2 && bone('l_elbow') < 0.4, `upper arm ${bone('l_elbow')}`);
    assert.ok(bone('l_hand') > 0.03 && bone('l_hand') < 0.15, `hand ${bone('l_hand')}`);
    assert.ok(b.slice(1).every(v => v > 0.01 && v < 0.7), b.map(v => v.toFixed(2)).join(' '));
  });
  test('tracked joints survive the round trip to MediaPipe world landmarks', () => {
    const back = worldFromSmpl(J);
    for (let j = 1; j < 13; j++) {
      const a = [f.world[j * 3], f.world[j * 3 + 1], f.world[j * 3 + 2]], b = [back[j * 3], back[j * 3 + 1], back[j * 3 + 2]];
      assert.ok(dist(a, b) < 1e-4, `joint ${j}: ${a} vs ${b}`);
    }
  });
  test('2D landmarks alone (MoveNet) give the same proportions', () => {
    const J2 = smplFromBody(bodyFromLandmarks(f.lm, 16 / 9), [0, 0.95, 0]);
    const thigh = dist(J3(J2, 'l_hip'), J3(J2, 'l_knee')), torso = dist(J3(J2, 'pelvis'), J3(J2, 'neck'));
    assert.ok(Math.abs(thigh / torso - dist(J3(J, 'l_hip'), J3(J, 'l_knee')) / dist(J3(J, 'pelvis'), J3(J, 'neck'))) < 0.15);
  });
  test('the converter estimates the pelvis height from the legs and moves the root with you', () => {
    const conv = createUserConverter({ smoothing: 1 });
    const a = conv.convert(frame(150), 16 / 9);
    assert.ok(a[1] > 0.7 && a[1] < 1.2, `pelvis height ${a[1]}`);
    const moved = frame(150);
    const lm = Float32Array.from(moved.lm); for (let i = 0; i < 33; i++) lm[i * 4] += 0.1; // 10% of the picture to the right
    const b = conv.convert({ lm, world: moved.world }, 16 / 9);
    assert.ok(b[0] - a[0] > 0.15, `root x moved ${b[0] - a[0]} m`);
    assert.ok(Math.abs(b[2] - a[2]) < 0.05, 'same distance from the camera');
  });
});

describe('SMPL-24 joints → landmarks for drawing', () => {
  const J = smplFromBody(bodyFromWorld(frame(1050).world), [0, 0.95, 0]); // arms up (35 s)
  const lm = landmarksFromSmpl(J);
  test('a partner facing the camera: their left on the picture\'s right, head at the top', () => {
    assert.ok(lm[11 * 4] > lm[12 * 4] && lm[23 * 4] > lm[24 * 4]);
    assert.ok(lm[1] < lm[11 * 4 + 1] && lm[11 * 4 + 1] < lm[23 * 4 + 1] && lm[23 * 4 + 1] < lm[27 * 4 + 1]);
    assert.ok(lm[15 * 4 + 1] < lm[11 * 4 + 1], 'arms up: wrists above shoulders');
  });
  test('the whole body fits in the virtual picture', () => {
    for (let i = 0; i < 33; i++) {
      assert.ok(lm[i * 4] > 0 && lm[i * 4] < 1 && lm[i * 4 + 1] > 0 && lm[i * 4 + 1] < 1, `landmark ${i}`);
      assert.ok(lm[i * 4 + 3] > 0.5);
    }
  });
});

test('resampling 30 fps to 60 fps interpolates between frames', () => {
  const rs = createResampler(60);
  const out = [];
  for (let k = 0; k <= 30; k++) rs.push(k * 1000 / 30, Float32Array.of(k), f => out.push(f[0]));
  assert.ok(out.length >= 60 && out.length <= 62, `${out.length} frames for 1 s`);
  assert.ok(Math.abs(out[1] - 0.5) < 0.05, `first in-between frame ${out[1]}`);
  for (let i = 1; i < out.length; i++) assert.ok(out[i] >= out[i - 1]);
});

test('a recorded session converts to 60 fps Altlando input', () => {
  const view = personView(session, 0);
  const frames = Array.from({ length: 300 }, (_, i) => ({ t: view.times[i], lm: view.lm.subarray(i * FRAME_SIZE, (i + 1) * FRAME_SIZE), world: view.world.subarray(i * WORLD_SIZE, (i + 1) * WORLD_SIZE) }));
  const seq = smplSequence(frames);
  assert.ok(Math.abs(seq.length - 600) <= 3, `${seq.length} frames for 10 s`);
  assert.ok(seq.every(f => f.length === 72 && f.every(Number.isFinite)));
});

// ---- the partner, against a fake model ----------------------------------------------------------
function fakeModel({ fail = 0, delay = 0 } = {}) {
  const calls = [];
  let failures = fail;
  const generate = async (frames, { shift, length }) => {
    calls.push({ frames, shift, length });
    if (delay) await new Promise(r => setTimeout(r, delay));
    if (failures > 0) { failures--; throw new Error('out of memory'); }
    // like Altlando: shift*8 frames re-reading the input, then length*8 new ones; root starts at 0
    const n = (shift + length) * CODE_FRAMES, r0 = frames[0];
    return Array.from({ length: n }, (_, k) => {
      const f = Float32Array.from(frames[k % frames.length]);
      for (let i = 0; i < 24; i++) { f[i * 3] -= r0[0]; f[i * 3 + 2] -= r0[2]; }
      if (k >= shift * CODE_FRAMES) f[0] += 0.001 * k; // mark the generated part
      return f;
    });
  };
  return { generate, calls };
}

describe('createPartner', () => {
  const f = frame(150);
  const J = smplFromBody(bodyFromWorld(f.world), [0, 0.95, -2.5]);
  const feed = (partner, fromMs, toMs, fps = 30) => { for (let t = fromMs; t <= toMs; t += 1000 / fps) partner.push(t, J); };

  test('waits for 3.7 s of movement, then asks the model with AIST++-style frames', async () => {
    const model = fakeModel();
    const statuses = [];
    const partner = createPartner({ generate: model.generate, onStatus: s => statuses.push(s.status) });
    partner.start();
    feed(partner, 0, 3000);
    assert.equal(partner.frameAt(3000), null);
    assert.equal(model.calls.length, 0, 'not enough movement yet');
    feed(partner, 3033, 4000);
    partner.frameAt(4000);
    assert.equal(model.calls.length, 1);
    const { frames, shift, length } = model.calls[0];
    assert.deepEqual([shift, length, frames.length], [28, 28, 224]);
    assert.ok(frames.every(fr => fr.length === 72 && fr.every(Number.isFinite)));
    await partner.settle();
    assert.deepEqual(statuses.slice(0, 3), ['watching', 'thinking', 'dancing']);
  });

  test('plays the generated continuation, not the re-read input, starting where the partner stands', async () => {
    const model = fakeModel();
    const partner = createPartner({ generate: model.generate });
    partner.start(); feed(partner, 0, 4000); partner.frameAt(4000); await partner.settle();
    const first = partner.frameAt(5000);
    assert.ok(first, 'dancing');
    assert.ok(Math.abs(first[0]) < 1e-6 && Math.abs(first[2]) < 1e-6, 'starts in the middle of its spot');
    const later = partner.frameAt(5500);
    assert.ok(later[0] > first[0], 'moving through the generated frames');
    assert.equal(partner.info().queued, 224 - 30);
  });

  test('with "mimic" it also shows the model\'s reading of your movement', async () => {
    const model = fakeModel();
    const partner = createPartner({ generate: model.generate, mimic: true });
    partner.start(); feed(partner, 0, 4000); partner.frameAt(4000); await partner.settle();
    partner.frameAt(5000);
    assert.equal(partner.info().queued, 448);
  });

  test('asks again before the reply runs out, and fades into the next one', async () => {
    const model = fakeModel();
    const partner = createPartner({ generate: model.generate });
    partner.start(); feed(partner, 0, 4000); partner.frameAt(4000); await partner.settle();
    partner.frameAt(4000);
    let t = 4000;
    while (model.calls.length < 2 && t < 9000) { t += 50; feed(partner, t - 50, t); partner.frameAt(t); }
    assert.equal(model.calls.length, 2, 'second request');
    const remaining = partner.info().queued;
    assert.ok(remaining > 0 && remaining < 224, `asked with ${remaining} frames left`);
    const before = Float32Array.from(partner.frameAt(t));
    await partner.settle();
    const after = partner.frameAt(t + 1);
    assert.ok(Math.abs(after[3 * 15 + 1] - before[3 * 15 + 1]) < 0.05, 'no jump at the switch');
  });

  test('reports a model error, backs off, and recovers', async () => {
    const model = fakeModel({ fail: 1 });
    const errors = [];
    const partner = createPartner({ generate: model.generate, onStatus: s => s.error && errors.push(s.error), retryMs: 50 });
    partner.start(); feed(partner, 0, 4000); partner.frameAt(4000); await partner.settle();
    assert.equal(partner.info().status, 'error');
    assert.equal(errors[0], 'out of memory');
    partner.frameAt(4100);
    assert.equal(model.calls.length, 1, 'waits before retrying');
    await new Promise(r => setTimeout(r, 80));
    feed(partner, 4000, 4200); partner.frameAt(4200); await partner.settle();
    assert.equal(model.calls.length, 2);
    assert.equal(partner.info().status, 'dancing');
  });

  test('stop() clears everything', async () => {
    const model = fakeModel();
    const partner = createPartner({ generate: model.generate });
    partner.start(); feed(partner, 0, 4000); partner.frameAt(4000); await partner.settle();
    partner.stop();
    assert.equal(partner.frameAt(5000), null);
    assert.equal(partner.info().buffered, 0);
  });
});
