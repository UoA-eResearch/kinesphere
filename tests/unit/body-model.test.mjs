// The fitted body and the force-based deformer.
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { createRig, buildBody, createDeformer } from '../../js/body-model.js';

const W = 1280, H = 720;
/** Projected MoveNet-style keypoints in pixels; `arm` swings the right forearm (radians). */
function joints(arm = 0, dx = 0) {
  const x = new Float32Array(33).fill(NaN), y = new Float32Array(33).fill(NaN), v = new Float32Array(33);
  const set = (k, px, py) => { x[k] = (px + dx) * W; y[k] = py * H; v[k] = 0.9; };
  set(0, 0.5, 0.2); set(7, 0.525, 0.2); set(8, 0.475, 0.2); set(11, 0.56, 0.3); set(12, 0.44, 0.3); set(23, 0.545, 0.53); set(24, 0.455, 0.53);
  set(13, 0.62, 0.42); set(15, 0.64, 0.52); set(25, 0.55, 0.7); set(27, 0.55, 0.86); set(26, 0.45, 0.7); set(28, 0.45, 0.86);
  set(14, 0.38, 0.42); set(16, 0.38 - 0.1 * Math.sin(arm), 0.42 + 0.1 * Math.cos(arm));
  return { x, y, v };
}
const finite = pts => pts.every(p => Number.isFinite(p[0]) && Number.isFinite(p[1]));

describe('buildBody', () => {
  test('builds closed outlines for the torso, head and all four limbs from 17 keypoints', () => {
    const B = buildBody(joints(), { samples: 10 });
    assert.ok(B && B.T > 0);
    assert.ok(B.torso.length > 10 && finite(B.torso));
    assert.ok(B.head.outline.length > 10 && finite(B.head.outline));
    for (const L of [...B.arms, ...B.legs]) {
      assert.ok(L, 'limb present');
      assert.ok(finite(L.outline) && L.outline.length > 10);
      assert.equal(L.joints.length, 4);
    }
  });
  test('returns null without shoulders and hips', () => {
    const J = joints(); J.v[23] = 0;
    assert.equal(buildBody(J), null);
  });
  test('the rig smooths a jump in a joint instead of snapping to it', () => {
    const rig = createRig();
    for (let i = 0; i < 30; i++) rig.update(joints(), 1 / 60);
    const before = rig.update(joints(), 1 / 60).x[16];
    const target = joints(Math.PI / 2).x[16];
    const after = rig.update(joints(Math.PI / 2), 1 / 60).x[16];
    assert.ok(Math.abs(after - before) < Math.abs(target - before) * 0.5);
  });
});

describe('createDeformer', () => {
  const run = (frames, strength = 1) => {
    const d = createDeformer();
    let info, B;
    for (const [arm, dx] of frames) { B = buildBody(joints(arm, dx), { samples: 12 }); info = d.apply(B, 1 / 60, strength); }
    return { info, B };
  };
  const still = Array.from({ length: 60 }, () => [0, 0]);
  const swing = [...still, ...Array.from({ length: 12 }, (_, i) => [i * 0.25, 0])]; // a fast forearm sweep

  test('a still body is not deformed', () => {
    const { info } = run(still);
    for (const L of info.limbs) { assert.ok(L.lag < 0.01, `lag ${L.lag}`); assert.ok(Math.abs(L.stretch - 1) < 0.01, `stretch ${L.stretch}`); }
    assert.ok(info.body < 0.01);
  });
  test('a fast swing bends the moving limb behind its hand; the others stay put', () => {
    const { info, B } = run(swing);
    const right = info.limbs[1];
    assert.ok(right.lag > 0.1, `right arm lag ${right.lag} torso lengths`);
    assert.ok(info.limbs[0].lag < 0.02 && info.limbs[2].lag < 0.02 && info.limbs[3].lag < 0.02);
    assert.ok(finite(B.arms[1].outline));
  });
  test('the deformation scales with strength', () => {
    const weak = run(swing, 0.5).info.limbs[1].lag, strong = run(swing, 1.25).info.limbs[1].lag;
    assert.ok(strong > weak * 2, `strong ${strong} vs weak ${weak}`);
  });
  test('the limb springs back after the move stops', () => {
    const end = swing[swing.length - 1][0];
    const { info } = run([...swing, ...Array.from({ length: 60 }, () => [end, 0])]);
    assert.ok(info.limbs[1].lag < 0.02, `lag after 1 s at rest ${info.limbs[1].lag}`);
  });
  test('the torso squashes and stretches when the body moves sharply', () => {
    const { info } = run([...still, ...Array.from({ length: 6 }, (_, i) => [0, i * 0.01])]);
    assert.ok(info.body > 0.02, `torso stretch ${info.body}`);
    assert.ok(info.body <= 0.35);
  });
  test('a teleport (a replay seek, a lost person) resets instead of smearing across the frame', () => {
    const { info } = run([...still, [0, 0.4]]);
    assert.ok(info.limbs.every(L => L.lag < 0.01));
  });
  test('stays finite through missing frames and long gaps', () => {
    const d = createDeformer();
    assert.equal(d.apply(null, 1 / 60), null);
    const B = buildBody(joints(), { samples: 12 });
    const info = d.apply(B, 5, 1); // a 5 s gap
    assert.ok(info.limbs.every(L => Number.isFinite(L.lag) && Number.isFinite(L.stretch)));
  });
});
