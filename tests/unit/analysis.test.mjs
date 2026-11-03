// Movement analysis against synthetic sessions with known ground truth.
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { phasesSession, floorSession, PHASES, serialize } from '../helpers/synth.mjs';
import { deserialize, personView } from '../../js/session.js';
import { analyze, KINESPHERE_CLASSES, LEVELS } from '../../js/analysis.js';

const phases = deserialize(serialize(phasesSession(1)));
const LABEL = Object.fromEntries(KINESPHERE_CLASSES.map(c => [c.key, c.label]));

/** The most common class per phase, ignoring 2.5 s at each edge (the body scale is a rolling estimate). */
function classesByPhase(s, a) {
  return PHASES.map((_, ph) => {
    const k = {}, l = {};
    for (let i = 0; i < s.frameCount; i++) {
      const t = s.times[i] / 1000;
      if (t < ph * 10 + 2.5 || t >= ph * 10 + 7.5) continue;
      const kc = KINESPHERE_CLASSES[a.frames.kClass[i]]?.key ?? 'none', lc = LEVELS[a.frames.level[i]]?.key ?? 'none';
      k[kc] = (k[kc] || 0) + 1; l[lc] = (l[lc] || 0) + 1;
    }
    const top = o => Object.entries(o).sort((x, y) => y[1] - x[1])[0][0];
    return { kinesphere: top(k), level: top(l), counts: k };
  });
}

describe('kinesphere detection (near / medium / far)', () => {
  test('the three kinesphere classes are labelled Near, Medium and Far', () => {
    assert.deepEqual(KINESPHERE_CLASSES.map(c => c.label), ['Near', 'Medium', 'Far']);
  });
  for (const mode of ['3d', '2d']) {
    describe(`measured in ${mode.toUpperCase()}${mode === '2d' ? ' (no world landmarks)' : ''}`, () => {
      const s = mode === '3d' ? phases : { ...phases, world: null };
      const a = analyze(personView(s, 0));
      const got = classesByPhase(s, a);
      test(`uses ${mode} analysis`, () => assert.equal(a.kinesphere.mode, mode));
      PHASES.forEach(([name, k3, , k2], ph) => {
        const want = mode === '3d' ? k3 : k2;
        test(`${name} → ${LABEL[want]}`, () => assert.equal(got[ph].kinesphere, want, `counts ${JSON.stringify(got[ph].counts)}`));
      });
      test('every class is detected somewhere and the shares sum to 1', () => {
        const shares = a.kinesphere.shares;
        for (const c of KINESPHERE_CLASSES) assert.ok(shares[c.key] > 0.05, `${c.label} share ${shares[c.key]}`);
        const sum = Object.values(shares).reduce((x, y) => x + y, 0);
        assert.ok(Math.abs(sum - 1) < 0.02, `shares sum to ${sum}`);
      });
    });
  }
});

describe('space levels (high / medium / low)', () => {
  const a = analyze(personView(phases, 0));
  const got = classesByPhase(phases, a);
  PHASES.forEach(([name, , level], ph) => {
    test(`${name} → ${level}`, () => assert.equal(got[ph].level, level));
  });
});

describe('floor path (bird\'s-eye view)', () => {
  const s = deserialize(serialize(floorSession(80)));
  const a = analyze(personView(s, 0));
  const sp = a.space;
  test('recovers the 2 m × 1.5 m square walked 2.5–4 m from the camera', () => {
    assert.ok(Math.abs(sp.width - 2) < 0.25, `width ${sp.width}`);
    assert.ok(Math.abs(sp.depth - 1.5) < 0.25, `depth ${sp.depth}`);
    assert.ok(Math.abs(sp.zRange[0] - 2.5) < 0.25 && Math.abs(sp.zRange[1] - 4) < 0.25, `z range ${sp.zRange}`);
    assert.ok(Math.abs(sp.xRange[0] + 1) < 0.2 && Math.abs(sp.xRange[1] - 1) < 0.2, `x range ${sp.xRange}`);
  });
  test('uses the 3D torso length (0.5 m)', () => assert.ok(Math.abs(sp.torsoM - 0.5) < 0.03, `torso ${sp.torsoM}`));
  test('distance travelled is close to the true 2 laps (~14 m, less the time too close)', () => {
    assert.ok(sp.travelled > 10 && sp.travelled < 16, `travelled ${sp.travelled}`);
  });
  test('the path is a list of timed floor points', () => {
    assert.ok(sp.path.length > 100);
    for (const p of sp.path) assert.ok(Number.isFinite(p.x) && Number.isFinite(p.z) && Number.isFinite(p.t));
  });
});

describe('speed outliers', () => {
  const clean = analyze(personView(deserialize(serialize(floorSession(80, { glitches: false }))), 0)).speed;
  const glitchy = analyze(personView(deserialize(serialize(floorSession(80))), 0)).speed;
  test('tracking glitches do not create speed spikes', () => {
    assert.ok(glitchy.peak <= clean.peak * 1.25, `peak with glitches ${glitchy.peak} vs clean ${clean.peak}`);
    assert.ok(Math.abs(glitchy.mean - clean.mean) / clean.mean < 0.1, `mean ${glitchy.mean} vs ${clean.mean}`);
  });
  test('only a small share of samples is removed', () => {
    assert.ok(glitchy.removed > 0 && glitchy.removed < 0.02, `removed ${glitchy.removed}`);
  });
});

describe('session format', () => {
  test('serialize → deserialize keeps frames, positions and world landmarks', () => {
    const s = phasesSession(2);
    const back = deserialize(serialize(s));
    assert.equal(back.frameCount, s.frameCount);
    assert.equal(back.people, 2);
    assert.equal(serialize(back), serialize(s));
  });
});
