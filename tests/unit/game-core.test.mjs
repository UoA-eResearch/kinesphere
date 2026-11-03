// Game mode scoring: every move matched against a player doing that move.
import { test, describe } from 'node:test';
import assert from 'node:assert/strict';
import { MOVES, SONGS, posePoints, buildChoreography, playerAngles, matchScore, rate, maxPointsFor, starsFor } from '../../js/game-core.js';

const aspect = 16 / 9;
// A player facing the camera doing move m as a mirror image; landmarks in raw (un-mirrored) camera space.
function landmarksFor(m, jitterDeg = 0, seed = 1) {
  let s = seed; const rnd = () => { s = (s * 16807) % 2147483647; return (s / 2147483647 - 0.5) * 2; };
  const jm = {}; for (const k of Object.keys(m)) jm[k] = typeof m[k] === 'number' ? m[k] + rnd() * jitterDeg : m[k];
  const P = posePoints(jm);
  const lm = new Float32Array(33 * 4);
  const put = (i, [x, y]) => { lm[i * 4] = 0.5 - x * 0.15 / aspect; lm[i * 4 + 1] = 0.4 + y * 0.15; lm[i * 4 + 3] = 1; };
  put(11, P.shL); put(12, P.shR); put(13, P.elL); put(14, P.elR); put(15, P.wrL); put(16, P.wrR);
  put(23, P.hipL); put(24, P.hipR); put(25, P.knL); put(26, P.knR); put(27, P.anL); put(28, P.anR); put(0, P.head);
  return lm;
}
const ids = Object.keys(MOVES);

describe('every move', () => {
  for (const id of ids) test(id, () => {
    const self = matchScore(MOVES[id], playerAngles(landmarksFor(MOVES[id]), aspect));
    const jittered = matchScore(MOVES[id], playerAngles(landmarksFor(MOVES[id], 12, 7), aspect));
    assert.equal(rate(self).key, 'perfect', `exact copy scores ${self}`);
    assert.equal(rate(jittered).key, 'perfect', `±12° copy scores ${jittered}`);
    for (const o of ids) if (o !== id) {
      const other = matchScore(MOVES[o], playerAngles(landmarksFor(MOVES[id]), aspect));
      assert.ok(other < 0.85, `${id} also scores ${other.toFixed(2)} as ${o}`);
    }
  });
});

describe('the wrong side does not score Perfect', () => {
  for (const id of ['reachL', 'discoL', 'pointL', 'waveL', 'kneeL', 'kickL', 'egypt']) test(id, () => {
    const wrong = id.endsWith('L') ? `${id.slice(0, -1)}R` : 'egyptB';
    assert.ok(matchScore(MOVES[id], playerAngles(landmarksFor(MOVES[wrong]), aspect)) < 0.7);
  });
});

test('arms out of view score badly; legs out of view are ignored', () => {
  const noArms = landmarksFor(MOVES.vup); for (const k of [13, 14, 15, 16]) noArms[k * 4 + 3] = 0;
  const noLegs = landmarksFor(MOVES.vup); for (const k of [25, 26, 27, 28]) noLegs[k * 4 + 3] = 0;
  assert.ok(matchScore(MOVES.vup, playerAngles(noArms, aspect)) < 0.4);
  assert.ok(matchScore(MOVES.vup, playerAngles(noLegs, aspect)) > 0.85);
});

describe('songs', () => {
  for (const song of SONGS) test(song.title, () => {
    const c = buildChoreography(song);
    assert.ok(c.moves.length > 10);
    assert.deepEqual(c.moves.filter(m => !m.move).map(m => m.id), [], 'every move exists');
    assert.ok(c.moves.some(m => m.gold), 'has gold moves');
    const max = maxPointsFor(c.moves);
    assert.equal(starsFor(0, max).stars, 0);
    assert.equal(starsFor(0.5 * max, max).stars, 3);
    assert.deepEqual([starsFor(max, max).stars, starsFor(max, max).superstar], [5, true]);
  });
});
