// Multiplayer wire format and link helpers.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { packPoses, unpackPoses, packWorlds, unpackWorlds, roomFromHash, signalOptions, makeRoomCode } from '../../js/multiplayer.js';

test('poses survive packing to within 1/30000 of the frame, including just outside the picture', () => {
  const lm = new Float32Array(132);
  for (let i = 0; i < 33; i++) { lm[i * 4] = -0.2 + i * 0.04; lm[i * 4 + 1] = 1.3 - i * 0.03; lm[i * 4 + 3] = i / 32; }
  const buf = packPoses([lm, lm]);
  assert.equal(buf.byteLength, 2 * (1 + 2 * 33 * 3));
  const back = unpackPoses(buf);
  assert.equal(back.length, 2);
  for (let i = 0; i < 33; i++) for (const c of [0, 1, 3]) assert.ok(Math.abs(back[1][i * 4 + c] - lm[i * 4 + c]) < 3.5e-5, `landmark ${i}.${c}`);
});

test('a truncated pose message unpacks to nothing', () => {
  const buf = packPoses([new Float32Array(132)]);
  assert.deepEqual(unpackPoses(buf.slice(0, 20)), []);
});

test('world landmarks survive packing to 0.1 mm; missing ones stay null', () => {
  const w = Float32Array.from({ length: 39 }, (_, i) => (i - 19) * 0.05);
  const back = unpackWorlds(packWorlds([w, null]), 2);
  assert.equal(back.length, 2);
  assert.equal(back[1], null);
  for (let i = 0; i < 39; i++) assert.ok(Math.abs(back[0][i] - w[i]) < 1e-4);
});

test('room codes come from #join= links', () => {
  assert.equal(roomFromHash('#join=abc23defgh'), 'abc23defgh');
  assert.equal(roomFromHash('#foo&join=ABCDEFGH'), 'abcdefgh');
  assert.equal(roomFromHash('#x'), null);
  assert.equal(roomFromHash(''), null);
  const code = makeRoomCode();
  assert.match(code, /^[a-km-np-z2-9]{10}$/);
  assert.equal(roomFromHash(`#join=${code}`), code);
});

test('?signal= points at a self-hosted PeerJS server', () => {
  assert.deepEqual(signalOptions('?signal=ws://localhost:9000/ks'), { host: 'localhost', port: 9000, path: '/ks', secure: false, key: 'peerjs' });
  assert.deepEqual(signalOptions('?signal=wss://peer.example.org'), { host: 'peer.example.org', port: 443, path: '/', secure: true, key: 'peerjs' });
  assert.deepEqual(signalOptions(''), {});
  assert.deepEqual(signalOptions('?signal=not a url'), {});
});
