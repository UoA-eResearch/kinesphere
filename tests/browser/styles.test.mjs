// Every overlay style draws something, stays fast enough not to freeze a computer, and copes with
// a person appearing and disappearing.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { openApp } from '../helpers/browser.mjs';

// Generous: CI runners draw on the CPU. The bug this guards against took ~4 s a frame.
const MAX_MS_PER_FRAME = Number(process.env.KINESPHERE_MAX_FRAME_MS || 250);
const IDS = ['skeleton', 'stickman', 'neon', 'trails', 'sparks', 'constellation', 'sandman', 'fuzzy', 'origami', 'pencil', 'hoops', 'animator', 'pleats'];
let app, results;

before(async () => {
  app = await openApp();
  results = await app.page.evaluate(async () => {
    const { createEffect, STYLES } = await import('/js/effects.js');
    const lm = new Float32Array(132);
    const set = (k, x, y) => { lm[k * 4] = x; lm[k * 4 + 1] = y; lm[k * 4 + 3] = 0.9; };
    set(0, 0.5, 0.2); set(7, 0.525, 0.2); set(8, 0.475, 0.2); set(11, 0.56, 0.3); set(12, 0.44, 0.3); set(23, 0.545, 0.53); set(24, 0.455, 0.53);
    set(13, 0.64, 0.33); set(15, 0.66, 0.22); set(14, 0.37, 0.36); set(16, 0.31, 0.44); set(25, 0.6, 0.62); set(27, 0.58, 0.72); set(26, 0.45, 0.7); set(28, 0.45, 0.86);
    const W = 1280, H = 720;
    const inked = ctx => { const d = ctx.getImageData(0, 0, W, H).data; let n = 0; for (let i = 3; i < d.length; i += 40) if (d[i]) n++; return n / (d.length / 40); };
    const out = {};
    for (const { id } of STYLES) {
      const r = out[id] = {};
      for (const videoVisible of [true, false]) {
        const c = document.createElement('canvas'); c.width = W; c.height = H;
        const ctx = c.getContext('2d'), e = createEffect(id);
        const opts = { person: 0, minVis: 0.5, videoVisible };
        const t0 = performance.now();
        for (let i = 0; i < 30; i++) {
          lm[16 * 4] = 0.31 + 0.12 * Math.sin(i * 0.5); lm[16 * 4 + 1] = 0.44 - 0.1 * Math.cos(i * 0.5);
          ctx.clearRect(0, 0, W, H);
          e.draw(ctx, lm, 0, W, H, 1 / 60, opts);
          ctx.getImageData(0, 0, 1, 1); // wait for the drawing to finish
        }
        const key = videoVisible ? 'Video' : 'NoVideo';
        r[`ms${key}`] = (performance.now() - t0) / 30;
        r[`ink${key}`] = inked(ctx);
        // the person leaves, then comes back
        e.draw(ctx, null, 0, W, H, 1 / 60, opts);
        e.reset?.();
        ctx.clearRect(0, 0, W, H);
        for (let i = 0; i < 3; i++) e.draw(ctx, lm, 0, W, H, 1 / 60, opts);
        r[`inkAfterReset${key}`] = inked(ctx);
      }
    }
    return out;
  });
  const ms = Object.fromEntries(Object.entries(results).map(([k, v]) => [k, +Math.max(v.msVideo, v.msNoVideo).toFixed(1)]));
  console.log('ms per frame:', JSON.stringify(ms));
});
after(async () => { await app?.close(); });

test('the style list is complete', () => assert.deepEqual(Object.keys(results), IDS));
for (const id of IDS) {
  test(`${id}: draws the dancer over the camera feed and on its own background`, () => {
    const r = results[id];
    assert.ok(r, 'style exists');
    assert.ok(r.inkVideo > 0.002, `ink over video ${r.inkVideo}`);
    assert.ok(r.inkNoVideo > 0.002, `ink without video ${r.inkNoVideo}`);
    assert.ok(r.inkAfterResetVideo > 0.002 && r.inkAfterResetNoVideo > 0.002, 'draws again after the person leaves and returns');
  });
  test(`${id}: under ${MAX_MS_PER_FRAME} ms a frame`, () => {
    const r = results[id];
    assert.ok(r.msVideo < MAX_MS_PER_FRAME && r.msNoVideo < MAX_MS_PER_FRAME, `${r.msVideo.toFixed(1)} / ${r.msNoVideo.toFixed(1)} ms`);
  });
}
// Poses a camera produces when someone is partly out of frame or the tracking glitches, drawn at
// 60, 30 and 15 frames a second (slow frames made the body rig unstable, so the fitted-body styles
// threw on non-finite gradients; a collapsed torso hung fuzzy blur; all joints on one point broke
// sandman).
test('every style copes with people partly out of frame and tracking glitches, at any frame rate', async () => {
  const failures = await app.page.evaluate(async () => {
    const { createEffect, STYLES } = await import('/js/effects.js');
    let seed = 1; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    const base = () => {
      const lm = new Float32Array(132);
      const set = (k, x, y, v = 0.95) => { lm[k * 4] = x; lm[k * 4 + 1] = y; lm[k * 4 + 3] = v; };
      set(0, 0.5, 0.2); for (let k = 1; k <= 10; k++) set(k, 0.5 + (k % 2 ? 0.02 : -0.02), 0.19);
      set(11, 0.56, 0.3); set(12, 0.44, 0.3); set(13, 0.64, 0.33); set(15, 0.66, 0.22); set(14, 0.37, 0.36); set(16, 0.31, 0.44);
      for (const k of [17, 19, 21]) set(k, 0.67, 0.2); for (const k of [18, 20, 22]) set(k, 0.3, 0.46);
      set(23, 0.545, 0.53); set(24, 0.455, 0.53); set(25, 0.6, 0.62); set(27, 0.58, 0.72); set(26, 0.45, 0.7); set(28, 0.45, 0.86);
      set(29, 0.58, 0.74); set(30, 0.45, 0.88); set(31, 0.6, 0.75); set(32, 0.43, 0.89);
      return lm;
    };
    const scenarios = {
      'walking up to the camera': i => { const lm = base(), sc = 1 + i * 0.04; for (let k = 0; k < 33; k++) { lm[k * 4] = 0.5 + (lm[k * 4] - 0.5) * sc; lm[k * 4 + 1] = 0.3 + (lm[k * 4 + 1] - 0.3) * sc; lm[k * 4 + 3] = lm[k * 4 + 1] > 1 || lm[k * 4] < 0 || lm[k * 4] > 1 ? 0.3 + 0.3 * rnd() : 0.9; } return lm; },
      'flickering visibility': () => { const lm = base(); for (let k = 0; k < 33; k++) if (rnd() < 0.3) lm[k * 4 + 3] = rnd() * 0.6; return lm; },
      'legs out of frame': i => { const lm = base(); for (const k of [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]) { lm[k * 4 + 1] = 1; lm[k * 4] = 0.5; lm[k * 4 + 3] = i % 3 ? 0.55 : 0.4; } return lm; },
      'collapsed torso': i => { const lm = base(); for (const k of [11, 12, 23, 24]) { lm[k * 4] = 0.5; lm[k * 4 + 1] = 0.4 + (i % 2) * 1e-7; } return lm; },
      'every joint on one point': () => { const lm = base(); for (let k = 0; k < 33; k++) { lm[k * 4] = 0.5; lm[k * 4 + 1] = 0.5; } return lm; },
      'a joint jumping away': i => { const lm = base(); if (i % 5 === 2) { lm[16 * 4] = 5; lm[16 * 4 + 1] = -3; } return lm; },
    };
    const out = [];
    for (const { id } of STYLES) {
      for (const [name, gen] of Object.entries(scenarios)) {
        const c = document.createElement('canvas'); c.width = 640; c.height = 360;
        const ctx = c.getContext('2d'), e = createEffect(id), t0 = performance.now();
        for (let i = 0; i < 45; i++) {
          try { e.draw(ctx, gen(i), 0, 640, 360, [1 / 60, 1 / 30, 1 / 15][i % 3], { person: 0, minVis: 0.5, videoVisible: i % 2 === 0 }); }
          catch (err) { out.push(`${id} / ${name}, frame ${i}: ${err.message}`); break; }
          if (performance.now() - t0 > 5000) { out.push(`${id} / ${name}: too slow (frame ${i})`); break; }
        }
      }
    }
    return out;
  });
  assert.deepEqual(failures, []);
});

test('no page errors', () => assert.deepEqual(app.errors, []));
