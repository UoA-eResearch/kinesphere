// The AI dance partner end to end, with the model running in the browser: a small random-weight
// export of the real Altlando architecture (npm run build:test-model -> test-results/altlando-tiny),
// a fake camera and a scripted dancer.
import { test as baseTest, before, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { openApp, startCameraWithDancer, setDancer, importSession, CAMERA_ARGS, ROOT } from '../helpers/browser.mjs';
import { phasesSession, serialize } from '../helpers/synth.mjs';
import { smplSequence } from '../../js/ai-partner.js';
import { FRAME_SIZE, WORLD_SIZE } from '../../js/pose.js';

/** 224 frames of a synthetic dancer (arms at 45 deg, then raising them), as Kinesphere sends them to the model. */
function humanInput() {
  const s = phasesSession(1), frames = [];
  for (let i = 400; i < 540; i++) frames.push({ t: s.times[i], lm: s.lm.subarray(i * FRAME_SIZE, (i + 1) * FRAME_SIZE), world: s.world.subarray(i * WORLD_SIZE, (i + 1) * WORLD_SIZE) });
  return smplSequence(frames, 16 / 9).slice(0, 224);
}

const test = (name, fn) => baseTest(name, { timeout: 120000 }, fn);
const MODEL = 'test-results/altlando-tiny/';
const MODEL_DIR = path.join(ROOT, MODEL);
let app;

/** A dancer swinging both arms, stepping side to side. */
function swingingDancer(t) {
  const lm = new Float32Array(132);
  const set = (k, x, y) => { lm[k * 4] = x; lm[k * 4 + 1] = y; lm[k * 4 + 3] = 0.95; };
  const cx = 0.5 + 0.05 * Math.sin(t * 1.3), a = 1.2 + Math.sin(t * 3);
  set(0, cx, 0.2); set(7, cx + 0.02, 0.19); set(8, cx - 0.02, 0.19);
  set(11, cx + 0.06, 0.3); set(12, cx - 0.06, 0.3); set(23, cx + 0.045, 0.53); set(24, cx - 0.045, 0.53);
  set(13, cx + 0.06 + 0.09 * Math.sin(a), 0.3 + 0.09 * Math.cos(a)); set(15, cx + 0.06 + 0.17 * Math.sin(a), 0.3 + 0.17 * Math.cos(a));
  set(14, cx - 0.06 - 0.09 * Math.sin(a), 0.3 + 0.09 * Math.cos(a)); set(16, cx - 0.06 - 0.17 * Math.sin(a), 0.3 + 0.17 * Math.cos(a));
  set(25, cx + 0.05, 0.7); set(27, cx + 0.05, 0.87); set(26, cx - 0.05, 0.7); set(28, cx - 0.05, 0.87);
  return lm;
}
const status = page => page.$eval('#ai-status', el => el.textContent.trim());
const waitStatus = (page, re, timeout = 30000) => page.waitForFunction(src => new RegExp(src).test(document.querySelector('#ai-status').textContent), re.source, { timeout });

before(async () => {
  assert.ok(fs.existsSync(path.join(MODEL_DIR, 'altlando.json')), `no test model in ${MODEL}: run "npm run build:test-model" (needs Python with PyTorch, see ai-partner/requirements-export.txt)`);
  app = await openApp({ viewport: { width: 1200, height: 900 }, permissions: ['camera'], args: CAMERA_ARGS, query: `?ai=${encodeURIComponent(MODEL)}` });
});
after(async () => { await app?.close(); });

test('in the browser, the model gives exactly what PyTorch gives', async () => {
  const ref = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, 'reference.json'), 'utf8'));
  const out = await app.page.evaluate(async ({ base, payload }) => {
    const { createAltlando } = await import('/js/altlando.js');
    const model = createAltlando();
    const info = await model.load({ base });
    const frames = await model.generate(payload.map(f => Float32Array.from(f)), 28);
    model.terminate();
    return { backend: info.backend, n: frames.length, codesUp: frames.codesUp, codesDown: frames.codesDown, raw: Array.from(frames.raw), root: [frames[0][0], frames[1][0], frames[2][0]] };
  }, { base: `/${MODEL}`, payload: ref.payload });
  assert.equal(out.n, 448, '(28 + 28 codes) x 8 frames');
  assert.deepEqual(out.codesUp, ref.codesUp);
  assert.deepEqual(out.codesDown, ref.codesDown);
  const flat = ref.rawPose.flat();
  let err = 0;
  for (let i = 0; i < flat.length; i++) err = Math.max(err, Math.abs(out.raw[i] - flat[i]));
  assert.ok(err < 1e-4, `decoded pose differs by ${err}`);
  assert.equal(out.root[0], 0, 'the root starts at the origin');
  assert.ok(Math.abs(out.root[1] - flat[0]) < 1e-4, 'and moves by the decoded root velocity');
});

for (const [name, folder, check] of [['Altlando', 'models/altlando/', /Altlando/], ['Bailando (music removed)', 'models/bailando/', /Bailando/]]) {
  test(`the bundled ${name} model loads from ${folder} and matches onnxruntime`, async () => {
    const ref = JSON.parse(fs.readFileSync(path.join(ROOT, folder, 'reference.json'), 'utf8'));
    const out = await app.page.evaluate(async ({ base, payload, human }) => {
      const { createAltlando } = await import('/js/altlando.js');
      const model = createAltlando();
      const info = await model.load({ base });
      const frames = await model.generate(payload.map(f => Float32Array.from(f)), 28);
      // a reply to a (synthetic) dancer: the generated half should still be a human body
      const reply = await model.generate(human.map(f => Float32Array.from(f)), 28);
      model.terminate();
      const P = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21];
      const bones = reply.slice(224).map(f => P.map((p, i) => (p < 0 ? 0 : Math.hypot(f[i * 3] - f[p * 3], f[i * 3 + 1] - f[p * 3 + 1], f[i * 3 + 2] - f[p * 3 + 2]))));
      return { model: info.manifest.model, vocab: info.manifest.vocabUp, n: frames.length, codesUp: frames.codesUp, codesDown: frames.codesDown, raw: Array.from(frames.raw), thigh: bones.map(b => b[4]), shin: bones.map(b => b[7]) };
    }, { base: `/${folder}`, payload: ref.payload, human: humanInput() });
    assert.match(out.model, check);
    assert.equal(out.model.replace(/ \(.*$/, ''), name.split(' ')[0], 'the name over the partner\'s head');
    assert.equal(out.vocab, name === 'Altlando' ? 2048 : 512);
    assert.equal(out.n, 448);
    assert.deepEqual(out.codesUp, ref.codesUp);
    assert.deepEqual(out.codesDown, ref.codesDown);
    const flat = ref.rawPose.flat();
    let err = 0;
    for (let i = 0; i < flat.length; i++) err = Math.max(err, Math.abs(out.raw[i] - flat[i]));
    assert.ok(err < 1e-3, `decoded pose differs by ${err}`);
    const range = a => `${Math.min(...a).toFixed(2)}..${Math.max(...a).toFixed(2)} m`;
    assert.ok(out.thigh.every(l => l > 0.25 && l < 0.6), `thigh length ${range(out.thigh)}`);
    assert.ok(out.shin.every(l => l > 0.25 && l < 0.6), `shin length ${range(out.shin)}`);
  });
}

test('Altlando is the default model, and Bailando can be chosen', async () => {
  const { page } = app;
  // a fresh page without ?ai=, so the default applies
  const ctx = page.context();
  const p2 = await ctx.newPage();
  await p2.goto(new URL('/', page.url()).href, { waitUntil: 'networkidle' });
  await p2.evaluate(() => { document.querySelector('#ai-panel').hidden = false; document.querySelector('.ai-advanced').open = true; });
  assert.equal(await p2.$eval('#ai-choice', s => s.value), 'altlando');
  const links = await p2.$$eval('#ai-panel a[href]', a => a.map(x => x.href));
  assert.ok(links.includes('https://hdl.handle.net/2292/74331'), "links to Jay Zhou's thesis");
  assert.ok(links.includes('https://arxiv.org/abs/2203.13055'), 'links to the Bailando paper');
  assert.equal(await p2.inputValue('#ai-model'), 'models/altlando/');
  await p2.selectOption('#ai-choice', 'bailando');
  assert.equal(await p2.inputValue('#ai-model'), 'models/bailando/');
  assert.equal(await p2.evaluate(() => localStorage.getItem('kinesphere:aiChoice')), 'bailando');
  await p2.reload({ waitUntil: 'networkidle' });
  assert.equal(await p2.$eval('#ai-choice', s => s.value), 'bailando', 'remembered');
  await p2.evaluate(() => { document.querySelector('#ai-panel').hidden = false; });
  await p2.selectOption('#ai-choice', 'altlando');
  await p2.close();
});

test('a stale cached model file is replaced, not used', async () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, 'altlando.json'), 'utf8'));
  const ok = await app.page.evaluate(async ({ base, sha }) => {
    // a wrong copy of the encoder already in the cache, under this exact version's key
    const url = new URL('encoder.onnx', new URL(base, location.href)).href;
    const cache = await caches.open('kinesphere-models');
    await cache.put(`${url}?v=${sha}`, new Response(new Uint8Array(1000)));
    const { createAltlando } = await import('/js/altlando.js');
    const model = createAltlando();
    await model.load({ base });
    model.terminate();
    const fixed = await cache.match(`${url}?v=${sha}`);
    return (await fixed.arrayBuffer()).byteLength;
  }, { base: `/${MODEL}`, sha: manifest.sha256.encoder });
  assert.equal(ok, manifest.bytes.encoder, 'the cache now holds the right file');
});

test('the AI partner button starts it: it watches for 3.7 s, then dances a reply beside you', async () => {
  const { page } = app;
  await startCameraWithDancer(page, swingingDancer);
  await page.click('#btn-ai');
  assert.ok(await page.isVisible('#ai-panel'));
  await waitStatus(page, /Watching you dance/);
  await waitStatus(page, /dancing a reply/, 60000);
  assert.match(await status(page), /random weights/, 'says when the model is only a test model');
  const partner = await page.evaluate(() => {
    const r = window.__state.mp.remotes.get('ai-partner');
    return r && { name: r.name, lane: r.lane, age: performance.now() - r.at, poses: r.poses.length };
  });
  assert.ok(partner, 'the partner is drawn as a dancer');
  assert.equal(partner.name, 'Altlando architecture', 'named after its model (the test model\'s manifest says "Altlando architecture (random weights, for testing)")');
  assert.ok(partner.age < 500 && partner.poses === 1);
  assert.ok(partner.lane > 0 && partner.lane < 1, `lane ${partner.lane}`);
});

test('a dancer partly out of frame, with the partner on: no drawing errors (the fitted-body styles used to throw)', async () => {
  const { page } = app;
  const before = app.errors.length;
  const warnings = [];
  const onConsole = m => { if (/failed on this pose/.test(m.text())) warnings.push(m.text()); };
  page.on('console', onConsole); // the app catches a style's error and warns, so check for that too
  // walks towards the camera and back: arms, legs and hips leave the picture, with flickering visibility
  await setDancer(page, function (t) {
    const lm = new Float32Array(132);
    const s = 1 + 1.6 * (0.5 - 0.5 * Math.cos(t * 1.5));
    const set = (k, x, y) => { const X = 0.5 + (x - 0.5) * s, Y = 0.25 + (y - 0.25) * s; lm[k * 4] = X; lm[k * 4 + 1] = Y; lm[k * 4 + 3] = X < 0 || X > 1 || Y < 0 || Y > 1 ? 0.25 + 0.3 * Math.abs(Math.sin(t * 7 + k)) : 0.97; };
    const a = 1.4 + 1.2 * Math.sin(t * 2.4);
    set(0, 0.5, 0.2); for (let k = 1; k <= 10; k++) set(k, 0.5 + (k % 2 ? 0.02 : -0.02), 0.19);
    set(11, 0.56, 0.3); set(12, 0.44, 0.3); set(23, 0.545, 0.53); set(24, 0.455, 0.53);
    set(13, 0.56 + 0.09 * Math.sin(a), 0.3 + 0.09 * Math.cos(a)); set(15, 0.56 + 0.17 * Math.sin(a), 0.3 + 0.17 * Math.cos(a));
    set(14, 0.44 - 0.09 * Math.sin(a), 0.3 + 0.09 * Math.cos(a)); set(16, 0.44 - 0.17 * Math.sin(a), 0.3 + 0.17 * Math.cos(a));
    set(25, 0.55, 0.7); set(27, 0.55, 0.87); set(26, 0.45, 0.7); set(28, 0.45, 0.87);
    return lm;
  });
  // a slow computer (the bug needed under ~50 frames a second): slow the page's CPU down
  const cdp = await page.context().newCDPSession(page);
  await cdp.send('Emulation.setCPUThrottlingRate', { rate: 6 });
  const fps = await page.evaluate(() => new Promise(r => { let n = 0; const t0 = performance.now(); const f = () => (++n, performance.now() - t0 < 1000 ? requestAnimationFrame(f) : r(n)); requestAnimationFrame(f); }));
  try {
    for (const style of ['pleats', 'animator']) {
      await page.selectOption('#sel-style', style);
      await page.waitForTimeout(5000);
    }
  } finally {
    await cdp.send('Emulation.setCPUThrottlingRate', { rate: 1 });
    await cdp.detach();
  }
  console.log(`# frame rate when throttling started: ${fps} a second`);
  await setDancer(page, swingingDancer);
  page.off('console', onConsole);
  assert.deepEqual(app.errors.slice(before), []);
  assert.deepEqual(warnings, []);
  assert.ok(await page.evaluate(() => performance.now() - window.__state.lastPeopleAt < 1000), 'the live view is still running');
});

test('the AI partner is recorded with you, and labelled in the dashboard', async () => {
  const { page } = app;
  await page.click('#btn-record');
  await page.waitForFunction(() => window.__state.recording, null, { timeout: 10000 });
  await page.waitForTimeout(2000);
  await page.click('#btn-stop');
  await page.waitForSelector('#view-dashboard:not([hidden]) #session-name', { timeout: 30000 });
  const tabs = await page.$$eval('#view-dashboard .tab[data-person]', t => t.map(x => x.textContent.trim()));
  assert.equal(tabs.length, 2, tabs.join(' | '));
  assert.ok(tabs.some(t => t.startsWith('Altlando architecture') && !t.includes('online')), tabs.join(' | '));
  await page.click('.nav-btn[data-view="live"]');
});

test('stop removes the partner; starting again reuses the loaded model', async () => {
  const { page } = app;
  await page.click('#ai-toggle');
  assert.equal(await status(page), 'Off.');
  await page.waitForTimeout(200);
  assert.equal(await page.evaluate(() => window.__state.mp.remotes.has('ai-partner')), false);
  await page.click('#ai-toggle');
  await waitStatus(page, /Watching you dance|thinking|dancing/, 5000);
  await page.click('#ai-toggle');
});

test('model files chosen on this computer work too', async () => {
  const { page } = app;
  const manifest = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, 'altlando.json'), 'utf8'));
  const names = ['altlando.json', ...Object.values(manifest.files).flat()];
  await page.click('.ai-advanced summary');
  await page.setInputFiles('#ai-files', names.map(n => path.join(MODEL_DIR, n)));
  await waitStatus(page, /Watching you dance|thinking|dancing/);
  assert.match(await page.inputValue('#ai-model'), /files chosen/);
  await page.click('#ai-toggle');
});

test('a folder without a model says so', async () => {
  const { page } = app;
  await page.fill('#ai-model', 'models/nothing-here/');
  await page.click('#ai-toggle');
  await waitStatus(page, /No model found/);
  assert.match(await status(page), /export_onnx\.py/);
  assert.equal(await page.textContent('#ai-toggle'), 'Start');
});

test('a dancer exports as Altlando input (60 fps SMPL-24 frames)', async () => {
  const { page } = app;
  await importSession(page, serialize(phasesSession(1)));
  const [dl] = await Promise.all([page.waitForEvent('download'), page.click('[data-action="export-smpl"]')]);
  assert.match(dl.suggestedFilename(), /\.altlando\.json$/);
  const chunks = [];
  for await (const c of await dl.createReadStream()) chunks.push(c);
  const seq = JSON.parse(Buffer.concat(chunks).toString('utf8'));
  assert.ok(Math.abs(seq.length - 100 * 60) < 10, `${seq.length} frames for 100 s`);
  assert.ok(seq.every(f => f.length === 72));
});

test('no page errors', () => assert.deepEqual(app.errors.filter(e => !/nothing-here/.test(e)), []));
