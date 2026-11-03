// The AI dance partner end to end, with the model running in the browser: a small random-weight
// export of the real Altlando architecture (npm run build:test-model -> test-results/altlando-tiny),
// a fake camera and a scripted dancer.
import { test as baseTest, before, after } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { openApp, startCameraWithDancer, importSession, CAMERA_ARGS, ROOT } from '../helpers/browser.mjs';
import { phasesSession, serialize } from '../helpers/synth.mjs';

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

test('the AI partner button starts it: it watches for 3.7 s, then dances a reply beside you', async () => {
  const { page } = app;
  await startCameraWithDancer(page, swingingDancer);
  await page.click('#btn-ai');
  assert.ok(await page.isVisible('#ai-panel'));
  await waitStatus(page, /Watching you dance/);
  await waitStatus(page, /Dancing a reply/, 60000);
  assert.match(await status(page), /random weights/, 'says when the model is only a test model');
  const partner = await page.evaluate(() => {
    const r = window.__state.mp.remotes.get('ai-partner');
    return r && { name: r.name, lane: r.lane, age: performance.now() - r.at, poses: r.poses.length };
  });
  assert.ok(partner, 'the partner is drawn as a dancer');
  assert.equal(partner.name, 'AI partner');
  assert.ok(partner.age < 500 && partner.poses === 1);
  assert.ok(partner.lane > 0 && partner.lane < 1, `lane ${partner.lane}`);
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
  assert.ok(tabs.some(t => t.startsWith('AI partner') && !t.includes('online')), tabs.join(' | '));
  await page.click('.nav-btn[data-view="live"]');
});

test('stop removes the partner; starting again reuses the loaded model', async () => {
  const { page } = app;
  await page.click('#ai-toggle');
  assert.equal(await status(page), 'Off.');
  await page.waitForTimeout(200);
  assert.equal(await page.evaluate(() => window.__state.mp.remotes.has('ai-partner')), false);
  await page.click('#ai-toggle');
  await waitStatus(page, /Watching you dance|Thinking|Dancing/, 5000);
  await page.click('#ai-toggle');
});

test('model files chosen on this computer work too', async () => {
  const { page } = app;
  const manifest = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, 'altlando.json'), 'utf8'));
  const names = ['altlando.json', ...Object.values(manifest.files).flat()];
  await page.click('.ai-advanced summary');
  await page.setInputFiles('#ai-files', names.map(n => path.join(MODEL_DIR, n)));
  await waitStatus(page, /Watching you dance|Thinking|Dancing/);
  assert.match(await page.inputValue('#ai-model'), /files chosen/);
  await page.click('#ai-toggle');
});

test('without a model on the site it says how to add one', async () => {
  const { page } = app;
  await page.fill('#ai-model', 'models/nothing-here/');
  await page.click('#ai-toggle');
  await waitStatus(page, /isn’t on this site yet/);
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
