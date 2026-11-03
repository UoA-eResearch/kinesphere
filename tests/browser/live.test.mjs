// The live view with a camera: Chromium's fake camera plus a scripted dancer in place of the pose
// model's output. Loads the real MediaPipe model first (from the CDN, or KINESPHERE_CDN_CACHE).
import { test, describe, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { openApp, startCameraWithDancer, setDancer, freshPose, dancerPose, CAMERA_ARGS } from '../helpers/browser.mjs';

let app;
before(async () => {
  app = await openApp({ viewport: { width: 1200, height: 900 }, permissions: ['camera'], args: CAMERA_ARGS });
  // a fake speech recogniser: tests "speak" by firing results on the live instance
  await app.page.addInitScript(() => {
    class FakeRec {
      constructor() { window.__rec = this; this.started = false; }
      start() { this.started = true; setTimeout(() => this.onstart?.(), 10); }
      stop() { this.started = false; setTimeout(() => this.onend?.(), 10); }
    }
    window.webkitSpeechRecognition = FakeRec; window.SpeechRecognition = FakeRec;
    window.__say = (...alts) => { const res = alts.map(t => ({ transcript: t })); res.isFinal = true; window.__rec.onresult({ resultIndex: 0, results: [res] }); };
  });
  await app.page.reload({ waitUntil: 'networkidle' });
  await app.page.addStyleTag({ content: '.topbar { position: static !important; }' });
  await startCameraWithDancer(app.page, dancerPose);
});
after(async () => { await app?.close(); });

const inked = page => page.$eval('#overlay', c => {
  const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
  let n = 0; for (let i = 3; i < d.length; i += 40) if (d[i]) n++;
  return n / (d.length / 40);
});

describe('overlay styles on the live camera', () => {
  for (const style of ['skeleton', 'neon', 'sandman', 'origami', 'pencil', 'hoops', 'animator', 'pleats']) {
    test(style, async () => {
      const { page } = app;
      await page.selectOption('#sel-style', style);
      await page.waitForTimeout(400); await freshPose(page);
      assert.ok(await inked(page) > 0.002);
    });
  }
  test('with the camera feed hidden', async () => {
    const { page } = app;
    await page.click('#btn-video');
    await page.waitForTimeout(300); await freshPose(page);
    assert.ok(await page.$eval('#stage', s => s.classList.contains('no-video')));
    assert.ok(await inked(page) > 0.002);
    await page.click('#btn-video');
  });
});

describe('"Body partially out of frame" hint', () => {
  test('appears when the hips and legs leave the picture, and goes when they return', async () => {
    const { page } = app;
    await setDancer(page, t => {
      const lm = new Float32Array(132);
      for (let k = 0; k < 33; k++) { lm[k * 4] = 0.5 + (k % 2 ? 0.05 : -0.05); lm[k * 4 + 1] = 0.15 + k * 0.01; lm[k * 4 + 3] = 0.95; }
      for (const k of [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]) { lm[k * 4 + 1] = 1.1 + k * 0.01; lm[k * 4 + 3] = 0.2; }
      return lm;
    });
    await page.waitForSelector('#frame-warn:not([hidden])', { timeout: 5000 });
    await setDancer(page, dancerPose);
    await page.waitForSelector('#frame-warn', { state: 'hidden', timeout: 5000 });
  });
});

describe('recording', () => {
  test('records the dancer and opens the dashboard', async () => {
    const { page } = app;
    await page.click('#btn-record');
    await page.waitForFunction(() => window.__state.recording, null, { timeout: 10000 });
    await page.waitForTimeout(2500);
    await page.click('#btn-stop');
    await page.waitForSelector('#view-dashboard:not([hidden]) #session-name', { timeout: 30000 });
    assert.ok(await page.$$eval('#view-dashboard .card h2', h => h.length) >= 5);
    const frames = await page.evaluate(async () => {
      const id = JSON.parse(localStorage.getItem('kinesphere:sessions'))[0];
      return id?.frameCount ?? 0;
    });
    assert.ok(frames > 20, `${frames} frames recorded`);
    await page.click('.nav-btn[data-view="live"]');
  });
});

describe('voice control', () => {
  let dialogs = 0;
  const say = async (...alts) => { await app.page.evaluate(a => window.__say(...a), alts); await app.page.waitForTimeout(150); };
  before(async () => {
    app.page.on('dialog', d => { dialogs++; d.accept(); });
    await app.page.keyboard.press('KeyM');
    await app.page.waitForFunction(() => window.__rec?.started, null, { timeout: 5000 });
  });
  test('asks for consent before listening', () => assert.ok(dialogs >= 1));
  test('"neon" changes the style', async () => {
    await say('neon');
    assert.equal(await app.page.$eval('#sel-style', s => s.value), 'neon');
  });
  test('"blue sand" and "gravity up" change the sand sliders', async () => {
    const { page } = app;
    await say('blue sand');
    assert.equal(await page.$eval('#sel-style', s => s.value), 'sandman');
    assert.equal(await page.$eval('#sand-hue', s => s.value), '215');
    const g = Number(await page.$eval('#sand-gravity', s => s.value));
    await say('gravity up');
    assert.ok(Number(await page.$eval('#sand-gravity', s => s.value)) > g);
  });
  test('"hide video" and "mirror off"', async () => {
    const { page } = app;
    await say('hide video');
    assert.ok(await page.$eval('#stage', s => s.classList.contains('no-video')));
    await say('show video');
    await say('mirror off');
    assert.equal(await page.$eval('#chk-mirror', c => c.checked), false);
  });
  test('uses a later alternative when the first is not a command', async () => {
    const { page } = app;
    const before = await page.$eval('#sel-style', s => s.value);
    await say('next stile', 'next style');
    assert.notEqual(await page.$eval('#sel-style', s => s.value), before);
  });
  test('chatter is ignored', async () => {
    const { page } = app;
    const before = await page.$eval('#sel-style', s => s.value);
    await say('that was fun');
    assert.equal(await page.$eval('#sel-style', s => s.value), before);
  });
  test('"start recording" counts down and "cancel" stops it', async () => {
    const { page } = app;
    await say('start recording');
    assert.ok(await page.evaluate(() => Boolean(window.__state.countdown)));
    await say('cancel');
    assert.equal(await page.evaluate(() => Boolean(window.__state.countdown)), false);
  });
  test('restarts listening when the browser ends recognition, until told to stop', async () => {
    const { page } = app;
    await page.evaluate(() => { const r = window.__rec; r.started = false; r.onend(); });
    await page.waitForFunction(() => window.__rec.started, null, { timeout: 3000 });
    await say('stop listening');
    await page.waitForTimeout(200);
    assert.equal(await page.evaluate(() => window.__rec.started), false);
  });
  test('no page errors', () => assert.deepEqual(app.errors, []));
});
