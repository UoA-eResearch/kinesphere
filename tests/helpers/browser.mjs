// Playwright helpers: serve the app from the working tree on a fake https origin (so camera,
// IndexedDB and module imports behave as on GitHub Pages) and collect page errors.
//
// Environment:
//   KINESPHERE_CDN_CACHE  a directory holding copies of CDN files (<dir>/<host>/<path>); when set,
//                         jsDelivr / Google Storage requests are served from it instead of the network
//   HTTPS_PROXY           used by the browser for any other outbound request
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

export const ROOT = path.resolve(fileURLToPath(new URL('../../', import.meta.url)));
export const ORIGIN = 'https://kinesphere.test';
const CDN_CACHE = process.env.KINESPHERE_CDN_CACHE;
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm', '.svg': 'image/svg+xml', '.png': 'image/png' };
// Test hooks appended to app.js so tests can reach its internal state.
const HOOKS = '\nwindow.__state = state; window.__mp = mp; window.__game = game;\n';
export const CAMERA_ARGS = ['--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--autoplay-policy=no-user-gesture-required'];
// Console noise that isn't an app error (TF.js backend probing, aborted model downloads).
const IGNORED = /tf-core|tfjs|Failed to load resource|WebGL|GPU stall/;

export async function launch(args = [], { proxy = true } = {}) {
  const opts = { args };
  if (proxy && process.env.HTTPS_PROXY) opts.proxy = { server: process.env.HTTPS_PROXY, bypass: 'localhost,127.0.0.1' };
  return chromium.launch(opts);
}

/** A new page in its own context, serving the app. `patch(file, text)` may rewrite served JS files. */
export async function newPage(browser, { viewport = { width: 1200, height: 900 }, permissions = [], label = '', patch = null, errors = [] } = {}) {
  const ctx = await browser.newContext({ viewport, ignoreHTTPSErrors: true, acceptDownloads: true, permissions });
  const page = await ctx.newPage();
  const tag = label ? `[${label}] ` : '';
  page.on('console', m => { if (m.type() === 'error' && !IGNORED.test(m.text())) errors.push(`${tag}[console] ${m.text().split('\n')[0]}`); });
  page.on('pageerror', e => errors.push(`${tag}[pageerror] ${e.message}`));
  // routes on the context, so requests from Web Workers (the AI partner model) are served too
  await ctx.route(`${ORIGIN}/**`, route => {
    let p = decodeURIComponent(new URL(route.request().url()).pathname);
    if (p === '/') p = '/index.html';
    const f = path.join(ROOT, p);
    if (!f.startsWith(ROOT + path.sep) || p.includes('/node_modules/') || !fs.existsSync(f) || fs.statSync(f).isDirectory()) return route.fulfill({ status: 404, body: '' });
    let body = fs.readFileSync(f);
    if (p === '/js/app.js') body = body.toString() + HOOKS;
    if (patch && p.endsWith('.js')) body = patch(p, body.toString()) ?? body;
    return route.fulfill({ status: 200, contentType: MIME[path.extname(f)] || 'application/octet-stream', body });
  });
  if (CDN_CACHE) {
    await ctx.route(/cdn\.jsdelivr\.net|storage\.googleapis\.com|tfhub\.dev|kaggle\.com/, route => {
      const u = new URL(route.request().url());
      const f = path.join(CDN_CACHE, u.hostname, u.pathname);
      if (!fs.existsSync(f)) { errors.push(`[cdn-miss] ${u.href}`); return route.fulfill({ status: 404, body: '' }); }
      return route.fulfill({ status: 200, contentType: MIME[path.extname(f)] || 'application/octet-stream', body: fs.readFileSync(f), headers: { 'access-control-allow-origin': '*' } });
    });
  }
  return page;
}

/** Launches a browser and opens the app in one page. Call `close()` when done. */
export async function openApp({ viewport, args = [], permissions = [], patch = null, query = '', proxy = true } = {}) {
  const errors = [];
  const browser = await launch(args, { proxy });
  const page = await newPage(browser, { viewport, permissions, patch, errors });
  await gotoApp(page, query);
  return { browser, page, errors, close: () => browser.close() };
}

export async function gotoApp(page, query = '') {
  await page.goto(`${ORIGIN}/${query}`, { waitUntil: 'networkidle' });
  // keep the sticky header from covering elements Playwright clicks
  await page.addStyleTag({ content: '.topbar { position: static !important; }' });
}

/** Imports a session (an object from tests/helpers/synth.mjs, or JSON text) through the file picker. */
export async function importSession(page, text, name = 'test.kinesphere.json') {
  // mark the current dashboard, so we can wait for the new session's dashboard to replace it
  await page.evaluate(() => { const n = document.querySelector('#session-name'); if (n) n.dataset.stale = '1'; });
  await page.setInputFiles('#file-import', { name, mimeType: 'application/json', buffer: Buffer.from(text) });
  await page.waitForSelector('#view-dashboard:not([hidden]) #session-name:not([data-stale])', { timeout: 60000 });
  await page.waitForTimeout(300);
}

export async function seek(page, ms) {
  await page.$eval('#replay-range', (el, v) => { el.value = String(v); el.dispatchEvent(new Event('input')); }, ms);
  await page.waitForTimeout(120);
}

export const hasOverflow = page => page.evaluate(() => document.documentElement.scrollWidth > innerWidth);

/**
 * Starts the camera (Chromium's fake device) and waits for the pose model, then replaces the
 * detector with a scripted dancer: `poseFn` runs in the page as (t seconds) => Float32Array(132)
 * (MediaPipe's 33 landmarks as x, y, z, visibility).
 */
export async function startCameraWithDancer(page, poseFn, arg = null) {
  await page.click('#btn-camera');
  await page.waitForFunction(() => window.__state?.detector, null, { timeout: 120000 });
  if (await page.isChecked('#chk-gesture').catch(() => false)) await page.uncheck('#chk-gesture');
  await setDancer(page, poseFn, arg);
}

export async function setDancer(page, poseFn, arg = null) {
  await page.evaluate(([src, arg]) => {
    const fn = new Function(`return (${src})`)();
    window.__state.detector.detect = () => [{ lm: fn(performance.now() / 1000, arg) }];
  }, [poseFn.toString(), arg]);
}

/** Waits until the live loop has drawn a fresh detection. */
export const freshPose = page => page.waitForFunction(() => performance.now() - window.__state.lastPeopleAt < 150, null, { timeout: 20000, polling: 20 });

/** A standing dancer, arms bent, one knee raised, gently moving (MoveNet's 17 keypoints only). */
export function dancerPose(t) {
  const lm = new Float32Array(132);
  const set = (k, x, y) => { lm[k * 4] = x; lm[k * 4 + 1] = y; lm[k * 4 + 3] = 0.9; };
  const sway = 0.02 * Math.sin(t * 2.5);
  set(0, 0.5 + sway, 0.2); set(2, 0.51 + sway, 0.19); set(5, 0.49 + sway, 0.19); set(7, 0.525 + sway, 0.2); set(8, 0.475 + sway, 0.2);
  set(11, 0.56, 0.3); set(12, 0.44, 0.3); set(23, 0.545, 0.53); set(24, 0.455, 0.53);
  set(13, 0.64, 0.33); set(15, 0.66, 0.22 + 0.02 * Math.sin(t * 3));
  set(14, 0.37, 0.36); set(16, 0.31 + 0.1 * Math.sin(t * 4), 0.44 - 0.08 * Math.cos(t * 4));
  set(25, 0.6, 0.62); set(27, 0.58, 0.72);
  set(26, 0.45, 0.7); set(28, 0.45, 0.86);
  return lm;
}
