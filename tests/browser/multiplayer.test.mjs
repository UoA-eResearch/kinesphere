// Dancing together online: a host and guests in separate browser contexts, connected through a
// local PeerJS signalling server (?signal=...), each with a fake camera and a scripted dancer.
import { test as baseTest, before, after } from 'node:test';
const test = (name, fn) => baseTest(name, { timeout: 120000 }, fn);
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { launch, newPage, gotoApp, startCameraWithDancer, CAMERA_ARGS, ORIGIN, ROOT } from '../helpers/browser.mjs';

const PORT = 9317;
const SIGNAL = `?signal=ws://127.0.0.1:${PORT}/ks`;
let server, browser;
const errors = [];

/** A dancer standing at x (0..1 of their frame), `size` torso lengths of frame height, arms raised by `arms` radians. */
function remoteDancer(t, { x, size, arms }) {
  const lm = new Float32Array(132);
  const set = (k, px, py) => { lm[k * 4] = px; lm[k * 4 + 1] = py; lm[k * 4 + 3] = 0.95; };
  for (let k = 0; k < 33; k++) set(k, 0.5, 0.2);
  const s = size, cx = x + 0.03 * Math.sin(t * 2), a = arms + Math.sin(t * 3) * 0.3;
  set(0, cx, 0.5 - 1.45 * s); set(7, cx + 0.03, 0.5 - 1.5 * s); set(8, cx - 0.03, 0.5 - 1.5 * s);
  set(11, cx + 0.25 * s, 0.5 - s); set(12, cx - 0.25 * s, 0.5 - s); set(23, cx + 0.15 * s, 0.5); set(24, cx - 0.15 * s, 0.5);
  set(25, cx + 0.16 * s, 0.5 + 0.9 * s); set(26, cx - 0.16 * s, 0.5 + 0.9 * s); set(27, cx + 0.16 * s, 0.5 + 1.75 * s); set(28, cx - 0.16 * s, 0.5 + 1.75 * s);
  set(13, cx + 0.25 * s + 0.6 * s * Math.sin(a), 0.5 - s + 0.6 * s * Math.cos(a)); set(15, cx + 0.25 * s + 1.15 * s * Math.sin(a), 0.5 - s + 1.15 * s * Math.cos(a));
  set(14, cx - 0.25 * s - 0.6 * s * Math.sin(a), 0.5 - s + 0.6 * s * Math.cos(a)); set(16, cx - 0.25 * s - 1.15 * s * Math.sin(a), 0.5 - s + 1.15 * s * Math.cos(a));
  return lm;
}

async function dancer(label) {
  return newPage(browser, { label, viewport: { width: 1100, height: 900 }, permissions: ['camera', 'clipboard-read', 'clipboard-write'], errors });
}
const remotes = page => page.evaluate(() => [...window.__state.mp.remotes.values()].map(r => r.name).sort());

before(async () => {
  // the signalling server runs in its own process (its timers would keep this one alive)
  const src = `const { PeerServer } = require('peer'); PeerServer({ port: ${PORT}, path: '/ks', host: '127.0.0.1' }, () => console.log('ready'));`;
  server = spawn(process.execPath, ['-e', src], { cwd: ROOT, stdio: ['ignore', 'pipe', 'inherit'] });
  await new Promise((resolve, reject) => {
    server.stdout.on('data', d => { if (String(d).includes('ready')) resolve(); });
    server.on('exit', code => reject(new Error(`signalling server exited (${code})`)));
  });
  // no proxy: everything this test needs is local (the app, the CDN cache or direct CDN access, the signalling server)
  browser = await launch([...CAMERA_ARGS, '--disable-features=WebRtcHideLocalIpsWithMdns'], { proxy: false });
});
after(async () => {
  await browser?.close();
  server?.kill();
});

let A, B, C, link;
test('the host gets an invite link', async () => {
  A = await dancer('host');
  await gotoApp(A, SIGNAL);
  await startCameraWithDancer(A, remoteDancer, { x: 0.5, size: 0.2, arms: 2.6 });
  await A.click('#btn-mp');
  await A.fill('#mp-name', 'Ana'); await A.dispatchEvent('#mp-name', 'change');
  await A.click('[data-mp="host"]');
  await A.waitForSelector('#mp-link', { timeout: 30000 });
  link = await A.$eval('#mp-link', i => i.value);
  assert.match(link, new RegExp(`^${ORIGIN.replace(/\./g, '\\.')}/\\?signal=.*#join=[a-z0-9]{10}$`));
});

test('a guest opening the link joins and each sees the other\'s avatar', async () => {
  B = await dancer('guest 1');
  await B.goto(link, { waitUntil: 'networkidle' });
  assert.ok(await B.isVisible('#join-name'), 'the guest is asked for a name');
  await B.fill('#join-name', 'Ben');
  await startCameraWithDancer(B, remoteDancer, { x: 0.45, size: 0.12, arms: 1.6 });
  await A.waitForFunction(() => window.__state.mp.remotes.size === 1, null, { timeout: 45000 });
  await B.waitForFunction(() => window.__state.mp.remotes.size === 1, null, { timeout: 45000 });
  assert.deepEqual(await remotes(A), ['Ben']);
  assert.deepEqual(await remotes(B), ['Ana']);
  // poses keep arriving
  await A.waitForTimeout(1000);
  const age = await A.evaluate(() => Math.min(...[...window.__state.mp.remotes.values()].map(r => performance.now() - r.at)));
  assert.ok(age < 1000, `last remote pose ${age} ms ago`);
});

test('a second guest sees both others (the host relays poses)', async () => {
  C = await dancer('guest 2');
  await C.goto(link, { waitUntil: 'networkidle' });
  await C.fill('#join-name', 'Cleo');
  await startCameraWithDancer(C, remoteDancer, { x: 0.6, size: 0.2, arms: 0.4 });
  await C.waitForFunction(() => window.__state.mp.remotes.size === 2, null, { timeout: 45000 });
  await B.waitForFunction(() => window.__state.mp.remotes.size === 2, null, { timeout: 45000 });
  assert.deepEqual(await remotes(C), ['Ana', 'Ben']);
  assert.deepEqual(await remotes(B), ['Ana', 'Cleo']);
});

test('remote dancers are recorded into the session', async () => {
  await A.click('#btn-record');
  await A.waitForFunction(() => window.__state.recording, null, { timeout: 10000 });
  await A.waitForTimeout(2000);
  await A.click('#btn-stop');
  await A.waitForSelector('#view-dashboard:not([hidden]) #session-name', { timeout: 30000 });
  const tabs = await A.$$eval('#view-dashboard .tab[data-person]', t => t.map(x => x.textContent.trim()));
  assert.equal(tabs.length, 3, tabs.join(' | '));
  assert.ok(tabs.some(t => t.includes('Ben')) && tabs.some(t => t.includes('Cleo')), tabs.join(' | '));
  await A.click('.nav-btn[data-view="live"]');
});

test('a guest leaving disappears for the others', async () => {
  await B.click('[data-mp="leave"]');
  await A.waitForFunction(() => window.__state.mp.remotes.size === 1, null, { timeout: 20000 });
  assert.deepEqual(await remotes(A), ['Cleo']);
  assert.equal(await B.evaluate(() => location.hash), '');
});

test('the host ending the session ends it for guests', async () => {
  await A.click('[data-mp="leave"]');
  await C.waitForFunction(() => window.__state.mp.status === 'ended', null, { timeout: 20000 });
  assert.equal(await C.evaluate(() => window.__state.mp.remotes.size), 0);
});

test('a link to a room that does not exist says so', async () => {
  const D = await dancer('bad link');
  await D.goto(`${ORIGIN}/${SIGNAL}#join=zzzzzzzzzz`, { waitUntil: 'networkidle' });
  await startCameraWithDancer(D, remoteDancer, { x: 0.5, size: 0.2, arms: 1 });
  await D.waitForFunction(() => window.__state.mp.status === 'ended', null, { timeout: 45000 });
  assert.ok((await D.textContent('#toast')).trim().length > 0);
});

test('no page errors', () => assert.deepEqual(errors, []));
