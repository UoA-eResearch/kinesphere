// The dashboard for a recorded session: panels, kinesphere (near / medium / far), speed, floor
// path, replay, export, storage and comparing dancers.
import { test, describe, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { openApp, importSession, seek, hasOverflow } from '../helpers/browser.mjs';
import { phasesSession, floorSession, serialize } from '../helpers/synth.mjs';

const PHASES_TEXT = serialize(phasesSession(1));
const DUET_TEXT = serialize(phasesSession(2));
const pct = s => Number(String(s).replace('%', ''));
let app;
before(async () => { app = await openApp({ viewport: { width: 1280, height: 900 } }); });
after(async () => { await app?.close(); });

describe('single dancer', () => {
  before(async () => { await importSession(app.page, PHASES_TEXT); });

  test('shows the analysis panels', async () => {
    const titles = await app.page.$$eval('#view-dashboard .card h2', h => h.map(x => x.textContent.trim()));
    assert.ok(titles.length >= 5, `panels: ${titles.join(', ')}`);
    assert.equal(await app.page.$eval('#session-name', e => e.value), 'Synthetic phases');
  });

  test('kinesphere figures are Near, Medium and Far with shares matching the phases', async () => {
    const figs = await app.page.$$eval('.kine-fig', els => els.map(e => ({
      name: e.querySelector('.kine-fig-name').textContent.trim(),
      share: e.querySelector('.kine-fig-share').textContent.trim(),
      rings: e.querySelectorAll('.viz-ring').length,
    })));
    assert.deepEqual(figs.map(f => f.name), ['Near', 'Medium', 'Far']);
    for (const f of figs) assert.ok(f.rings >= 3, `${f.name} draws the three spheres`);
    // ground truth (3D): 4 phases near, 1 medium, 5 far, of 10 equal phases
    const [near, medium, far] = figs.map(f => pct(f.share));
    assert.ok(near > 25 && near < 55, `near ${near}%`);
    assert.ok(medium > 3 && medium < 25, `medium ${medium}%`);
    assert.ok(far > 35 && far < 65, `far ${far}%`);
    assert.ok(Math.abs(near + medium + far - 100) <= 2);
  });

  test('upper / lower and left / right body figures are drawn', async () => {
    for (const sel of ['.viz-side-figure', '.viz-updown-figure', '.viz-levels-figure'])
      assert.ok(await app.page.$eval(sel, s => s.getAttribute('aria-label')), `${sel} has a description`);
  });

  test('"Show values" adds numbers to the body figure', async () => {
    const { page } = app;
    assert.equal(await page.$$eval('#body-figure .viz-val', e => e.length), 0);
    await page.check('#show-values');
    const vals = await page.$$eval('#body-figure .viz-val', e => e.map(t => t.textContent));
    assert.ok(vals.length >= 12 && vals.every(v => /%$/.test(v)), vals.join(' '));
  });

  test('explanations can be hidden, and stay hidden', async () => {
    const { page } = app;
    const visible = () => page.$$eval('#view-dashboard .card-sub', e => e.filter(x => x.offsetParent).length);
    assert.ok(await visible() > 0);
    await page.uncheck('#show-help');
    assert.equal(await visible(), 0);
    await importSession(page, PHASES_TEXT);
    assert.equal(await visible(), 0);
    await page.check('#show-help');
  });

  test('the replay overlay draws the kinesphere and is remembered', async () => {
    const { page } = app;
    await page.check('#replay-overlay');
    await seek(page, 35000);
    assert.equal(await page.evaluate(() => localStorage.getItem('kinesphere:replayOverlay')), 'on');
    const inked = await page.$eval('#replay-canvas', c => { const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 3; i < d.length; i += 40) if (d[i]) n++; return n; });
    assert.ok(inked > 100);
  });

  test('exports JSON identical to the import, and CSV with world columns', async () => {
    const { page } = app;
    const [dl] = await Promise.all([page.waitForEvent('download'), page.click('[data-action="export-json"]')]);
    const out = JSON.parse(await streamText(await dl.createReadStream()));
    const src = JSON.parse(PHASES_TEXT);
    assert.equal(out.positions, src.positions);
    assert.equal(out.worldPositions, src.worldPositions);
    const [dl2] = await Promise.all([page.waitForEvent('download'), page.click('[data-action="export-csv"]')]);
    const head = (await streamText(await dl2.createReadStream())).split('\n')[0].split(',');
    assert.ok(head.filter(h => h.includes('_world_')).length > 0);
  });
});

describe('floor path and speed', () => {
  before(async () => { await importSession(app.page, serialize(floorSession(60))); });

  test('the bird\'s-eye floor plot draws the path', async () => {
    assert.ok(await app.page.$$eval('#space-plot svg line', l => l.length) > 100, 'path segments');
    assert.equal(await app.page.$$eval('#space-plot .viz-start, #space-plot .viz-end', c => c.length), 2, 'start and end markers');
  });

  test('speed values are hidden until "Show values" is ticked', async () => {
    const { page } = app;
    const ticks = () => page.$$eval('#speed-chart .viz-tick', t => t.map(x => x.textContent).join(' '));
    assert.doesNotMatch(await ticks(), /\d\.\d/);
    await page.check('#show-speed-values');
    assert.match(await ticks(), /\d/);
  });
});

describe('storage', () => {
  test('sessions are kept in IndexedDB across reloads, and can be deleted', async () => {
    const { page } = app;
    await page.reload({ waitUntil: 'networkidle' });
    await page.click('.nav-btn[data-view="sessions"]');
    await page.waitForSelector('.session-item');
    const names = await page.$$eval('.session-item h2', h => h.map(x => x.textContent.trim()));
    assert.ok(names.some(n => n.includes('Floor test')) && names.some(n => n.includes('Synthetic phases')), names.join(', '));
    assert.equal(await page.evaluate(() => Object.keys(localStorage).filter(k => k.startsWith('kinesphere:session:')).length), 0, 'no session payloads in localStorage');
    page.once('dialog', d => d.accept());
    const before = names.length;
    await page.click('.session-item:first-child [data-action="delete"]');
    await page.waitForFunction(n => document.querySelectorAll('.session-item').length === n - 1, before);
  });

  test('a 15-minute session imports and saves', async () => {
    const { page } = app;
    await importSession(page, serialize(floorSession(900, { glitches: false })));
    const banners = await page.$$eval('#view-dashboard .banner', b => b.map(x => x.textContent.trim()));
    assert.ok(!banners.some(b => /could not|full|quota/i.test(b)), banners.join(' | '));
    await page.click('.nav-btn[data-view="sessions"]');
    await page.waitForSelector('.session-item');
    assert.ok((await page.$$eval('.session-item h2', h => h.map(x => x.textContent))).some(n => n.includes('900s')));
  });
});

describe('comparing dancers', () => {
  test('choosing several people switches to MoveNet MultiPose', async () => {
    const { page } = app;
    await page.click('.nav-btn[data-view="live"]');
    await page.selectOption('#sel-model', 'lite');
    await page.selectOption('#sel-people', '2');
    assert.equal(await page.$eval('#sel-model', s => s.value), 'movenet-multipose');
    await page.selectOption('#sel-people', '1');
  });

  test('two people in one session side by side', async () => {
    const { page } = app;
    await importSession(page, DUET_TEXT);
    await page.click('.tab[data-compare]');
    await page.waitForSelector('.cmp-table', { timeout: 20000 });
    const rows = await page.$$eval('.cmp-table tbody tr td:first-child', t => t.map(x => x.textContent.trim()));
    assert.ok(rows.length >= 6, rows.join(' | '));
    assert.ok(await page.$$eval('#compare-body svg', s => s.length) >= 4);
    assert.equal(await page.$$eval('[id^=cmp-updown] svg', s => s.length), 2);
  });

  test('a dancer from another session', async () => {
    const { page } = app;
    const other = (await page.$$eval('#cmp-b option', o => o.map(x => [x.value, x.textContent]))).find(([, t]) => t.includes('Synthetic phases'));
    assert.ok(other, 'the earlier session is offered');
    await page.selectOption('#cmp-b', other[0]);
    await page.waitForFunction(() => document.querySelector('#cmp-b')?.selectedOptions[0]?.textContent.includes('phases') && document.querySelector('.cmp-table'), null, { timeout: 20000 });
  });

  test('fits a phone screen', async () => {
    const { page } = app;
    await page.setViewportSize({ width: 390, height: 844 });
    await page.waitForTimeout(300);
    assert.equal(await hasOverflow(page), false);
    await page.click('.tab[data-person="0"]');
    await page.waitForTimeout(300);
    assert.equal(await hasOverflow(page), false);
    assert.deepEqual(app.errors, []);
  });
});

async function streamText(stream) {
  const chunks = [];
  for await (const c of stream) chunks.push(c);
  return Buffer.concat(chunks).toString('utf8');
}
