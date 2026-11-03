// The help panel: layout (closed and open, desktop and phone) and the overlay-styles gallery.
import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { openApp, hasOverflow } from '../helpers/browser.mjs';

let app;
before(async () => { app = await openApp({ viewport: { width: 1200, height: 900 } }); });
after(async () => { await app?.close(); });

const box = sel => app.page.$eval(sel, el => { const r = el.getBoundingClientRect(); return { w: r.width, h: r.height }; });

for (const [name, width] of [['desktop', 1200], ['phone', 390]]) {
  test(`closed help panel is a readable one-line link (${name})`, async () => {
    const { page } = app;
    await page.setViewportSize({ width, height: 900 });
    await page.evaluate(() => { document.querySelector('#help-panel').open = false; });
    const panel = await box('#help-panel'), summary = await box('#help-panel summary');
    assert.ok(panel.w > Math.min(300, width - 60), `panel width ${panel.w}`);
    assert.ok(summary.h < 60, `summary height ${summary.h} (the text is wrapping one letter per line)`);
    assert.equal(await hasOverflow(page), false);
  });
}

test('open help panel fills its column', async () => {
  const { page } = app;
  await page.setViewportSize({ width: 1200, height: 900 });
  await page.click('#btn-help');
  await page.waitForSelector('#help-panel[open]');
  assert.ok((await box('#help-panel')).w > 600);
});

test('the gallery shows every overlay style with a drawn preview', async () => {
  const { page } = app;
  const ids = await page.evaluate(async () => (await import('/js/effects.js')).STYLES.map(s => s.id));
  await page.waitForFunction(n => document.querySelectorAll('.style-card canvas').length === n, ids.length, { timeout: 10000 });
  await page.waitForTimeout(1500); // previews are drawn after the panel opens
  assert.deepEqual(await page.$$eval('.style-card', c => c.map(x => x.dataset.style)), ids);
  const blank = await page.$$eval('.style-card canvas', cs => cs.filter(c => {
    const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data;
    for (let i = 3; i < d.length; i += 16) if (d[i]) return false;
    return true;
  }).map(c => c.closest('.style-card').dataset.style));
  assert.deepEqual(blank, []);
  for (const card of await page.$$('.style-card')) assert.ok((await card.textContent()).length > 80, 'card describes the style');
});

test('clicking a card switches to that style', async () => {
  const { page } = app;
  await page.click('.style-card[data-style="neon"]');
  assert.equal(await page.$eval('#sel-style', s => s.value), 'neon');
  assert.equal(await page.$eval('.style-card.is-current', c => c.dataset.style), 'neon');
});

test('no horizontal scroll with the panel open on a phone', async () => {
  const { page } = app;
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(300);
  assert.equal(await hasOverflow(page), false);
  assert.deepEqual(app.errors, []);
});
