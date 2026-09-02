// Small SVG/HTML chart primitives used by the dashboard (no external libraries).
// Colour roles follow one blue sequential ramp for magnitude/time and an ordinal
// subset of the same ramp for small/medium/large; text always uses ink tokens.

import { fmtClock, fmtPct, escapeHtml, clamp } from './util.js';
import { MAJOR_CONNECTIONS } from './analysis.js';

const NS = 'http://www.w3.org/2000/svg';

export function svg(tag, attrs = {}, children = []) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === 'style') el.style.cssText = v;
    else el.setAttribute(k, v);
  }
  for (const c of children) el.append(c);
  return el;
}

export function isDark() {
  const t = document.documentElement.dataset.theme;
  if (t === 'dark') return true;
  if (t === 'light') return false;
  return window.matchMedia('(prefers-color-scheme: dark)').matches;
}

// Blue sequential ramp, steps 100..700 (light -> dark).
const SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
  '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b'];

/** Map 0..1 to a ramp colour. Low values recede toward the surface in both themes. */
export function heat(v) {
  const idx = clamp(Math.round((Number.isFinite(v) ? v : 0) * (SEQ.length - 1)), 0, SEQ.length - 1);
  const i = isDark() ? SEQ.length - 1 - idx : idx;
  return { fill: SEQ[i], ink: i >= 6 ? '#ffffff' : '#0b0b0b' };
}

/** Ordinal colours for small / medium / large. */
export function ordinalColors() {
  return isDark()
    ? [{ fill: '#184f95', ink: '#fff' }, { fill: '#3987e5', ink: '#fff' }, { fill: '#86b6ef', ink: '#0b0b0b' }]
    : [{ fill: '#86b6ef', ink: '#0b0b0b' }, { fill: '#2a78d6', ink: '#fff' }, { fill: '#104281', ink: '#fff' }];
}

export function heatLegend(low = 'Less', high = 'More') {
  const el = document.createElement('div');
  el.className = 'viz-legend';
  const stops = Array.from({ length: 13 }, (_, i) => heat(i / 12).fill).join(',');
  el.innerHTML = `<span>${escapeHtml(low)}</span><span class="viz-legend-bar" style="background:linear-gradient(90deg,${stops})"></span><span>${escapeHtml(high)}</span>`;
  return el;
}

// ---- tooltip ------------------------------------------------------------------------

let tip = null;
function tipEl() {
  if (!tip) {
    tip = document.createElement('div');
    tip.className = 'viz-tip';
    tip.hidden = true;
    document.body.append(tip);
  }
  return tip;
}
export function showTip(x, y, html) {
  const t = tipEl();
  t.innerHTML = html;
  t.hidden = false;
  const r = t.getBoundingClientRect();
  let left = x + 14, top = y + 14;
  if (left + r.width > window.innerWidth - 8) left = x - r.width - 14;
  if (top + r.height > window.innerHeight - 8) top = y - r.height - 14;
  t.style.left = `${left}px`;
  t.style.top = `${top}px`;
}
export function hideTip() {
  if (tip) tip.hidden = true;
}

// ---- helpers ------------------------------------------------------------------------

function niceCeil(v) {
  if (!(v > 0)) return 1;
  const exp = Math.pow(10, Math.floor(Math.log10(v)));
  const f = v / exp;
  const nice = f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10;
  return nice * exp;
}

function pickInterval(tMaxMs) {
  const candidates = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1200, 1800];
  const s = tMaxMs / 1000;
  return (candidates.find(c => s / c <= 7) ?? 3600) * 1000;
}

function nearestByT(series, t) {
  if (!series.length) return null;
  let lo = 0, hi = series.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (series[mid].t < t) lo = mid + 1; else hi = mid;
  }
  if (lo > 0 && Math.abs(series[lo - 1].t - t) < Math.abs(series[lo].t - t)) lo--;
  return series[lo];
}

// ---- line chart ---------------------------------------------------------------------

/**
 * Single-series line chart with a hairline grid, crosshair + tooltip.
 * series: [{ t (ms), v }]. Returns { setCursor(tMs|null), destroy() }.
 */
export function lineChart(container, { series, yMax: fixedYMax, yLabel = '', yUnit = '', yFormat = v => v.toFixed(2), xMax, height = 220 }) {
  container.classList.add('viz-line');
  let scales = null, cursorLine = null, cursorDot = null, cursorT = null;

  function positionCursor(t) {
    if (!scales || !cursorLine) return;
    const p = t == null ? null : nearestByT(series, t);
    if (!p) { cursorLine.style.display = 'none'; cursorDot.style.display = 'none'; return; }
    const x = scales.sx(clamp(t, 0, scales.tMax));
    cursorLine.style.display = '';
    cursorLine.setAttribute('x1', x); cursorLine.setAttribute('x2', x);
    cursorDot.style.display = '';
    cursorDot.setAttribute('cx', scales.sx(p.t)); cursorDot.setAttribute('cy', scales.sy(p.v));
  }

  function render() {
    container.replaceChildren();
    const W = Math.max(280, Math.floor(container.clientWidth) || 600), H = height;
    const m = { l: 48, r: 14, t: 24, b: 30 };
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const tMax = Math.max(1, xMax ?? (series.length ? series[series.length - 1].t : 1));
    const yTicks = 4;
    let vMax = fixedYMax;
    if (vMax == null) {
      const step = niceCeil(Math.max(0.01, ...series.map(p => p.v)) * 1.05 / yTicks);
      vMax = step * Math.ceil(Math.max(0.01, ...series.map(p => p.v)) * 1.05 / step);
    }
    const sx = t => m.l + (t / tMax) * pw;
    const sy = v => m.t + ph - (clamp(v, 0, vMax) / vMax) * ph;
    scales = { sx, sy, tMax, vMax };
    const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, width: '100%', height: H, class: 'viz-svg', role: 'img' });

    for (let i = 0; i <= yTicks; i++) {
      const v = vMax * i / yTicks, y = sy(v);
      root.append(svg('line', { x1: m.l, x2: W - m.r, y1: y, y2: y, class: i === 0 ? 'viz-axis' : 'viz-grid' }));
      root.append(svg('text', { x: m.l - 8, y: y + 4, class: 'viz-tick', 'text-anchor': 'end' }, [yFormat(v)]));
    }
    const interval = pickInterval(tMax);
    for (let t = 0; t <= tMax; t += interval) {
      const x = sx(t);
      root.append(svg('line', { x1: x, x2: x, y1: m.t + ph, y2: m.t + ph + 4, class: 'viz-axis' }));
      root.append(svg('text', { x, y: m.t + ph + 18, class: 'viz-tick', 'text-anchor': 'middle' }, [fmtClock(t)]));
    }
    if (yLabel) root.append(svg('text', { x: m.l, y: 12, class: 'viz-label' }, [yLabel]));

    if (series.length) {
      const d = series.map((p, i) => `${i ? 'L' : 'M'}${sx(p.t).toFixed(1)},${sy(p.v).toFixed(1)}`).join('');
      const area = `${d}L${sx(series[series.length - 1].t).toFixed(1)},${sy(0)}L${sx(series[0].t).toFixed(1)},${sy(0)}Z`;
      root.append(svg('path', { d: area, class: 'viz-area' }));
      root.append(svg('path', { d, class: 'viz-line-path' }));
    }
    cursorLine = svg('line', { class: 'viz-cursor', y1: m.t, y2: m.t + ph, style: 'display:none' });
    cursorDot = svg('circle', { r: 4.5, class: 'viz-cursor-dot', style: 'display:none' });
    root.append(cursorLine, cursorDot);

    const hit = svg('rect', { x: m.l, y: m.t, width: pw, height: ph, fill: 'transparent', style: 'cursor:crosshair' });
    hit.addEventListener('pointermove', e => {
      const r = root.getBoundingClientRect();
      const scale = r.width / W;
      const t = clamp((e.clientX - r.left) / scale - m.l, 0, pw) / pw * tMax;
      const p = nearestByT(series, t);
      if (!p) return;
      positionCursor(p.t);
      showTip(e.clientX, e.clientY, `<b>${fmtClock(p.t)}</b><br>${yFormat(p.v)}${yUnit ? ` ${escapeHtml(yUnit)}` : ''}`);
    });
    hit.addEventListener('pointerleave', () => { hideTip(); positionCursor(cursorT); });
    root.append(hit);
    container.append(root);
    positionCursor(cursorT);
  }

  const ro = new ResizeObserver(() => render());
  render();
  ro.observe(container);
  return {
    setCursor(t) { cursorT = t; positionCursor(t); },
    destroy() { ro.disconnect(); hideTip(); },
  };
}

// ---- stacked 100% bar (HTML) --------------------------------------------------------

/** segments: [{ label, value (fraction 0..1), fill, ink }] */
export function stackedBar(container, segments) {
  container.classList.add('viz-stack');
  const bar = document.createElement('div');
  bar.className = 'viz-stack-bar';
  const legend = document.createElement('div');
  legend.className = 'viz-stack-legend';
  for (const s of segments) {
    const seg = document.createElement('div');
    seg.className = 'viz-stack-seg';
    seg.style.flexGrow = String(Math.max(s.value, 0.002) * 1000);
    seg.style.background = s.fill;
    seg.style.color = s.ink;
    seg.title = `${s.label}: ${fmtPct(s.value)}`;
    if (s.value >= 0.1) seg.textContent = fmtPct(s.value);
    bar.append(seg);
    const item = document.createElement('span');
    item.className = 'viz-legend-item';
    item.innerHTML = `<i class="swatch" style="background:${s.fill}"></i>${escapeHtml(s.label)} <b>${fmtPct(s.value)}</b>`;
    legend.append(item);
  }
  container.replaceChildren(bar, legend);
}

// ---- heat grid (HTML) ---------------------------------------------------------------

export function heatGrid(container, { values, rowLabels, colLabels, format = v => fmtPct(v), tooltip }) {
  container.classList.add('viz-heatgrid');
  container.replaceChildren();
  const cols = colLabels.length;
  container.style.gridTemplateColumns = `auto repeat(${cols}, minmax(0, 1fr))`;
  const vmax = Math.max(1e-9, ...values.flat());
  container.append(Object.assign(document.createElement('div'), { className: 'hg-corner' }));
  for (const c of colLabels) {
    const h = document.createElement('div');
    h.className = 'hg-col';
    h.textContent = c;
    container.append(h);
  }
  values.forEach((row, r) => {
    const h = document.createElement('div');
    h.className = 'hg-row';
    h.textContent = rowLabels[r];
    container.append(h);
    row.forEach((v, c) => {
      const cell = document.createElement('div');
      cell.className = 'hg-cell';
      const col = heat(v / vmax);
      cell.style.background = col.fill;
      cell.style.color = col.ink;
      cell.textContent = format(v);
      const text = tooltip ? tooltip(r, c, v) : `${rowLabels[r]} · ${colLabels[c]}: ${format(v)}`;
      cell.addEventListener('pointermove', e => showTip(e.clientX, e.clientY, escapeHtml(text)));
      cell.addEventListener('pointerleave', hideTip);
      container.append(cell);
    });
  });
}

// ---- trajectory in the frame ---------------------------------------------------------

/** path: [{ t, x, y }] in normalised frame coords; occupancy: { cols, rows, cells }. */
export function trajectoryPlot(container, { path, occupancy, aspect, durationMs }) {
  container.classList.add('viz-traj-wrap');
  container.replaceChildren();
  const W = 1000, H = Math.round(W / aspect);
  const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'viz-svg viz-traj', width: '100%', role: 'img' });
  root.style.aspectRatio = `${W} / ${H}`;
  root.append(svg('rect', { x: 0, y: 0, width: W, height: H, rx: 8, class: 'viz-frame' }));

  const { cols, rows, cells } = occupancy;
  const maxCell = Math.max(1e-9, ...cells);
  const cw = W / cols, ch = H / rows;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const v = cells[r * cols + c];
      if (!v) continue;
      const rect = svg('rect', {
        x: c * cw + 1, y: r * ch + 1, width: cw - 2, height: ch - 2, rx: 3, class: 'viz-occ',
        style: `fill-opacity:${(0.06 + 0.34 * v / maxCell).toFixed(3)}`,
      });
      const text = `${fmtPct(v, 1)} of the time here`;
      rect.addEventListener('pointermove', e => showTip(e.clientX, e.clientY, text));
      rect.addEventListener('pointerleave', hideTip);
      root.append(rect);
    }
  }

  const T = Math.max(1, durationMs || (path.length ? path[path.length - 1].t : 1));
  const g = svg('g', { 'stroke-width': 4, 'stroke-linecap': 'round', fill: 'none', style: 'pointer-events:none' });
  for (let i = 1; i < path.length; i++) {
    const a = path[i - 1], b = path[i];
    g.append(svg('line', { x1: a.x * W, y1: a.y * H, x2: b.x * W, y2: b.y * H, stroke: heat(b.t / T).fill }));
  }
  root.append(g);
  if (path.length) {
    const s = path[0], e = path[path.length - 1];
    root.append(svg('circle', { cx: s.x * W, cy: s.y * H, r: 9, class: 'viz-start' }));
    root.append(svg('circle', { cx: e.x * W, cy: e.y * H, r: 9, style: `fill:${heat(1).fill}`, class: 'viz-end' }));
  }
  const hit = svg('rect', { x: 0, y: 0, width: W, height: H, fill: 'transparent' });
  hit.addEventListener('pointermove', e => {
    if (!path.length) return;
    const r = root.getBoundingClientRect();
    const x = (e.clientX - r.left) / r.width, y = (e.clientY - r.top) / r.height;
    let best = null, bd = Infinity;
    for (const p of path) {
      const d = Math.hypot((p.x - x) * aspect, p.y - y);
      if (d < bd) { bd = d; best = p; }
    }
    if (best && bd < 0.08) showTip(e.clientX, e.clientY, `<b>${fmtClock(best.t)}</b><br>x ${fmtPct(best.x)} · y ${fmtPct(best.y)}`);
    else hideTip();
  });
  hit.addEventListener('pointerleave', hideTip);
  root.append(hit);
  container.append(root, heatLegend('Start', 'End'));
}

// ---- stick figure ---------------------------------------------------------------------

/**
 * pose: 13 [x, y] points (MAJOR order) relative to the hips, in torso lengths, y down.
 * values: optional 13 numbers in 0..1 drawn as a per-joint heat.
 */
export function bodyFigure(container, { pose, values = null, names = null, tooltip = null, showSides = true }) {
  container.classList.add('viz-figure-wrap');
  container.replaceChildren();
  const xs = pose.map(p => p[0]), ys = pose.map(p => p[1]);
  const pad = 0.45;
  const minX = Math.min(...xs) - pad, maxX = Math.max(...xs) + pad;
  const minY = Math.min(...ys) - pad, maxY = Math.max(...ys) + pad;
  const root = svg('svg', { viewBox: `${minX} ${minY} ${maxX - minX} ${maxY - minY}`, class: 'viz-figure', role: 'img' });
  root.style.aspectRatio = `${maxX - minX} / ${maxY - minY}`;
  const [nx, ny] = pose[0];
  const neckX = (pose[1][0] + pose[2][0]) / 2, neckY = (pose[1][1] + pose[2][1]) / 2;
  const bones = svg('g', { class: 'viz-bones' });
  for (const [a, b] of MAJOR_CONNECTIONS) {
    bones.append(svg('line', { x1: pose[a][0], y1: pose[a][1], x2: pose[b][0], y2: pose[b][1] }));
  }
  bones.append(svg('line', { x1: neckX, y1: neckY, x2: nx, y2: ny }));
  root.append(bones);
  root.append(svg('circle', { cx: nx, cy: ny, r: 0.2, class: 'viz-head' }));
  pose.forEach(([x, y], j) => {
    if (j === 0 && !values) return;
    const v = values ? values[j] : null;
    const r = values ? 0.08 + 0.15 * clamp(v, 0, 1) : 0.06;
    const circle = svg('circle', { cx: x, cy: y, r, class: values ? 'viz-joint-heat' : 'viz-joint', style: values ? `fill:${heat(v).fill}` : '' });
    if (names) {
      const text = tooltip ? tooltip(j, v) : names[j];
      circle.addEventListener('pointermove', e => showTip(e.clientX, e.clientY, escapeHtml(text)));
      circle.addEventListener('pointerleave', hideTip);
    }
    root.append(circle);
  });
  if (showSides) {
    root.append(svg('text', { x: minX + 0.12, y: minY + 0.3, class: 'viz-side' }, ['L']));
    root.append(svg('text', { x: maxX - 0.12, y: minY + 0.3, class: 'viz-side', 'text-anchor': 'end' }, ['R']));
  }
  container.append(root);
}

// ---- table twin ------------------------------------------------------------------------

export function dataTable(summary, columns, rows) {
  const details = document.createElement('details');
  details.className = 'viz-table';
  const head = columns.map(c => `<th>${escapeHtml(c)}</th>`).join('');
  const body = rows.map(r => `<tr>${r.map(c => `<td>${escapeHtml(c)}</td>`).join('')}</tr>`).join('');
  details.innerHTML = `<summary>${escapeHtml(summary)}</summary><div class="table-scroll"><table class="data-table"><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
  return details;
}
