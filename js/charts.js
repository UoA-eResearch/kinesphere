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
export function lineChart(container, { series, yMax: fixedYMax, yLabel = '', yUnit = '', yFormat = v => v.toFixed(2), tipFormat = null, xMax, height = 220 }) {
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
      showTip(e.clientX, e.clientY, `<b>${fmtClock(p.t)}</b><br>${tipFormat ? escapeHtml(tipFormat(p.v)) : `${yFormat(p.v)}${yUnit ? ` ${escapeHtml(yUnit)}` : ''}`}`);
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
/**
 * Bird's eye view of the floor: the camera at the top centre with its field of view, the dancer's
 * estimated path (coloured start to end), and shaded 0.25 m cells where they spent the most time.
 * x is metres to the dancer's right as they face the camera, z metres from the camera.
 */
export function floorPlot(container, { path, cells, fov, durationMs }) {
  container.classList.add('viz-floor-wrap');
  container.replaceChildren();
  const pts = path.filter(p => Number.isFinite(p.x) && Number.isFinite(p.z));
  const tanH = Math.tan((fov / 2) * Math.PI / 180);
  let xHalf = 1.5, zMax = 3;
  for (const p of pts) { xHalf = Math.max(xHalf, Math.abs(p.x) + 0.5); zMax = Math.max(zMax, p.z + 0.6); }
  for (const c of cells) { xHalf = Math.max(xHalf, Math.abs(c.x) + c.size + 0.3, Math.abs(c.x + c.size) + 0.3); zMax = Math.max(zMax, c.z + c.size + 0.3); }
  zMax = Math.ceil(zMax * 2) / 2;
  xHalf = Math.max(xHalf, zMax / (2 * 0.85)); // keep the plot no taller than it is wide
  xHalf = Math.ceil(xHalf * 2) / 2;
  const W = 1000, pad = { l: 70, r: 16, t: 44, b: 62 };
  const pw = W - pad.l - pad.r, k = pw / (2 * xHalf), ph = zMax * k, H = Math.round(ph + pad.t + pad.b);
  const X = x => pad.l + (x + xHalf) * k, Z = z => pad.t + z * k;
  const root = svg('svg', { viewBox: `0 0 ${W} ${H}`, class: 'viz-svg viz-floor', width: '100%', role: 'img',
    'aria-label': 'Bird\'s eye view of the estimated path on the floor, camera at the top' });
  root.style.aspectRatio = `${W} / ${H}`;
  root.append(svg('rect', { x: pad.l, y: pad.t, width: pw, height: ph, rx: 6, class: 'viz-frame' }));
  // grid every 0.5 m (1 m on big floors), labelled in metres
  const step = xHalf > 4 || zMax > 8 ? 1 : 0.5;
  for (let x = -Math.floor(xHalf / step) * step; x <= xHalf + 1e-6; x += step) {
    const major = Math.abs(x - Math.round(x)) < 1e-6;
    root.append(svg('line', { x1: X(x), x2: X(x), y1: pad.t, y2: pad.t + ph, class: 'viz-grid', style: major ? '' : 'stroke-dasharray:2 4' }));
    if (major) root.append(svg('text', { x: X(x), y: pad.t + ph + 26, class: 'viz-tick', 'text-anchor': 'middle' }, [x === 0 ? '0' : `${x > 0 ? '+' : '−'}${Math.abs(x)} m`]));
  }
  for (let z = step; z <= zMax + 1e-6; z += step) {
    const major = Math.abs(z - Math.round(z)) < 1e-6;
    root.append(svg('line', { x1: pad.l, x2: pad.l + pw, y1: Z(z), y2: Z(z), class: 'viz-grid', style: major ? '' : 'stroke-dasharray:2 4' }));
    if (major && z < zMax - 1e-6) root.append(svg('text', { x: pad.l - 10, y: Z(z) + 8, class: 'viz-tick', 'text-anchor': 'end' }, [`${z} m`]));
  }
  // camera field of view
  const zSide = xHalf / tanH; // where the edge of view leaves the plot through its sides
  const corners = zSide >= zMax
    ? [[0, 0], [-zMax * tanH, zMax], [zMax * tanH, zMax]]
    : [[0, 0], [-xHalf, zSide], [-xHalf, zMax], [xHalf, zMax], [xHalf, zSide]];
  const wedge = corners.map(([x, z]) => `${X(x).toFixed(1)},${Z(z).toFixed(1)}`);
  root.append(svg('polygon', { points: wedge.join(' '), class: 'viz-fov' }));
  // time spent in each floor cell
  const maxShare = Math.max(1e-9, ...cells.map(c => c.share));
  for (const c of cells) {
    const rect = svg('rect', {
      x: X(c.x) + 1, y: Z(c.z) + 1, width: c.size * k - 2, height: c.size * k - 2, rx: 3, class: 'viz-occ',
      style: `fill-opacity:${(0.05 + 0.35 * c.share / maxShare).toFixed(3)}`,
    });
    const text = `${fmtPct(c.share, 1)} of the time in this ${Math.round(c.size * 100)} cm square`;
    rect.addEventListener('pointermove', e => showTip(e.clientX, e.clientY, text));
    rect.addEventListener('pointerleave', hideTip);
    root.append(rect);
  }
  // camera
  const cx = X(0), cy = Z(0);
  root.append(svg('rect', { x: cx - 22, y: cy - 30, width: 44, height: 26, rx: 5, class: 'viz-camera' }));
  root.append(svg('text', { x: cx + 32, y: cy - 10, class: 'viz-label' }, ['Camera']));
  // path
  const T = Math.max(1, durationMs || (pts.length ? pts[pts.length - 1].t : 1));
  const g = svg('g', { 'stroke-width': 4, 'stroke-linecap': 'round', fill: 'none', style: 'pointer-events:none' });
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1], b = pts[i];
    g.append(svg('line', { x1: X(a.x), y1: Z(a.z), x2: X(b.x), y2: Z(b.z), stroke: heat(b.t / T).fill }));
  }
  root.append(g);
  if (pts.length) {
    const s0 = pts[0], e = pts[pts.length - 1];
    root.append(svg('circle', { cx: X(s0.x), cy: Z(s0.z), r: 9, class: 'viz-start' }));
    root.append(svg('circle', { cx: X(e.x), cy: Z(e.z), r: 9, style: `fill:${heat(1).fill}`, class: 'viz-end' }));
  }
  root.append(svg('text', { x: pad.l, y: H - 6, class: 'viz-tick' }, ["← dancer's left"]));
  root.append(svg('text', { x: pad.l + pw, y: H - 6, class: 'viz-tick', 'text-anchor': 'end' }, ["dancer's right →"]));
  const hit = svg('rect', { x: pad.l, y: pad.t, width: pw, height: ph, fill: 'transparent' });
  hit.addEventListener('pointermove', e => {
    if (!pts.length) return;
    const r = root.getBoundingClientRect(), sc = r.width / W;
    const mx = (e.clientX - r.left) / sc, my = (e.clientY - r.top) / sc;
    let best = null, bd = Infinity;
    for (const p of pts) {
      const d = Math.hypot(X(p.x) - mx, Z(p.z) - my);
      if (d < bd) { bd = d; best = p; }
    }
    if (best && bd < 30) {
      const side = Math.abs(best.x) < 0.05 ? 'centre' : `${Math.abs(best.x).toFixed(1)} m to the ${best.x > 0 ? 'right' : 'left'}`;
      showTip(e.clientX, e.clientY, `<b>${fmtClock(best.t)}</b><br>${best.z.toFixed(1)} m from the camera · ${side}`);
    } else hideTip();
  });
  hit.addEventListener('pointerleave', hideTip);
  root.append(hit);
  container.append(root, heatLegend('Start', 'End'));
}

// ---- stick figure ---------------------------------------------------------------------

const LEFT_SIDE = new Set([1, 3, 5, 7, 9, 11]);

function drawStick(root, pose, { values = null, names = null, tooltip = null } = {}) {
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
}

/**
 * pose: 13 [x, y] points (MAJOR order) relative to the mid-hip, in torso lengths, y down.
 * values: optional 13 numbers in 0..1 drawn as a per-joint heat (colour and size).
 * labels: optional 13 strings printed beside the joints (the "Show values" view).
 * rings: optional [{ r, label, active }] concentric circles around the centre of the torso (midway
 *   between mid-hip and mid-shoulder), the kinesphere sizes.
 */
export function bodyFigure(container, { pose, values = null, names = null, tooltip = null, showSides = true, labels = null, rings = null }) {
  container.classList.add('viz-figure-wrap');
  container.replaceChildren();
  const xs = pose.map(p => p[0]), ys = pose.map(p => p[1]);
  const pad = 0.45;
  let minX = Math.min(...xs) - pad, maxX = Math.max(...xs) + pad;
  let minY = Math.min(...ys) - pad, maxY = Math.max(...ys) + pad;
  const rcx = (pose[1][0] + pose[2][0] + pose[7][0] + pose[8][0]) / 4;
  const rcy = (pose[1][1] + pose[2][1] + pose[7][1] + pose[8][1]) / 4;
  if (rings?.length) {
    const rMax = Math.max(...rings.map(r => r.r)) + 0.1;
    minX = Math.min(minX, rcx - rMax); maxX = Math.max(maxX, rcx + rMax);
    minY = Math.min(minY, rcy - rMax); maxY = Math.max(maxY, rcy + rMax);
  }
  if (labels) { minX -= 0.75; maxX += 0.75; minY -= 0.15; }
  const root = svg('svg', { viewBox: `${minX} ${minY} ${maxX - minX} ${maxY - minY}`, class: 'viz-figure', role: 'img' });
  root.style.aspectRatio = `${maxX - minX} / ${maxY - minY}`;
  if (rings?.length) {
    const g = svg('g', { class: 'viz-rings' });
    for (const ring of rings) {
      g.append(svg('circle', { cx: rcx, cy: rcy, r: ring.r, class: ring.active ? 'viz-ring is-active' : 'viz-ring', style: ring.color ? `stroke:${ring.color}` : '' }));
      if (ring.label) g.append(svg('text', { x: rcx + ring.r * 0.71 + 0.04, y: rcy - ring.r * 0.71 - 0.04, class: 'viz-ring-label' }, [ring.label]));
    }
    g.append(svg('circle', { cx: rcx, cy: rcy, r: 0.05, class: 'viz-ring-centre' }));
    root.append(g);
  }
  drawStick(root, pose, { values, names, tooltip });
  if (labels) {
    pose.forEach(([x, y], j) => {
      if (!labels[j]) return;
      const r = values ? 0.08 + 0.15 * clamp(values[j], 0, 1) : 0.06;
      const attrs = j === 0 ? { x, y: y - 0.32, 'text-anchor': 'middle' }
        : LEFT_SIDE.has(j) ? { x: x - r - 0.06, y: y + 0.06, 'text-anchor': 'end' }
          : { x: x + r + 0.06, y: y + 0.06, 'text-anchor': 'start' };
      root.append(svg('text', { ...attrs, class: 'viz-val' }, [labels[j]]));
    });
  }
  if (showSides) {
    root.append(svg('text', { x: minX + 0.12, y: minY + 0.3, class: 'viz-side' }, ['L']));
    root.append(svg('text', { x: maxX - 0.12, y: minY + 0.3, class: 'viz-side', 'text-anchor': 'end' }, ['R']));
  }
  container.append(root);
}

/** Fill colour for a share, on the lighter half of the ramp so large blocks stay calm. */
const blockFill = (v, max) => heat(0.08 + 0.5 * clamp(max > 0 ? v / max : 0, 0, 1)).fill;

/** Left / right display: the figure split down the middle, each half shaded by its share. */
export function sideFigure(container, { pose, left, right }) {
  container.classList.add('viz-figure-wrap');
  container.replaceChildren();
  const xs = pose.map(p => p[0]), ys = pose.map(p => p[1]);
  const minX = Math.min(...xs) - 0.6, maxX = Math.max(...xs) + 0.6;
  const minY = Math.min(...ys) - 0.5, maxY = Math.max(...ys) + 0.35, footY = maxY + 0.75;
  const root = svg('svg', { viewBox: `${minX} ${minY} ${maxX - minX} ${footY - minY}`, class: 'viz-figure viz-side-figure', role: 'img' });
  root.style.aspectRatio = `${maxX - minX} / ${footY - minY}`;
  root.setAttribute('aria-label', `Left side ${fmtPct(left)}, right side ${fmtPct(right)} of movement`);
  const m = Math.max(left, right);
  root.append(svg('rect', { x: minX, y: minY, width: -minX - 0.02, height: maxY - minY, rx: 0.12, style: `fill:${blockFill(left, m)}` }));
  root.append(svg('rect', { x: 0.02, y: minY, width: maxX - 0.02, height: maxY - minY, rx: 0.12, style: `fill:${blockFill(right, m)}` }));
  root.append(svg('line', { x1: 0, x2: 0, y1: minY, y2: maxY, style: 'stroke:var(--surface);stroke-width:0.04' }));
  drawStick(root, pose);
  root.append(svg('text', { x: minX + 0.15, y: minY + 0.35, class: 'viz-side' }, ['L']));
  root.append(svg('text', { x: maxX - 0.15, y: minY + 0.35, class: 'viz-side', 'text-anchor': 'end' }, ['R']));
  root.append(svg('text', { x: minX / 2, y: maxY + 0.55, class: 'viz-big', 'text-anchor': 'middle' }, [fmtPct(left)]));
  root.append(svg('text', { x: maxX / 2, y: maxY + 0.55, class: 'viz-big', 'text-anchor': 'middle' }, [fmtPct(right)]));
  container.append(root);
}

/**
 * Levels: three horizontal bands (high above standing shoulder height, medium between
 * shoulder and knee height, low below the knee), each shaded and labelled with its share.
 */
export function levelsFigure(container, { pose, shares, labels = { high: 'High', mid: 'Medium', low: 'Low' } }) {
  container.classList.add('viz-figure-wrap');
  container.replaceChildren();
  const xs = pose.map(p => p[0]), ys = pose.map(p => p[1]);
  const shoulderY = (pose[1][1] + pose[2][1]) / 2, kneeY = (pose[9][1] + pose[10][1]) / 2;
  const figMinX = Math.min(...xs) - 0.4, figMaxX = Math.max(...xs) + 0.4;
  const top = Math.min(...ys) - 0.9, bottom = Math.max(...ys) + 0.3;
  const minX = figMinX - 1.35, maxX = figMaxX + 1.6;
  const root = svg('svg', { viewBox: `${minX} ${top} ${maxX - minX} ${bottom - top}`, class: 'viz-figure viz-levels-figure', role: 'img' });
  root.style.aspectRatio = `${maxX - minX} / ${bottom - top}`;
  root.setAttribute('aria-label', `High ${fmtPct(shares.high)}, medium ${fmtPct(shares.mid)}, low ${fmtPct(shares.low)} of the time`);
  const m = Math.max(shares.high, shares.mid, shares.low);
  const bands = [['high', top, shoulderY], ['mid', shoulderY, kneeY], ['low', kneeY, bottom]];
  for (const [key, y0, y1] of bands) {
    const gap = 0.025;
    const rect = svg('rect', { x: figMinX, y: y0 + gap, width: figMaxX - figMinX, height: y1 - y0 - 2 * gap, rx: 0.08, style: `fill:${blockFill(shares[key], m)}` });
    const tip = `${labels[key]}: ${fmtPct(shares[key], 1)} of the time`;
    rect.addEventListener('pointermove', e => showTip(e.clientX, e.clientY, escapeHtml(tip)));
    rect.addEventListener('pointerleave', hideTip);
    root.append(rect);
    const cy = (y0 + y1) / 2 + 0.12;
    root.append(svg('text', { x: figMinX - 0.15, y: cy, class: 'viz-big', 'text-anchor': 'end' }, [fmtPct(shares[key])]));
    root.append(svg('text', { x: figMaxX + 0.15, y: cy, class: 'viz-level-label' }, [labels[key]]));
  }
  drawStick(root, pose);
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
