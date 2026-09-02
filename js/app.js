// Kinesphere: live pose view + recording, analysis dashboard and session library.

import { createPoseDetector, flattenLandmarks, handsAboveHead, drawSkeleton, FRAME_SIZE } from './pose.js';
import { store, buildSession, serialize, deserialize, toCSV, download, safeFilename, StorageFullError } from './session.js';
import { analyze, KINESPHERE_CLASSES, GRID_ROWS, GRID_COLS, MAJOR_NAMES } from './analysis.js';
import { lineChart, stackedBar, heatGrid, trajectoryPlot, bodyFigure, dataTable, heatLegend, ordinalColors, isDark, hideTip } from './charts.js';
import { fmtClock, fmtDuration, fmtPct, fmtBytes, fmtDate, escapeHtml, clamp } from './util.js';

const $ = s => document.querySelector(s);
const HOLD_MS = 1500;          // how long both hands must stay up to trigger start/stop
const COUNTDOWN_MS = 3000;     // countdown after the start gesture
const LOCKOUT_MS = 4000;       // ignore the gesture for a while after it fired
const STORE_INTERVAL_MS = 1000 / 30; // store at most 30 frames per second
const STORAGE_BUDGET = 5 * 1024 * 1024;
const THEME_KEY = 'kinesphere:theme';
const OVERLAY_KEY = 'kinesphere:overlay';
const CAMERA_KEY = 'kinesphere:camera';

/** Neutral standing pose (MAJOR order, torso units, selfie view) used for the body heat diagram. */
const TEMPLATE_POSE = [
  [0, -1.45], [-0.45, -1.0], [0.45, -1.0], [-0.85, -0.55], [0.85, -0.55], [-1.15, -0.1], [1.15, -0.1],
  [-0.3, 0], [0.3, 0], [-0.35, 1.0], [0.35, 1.0], [-0.4, 2.0], [0.4, 2.0],
];

const ui = {
  video: $('#video'), canvas: $('#overlay'), stage: $('#stage'), placeholder: $('#stage-placeholder'),
  cameraError: $('#camera-error'), btnCamera: $('#btn-camera'), btnRecord: $('#btn-record'), btnStop: $('#btn-stop'),
  chkGesture: $('#chk-gesture'), selModel: $('#sel-model'), selCamera: $('#sel-camera'), cameraWrap: $('#camera-wrap'), status: $('#status'), fps: $('#fps'),
  recBadge: $('#rec-badge'), recTime: $('#rec-time'), countdown: $('#countdown'),
  hold: $('#hold'), holdFill: $('#hold-fill'), holdLabel: $('#hold-label'),
  views: { live: $('#view-live'), dashboard: $('#view-dashboard'), sessions: $('#view-sessions') },
  sessionCount: $('#session-count'), btnImport: $('#btn-import'), fileImport: $('#file-import'),
  btnTheme: $('#btn-theme'), toast: $('#toast'),
  hudRecord: $('#hud-record'), hudStop: $('#hud-stop'), btnOverlay: $('#btn-overlay'), btnFullscreen: $('#btn-fullscreen'),
};
const overlayCtx = ui.canvas.getContext('2d');

const state = {
  view: 'live',
  detector: null, detectorPromise: null,
  stream: null, deviceId: null, mirrored: true, loopRunning: false, lastVideoTime: -1,
  recording: null, countdown: null, hold: null, lockoutUntil: 0,
  fps: { count: 0, since: performance.now() },
  lastStatus: '',
  showOverlay: localStorage.getItem(OVERLAY_KEY) !== 'off',
  current: null,        // { session, analysis, saved, saveError }
  charts: [], replay: null,
};

// ---------------------------------------------------------------------------------------
// Generic UI helpers

let toastTimer = 0;
function toast(msg, ms = 4000) {
  ui.toast.textContent = msg;
  ui.toast.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { ui.toast.hidden = true; }, ms);
}

function setStatus(msg) {
  if (msg === state.lastStatus) return;
  state.lastStatus = msg;
  ui.status.textContent = msg;
}

function setView(name) {
  state.view = name;
  for (const [k, el] of Object.entries(ui.views)) el.hidden = k !== name;
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.toggle('is-active', b.dataset.view === name));
  hideTip();
  if (name === 'sessions') renderSessions();
  if (name === 'live' && state.stream) startLoop();
  window.scrollTo({ top: 0 });
}

function updateSessionCount() {
  const n = store.list().length;
  ui.sessionCount.textContent = n ? String(n) : '';
}

function applyTheme(theme) {
  if (theme === 'auto') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  ui.btnTheme.textContent = { auto: '◐', light: '☀', dark: '☾' }[theme] ?? '◐';
  ui.btnTheme.title = `Theme: ${theme}`;
}
function currentTheme() {
  return localStorage.getItem(THEME_KEY) || 'auto';
}
function onThemeChanged() {
  if (state.current && state.view === 'dashboard') renderDashboard();
}

// ---------------------------------------------------------------------------------------
// Camera + detection loop

function showCameraError(msg) {
  ui.cameraError.textContent = msg;
  ui.cameraError.hidden = false;
}

/** Open a camera stream, by device id when given, otherwise the default front-facing camera. */
function openStream(deviceId) {
  const video = { width: { ideal: 1280 }, height: { ideal: 720 } };
  if (deviceId) video.deviceId = { exact: deviceId };
  else video.facingMode = 'user';
  return navigator.mediaDevices.getUserMedia({ video, audio: false });
}

/** Show a stream in the preview, replacing (and stopping) the previous one. */
async function attachStream(stream, requestedDeviceId = null) {
  state.stream?.getTracks().forEach(t => t.stop());
  state.stream = stream;
  state.lastVideoTime = -1;
  ui.video.srcObject = stream;
  await new Promise(resolve => {
    if (ui.video.readyState >= 1) resolve();
    else ui.video.addEventListener('loadedmetadata', resolve, { once: true });
  });
  try { await ui.video.play(); } catch { /* autoplay is muted so this should not fail */ }
  resizeCanvas();
  const settings = stream.getVideoTracks()[0]?.getSettings?.() ?? {};
  state.deviceId = requestedDeviceId || settings.deviceId || null;
  // Rear (environment-facing) cameras show the scene as it is; everything else is a mirror.
  state.mirrored = settings.facingMode !== 'environment';
  ui.stage.classList.toggle('no-mirror', !state.mirrored);
  await refreshCameraList();
}

/** Populate the camera selector; it is only shown when more than one camera exists. */
async function refreshCameraList() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  let devices;
  try { devices = await navigator.mediaDevices.enumerateDevices(); } catch { return; }
  const cams = devices.filter(d => d.kind === 'videoinput');
  ui.selCamera.replaceChildren(...cams.map((d, i) => {
    const opt = document.createElement('option');
    opt.value = d.deviceId;
    opt.textContent = d.label || `Camera ${i + 1}`;
    return opt;
  }));
  if (state.deviceId && cams.some(d => d.deviceId === state.deviceId)) ui.selCamera.value = state.deviceId;
  ui.cameraWrap.hidden = cams.length < 2;
}

async function switchCamera(deviceId) {
  if (!state.stream || state.recording || !deviceId || deviceId === state.deviceId) return;
  ui.selCamera.disabled = true;
  try {
    await attachStream(await openStream(deviceId), deviceId);
    localStorage.setItem(CAMERA_KEY, deviceId);
  } catch (err) {
    toast(`Could not switch camera: ${err.message || err.name}`);
    if (state.deviceId) ui.selCamera.value = state.deviceId;
  } finally {
    ui.selCamera.disabled = false;
  }
}

async function startCamera() {
  ui.btnCamera.disabled = true;
  ui.cameraError.hidden = true;
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('This browser cannot access the camera here. Camera access needs a secure (https) page or localhost.');
    }
    const preferred = localStorage.getItem(CAMERA_KEY);
    let stream;
    try {
      stream = await openStream(preferred);
    } catch (err) {
      if (!preferred) throw err;
      stream = await openStream(null); // the remembered camera is gone; fall back to the default
    }
    await attachStream(stream);
  } catch (err) {
    showCameraError(`Camera error: ${err.message || err.name}`);
    ui.btnCamera.disabled = false;
    return;
  }
  ui.placeholder.hidden = true;
  try {
    await ensureDetector();
  } catch (err) {
    console.error(err);
    setStatus(`Could not load the pose model: ${err.message}`);
    toast('MediaPipe could not be loaded. Check your network connection and reload.');
    return;
  }
  ui.btnRecord.disabled = false;
  ui.hudRecord.disabled = false;
  startLoop();
}

function setOverlay(on) {
  state.showOverlay = on;
  localStorage.setItem(OVERLAY_KEY, on ? 'on' : 'off');
  ui.btnOverlay.textContent = on ? 'Overlay: on' : 'Overlay: off';
  ui.btnOverlay.setAttribute('aria-pressed', String(on));
  if (!on) overlayCtx.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
}

function toggleFullscreen() {
  if (document.fullscreenElement) {
    document.exitFullscreen?.();
  } else if (ui.stage.requestFullscreen) {
    ui.stage.requestFullscreen().catch(err => toast(`Fullscreen is not available: ${err.message}`));
  } else {
    toast('Fullscreen is not supported in this browser.');
  }
}

function resizeCanvas() {
  const w = ui.video.videoWidth, h = ui.video.videoHeight;
  if (w && h && (ui.canvas.width !== w || ui.canvas.height !== h)) {
    ui.canvas.width = w;
    ui.canvas.height = h;
  }
}

function ensureDetector() {
  const model = ui.selModel.value;
  if (state.detector && state.detector.model === model) return Promise.resolve(state.detector);
  if (state.detectorPromise) return state.detectorPromise;
  const old = state.detector;
  state.detector = null;
  state.detectorPromise = createPoseDetector({ model, onStatus: setStatus }).then(
    d => { state.detector = d; state.detectorPromise = null; old?.close(); return d; },
    err => { state.detectorPromise = null; throw err; },
  );
  return state.detectorPromise;
}

function startLoop() {
  if (state.loopRunning) return;
  state.loopRunning = true;
  requestAnimationFrame(loop);
}

function loop() {
  if (!state.stream || (state.view !== 'live' && !state.recording)) {
    state.loopRunning = false;
    return;
  }
  const v = ui.video;
  if (state.detector && v.readyState >= 2 && v.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = v.currentTime;
    resizeCanvas();
    const now = performance.now();
    let lm = null;
    try {
      const result = state.detector.detect(v, now);
      const first = result?.landmarks?.[0];
      if (first) lm = flattenLandmarks(first);
    } catch (err) {
      console.error(err);
    }
    onFrame(lm, now);
  }
  requestAnimationFrame(loop);
}

function onFrame(lm, now) {
  state.fps.count++;
  if (now - state.fps.since >= 1000) {
    const fps = state.fps.count * 1000 / (now - state.fps.since);
    state.fps.count = 0;
    state.fps.since = now;
    ui.fps.textContent = `${fps.toFixed(0)} fps · ${state.detector.model} · ${state.detector.delegate}`;
  }

  const { width, height } = ui.canvas;
  overlayCtx.clearRect(0, 0, width, height);
  if (lm && state.showOverlay) {
    drawSkeleton(overlayCtx, lm, 0, width, height, {
      lineWidth: Math.max(2, width / 320), radius: Math.max(3, width / 240),
    });
  }

  if (state.countdown) {
    const left = state.countdown.endsAt - now;
    if (left <= 0) {
      state.countdown = null;
      ui.countdown.hidden = true;
      startRecording('pose', now);
    } else {
      ui.countdown.hidden = false;
      ui.countdown.textContent = String(Math.ceil(left / 1000));
    }
  }

  if (state.recording) {
    const rec = state.recording;
    if (lm && now - rec.lastStored >= STORE_INTERVAL_MS - 1) {
      rec.frames.push({ t: now - rec.startedAt, lm });
      rec.lastStored = now;
    }
    ui.recTime.textContent = fmtClock(now - rec.startedAt);
  }

  updateGesture(lm, now);
  updateStatus(lm);
}

function updateGesture(lm, now) {
  const active = ui.chkGesture.checked && lm && !state.countdown && now >= state.lockoutUntil && handsAboveHead(lm);
  if (!active) {
    if (state.hold) { state.hold = null; ui.hold.hidden = true; }
    return;
  }
  state.hold ??= { since: now };
  const p = clamp((now - state.hold.since) / HOLD_MS, 0, 1);
  ui.hold.hidden = false;
  ui.holdFill.style.width = `${(p * 100).toFixed(0)}%`;
  ui.holdLabel.textContent = state.recording ? 'Keep both hands up to stop' : 'Keep both hands up to start';
  if (p >= 1) {
    const since = state.hold.since;
    state.hold = null;
    ui.hold.hidden = true;
    state.lockoutUntil = now + LOCKOUT_MS;
    if (state.recording) stopRecording(since - state.recording.startedAt);
    else state.countdown = { endsAt: now + COUNTDOWN_MS };
  }
}

function updateStatus(lm) {
  let msg;
  if (state.countdown) msg = 'Get ready…';
  else if (state.recording) msg = 'Recording. Press Stop, or raise both hands above your head to finish.';
  else if (!lm) msg = 'No pose detected. Step back so your whole body is visible.';
  else if (ui.chkGesture.checked) msg = 'Pose detected. Press Record, or raise both hands above your head to start.';
  else msg = 'Pose detected. Press Record to start.';
  setStatus(msg);
}

// ---------------------------------------------------------------------------------------
// Recording

function startRecording(trigger, now = performance.now()) {
  if (state.recording || !state.detector) return;
  state.countdown = null;
  ui.countdown.hidden = true;
  state.recording = { startedAt: now, wallStart: Date.now(), lastStored: -Infinity, frames: [], trigger };
  ui.stage.classList.add('is-recording');
  ui.recBadge.hidden = false;
  ui.recTime.textContent = '0:00';
  ui.btnRecord.hidden = true;
  ui.btnStop.hidden = false;
  ui.hudRecord.hidden = true;
  ui.hudStop.hidden = false;
  ui.selModel.disabled = true;
  ui.selCamera.disabled = true;
}

/** Stop recording. `trimBeforeT` drops frames from that time on (used to cut the stop gesture). */
function stopRecording(trimBeforeT = null) {
  const rec = state.recording;
  if (!rec) return;
  state.recording = null;
  ui.stage.classList.remove('is-recording');
  ui.recBadge.hidden = true;
  ui.btnRecord.hidden = false;
  ui.btnStop.hidden = true;
  ui.hudRecord.hidden = false;
  ui.hudStop.hidden = true;
  ui.selModel.disabled = false;
  ui.selCamera.disabled = false;
  let frames = rec.frames;
  if (trimBeforeT != null) frames = frames.filter(f => f.t < trimBeforeT);
  if (frames.length < 10) {
    toast('The recording was too short to analyse, so nothing was saved.');
    return;
  }
  const session = buildSession({
    frames,
    width: ui.video.videoWidth || 1280,
    height: ui.video.videoHeight || 720,
    model: state.detector?.model,
    trigger: rec.trigger,
    startedAt: rec.wallStart,
    mirrored: state.mirrored,
  });
  openSession(session, { save: true });
}

// ---------------------------------------------------------------------------------------
// Sessions

function openSession(session, { save = false } = {}) {
  const analysis = analyze(session);
  state.current = { session, analysis, saved: false, saveError: null };
  if (save) {
    try {
      store.save(session);
      state.current.saved = true;
    } catch (err) {
      if (!(err instanceof StorageFullError)) throw err;
      state.current.saveError = err.message;
    }
  } else {
    state.current.saved = true;
  }
  updateSessionCount();
  renderDashboard();
  setView('dashboard');
}

function exportJSON(session) {
  download(`${safeFilename(session.name)}.kinesphere.json`, serialize(session), 'application/json');
}
function exportCSV(session) {
  download(`${safeFilename(session.name)}.csv`, toCSV(session), 'text/csv');
}

async function importFile(file) {
  if (!file) return;
  try {
    const session = deserialize(await file.text());
    openSession(session, { save: true });
    toast(`Imported “${session.name}”.`);
  } catch (err) {
    toast(`Import failed: ${err.message}`);
  } finally {
    ui.fileImport.value = '';
  }
}

// ---------------------------------------------------------------------------------------
// Dashboard

function stat(label, value, unit = '') {
  return `<div class="stat"><div class="stat-label">${escapeHtml(label)}</div><div class="stat-value">${value}${unit ? `<small>${escapeHtml(unit)}</small>` : ''}</div></div>`;
}

function renderDashboard() {
  const { session, analysis: a, saved, saveError } = state.current;
  for (const c of state.charts) c.destroy?.();
  state.charts = [];
  state.replay?.destroy();
  state.replay = null;

  const fpsText = a.ok ? ` · ${a.fps.toFixed(0)} fps` : '';
  const triggerText = session.trigger === 'pose' ? 'started by pose' : session.trigger === 'button' ? 'started by button' : '';
  const root = ui.views.dashboard;
  root.innerHTML = `
    <div class="dash-header">
      <div>
        <input id="session-name" class="name-input" type="text" value="${escapeHtml(session.name)}" aria-label="Session name" spellcheck="false">
        <div class="meta">${escapeHtml(fmtDate(session.createdAt))} · ${fmtDuration(session.durationMs)} · ${session.frameCount} frames${fpsText} · ${escapeHtml(session.model)} model${triggerText ? ` · ${triggerText}` : ''}</div>
      </div>
      <div class="dash-actions">
        <button class="btn" data-action="export-json">Export JSON</button>
        <button class="btn" data-action="export-csv">Export CSV</button>
        <button class="btn btn-ghost" data-action="delete">Delete</button>
        <button class="btn btn-primary" data-action="new">New recording</button>
      </div>
    </div>
    ${saveError ? `<div class="banner banner-warn"><b>Not saved.</b> ${escapeHtml(saveError)}</div>` : ''}
    ${!saved && !saveError ? '' : ''}
    <section class="card">
      <h2>Replay</h2>
      <p class="card-sub">The recorded pose, as you saw it on screen. Scrub to move the cursor on the charts below.</p>
      <div class="replay"><canvas id="replay-canvas"></canvas></div>
      <div class="replay-controls">
        <button class="btn" id="replay-play" aria-label="Play">▶</button>
        <input type="range" id="replay-range" min="0" max="${Math.max(1, session.durationMs)}" value="0" step="1" aria-label="Playback position">
        <span id="replay-time" class="mono">0:00 / ${fmtClock(session.durationMs)}</span>
      </div>
    </section>
    ${a.ok ? dashboardCards(a) : `<div class="banner">Not enough pose data to analyse this session. ${escapeHtml(a.reason)}</div>`}
  `;

  root.querySelector('#session-name').addEventListener('change', e => {
    const name = e.target.value.trim() || session.name;
    e.target.value = name;
    session.name = name;
    if (state.current.saved) {
      try { store.save(session); } catch (err) { toast(err.message); }
    }
    updateSessionCount();
  });
  root.querySelector('[data-action="export-json"]').onclick = () => exportJSON(session);
  root.querySelector('[data-action="export-csv"]').onclick = () => exportCSV(session);
  root.querySelector('[data-action="delete"]').onclick = () => {
    if (!confirm(`Delete “${session.name}”? This cannot be undone unless you exported it.`)) return;
    store.remove(session.id);
    state.current = null;
    updateSessionCount();
    setView('sessions');
  };
  root.querySelector('[data-action="new"]').onclick = () => setView('live');

  const onTime = t => { for (const c of state.charts) c.setCursor?.(t); };
  state.replay = setupReplay(session, root, onTime);
  if (a.ok) mountCharts(a, session, root);
}

function dashboardCards(a) {
  const k = a.kinesphere;
  const angleRows = a.angles.map(x => `
      <tr><td>${x.label}</td><td>${x.side === 'L' ? 'Left' : 'Right'}</td>
      <td class="num">${Number.isFinite(x.mean) ? `${x.mean.toFixed(0)}°` : '–'}</td>
      <td class="num">${Number.isFinite(x.p5) ? `${x.p5.toFixed(0)}° – ${x.p95.toFixed(0)}°` : '–'}</td></tr>`).join('');
  const z = a.zones;
  const shapes = a.shapes.map((s, i) => `
      <div class="shape"><div class="shape-rank">#${i + 1}</div><div id="shape-${i}"></div><div class="shape-share">${fmtPct(s.share)}<small class="meta"> of the time</small></div></div>`).join('');
  return `
    <section class="card">
      <h2>Kinesphere</h2>
      <p class="card-sub">How far the limbs reached into the space around the body, as a share of the session.</p>
      <div id="kine-bar"></div>
      <div class="stat-row">
        ${stat('Mean openness', fmtPct(k.meanOpenness))}
        ${stat('Pose tracked', fmtPct(a.validFraction), 'of frames')}
      </div>
      <h3>Openness over time</h3>
      <div id="kine-chart"></div>
      <h3>Joint angles</h3>
      <div class="table-scroll"><table class="data-table">
        <thead><tr><th>Joint</th><th>Side</th><th class="num">Mean</th><th class="num">Typical range (5th–95th pct)</th></tr></thead>
        <tbody>${angleRows}</tbody></table></div>
      <p class="fine">Openness averages the shoulder, elbow, hip and knee angles (0° folded → 180° fully extended), weighting arms 60% and legs 40%.
        Below 55% counts as a small kinesphere, 55–80% medium, above 80% large. Shoulder angle is measured between the torso and the upper arm.</p>
    </section>

    <section class="card">
      <h2>Where was the activity?</h2>
      <p class="card-sub">Share of total movement by body region, from the mean speed of each joint. Left and right are the dancer's own sides.</p>
      <div class="two-col">
        <div>
          <div id="zone-grid"></div>
          <div class="stat-row">
            ${stat('Upper', fmtPct(z.vertical.upper))}${stat('Mid', fmtPct(z.vertical.mid))}${stat('Lower', fmtPct(z.vertical.lower))}
          </div>
          <div class="stat-row">
            ${stat('Left side', fmtPct(z.lateral.left))}${stat('Right side', fmtPct(z.lateral.right))}
          </div>
        </div>
        <div>
          <div id="body-figure"></div>
          <div id="body-legend"></div>
        </div>
      </div>
      <p class="fine">Upper = elbows and wrists, mid = shoulders and hips, lower = knees and ankles. Joint dots are sized and coloured by their mean speed relative to the most active joint.</p>
    </section>

    <section class="card">
      <h2>Speed</h2>
      <p class="card-sub">Average joint speed over time, in torso lengths per second (so it does not depend on how far you stood from the camera).</p>
      <div class="stat-row">
        ${stat('Mean speed', a.speed.mean.toFixed(2), 'torso lengths / s')}
        ${stat('Peak', Number.isFinite(a.speed.peak) ? a.speed.peak.toFixed(2) : '–', `at ${fmtClock(a.speed.peakT)}`)}
      </div>
      <div id="speed-chart"></div>
      <div id="speed-table"></div>
    </section>

    <section class="card">
      <h2>Space</h2>
      <p class="card-sub">Where the hips were in the camera frame, from start (ring) to end (filled dot). Shaded cells show where you spent the most time.</p>
      <div class="two-col">
        <div id="space-plot"></div>
        <div>
          <div class="stat-row">
            ${stat('Horizontal coverage', fmtPct(a.space.xRange[1] - a.space.xRange[0]), 'of frame width')}
            ${stat('Vertical coverage', fmtPct(a.space.yRange[1] - a.space.yRange[0]), 'of frame height')}
            ${stat('Path length', a.space.pathLength.toFixed(1), 'torso lengths')}
          </div>
          <div id="space-table"></div>
        </div>
      </div>
    </section>

    ${a.shapes.length ? `
    <section class="card">
      <h2>Top ${a.shapes.length} shapes</h2>
      <p class="card-sub">The most common body shapes, found by clustering every frame's pose. Experimental.</p>
      <div class="shapes">${shapes}</div>
    </section>` : ''}
  `;
}

function mountCharts(a, session, root) {
  const ord = ordinalColors();
  stackedBar(root.querySelector('#kine-bar'), KINESPHERE_CLASSES.map((c, i) => ({
    label: c.label, value: a.kinesphere.shares[c.key], fill: ord[i].fill, ink: ord[i].ink,
  })));
  state.charts.push(lineChart(root.querySelector('#kine-chart'), {
    series: a.kinesphere.series, yMax: 1, xMax: session.durationMs, height: 160,
    yLabel: 'Openness', yFormat: v => fmtPct(v),
  }));

  const z = a.zones;
  heatGrid(root.querySelector('#zone-grid'), {
    values: z.grid, rowLabels: GRID_ROWS, colLabels: GRID_COLS,
    tooltip: (r, c, v) => `${GRID_ROWS[r]} · ${GRID_COLS[c]}: ${fmtPct(v, 1)} of movement`,
  });
  bodyFigure(root.querySelector('#body-figure'), {
    pose: TEMPLATE_POSE, values: z.joints.map(j => j.relative), names: MAJOR_NAMES,
    tooltip: (j, v) => `${MAJOR_NAMES[j]}: ${fmtPct(v)} of the most active joint (${z.joints[j].meanSpeed.toFixed(2)} torso lengths/s)`,
  });
  root.querySelector('#body-legend').append(heatLegend('Less movement', 'More movement'));

  state.charts.push(lineChart(root.querySelector('#speed-chart'), {
    series: a.speed.series, xMax: session.durationMs, yLabel: 'Speed (torso lengths / s)', yUnit: 'torso lengths/s',
  }));
  root.querySelector('#speed-table').append(dataTable('Show speed data', ['Time', 'Speed (torso lengths/s)'],
    a.speed.series.map(p => [fmtClock(p.t), p.v.toFixed(2)])));

  trajectoryPlot(root.querySelector('#space-plot'), {
    path: a.space.path, occupancy: a.space.occupancy, aspect: session.width / session.height, durationMs: session.durationMs,
  });
  root.querySelector('#space-table').append(dataTable('Show position data', ['Time', 'X (% of width)', 'Y (% of height)'],
    a.space.path.map(p => [fmtClock(p.t), (p.x * 100).toFixed(1), (p.y * 100).toFixed(1)])));

  a.shapes.forEach((s, i) => bodyFigure(root.querySelector(`#shape-${i}`), { pose: s.pose, showSides: false }));
}

function setupReplay(session, root, onTime) {
  const canvas = root.querySelector('#replay-canvas');
  const range = root.querySelector('#replay-range');
  const btn = root.querySelector('#replay-play');
  const timeEl = root.querySelector('#replay-time');
  const w = session.width, h = session.height;
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');
  const { times, lm, durationMs } = session;
  let t = 0, playing = false, raf = 0, last = 0;

  const frameAt = ms => {
    let lo = 0, hi = times.length - 1;
    if (!times.length) return -1;
    if (ms < times[0]) return times[0] - ms < 500 ? 0 : -1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (times[mid] <= ms) lo = mid; else hi = mid - 1;
    }
    return lo;
  };
  const draw = () => {
    ctx.clearRect(0, 0, w, h);
    const i = frameAt(t);
    if (i >= 0 && t - times[i] < 500) {
      drawSkeleton(ctx, lm, i * FRAME_SIZE, w, h, {
        mirror: session.mirrored, lineWidth: Math.max(2, w / 320), radius: Math.max(3, w / 240),
        midColor: isDark() ? '#f4f4f2' : '#3a3a38',
      });
    }
  };
  const setTime = nt => {
    t = clamp(nt, 0, durationMs);
    range.value = String(Math.round(t));
    timeEl.textContent = `${fmtClock(t)} / ${fmtClock(durationMs)}`;
    draw();
    onTime(t);
  };
  const tick = now => {
    if (!playing) return;
    setTime(t + (now - last));
    last = now;
    if (t >= durationMs) { playing = false; btn.textContent = '▶'; return; }
    raf = requestAnimationFrame(tick);
  };
  btn.onclick = () => {
    playing = !playing;
    btn.textContent = playing ? '❚❚' : '▶';
    if (playing) {
      if (t >= durationMs) t = 0;
      last = performance.now();
      raf = requestAnimationFrame(tick);
    } else {
      cancelAnimationFrame(raf);
    }
  };
  range.oninput = () => setTime(Number(range.value));
  setTime(0);
  return { destroy() { playing = false; cancelAnimationFrame(raf); } };
}

// ---------------------------------------------------------------------------------------
// Session library

function renderSessions() {
  const list = store.list();
  const usage = store.usage();
  const pct = clamp(usage / STORAGE_BUDGET * 100, 0, 100);
  const items = list.map(m => `
    <div class="card session-item" data-id="${escapeHtml(m.id)}">
      <div class="grow">
        <h2>${escapeHtml(m.name)}</h2>
        <div class="meta">${escapeHtml(fmtDate(m.createdAt))} · ${fmtDuration(m.durationMs)} · ${m.frameCount} frames · ${fmtBytes(m.bytes || 0)}</div>
      </div>
      <div class="actions">
        <button class="btn btn-sm btn-primary" data-action="open">Open</button>
        <button class="btn btn-sm" data-action="export-json">Export JSON</button>
        <button class="btn btn-sm" data-action="export-csv">CSV</button>
        <button class="btn btn-sm btn-ghost" data-action="delete">Delete</button>
      </div>
    </div>`).join('');
  ui.views.sessions.innerHTML = `
    <div class="sessions-header">
      <div>
        <h2>Sessions</h2>
        <div class="usage"><span class="usage-bar"><span class="usage-fill" style="width:${pct.toFixed(0)}%"></span></span>${fmtBytes(usage)} of about 5 MB browser storage used</div>
      </div>
      <div class="dash-actions">
        <button class="btn" data-action="import">Import…</button>
        <button class="btn btn-primary" data-action="new">New recording</button>
      </div>
    </div>
    ${list.length ? `<div class="session-list">${items}</div>` : `
      <div class="card empty">No sessions yet. Record one from the Live view, or import a session file that someone shared with you.<br>
      Sessions are stored only in this browser (localStorage).</div>`}
  `;
}

ui.views.sessions.addEventListener('click', e => {
  const btn = e.target.closest('[data-action]');
  if (!btn) return;
  const action = btn.dataset.action;
  if (action === 'import') { ui.fileImport.click(); return; }
  if (action === 'new') { setView('live'); return; }
  const id = btn.closest('[data-id]')?.dataset.id;
  if (!id) return;
  if (action === 'delete') {
    const meta = store.list().find(m => m.id === id);
    if (!confirm(`Delete “${meta?.name ?? 'this session'}”? This cannot be undone unless you exported it.`)) return;
    store.remove(id);
    if (state.current?.session.id === id) state.current = null;
    updateSessionCount();
    renderSessions();
    return;
  }
  const session = store.load(id);
  if (!session) { toast('This session could not be read from storage.'); return; }
  if (action === 'open') openSession(session);
  else if (action === 'export-json') exportJSON(session);
  else if (action === 'export-csv') exportCSV(session);
});

// ---------------------------------------------------------------------------------------
// Wiring

function init() {
  applyTheme(currentTheme());
  ui.btnTheme.addEventListener('click', () => {
    const order = ['auto', 'light', 'dark'];
    const next = order[(order.indexOf(currentTheme()) + 1) % order.length];
    localStorage.setItem(THEME_KEY, next);
    applyTheme(next);
    onThemeChanged();
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
    if (currentTheme() === 'auto') onThemeChanged();
  });

  document.querySelectorAll('[data-view]').forEach(el => {
    el.addEventListener('click', e => {
      e.preventDefault();
      const view = el.dataset.view;
      if (view === 'dashboard') return;
      if (view === 'live' && el.classList.contains('brand') && state.current && state.view === 'live') return;
      setView(view);
    });
  });

  ui.btnCamera.addEventListener('click', startCamera);
  ui.btnRecord.addEventListener('click', () => startRecording('button'));
  ui.btnStop.addEventListener('click', () => stopRecording());
  ui.hudRecord.addEventListener('click', () => startRecording('button'));
  ui.hudStop.addEventListener('click', () => stopRecording());
  ui.btnOverlay.addEventListener('click', () => setOverlay(!state.showOverlay));
  setOverlay(state.showOverlay);
  ui.btnFullscreen.addEventListener('click', toggleFullscreen);
  if (!document.fullscreenEnabled && !ui.stage.requestFullscreen) ui.btnFullscreen.hidden = true;
  document.addEventListener('fullscreenchange', () => {
    const fs = Boolean(document.fullscreenElement);
    ui.btnFullscreen.textContent = fs ? '✕' : '⛶';
    ui.btnFullscreen.title = fs ? 'Exit fullscreen (F)' : 'Fullscreen (F)';
  });
  ui.selModel.addEventListener('change', () => {
    if (!state.stream) return;
    ui.btnRecord.disabled = true;
    ui.hudRecord.disabled = true;
    ensureDetector().then(() => { ui.btnRecord.disabled = false; ui.hudRecord.disabled = false; }, err => toast(`Could not load model: ${err.message}`));
  });
  ui.selCamera.addEventListener('change', () => switchCamera(ui.selCamera.value));
  navigator.mediaDevices?.addEventListener?.('devicechange', () => { if (state.stream) refreshCameraList(); });
  ui.btnImport.addEventListener('click', () => ui.fileImport.click());
  ui.fileImport.addEventListener('change', () => importFile(ui.fileImport.files?.[0]));
  ui.video.addEventListener('resize', resizeCanvas);

  document.addEventListener('keydown', e => {
    if (state.view !== 'live' || e.ctrlKey || e.metaKey || e.altKey) return;
    const tag = document.activeElement?.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (e.code === 'KeyF' && state.stream) { e.preventDefault(); toggleFullscreen(); return; }
    if (e.code === 'KeyO') { e.preventDefault(); setOverlay(!state.showOverlay); return; }
    if (e.code !== 'Space' || tag === 'BUTTON' || !state.detector) return;
    e.preventDefault();
    if (state.recording) stopRecording();
    else startRecording('button');
  });
  window.addEventListener('beforeunload', e => {
    if (state.recording) { e.preventDefault(); e.returnValue = ''; }
  });

  if (!navigator.mediaDevices?.getUserMedia) {
    showCameraError(window.isSecureContext
      ? 'This browser does not support camera access.'
      : 'Camera access needs a secure page. Open this app over https:// or from localhost.');
  }
  updateSessionCount();
}

init();
