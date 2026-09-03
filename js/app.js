// Kinesphere: live pose view + recording, analysis dashboard and session library.

import {
  createPoseDetector, createTracker, handsAboveHead, minVisibilityFor, engineInfo, ENGINES, MAX_PEOPLE, FRAME_SIZE,
} from './pose.js';
import {
  store, buildSession, serialize, deserialize, toCSV, download, safeFilename, StorageFullError, personView, frameOffset, personPresent,
} from './session.js';
import { analyze, KINESPHERE_CLASSES, GRID_ROWS, GRID_COLS, MAJOR_NAMES } from './analysis.js';
import { lineChart, stackedBar, heatGrid, trajectoryPlot, bodyFigure, dataTable, heatLegend, ordinalColors, hideTip } from './charts.js';
import { fmtClock, fmtDuration, fmtPct, fmtBytes, fmtDate, escapeHtml, clamp } from './util.js';
import { STYLES, DEFAULT_STYLE, createEffect, personColors, drawPersonBadge } from './effects.js';

const $ = s => document.querySelector(s);
const HOLD_MS = 1500;          // how long both hands must stay up to trigger start/stop
const COUNTDOWN_MS = 3000;     // countdown after the start gesture
const LOCKOUT_MS = 4000;       // ignore the gesture for a while after it fired
const STORE_INTERVAL_MS = 1000 / 30; // store at most 30 frames per second
const STORAGE_BUDGET = 5 * 1024 * 1024;
const THEME_KEY = 'kinesphere:theme';
const OVERLAY_KEY = 'kinesphere:overlay';
const VIDEO_KEY = 'kinesphere:video';
const CAMERA_KEY = 'kinesphere:camera';
const STYLE_KEY = 'kinesphere:style';
const MODEL_KEY = 'kinesphere:model';
const PEOPLE_KEY = 'kinesphere:people';
const CAMERA_TIMEOUT_MS = 15000;

/** Neutral standing pose (MAJOR order, torso units, selfie view) used for the body heat diagram. */
const TEMPLATE_POSE = [
  [0, -1.45], [-0.45, -1.0], [0.45, -1.0], [-0.85, -0.55], [0.85, -0.55], [-1.15, -0.1], [1.15, -0.1],
  [-0.3, 0], [0.3, 0], [-0.35, 1.0], [0.35, 1.0], [-0.4, 2.0], [0.4, 2.0],
];

const ui = {
  video: $('#video'), canvas: $('#overlay'), stage: $('#stage'), placeholder: $('#stage-placeholder'),
  cameraError: $('#camera-error'), cameraStatus: $('#camera-status'), btnCamera: $('#btn-camera'), btnRecord: $('#btn-record'), btnStop: $('#btn-stop'),
  chkGesture: $('#chk-gesture'), selModel: $('#sel-model'), selPeople: $('#sel-people'), selCamera: $('#sel-camera'), cameraWrap: $('#camera-wrap'),
  settingHelp: $('#setting-help'), helpPanel: $('#help-panel'), btnHelp: $('#btn-help'),
  status: $('#status'), fps: $('#fps'),
  recBadge: $('#rec-badge'), recTime: $('#rec-time'), countdown: $('#countdown'),
  hold: $('#hold'), holdFill: $('#hold-fill'), holdLabel: $('#hold-label'),
  views: { live: $('#view-live'), dashboard: $('#view-dashboard'), sessions: $('#view-sessions') },
  sessionCount: $('#session-count'), btnImport: $('#btn-import'), fileImport: $('#file-import'),
  btnTheme: $('#btn-theme'), toast: $('#toast'),
  hudRecord: $('#hud-record'), hudStop: $('#hud-stop'), btnOverlay: $('#btn-overlay'), btnFullscreen: $('#btn-fullscreen'),
  btnStyle: $('#btn-style'), selStyle: $('#sel-style'), btnVideo: $('#btn-video'),
};
const overlayCtx = ui.canvas.getContext('2d');

const state = {
  view: 'live',
  detector: null, detectorPromise: null, detecting: false,
  tracker: createTracker(1), effects: [],
  stream: null, deviceId: null, mirrored: true, loopRunning: false, lastVideoTime: -1,
  recording: null, countdown: null, hold: null, lockoutUntil: 0,
  fps: { count: 0, since: performance.now() },
  lastStatus: '',
  showOverlay: localStorage.getItem(OVERLAY_KEY) !== 'off',
  showVideo: localStorage.getItem(VIDEO_KEY) !== 'off',
  styleId: localStorage.getItem(STYLE_KEY) || DEFAULT_STYLE,
  lastPeople: null, lastPeopleAt: 0, lastRender: 0,
  current: null,        // { session, analyses, person, saved, saveError }
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
  // The camera stage may be fullscreen; leaving it in the top layer over a hidden view
  // would block every click on the dashboard, so always drop out of fullscreen first.
  if (name !== 'live' && document.fullscreenElement) document.exitFullscreen?.().catch(() => {});
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
// Detection settings (model + number of people)

function settings() {
  const info = engineInfo(ui.selModel.value);
  const people = clamp(Number(ui.selPeople.value) || 1, 1, info.maxPeople);
  return { model: info.id, people, info };
}

/** Keep the People selector consistent with the chosen model and explain the implications. */
function syncSettingsUI() {
  const info = engineInfo(ui.selModel.value);
  const wanted = Number(ui.selPeople.value) || 1;
  if (wanted > info.maxPeople) ui.selPeople.value = String(info.maxPeople);
  ui.selPeople.disabled = info.maxPeople === 1 || Boolean(state.recording);
  const people = Number(ui.selPeople.value) || 1;
  let peopleNote;
  if (info.maxPeople === 1) peopleNote = 'This model tracks one person only; pick MediaPipe or MoveNet MultiPose to track a group.';
  else if (people > 1) peopleNote = `Tracking up to ${people} people: expect a lower frame rate, one dashboard tab per person, and pose control reacting to anyone's raised hands.`;
  else peopleNote = 'Tracking one person.';
  ui.settingHelp.textContent = `${info.help} ${peopleNote}`;
  localStorage.setItem(MODEL_KEY, info.id);
  localStorage.setItem(PEOPLE_KEY, String(people));
}

function onSettingsChanged() {
  syncSettingsUI();
  if (!state.stream) return;
  ui.btnRecord.disabled = true;
  ui.hudRecord.disabled = true;
  ensureDetector().then(
    () => { ui.btnRecord.disabled = false; ui.hudRecord.disabled = false; },
    err => { console.error(err); toast(`Could not load the model: ${err.message}`, 6000); setStatus(`Could not load the model: ${err.message}`); },
  );
}

// ---------------------------------------------------------------------------------------
// Camera + detection loop

function showCameraError(msg) {
  ui.cameraError.textContent = msg;
  ui.cameraError.hidden = false;
}

function withTimeout(promise, ms, message) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(Object.assign(new Error(message), { name: 'TimeoutError' })), ms);
  });
  return Promise.race([promise.finally(() => clearTimeout(timer)), timeout]);
}

/**
 * Open a camera stream, by device id when given, otherwise the default front-facing camera.
 * Some platforms never settle getUserMedia while another app holds the camera, so the
 * request is given a deadline; a stream that arrives after we gave up is stopped again.
 */
function openStream(deviceId) {
  const video = { width: { ideal: 1280 }, height: { ideal: 720 } };
  if (deviceId) video.deviceId = { exact: deviceId };
  else video.facingMode = 'user';
  const request = navigator.mediaDevices.getUserMedia({ video, audio: false });
  const guarded = withTimeout(request, CAMERA_TIMEOUT_MS,
    'The camera did not respond. It is probably in use by another app or browser tab: close that and try again.');
  guarded.catch(() => request.then(s => { if (state.stream !== s) s.getTracks().forEach(t => t.stop()); }, () => {}));
  return guarded;
}

function cameraErrorMessage(err) {
  switch (err?.name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
      return 'Camera access was denied. Allow the camera for this site in your browser settings, then try again.';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return 'No camera was found on this device.';
    case 'NotReadableError':
    case 'TrackStartError':
    case 'AbortError':
      return 'The camera could not be started because another app or browser tab is using it. Close that and try again.';
    case 'TimeoutError':
      return err.message;
    case 'SecurityError':
      return 'Camera access is blocked here. The page must be opened over https:// or from localhost.';
    default:
      return `Camera error: ${err?.message || err?.name || err}`;
  }
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
  const settingsOf = stream.getVideoTracks()[0]?.getSettings?.() ?? {};
  state.deviceId = requestedDeviceId || settingsOf.deviceId || null;
  // Rear (environment-facing) cameras show the scene as it is; everything else is a mirror.
  state.mirrored = settingsOf.facingMode !== 'environment';
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
    toast(`Could not switch camera. ${cameraErrorMessage(err)}`, 6000);
    if (state.deviceId) ui.selCamera.value = state.deviceId;
  } finally {
    ui.selCamera.disabled = false;
  }
}

async function startCamera() {
  ui.btnCamera.disabled = true;
  ui.btnCamera.textContent = 'Starting camera…';
  ui.cameraError.hidden = true;
  ui.cameraStatus.textContent = 'Waiting for the camera. If nothing happens, another app or tab may be using it.';
  ui.cameraStatus.hidden = false;
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('This browser cannot access the camera here. Camera access needs a secure (https) page or localhost.');
    }
    const preferred = localStorage.getItem(CAMERA_KEY);
    let stream;
    try {
      stream = await openStream(preferred);
    } catch (err) {
      // The remembered camera is gone or refuses these constraints: fall back to the default one.
      if (!preferred || !['OverconstrainedError', 'NotFoundError', 'ConstraintNotSatisfiedError'].includes(err?.name)) throw err;
      localStorage.removeItem(CAMERA_KEY);
      stream = await openStream(null);
    }
    await attachStream(stream);
  } catch (err) {
    console.warn('Camera failed to start', err);
    showCameraError(cameraErrorMessage(err));
    ui.cameraStatus.hidden = true;
    ui.btnCamera.disabled = false;
    ui.btnCamera.textContent = 'Try again';
    return;
  }
  ui.cameraStatus.hidden = true;
  ui.stage.classList.remove('is-idle');
  ui.placeholder.hidden = true;
  try {
    await ensureDetector();
  } catch (err) {
    console.error(err);
    setStatus(`Could not load the pose model: ${err.message}`);
    toast('The pose model could not be loaded. Check your network connection and reload.', 6000);
    return;
  }
  ui.btnRecord.disabled = false;
  ui.hudRecord.disabled = false;
  startLoop();
}

/** Show or hide the camera feed. The stream keeps running so detection and recording continue. */
function setVideo(on) {
  state.showVideo = on;
  localStorage.setItem(VIDEO_KEY, on ? 'on' : 'off');
  ui.stage.classList.toggle('no-video', !on);
  ui.btnVideo.textContent = on ? 'Video: on' : 'Video: off';
  ui.btnVideo.setAttribute('aria-pressed', String(on));
}

function setOverlay(on) {
  state.showOverlay = on;
  localStorage.setItem(OVERLAY_KEY, on ? 'on' : 'off');
  ui.btnOverlay.textContent = on ? 'Overlay: on' : 'Overlay: off';
  ui.btnOverlay.setAttribute('aria-pressed', String(on));
  if (!on) overlayCtx.clearRect(0, 0, ui.canvas.width, ui.canvas.height);
}

function rebuildEffects() {
  const n = state.detector?.people ?? 1;
  state.effects = Array.from({ length: n }, () => createEffect(state.styleId));
}

function setStyle(id) {
  state.styleId = createEffect(id).id;
  localStorage.setItem(STYLE_KEY, state.styleId);
  ui.selStyle.value = state.styleId;
  ui.btnStyle.textContent = `Style: ${STYLES.find(st => st.id === state.styleId)?.label ?? state.styleId}`;
  rebuildEffects();
  if (!state.showOverlay) setOverlay(true);
}

function nextStyle() {
  const i = STYLES.findIndex(st => st.id === state.styleId);
  setStyle(STYLES[(i + 1) % STYLES.length].id);
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

/** Create (or re-create) the detector for the current model and people settings. */
function ensureDetector() {
  const { model, people } = settings();
  if (state.detector && state.detector.model === model && state.detector.people === people) return Promise.resolve(state.detector);
  if (state.detectorPromise) return state.detectorPromise;
  const old = state.detector;
  state.detector = null;
  state.lastPeople = null;
  state.detectorPromise = createPoseDetector({ model, people, onStatus: setStatus }).then(
    d => {
      state.detector = d;
      state.detectorPromise = null;
      state.tracker = createTracker(d.people);
      rebuildEffects();
      old?.close();
      // settings may have changed again while this one was loading
      const now = settings();
      if (now.model !== d.model || now.people !== d.people) return ensureDetector();
      return d;
    },
    err => { state.detectorPromise = null; if (old) state.detector = old; throw err; },
  );
  return state.detectorPromise;
}

function startLoop() {
  if (state.loopRunning) return;
  state.loopRunning = true;
  requestAnimationFrame(loop);
}

function loop(now) {
  if (!state.stream || (state.view !== 'live' && !state.recording)) {
    state.loopRunning = false;
    return;
  }
  const v = ui.video;
  if (state.detector && !state.detecting && v.readyState >= 2 && v.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = v.currentTime;
    resizeCanvas();
    const ts = performance.now();
    const detector = state.detector;
    state.detecting = true;
    let result;
    try { result = detector.detect(v, ts); } catch (err) { console.error(err); result = []; }
    Promise.resolve(result).then(
      poses => {
        if (state.detector !== detector) return; // settings changed mid-flight
        const slots = state.tracker.update(poses ?? [], ts);
        state.lastPeople = slots;
        state.lastPeopleAt = ts;
        onDetection(slots, ts);
      },
      err => console.error(err),
    ).finally(() => { state.detecting = false; });
  }
  renderOverlay(now);
  requestAnimationFrame(loop);
}

/** Draw the overlay every animation frame so particle/trail styles stay smooth between detections. */
function renderOverlay(now) {
  const dt = state.lastRender ? clamp((now - state.lastRender) / 1000, 0, 0.1) : 1 / 60;
  state.lastRender = now;
  const { width, height } = ui.canvas;
  overlayCtx.clearRect(0, 0, width, height);
  if (!state.showOverlay || !state.detector) return;
  const slots = state.lastPeople && now - state.lastPeopleAt < 600 ? state.lastPeople : null;
  const minVis = minVisibilityFor(state.detector.engine);
  state.effects.forEach((effect, p) => effect.draw(overlayCtx, slots?.[p] ?? null, 0, width, height, dt, { person: p, minVis }));
  if (slots && state.detector.people > 1) {
    slots.forEach((lm, p) => { if (lm) drawPersonBadge(overlayCtx, lm, 0, width, height, p, { mirrorText: state.mirrored, minVis }); });
  }
}

function onDetection(slots, now) {
  const present = slots.filter(Boolean);
  state.fps.count++;
  if (now - state.fps.since >= 1000) {
    const fps = state.fps.count * 1000 / (now - state.fps.since);
    state.fps.count = 0;
    state.fps.since = now;
    const d = state.detector;
    const who = d.people > 1 ? ` · ${present.length}/${d.people} people` : '';
    ui.fps.textContent = `${fps.toFixed(0)} fps · ${d.label} · ${d.delegate}${who}`;
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
    if (present.length && now - rec.lastStored >= STORE_INTERVAL_MS - 1) {
      const frame = new Float32Array(rec.people * FRAME_SIZE);
      slots.forEach((lm, p) => { if (lm && p < rec.people) frame.set(lm, p * FRAME_SIZE); });
      rec.frames.push({ t: now - rec.startedAt, lm: frame });
      rec.lastStored = now;
    }
    ui.recTime.textContent = fmtClock(now - rec.startedAt);
  }

  updateGesture(present, now);
  updateStatus(present.length);
}

function updateGesture(present, now) {
  const minVis = minVisibilityFor(state.detector?.engine);
  const active = ui.chkGesture.checked && !state.countdown && now >= state.lockoutUntil && present.some(lm => handsAboveHead(lm, 0, minVis));
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

function updateStatus(count) {
  const people = state.detector?.people ?? 1;
  const who = people > 1 ? (count === 1 ? '1 person detected.' : `${count} people detected.`) : 'Pose detected.';
  let msg;
  if (state.countdown) msg = 'Get ready…';
  else if (state.recording) msg = 'Recording. Press Stop, or raise both hands above your head to finish.';
  else if (!count) msg = people > 1 ? 'Nobody detected yet. Step back so whole bodies are visible.' : 'No pose detected. Step back so your whole body is visible.';
  else if (ui.chkGesture.checked) msg = `${who} Press Record, or raise both hands above your head to start.`;
  else msg = `${who} Press Record to start.`;
  setStatus(msg);
}

// ---------------------------------------------------------------------------------------
// Recording

function startRecording(trigger, now = performance.now()) {
  if (state.recording || !state.detector) return;
  state.countdown = null;
  ui.countdown.hidden = true;
  state.recording = { startedAt: now, wallStart: Date.now(), lastStored: -Infinity, frames: [], trigger, people: state.detector.people };
  ui.stage.classList.add('is-recording');
  ui.recBadge.hidden = false;
  ui.recTime.textContent = '0:00';
  ui.btnRecord.hidden = true;
  ui.btnStop.hidden = false;
  ui.hudRecord.hidden = true;
  ui.hudStop.hidden = false;
  ui.selModel.disabled = true;
  ui.selPeople.disabled = true;
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
  syncSettingsUI();
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
    engine: state.detector?.engine,
    people: rec.people,
    trigger: rec.trigger,
    startedAt: rec.wallStart,
    mirrored: state.mirrored,
  });
  openSession(session, { save: true });
}

// ---------------------------------------------------------------------------------------
// Sessions

function openSession(session, { save = false } = {}) {
  const analyses = Array.from({ length: session.people || 1 }, (_, p) => analyze(personView(session, p)));
  const person = Math.max(0, analyses.findIndex(a => a.ok));
  state.current = { session, analyses, person, saved: false, saveError: null };
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

function describeSession(s) {
  const info = ENGINES.find(e => e.id === s.model);
  const modelText = info ? info.label : `${escapeHtml(s.model)} model`;
  const people = (s.people || 1) > 1 ? ` · ${s.people} people` : '';
  return `${modelText}${people}`;
}

// ---------------------------------------------------------------------------------------
// Dashboard

function stat(label, value, unit = '') {
  return `<div class="stat"><div class="stat-label">${escapeHtml(label)}</div><div class="stat-value">${value}${unit ? `<small>${escapeHtml(unit)}</small>` : ''}</div></div>`;
}

function renderDashboard() {
  const { session, analyses, person, saved, saveError } = state.current;
  const a = analyses[person];
  for (const c of state.charts) c.destroy?.();
  state.charts = [];
  state.replay?.destroy();
  state.replay = null;

  const people = session.people || 1;
  const fpsText = a.ok ? ` · ${a.fps.toFixed(0)} fps` : '';
  const triggerText = session.trigger === 'pose' ? 'started by pose' : session.trigger === 'button' ? 'started by button' : '';
  const tabs = people > 1 ? `
    <div class="person-tabs" role="tablist" aria-label="Person">
      ${analyses.map((an, p) => `<button class="tab${p === person ? ' is-active' : ''}" role="tab" data-person="${p}" aria-selected="${p === person}">
        <i class="swatch" style="background:${personColors(p).left}"></i>Person ${p + 1}
        <small>${an.ok ? `${fmtPct(an.validFraction)} tracked` : 'no data'}</small></button>`).join('')}
    </div>` : '';
  const root = ui.views.dashboard;
  root.innerHTML = `
    <div class="dash-header">
      <div>
        <input id="session-name" class="name-input" type="text" value="${escapeHtml(session.name)}" aria-label="Session name" spellcheck="false">
        <div class="meta">${escapeHtml(fmtDate(session.createdAt))} · ${fmtDuration(session.durationMs)} · ${session.frameCount} frames${fpsText} · ${describeSession(session)}${triggerText ? ` · ${triggerText}` : ''}</div>
      </div>
      <div class="dash-actions">
        <button class="btn" data-action="export-json">Export JSON</button>
        <button class="btn" data-action="export-csv">Export CSV</button>
        <button class="btn btn-ghost" data-action="delete">Delete</button>
        <button class="btn btn-primary" data-action="new">New recording</button>
      </div>
    </div>
    ${saveError ? `<div class="banner banner-warn"><b>Not saved.</b> ${escapeHtml(saveError)}</div>` : ''}
    <section class="card">
      <h2>Replay</h2>
      <p class="card-sub">The recorded pose${people > 1 ? 's' : ''}, as you saw ${people > 1 ? 'them' : 'it'} on screen. Scrub to move the cursor on the charts below.</p>
      <div class="replay"><canvas id="replay-canvas"></canvas></div>
      <div class="replay-controls">
        <button class="btn" id="replay-play" aria-label="Play">▶</button>
        <input type="range" id="replay-range" min="0" max="${Math.max(1, session.durationMs)}" value="0" step="1" aria-label="Playback position">
        <span id="replay-time" class="mono">0:00 / ${fmtClock(session.durationMs)}</span>
      </div>
    </section>
    ${tabs}
    ${a.ok ? dashboardCards(a) : `<div class="banner">Not enough pose data to analyse ${people > 1 ? `person ${person + 1}` : 'this session'}. ${escapeHtml(a.reason)}</div>`}
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
  root.querySelectorAll('.person-tabs .tab').forEach(tab => {
    tab.onclick = () => { state.current.person = Number(tab.dataset.person); renderDashboard(); };
  });

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
  const people = session.people || 1;
  const effects = Array.from({ length: people }, () => createEffect(state.styleId));
  const minVis = minVisibilityFor(session.engine);
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
  const draw = (dt = 1 / 30) => {
    ctx.clearRect(0, 0, w, h);
    const i = frameAt(t);
    const hasFrame = i >= 0 && t - times[i] < 500;
    for (let p = 0; p < people; p++) {
      const offset = frameOffset(session, Math.max(0, i), p);
      const visible = hasFrame && personPresent(lm, offset);
      effects[p].draw(ctx, visible ? lm : null, offset, w, h, dt, { mirror: session.mirrored, minVis, person: p });
      if (visible && people > 1) drawPersonBadge(ctx, lm, offset, w, h, p, { mirror: session.mirrored, minVis });
    }
  };
  const setTime = (nt, dt) => {
    t = clamp(nt, 0, durationMs);
    range.value = String(Math.round(t));
    timeEl.textContent = `${fmtClock(t)} / ${fmtClock(durationMs)}`;
    draw(dt);
    onTime(t);
  };
  const tick = now => {
    if (!playing) return;
    setTime(t + (now - last), clamp((now - last) / 1000, 0, 0.1));
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
  range.oninput = () => { effects.forEach(e => e.reset()); setTime(Number(range.value)); };
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
        <div class="meta">${escapeHtml(fmtDate(m.createdAt))} · ${fmtDuration(m.durationMs)} · ${m.frameCount} frames${m.model ? ` · ${describeSession(m)}` : ''} · ${fmtBytes(m.bytes || 0)}</div>
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

  // Detection settings
  ui.selModel.replaceChildren(...ENGINES.map(e => {
    const opt = document.createElement('option');
    opt.value = e.id;
    opt.textContent = e.label;
    opt.title = e.help;
    return opt;
  }));
  ui.selPeople.replaceChildren(...Array.from({ length: MAX_PEOPLE }, (_, i) => {
    const opt = document.createElement('option');
    opt.value = String(i + 1);
    opt.textContent = i === 0 ? '1 person' : `${i + 1} people`;
    return opt;
  }));
  const savedModel = localStorage.getItem(MODEL_KEY);
  if (savedModel && ENGINES.some(e => e.id === savedModel)) ui.selModel.value = savedModel;
  const savedPeople = Number(localStorage.getItem(PEOPLE_KEY));
  if (savedPeople >= 1 && savedPeople <= MAX_PEOPLE) ui.selPeople.value = String(savedPeople);
  syncSettingsUI();
  ui.selModel.addEventListener('change', onSettingsChanged);
  ui.selPeople.addEventListener('change', onSettingsChanged);
  ui.btnHelp.addEventListener('click', () => {
    ui.helpPanel.open = !ui.helpPanel.open;
    ui.btnHelp.setAttribute('aria-expanded', String(ui.helpPanel.open));
    if (ui.helpPanel.open) ui.helpPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });
  ui.helpPanel.addEventListener('toggle', () => ui.btnHelp.setAttribute('aria-expanded', String(ui.helpPanel.open)));

  ui.btnCamera.addEventListener('click', startCamera);
  ui.btnRecord.addEventListener('click', () => startRecording('button'));
  ui.btnStop.addEventListener('click', () => stopRecording());
  ui.hudRecord.addEventListener('click', () => startRecording('button'));
  ui.hudStop.addEventListener('click', () => stopRecording());
  ui.btnOverlay.addEventListener('click', () => setOverlay(!state.showOverlay));
  setOverlay(state.showOverlay);
  ui.selStyle.replaceChildren(...STYLES.map(st => {
    const opt = document.createElement('option');
    opt.value = st.id;
    opt.textContent = st.label;
    return opt;
  }));
  ui.selStyle.addEventListener('change', () => setStyle(ui.selStyle.value));
  ui.btnStyle.addEventListener('click', nextStyle);
  setStyle(state.styleId);
  ui.btnVideo.addEventListener('click', () => setVideo(!state.showVideo));
  setVideo(state.showVideo);
  ui.btnFullscreen.addEventListener('click', toggleFullscreen);
  if (!document.fullscreenEnabled && !ui.stage.requestFullscreen) ui.btnFullscreen.hidden = true;
  document.addEventListener('fullscreenchange', () => {
    const fs = Boolean(document.fullscreenElement);
    ui.btnFullscreen.textContent = fs ? '✕' : '⛶';
    ui.btnFullscreen.title = fs ? 'Exit fullscreen (F)' : 'Fullscreen (F)';
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
    if (e.code === 'KeyS') { e.preventDefault(); nextStyle(); return; }
    if (e.code === 'KeyV') { e.preventDefault(); setVideo(!state.showVideo); return; }
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
