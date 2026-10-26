// Kinesphere: live pose view + recording, analysis dashboard and session library.

import {
  createPoseDetector, createTracker, handsAboveHead, bodyInFrame, minVisibilityFor, engineInfo, ENGINES, MAX_PEOPLE, FRAME_SIZE, HAND_SIZE, WORLD_SIZE,
  modelCacheInfo, clearModelCache,
} from './pose.js';
import {
  store, buildSession, serialize, deserialize, toCSV, download, safeFilename, StorageFullError, personView, frameOffset, handOffset, personPresent,
} from './session.js';
import { analyze, KINESPHERE_CLASSES, LEVELS, MAJOR_NAMES, LEFT_SLOTS, RIGHT_SLOTS, UPPER_SLOTS, LOWER_SLOTS } from './analysis.js';
import { lineChart, stackedBar, floorPlot, bodyFigure, sideFigure, upperLowerFigure, levelsFigure, dataTable, heatLegend, ordinalColors, hideTip } from './charts.js';
import { fmtClock, fmtDuration, fmtPct, fmtBytes, fmtDate, escapeHtml, clamp } from './util.js';
import { STYLES, DEFAULT_STYLE, createEffect, personColors, drawPersonBadge, SAND_DEFAULTS, sandSettings, setSandSettings } from './effects.js';
import { SMOOTHING_LEVELS, DEFAULT_SMOOTHING, createSmoother } from './smoothing.js';
import { createViewer3D, xrSupport, estimateBodyScale, loadThree, DEFAULT_DEPTH } from './viewer3d.js';
import { createGame } from './game.js';
import { parseCommand, createVoiceControl, voiceSupported, VOICE_HELP } from './voice.js';
import { SONGS } from './game-core.js';

const $ = s => document.querySelector(s);
const HOLD_MS = 1500;          // how long both hands must stay up to trigger start/stop
const COUNTDOWN_MS = 3000;     // countdown after the start gesture
const LOCKOUT_MS = 4000;       // ignore the gesture for a while after it fired
const STORE_INTERVAL_MS = 1000 / 30; // store at most 30 frames per second
const THEME_KEY = 'kinesphere:theme';
const OVERLAY_KEY = 'kinesphere:overlay';
const VIDEO_KEY = 'kinesphere:video';
const CAMERA_KEY = 'kinesphere:camera';
const STYLE_KEY = 'kinesphere:style';
const MODEL_KEY = 'kinesphere:model';
const PEOPLE_KEY = 'kinesphere:people';
const SMOOTH_KEY = 'kinesphere:smoothing';
const DEPTH_KEY = 'kinesphere:depth';
const MIRROR3D_KEY = 'kinesphere:mirror3d';
const OVERLAY_METRICS_KEY = 'kinesphere:replayOverlay';
const SHOW_VALUES_KEY = 'kinesphere:showValues';
const SPEED_VALUES_KEY = 'kinesphere:showSpeedValues';
const SAND_KEY = 'kinesphere:sand';
const HELP_KEY = 'kinesphere:dashHelp';
const VOICE_CONSENT_KEY = 'kinesphere:voiceConsent';
const CAMERA_TIMEOUT_MS = 15000;

/** Neutral standing pose (MAJOR order, torso units, selfie view) used for the body heat diagram. */
const TEMPLATE_POSE = [
  [0, -1.45], [-0.45, -1.0], [0.45, -1.0], [-0.85, -0.55], [0.85, -0.55], [-1.15, -0.1], [1.15, -0.1],
  [-0.3, 0], [0.3, 0], [-0.35, 1.0], [0.35, 1.0], [-0.4, 2.0], [0.4, 2.0],
];

const ui = {
  video: $('#video'), canvas: $('#overlay'), stage: $('#stage'), placeholder: $('#stage-placeholder'),
  cameraError: $('#camera-error'), cameraStatus: $('#camera-status'), btnCamera: $('#btn-camera'), btnRecord: $('#btn-record'), btnStop: $('#btn-stop'),
  chkGesture: $('#chk-gesture'), chkMirror: $('#chk-mirror'), selModel: $('#sel-model'), selPeople: $('#sel-people'), selCamera: $('#sel-camera'), cameraWrap: $('#camera-wrap'),
  settingHelp: $('#setting-help'), helpPanel: $('#help-panel'), btnHelp: $('#btn-help'),
  cacheInfo: $('#cache-info'), btnClearCache: $('#btn-clear-cache'),
  status: $('#status'), fps: $('#fps'),
  recBadge: $('#rec-badge'), recTime: $('#rec-time'), countdown: $('#countdown'),
  hold: $('#hold'), holdFill: $('#hold-fill'), holdLabel: $('#hold-label'),
  frameWarn: $('#frame-warn'), load: $('#load'), loadFill: $('#load-fill'), loadLabel: $('#load-label'),
  views: { live: $('#view-live'), dashboard: $('#view-dashboard'), sessions: $('#view-sessions') },
  sessionCount: $('#session-count'), btnImport: $('#btn-import'), fileImport: $('#file-import'),
  btnTheme: $('#btn-theme'), toast: $('#toast'),
  hudRecord: $('#hud-record'), hudStop: $('#hud-stop'), btnOverlay: $('#btn-overlay'), btnFullscreen: $('#btn-fullscreen'),
  btnVoice: $('#btn-voice'), btnVoiceHud: $('#btn-voice-hud'), voiceChip: $('#voice-chip'), voiceHelp: $('#voice-help'),
  sandControls: $('#sand-controls'), sandHue: $('#sand-hue'), sandGravity: $('#sand-gravity'), sandGravityVal: $('#sand-gravity-val'),
  sandDecay: $('#sand-decay'), sandDecayVal: $('#sand-decay-val'), sandMomentum: $('#sand-momentum'), sandMomentumVal: $('#sand-momentum-val'), sandReset: $('#sand-reset'),
  btnStyle: $('#btn-style'), selStyle: $('#sel-style'), btnVideo: $('#btn-video'), selSmooth: $('#sel-smooth'),
  live3dBar: $('#live-3d-bar'), live3d: $('#live-3d'), liveDepth: $('#live-depth'), liveDepthVal: $('#live-depth-val'), btnLive3d: $('#btn-live-3d'), btnLiveVr: $('#btn-live-vr'), btnLiveAr: $('#btn-live-ar'), liveVrNote: $('#live-vr-note'),
};
const overlayCtx = ui.canvas.getContext('2d');

const state = {
  view: 'live',
  detector: null, detectorPromise: null, detecting: false,
  tracker: createTracker(1), effects: [],
  stream: null, deviceId: null, mirrored: true, facing: 'user', loopRunning: false, lastVideoTime: -1,
  recording: null, countdown: null, hold: null, lockoutUntil: 0,
  framing: { bad: false, since: 0 }, // "body partially out of frame" hint, debounced
  fps: { count: 0, since: performance.now() },
  lastStatus: '',
  showOverlay: localStorage.getItem(OVERLAY_KEY) !== 'off',
  showVideo: localStorage.getItem(VIDEO_KEY) !== 'off',
  styleId: localStorage.getItem(STYLE_KEY) || DEFAULT_STYLE,
  smoother: createSmoother(localStorage.getItem(SMOOTH_KEY) || DEFAULT_SMOOTHING, MAX_PEOPLE),
  handSmoother: createSmoother(localStorage.getItem(SMOOTH_KEY) || DEFAULT_SMOOTHING, MAX_PEOPLE, HAND_SIZE),
  lastPeople: null, lastPeopleAt: 0, lastRender: 0,
  current: null,        // { session, analyses, person, saved, saveError, compare: { a, b } | null }
  compareCache: new Map(), compareToken: 0, compareChart: null,
  charts: [], replay: null,
  liveViewer: null, liveViewerPromise: null, liveBody: { torso: 0, floor: 0 },
  xr: { vr: false, ar: false, checked: false },
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

/** Progress bar under the status while model files download; `progress` is { loaded, total } or null. */
function showLoadProgress(progress) {
  if (!progress || !(progress.total > 0)) { ui.load.hidden = true; return; }
  const frac = clamp(progress.loaded / progress.total, 0, 1);
  const mb = b => (b / 1e6).toFixed(1);
  ui.load.hidden = false;
  ui.loadFill.style.width = `${(frac * 100).toFixed(1)}%`;
  ui.loadLabel.textContent = `${mb(progress.loaded)} / ${mb(progress.total)} MB`;
  ui.load.setAttribute('aria-valuenow', String(Math.round(frac * 100)));
}

function onLoadStatus(msg, progress = null) {
  setStatus(msg);
  showLoadProgress(progress);
}

function setView(name) {
  // The game is played on the live stage, so it is the live view with the game layer open.
  const view = name === 'game' ? 'live' : name;
  // The camera stage may be fullscreen; leaving it in the top layer over a hidden view
  // would block every click on the dashboard, so always drop out of fullscreen first.
  if (view !== 'live' && document.fullscreenElement) document.exitFullscreen?.().catch(() => {});
  state.view = view;
  for (const [k, el] of Object.entries(ui.views)) el.hidden = k !== view;
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.toggle('is-active', b.dataset.view === name));
  hideTip();
  if (name === 'game') {
    state.countdown = null;
    ui.countdown.hidden = true;
    if (state.hold) { state.hold = null; ui.hold.hidden = true; }
    if (state.recording && stopRecording(null, { navigate: false })) toast('Your recording was saved to Sessions.');
    ui.views.live.classList.add('in-game');
    game.open();
  } else {
    ui.views.live.classList.remove('in-game');
    game.close();
  }
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
  ui.selPeople.disabled = Boolean(state.recording);
  const people = Number(ui.selPeople.value) || 1;
  let peopleNote;
  if (info.maxPeople === 1) peopleNote = 'This model tracks one person only; choosing more people switches to MoveNet MultiPose.';
  else if (people > 1) peopleNote = `Tracking up to ${people} people: expect a lower frame rate, one dashboard tab per person, and pose control reacting to anyone's raised hands.`;
  else peopleNote = 'Tracking one person.';
  ui.settingHelp.textContent = `${info.help} ${peopleNote}`;
  localStorage.setItem(MODEL_KEY, info.id);
  localStorage.setItem(PEOPLE_KEY, String(people));
}

function renderModelMatrix() {
  const yes = '✓', no = '–';
  const rows = ENGINES.map(e => `<tr><td>${escapeHtml(e.label)}</td><td>${escapeHtml(e.runtime)}</td><td class="num">${e.landmarks}</td>
    <td>${e.depth ? yes : no}</td><td>${e.hands ? yes : no}</td><td>${e.face === 'mesh' ? 'mesh' : '11 points'}</td>
    <td class="num">${e.maxPeople === 1 ? '1' : `up to ${e.maxPeople}`}</td><td>${e.maxPeople > 1 && e.engine === 'movenet' ? 'built-in' : e.maxPeople > 1 ? 'by position' : no}</td>
    <td>${escapeHtml(e.speed)}</td><td class="num">${escapeHtml(e.download)}</td></tr>`).join('');
  $('#model-matrix').innerHTML = `<thead><tr><th>Model</th><th>Runtime</th><th class="num">Landmarks</th><th>Depth (z)</th><th>Hands</th><th>Face</th><th class="num">People</th><th>Identity tracking</th><th>Speed</th><th class="num">Download</th></tr></thead><tbody>${rows}</tbody>`;
}

async function refreshCacheInfo() {
  const info = await modelCacheInfo();
  if (!info.supported) {
    ui.cacheInfo.textContent = 'This browser cannot keep downloaded models between visits here (it needs a secure https page), so they are fetched each time.';
    ui.btnClearCache.hidden = true;
    return;
  }
  ui.cacheInfo.textContent = info.files
    ? `Downloaded models are kept in this browser so they load instantly next time: ${info.files} file${info.files === 1 ? '' : 's'}, ${fmtBytes(info.bytes)}.`
    : 'Downloaded models are kept in this browser so they load instantly next time. Nothing is cached yet.';
  ui.btnClearCache.hidden = !info.files;
}

const MULTI_MODEL = 'movenet-multipose';
/**
 * Choosing several people switches to MoveNet MultiPose: its built-in tracker keeps identities
 * stable, where the other models only tell people apart by position and mix them up when they cross.
 */
function onPeopleChanged() {
  const people = Number(ui.selPeople.value) || 1;
  if (people > 1 && ui.selModel.value !== MULTI_MODEL) {
    ui.selModel.value = MULTI_MODEL;
    toast('Switched the model to MoveNet MultiPose, the best at keeping people apart. It has no 3D depth or hands; pick a MediaPipe model under Model if you need those.', 8000);
  }
  onSettingsChanged();
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
  if (game.isOpen) toast(msg, 8000); // the game covers the intro panel that shows the error
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
  // Rear (environment-facing) cameras show the scene as it is; everything else is a mirror,
  // unless the user chose otherwise for this kind of camera.
  state.facing = settingsOf.facingMode === 'environment' ? 'environment' : 'user';
  const stored = localStorage.getItem(mirrorKeyFor(state.facing));
  applyLiveMirror(stored ? stored === 'on' : state.facing !== 'environment', { remember: false });
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
  ui.live3dBar.hidden = false;
  wireXrButtons(ui.btnLiveVr, ui.btnLiveAr, ui.liveVrNote, ensureLiveViewer);
  startLoop();
  game.refresh();
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

function setSmoothing(id) {
  state.smoother = createSmoother(id, MAX_PEOPLE);
  state.handSmoother = createSmoother(state.smoother.id, MAX_PEOPLE, HAND_SIZE);
  localStorage.setItem(SMOOTH_KEY, state.smoother.id);
  ui.selSmooth.value = state.smoother.id;
  ui.selSmooth.title = `${SMOOTHING_LEVELS.find(l => l.id === state.smoother.id)?.help ?? ''} Display only: recordings keep the raw landmarks.`;
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
  ui.sandControls.hidden = state.styleId !== 'sandman';
  if (!state.showOverlay) setOverlay(true);
}

/** Sandman particle sliders: applied live (every sandman effect reads them each frame) and remembered. */
function syncSandUI() {
  ui.sandHue.value = String(sandSettings.hue);
  ui.sandGravity.value = String(sandSettings.gravity);
  ui.sandDecay.value = String(sandSettings.decay);
  ui.sandMomentum.value = String(sandSettings.momentum);
  ui.sandMomentumVal.textContent = `${sandSettings.momentum.toFixed(1)}×`;
  ui.sandGravityVal.textContent = `${sandSettings.gravity.toFixed(1)}×`;
  ui.sandDecayVal.textContent = `${sandSettings.decay.toFixed(1)} s`;
  ui.sandControls.style.setProperty('--sand', `hsl(${sandSettings.hue}, 70%, 65%)`);
}
/** Change sandman settings (clamped to the sliders' ranges), remember them and update the sliders. */
const SAND_RANGES = { hue: [0, 360], gravity: [-1, 3], decay: [0.3, 8], momentum: [0, 2] };
function applySand(patch) {
  const clean = {};
  for (const [k, v] of Object.entries(patch)) if (SAND_RANGES[k] && Number.isFinite(v)) clean[k] = clamp(v, ...SAND_RANGES[k]);
  setSandSettings(clean);
  localStorage.setItem(SAND_KEY, JSON.stringify(sandSettings));
  syncSandUI();
}

function initSandControls() {
  try { setSandSettings(JSON.parse(localStorage.getItem(SAND_KEY)) ?? {}); } catch { /* defaults */ }
  const update = () => applySand({ hue: Number(ui.sandHue.value), gravity: Number(ui.sandGravity.value), decay: Number(ui.sandDecay.value), momentum: Number(ui.sandMomentum.value) });
  for (const el of [ui.sandHue, ui.sandGravity, ui.sandDecay, ui.sandMomentum]) el.addEventListener('input', update);
  ui.sandReset.addEventListener('click', () => { setSandSettings(SAND_DEFAULTS); localStorage.removeItem(SAND_KEY); syncSandUI(); });
  syncSandUI();
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

// ---------------------------------------------------------------------------------------
// 3D view and WebXR

/** Depth exaggeration for the 3D figure (0..1.5 of the model's z estimate), remembered across visits. */
function currentDepth() {
  const v = Number(localStorage.getItem(DEPTH_KEY));
  return Number.isFinite(v) && localStorage.getItem(DEPTH_KEY) !== null ? Math.max(0, Math.min(1.5, v)) : DEFAULT_DEPTH;
}
function setDepth(v) {
  localStorage.setItem(DEPTH_KEY, String(v));
  state.liveViewer?.setScale({ depth: v });
  state.replay?.setDepth?.(v);
  document.querySelectorAll('[data-depth-slider]').forEach(el => { el.value = String(Math.round(v * 100)); });
  document.querySelectorAll('[data-depth-value]').forEach(el => { el.textContent = `${Math.round(v * 100)}%`; });
}
/**
 * Mirror view for the live stage: flips the camera feed, overlay and 3D figure together.
 * Defaults to mirrored for front cameras and un-mirrored for rear ones; an explicit choice
 * is remembered per camera facing.
 */
function mirrorKeyFor(facing) { return `${MIRROR3D_KEY}:${facing === 'environment' ? 'rear' : 'front'}`; }
function applyLiveMirror(on, { remember = true } = {}) {
  state.mirrored = Boolean(on);
  ui.chkMirror.checked = state.mirrored;
  ui.stage.classList.toggle('no-mirror', !state.mirrored);
  state.liveViewer?.setMirror(state.mirrored);
  if (remember) localStorage.setItem(mirrorKeyFor(state.facing), state.mirrored ? 'on' : 'off');
}

function wireDepthSlider(input, label) {
  input.dataset.depthSlider = '1';
  label.dataset.depthValue = '1';
  input.value = String(Math.round(currentDepth() * 100));
  label.textContent = `${Math.round(currentDepth() * 100)}%`;
  input.addEventListener('input', () => setDepth(Number(input.value) / 100));
}

/** Show Enter VR / AR buttons when the browser offers immersive sessions, else a short note. */
async function wireXrButtons(btnVr, btnAr, note, getViewer) {
  if (!state.xr.checked) { state.xr = { ...(await xrSupport()), checked: true }; }
  btnVr.hidden = !state.xr.vr;
  btnAr.hidden = !state.xr.ar;
  if (!state.xr.vr && !state.xr.ar) {
    note.textContent = 'No VR headset detected: open this page in a WebXR browser (for example Meta Quest Browser) for Enter VR. The 3D view works everywhere.';
  } else {
    note.textContent = 'In VR the figure stands life-size in front of you; a controller trigger toggles play/pause.';
    loadThree().catch(() => {}); // warm up so Enter VR can start within the click's user activation
  }
  const enter = mode => async () => {
    try {
      const viewer = await getViewer();
      await viewer.enterXR(mode);
    } catch (err) {
      console.warn(err);
      toast(`Could not start ${mode === 'immersive-ar' ? 'AR' : 'VR'}: ${err.message || err.name}`, 6000);
    }
  };
  btnVr.onclick = enter('immersive-vr');
  btnAr.onclick = enter('immersive-ar');
}

function ensureLiveViewer() {
  if (state.liveViewer) return Promise.resolve(state.liveViewer);
  if (state.liveViewerPromise) return state.liveViewerPromise;
  ui.live3d.hidden = false;
  $('.viewer3d-help[data-for="live-3d"]').hidden = false;
  ui.btnLive3d.setAttribute('aria-pressed', 'true');
  const aspect = (ui.video.videoWidth || 16) / (ui.video.videoHeight || 9);
  state.liveViewerPromise = createViewer3D(ui.live3d, { people: MAX_PEOPLE, aspect, mirrored: state.mirrored }).then(
    v => { state.liveViewer = v; state.liveViewerPromise = null; v.setScale({ depth: currentDepth() }); return v; },
    err => { state.liveViewerPromise = null; ui.live3d.hidden = true; ui.btnLive3d.setAttribute('aria-pressed', 'false'); toast(`Could not load the 3D view: ${err.message}`, 6000); throw err; },
  );
  return state.liveViewerPromise;
}

function toggleLiveViewer() {
  if (state.liveViewer) {
    state.liveViewer.destroy();
    state.liveViewer = null;
    ui.live3d.hidden = true;
    $('.viewer3d-help[data-for="live-3d"]').hidden = true;
    ui.btnLive3d.setAttribute('aria-pressed', 'false');
    return;
  }
  ensureLiveViewer().catch(() => {});
}

/** Keep a running estimate of body scale and floor level for the live 3D figure. */
function updateLiveBody(pose) {
  if (!pose) return;
  const lm = pose.lm, aspect = (ui.video.videoWidth || 16) / (ui.video.videoHeight || 9);
  const v = i => lm[i * STRIDE_LOCAL + 3];
  if (v(11) < 0.5 || v(12) < 0.5 || v(23) < 0.5 || v(24) < 0.5) return;
  const X = i => lm[i * STRIDE_LOCAL] * aspect, Y = i => lm[i * STRIDE_LOCAL + 1];
  const torso = Math.hypot((X(11) + X(12)) / 2 - (X(23) + X(24)) / 2, (Y(11) + Y(12)) / 2 - (Y(23) + Y(24)) / 2);
  const body = state.liveBody;
  body.torso = body.torso ? body.torso * 0.9 + torso * 0.1 : torso;
  const ankle = Math.max(v(27) >= 0.5 ? Y(27) : -1, v(28) >= 0.5 ? Y(28) : -1);
  const floor = ankle >= 0 ? ankle + torso * 0.15 : (Y(23) + Y(24)) / 2 + torso * 2.3;
  body.floor = body.floor ? Math.max(body.floor * 0.995, floor) : floor; // follow the lowest point seen, slowly forgetting
  state.liveViewer?.setScale({ scale: 0.5 / body.torso, floorY: body.floor, depth: currentDepth() });
}
const STRIDE_LOCAL = 4;

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
  state.detectorPromise = createPoseDetector({ model, people, onStatus: onLoadStatus }).then(
    d => {
      showLoadProgress(null);
      refreshCacheInfo();
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
    err => { showLoadProgress(null); state.detectorPromise = null; if (old) state.detector = old; throw err; },
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
  const raw = state.lastPeople && now - state.lastPeopleAt < 600 ? state.lastPeople : null;
  const minVis = minVisibilityFor(state.detector.engine);
  const slots = raw ? raw.map((pose, p) => {
    if (!pose) { state.smoother.reset(p); state.handSmoother.reset(p); return null; }
    return {
      lm: state.smoother.apply(pose.lm, 0, p, dt),
      hands: pose.hands ? state.handSmoother.apply(pose.hands, 0, p, dt) : null,
      face: pose.face ?? null,
    };
  }) : null;
  if (!slots) { state.smoother.reset(); state.handSmoother.reset(); }
  const faceContours = state.detector.faceContours;
  if (state.liveViewer) state.liveViewer.setPeople(slots?.map(pose => pose && { lm: pose.lm, offset: 0 }) ?? null, minVis);
  state.effects.forEach((effect, p) => {
    const pose = slots?.[p] ?? null;
    effect.draw(overlayCtx, pose?.lm ?? null, 0, width, height, dt, { person: p, minVis, hands: pose?.hands ?? null, face: pose?.face ?? null, faceContours, videoVisible: state.showVideo });
  });
  if (slots && state.detector.people > 1) {
    const mirrorText = state.mirrored || game.isPlaying; // the game always shows the player mirrored
    slots.forEach((pose, p) => { if (pose) drawPersonBadge(overlayCtx, pose.lm, 0, width, height, p, { mirrorText, minVis }); });
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
      const trigger = state.countdown.trigger ?? 'pose';
      state.countdown = null;
      ui.countdown.hidden = true;
      startRecording(trigger, now);
    } else {
      ui.countdown.hidden = false;
      ui.countdown.textContent = String(Math.ceil(left / 1000));
    }
  }

  if (state.recording) {
    const rec = state.recording;
    if (present.length && now - rec.lastStored >= STORE_INTERVAL_MS - 1) {
      const frame = new Float32Array(rec.people * FRAME_SIZE);
      const hands = rec.hands ? new Float32Array(rec.people * HAND_SIZE) : null;
      const world = rec.world ? new Float32Array(rec.people * WORLD_SIZE) : null;
      slots.forEach((pose, p) => {
        if (!pose || p >= rec.people) return;
        frame.set(pose.lm, p * FRAME_SIZE);
        if (hands && pose.hands) hands.set(pose.hands, p * HAND_SIZE);
        if (world && pose.world) world.set(pose.world, p * WORLD_SIZE);
      });
      rec.frames.push({ t: now - rec.startedAt, lm: frame, hands, world });
      rec.lastStored = now;
    }
    ui.recTime.textContent = fmtClock(now - rec.startedAt);
  }

  if (game.isOpen) game.onPoses(slots, now, (ui.video.videoWidth || 16) / (ui.video.videoHeight || 9));
  else updateGesture(present, now);
  updateFraming(present, now);
  updateStatus(present.length);
  if (state.liveViewer) updateLiveBody(slots[0] ?? present[0]);
}

/** Show a quiet hint while someone tracked has less than head-to-hips in view (held 0.6 s to avoid flicker). */
function updateFraming(present, now) {
  const minVis = minVisibilityFor(state.detector?.engine);
  const bad = present.some(pose => !bodyInFrame(pose.lm, 0, minVis));
  const f = state.framing;
  if (bad !== f.bad) { f.bad = bad; f.since = now; }
  const show = present.length > 0 && f.bad && now - f.since > 600;
  if (show !== !ui.frameWarn.hidden && (show || now - f.since > 600)) ui.frameWarn.hidden = !show;
  if (!present.length) ui.frameWarn.hidden = true;
}

function updateGesture(present, now) {
  const minVis = minVisibilityFor(state.detector?.engine);
  const active = ui.chkGesture.checked && !state.countdown && now >= state.lockoutUntil && present.some(pose => handsAboveHead(pose.lm, 0, minVis));
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
  state.recording = { startedAt: now, wallStart: Date.now(), lastStored: -Infinity, frames: [], trigger, people: state.detector.people, hands: Boolean(state.detector.hands), world: Boolean(state.detector.world) };
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

/**
 * Stop recording. `trimBeforeT` drops frames from that time on (used to cut the stop gesture).
 * Normally the new session is saved and opened in the dashboard; with `navigate: false` (the game)
 * it is saved and returned instead, and with `discard: true` it is dropped.
 */
function stopRecording(trimBeforeT = null, { navigate = true, discard = false, name = null } = {}) {
  const rec = state.recording;
  if (!rec) return null;
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
  if (discard) return null;
  if (frames.length < 10) {
    toast('The recording was too short to analyse, so nothing was saved.');
    return null;
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
  if (name) session.name = `${name} · ${session.name}`;
  if (navigate) { openSession(session, { save: true }); return session; }
  store.save(session).then(updateSessionCount, err => {
    if (!(err instanceof StorageFullError)) console.error(err);
    toast(err.message, 8000);
  });
  return session;
}

// ---------------------------------------------------------------------------------------
// Sessions

async function openSession(session, { save = false } = {}) {
  const analyses = Array.from({ length: session.people || 1 }, (_, p) => analyze(personView(session, p)));
  const person = Math.max(0, analyses.findIndex(a => a.ok));
  const current = { session, analyses, person, saved: false, saveError: null, compare: null };
  state.compareCache.clear();
  if (save) {
    try {
      await store.save(session);
      current.saved = true;
    } catch (err) {
      if (!(err instanceof StorageFullError)) console.error(err);
      current.saveError = err.message;
    }
  } else {
    current.saved = true;
  }
  state.current = current;
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
  const hands = s.hands ? ' · with hands' : '';
  return `${modelText}${people}${hands}`;
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
  const triggerText = { pose: 'started by pose', button: 'started by button', voice: 'started by voice', game: 'danced in the game' }[session.trigger] ?? '';
  const comparing = Boolean(state.current.compare);
  const tabs = `
    <div class="person-tabs" role="tablist" aria-label="Person">
      ${people > 1
        ? analyses.map((an, p) => `<button class="tab${p === person && !comparing ? ' is-active' : ''}" role="tab" data-person="${p}" aria-selected="${p === person && !comparing}">
        <i class="swatch" style="background:${personColors(p).left}"></i>Person ${p + 1}
        <small>${an.ok ? `${fmtPct(an.validFraction)} tracked` : 'no data'}</small></button>`).join('')
        : `<button class="tab${comparing ? '' : ' is-active'}" role="tab" data-person="0" aria-selected="${!comparing}">This dancer</button>`}
      <button class="tab tab-compare${comparing ? ' is-active' : ''}" role="tab" data-compare aria-selected="${comparing}" title="Compare two dancers: two people in this session, or someone from another saved session">⇄ Compare two dancers</button>
    </div>`;
  const root = ui.views.dashboard;
  root.innerHTML = `
    <div class="dash-header">
      <div>
        <input id="session-name" class="name-input" type="text" value="${escapeHtml(session.name)}" aria-label="Session name" spellcheck="false">
        <div class="meta">${escapeHtml(fmtDate(session.createdAt))} · ${fmtDuration(session.durationMs)} · ${session.frameCount} frames${fpsText} · ${describeSession(session)}${triggerText ? ` · ${triggerText}` : ''}</div>
      </div>
      <div class="dash-actions">
        <label class="check help-toggle" title="Show or hide the explanatory text under each panel"><input type="checkbox" id="show-help"> Explanations</label>
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
        <label class="check" title="Draw the kinesphere spheres, the standing shoulder and knee lines and the live body metrics on the replay"><input type="checkbox" id="replay-overlay"${a.ok ? '' : ' disabled'}> Metrics overlay</label>
      </div>
      <div class="viewer3d-bar">
        <button class="btn btn-sm" id="replay-3d-toggle" aria-pressed="false" title="Show the recording as a 3D figure you can orbit">3D view</button>
        <button class="btn btn-sm" id="replay-vr" hidden>Enter VR</button>
        <button class="btn btn-sm" id="replay-ar" hidden>Enter AR</button>
        <label class="check" title="Show the replay and 3D figure mirrored, as the dancer saw the preview. Untick to see the dancer as someone facing them would.">
          <input type="checkbox" id="replay-mirror"> Mirror view</label>
        <label class="select depth-ctl" title="How much of the model's depth estimate to show. MediaPipe's z is noisy and exaggerated, so less than full is usually more natural.">Depth
          <input type="range" id="replay-depth" min="0" max="100" step="5" aria-label="Depth exaggeration"> <span id="replay-depth-val" class="mono"></span>
        </label>
        <span class="meta" id="replay-vr-note"></span>
      </div>
      <div id="replay-3d" class="viewer3d" hidden></div>
      <p class="viewer3d-help" id="replay-3d-help" hidden>Drag to orbit, scroll or pinch to zoom, right-drag or two-finger drag to pan. Axes at the dancer's feet: <b style="color:#e34948">red x</b> = the dancer's left/right, <b style="color:#2ea043">green y</b> = up, <b style="color:#3987e5">blue z</b> = towards the camera (where you stand in VR). Grid squares are 0.5 m.</p>
    </section>
    ${tabs}
    ${comparing ? '<div id="compare-root"></div>'
      : a.ok ? dashboardCards(a) : `<div class="banner">Not enough pose data to analyse ${people > 1 ? `person ${person + 1}` : 'this session'}. ${escapeHtml(a.reason)}</div>`}
  `;
  const helpBox = root.querySelector('#show-help');
  helpBox.checked = localStorage.getItem(HELP_KEY) !== 'off';
  root.classList.toggle('hide-help', !helpBox.checked);
  helpBox.addEventListener('change', () => {
    localStorage.setItem(HELP_KEY, helpBox.checked ? 'on' : 'off');
    root.classList.toggle('hide-help', !helpBox.checked);
  });

  root.querySelector('#session-name').addEventListener('change', e => {
    const name = e.target.value.trim() || session.name;
    e.target.value = name;
    session.name = name;
    if (state.current.saved) {
      store.save(session).then(updateSessionCount, err => toast(err.message));
    }
  });
  root.querySelector('[data-action="export-json"]').onclick = () => exportJSON(session);
  root.querySelector('[data-action="export-csv"]').onclick = () => exportCSV(session);
  root.querySelector('[data-action="delete"]').onclick = () => {
    if (!confirm(`Delete “${session.name}”? This cannot be undone unless you exported it.`)) return;
    store.remove(session.id).catch(err => toast(`Could not delete: ${err.message}`)).finally(() => {
      state.current = null;
      updateSessionCount();
      setView('sessions');
    });
  };
  root.querySelector('[data-action="new"]').onclick = () => setView('live');
  root.querySelectorAll('.person-tabs .tab[data-person]').forEach(tab => {
    tab.onclick = () => { state.current.compare = null; state.current.person = Number(tab.dataset.person); renderDashboard(); };
  });
  root.querySelector('.person-tabs .tab[data-compare]').onclick = () => {
    if (!state.current.compare) { state.current.compare = defaultCompare(); renderDashboard(); }
  };

  const onTime = t => { for (const c of state.charts) c.setCursor?.(t); };
  state.replay = setupReplay(session, root, onTime, a.ok ? { analysis: a, person } : null);
  setupReplay3D(session, analyses, root, state.replay);
  if (comparing) renderCompare(root);
  else if (a.ok) mountCharts(a, session, root);
}

// ---------------------------------------------------------------------------------------
// Compare two dancers: two people in this session, or someone from another saved session.

/** Everyone who can be compared: { own: [...], others: [...] } of { key: 'sessionId:person', label }. */
function compareChoices() {
  const { session, analyses } = state.current;
  const many = (session.people || 1) > 1;
  const own = analyses.map((an, p) => ({ key: `${session.id}:${p}`, label: many ? `Person ${p + 1} (this session)` : `This session${session.name ? ` (${session.name})` : ''}` }));
  const others = store.list().filter(m => m.id !== session.id).flatMap(m => Array.from({ length: m.people || 1 }, (_, p) => ({
    key: `${m.id}:${p}`, label: `${m.name}${(m.people || 1) > 1 ? ` · Person ${p + 1}` : ''}`,
  })));
  return { own, others };
}

function defaultCompare() {
  const { own, others } = compareChoices();
  const first = own[state.current.person] ?? own[0];
  const second = own.find(o => o.key !== first.key) ?? others[0] ?? null;
  return { a: first.key, b: second?.key ?? null };
}

/** Load one dancer to compare: { session, a: analysis }, from this session or storage (cached). */
async function loadCompareRef(key) {
  const cut = key.lastIndexOf(':');
  const id = key.slice(0, cut), p = Number(key.slice(cut + 1));
  const { session, analyses } = state.current;
  if (id === session.id) return { session, a: analyses[p] };
  let entry = state.compareCache.get(id);
  if (!entry) {
    const loaded = await store.load(id);
    if (!loaded) return null;
    entry = { session: loaded, analyses: [] };
    state.compareCache.set(id, entry);
  }
  entry.analyses[p] ??= analyze(personView(entry.session, p));
  return { session: entry.session, a: entry.analyses[p] };
}

async function renderCompare(root) {
  const host = root.querySelector('#compare-root');
  if (!host) return;
  const cmp = state.current.compare;
  const { own, others } = compareChoices();
  const all = [...own, ...others];
  const options = selected => `<optgroup label="This session">${own.map(o => `<option value="${escapeHtml(o.key)}"${o.key === selected ? ' selected' : ''}>${escapeHtml(o.label)}</option>`).join('')}</optgroup>
    ${others.length ? `<optgroup label="Other saved sessions">${others.map(o => `<option value="${escapeHtml(o.key)}"${o.key === selected ? ' selected' : ''}>${escapeHtml(o.label)}</option>`).join('')}</optgroup>` : ''}`;
  host.innerHTML = `
    <section class="card">
      <h2>Compare two dancers</h2>
      <p class="card-sub">Pick any two dancers: two people tracked in this session, or someone from another session saved in this browser. Every metric is worked out for each dancer separately (relative to their own body size), so people of different heights and recordings of different lengths can be compared fairly.</p>
      <div class="compare-picker">
        <label class="select"><span class="cmp-tag cmp-a">A</span><select id="cmp-a" aria-label="Dancer A">${options(cmp.a)}</select></label>
        <label class="select"><span class="cmp-tag cmp-b">B</span><select id="cmp-b" aria-label="Dancer B">${cmp.b ? '' : '<option value="" selected>Choose a second dancer…</option>'}${options(cmp.b)}</select></label>
      </div>
    </section>
    <div id="compare-body"></div>`;
  host.querySelector('#cmp-a').onchange = e => { cmp.a = e.target.value; renderCompare(root); };
  host.querySelector('#cmp-b').onchange = e => { cmp.b = e.target.value || null; renderCompare(root); };
  const body = host.querySelector('#compare-body');
  if (state.compareChart) { state.compareChart.destroy(); state.charts = state.charts.filter(c => c !== state.compareChart); state.compareChart = null; }
  if (!cmp.b) {
    body.innerHTML = `<div class="banner">${all.length < 2 ? 'There is only one dancer to compare. Track several people (People, in the Live view) or record or import another session, then come back here.' : 'Choose a second dancer above.'}</div>`;
    return;
  }
  body.innerHTML = '<p class="meta">Loading…</p>';
  const token = ++state.compareToken;
  const [A, B] = await Promise.all([loadCompareRef(cmp.a), loadCompareRef(cmp.b)]);
  if (token !== state.compareToken || !root.contains(body)) return; // a newer choice or view replaced this one
  const labelOf = key => all.find(o => o.key === key)?.label ?? key;
  const fail = [[A, cmp.a], [B, cmp.b]].find(([X]) => !X || !X.a.ok);
  if (fail) {
    body.innerHTML = `<div class="banner">Not enough pose data to compare ${escapeHtml(labelOf(fail[1]))}${fail[0]?.a.reason ? `: ${escapeHtml(fail[0].a.reason)}` : '.'}</div>`;
    return;
  }
  A.label = labelOf(cmp.a); B.label = labelOf(cmp.b);
  body.innerHTML = compareCards(A, B);
  mountCompareCharts(A, B, body);
}

const tag = k => `<span class="cmp-tag cmp-${k}">${k.toUpperCase()}</span>`;

function compareCards(A, B) {
  const a = A.a, b = B.a;
  const pct = v => (Number.isFinite(v) ? fmtPct(v) : '–');
  const signed = (v, d) => { const r = Number(v.toFixed(d)); return r === 0 ? (0).toFixed(d) : `${r > 0 ? '+' : '−'}${Math.abs(r).toFixed(d)}`; };
  const ptsDiff = (x, y) => (Number.isFinite(x) && Number.isFinite(y) ? `${signed((y - x) * 100, 0)} pts` : '');
  const num = (v, d, unit = '') => (Number.isFinite(v) ? `${v.toFixed(d)}${unit}` : '–');
  const numDiff = (x, y, d, unit = '') => (Number.isFinite(x) && Number.isFinite(y) ? `${signed(y - x, d)}${unit}` : '');
  const mps = X => v => v * X.a.speed.metresPerTorso;
  const top = X => X.a.body.joints.reduce((m, j) => (j.share > m.share ? j : m));
  const perMin = X => X.a.space.travelled / Math.max(1e-6, X.session.durationMs / 60000);
  const share = (key, X) => X.a.kinesphere.shares[key];
  const rows = [
    ['Duration', fmtDuration(A.session.durationMs), fmtDuration(B.session.durationMs), ''],
    ['Pose tracked', pct(a.validFraction), pct(b.validFraction), ptsDiff(a.validFraction, b.validFraction)],
    ...KINESPHERE_CLASSES.map(c => [`${c.label} kinesphere`, pct(share(c.key, A)), pct(share(c.key, B)), ptsDiff(share(c.key, A), share(c.key, B))]),
    ['Left side of the body', pct(a.body.lateral.left), pct(b.body.lateral.left), ptsDiff(a.body.lateral.left, b.body.lateral.left)],
    ['Right side of the body', pct(a.body.lateral.right), pct(b.body.lateral.right), ptsDiff(a.body.lateral.right, b.body.lateral.right)],
    ['Upper body', pct(a.body.vertical.upper), pct(b.body.vertical.upper), ptsDiff(a.body.vertical.upper, b.body.vertical.upper)],
    ['Lower body', pct(a.body.vertical.lower), pct(b.body.vertical.lower), ptsDiff(a.body.vertical.lower, b.body.vertical.lower)],
    ...LEVELS.map(l => [`${l.label} level`, pct(a.levels.shares[l.key]), pct(b.levels.shares[l.key]), ptsDiff(a.levels.shares[l.key], b.levels.shares[l.key])]),
    ['Most active body part', `${top(A).name} (${fmtPct(top(A).share)})`, `${top(B).name} (${fmtPct(top(B).share)})`, ''],
    ['Mean speed', num(mps(A)(a.speed.mean), 2, ' m/s'), num(mps(B)(b.speed.mean), 2, ' m/s'), numDiff(mps(A)(a.speed.mean), mps(B)(b.speed.mean), 2, ' m/s')],
    ['Peak speed', num(mps(A)(a.speed.peak), 2, ' m/s'), num(mps(B)(b.speed.peak), 2, ' m/s'), numDiff(mps(A)(a.speed.peak), mps(B)(b.speed.peak), 2, ' m/s')],
    ['Floor area used', `${num(a.space.width, 1)} × ${num(a.space.depth, 1)} m`, `${num(b.space.width, 1)} × ${num(b.space.depth, 1)} m`, ''],
    ['Distance travelled', num(a.space.travelled, 1, ' m'), num(b.space.travelled, 1, ' m'), numDiff(a.space.travelled, b.space.travelled, 1, ' m')],
    ['Distance travelled per minute', num(perMin(A), 1, ' m'), num(perMin(B), 1, ' m'), numDiff(perMin(A), perMin(B), 1, ' m')],
    ['Typical distance from camera', num(a.space.distance, 1, ' m'), num(b.space.distance, 1, ' m'), ''],
  ];
  const pair = (title, id) => `
    <h3 class="cmp-row-title">${title}</h3>
    <div><h3>${tag('a')}${escapeHtml(A.label)}</h3><div id="${id}-a"></div></div>
    <div><h3>${tag('b')}${escapeHtml(B.label)}</h3><div id="${id}-b"></div></div>`;
  return `
    <section class="card">
      <h2>At a glance</h2>
      <p class="card-sub">Shares are of each dancer's own session. The last column is B minus A (percentage points for shares).</p>
      <div class="table-scroll"><table class="data-table cmp-table">
        <thead><tr><th>Measure</th><th class="num">${tag('a')}A</th><th class="num">${tag('b')}B</th><th class="num">B − A</th></tr></thead>
        <tbody>${rows.map(r => `<tr><td>${r[0]}</td><td class="num">${r[1]}</td><td class="num">${r[2]}</td><td class="num diff">${r[3]}</td></tr>`).join('')}</tbody>
      </table></div>
    </section>
    <section class="card">
      <h2>Kinesphere</h2>
      <p class="card-sub">Share of time in a near, medium or far kinesphere, each measured against that dancer's own limb lengths.</p>
      <div class="cmp-grid">
        <div><h3>${tag('a')}${escapeHtml(A.label)}</h3><div id="cmp-kine-a"></div></div>
        <div><h3>${tag('b')}${escapeHtml(B.label)}</h3><div id="cmp-kine-b"></div></div>
      </div>
    </section>
    <section class="card">
      <h2>Body</h2>
      <p class="card-sub">Which body parts moved, on which side, in the upper or lower body, and at what level. Both activity figures share one colour scale, so the same colour means the same share of movement.</p>
      <div class="cmp-grid">
        ${pair('Body part activity', 'cmp-body')}
        <div class="cmp-row-title"><div id="cmp-body-legend"></div><label class="check"><input type="checkbox" id="cmp-show-values"> Show values</label></div>
        ${pair('Left / right', 'cmp-side')}
        ${pair('Upper / lower', 'cmp-updown')}
        ${pair('Level: where was the activity?', 'cmp-level')}
      </div>
    </section>
    <section class="card">
      <h2>Speed</h2>
      <p class="card-sub">Average joint speed over each session, in approximate metres per second, from the start of each recording.</p>
      <div id="cmp-speed"></div>
    </section>
    <section class="card">
      <h2>Path on the floor</h2>
      <p class="card-sub">Both dancers' estimated paths from the bird's eye view, each from start (ring) to end (filled dot). For two sessions, positions are relative to each session's own camera.</p>
      <div id="cmp-floor"></div>
    </section>`;
}

function mountCompareCharts(A, B, root) {
  const ord = ordinalColors();
  for (const [X, k] of [[A, 'a'], [B, 'b']]) {
    stackedBar(root.querySelector(`#cmp-kine-${k}`), KINESPHERE_CLASSES.map((c, i) => ({ label: c.label, value: X.a.kinesphere.shares[c.key], fill: ord[i].fill, ink: ord[i].ink })));
    sideFigure(root.querySelector(`#cmp-side-${k}`), { pose: TEMPLATE_POSE, left: X.a.body.lateral.left, right: X.a.body.lateral.right });
    upperLowerFigure(root.querySelector(`#cmp-updown-${k}`), { pose: TEMPLATE_POSE, upper: X.a.body.vertical.upper, lower: X.a.body.vertical.lower });
    levelsFigure(root.querySelector(`#cmp-level-${k}`), { pose: TEMPLATE_POSE, shares: X.a.levels.shares, labels: Object.fromEntries(LEVELS.map(l => [l.key, l.label])) });
  }
  const maxShare = Math.max(1e-6, ...[A, B].flatMap(X => X.a.body.joints.map(j => j.share)));
  const showBox = root.querySelector('#cmp-show-values');
  showBox.checked = localStorage.getItem(SHOW_VALUES_KEY) === 'on';
  const drawBodies = () => {
    for (const [X, k] of [[A, 'a'], [B, 'b']]) {
      const joints = X.a.body.joints;
      bodyFigure(root.querySelector(`#cmp-body-${k}`), {
        pose: TEMPLATE_POSE, values: joints.map(j => j.share / maxShare), names: MAJOR_NAMES,
        labels: showBox.checked ? joints.map(j => fmtPct(j.share)) : null,
        tooltip: j => `${MAJOR_NAMES[j]}: ${fmtPct(joints[j].share, 1)} of all movement, active ${fmtPct(joints[j].activeTime)} of the time`,
      });
    }
  };
  showBox.addEventListener('change', () => { localStorage.setItem(SHOW_VALUES_KEY, showBox.checked ? 'on' : 'off'); drawBodies(); });
  drawBodies();
  root.querySelector('#cmp-body-legend').append(heatLegend('Less movement', 'More movement'));
  const mps = X => X.a.speed.series.map(p => ({ t: p.t, v: p.v * X.a.speed.metresPerTorso }));
  state.compareChart = lineChart(root.querySelector('#cmp-speed'), {
    series: mps(A), color: 'var(--cmp-a)', label: `A: ${A.label}`,
    extra: [{ series: mps(B), color: 'var(--cmp-b)', label: `B: ${B.label}` }],
    xMax: Math.max(A.session.durationMs, B.session.durationMs), yLabel: 'Speed (m/s, approximate)', yUnit: 'm/s',
  });
  state.charts.push(state.compareChart);
  floorPlot(root.querySelector('#cmp-floor'), {
    fov: A.a.space.fov,
    paths: [{ path: A.a.space.path, color: 'var(--cmp-a)', label: `A: ${A.label}` }, { path: B.a.space.path, color: 'var(--cmp-b)', label: `B: ${B.label}` }],
  });
}

function setupReplay3D(session, analyses, root, replay) {
  const panel = root.querySelector('#replay-3d');
  const toggle = root.querySelector('#replay-3d-toggle');
  let viewer = null, viewerPromise = null;
  const ensure = () => {
    if (viewer) return Promise.resolve(viewer);
    if (viewerPromise) return viewerPromise;
    panel.hidden = false;
    root.querySelector('#replay-3d-help').hidden = false;
    toggle.setAttribute('aria-pressed', 'true');
    viewerPromise = createViewer3D(panel, { people: session.people || 1, aspect: session.width / session.height, mirrored: replay.mirrorView }).then(
      v => {
        viewer = v; viewerPromise = null;
        const slot = Math.max(0, analyses.findIndex(x => x.ok));
        v.setScale({ ...estimateBodyScale(session, slot), depth: currentDepth() });
        replay.setDepth = d => v.setScale({ depth: d });
        replay.setMirror = on => v.setMirror(on);
        v.onSelect(() => replay.togglePlay());
        replay.attachViewer(v);
        return v;
      },
      err => { viewerPromise = null; panel.hidden = true; toggle.setAttribute('aria-pressed', 'false'); toast(`Could not load the 3D view: ${err.message}`, 6000); throw err; },
    );
    return viewerPromise;
  };
  toggle.onclick = () => {
    if (viewer) { replay.attachViewer(null); viewer.destroy(); viewer = null; panel.hidden = true; root.querySelector('#replay-3d-help').hidden = true; toggle.setAttribute('aria-pressed', 'false'); return; }
    ensure().catch(() => {});
  };
  wireXrButtons(root.querySelector('#replay-vr'), root.querySelector('#replay-ar'), root.querySelector('#replay-vr-note'), ensure);
  wireDepthSlider(root.querySelector('#replay-depth'), root.querySelector('#replay-depth-val'));
  const mirrorBox = root.querySelector('#replay-mirror');
  mirrorBox.checked = replay.mirrorView;
  mirrorBox.addEventListener('change', () => replay.setMirrorView(mirrorBox.checked));
  const destroyOld = replay.destroy;
  replay.destroy = () => { viewer?.destroy(); viewer = null; replay.setDepth = null; replay.setMirror = null; destroyOld(); };
}

function dashboardCards(a) {
  const k = a.kinesphere;
  const shapes = a.shapes.map((s, i) => `
      <div class="shape"><div class="shape-rank">#${i + 1}</div><div id="shape-${i}"></div><div class="shape-share">${fmtPct(s.share)}<small class="meta"> of the time</small></div></div>`).join('');
  const kineFigs = KINESPHERE_CLASSES.map(c => `
      <div class="kine-fig">
        <div id="kine-fig-${c.key}"></div>
        <div class="kine-fig-name">${c.label}</div>
        <div class="kine-fig-share">${fmtPct(k.shares[c.key])}</div>
        ${k.poses[c.key] ? '' : '<div class="meta">not reached</div>'}
      </div>`).join('');
  return `
    <section class="card">
      <h2>Kinesphere</h2>
      <p class="card-sub">How far the limbs reached from the body, as a share of the session. Three concentric spheres around the centre of the torso (midway between the hips and the shoulders) are sized from this dancer's own limb lengths, measured from the skeleton
        ${k.mode === '3d' ? 'in 3D (MediaPipe world landmarks), so reaches towards and away from the camera count fully.' : 'on screen in 2D, because this session has no 3D world landmarks (MoveNet, or recorded before 3D support).'}</p>
      <div class="kine-figs">${kineFigs}</div>
      <div id="kine-bar"></div>
      <div class="stat-row">
        ${stat('Pose tracked', fmtPct(a.validFraction), 'of frames')}
        ${stat('Measured in', k.mode === '3d' ? '3D' : '2D', k.mode === '3d' ? `world landmarks, ${fmtPct(k.coverage3d)} of frames` : 'screen positions')}
        ${k.metres ? stat('Arm length', ((k.metres.upperArm + k.metres.forearm) * 100).toFixed(0), 'cm, shoulder to wrist') : stat('Reach', k.radii.shared.large.toFixed(2), 'torso lengths from the torso centre')}
      </div>
      <div id="kine-table"></div>
      <p class="fine"><b>Near</b>: the sphere at the distance of the proximal joints (the shoulders). <b>Medium</b>: the sphere at the distance of the mid-limb joints (elbows) when the arm is outstretched. <b>Far</b>: the sphere at the distance of the hands when outstretched.
        The spheres are sized from the arms and stay fixed during the session; lifted legs are compared with the same spheres (a standing leg is support, not reach). Each hand or foot is classed by the sphere it is nearest to, and the frame takes the farthest of those, so relaxed arms hanging at the sides count as near. Each figure above is the most typical pose in that class.
        ${k.mode === '3d'
          ? 'Limb lengths are the median 3D length of each segment from MediaPipe\'s world landmarks (metric, in metres); distances are in torso lengths. The replay overlay draws the spheres as circles on screen, so a hand reaching towards the camera can be in the far sphere while it looks inside the circle.'
          : 'Limb lengths are the 90th percentile of each segment\'s length on screen, so reaching straight towards or away from the camera is under-counted.'}</p>
    </section>

    <section class="card">
      <h2>Body</h2>
      <p class="card-sub">Which body parts moved, on which side, in the upper or lower body, and at what height the activity happened. Left and right are the dancer's own sides.</p>
      <div class="body-grid">
        <div>
          <h3>Body part activity</h3>
          <div id="body-figure"></div>
          <div id="body-legend"></div>
          <label class="check"><input type="checkbox" id="show-values"> Show values</label>
        </div>
        <div>
          <h3>Left / right</h3>
          <div id="side-figure"></div>
        </div>
        <div>
          <h3>Upper / lower</h3>
          <div id="updown-figure"></div>
        </div>
        <div>
          <h3>Level: where was the activity?</h3>
          <div id="level-figure"></div>
        </div>
      </div>
      <div id="body-table"></div>
      <p class="fine"><b>Body part activity</b>: each of the 13 markers (head, shoulders, elbows, hands, hips, knees, feet) is sized and coloured by its share of all movement; <em>active time</em> is the share of the session that part moved faster than ${'0.5'} torso lengths per second.
        <b>Left / right</b>: share of movement on each side of the body (head excluded).
        <b>Upper / lower</b>: share of movement above the waist (shoulders, elbows, hands) and below it (hips, knees, feet), head excluded.
        <b>Level</b> is about position, not which limb moved: <em>high</em> when at least one limb is above standing shoulder height, <em>low</em> when a hand or the hips are below standing knee height, <em>medium</em> otherwise. The head is not counted; standing heights are measured from this session (standing shoulder ${a.levels.standing.shoulder.toFixed(2)} and knee ${a.levels.standing.knee.toFixed(2)} torso lengths above the floor).
        The floor is only measured while a foot is in view (${fmtPct(a.levels.feetInView)} of this session); when the feet are out of frame the dancer is assumed to be standing, so low is then only detected from the hands.</p>
    </section>

    <section class="card">
      <h2>Speed</h2>
      <p class="card-sub">How fast the body moved over time: the average speed of the 13 major joints, smoothed over a second. Higher means faster movement.</p>
      <div class="stat-row" id="speed-stats"></div>
      <div id="speed-chart"></div>
      <label class="check"><input type="checkbox" id="show-speed-values"> Show values</label>
      <div id="speed-table"></div>
      <p class="fine">Speeds are measured on screen relative to the dancer's own size (torso lengths per second), so they do not depend on how far from the camera you stood; <em>Show values</em> converts them to approximate metres per second using ${a.speed.metresMeasured ? `this dancer's torso length measured in 3D (${(a.speed.metresPerTorso * 100).toFixed(0)} cm)` : `a typical torso length of ${(a.speed.metresPerTorso * 100).toFixed(0)} cm (this session has no 3D data)`}. Movement straight towards or away from the camera is under-counted.
        Tracking glitches are removed before averaging: a joint speed far beyond that joint's usual range for the session (an extreme outlier) or faster than any human limb, and frames where several joints jump at once, typically when the body is very close to the camera or partly out of view${a.speed.removed > 0 ? ` (${fmtPct(a.speed.removed, 1)} of joint speed samples in this session)` : ' (none in this session)'}.</p>
    </section>

    <section class="card">
      <h2>Path on the floor</h2>
      <p class="card-sub">A bird's eye view of where you moved in the room, estimated from the camera: the camera is at the top, and the shaded wedge is what it could see. Your path runs from start (ring) to end (filled dot); shaded squares show where you spent the most time.</p>
      <div class="two-col">
        <div id="space-plot"></div>
        <div>
          <div class="stat-row">
            ${stat('Floor area used', `${a.space.width.toFixed(1)} × ${a.space.depth.toFixed(1)}`, 'm, side to side × front to back')}
            ${stat('Distance travelled', a.space.travelled.toFixed(1), 'm')}
            ${stat('Distance from camera', a.space.distance.toFixed(1), 'm, typical')}
          </div>
          <div id="space-table"></div>
        </div>
      </div>
      <p class="fine">Estimated, not measured: distance from the camera comes from how big your torso looks in the picture (${a.space.torsoMeasured ? `your torso is ${(a.space.torsoM * 100).toFixed(0)} cm in 3D` : `assuming a typical ${(a.space.torsoM * 100).toFixed(0)} cm torso`}) and an assumed webcam field of view of ${a.space.fov}°, and side-to-side position from where your hips are across the picture. Expect roughly ±20% on distances; bending or turning briefly can look like stepping back, so positions are smoothed over a second. Areas cover 96% of the time (2nd–98th percentile).</p>
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
  const k = a.kinesphere;
  stackedBar(root.querySelector('#kine-bar'), KINESPHERE_CLASSES.map((c, i) => ({
    label: c.label, value: k.shares[c.key], fill: ord[i].fill, ink: ord[i].ink,
  })));
  const R = k.radii.shared;
  KINESPHERE_CLASSES.forEach((c, ci) => {
    bodyFigure(root.querySelector(`#kine-fig-${c.key}`), {
      pose: k.poses[c.key] ?? TEMPLATE_POSE, showSides: false,
      rings: [R.small, R.medium, R.large].map((r, ri) => ({ r, active: ri === ci, label: KINESPHERE_CLASSES[ri].letter, color: ri === ci ? ord[ci].fill : null })),
    });
  });
  const L = k.limbs, fmt = v => v.toFixed(2);
  const M = k.metres, cm = key => (M && Number.isFinite(M[key]) ? `${(M[key] * 100).toFixed(1)} cm` : '–');
  const limbRow = (label, key) => (M ? [label, fmt(L[key]), cm(key)] : [label, fmt(L[key])]);
  const pad = row => (M && row.length < 3 ? [...row, ''] : row);
  root.querySelector('#kine-table').append(dataTable('Show limb lengths and sphere sizes', M ? ['Measure', 'Torso lengths', 'Length (3D)'] : ['Measure', 'Torso lengths'], [
    ...(M ? [['Torso (mid-hip to mid-shoulder)', '1.00', cm('torso')]] : []),
    limbRow('Upper arm (shoulder to elbow)', 'upperArm'), limbRow('Forearm (elbow to wrist)', 'forearm'),
    limbRow('Thigh (hip to knee)', 'thigh'), limbRow('Shin (knee to ankle)', 'shin'),
    limbRow('Torso centre to shoulder', 'centreShoulder'), limbRow('Torso centre to hip', 'centreHip'),
    ['Sphere radii: near / medium / far (shoulder / + upper arm / + forearm)', `${fmt(R.small)} / ${fmt(R.medium)} / ${fmt(R.large)}`],
    ['Class boundaries (halfway between spheres)', `${fmt((R.small + R.medium) / 2)} / ${fmt((R.medium + R.large) / 2)}`],
    ...KINESPHERE_CLASSES.map(c => [`Time in ${c.label.toLowerCase()} kinesphere`, fmtPct(k.shares[c.key], 1)]),
  ].map(pad)));

  const b = a.body;
  const drawBody = () => {
    const show = root.querySelector('#show-values').checked;
    bodyFigure(root.querySelector('#body-figure'), {
      pose: TEMPLATE_POSE, values: b.joints.map(j => j.relative), names: MAJOR_NAMES,
      labels: show ? b.joints.map(j => fmtPct(j.share)) : null,
      tooltip: j => `${MAJOR_NAMES[j]}: ${fmtPct(b.joints[j].share, 1)} of all movement, active ${fmtPct(b.joints[j].activeTime)} of the time`,
    });
  };
  const showBox = root.querySelector('#show-values');
  showBox.checked = localStorage.getItem(SHOW_VALUES_KEY) === 'on';
  showBox.addEventListener('change', () => { localStorage.setItem(SHOW_VALUES_KEY, showBox.checked ? 'on' : 'off'); drawBody(); });
  drawBody();
  root.querySelector('#body-legend').append(heatLegend('Less movement', 'More movement'));
  sideFigure(root.querySelector('#side-figure'), { pose: TEMPLATE_POSE, left: b.lateral.left, right: b.lateral.right });
  upperLowerFigure(root.querySelector('#updown-figure'), { pose: TEMPLATE_POSE, upper: b.vertical.upper, lower: b.vertical.lower });
  levelsFigure(root.querySelector('#level-figure'), {
    pose: TEMPLATE_POSE, shares: a.levels.shares,
    labels: Object.fromEntries(LEVELS.map(l => [l.key, l.label])),
  });
  root.querySelector('#body-table').append(dataTable('Show body part data',
    ['Body part', 'Share of movement', 'Active time', 'Mean speed (torso lengths/s)'],
    [...b.joints.map(j => [j.name, fmtPct(j.share, 1), fmtPct(j.activeTime, 1), Number.isFinite(j.meanSpeed) ? j.meanSpeed.toFixed(2) : '–']),
      ['Left side', fmtPct(b.lateral.left, 1), '', ''], ['Right side', fmtPct(b.lateral.right, 1), '', ''],
      ['Upper body', fmtPct(b.vertical.upper, 1), '', ''], ['Lower body', fmtPct(b.vertical.lower, 1), '', ''],
      ...LEVELS.map(l => [`${l.label} level`, '', fmtPct(a.levels.shares[l.key], 1), ''])]));

  // Speed: values are hidden by default (torso lengths per second mean little on their own);
  // "Show values" adds numbers, converted to approximate metres per second.
  const sp = a.speed, mps = v => v * sp.metresPerTorso;
  let speedChart = null;
  const speedBox = root.querySelector('#show-speed-values');
  speedBox.checked = localStorage.getItem(SPEED_VALUES_KEY) === 'on';
  const peakOf = Math.max(1e-6, ...sp.series.map(p => p.v));
  const drawSpeed = () => {
    const show = speedBox.checked;
    root.querySelector('#speed-stats').innerHTML = show
      ? `${stat('Mean speed', mps(sp.mean).toFixed(2), `m/s (≈ ${sp.mean.toFixed(2)} torso lengths/s)`)}
         ${stat('Peak', Number.isFinite(sp.peak) ? mps(sp.peak).toFixed(2) : '–', `m/s, at ${fmtClock(sp.peakT)}`)}`
      : `${stat('Fastest moment', Number.isFinite(sp.peak) ? fmtClock(sp.peakT) : '–', 'into the session')}
         ${stat('Average', Number.isFinite(sp.peak) && sp.peak > 0 ? fmtPct(sp.mean / sp.peak) : '–', 'of the peak speed')}`;
    if (speedChart) { speedChart.destroy(); state.charts.splice(state.charts.indexOf(speedChart), 1); }
    speedChart = lineChart(root.querySelector('#speed-chart'), show
      ? { series: sp.series.map(p => ({ t: p.t, v: mps(p.v) })), xMax: session.durationMs, yLabel: 'Speed (m/s, approximate)', yUnit: 'm/s' }
      : { series: sp.series.map(p => ({ t: p.t, v: p.v / peakOf })), yMax: 1, xMax: session.durationMs, yLabel: 'Speed (slower ↓ ↑ faster)', yFormat: () => '', tipFormat: v => `${fmtPct(v)} of the peak` });
    state.charts.push(speedChart);
    if (state.replay) speedChart.setCursor(state.replay.time?.() ?? null);
  };
  speedBox.addEventListener('change', () => { localStorage.setItem(SPEED_VALUES_KEY, speedBox.checked ? 'on' : 'off'); drawSpeed(); });
  drawSpeed();
  root.querySelector('#speed-table').append(dataTable('Show speed data', ['Time', 'Speed (m/s, approx.)', 'Speed (torso lengths/s)'],
    sp.series.map(p => [fmtClock(p.t), mps(p.v).toFixed(2), p.v.toFixed(2)])));

  floorPlot(root.querySelector('#space-plot'), { path: a.space.path, cells: a.space.cells, fov: a.space.fov, durationMs: session.durationMs });
  root.querySelector('#space-table').append(dataTable('Show floor position data', ['Time', 'Side to side (m, + = dancer\'s right)', 'From camera (m)'],
    a.space.path.map(p => [fmtClock(p.t), p.x.toFixed(2), p.z.toFixed(2)])));

  a.shapes.forEach((s, i) => bodyFigure(root.querySelector(`#shape-${i}`), { pose: s.pose, showSides: false }));
}

// ---------------------------------------------------------------------------------------
// Replay metrics overlay: the kinesphere spheres, the standing shoulder / knee lines and the
// live Panel 2 metrics, drawn on top of the replay for one person.

const OV_ACTIVE = 0.5;   // torso lengths / s, matches the analysis "active" threshold
const OV_KCOLORS = ['#86b6ef', '#3987e5', '#ffd166'];
const OV_LCOLORS = { 0: 'rgba(255, 209, 102, 0.12)', 1: 'rgba(134, 182, 239, 0.10)', 2: 'rgba(235, 104, 52, 0.14)' };
const lerpHex = (a, b, t) => {
  const pa = [1, 3, 5].map(i => parseInt(a.slice(i, i + 2), 16)), pb = [1, 3, 5].map(i => parseInt(b.slice(i, i + 2), 16));
  return `rgb(${pa.map((v, i) => Math.round(v + (pb[i] - v) * clamp(t, 0, 1))).join(',')})`;
};

function drawMetricsOverlay(ctx, a, i, lm, w, h, mirror, times, person) {
  const F = a.frames;
  if (!F || i < 0 || i >= F.kClass.length || !lm) return;
  const sc = F.scale[i], floor = F.floor[i];
  const X = x => (mirror ? 1 - x : x) * w;
  const u = Math.max(1, w / 1280);
  ctx.save();
  // Levels: bands and the standing shoulder / knee lines.
  if (Number.isFinite(sc) && Number.isFinite(floor)) {
    const yS = (floor - a.levels.standing.shoulder * sc) * h, yK = (floor - a.levels.standing.knee * sc) * h;
    const lvl = F.level[i];
    if (lvl >= 0) {
      const [y0, y1] = lvl === 0 ? [0, yS] : lvl === 1 ? [yS, yK] : [yK, h];
      ctx.fillStyle = OV_LCOLORS[lvl];
      ctx.fillRect(0, y0, w, y1 - y0);
    }
    ctx.setLineDash([10 * u, 8 * u]);
    ctx.lineWidth = 1.5 * u;
    ctx.strokeStyle = 'rgba(255,255,255,0.55)';
    ctx.font = `${Math.round(13 * u)}px system-ui, sans-serif`;
    ctx.fillStyle = 'rgba(255,255,255,0.8)';
    ctx.textAlign = 'right';
    for (const [y, text] of [[yS, 'standing shoulder height'], [yK, 'standing knee height']]) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      ctx.fillText(text, w - 10 * u, y - 6 * u);
    }
    ctx.setLineDash([]);
    // Kinesphere spheres around the centre of the torso (midway between mid-hip and mid-shoulder).
    const wx = X((lm[11 * 4] + lm[12 * 4] + lm[23 * 4] + lm[24 * 4]) / 4);
    const wy = (lm[11 * 4 + 1] + lm[12 * 4 + 1] + lm[23 * 4 + 1] + lm[24 * 4 + 1]) / 4 * h;
    const kc = F.kClass[i];
    const R = a.kinesphere.radii.shared; // the same spheres in every frame
    [R.small, R.medium, R.large].forEach((r, ri) => {
      ctx.beginPath();
      ctx.arc(wx, wy, r * sc * h, 0, Math.PI * 2);
      ctx.lineWidth = (ri === kc ? 4 : 1.5) * u;
      ctx.strokeStyle = ri === kc ? OV_KCOLORS[ri] : 'rgba(255,255,255,0.35)';
      ctx.stroke();
      ctx.fillStyle = ri === kc ? OV_KCOLORS[ri] : 'rgba(255,255,255,0.6)';
      ctx.textAlign = 'left';
      ctx.font = `600 ${Math.round(13 * u)}px system-ui, sans-serif`;
      ctx.fillText(KINESPHERE_CLASSES[ri].letter, wx + r * sc * h * 0.71 + 4 * u, wy - r * sc * h * 0.71 - 4 * u);
    });
    ctx.beginPath(); ctx.arc(wx, wy, 3 * u, 0, Math.PI * 2); ctx.fillStyle = '#fff'; ctx.fill();
  }
  // Body part activity markers, sized and coloured by current speed.
  const speedNow = j => { const v = F.jointSpeed[j][i]; return Number.isFinite(v) ? v : 0; };
  for (let j = 0; j < 13; j++) {
    const k = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28][j];
    if (lm[k * 4 + 3] < 0.3) continue;
    const v = clamp(speedNow(j) / 3, 0, 1);
    ctx.beginPath();
    ctx.arc(X(lm[k * 4]), lm[k * 4 + 1] * h, (4 + 12 * v) * u, 0, Math.PI * 2);
    ctx.fillStyle = lerpHex('#184f95', '#cde2fb', v);
    ctx.globalAlpha = 0.85;
    ctx.fill();
    ctx.globalAlpha = 1;
  }
  // Left / right and upper / lower over the last second, and the most active part right now.
  let left = 0, right = 0, upper = 0, lower = 0;
  for (let j = i; j >= 0 && times[j] >= times[i] - 1000; j--) {
    for (const s of LEFT_SLOTS) { const v = F.jointSpeed[s][j]; if (Number.isFinite(v)) left += v; }
    for (const s of RIGHT_SLOTS) { const v = F.jointSpeed[s][j]; if (Number.isFinite(v)) right += v; }
    for (const s of UPPER_SLOTS) { const v = F.jointSpeed[s][j]; if (Number.isFinite(v)) upper += v; }
    for (const s of LOWER_SLOTS) { const v = F.jointSpeed[s][j]; if (Number.isFinite(v)) lower += v; }
  }
  let top = -1, topV = OV_ACTIVE;
  for (let j = 1; j < 13; j++) if (speedNow(j) > topV) { topV = speedNow(j); top = j; }
  const lr = left + right;
  const kc = F.kClass[i], lvl = F.level[i];
  const rows = [
    [`Kinesphere ${a.kinesphere.mode === '3d' ? '3D' : '2D'}`, kc >= 0 ? `${KINESPHERE_CLASSES[kc].label} · ${a.kinesphere.limbNames[F.kLimb[i]] ?? ''}` : '–', kc >= 0 ? OV_KCOLORS[kc] : '#fff'],
    ['Level', lvl >= 0 ? `${LEVELS[lvl].label}${F.floorMeasured[i] ? '' : ' (feet out of view)'}` : '–', '#fff'],
    ['Left / right', lr > 0 ? `${fmtPct(left / lr)} / ${fmtPct(right / lr)}` : '–', '#fff'],
    ['Upper / lower', upper + lower > 0 ? `${fmtPct(upper / (upper + lower))} / ${fmtPct(lower / (upper + lower))}` : '–', '#fff'],
    ['Most active', top >= 0 ? MAJOR_NAMES[top] : 'still', '#fff'],
  ];
  if (person != null) rows.unshift(['Person', String(person + 1), '#fff']);
  const fs = Math.round(15 * u), lh = fs * 1.45, padX = 12 * u, boxW = 260 * u, boxH = rows.length * lh + 14 * u;
  ctx.fillStyle = 'rgba(0,0,0,0.62)';
  ctx.fillRect(10 * u, 10 * u, boxW, boxH);
  rows.forEach(([label, value, color], r) => {
    const y = 10 * u + 7 * u + lh * (r + 0.75);
    ctx.textAlign = 'left';
    ctx.font = `${fs}px system-ui, sans-serif`;
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.fillText(label, 10 * u + padX, y);
    ctx.font = `700 ${fs}px system-ui, sans-serif`;
    ctx.fillStyle = color;
    ctx.fillText(value, 10 * u + padX + 110 * u, y);
  });
  ctx.restore();
}

/** The quiet "body partially out of frame" hint, bottom right of a canvas. */
function drawFramingHint(ctx, w, h) {
  ctx.save();
  ctx.font = `${Math.round(Math.max(12, h * 0.028))}px system-ui, sans-serif`;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'bottom';
  ctx.fillStyle = 'rgba(205, 209, 216, 0.85)';
  ctx.shadowColor = 'rgba(0, 0, 0, 0.6)'; ctx.shadowBlur = 3;
  ctx.fillText('Body partially out of frame', w - h * 0.025, h - h * 0.025);
  ctx.restore();
}

function setupReplay(session, root, onTime, metrics = null) {
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
  const smoother = createSmoother(state.smoother.id, people);
  const handSmoother = session.hands ? createSmoother(state.smoother.id, people, HAND_SIZE) : null;
  const minVis = minVisibilityFor(session.engine);
  let t = 0, playing = false, raf = 0, last = 0;
  let viewer = null;
  let mirrorView = session.mirrored; // how the dancer saw the preview; the toggle can flip it
  const poses3d = new Array(people).fill(null);
  const overlayBox = root.querySelector('#replay-overlay');
  let showMetrics = Boolean(metrics) && localStorage.getItem(OVERLAY_METRICS_KEY) === 'on';
  if (overlayBox) {
    overlayBox.checked = showMetrics;
    overlayBox.addEventListener('change', () => {
      showMetrics = Boolean(metrics) && overlayBox.checked;
      localStorage.setItem(OVERLAY_METRICS_KEY, overlayBox.checked ? 'on' : 'off');
      draw();
    });
  }
  const schedule = cb => { if (viewer) viewer.requestFrame(cb); else raf = requestAnimationFrame(cb); };

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
      if (!visible) { smoother.reset(p); handSmoother?.reset(p); poses3d[p] = null; effects[p].draw(ctx, null, 0, w, h, dt, { mirror: mirrorView, minVis, person: p, videoVisible: false }); continue; }
      const shown = smoother.apply(lm, offset, p, dt);
      const hands = handSmoother ? handSmoother.apply(session.hands, handOffset(session, Math.max(0, i), p), p, dt) : null;
      poses3d[p] = { lm: shown, offset: 0 };
      effects[p].draw(ctx, shown, 0, w, h, dt, { mirror: mirrorView, minVis, person: p, videoVisible: false, hands });
      if (people > 1) drawPersonBadge(ctx, shown, 0, w, h, p, { mirror: mirrorView, minVis });
    }
    if (showMetrics && hasFrame && poses3d[metrics.person]) {
      drawMetricsOverlay(ctx, metrics.analysis, i, poses3d[metrics.person].lm, w, h, mirrorView, times, people > 1 ? metrics.person : null);
    }
    if (hasFrame && Array.from({ length: people }, (_, p) => frameOffset(session, i, p)).some(o => personPresent(lm, o) && !bodyInFrame(lm, o, minVis))) {
      drawFramingHint(ctx, w, h);
    }
    viewer?.setPeople(poses3d, minVis);
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
    schedule(tick);
  };
  const togglePlay = () => {
    playing = !playing;
    btn.textContent = playing ? '❚❚' : '▶';
    if (playing) {
      if (t >= durationMs) t = 0;
      last = performance.now();
      schedule(tick);
    } else {
      cancelAnimationFrame(raf);
    }
  };
  btn.onclick = togglePlay;
  range.oninput = () => { effects.forEach(e => e.reset()); smoother.reset(); handSmoother?.reset(); setTime(Number(range.value)); };
  setTime(0);
  const api = {
    togglePlay,
    get mirrorView() { return mirrorView; },
    /** Flip the 2D replay and the 3D figure together. */
    setMirrorView(on) {
      mirrorView = Boolean(on);
      effects.forEach(e => e.reset());
      draw();
      api.setMirror?.(mirrorView);
    },
    /** Drive playback from the 3D viewer's loop (keeps running inside a WebXR session). */
    attachViewer(v) { viewer = v; if (v) { v.setPeople(poses3d, minVis); if (playing) { last = performance.now(); schedule(tick); } } },
    destroy() { playing = false; cancelAnimationFrame(raf); },
  };
  return api;
}

// ---------------------------------------------------------------------------------------
// Session library

async function renderSessions() {
  const list = store.list();
  const { used: usage, quota, backend } = await store.usage();
  const pct = quota ? clamp(usage / quota * 100, 0, 100) : 0;
  const quotaText = backend === 'localstorage' ? 'of about 5 MB browser storage used'
    : quota ? `of about ${fmtBytes(quota)} available in this browser used` : 'of browser storage used';
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
        <div class="usage"><span class="usage-bar"><span class="usage-fill" style="width:${pct.toFixed(0)}%"></span></span>${fmtBytes(usage)} ${quotaText}</div>
      </div>
      <div class="dash-actions">
        <button class="btn" data-action="import">Import…</button>
        <button class="btn btn-primary" data-action="new">New recording</button>
      </div>
    </div>
    ${list.length ? `<div class="session-list">${items}</div>` : `
      <div class="card empty">No sessions yet. Record one from the Live view, or import a session file that someone shared with you.<br>
      Sessions are stored only in this browser.</div>`}
  `;
}

ui.views.sessions.addEventListener('click', async e => {
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
    store.remove(id).catch(err => toast(`Could not delete: ${err.message}`)).finally(() => {
      if (state.current?.session.id === id) state.current = null;
      updateSessionCount();
      renderSessions();
    });
    return;
  }
  if (btn.disabled) return;
  btn.disabled = true;
  const session = await store.load(id).finally(() => { btn.disabled = false; });
  if (!session) { toast('This session could not be read from storage.'); return; }
  if (action === 'open') openSession(session);
  else if (action === 'export-json') exportJSON(session);
  else if (action === 'export-csv') exportCSV(session);
});

// ---------------------------------------------------------------------------------------
// Voice control: spoken commands change settings hands-free (Web Speech API).

const voice = createVoiceControl({ onPhrase: handlePhrase, onState: onVoiceState });
let voiceChipTimer = 0;

function onVoiceState({ listening, local, error }) {
  for (const btn of [ui.btnVoice, ui.btnVoiceHud]) {
    btn.classList.toggle('is-listening', Boolean(listening));
    btn.setAttribute('aria-pressed', String(Boolean(listening || voice.on)));
  }
  ui.btnVoice.textContent = listening ? '🎤 Voice: listening' : '🎤 Voice control';
  ui.btnVoiceHud.textContent = listening ? '🎤 Voice: on' : '🎤 Voice: off';
  const where = local ? 'transcribed on this device' : 'transcribed by your browser\'s speech service';
  ui.btnVoice.title = listening ? `Listening (${where}). Say “help” for commands, “stop listening” or press M to turn off.` : 'Change settings hands-free by speaking (M). Say “help” for the list of commands.';
  if (error) toast(error, 8000);
}

async function toggleVoice() {
  if (voice.on) { voice.disable(); showVoiceChip(null, 'Voice control off'); return; }
  if (!voice.supported) { toast('This browser has no speech recognition. Try Chrome, Edge or Safari.', 6000); return; }
  if (localStorage.getItem(VOICE_CONSENT_KEY) !== 'yes' && !(await voice.canRunLocally())) {
    const ok = confirm('Voice control uses your browser\'s speech recognition. This browser cannot transcribe on the device, so while voice control is on your voice is sent to its speech service (Google for Chrome, Apple for Safari, Microsoft for Edge) to be turned into text. Video never leaves your device.\n\nTurn on voice control?');
    if (!ok) return;
    localStorage.setItem(VOICE_CONSENT_KEY, 'yes');
  }
  await voice.enable();
  showVoiceChip(null, 'Listening: say “help” for commands');
}

function showVoiceChip(heard, message, miss = false) {
  if (state.view !== 'live') { // the chip lives on the camera stage; elsewhere use the toast
    if (!miss) toast(heard ? `“${heard}” → ${message}` : message, 2500);
    return;
  }
  const chip = ui.voiceChip;
  chip.innerHTML = `${heard ? `<q>${escapeHtml(heard)}</q>${message ? ' → ' : ''}` : '🎤 '}${message ? `<b>${escapeHtml(message)}</b>` : ''}`;
  chip.classList.toggle('is-miss', miss);
  chip.hidden = false;
  clearTimeout(voiceChipTimer);
  voiceChipTimer = setTimeout(() => { chip.hidden = true; }, miss ? 1500 : 2800);
}

function voiceHelpHTML(compact = false) {
  return VOICE_HELP.map(([group, items]) => `<div>${compact ? `<h4>${escapeHtml(group)}</h4>` : `<h3>${escapeHtml(group)}</h3>`}<ul>${items.map(([say, does]) => `<li>${escapeHtml(say)}${does ? ` <span>${escapeHtml(does)}</span>` : ''}</li>`).join('')}</ul></div>`).join('');
}
function showVoiceHelp(on) {
  ui.voiceHelp.hidden = !on;
  if (on) {
    ui.voiceHelp.innerHTML = `<button class="hud-btn vh-close" type="button" aria-label="Close">✕</button><h2>Say…</h2><div class="voice-help-grid">${voiceHelpHTML()}</div>`;
    ui.voiceHelp.querySelector('.vh-close').onclick = () => showVoiceHelp(false);
  }
}

/** Try each transcript the recogniser offers until one is a command. */
function handlePhrase(alternatives) {
  for (const text of alternatives) {
    const cmd = parseCommand(text);
    if (!cmd) continue;
    let msg;
    try { msg = runVoiceCommand(cmd); } catch (err) { console.error(err); msg = 'Could not do that'; }
    showVoiceChip(text, msg);
    return;
  }
  showVoiceChip(alternatives[0], 'not a command (say “help”)', true);
}

const onOffText = on => (on ? 'on' : 'off');
function runVoiceCommand(cmd) {
  const v = cmd.value;
  const cancelCountdown = () => { state.countdown = null; ui.countdown.hidden = true; };
  switch (cmd.type) {
    case 'voice:off': voice.disable(); return 'Voice control off';
    case 'help:show':
      if (state.view !== 'live') return 'Say “live view”, then “help”, to see the commands';
      showVoiceHelp(true);
      return 'Voice commands';
    case 'help:hide': showVoiceHelp(false); return 'Closed';
    case 'record:start':
      if (state.view !== 'live' || game.isOpen) return 'Recording starts from the Live view';
      if (!state.detector) return 'Start the camera first';
      if (state.recording) return 'Already recording';
      state.countdown = { endsAt: performance.now() + COUNTDOWN_MS, trigger: 'voice' };
      return 'Recording in 3 seconds';
    case 'record:cancel':
      if (state.countdown) { cancelCountdown(); return 'Countdown cancelled'; }
      return 'Nothing to cancel';
    case 'stop':
      if (game.isPlaying) { game.back(); return 'Song stopped'; }
      if (state.countdown) { cancelCountdown(); return 'Countdown cancelled'; }
      if (state.recording) { stopRecording(); return 'Recording stopped'; }
      return 'Not recording';
    case 'game:stop':
      if (game.isPlaying) { game.back(); return 'Song stopped'; }
      return 'No song is playing';
    case 'game:play': {
      if (!game.isOpen) setView('game');
      if (!(state.stream && state.detector)) return 'Start the camera first';
      if (state.recording && !game.isPlaying) return 'Stop the recording first';
      game.play(v);
      return `Playing ${SONGS.find(sg => sg.id === v)?.title ?? v}`;
    }
    case 'game:volume': return `Music volume ${fmtPct(game.adjustVolume(v, cmd.absolute))}`;
    case 'nav':
      if (v === 'dashboard') {
        if (!state.current) return 'No session is open yet';
        setView('dashboard');
        return 'Dashboard';
      }
      if (state.recording && v !== 'live') return 'Stop the recording first';
      setView(v);
      return { live: 'Live view', game: 'Game', sessions: 'Sessions' }[v];
    case 'fullscreen': {
      const want = v ?? !document.fullscreenElement;
      if (!want) { if (document.fullscreenElement) document.exitFullscreen?.(); return 'Fullscreen off'; }
      if (document.fullscreenElement) return 'Already fullscreen';
      // Browsers only allow entering fullscreen from a click or key press.
      ui.stage.requestFullscreen?.().catch(() => showVoiceChip(null, 'Fullscreen needs a click or the F key (a browser rule)'));
      return 'Fullscreen';
    }
    case 'overlay': { const on = v ?? !state.showOverlay; setOverlay(on); return `Overlay ${onOffText(on)}`; }
    case 'video': setVideo(v); return `Camera feed ${onOffText(v)}`;
    case 'mirror': { const on = v ?? !state.mirrored; applyLiveMirror(on); return `Mirror view ${onOffText(on)}`; }
    case 'view3d':
      if (!state.stream) return 'Start the camera first';
      if (Boolean(state.liveViewer) !== v) toggleLiveViewer();
      return `3D view ${onOffText(v)}`;
    case 'smoothing': setSmoothing(v); return `Smoothing: ${SMOOTHING_LEVELS.find(l => l.id === v)?.label ?? v}`;
    case 'model':
      if (state.recording) return 'Can’t change the model while recording';
      ui.selModel.value = v;
      onSettingsChanged();
      return `Model: ${engineInfo(v).label}`;
    case 'people':
      if (state.recording) return 'Can’t change this while recording';
      ui.selPeople.value = String(clamp(v, 1, MAX_PEOPLE));
      onPeopleChanged();
      return v === 1 ? 'Tracking one person' : `Tracking up to ${v} people`;
    case 'sand': {
      if (state.styleId !== 'sandman') setStyle('sandman');
      if (v === 'reset') { applySand(SAND_DEFAULTS); localStorage.removeItem(SAND_KEY); return 'Sandman settings reset'; }
      const patch = {};
      for (const k of ['hue', 'gravity', 'decay', 'momentum']) if (k in v) patch[k] = v.relative ? sandSettings[k] + v[k] : v[k];
      applySand(patch);
      if ('hue' in v) return 'Sand colour changed';
      if ('gravity' in v) return `Gravity ${sandSettings.gravity.toFixed(1)}×`;
      if ('decay' in v) return `Decay ${sandSettings.decay.toFixed(1)} s`;
      return `Momentum ${sandSettings.momentum.toFixed(1)}×`;
    }
    case 'style': {
      if (v === 'next') nextStyle();
      else if (v === 'previous') { const i = STYLES.findIndex(st => st.id === state.styleId); setStyle(STYLES[(i - 1 + STYLES.length) % STYLES.length].id); }
      else setStyle(v);
      return `Style: ${STYLES.find(st => st.id === state.styleId)?.label}`;
    }
    default: return null;
  }
}

function initVoice() {
  if (!voiceSupported) {
    $('#voice-help-text').insertAdjacentHTML('beforeend', ' <b>This browser has no speech recognition, so the voice control button is hidden.</b>');
  } else {
    ui.btnVoice.hidden = false;
    ui.btnVoiceHud.hidden = false;
    ui.btnVoice.addEventListener('click', toggleVoice);
    ui.btnVoiceHud.addEventListener('click', toggleVoice);
  }
  $('#voice-commands').innerHTML = voiceHelpHTML(true);
}

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
  renderModelMatrix();
  ui.selModel.addEventListener('change', onSettingsChanged);
  ui.selPeople.addEventListener('change', onPeopleChanged);
  ui.btnHelp.addEventListener('click', () => {
    ui.helpPanel.open = !ui.helpPanel.open;
    ui.btnHelp.setAttribute('aria-expanded', String(ui.helpPanel.open));
    if (ui.helpPanel.open) ui.helpPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });
  ui.helpPanel.addEventListener('toggle', () => {
    ui.btnHelp.setAttribute('aria-expanded', String(ui.helpPanel.open));
    if (ui.helpPanel.open) refreshCacheInfo();
  });
  ui.btnClearCache.addEventListener('click', async () => {
    await clearModelCache();
    await refreshCacheInfo();
    toast('Cached models removed. The next model load will download again.');
  });
  refreshCacheInfo();

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
  initSandControls();
  initVoice();
  ui.btnStyle.addEventListener('click', nextStyle);
  setStyle(state.styleId);
  ui.btnVideo.addEventListener('click', () => setVideo(!state.showVideo));
  setVideo(state.showVideo);
  ui.btnLive3d.addEventListener('click', toggleLiveViewer);
  wireDepthSlider(ui.liveDepth, ui.liveDepthVal);
  ui.chkMirror.addEventListener('change', () => applyLiveMirror(ui.chkMirror.checked));
  ui.selSmooth.replaceChildren(...SMOOTHING_LEVELS.map(l => {
    const opt = document.createElement('option');
    opt.value = l.id;
    opt.textContent = l.label;
    opt.title = l.help;
    return opt;
  }));
  ui.selSmooth.addEventListener('change', () => setSmoothing(ui.selSmooth.value));
  setSmoothing(state.smoother.id);
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
    if (e.code === 'Escape' && !ui.voiceHelp.hidden) { e.preventDefault(); showVoiceHelp(false); return; }
    if (game.isOpen && e.code === 'Escape' && !document.fullscreenElement) { e.preventDefault(); game.back(); return; }
    if (e.code === 'KeyF' && (state.stream || game.isOpen)) { e.preventDefault(); toggleFullscreen(); return; }
    if (e.code === 'KeyO') { e.preventDefault(); setOverlay(!state.showOverlay); return; }
    if (e.code === 'KeyS') { e.preventDefault(); nextStyle(); return; }
    if (e.code === 'KeyV') { e.preventDefault(); setVideo(!state.showVideo); return; }
    if (e.code === 'KeyM' && voiceSupported) { e.preventDefault(); toggleVoice(); return; }
    if (e.code !== 'Space' || tag === 'BUTTON' || !state.detector || game.isOpen) return;
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
  store.init().then(() => { if (state.view === 'sessions') renderSessions(); });
}

const game = createGame({
  stage: ui.stage, video: ui.video, overlay: ui.canvas,
  isReady: () => Boolean(state.stream && state.detector),
  players: () => state.detector?.people ?? 1,
  startCamera,
  startRecording: () => startRecording('game'),
  stopRecording: ({ discard = false, name = null } = {}) => stopRecording(null, { navigate: false, discard, name }),
  openDashboard: session => openSession(session, { save: true }),
  toggleFullscreen,
  onClose: () => setView('live'),
});

init();
