// Dance game mode, in the spirit of Just Dance: an animated coach demonstrates a routine to an
// original, procedurally generated track; pictograms of the upcoming moves scroll towards a target;
// every tracked player is scored on each move's beat. Pure logic lives in game-core.js.

import {
  SONGS, MOVES, buildChoreography, posePoints, lerpMove, playerAngles, matchScore, rate, RATINGS,
  starsFor, maxPointsFor, STAR_LEVELS,
} from './game-core.js';
import { personColors } from './effects.js';
import { bodyInFrame } from './pose.js';

const BEST_KEY = 'kinesphere:gameBest';
const VOLUME_KEY = 'kinesphere:gameVolume';
const RECORD_KEY = 'kinesphere:gameRecord';
const INPUT_LAG = 0.1;       // s: camera + detection latency, the player is seen this much late
const WINDOW_BEFORE = 0.4;   // s: a move can be hit this early ...
const WINDOW_AFTER = 0.3;    // ... and this late
const LOOKAHEAD = 3.2;       // s of upcoming pictograms on the lane
const COMBO_KEYS = new Set(['perfect', 'super', 'good']);

const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const ease = x => (x < 0.5 ? 2 * x * x : 1 - (-2 * x + 2) ** 2 / 2);
const fmtTime = s => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, '0')}`;

function readJSON(key, fallback) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch { return fallback; }
}

// ---- music: a small Web Audio synth playing an original track per song ------------------------

function createMusic(song, chor, volume) {
  const AC = window.AudioContext || window.webkitAudioContext;
  const ctx = new AC();
  const master = ctx.createGain();
  master.gain.value = volume;
  const comp = ctx.createDynamicsCompressor();
  comp.threshold.value = -14; comp.ratio.value = 4;
  comp.connect(master); master.connect(ctx.destination);
  const noise = ctx.createBuffer(1, ctx.sampleRate, ctx.sampleRate);
  const nd = noise.getChannelData(0);
  for (let i = 0; i < nd.length; i++) nd[i] = Math.random() * 2 - 1;
  const hz = m => 440 * 2 ** ((m - 69) / 12);
  const env = (g, t, peak, attack, decay) => {
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(peak, t + attack);
    g.gain.exponentialRampToValueAtTime(0.0001, t + attack + decay);
  };
  const voice = (type, freq, t, peak, attack, decay, filter) => {
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.type = type; o.frequency.setValueAtTime(freq, t);
    let node = o;
    if (filter) { const f = ctx.createBiquadFilter(); f.type = 'lowpass'; f.frequency.value = filter; f.Q.value = 3; o.connect(f); node = f; }
    node.connect(g); g.connect(comp);
    env(g, t, peak, attack, decay);
    o.start(t); o.stop(t + attack + decay + 0.05);
    return o;
  };
  const noiseHit = (t, type, freq, peak, decay) => {
    const s = ctx.createBufferSource(), f = ctx.createBiquadFilter(), g = ctx.createGain();
    s.buffer = noise; f.type = type; f.frequency.value = freq;
    s.connect(f); f.connect(g); g.connect(comp);
    env(g, t, peak, 0.002, decay);
    s.start(t, Math.random() * 0.5); s.stop(t + decay + 0.05);
  };
  const kick = (t, v = 1) => {
    const o = voice('sine', 150, t, 0.9 * v, 0.003, 0.28);
    o.frequency.exponentialRampToValueAtTime(42, t + 0.14);
  };
  const snare = t => { noiseHit(t, 'bandpass', 1900, 0.32, 0.16); voice('triangle', 190, t, 0.18, 0.002, 0.08); };
  const hat = (t, open = false, v = 1) => noiseHit(t, 'highpass', 7500, (open ? 0.12 : 0.07) * v, open ? 0.2 : 0.035);
  const click = (t, accent) => voice('sine', accent ? 1760 : 1320, t, 0.25, 0.002, 0.06);
  const crash = t => noiseHit(t, 'highpass', 4000, 0.2, 1.4);

  const beat = chor.beat, step = beat / 4, totalSteps = chor.bars * 16;
  let startAt = 0, next = 0, timer = 0;
  const chordAt = bar => {
    const [root, kind] = song.chords[(bar - 1 + song.chords.length * 16) % song.chords.length];
    return { root, third: root + (kind === 'maj' ? 4 : 3), fifth: root + 7 };
  };
  function schedule(n, t) {
    const bar = Math.floor(n / 16), s = n % 16;
    if (bar === 0) { // count-in
      if (s % 4 === 0) click(t, s === 0);
      if (s % 2 === 0) hat(t, false, 0.6);
      return;
    }
    if (bar === chor.bars - 1) { // outro: one big final hit
      if (s === 0) { kick(t); crash(t); const c = chordAt(bar); for (const m of [c.root, c.third, c.fifth]) voice('sawtooth', hz(m), t, 0.05, 0.01, 1.6, 1800); voice('sawtooth', hz(c.root - 24), t, 0.25, 0.01, 1.4, 500); }
      return;
    }
    const c = chordAt(bar);
    const chorus = Math.floor((bar - 1) / 8) % 2 === 1;
    const st = song.style;
    // drums
    if (st === 'pop') { if (s === 0 || s === 8 || (s === 10 && bar % 2)) kick(t); }
    else if (s % 4 === 0) kick(t);
    if (s === 4 || s === 12) snare(t);
    if (st === 'edm') hat(t, false, s % 2 ? 0.6 : 1);
    else if (st === 'disco') { if (s % 4 === 2) hat(t, true); else if (s % 2 === 0) hat(t, false, 0.7); }
    else if (s % 2 === 0) hat(t, false, s % 4 ? 0.7 : 1);
    if (s === 0 && (bar - 1) % 8 === 0) crash(t);
    // bass
    if (st === 'disco') { if (s % 2 === 0) voice('sawtooth', hz(c.root - 24 + (s % 4 ? 12 : 0)), t, 0.22, 0.005, step * 1.6, 700); }
    else if (st === 'edm') { if (s % 4 === 2) voice('sawtooth', hz(c.root - 24), t, 0.26, 0.005, step * 1.8, 600); }
    else if (s === 0 || s === 6 || s === 8) voice('triangle', hz(c.root - 24), t, 0.3, 0.005, step * 3, 900);
    // chords
    if (st === 'edm') { if (s === 0) for (const m of [c.root, c.third, c.fifth]) voice('sawtooth', hz(m), t, 0.035, 0.08, beat * 3.6, 1400); }
    else if ((st === 'disco' && s % 8 === 4) || (st === 'pop' && (s === 4 || s === 12))) {
      for (const m of [c.root, c.third, c.fifth]) voice(st === 'disco' ? 'square' : 'triangle', hz(m), t, 0.045, 0.005, step * 1.5, 2600);
    }
    // lead arpeggio in the chorus
    if (chorus) {
      const notes = [c.root + 12, c.third + 12, c.fifth + 12, c.third + 12];
      const every = st === 'edm' ? 1 : 2;
      if (s % every === 0) voice('square', hz(notes[(s / every) % 4]), t, 0.05, 0.004, step * every * 0.9, 3200);
    }
  }
  function tick() {
    while (next < totalSteps && startAt + next * step < ctx.currentTime + 0.15) { schedule(next, startAt + next * step); next++; }
  }
  return {
    async start() {
      await ctx.resume();
      startAt = ctx.currentTime + 0.15;
      next = 0;
      tick();
      timer = setInterval(tick, 25);
    },
    /** Song time in seconds, as heard (output latency taken off). */
    time() { return ctx.currentTime - startAt - (ctx.outputLatency || ctx.baseLatency || 0); },
    setVolume(v) { master.gain.setTargetAtTime(v, ctx.currentTime, 0.05); },
    stop() {
      clearInterval(timer);
      try { master.gain.setTargetAtTime(0.0001, ctx.currentTime, 0.08); } catch { /* closed */ }
      setTimeout(() => ctx.close().catch(() => {}), 400);
    },
  };
}

// ---- drawing helpers ----------------------------------------------------------------------------

const BONES = [['shL', 'shR'], ['shL', 'elL'], ['elL', 'wrL'], ['shR', 'elR'], ['elR', 'wrR'], ['neck', 'head'],
  ['shL', 'hipL'], ['shR', 'hipR'], ['hipL', 'hipR'], ['hipL', 'knL'], ['knL', 'anL'], ['hipR', 'knR'], ['knR', 'anR']];

function drawFigure(ctx, P, cx, floorY, u, { color = '#8ec5ff', glow = 18, width = 0.16, headFill = null } = {}) {
  const X = p => cx + p[0] * u, Y = p => floorY + (p[1] - 1.75) * u;
  ctx.save();
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.strokeStyle = color; ctx.shadowColor = color; ctx.shadowBlur = glow;
  ctx.lineWidth = width * u;
  ctx.beginPath();
  for (const [a, b] of BONES) {
    if (a === 'neck') continue;
    ctx.moveTo(X(P[a]), Y(P[a])); ctx.lineTo(X(P[b]), Y(P[b]));
  }
  ctx.moveTo(X(P.neck), Y(P.neck)); ctx.lineTo(X(P.head), Y(P.head) + 0.26 * u);
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(X(P.head), Y(P.head), 0.28 * u, 0, Math.PI * 2);
  if (headFill) { ctx.fillStyle = headFill; ctx.fill(); }
  ctx.stroke();
  ctx.restore();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
}

function starRow(n, superstar) {
  return Array.from({ length: 5 }, (_, i) => `<span class="${i < n ? 'on' : ''}">★</span>`).join('') + (superstar ? ' <b class="superstar">Superstar!</b>' : '');
}

// ---- the game -------------------------------------------------------------------------------------

/**
 * @param {object} deps hooks into the app: stage, video, overlay (canvas), isReady(), players(),
 *   startCamera(), startRecording(), stopRecording() -> session|null, openDashboard(session),
 *   toggleFullscreen(), onClose()
 */
export function createGame(deps) {
  const layer = document.createElement('div');
  layer.className = 'game-layer';
  layer.hidden = true;
  layer.innerHTML = `
    <canvas class="game-canvas" aria-hidden="true"></canvas>
    <div class="game-panel game-menu" role="dialog" aria-label="Dance game"></div>
    <div class="game-panel game-results" role="dialog" aria-label="Results" hidden></div>
    <div class="game-buttons">
      <button class="hud-btn" data-g="fullscreen" title="Fullscreen (F)">⛶</button>
      <button class="hud-btn" data-g="quit" title="Back to the songs (Esc)">✕</button>
    </div>`;
  deps.stage.append(layer);
  const canvas = layer.querySelector('.game-canvas');
  const ctx = canvas.getContext('2d');
  const menu = layer.querySelector('.game-menu');
  const results = layer.querySelector('.game-results');
  const quitBtn = layer.querySelector('[data-g="quit"]');
  layer.querySelector('[data-g="fullscreen"]').onclick = () => deps.toggleFullscreen();
  quitBtn.onclick = () => back();

  let open = false, phase = 'menu'; // 'menu' | 'playing' | 'results'
  let run = null, raf = 0, layout = null;
  const savedStyles = new Map();

  // ---- layout: coach area, player picture-in-picture, pictogram lane ----
  function computeLayout() {
    const r = deps.stage.getBoundingClientRect();
    const W = Math.max(320, r.width), H = Math.max(200, r.height);
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
      canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    }
    const wide = W / H >= 1.2;
    const top = Math.max(0.1 * H, 50); // clear of the fullscreen / quit buttons
    // The camera picture-in-picture takes the video's own shape, fitted into its slot.
    const va = deps.video.videoWidth && deps.video.videoHeight ? deps.video.videoWidth / deps.video.videoHeight : 16 / 9;
    const fit = (x, y, w, h, alignRight) => {
      const fw = Math.min(w, h * va), fh = fw / va;
      return { x: alignRight ? x + w - fw : x + (w - fw) / 2, y, w: fw, h: fh };
    };
    if (wide) {
      const pip = fit(0.62 * W, top, 0.36 * W, 0.58 * H, false);
      const coach = { x: 0.02 * W, y: top, w: 0.58 * W, h: 0.79 * H - top };
      layout = {
        W, H, dpr, wide, pip, coach,
        meter: { x: pip.x, y: pip.y + pip.h + Math.max(26, 0.07 * H), w: pip.w },
        title: { x: coach.x + coach.w / 2, y: top * 0.62 },
      };
    } else {
      const pip = fit(0.5 * W, top, 0.48 * W, 0.22 * H, true);
      const coachTop = Math.max(pip.y + pip.h, 0.3 * H) + 0.02 * H;
      layout = {
        W, H, dpr, wide, pip,
        coach: { x: 0.02 * W, y: coachTop, w: 0.96 * W, h: 0.79 * H - coachTop },
        meter: { x: 14, y: top + Math.max(0.08 * H, 30), w: 0.44 * W - 14 },
        title: { x: 0.25 * W, y: coachTop - 0.02 * H },
      };
    }
    layout.lane = { x: 0.02 * W, y: 0.81 * H, w: 0.96 * W, h: 0.16 * H };
    if (phase === 'playing') placePip();
  }
  function placePip() {
    const { pip, W, H } = layout;
    for (const el of [deps.video, deps.overlay]) {
      if (!savedStyles.has(el)) savedStyles.set(el, el.getAttribute('style'));
      Object.assign(el.style, {
        inset: 'auto', left: `${(pip.x / W) * 100}%`, top: `${(pip.y / H) * 100}%`,
        width: `${(pip.w / W) * 100}%`, height: `${(pip.h / H) * 100}%`, zIndex: '2',
        transform: 'scaleX(-1)', // always a mirror, as the player is matched against the coach like one
      });
    }
  }
  function restorePip() {
    for (const [el, st] of savedStyles) { if (st == null) el.removeAttribute('style'); else el.setAttribute('style', st); }
    savedStyles.clear();
  }
  const ro = new ResizeObserver(() => { if (open) { computeLayout(); if (phase !== 'playing') drawIdle(); } });
  ro.observe(deps.stage);

  // ---- menu ----
  function renderMenu() {
    const best = readJSON(BEST_KEY, {});
    const ready = deps.isReady();
    const volume = Number(localStorage.getItem(VOLUME_KEY) ?? 0.7);
    const record = localStorage.getItem(RECORD_KEY) === 'on';
    menu.innerHTML = `
      <h2>Dance game</h2>
      <p class="game-how">Follow the coach <b>like a mirror</b>: when the coach moves the arm on the left of the screen, move your left arm.
        Each move is scored on its beat: <b>Perfect</b>, <b>Super</b>, <b>Good</b>, <b>OK</b> or <b>X</b>. Gold moves score double.
        Everyone the camera tracks is scored (set <i>People</i> below the preview for a group).</p>
      ${ready ? '' : `<div class="game-needcam"><span>The game needs the camera and pose model.</span> <button class="btn btn-primary btn-sm" data-g="camera">Start camera</button></div>`}
      <div class="game-songs">
        ${SONGS.map(s => {
          const c = buildChoreography(s);
          const b = best[s.id];
          const st = b ? starsFor(b.points, maxPointsFor(c.moves)) : null;
          return `<button class="game-song" data-song="${s.id}"${ready ? '' : ' disabled'}>
            <span class="game-song-title">${esc(s.title)}</span>
            <span class="game-chip game-chip-${s.difficulty.toLowerCase()}">${esc(s.difficulty)}</span>
            <span class="game-song-meta">${s.bpm} BPM · ${fmtTime(c.duration)} · ${c.moves.length} moves</span>
            <span class="game-song-blurb">${esc(s.blurb)}</span>
            <span class="game-song-best">${b ? `Best ${b.points.toLocaleString()} <span class="game-stars">${starRow(st.stars, st.superstar)}</span>` : 'Not played yet'}</span>
          </button>`;
        }).join('')}
      </div>
      <div class="game-options">
        <label class="select">Music volume <input type="range" min="0" max="1" step="0.05" value="${volume}" data-g="volume" aria-label="Music volume"></label>
        <label class="check"><input type="checkbox" data-g="record"${record ? ' checked' : ''}> Record my dance for the dashboard</label>
      </div>`;
    menu.querySelector('[data-g="camera"]')?.addEventListener('click', async e => {
      e.currentTarget.disabled = true;
      e.currentTarget.textContent = 'Starting camera and pose model…';
      await deps.startCamera();
      if (open && phase === 'menu') renderMenu();
    });
    menu.querySelector('[data-g="volume"]').addEventListener('input', e => { localStorage.setItem(VOLUME_KEY, e.target.value); run?.music.setVolume(Number(e.target.value)); });
    menu.querySelector('[data-g="record"]').addEventListener('change', e => localStorage.setItem(RECORD_KEY, e.target.checked ? 'on' : 'off'));
    menu.querySelectorAll('[data-song]').forEach(b => { b.onclick = () => startSong(b.dataset.song); });
  }

  function setPhase(p) {
    phase = p;
    menu.hidden = p !== 'menu';
    results.hidden = p !== 'results';
    deps.stage.classList.toggle('game-playing', p === 'playing');
    quitBtn.title = p === 'playing' ? 'Stop the song (Esc)' : p === 'results' ? 'Back to the songs (Esc)' : 'Leave the game (Esc)';
    if (p === 'playing') placePip(); else restorePip();
  }

  // ---- playing ----
  async function startSong(id) {
    if (!deps.isReady()) { renderMenu(); return; }
    const song = SONGS.find(s => s.id === id);
    const chor = buildChoreography(song);
    const players = Math.max(1, deps.players());
    const volume = Number(localStorage.getItem(VOLUME_KEY) ?? 0.7);
    run = {
      song, chor, players,
      music: createMusic(song, chor, volume),
      moves: chor.moves.map(m => ({ ...m, best: new Array(players).fill(0), result: null })),
      scores: Array.from({ length: players }, () => ({ points: 0, combo: 0, maxCombo: 0, counts: Object.fromEntries(RATINGS.map(r => [r.key, 0])), seen: false })),
      popups: [], live: new Array(players).fill(0), aspect: 16 / 9, framing: { bad: false, since: 0, show: false },
      recording: localStorage.getItem(RECORD_KEY) === 'on', session: null,
    };
    computeLayout();
    setPhase('playing');
    await run.music.start();
    if (run.recording) deps.startRecording();
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(frame);
  }

  function finalize(m) {
    m.result = m.best.map((b, p) => {
      const r = rate(b);
      const sc = run.scores[p];
      sc.points += r.points * (m.gold ? 2 : 1);
      sc.counts[r.key]++;
      sc.combo = COMBO_KEYS.has(r.key) ? sc.combo + 1 : 0;
      sc.maxCombo = Math.max(sc.maxCombo, sc.combo);
      if (sc.seen || r.key !== 'x') run.popups.push({ p, r, gold: m.gold, at: performance.now() });
      return r;
    });
  }

  function frame() {
    if (phase !== 'playing' || !run) return;
    const t = run.music.time();
    for (const m of run.moves) if (!m.result && t - INPUT_LAG > m.time + WINDOW_AFTER) finalize(m);
    draw(t);
    if (t >= run.chor.duration) { finish(); return; }
    raf = requestAnimationFrame(frame);
  }

  function finish() {
    const r = run;
    r.music.stop();
    if (r.recording) r.session = deps.stopRecording({ name: r.song.title });
    const maxPts = maxPointsFor(r.chor.moves);
    const best = readJSON(BEST_KEY, {});
    const top = Math.max(...r.scores.map(s => s.points));
    const prev = best[r.song.id]?.points ?? -1;
    const newBest = top > prev;
    if (newBest) { best[r.song.id] = { points: top, at: new Date().toISOString() }; localStorage.setItem(BEST_KEY, JSON.stringify(best)); }
    results.innerHTML = `
      <h2>${esc(r.song.title)}</h2>
      ${newBest && prev >= 0 ? '<p class="game-newbest">New personal best!</p>' : newBest ? '<p class="game-newbest">First score on this song!</p>' : ''}
      <div class="game-result-players">
        ${r.scores.map((s, p) => {
          const st = starsFor(s.points, maxPts);
          return `<div class="game-result">
            ${r.players > 1 ? `<div class="game-result-who"><i style="background:${personColors(p).left}"></i>Player ${p + 1}</div>` : ''}
            <div class="game-stars big">${starRow(st.stars, st.superstar)}</div>
            <div class="game-result-score">${s.points.toLocaleString()}</div>
            <div class="meta">${Math.round(st.fraction * 100)}% of ${maxPts.toLocaleString()} · best combo ${s.maxCombo}</div>
            <div class="game-counts">${RATINGS.map(x => `<span style="--c:${x.color}"><b>${s.counts[x.key]}</b> ${x.label}</span>`).join('')}</div>
          </div>`;
        }).join('')}
      </div>
      <div class="game-result-actions">
        <button class="btn btn-primary" data-g="again">Play again</button>
        <button class="btn" data-g="songs">Choose a song</button>
        ${r.session ? '<button class="btn" data-g="dash">Open the dance in the dashboard</button>' : ''}
      </div>`;
    results.querySelector('[data-g="again"]').onclick = () => startSong(r.song.id);
    results.querySelector('[data-g="songs"]').onclick = () => { setPhase('menu'); renderMenu(); drawIdle(); };
    results.querySelector('[data-g="dash"]')?.addEventListener('click', () => deps.openDashboard(r.session));
    setPhase('results');
    drawIdle();
  }

  function stopSong() {
    if (!run) return;
    cancelAnimationFrame(raf);
    run.music.stop();
    if (run.recording) deps.stopRecording({ discard: true });
    run = null;
  }

  /** Called by the app with each detection: per-slot poses ({ lm } or null). */
  function onPoses(slots, perfNow, aspect) {
    if (phase !== 'playing' || !run) return;
    run.aspect = aspect || run.aspect;
    const t = run.music.time() - (performance.now() - perfNow) / 1000 - INPUT_LAG;
    const angles = slots.slice(0, run.players).map(pose => (pose ? playerAngles(pose.lm, run.aspect) : null));
    angles.forEach((a, p) => { if (a) run.scores[p].seen = true; });
    const f = run.framing, present = slots.slice(0, run.players).filter(Boolean);
    const bad = present.some(pose => !bodyInFrame(pose.lm, 0, 0.5));
    if (bad !== f.bad) { f.bad = bad; f.since = perfNow; }
    if (perfNow - f.since > 600) f.show = bad && present.length > 0;
    let current = null;
    for (const m of run.moves) {
      if (m.result) continue;
      if (t < m.time - WINDOW_BEFORE) { current ??= m; break; }
      if (t > m.time + WINDOW_AFTER) continue;
      current ??= m;
      angles.forEach((a, p) => { if (a) m.best[p] = Math.max(m.best[p], matchScore(m.move, a)); });
    }
    if (current) angles.forEach((a, p) => { run.live[p] = a ? matchScore(current.move, a) : 0; });
  }

  // ---- rendering ----
  function coachPose(t) {
    const ms = run.moves;
    let j = ms.findIndex(m => m.time >= t);
    if (j < 0) return { pose: ms[ms.length - 1].move, gold: false, next: null };
    const prev = j > 0 ? ms[j - 1] : { move: MOVES.ready, time: 0 };
    const target = ms[j];
    const trans = Math.min(0.45, 0.6 * (target.time - prev.time));
    const k = clamp((t - (target.time - trans)) / trans, 0, 1);
    return { pose: lerpMove(prev.move, target.move, ease(k)), gold: target.gold && k > 0, next: target };
  }

  function draw(t) {
    const { W, H, dpr, coach, pip, lane, meter, title } = layout;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const bg = ctx.createLinearGradient(0, 0, W, H);
    bg.addColorStop(0, '#1a1036'); bg.addColorStop(1, '#0b1a2e');
    ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);
    const beat = run.chor.beat;
    const beatPhase = ((t / beat) % 1 + 1) % 1;
    // song progress
    ctx.fillStyle = 'rgba(255,255,255,0.12)'; ctx.fillRect(0, 0, W, 4);
    ctx.fillStyle = '#ff7ad9'; ctx.fillRect(0, 0, W * clamp(t / run.chor.duration, 0, 1), 4);
    // floor glow pulsing on the beat
    const floorY = coach.y + coach.h * 0.97;
    const glow = ctx.createRadialGradient(coach.x + coach.w / 2, floorY, 2, coach.x + coach.w / 2, floorY, coach.w * 0.45);
    glow.addColorStop(0, `rgba(255,122,217,${0.35 - 0.25 * beatPhase})`); glow.addColorStop(1, 'rgba(255,122,217,0)');
    ctx.fillStyle = glow; ctx.fillRect(coach.x, floorY - coach.w * 0.2, coach.w, coach.w * 0.25);
    // coach
    const { pose, gold, next } = coachPose(t);
    const u = coach.h / 4.15; // floor to raised hands is about 3.9 torso lengths
    const bounce = -0.05 * Math.abs(Math.sin(Math.PI * t / beat)) * u;
    drawFigure(ctx, posePoints(pose), coach.x + coach.w / 2, floorY + bounce, u, { color: gold ? '#ffd166' : '#7cf6ff', glow: 24, headFill: 'rgba(124,246,255,0.15)' });
    // count-in / move name
    ctx.textAlign = 'center';
    if (t < run.chor.countInBeats * beat) {
      const n = Math.floor(t / beat);
      const text = n < 0 ? '' : n < 3 ? String(3 - n) : 'Dance!';
      ctx.font = `800 ${Math.round(coach.h * 0.28)}px system-ui, sans-serif`;
      ctx.fillStyle = 'rgba(255,255,255,0.92)'; ctx.shadowColor = '#ff7ad9'; ctx.shadowBlur = 20;
      ctx.fillText(text, coach.x + coach.w / 2, coach.y + coach.h * 0.45);
      ctx.shadowBlur = 0;
    }
    if (next) {
      ctx.font = `600 ${Math.round(Math.max(13, H * 0.036))}px system-ui, sans-serif`;
      ctx.fillStyle = next.gold ? '#ffd166' : 'rgba(255,255,255,0.8)';
      ctx.fillText(`${next.gold ? '★ ' : ''}${next.move.name}`, title.x, title.y);
    }
    // player picture-in-picture frame, glowing with how well player 1 matches right now
    const live = run.live[0] || 0;
    ctx.save();
    roundRect(ctx, pip.x - 3, pip.y - 3, pip.w + 6, pip.h + 6, 12);
    ctx.lineWidth = 3; ctx.strokeStyle = rate(live).color; ctx.shadowColor = rate(live).color; ctx.shadowBlur = 8 + 20 * live;
    ctx.stroke(); ctx.restore();
    ctx.fillStyle = 'rgba(0,0,0,0.35)'; ctx.fillRect(pip.x, pip.y, pip.w, pip.h);
    if (run.framing.show) {
      ctx.font = `${Math.round(Math.max(11, H * 0.026))}px system-ui, sans-serif`;
      ctx.fillStyle = 'rgba(205,209,216,0.85)';
      // the video covers the PiP box, so the hint goes just outside it, clear of the buttons and meter
      if (layout.wide) { ctx.textAlign = 'left'; ctx.fillText('Body partially out of frame', pip.x, pip.y - 7); }
      else { ctx.textAlign = 'right'; ctx.fillText('Body partially out of frame', pip.x + pip.w, pip.y + pip.h + Math.max(14, H * 0.03)); }
    }
    // scores
    const maxPts = maxPointsFor(run.chor.moves);
    const fs = Math.round(Math.max(13, H * 0.04));
    run.scores.forEach((s, p) => {
      const y = 14 + fs * 1.3 * (p + 0.8);
      ctx.textAlign = 'left';
      ctx.font = `700 ${fs}px system-ui, sans-serif`;
      ctx.fillStyle = run.players > 1 ? personColors(p).left : '#fff';
      ctx.fillText(`${run.players > 1 ? `P${p + 1} ` : ''}${s.points.toLocaleString()}`, 14, y);
      if (s.combo >= 3) { ctx.font = `600 ${Math.round(fs * 0.7)}px system-ui, sans-serif`; ctx.fillStyle = '#ffd166'; ctx.fillText(`combo ×${s.combo}`, 14 + ctx.measureText('0').width * 9, y); }
    });
    // star meter for player 1
    const f = clamp(run.scores[0].points / maxPts, 0, 1);
    const mx = meter.x, my = meter.y, mw = meter.w;
    ctx.fillStyle = 'rgba(255,255,255,0.15)'; roundRect(ctx, mx, my, mw, 8, 4); ctx.fill();
    ctx.fillStyle = '#ffd166'; roundRect(ctx, mx, my, Math.max(8, mw * Math.min(1, f / STAR_LEVELS[4])), 8, 4); ctx.fill();
    ctx.font = `${Math.round(Math.max(12, H * 0.035))}px system-ui, sans-serif`;
    ctx.textAlign = 'center';
    STAR_LEVELS.forEach((lvl, i) => { ctx.fillStyle = f >= lvl ? '#ffd166' : 'rgba(255,255,255,0.35)'; ctx.fillText('★', mx + mw * (lvl / STAR_LEVELS[4]), my - 4); });
    // pictogram lane
    ctx.fillStyle = 'rgba(0,0,0,0.35)'; roundRect(ctx, lane.x, lane.y, lane.w, lane.h, 12); ctx.fill();
    const box = lane.h * 0.86, targetX = lane.x + box * 0.6 + 8, top = lane.y + (lane.h - box) / 2;
    ctx.save();
    ctx.lineWidth = 3; ctx.strokeStyle = `rgba(255,255,255,${0.5 + 0.5 * (1 - beatPhase)})`;
    roundRect(ctx, targetX - box / 2 - 4, top - 4, box + 8, box + 8, 12); ctx.stroke();
    ctx.restore();
    const span = lane.x + lane.w - box / 2 - targetX;
    for (const m of run.moves) {
      const dt = m.time - t;
      if (dt > LOOKAHEAD || dt < -0.8) continue;
      const x = targetX + (dt / LOOKAHEAD) * span;
      ctx.save();
      ctx.globalAlpha = dt < 0 ? clamp(1 + dt / 0.8, 0, 1) : 1;
      roundRect(ctx, x - box / 2, top, box, box, 10);
      ctx.fillStyle = m.gold ? 'rgba(255,209,102,0.25)' : 'rgba(124,246,255,0.12)'; ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = m.result ? m.result[0].color : m.gold ? '#ffd166' : 'rgba(124,246,255,0.6)'; ctx.stroke();
      const pu = box / 4.2;
      drawFigure(ctx, posePoints(m.move), x, top + box * 0.94, pu, { color: m.gold ? '#ffd166' : '#ffffff', glow: 6, width: 0.22 });
      ctx.restore();
    }
    // rating pop-ups
    const now = performance.now();
    run.popups = run.popups.filter(pp => now - pp.at < 900);
    for (const pp of run.popups) {
      const age = (now - pp.at) / 900;
      const scale = 0.7 + 0.5 * Math.min(1, age * 5);
      ctx.save();
      ctx.globalAlpha = 1 - age ** 3;
      ctx.font = `900 ${Math.round(Math.max(18, H * 0.065))}px system-ui, sans-serif`;
      const label = `${run.players > 1 ? `P${pp.p + 1} ` : ''}${pp.r.label}${pp.gold && pp.r.key !== 'x' ? ' ★' : ''}`;
      // above the lane's target box, where the move was judged; players stack upwards
      const half = ctx.measureText(label).width * 0.6;
      ctx.translate(clamp(targetX, half + 8, W - half - 8), lane.y - 0.025 * H - pp.p * H * 0.07 - age * 20);
      ctx.scale(scale, scale);
      ctx.textAlign = 'center';
      ctx.lineWidth = 5; ctx.strokeStyle = 'rgba(0,0,0,0.6)';
      ctx.strokeText(label, 0, 0);
      ctx.fillStyle = pp.r.color; ctx.fillText(label, 0, 0);
      ctx.restore();
    }
  }

  function drawIdle() {
    if (!layout) computeLayout();
    const { W, H, dpr } = layout;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const bg = ctx.createLinearGradient(0, 0, W, H);
    bg.addColorStop(0, '#1a1036'); bg.addColorStop(1, '#0b1a2e');
    ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);
  }

  // ---- open / close ----
  function back() {
    if (phase === 'playing') { stopSong(); setPhase('menu'); renderMenu(); drawIdle(); return; }
    if (phase === 'results') { setPhase('menu'); renderMenu(); drawIdle(); return; }
    deps.onClose();
  }

  return {
    open() {
      if (open) return;
      open = true;
      layer.hidden = false;
      deps.stage.classList.add('game-mode');
      computeLayout();
      setPhase('menu');
      renderMenu();
      drawIdle();
    },
    close() {
      if (!open) return;
      stopSong();
      open = false;
      layer.hidden = true;
      setPhase('menu');
      deps.stage.classList.remove('game-mode', 'game-playing');
      restorePip();
    },
    refresh() { if (open && phase === 'menu') renderMenu(); },
    /** Start a song by id (from the menu or results, or replacing the song playing). */
    play(id) {
      if (!open || !SONGS.some(s => s.id === id)) return false;
      if (phase === 'playing') stopSong();
      startSong(id);
      return deps.isReady();
    },
    /** Change the music volume by `delta` (or set it with absolute), returns the new volume 0..1. */
    adjustVolume(delta, absolute = false) {
      const cur = Number(localStorage.getItem(VOLUME_KEY) ?? 0.7);
      const v = clamp(absolute ? delta : cur + delta, 0, 1);
      localStorage.setItem(VOLUME_KEY, String(v));
      run?.music.setVolume(v);
      const slider = menu.querySelector('[data-g="volume"]');
      if (slider) slider.value = String(v);
      return v;
    },
    back,
    onPoses,
    get isOpen() { return open; },
    get isPlaying() { return open && phase === 'playing'; },
  };
}
