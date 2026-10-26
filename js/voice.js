// Hands-free voice control with the browser's Web Speech API (SpeechRecognition). The phrase
// parser is pure (no DOM) so it can be tested on its own; createVoiceControl() wraps the
// recogniser: continuous listening, automatic restarts, on-device recognition where offered.

const NUMBER_WORDS = {
  zero: 0, one: 1, won: 1, two: 2, three: 3, four: 4, five: 5, six: 6,
  seven: 7, eight: 8, nine: 9, ten: 10, half: 0.5,
};

/** Colour names to sand hues (degrees). */
export const COLOUR_HUES = {
  red: 0, orange: 25, gold: 42, golden: 42, yellow: 52, lime: 90, green: 125, teal: 170,
  turquoise: 175, cyan: 188, aqua: 188, blue: 215, navy: 230, indigo: 250, purple: 272, violet: 282,
  magenta: 305, pink: 325, rose: 340,
};

const STYLE_WORDS = [
  [/^(the )?skeleton$/, 'skeleton'],
  [/^(the )?stick ?(figure|man|men)$/, 'stickman'],
  [/^(the )?neon( glow)?$/, 'neon'],
  [/^(the )?(light )?trails?$/, 'trails'],
  [/^(the )?sparks?$/, 'sparks'],
  [/^(the )?(constellations?|stars)$/, 'constellation'],
  [/^(the )?(sand ?man|sand|falling (particles|sand))$/, 'sandman'],
  [/^(the )?(fuzzy( blur)?|blur|blurry)$/, 'fuzzy'],
  [/^(the )?(origami|paper)$/, 'origami'],
  [/^(the )?(pencil( sketch)?|sketch|drawing)$/, 'pencil'],
];
const SONG_WORDS = [
  [/sunrise|warm ?up|easy/, 'sunrise'],
  [/neon|night|medium/, 'neon'],
  [/rush|hard/, 'rush'],
];
const MODEL_WORDS = [
  [/lightning/, 'movenet-lightning'],
  [/thunder/, 'movenet-thunder'],
  [/multi ?pose|multi person|multiple/, 'movenet-multipose'],
  [/holistic/, 'holistic'],
  [/heavy/, 'heavy'],
  [/full/, 'full'],
  [/lite|light/, 'lite'],
];

const ON = /\b(on|show|enable|open|start|display)\b/;
const OFF = /\b(off|hide|disable|close|stop|remove|exit|leave)\b/;
const onOff = text => (OFF.test(text) ? false : ON.test(text) ? true : null);

/** Lower-case, drop punctuation and polite fillers, turn number words into digits. */
export function normalise(text) {
  let t = ` ${String(text).toLowerCase().replace(/[-_]/g, ' ').replace(/(\d)\.(\d)/g, '$1,$2').replace(/[^a-z0-9, ]+/g, ' ').replace(/(\d),(\d)/g, '$1.$2').replace(/,/g, ' ')} `;
  t = t.replace(/\b(please|thanks|thank you|kinesphere|okay|ok|hey|now|could you|can you|would you|i want to|i want|let's|lets|go ahead and)\b/g, ' ');
  // "to/too people" and "for people" are how a recogniser often hears "two" and "four"
  t = t.replace(/\b(to|too)\s+(people|persons|dancers|players)\b/g, '2 $2').replace(/\bfor\s+(people|persons|dancers|players)\b/g, '4 $1');
  t = t.replace(/\bthree d\b|\b3 d\b|\b3d\b/g, '3d');
  t = t.replace(/\b(zero|one|won|two|three|four|five|six|seven|eight|nine|ten|half)\b/g, w => String(NUMBER_WORDS[w]));
  return t.replace(/\s+/g, ' ').trim();
}

/**
 * Turn a spoken phrase into a command, or null. Commands are { type, value }:
 *   voice:off, help:show, help:hide, record:start, stop (record or song, as context decides),
 *   record:cancel, style (id | 'next' | 'previous'), overlay / video / mirror / view3d /
 *   fullscreen (true | false | null = toggle), smoothing (id), model (id), people (n),
 *   sand ({ hue } | { gravity, relative } | { decay, relative } | { momentum, relative } | 'reset'),
 *   nav (live | game | sessions | dashboard), game:play (song id | null), game:volume (+-n | 0).
 */
export function parseCommand(text) {
  const t = normalise(text);
  if (!t) return null;
  let m;
  // voice and help
  if (/^(stop|end|quit) (listening|voice( control)?)$|^voice( control)? off$|^(turn off|disable) voice( control)?$|^mute (the )?(mic|microphone)$/.test(t)) return { type: 'voice:off' };
  if (/^(help|voice help|commands|voice commands|what can i say|what (are|can) (the )?commands?( be)?|show (the )?commands)$/.test(t)) return { type: 'help:show' };
  if (/^(close|hide)( the)?( help| commands)?$|^got it$/.test(t)) return { type: 'help:hide' };
  // recording
  if (/^cancel( (the )?(countdown|recording))?$/.test(t)) return { type: 'record:cancel' };
  if (/^((start|begin)( the| a)?( recording| dance)?|record( now| me| this)?|action|start dancing)$/.test(t)) return { type: 'record:start' };
  if (/^(stop|finish|cut|end|done|that's it|that is it)( the)?( recording| dance| dancing| it| now)?$/.test(t)) return { type: 'stop' };
  // game
  if (/^(stop|quit|end|exit) (the )?(song|game|music|track)$/.test(t)) return { type: 'game:stop' };
  if ((m = t.match(/^(play|start|dance to|let's play)( the)?( song)? (.+?)( song)?$/))) {
    const song = SONG_WORDS.find(([re]) => re.test(m[4]));
    if (song) return { type: 'game:play', value: song[1] };
    if (/^(a )?(game|song)$/.test(m[4])) return { type: 'nav', value: 'game' };
  }
  if (/^(volume up|louder|turn (it|the music|the volume) up|more volume)$/.test(t)) return { type: 'game:volume', value: 0.15 };
  if (/^(volume down|quieter|softer|turn (it|the music|the volume) down|less volume)$/.test(t)) return { type: 'game:volume', value: -0.15 };
  if (/^(mute|mute (the )?(music|sound|game)|silence)$/.test(t)) return { type: 'game:volume', value: 0, absolute: true };
  // navigation
  if ((m = t.match(/^((go|switch|back) to |open |show )?(the )?(live( view)?|camera view|game( mode)?|games|sessions?( list)?|library|dashboard|results)( view| tab| page)?$/))) {
    const w = m[4];
    if (/^live|^camera/.test(w)) return { type: 'nav', value: 'live' };
    if (/^game/.test(w)) return { type: 'nav', value: 'game' };
    if (/^session|^library/.test(w)) return { type: 'nav', value: 'sessions' };
    return { type: 'nav', value: 'dashboard' };
  }
  // display toggles
  if (/full ?screen/.test(t)) return { type: 'fullscreen', value: /\b(exit|leave|close|off|stop|end)\b/.test(t) ? false : /\b(toggle)\b/.test(t) ? null : true };
  if (/\b3d\b/.test(t)) return { type: 'view3d', value: !OFF.test(t) };
  if (/\bmirror|\bflip\b/.test(t)) return { type: 'mirror', value: /\bflip\b/.test(t) ? null : onOff(t) ?? true };
  if (/\b(overlay|pose|tracking lines)\b/.test(t) && !/\bstyle\b/.test(t)) return { type: 'overlay', value: onOff(t) };
  if (/\b(video|camera feed|camera|feed|picture)\b/.test(t) && onOff(t) !== null) return { type: 'video', value: onOff(t) };
  // smoothing
  if (/\bsmooth/.test(t)) {
    if (/\b(off|none|no)\b|less smooth/.test(t)) return { type: 'smoothing', value: 'off' };
    if (/\b(strong|high|max|maximum|more|extra)\b/.test(t)) return { type: 'smoothing', value: 'strong' };
    return { type: 'smoothing', value: 'normal' };
  }
  // model and people
  if ((m = t.match(/^(use |switch to |change to |set )?(the )?(model|mediapipe|move ?net)? ?(.*?)( model)?$/))
    && (/\bmodel\b|mediapipe|move ?net|holistic|multi ?pose|lightning|thunder/.test(t) || /^(use|switch to|change to) (the )?(lite|light|full|heavy)( model)?$/.test(t))) {
    const model = MODEL_WORDS.find(([re]) => re.test(m[4] || t));
    if (model) return { type: 'model', value: model[1] };
  }
  if ((m = t.match(/\b(\d+)\s*(people|person|persons|dancers?|players?)\b/)) || (m = t.match(/\b(just me|solo|alone|only me)\b/))) {
    const n = /\d/.test(m[1]) ? Number(m[1]) : 1;
    if (n >= 1 && n <= 6) return { type: 'people', value: n };
  }
  // sandman particles
  if (/^reset (the )?(sand|particles?|sliders?|settings)$/.test(t)) return { type: 'sand', value: 'reset' };
  const colour = Object.keys(COLOUR_HUES).find(c => new RegExp(`\\b${c}\\b`).test(t));
  if (colour && (/\b(colou?r|sand|particles?|grains?)\b/.test(t) || /^(make it |go )?\w+$/.test(t))) return { type: 'sand', value: { hue: COLOUR_HUES[colour] } };
  if (/\b(anti ?gravity|float|floating|reverse gravity|upside down)\b/.test(t)) return { type: 'sand', value: { gravity: -0.6 } };
  if (/\bgravity\b/.test(t)) {
    if ((m = t.match(/(\d+(\.\d+)?)/))) return { type: 'sand', value: { gravity: Number(m[1]) } };
    if (/\b(off|zero|none|no)\b/.test(t)) return { type: 'sand', value: { gravity: 0 } };
    if (/\b(normal|reset|default)\b/.test(t)) return { type: 'sand', value: { gravity: 1 } };
    if (/\b(up|more|higher|stronger|heavier|increase)\b/.test(t)) return { type: 'sand', value: { gravity: 0.5, relative: true } };
    if (/\b(down|less|lower|weaker|lighter|decrease)\b/.test(t)) return { type: 'sand', value: { gravity: -0.5, relative: true } };
  }
  if (/\b(decay|fade|trails?|lifetime)\b/.test(t)) {
    if ((m = t.match(/(\d+(\.\d+)?)/))) return { type: 'sand', value: { decay: Number(m[1]) } };
    if (/\b(longer|slower|more|up|increase|longest)\b/.test(t)) return { type: 'sand', value: { decay: 1, relative: true } };
    if (/\b(shorter|faster|less|down|decrease|quicker)\b/.test(t)) return { type: 'sand', value: { decay: -1, relative: true } };
  }
  if (/\bmomentum\b/.test(t)) {
    if ((m = t.match(/(\d+(\.\d+)?)/))) return { type: 'sand', value: { momentum: Number(m[1]) } };
    if (/\b(off|zero|none|no)\b/.test(t)) return { type: 'sand', value: { momentum: 0 } };
    if (/\b(normal|reset|default|on)\b/.test(t)) return { type: 'sand', value: { momentum: 1 } };
    if (/\b(more|up|higher|stronger|increase)\b/.test(t)) return { type: 'sand', value: { momentum: 0.5, relative: true } };
    if (/\b(less|down|lower|weaker|decrease)\b/.test(t)) return { type: 'sand', value: { momentum: -0.5, relative: true } };
  }
  // overlay style
  if (/^(next|another|change( the)?|switch( the)?|new) (style|avatar|look|effect)$|^next$/.test(t)) return { type: 'style', value: 'next' };
  if (/^(previous|last|back) (style|avatar|look|effect)$/.test(t)) return { type: 'style', value: 'previous' };
  const bare = t.replace(/^((switch|change|go) to |use |make (it |me )?|show (me )?|be (a )?)/, '').replace(/^(style|avatar|look|effect) /, '').replace(/ (style|avatar|look|effect|mode)$/, '');
  const style = STYLE_WORDS.find(([re]) => re.test(bare));
  if (style) return { type: 'style', value: style[1] };
  return null;
}

/** What to tell people they can say, grouped for the help panel and the README. */
export const VOICE_HELP = [
  ['Recording', [['“Start recording” / “Record”', 'start after a 3-second countdown'], ['“Stop” / “Finish”', 'stop and open the dashboard'], ['“Cancel”', 'cancel the countdown']]],
  ['Look', [['“Neon”, “Sandman”, “Pencil”, “Origami”, “Fuzzy”…', 'switch overlay style'], ['“Next style”', 'the next style'], ['“Overlay off” / “Show overlay”', 'hide or show the pose'], ['“Hide video” / “Video on”', 'hide or show the camera feed'], ['“Fullscreen” / “Exit fullscreen”', ''], ['“Mirror off” / “Flip”', 'mirror view'], ['“3D view” / “Close 3D”', ''], ['“Smoothing off / normal / strong”', '']]],
  ['Sandman particles', [['“Blue sand”, “Colour pink”…', 'particle colour'], ['“Gravity up / down / off / 2”, “Float”', 'gravity'], ['“Longer trails” / “Decay 5 seconds”', 'decay time'], ['“More / less momentum”', 'momentum'], ['“Reset sand”', 'defaults']]],
  ['Tracking', [['“Two people”, “Just me”', 'how many people to track'], ['“Use MultiPose / Lite / Heavy / Holistic”', 'pose model']]],
  ['Game and pages', [['“Game”, “Sessions”, “Dashboard”, “Live”', 'go to a page'], ['“Play Sunrise / Neon / Rush”', 'start a song'], ['“Stop song”', 'back to the song list'], ['“Louder” / “Quieter” / “Mute”', 'music volume']]],
  ['Voice', [['“Help” / “What can I say”', 'show this list'], ['“Close”', 'hide it'], ['“Stop listening”', 'turn voice control off']]],
];

const Recognition = typeof window !== 'undefined' ? (window.SpeechRecognition || window.webkitSpeechRecognition) : null;
export const voiceSupported = Boolean(Recognition);

/** English for the recogniser, in the user's regional variant when they have one. */
export function recognitionLang() {
  const nav = (typeof navigator !== 'undefined' && navigator.language) || 'en-US';
  return /^en\b/i.test(nav) ? nav : 'en-US';
}

/**
 * Continuous listening. `onPhrase(alternatives: string[])` gets each final result's transcripts
 * (best first); `onState({ listening, local, error })` follows the recogniser's state.
 */
export function createVoiceControl({ onPhrase, onState }) {
  let rec = null, wanted = false, local = false, restartTimer = 0, failures = 0;
  const lang = recognitionLang();

  let checked = null;
  /**
   * Chrome 139+ can transcribe on the device. Use that when the language pack is installed; when it
   * can be downloaded, start the download in the background so a later session can use it. The
   * check is time-limited because some browsers never answer it.
   */
  function preferOnDevice() {
    checked ??= (async () => {
      try {
        if (!Recognition?.available) return false;
        const opts = { langs: [lang], processLocally: true };
        const status = await Promise.race([Recognition.available(opts), new Promise(res => setTimeout(() => res('timeout'), 2500))]);
        if (status === 'downloadable' && Recognition.install) Promise.resolve(Recognition.install(opts)).catch(() => {});
        return status === 'available';
      } catch {
        return false;
      }
    })();
    return checked;
  }

  function build() {
    const r = new Recognition();
    r.lang = lang;
    r.continuous = true;
    r.interimResults = false;
    r.maxAlternatives = 3;
    if (local && 'processLocally' in r) r.processLocally = true;
    r.onstart = () => onState({ listening: true, local });
    r.onresult = e => {
      failures = 0; // recognition is working
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const res = e.results[i];
        if (!res.isFinal) continue;
        onPhrase(Array.from(res, alt => alt.transcript.trim()).filter(Boolean));
      }
    };
    r.onerror = e => {
      if (e.error === 'no-speech' || e.error === 'aborted') return;
      if (e.error === 'not-allowed' || e.error === 'service-not-allowed') { wanted = false; onState({ listening: false, error: 'The microphone is blocked for this page, or speech recognition is not allowed. Check the site permissions.' }); return; }
      if (e.error === 'language-not-supported' && local) { local = false; return; } // retry with the online service
      failures++;
      if (failures > 3) { wanted = false; onState({ listening: false, error: e.error === 'network' ? 'Speech recognition needs a network connection in this browser.' : `Speech recognition stopped (${e.error}).` }); }
    };
    // Browsers end continuous recognition after silence or a time limit: start again while wanted.
    r.onend = () => {
      rec = null;
      if (!wanted) { onState({ listening: false }); return; }
      clearTimeout(restartTimer);
      restartTimer = setTimeout(() => { if (wanted) start(); }, failures ? 600 * failures : 120);
    };
    return r;
  }

  async function start() {
    if (!Recognition || rec) return;
    wanted = true;
    rec = build();
    try { rec.start(); } catch (err) { rec = null; console.warn('Speech recognition failed to start', err); }
  }

  return {
    supported: Boolean(Recognition),
    get on() { return wanted; },
    get local() { return local; },
    /** Check for on-device recognition, then start listening. */
    async enable() {
      if (!Recognition) return false;
      local = await preferOnDevice();
      await start();
      return true;
    },
    canRunLocally: preferOnDevice,
    disable() {
      wanted = false;
      clearTimeout(restartTimer);
      try { rec?.stop(); } catch { /* already stopped */ }
      rec = null;
      onState({ listening: false });
    },
  };
}
