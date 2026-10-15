// Dance game core (no DOM, no audio): the move library, the songs and their choreography, and
// pose matching. Poses are described as screen-space limb directions as the player sees the coach,
// and players are matched as a mirror image of the coach (raise the arm on the same side of the
// screen), whatever the camera preview's mirror setting.

/** Angle convention: degrees, 0 = pointing straight down, +90 = towards screen right, 180 = up. */
const dir = a => [Math.sin(a * Math.PI / 180), Math.cos(a * Math.PI / 180)];

// Limb segments: screen-left (l) and screen-right (r) upper arm, forearm, thigh, shin.
export const SEGMENTS = ['lu', 'lf', 'ru', 'rf', 'lt', 'ls', 'rt', 'rs'];
const ARMS_DOWN = { lu: -10, lf: -8, ru: 10, rf: 8 };
const LEGS = { lt: -6, ls: -4, rt: 6, rs: 4 };
const WIDE = { lt: -22, ls: -18, rt: 22, rs: 18 };

const BASE_MOVES = {
  ready: { name: 'Ready', ...ARMS_DOWN, ...LEGS },
  vup: { name: 'Arms up', lu: -162, lf: -165, ru: 162, rf: 165, ...LEGS },
  t: { name: 'T', lu: -90, lf: -90, ru: 90, rf: 90, ...LEGS },
  reachL: { name: 'Reach up', lu: -172, lf: -178, ru: 45, rf: -60, ...LEGS },
  hips: { name: 'Hands on hips', lu: -45, lf: 60, ru: 45, rf: -60, ...WIDE },
  clap: { name: 'Clap', lu: -20, lf: 112, ru: 20, rf: -112, ...LEGS },
  discoL: { name: 'Disco point', lu: -135, lf: -140, ru: 40, rf: 45, ...LEGS },
  muscle: { name: 'Muscles', lu: -90, lf: -180, ru: 90, rf: 180, ...WIDE },
  pointL: { name: 'Point', lu: -90, lf: -90, ru: -80, rf: -90, ...LEGS },
  egypt: { name: 'Egypt', lu: -90, lf: -180, ru: 90, rf: 0, ...LEGS },
  egyptB: { name: 'Egypt flip', lu: -90, lf: 0, ru: 90, rf: 180, ...LEGS },
  star: { name: 'Star', lu: -122, lf: -120, ru: 122, rf: 120, lt: -34, ls: -34, rt: 34, rs: 34 },
  waveL: { name: 'Wave', lu: -120, lf: -150, ru: -160, rf: -140, ...LEGS },
  kneeL: { name: 'Knee up', lu: -45, lf: -10, ru: 160, rf: 166, lt: -75, ls: -5, rt: 6, rs: 4 },
  kickL: { name: 'Side kick', lu: -90, lf: -90, ru: 165, rf: 170, lt: -60, ls: -70, rt: 6, rs: 4 },
  squat: { name: 'Squat', lu: -60, lf: -90, ru: 60, rf: 90, lt: -62, ls: 22, rt: 62, rs: -22 },
};

/** Mirror a pose left-to-right (swap screen sides, negate angles). */
function mirrorMove(m) {
  return { name: m.name, lu: -m.ru, lf: -m.rf, ru: -m.lu, rf: -m.lf, lt: -m.rt, ls: -m.rs, rt: -m.lt, rs: -m.ls };
}

/** Every move by id; "...L" moves get an "...R" mirror twin. */
export const MOVES = (() => {
  const out = {};
  for (const [id, m] of Object.entries(BASE_MOVES)) {
    out[id] = m;
    if (id.endsWith('L')) out[`${id.slice(0, -1)}R`] = mirrorMove(m);
  }
  return out;
})();

// Coach skeleton in torso units (screen space, y down): shoulders, hips, segment lengths.
const SH = [[-0.45, -1.0], [0.45, -1.0]], HP = [[-0.28, 0], [0.28, 0]];
const LEN = { u: 0.6, f: 0.55, t: 0.9, s: 0.85 };
export const FLOOR_Y = 1.75;

/**
 * Joint positions for a pose (torso units, screen space): the lower foot rests on the floor, so a
 * squat lowers the body and a knee lift keeps the standing leg straight.
 */
export function posePoints(m) {
  const add = (p, a, l) => { const d = dir(a); return [p[0] + d[0] * l, p[1] + d[1] * l]; };
  const P = {
    shL: SH[0], shR: SH[1], hipL: HP[0], hipR: HP[1], neck: [0, -1.0], head: [0, -1.42],
  };
  P.elL = add(P.shL, m.lu, LEN.u); P.wrL = add(P.elL, m.lf, LEN.f);
  P.elR = add(P.shR, m.ru, LEN.u); P.wrR = add(P.elR, m.rf, LEN.f);
  P.knL = add(P.hipL, m.lt, LEN.t); P.anL = add(P.knL, m.ls, LEN.s);
  P.knR = add(P.hipR, m.rt, LEN.t); P.anR = add(P.knR, m.rs, LEN.s);
  const drop = FLOOR_Y - Math.max(P.anL[1], P.anR[1]);
  for (const k of Object.keys(P)) P[k] = [P[k][0], P[k][1] + drop];
  return P;
}

/** Interpolate two poses angle-by-angle along the shorter way round. */
export function lerpMove(a, b, t) {
  const out = {};
  for (const s of SEGMENTS) {
    let d = ((b[s] - a[s] + 540) % 360) - 180;
    out[s] = a[s] + d * t;
  }
  return out;
}

// ---- songs ------------------------------------------------------------------------------------

// Choreography patterns; a trailing "*" marks a gold move (double points).
const PAT = {
  warmA: ['vup', 't', 'reachL', 'reachR', 'hips', 'clap', 'discoL', 'discoR'],
  warmB: ['star', 'muscle', 'pointL', 'pointR', 'egypt', 'egyptB', 'waveL', 'waveR*'],
  warmC: ['vup', 't', 'clap', 'star', 'reachL', 'reachR', 'muscle', 'vup*'],
  discoA: ['discoL', 'discoR', 'discoL', 'discoR', 'pointL', 'pointR', 'clap', 'clap'],
  discoB: ['egypt', 'egyptB', 'egypt', 'egyptB', 'waveL', 'waveR', 'vup', 'vup'],
  discoC: ['kneeL', 'kneeR', 'kneeL', 'kneeR', 'muscle', 't', 'star', 'star*'],
  rushD: ['kickL', 'kickR', 'squat', 'star', 'reachL', 'reachR', 'squat', 'vup*'],
  rushE: ['waveL', 'waveR', 'pointL', 'pointR', 'kneeL', 'kneeR', 'muscle', 't'],
};

/**
 * Songs: original procedural tracks (see game.js for the synth). `chords` are [root MIDI note,
 * 'maj' | 'min'], one per bar, cycled. Choreography starts after a one-bar count-in.
 */
export const SONGS = [
  {
    id: 'sunrise', title: 'Sunrise Warm-up', bpm: 96, difficulty: 'Easy', style: 'pop',
    beatsPerMove: 4, routine: ['warmA', 'warmB', 'warmC'],
    chords: [[60, 'maj'], [57, 'min'], [53, 'maj'], [55, 'maj']],
    blurb: 'One move per bar to learn the ropes. Arms only.',
  },
  {
    id: 'neon', title: 'Neon Nights', bpm: 116, difficulty: 'Medium', style: 'disco',
    beatsPerMove: 2, routine: ['discoA', 'discoA', 'discoB', 'discoB', 'discoC', 'discoA', 'discoB', 'discoC'],
    chords: [[57, 'min'], [53, 'maj'], [60, 'maj'], [55, 'maj']],
    blurb: 'Disco points, egypt arms and knee lifts, two beats per move.',
  },
  {
    id: 'rush', title: 'Kinesphere Rush', bpm: 128, difficulty: 'Hard', style: 'edm',
    beatsPerMove: 2, routine: ['rushE', 'rushD', 'rushE', 'rushD', 'discoB', 'rushD', 'discoC', 'rushD'],
    chords: [[62, 'min'], [58, 'maj'], [53, 'maj'], [60, 'maj']],
    blurb: 'Fast, full body: kicks, squats and stars. Make sure your legs are in view.',
  },
];

/**
 * Timeline for a song: { beat (seconds per beat), countInBeats, moves: [{ id, move, time, beats,
 * gold }], duration } with times in seconds from the start of the music.
 */
export function buildChoreography(song) {
  const beat = 60 / song.bpm;
  const countInBeats = 4;
  const moves = [];
  let b = countInBeats;
  for (const pat of song.routine) {
    for (const entry of PAT[pat]) {
      const gold = entry.endsWith('*');
      const id = gold ? entry.slice(0, -1) : entry;
      b += song.beatsPerMove;
      moves.push({ id, move: MOVES[id], time: b * beat, beats: song.beatsPerMove, gold });
    }
  }
  const totalBeats = Math.ceil((b + 4) / 4) * 4; // finish the bar, plus one bar of outro
  return { beat, countInBeats, moves, duration: totalBeats * beat, bars: totalBeats / 4 };
}

// ---- matching ---------------------------------------------------------------------------------

/** Segment endpoints by anatomical landmark; screen-left in mirror space = the player's left side. */
const PLAYER_SEGMENTS = {
  lu: [11, 13], lf: [13, 15], ru: [12, 14], rf: [14, 16],
  lt: [23, 25], ls: [25, 27], rt: [24, 26], rs: [26, 28],
};
const WEIGHTS = { lu: 1, lf: 1, ru: 1, rf: 1, lt: 0.6, ls: 0.4, rt: 0.6, rs: 0.4 };
const FULL_CREDIT = 18, NO_CREDIT = 75; // degrees of error

/** Segment directions of a player pose (flat MediaPipe-order landmarks) in mirror space. */
export function playerAngles(lm, aspect = 16 / 9, minVis = 0.4) {
  const out = {};
  for (const [s, [a, b]] of Object.entries(PLAYER_SEGMENTS)) {
    if (lm[a * 4 + 3] < minVis || lm[b * 4 + 3] < minVis) { out[s] = null; continue; }
    const dx = -(lm[b * 4] - lm[a * 4]) * aspect; // flip x: the player is a mirror image of the coach
    const dy = lm[b * 4 + 1] - lm[a * 4 + 1];
    out[s] = Math.atan2(dx, dy) * 180 / Math.PI;
  }
  return out;
}

/**
 * How well a player's segment angles match a target move, 0..1. Arms always count (an arm out of
 * view scores zero); legs count when visible.
 */
export function matchScore(target, angles) {
  let sum = 0, weight = 0;
  for (const s of SEGMENTS) {
    const a = angles[s];
    const isLeg = s[1] === 't' || s[1] === 's';
    if (a == null) { if (!isLeg) weight += WEIGHTS[s]; continue; }
    const err = Math.abs(((a - target[s] + 540) % 360) - 180);
    const credit = Math.max(0, Math.min(1, 1 - (err - FULL_CREDIT) / (NO_CREDIT - FULL_CREDIT)));
    sum += credit * WEIGHTS[s];
    weight += WEIGHTS[s];
  }
  return weight ? sum / weight : 0;
}

export const RATINGS = [
  { key: 'perfect', label: 'Perfect', min: 0.85, points: 100, color: '#ffd166' },
  { key: 'super', label: 'Super', min: 0.7, points: 70, color: '#7ee787' },
  { key: 'good', label: 'Good', min: 0.55, points: 45, color: '#8ec5ff' },
  { key: 'ok', label: 'OK', min: 0.4, points: 20, color: '#c9b8ff' },
  { key: 'x', label: 'X', min: -Infinity, points: 0, color: '#ff7b72' },
];
export const rate = score => RATINGS.find(r => score >= r.min);

/** Star thresholds as a share of the song's maximum points; superstar above the last. */
export const STAR_LEVELS = [0.2, 0.35, 0.5, 0.65, 0.8];
export const SUPERSTAR = 0.9;
export function starsFor(points, maxPoints) {
  const f = maxPoints ? points / maxPoints : 0;
  return { stars: STAR_LEVELS.filter(s => f >= s).length, superstar: f >= SUPERSTAR, fraction: f };
}
export const maxPointsFor = moves => moves.reduce((acc, m) => acc + 100 * (m.gold ? 2 : 1), 0);
