// Pose detection engines (MediaPipe Pose Landmarker or TensorFlow.js MoveNet) behind one
// interface, plus a small tracker that keeps people in stable "slots" across frames and the
// skeleton drawing helper.
//
// Every engine produces poses in the same layout: a Float32Array of 33 landmarks x
// [x, y, z, visibility] in MediaPipe order and normalised image coordinates. MoveNet only
// knows 17 of those joints, so it fills the matching slots and leaves the rest at visibility 0.
// Libraries are loaded lazily from the CDN so the rest of the app works without them.

const MP_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1';
const MP_MODEL_BASE = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker';
const TF_CDN = 'https://cdn.jsdelivr.net/npm';
const TF_VERSION = '4.22.0';
const POSE_DETECTION_VERSION = '2.1.3';

export const MODELS = {
  lite: `${MP_MODEL_BASE}/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
  full: `${MP_MODEL_BASE}/pose_landmarker_full/float16/1/pose_landmarker_full.task`,
  heavy: `${MP_MODEL_BASE}/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task`,
};

/** The 33 BlazePose landmarks, in MediaPipe index order. */
export const LANDMARK_NAMES = [
  'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer', 'right_eye_inner', 'right_eye', 'right_eye_outer',
  'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
  'left_pinky', 'right_pinky', 'left_index', 'right_index', 'left_thumb', 'right_thumb',
  'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
  'left_heel', 'right_heel', 'left_foot_index', 'right_foot_index',
];
export const NUM_LANDMARKS = 33;
/** Values stored per landmark: x, y, z (normalised image coords) and visibility. */
export const STRIDE = 4;
/** Floats per person per frame. */
export const FRAME_SIZE = NUM_LANDMARKS * STRIDE;
export const MAX_PEOPLE = 6;

/** MoveNet's 17 COCO keypoints, mapped onto the MediaPipe landmark index they correspond to. */
export const MOVENET_TO_MP = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];

/**
 * Selectable detection models. `id` is what sessions store in their `model` field.
 * `maxPeople` is how many people the model can track at once.
 */
export const ENGINES = [
  {
    id: 'lite', label: 'MediaPipe Lite', engine: 'mediapipe', variant: 'lite', maxPeople: MAX_PEOPLE,
    help: 'Fastest MediaPipe model. All 33 landmarks including face, hands and an estimated depth for each joint. Good default for one person on most devices.',
  },
  {
    id: 'full', label: 'MediaPipe Full', engine: 'mediapipe', variant: 'full', maxPeople: MAX_PEOPLE,
    help: 'More accurate than Lite, roughly half the frame rate. 33 landmarks with depth.',
  },
  {
    id: 'heavy', label: 'MediaPipe Heavy', engine: 'mediapipe', variant: 'heavy', maxPeople: MAX_PEOPLE,
    help: 'Most accurate MediaPipe model and by far the slowest (a 30 MB download). Best for a powerful laptop or desktop with a GPU.',
  },
  {
    id: 'movenet-lightning', label: 'MoveNet Lightning', engine: 'movenet', variant: 'SinglePose.Lightning', maxPeople: 1,
    help: 'Very fast TensorFlow.js model, often the smoothest on phones and laptops without a GPU. Tracks 17 joints only: no depth, no hand or face detail, so the overlay is simpler. Single person only.',
  },
  {
    id: 'movenet-thunder', label: 'MoveNet Thunder', engine: 'movenet', variant: 'SinglePose.Thunder', maxPeople: 1,
    help: 'More accurate MoveNet model, slower than Lightning. 17 joints, single person only.',
  },
  {
    id: 'movenet-multipose', label: 'MoveNet MultiPose', engine: 'movenet', variant: 'MultiPose.Lightning', maxPeople: MAX_PEOPLE,
    help: 'MoveNet model built for groups: tracks up to 6 people with a built-in tracker that keeps identities stable. 17 joints per person, no depth. Slower than Lightning.',
  },
];

export function engineInfo(id) {
  return ENGINES.find(e => e.id === id) ?? ENGINES[0];
}

/** Standard POSE_CONNECTIONS from MediaPipe. */
export const CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 7], [0, 4], [4, 5], [5, 6], [6, 8], [9, 10],
  [11, 12], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
  [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [11, 23], [12, 24], [23, 24], [23, 25], [24, 26], [25, 27], [26, 28],
  [27, 29], [28, 30], [29, 31], [30, 32], [27, 31], [28, 32],
];
const LEFT = new Set(LANDMARK_NAMES.map((n, i) => (n.startsWith('left') || n.endsWith('_left') ? i : -1)).filter(i => i >= 0));
const RIGHT = new Set(LANDMARK_NAMES.map((n, i) => (n.startsWith('right') || n.endsWith('_right') ? i : -1)).filter(i => i >= 0));

/** Visibility threshold to use for an engine: MoveNet scores run lower than MediaPipe visibility. */
export function minVisibilityFor(engine) {
  return engine === 'movenet' ? 0.3 : 0.5;
}

// ---- MediaPipe ------------------------------------------------------------------------------

let mpModulePromise = null;
function loadMediaPipe() {
  mpModulePromise ??= import(`${MP_CDN}/vision_bundle.mjs`);
  return mpModulePromise;
}

async function createMediaPipe(info, people, onStatus) {
  onStatus('Loading MediaPipe…');
  const { PoseLandmarker, FilesetResolver } = await loadMediaPipe();
  const vision = await FilesetResolver.forVisionTasks(`${MP_CDN}/wasm`);
  const options = {
    baseOptions: { modelAssetPath: MODELS[info.variant] ?? MODELS.lite, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numPoses: people,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    outputSegmentationMasks: false,
  };
  onStatus(`Downloading the ${info.label} model…`);
  let landmarker;
  let delegate = 'GPU';
  try {
    landmarker = await PoseLandmarker.createFromOptions(vision, options);
  } catch (err) {
    console.warn('GPU delegate unavailable, falling back to CPU', err);
    delegate = 'CPU';
    options.baseOptions.delegate = 'CPU';
    landmarker = await PoseLandmarker.createFromOptions(vision, options);
  }
  return {
    delegate,
    detect(video, timestampMs) {
      const result = landmarker.detectForVideo(video, timestampMs);
      return (result?.landmarks ?? []).map(l => ({ lm: flattenLandmarks(l) }));
    },
    close() { landmarker.close(); },
  };
}

// ---- MoveNet (TensorFlow.js) -----------------------------------------------------------------

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing?.dataset.loaded) return resolve();
    const s = existing ?? document.createElement('script');
    s.addEventListener('load', () => { s.dataset.loaded = '1'; resolve(); }, { once: true });
    s.addEventListener('error', () => reject(new Error(`Could not load ${src}`)), { once: true });
    if (!existing) { s.src = src; s.crossOrigin = 'anonymous'; document.head.append(s); }
  });
}

let tfPromise = null;
function loadTensorFlow(onStatus) {
  tfPromise ??= (async () => {
    onStatus('Loading TensorFlow.js…');
    for (const pkg of ['tfjs-core/dist/tf-core.min.js', 'tfjs-converter/dist/tf-converter.min.js', 'tfjs-backend-webgl/dist/tf-backend-webgl.min.js', 'tfjs-backend-cpu/dist/tf-backend-cpu.min.js']) {
      const [name, file] = pkg.split('/dist/');
      await loadScript(`${TF_CDN}/@tensorflow/${name}@${TF_VERSION}/dist/${file}`);
    }
    await loadScript(`${TF_CDN}/@tensorflow-models/pose-detection@${POSE_DETECTION_VERSION}/dist/pose-detection.min.js`);
    const tf = window.tf;
    // WebGL is much faster; the plain CPU backend is the fallback when the browser has no usable GPU context.
    let ok = false;
    try { ok = await tf.setBackend('webgl'); } catch (err) { console.warn('WebGL backend unavailable', err); }
    if (!ok) {
      onStatus('No GPU available for TensorFlow.js, using the (slow) CPU backend…');
      await tf.setBackend('cpu');
    }
    await tf.ready();
    return window.poseDetection;
  })();
  tfPromise.catch(() => { tfPromise = null; });
  return tfPromise;
}

async function createMoveNet(info, people, onStatus) {
  const pd = await loadTensorFlow(onStatus);
  onStatus(`Downloading the ${info.label} model…`);
  const multi = info.variant === 'MultiPose.Lightning';
  const modelType = multi ? pd.movenet.modelType.MULTIPOSE_LIGHTNING
    : info.variant === 'SinglePose.Thunder' ? pd.movenet.modelType.SINGLEPOSE_THUNDER
      : pd.movenet.modelType.SINGLEPOSE_LIGHTNING;
  const config = { modelType, enableSmoothing: true };
  if (multi) { config.enableTracking = true; config.trackerType = pd.TrackerType.BoundingBox; }
  const detector = await pd.createDetector(pd.SupportedModels.MoveNet, config);
  return {
    delegate: window.tf.getBackend(),
    async detect(video, timestampMs) {
      const w = video.videoWidth, h = video.videoHeight;
      if (!w || !h) return [];
      const poses = await detector.estimatePoses(video, { maxPoses: multi ? people : 1, flipHorizontal: false }, timestampMs);
      const out = [];
      for (const pose of poses) {
        const lm = new Float32Array(FRAME_SIZE);
        let scoreSum = 0;
        pose.keypoints.forEach((k, i) => {
          const slot = MOVENET_TO_MP[i];
          if (slot == null) return;
          const o = slot * STRIDE;
          lm[o] = k.x / w; lm[o + 1] = k.y / h; lm[o + 2] = 0; lm[o + 3] = k.score ?? 0;
          scoreSum += k.score ?? 0;
        });
        const score = pose.score ?? scoreSum / Math.max(1, pose.keypoints.length);
        if (score < 0.2) continue; // single-pose models always return one pose, even for an empty frame
        out.push({ lm, id: pose.id });
      }
      return out;
    },
    close() { detector.dispose(); },
  };
}

// ---- public API ------------------------------------------------------------------------------

/**
 * Create a detector. `detect(video, timestampMs)` returns (possibly a promise of) an array of
 * poses `{ lm: Float32Array(132), id? }` in no particular order; use a tracker to keep people
 * in stable slots.
 * @param {{model?: string, people?: number, onStatus?: (msg: string) => void}} opts
 */
export async function createPoseDetector({ model = 'lite', people = 1, onStatus = () => {} } = {}) {
  const info = engineInfo(model);
  const n = Math.max(1, Math.min(info.maxPeople, people | 0 || 1));
  const impl = info.engine === 'movenet' ? await createMoveNet(info, n, onStatus) : await createMediaPipe(info, n, onStatus);
  return { id: info.id, model: info.id, engine: info.engine, label: info.label, people: n, ...impl };
}

/** Convert MediaPipe's landmark objects into a flat Float32Array [x,y,z,visibility]*33. */
export function flattenLandmarks(landmarks) {
  const out = new Float32Array(FRAME_SIZE);
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    const p = landmarks[i];
    if (!p) continue;
    const o = i * STRIDE;
    out[o] = p.x;
    out[o + 1] = p.y;
    out[o + 2] = p.z;
    out[o + 3] = p.visibility ?? 1;
  }
  return out;
}

/** Body centre of a pose in normalised coordinates (mid-hip, else the mean of visible joints). */
export function poseCentre(lm, offset = 0, minVis = 0.3) {
  const v = i => lm[offset + i * STRIDE + 3];
  if (v(23) >= minVis && v(24) >= minVis) {
    return [(lm[offset + 23 * STRIDE] + lm[offset + 24 * STRIDE]) / 2, (lm[offset + 23 * STRIDE + 1] + lm[offset + 24 * STRIDE + 1]) / 2];
  }
  let sx = 0, sy = 0, n = 0;
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    if (v(i) < minVis) continue;
    sx += lm[offset + i * STRIDE]; sy += lm[offset + i * STRIDE + 1]; n++;
  }
  return n ? [sx / n, sy / n] : null;
}

/**
 * Keeps detected people in stable slots (0..maxPeople-1) between frames, so each dancer keeps
 * the same colour and ends up in the same track of the recording. Matches by detector id when
 * the engine provides one, otherwise by nearest body centre; new people take the slot that has
 * been empty the longest.
 */
export function createTracker(maxPeople) {
  const slots = Array.from({ length: maxPeople }, () => ({ cx: 0, cy: 0, lastSeen: -Infinity, id: null }));
  const MATCH_DIST = 0.3;   // normalised units: how far a person may move between matches
  const MEMORY_MS = 2000;   // how long a slot remembers where its person was
  return {
    /** @param {{lm: Float32Array, id?: number}[]} poses @returns {(Float32Array|null)[]} */
    update(poses, now) {
      const out = new Array(maxPeople).fill(null);
      const cands = poses.map(p => ({ ...p, c: poseCentre(p.lm) })).filter(p => p.c);
      const taken = new Set();
      const assigned = new Array(cands.length).fill(-1);
      cands.forEach((p, i) => {
        if (p.id == null) return;
        const k = slots.findIndex((s, j) => s.id === p.id && !taken.has(j) && now - s.lastSeen < MEMORY_MS);
        if (k >= 0) { assigned[i] = k; taken.add(k); }
      });
      const pairs = [];
      cands.forEach((p, i) => {
        if (assigned[i] >= 0) return;
        slots.forEach((s, k) => {
          if (taken.has(k) || now - s.lastSeen > MEMORY_MS) return;
          const d = Math.hypot(p.c[0] - s.cx, p.c[1] - s.cy);
          if (d < MATCH_DIST) pairs.push([d, i, k]);
        });
      });
      pairs.sort((a, b) => a[0] - b[0]);
      for (const [, i, k] of pairs) {
        if (assigned[i] >= 0 || taken.has(k)) continue;
        assigned[i] = k; taken.add(k);
      }
      const free = slots.map((s, k) => k).filter(k => !taken.has(k)).sort((a, b) => slots[a].lastSeen - slots[b].lastSeen);
      cands.forEach((p, i) => {
        if (assigned[i] >= 0 || !free.length) return;
        const k = free.shift();
        assigned[i] = k; taken.add(k);
      });
      cands.forEach((p, i) => {
        const k = assigned[i];
        if (k < 0) return;
        const s = slots[k];
        s.cx = p.c[0]; s.cy = p.c[1]; s.lastSeen = now; s.id = p.id ?? null;
        out[k] = p.lm;
      });
      return out;
    },
    reset() { for (const s of slots) { s.lastSeen = -Infinity; s.id = null; } },
  };
}

/** True when both wrists are clearly above the nose (the "raise both hands" gesture). */
export function handsAboveHead(lm, offset = 0, minVis = 0.3) {
  const y = i => lm[offset + i * STRIDE + 1];
  const v = i => lm[offset + i * STRIDE + 3];
  if (v(0) < minVis || v(15) < minVis || v(16) < minVis) return false;
  const margin = 0.04;
  return y(15) < y(0) - margin && y(16) < y(0) - margin;
}

/**
 * Draw a skeleton from a flat landmark array onto a 2D canvas context.
 * Left-side limbs are orange, right-side limbs are blue, the rest is neutral (by default).
 */
export function drawSkeleton(ctx, lm, offset, width, height, opts = {}) {
  const {
    mirror = false, minVis = 0.5, lineWidth = 3, radius = 4,
    leftColor = '#eb6834', rightColor = '#3987e5', midColor = '#f4f4f2',
  } = opts;
  const px = i => {
    const o = offset + i * STRIDE;
    const x = mirror ? 1 - lm[o] : lm[o];
    return [x * width, lm[o + 1] * height, lm[o + 3]];
  };
  const sideColor = (a, b) => {
    if (LEFT.has(a) && LEFT.has(b)) return leftColor;
    if (RIGHT.has(a) && RIGHT.has(b)) return rightColor;
    return midColor;
  };
  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.lineWidth = lineWidth;
  for (const [a, b] of CONNECTIONS) {
    const pa = px(a), pb = px(b);
    if (pa[2] < minVis || pb[2] < minVis) continue;
    ctx.strokeStyle = sideColor(a, b);
    ctx.beginPath();
    ctx.moveTo(pa[0], pa[1]);
    ctx.lineTo(pb[0], pb[1]);
    ctx.stroke();
  }
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    const p = px(i);
    if (p[2] < minVis) continue;
    ctx.fillStyle = LEFT.has(i) ? leftColor : RIGHT.has(i) ? rightColor : midColor;
    ctx.beginPath();
    ctx.arc(p[0], p[1], i >= 1 && i <= 10 ? radius * 0.6 : radius, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}
