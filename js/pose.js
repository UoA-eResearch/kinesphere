// MediaPipe Pose Landmarker wrapper + skeleton drawing.
// The MediaPipe module is loaded lazily from the CDN so the rest of the app
// (session library, dashboard, import/export) works even when it is unreachable.

const CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1';
const MODEL_BASE = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker';

export const MODELS = {
  lite: `${MODEL_BASE}/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
  full: `${MODEL_BASE}/pose_landmarker_full/float16/1/pose_landmarker_full.task`,
  heavy: `${MODEL_BASE}/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task`,
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
export const FRAME_SIZE = NUM_LANDMARKS * STRIDE;

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

let modulePromise = null;
function loadModule() {
  modulePromise ??= import(`${CDN}/vision_bundle.mjs`);
  return modulePromise;
}

/**
 * Create a pose detector for video. Tries the GPU delegate first and falls back to CPU.
 * @param {{model?: 'lite'|'full'|'heavy', onStatus?: (msg: string) => void}} opts
 */
export async function createPoseDetector({ model = 'lite', onStatus = () => {} } = {}) {
  onStatus('Loading MediaPipe…');
  const { PoseLandmarker, FilesetResolver } = await loadModule();
  const vision = await FilesetResolver.forVisionTasks(`${CDN}/wasm`);
  const options = {
    baseOptions: { modelAssetPath: MODELS[model] ?? MODELS.lite, delegate: 'GPU' },
    runningMode: 'VIDEO',
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    outputSegmentationMasks: false,
  };
  onStatus(`Downloading the ${model} pose model…`);
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
    model,
    delegate,
    /** Run detection on the current video frame. `timestampMs` must increase monotonically. */
    detect(video, timestampMs) {
      return landmarker.detectForVideo(video, timestampMs);
    },
    close() {
      landmarker.close();
    },
  };
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

/** True when both wrists are clearly above the nose (the "raise both hands" gesture). */
export function handsAboveHead(lm, offset = 0, minVis = 0.5) {
  const y = i => lm[offset + i * STRIDE + 1];
  const v = i => lm[offset + i * STRIDE + 3];
  if (v(0) < minVis || v(15) < minVis || v(16) < minVis) return false;
  const margin = 0.04;
  return y(15) < y(0) - margin && y(16) < y(0) - margin;
}

/**
 * Draw a skeleton from a flat landmark array onto a 2D canvas context.
 * Left-side limbs are orange, right-side limbs are blue, the rest is neutral.
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
