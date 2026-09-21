// 3D and WebXR viewer for tracked poses (three.js, loaded on demand).
// Landmarks are placed in metres in front of the viewer: x/y from the frame, depth from the
// engine's z estimate (MediaPipe; MoveNet sessions are flat), scaled so a torso is ~0.5 m and
// the feet stand on the floor. "Enter VR" / "Enter AR" use WebXR when the browser offers it.

import { CONNECTIONS, NUM_LANDMARKS, STRIDE, LANDMARK_NAMES } from './pose.js';
import { PERSON_COLORS } from './effects.js';
import { median, percentile } from './util.js';

const LEFT = new Set(LANDMARK_NAMES.map((n, i) => (n.startsWith('left') || n.endsWith('_left') ? i : -1)).filter(i => i >= 0));
const RIGHT = new Set(LANDMARK_NAMES.map((n, i) => (n.startsWith('right') || n.endsWith('_right') ? i : -1)).filter(i => i >= 0));
const FIGURE_Z = -1.8;   // metres in front of the viewer
const TORSO_M = 0.5;     // assumed real torso length used to scale the figure
/**
 * MediaPipe's per-landmark z is nominally on the same scale as x, but its magnitude runs well
 * above real depth and it is by far the noisiest coordinate, so it is scaled down by default
 * and clamped to a plausible reach around the hips.
 */
export const DEFAULT_DEPTH = 0.35;
const MAX_DEPTH_TORSOS = 2.5;

let threePromise = null;
/** Load three.js (and OrbitControls) from the CDN, once. */
export function loadThree() {
  threePromise ??= Promise.all([import('three'), import('three/addons/controls/OrbitControls.js')])
    .then(([THREE, mod]) => ({ THREE, OrbitControls: mod.OrbitControls }));
  threePromise.catch(() => { threePromise = null; });
  return threePromise;
}

/** Which immersive WebXR modes this browser offers. */
export async function xrSupport() {
  if (!navigator.xr?.isSessionSupported) return { vr: false, ar: false };
  const probe = mode => navigator.xr.isSessionSupported(mode).catch(() => false);
  const [vr, ar] = await Promise.all([probe('immersive-vr'), probe('immersive-ar')]);
  return { vr, ar };
}

/**
 * Estimate how to place a recorded session in metres: `scale` (metres per frame-height unit)
 * from the median torso length, and `floorY` (frame y of the floor) from where the ankles were.
 */
export function estimateBodyScale(session, slot = 0) {
  const people = session.people || 1;
  const aspect = session.width / session.height;
  const torsos = [], ankles = [], hips = [];
  const { lm, frameCount } = session;
  const v = (o, i) => lm[o + i * STRIDE + 3];
  for (let f = 0; f < frameCount; f++) {
    const o = (f * people + slot) * NUM_LANDMARKS * STRIDE;
    if (v(o, 11) < 0.5 || v(o, 12) < 0.5 || v(o, 23) < 0.5 || v(o, 24) < 0.5) continue;
    const sx = (lm[o + 11 * STRIDE] + lm[o + 12 * STRIDE]) / 2 * aspect, sy = (lm[o + 11 * STRIDE + 1] + lm[o + 12 * STRIDE + 1]) / 2;
    const hx = (lm[o + 23 * STRIDE] + lm[o + 24 * STRIDE]) / 2 * aspect, hy = (lm[o + 23 * STRIDE + 1] + lm[o + 24 * STRIDE + 1]) / 2;
    torsos.push(Math.hypot(sx - hx, sy - hy));
    hips.push(hy);
    const ay = Math.max(v(o, 27) >= 0.5 ? lm[o + 27 * STRIDE + 1] : -1, v(o, 28) >= 0.5 ? lm[o + 28 * STRIDE + 1] : -1);
    if (ay >= 0) ankles.push(ay);
  }
  const torso = median(torsos);
  if (!Number.isFinite(torso) || torso <= 0) return { scale: 2.2, floorY: 0.95 };
  const floorY = ankles.length >= 10 ? percentile(ankles, 0.95) + torso * 0.15 : median(hips) + torso * 2.3;
  return { scale: TORSO_M / torso, floorY };
}

/**
 * Create a viewer inside `container`. Feed it poses with `setPeople([{ lm, offset } | null, ...])`.
 * The viewer runs its own render loop (also inside a WebXR session); `requestFrame(cb)` lets a
 * playback loop ride on it so playback keeps going while presenting to a headset.
 */
export async function createViewer3D(container, { people = 1, aspect = 16 / 9, mirrored: mirrorInit = true } = {}) {
  let mirrored = Boolean(mirrorInit);
  const { THREE, OrbitControls } = await loadThree();
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.xr.enabled = true;
  renderer.domElement.className = 'viewer3d-canvas';
  container.append(renderer.domElement);

  const scene = new THREE.Scene();
  const background = new THREE.Color(0x111111);
  scene.background = background;
  const camera = new THREE.PerspectiveCamera(60, 1, 0.05, 60);
  camera.position.set(0.6, 1.5, 1.2);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.9, FIGURE_Z);
  controls.enableDamping = true;
  controls.minDistance = 0.3;
  controls.maxDistance = 12;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x334, 1.2));
  const sun = new THREE.DirectionalLight(0xffffff, 0.9);
  sun.position.set(2, 4, 2);
  scene.add(sun);
  const grid = new THREE.GridHelper(8, 16, 0x5a5a5a, 0x2c2c2c);
  grid.position.set(0, 0, FIGURE_Z);
  scene.add(grid);

  // Annotated axes at the dancer's feet: x left/right, y up, z towards the camera / viewer.
  const axes = new THREE.Group();
  axes.position.set(0, 0.005, FIGURE_Z);
  const label = (text, color, x, y, z) => {
    const c = document.createElement('canvas');
    c.width = 512; c.height = 64;
    const g = c.getContext('2d');
    g.font = 'bold 34px system-ui, sans-serif';
    g.textAlign = 'center'; g.textBaseline = 'middle';
    g.lineWidth = 6; g.strokeStyle = 'rgba(0,0,0,0.8)'; g.strokeText(text, 256, 32);
    g.fillStyle = color; g.fillText(text, 256, 32);
    const tex = new THREE.CanvasTexture(c);
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
    sp.scale.set(1.0, 0.125, 1);
    sp.position.set(x, y, z);
    return sp;
  };
  const arrowLen = 0.6;
  axes.add(new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), arrowLen, 0xe34948, 0.08, 0.05));
  axes.add(new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), arrowLen, 0x2ea043, 0.08, 0.05));
  axes.add(new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), arrowLen, 0x3987e5, 0.08, 0.05));
  let xLabel = null;
  const setXLabel = () => {
    if (xLabel) { axes.remove(xLabel); xLabel.material.map.dispose(); xLabel.material.dispose(); }
    xLabel = label(mirrored ? "x · dancer's right" : "x · dancer's left", '#ff8a8a', arrowLen + 0.5, 0.05, 0);
    axes.add(xLabel);
  };
  setXLabel();
  axes.add(label('y · up', '#7ee787', 0, arrowLen + 0.1, 0));
  axes.add(label('z · to camera', '#8ec5ff', 0, 0.05, arrowLen + 0.25));
  scene.add(axes);

  const sphere = new THREE.SphereGeometry(1, 12, 8);
  const cylinder = new THREE.CylinderGeometry(1, 1, 1, 8, 1);
  const hidden = new THREE.Matrix4().makeScale(0, 0, 0);
  const figures = Array.from({ length: people }, (_, p) => {
    const pc = PERSON_COLORS[p % PERSON_COLORS.length];
    const material = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.05 });
    const joints = new THREE.InstancedMesh(sphere, material, NUM_LANDMARKS);
    const bones = new THREE.InstancedMesh(cylinder, material, CONNECTIONS.length);
    const color = new THREE.Color();
    for (let i = 0; i < NUM_LANDMARKS; i++) {
      joints.setMatrixAt(i, hidden);
      joints.setColorAt(i, color.set(LEFT.has(i) ? pc.left : RIGHT.has(i) ? pc.right : pc.mid));
    }
    CONNECTIONS.forEach(([a, b], i) => {
      bones.setMatrixAt(i, hidden);
      const side = LEFT.has(a) && LEFT.has(b) ? pc.left : RIGHT.has(a) && RIGHT.has(b) ? pc.right : pc.mid;
      bones.setColorAt(i, color.set(side));
    });
    joints.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    bones.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    // Instances move every frame and start hidden (scaled to zero at the origin), so the cached
    // bounding sphere would be wrong and could cull a dancer who appears later. Never cull.
    joints.frustumCulled = false;
    bones.frustumCulled = false;
    scene.add(joints, bones);
    return { joints, bones };
  });

  let scale = 2.2, floorY = 0.95, depth = DEFAULT_DEPTH;
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), qi = new THREE.Quaternion();
  const va = new THREE.Vector3(), vb = new THREE.Vector3(), mid = new THREE.Vector3(), sz = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0);
  const toWorld = (lm, o, i, out) => {
    const k = o + i * STRIDE;
    const zMax = MAX_DEPTH_TORSOS * TORSO_M;
    const z = Math.max(-zMax, Math.min(zMax, lm[k + 2] * aspect * scale * depth));
    out.set((mirrored ? 0.5 - lm[k] : lm[k] - 0.5) * aspect * scale, (floorY - lm[k + 1]) * scale, FIGURE_Z - z);
  };

  let lastSlots = null, lastMinVis = 0.5;
  function setPeople(slots, minVis = 0.5) {
    lastSlots = slots; lastMinVis = minVis;
    figures.forEach((fig, p) => {
      const slot = slots?.[p];
      const { joints, bones } = fig;
      if (!slot) {
        for (let i = 0; i < NUM_LANDMARKS; i++) joints.setMatrixAt(i, hidden);
        for (let i = 0; i < CONNECTIONS.length; i++) bones.setMatrixAt(i, hidden);
      } else {
        const { lm, offset = 0 } = slot;
        const vis = i => lm[offset + i * STRIDE + 3] >= minVis;
        for (let i = 0; i < NUM_LANDMARKS; i++) {
          if (!vis(i)) { joints.setMatrixAt(i, hidden); continue; }
          const r = i >= 1 && i <= 10 ? 0.012 : 0.022;
          toWorld(lm, offset, i, va);
          joints.setMatrixAt(i, m.compose(va, qi, sz.set(r, r, r)));
        }
        CONNECTIONS.forEach(([a, b], i) => {
          if (!vis(a) || !vis(b)) { bones.setMatrixAt(i, hidden); return; }
          toWorld(lm, offset, a, va);
          toWorld(lm, offset, b, vb);
          mid.addVectors(va, vb).multiplyScalar(0.5);
          vb.sub(va);
          const len = vb.length();
          if (len < 1e-4) { bones.setMatrixAt(i, hidden); return; }
          q.setFromUnitVectors(up, vb.divideScalar(len));
          const r = a <= 10 && b <= 10 ? 0.006 : 0.013;
          bones.setMatrixAt(i, m.compose(mid, q, sz.set(r, len, r)));
        });
      }
      joints.instanceMatrix.needsUpdate = true;
      bones.instanceMatrix.needsUpdate = true;
    });
  }

  /** Flip the figure left-to-right (mirror view) and re-project at once. */
  function setMirror(on) {
    if (mirrored === Boolean(on)) return;
    mirrored = Boolean(on);
    setXLabel();
    if (lastSlots) setPeople(lastSlots, lastMinVis);
  }

  /** Update placement; the current poses are re-projected at once so slider changes show live. */
  function setScale(opts = {}) {
    const before = `${scale}|${floorY}|${depth}`;
    if (opts.scale > 0) scale = opts.scale;
    if (Number.isFinite(opts.floorY)) floorY = opts.floorY;
    if (Number.isFinite(opts.depth)) depth = Math.max(0, Math.min(1.5, opts.depth));
    if (lastSlots && before !== `${scale}|${floorY}|${depth}`) setPeople(lastSlots, lastMinVis);
  }

  function resize() {
    const w = container.clientWidth || 640, h = container.clientHeight || Math.round((container.clientWidth || 640) * 9 / 16);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  const ro = new ResizeObserver(resize);
  ro.observe(container);
  resize();

  let pendingFrame = null;
  const frameListeners = new Set(), selectListeners = new Set(), endListeners = new Set();
  renderer.setAnimationLoop(time => {
    const cb = pendingFrame;
    pendingFrame = null;
    if (cb) cb(time);
    for (const f of frameListeners) f(time);
    if (!renderer.xr.isPresenting) controls.update();
    renderer.render(scene, camera);
  });

  for (let i = 0; i < 2; i++) {
    const controller = renderer.xr.getController(i);
    controller.addEventListener('select', () => { for (const f of selectListeners) f(); });
    scene.add(controller);
  }

  let session = null;
  async function enterXR(mode = 'immersive-vr') {
    if (session) { await session.end(); return; }
    const s = await navigator.xr.requestSession(mode, { optionalFeatures: ['local-floor', 'bounded-floor', 'hand-tracking'] });
    session = s;
    renderer.xr.setReferenceSpaceType('local-floor');
    if (mode === 'immersive-ar') { scene.background = null; grid.visible = false; axes.visible = false; }
    s.addEventListener('end', () => {
      session = null;
      scene.background = background;
      grid.visible = true;
      axes.visible = true;
      resize();
      for (const f of endListeners) f();
    });
    await renderer.xr.setSession(s);
  }

  const posOut = new THREE.Vector3();
  return {
    setPeople,
    setScale,
    setMirror,
    get mirrored() { return mirrored; },
    /** World position of a joint as last projected (null when hidden). Mainly for tests. */
    jointPosition(person, index) {
      const fig = figures[person];
      if (!fig) return null;
      fig.joints.getMatrixAt(index, m);
      m.decompose(posOut, q, sz);
      return sz.x === 0 ? null : { x: posOut.x, y: posOut.y, z: posOut.z };
    },
    requestFrame(cb) { pendingFrame = cb; },
    onFrame(f) { frameListeners.add(f); return () => frameListeners.delete(f); },
    onSelect(f) { selectListeners.add(f); return () => selectListeners.delete(f); },
    onSessionEnd(f) { endListeners.add(f); return () => endListeners.delete(f); },
    enterXR,
    get presenting() { return Boolean(session); },
    destroy() {
      ro.disconnect();
      renderer.setAnimationLoop(null);
      session?.end().catch(() => {});
      controls.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}
