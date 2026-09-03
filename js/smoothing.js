// Display smoothing for the pose overlay: a One Euro filter per landmark value.
// It is applied at render time, so between detections the drawn pose eases toward the latest
// sample instead of jumping, and at rest the cutoff drops to kill jitter while fast movement
// raises it to keep the overlay responsive. Recorded landmarks are never smoothed here.

import { FRAME_SIZE } from './pose.js';

export const SMOOTHING_LEVELS = [
  { id: 'off', label: 'Off', help: 'Raw detections, no smoothing.' },
  { id: 'normal', label: 'Normal', minCutoff: 1.0, beta: 4.0, dCutoff: 1.0, help: 'Removes jitter while keeping fast moves responsive.' },
  { id: 'strong', label: 'Strong', minCutoff: 0.4, beta: 1.5, dCutoff: 1.0, help: 'Very steady but adds visible lag; good for slow, held movement.' },
];
export const DEFAULT_SMOOTHING = 'normal';

function alphaFor(cutoff, dt) {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

/**
 * Smoother for up to `maxPeople` person slots over buffers of `size` floats (x,y,z,visibility
 * groups). `apply()` returns a buffer owned by the smoother.
 */
export function createSmoother(levelId = DEFAULT_SMOOTHING, maxPeople = 6, size = FRAME_SIZE) {
  const level = SMOOTHING_LEVELS.find(l => l.id === levelId) ?? SMOOTHING_LEVELS[1];
  const slots = Array.from({ length: maxPeople }, () => ({ x: new Float32Array(size), dx: new Float32Array(size), primed: false }));
  return {
    id: level.id,
    /**
     * @param {Float32Array} lm raw landmarks; `offset` selects the person inside a larger array
     * @param {number} person slot index
     * @param {number} dt seconds since the previous call for this slot
     */
    apply(lm, offset, person, dt) {
      const s = slots[person];
      if (level.id === 'off') return offset || lm.length !== size ? lm.subarray(offset, offset + size) : lm;
      if (!s.primed || !(dt > 0)) {
        s.x.set(lm.subarray(offset, offset + size));
        s.dx.fill(0);
        s.primed = true;
        return s.x;
      }
      const step = Math.min(dt, 0.25);
      const aD = alphaFor(level.dCutoff, step);
      for (let i = 0; i < size; i++) {
        const x = lm[offset + i];
        const prev = s.x[i];
        const dx = (x - prev) / step;
        const dxHat = s.dx[i] + aD * (dx - s.dx[i]);
        s.dx[i] = dxHat;
        // visibility (every 4th value) only needs the base cutoff
        const cutoff = (i & 3) === 3 ? level.minCutoff : level.minCutoff + level.beta * Math.abs(dxHat);
        const a = alphaFor(cutoff, step);
        s.x[i] = prev + a * (x - prev);
      }
      return s.x;
    },
    reset(person) {
      if (person == null) for (const s of slots) s.primed = false;
      else if (slots[person]) slots[person].primed = false;
    },
  };
}
