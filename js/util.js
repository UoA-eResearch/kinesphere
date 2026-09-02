// Small shared helpers (formatting + basic statistics).

export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** 83000 -> "1:23" */
export function fmtClock(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

/** 220000 -> "3m 40s" */
export function fmtDuration(ms) {
  const s = Math.max(0, Math.round(ms / 1000));
  const m = Math.floor(s / 60);
  return m ? `${m}m ${s % 60}s` : `${s}s`;
}

export function fmtPct(fraction, digits = 0) {
  if (!Number.isFinite(fraction)) return '–';
  return `${(fraction * 100).toFixed(digits)}%`;
}

export function fmtBytes(b) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(0)} KB`;
  return `${(b / 1024 / 1024).toFixed(2)} MB`;
}

export function fmtDate(iso) {
  const d = new Date(iso);
  return d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

export function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

export function mean(arr) {
  let s = 0, n = 0;
  for (const v of arr) if (Number.isFinite(v)) { s += v; n++; }
  return n ? s / n : NaN;
}

/** Percentile (0..1) of the finite values in arr. */
export function percentile(arr, p) {
  const a = Array.from(arr).filter(Number.isFinite).sort((x, y) => x - y);
  if (!a.length) return NaN;
  const idx = clamp((a.length - 1) * p, 0, a.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  return a[lo] + (a[hi] - a[lo]) * (idx - lo);
}

export const median = arr => percentile(arr, 0.5);

/** Deterministic PRNG (mulberry32) so repeated analyses give identical results. */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
