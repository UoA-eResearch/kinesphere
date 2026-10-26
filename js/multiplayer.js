// Dance together online: share a link, and everyone in the session sees everyone's avatar.
// Browsers talk to each other directly over encrypted WebRTC data channels (PeerJS); a PeerJS
// signalling server only introduces them, and PeerJS's TURN servers relay the data when a direct
// connection is impossible. Only pose landmarks are sent, never video. The person who creates the
// link hosts the session: guests connect to the host, who passes each guest's poses on to the rest.

const PEERJS_URL = 'https://cdn.jsdelivr.net/npm/peerjs@1.5.5/dist/peerjs.min.js';
const ID_PREFIX = 'kinesphere-';
export const MAX_GUESTS = 5;
const SEND_INTERVAL_MS = 50;       // up to 20 pose updates per second
const KEEPALIVE_MS = 1000;         // send an empty update this often when nobody is in view
const MAX_BUFFERED = 256 * 1024;   // skip a frame rather than queue behind a slow connection
const CONNECT_TIMEOUT_MS = 20000;
const POSE_VALUES = 33 * 3;        // x, y, visibility per landmark

let peerPromise = null;
function loadPeerJS() {
  if (window.Peer) return Promise.resolve(window.Peer);
  peerPromise ??= new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = PEERJS_URL;
    s.async = true;
    s.onload = () => (window.Peer ? resolve(window.Peer) : reject(new Error('PeerJS did not load')));
    s.onerror = () => { peerPromise = null; reject(new Error('Could not load the connection library. Check your network.')); };
    document.head.append(s);
  });
  return peerPromise;
}

/**
 * Pack poses (flat MediaPipe-order Float32Arrays, 33 x [x, y, z, visibility]) into a small binary
 * message: uint16 count, then x, y, visibility per landmark. Coordinates cover -0.5..1.5 (landmarks
 * can sit just outside the picture) at about 1/32768 of the frame.
 */
export function packPoses(poses) {
  const out = new Uint16Array(1 + poses.length * POSE_VALUES);
  out[0] = poses.length;
  const q = v => Math.max(0, Math.min(65535, Math.round(((Number.isFinite(v) ? v : -1) + 0.5) * 32767.5)));
  poses.forEach((lm, p) => {
    const o = 1 + p * POSE_VALUES;
    for (let i = 0; i < 33; i++) {
      out[o + i * 3] = q(lm[i * 4]);
      out[o + i * 3 + 1] = q(lm[i * 4 + 1]);
      out[o + i * 3 + 2] = Math.round(Math.max(0, Math.min(1, lm[i * 4 + 3] || 0)) * 65535);
    }
  });
  return out.buffer;
}

export function unpackPoses(buffer) {
  const u = buffer instanceof Uint16Array ? buffer : new Uint16Array(buffer instanceof ArrayBuffer ? buffer : buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength));
  const n = Math.min(6, u[0] || 0);
  if (u.length < 1 + n * POSE_VALUES) return [];
  const poses = [];
  for (let p = 0; p < n; p++) {
    const lm = new Float32Array(132), o = 1 + p * POSE_VALUES;
    for (let i = 0; i < 33; i++) {
      lm[i * 4] = u[o + i * 3] / 32767.5 - 0.5;
      lm[i * 4 + 1] = u[o + i * 3 + 1] / 32767.5 - 0.5;
      lm[i * 4 + 3] = u[o + i * 3 + 2] / 65535;
    }
    poses.push(lm);
  }
  return poses;
}

/** A room code: 10 characters a person can read out (no 0/o, 1/l). */
export function makeRoomCode() {
  const abc = 'abcdefghijkmnpqrstuvwxyz23456789';
  const bytes = crypto.getRandomValues(new Uint8Array(10));
  return Array.from(bytes, b => abc[b % abc.length]).join('');
}

/** The room code in a link's #join=... fragment, or null. */
export function roomFromHash(hash = location.hash) {
  const m = /[#&]join=([a-z0-9]{6,24})\b/i.exec(hash || '');
  return m ? m[1].toLowerCase() : null;
}

/** The link that joins a room (the current page with #join=code; ?signal= is kept). */
export function inviteLink(code) {
  return `${location.origin}${location.pathname}${location.search}#join=${code}`;
}

/**
 * Signalling server: PeerJS's free cloud server by default. `?signal=wss://host[:port][/path]` on
 * the page address points at your own PeerJS server instead (e.g. for an organisation's network).
 */
export function signalOptions(search = location.search) {
  const raw = new URLSearchParams(search).get('signal');
  if (!raw) return {};
  try {
    const u = new URL(raw);
    const secure = u.protocol === 'wss:' || u.protocol === 'https:';
    return { host: u.hostname, port: Number(u.port) || (secure ? 443 : 80), path: u.pathname || '/', secure, key: 'peerjs' };
  } catch {
    return {};
  }
}

const cleanName = n => String(n ?? '').replace(/[\u0000-\u001f<>]/g, '').trim().slice(0, 24) || 'Dancer';

/**
 * A dance-together session. Callbacks:
 *   onPeers(list of { id, name, self, host })   whenever someone joins or leaves
 *   onPose(id, poses, aspect)                   a participant's latest poses (normalised to their frame)
 *   onStatus({ state: 'connecting'|'hosting'|'joined'|'ended', message })
 */
export function createSession({ onPeers, onPose, onStatus }) {
  let peer = null, role = null, code = null, myName = 'Dancer', myAspect = 16 / 9, ended = false;
  const conns = new Map();   // host: guest id -> { conn, name }; guest: host id -> { conn, name }
  const names = new Map();   // every participant's name by peer id
  let lastSent = 0, connectTimer = 0;

  const status = (state, message = '') => onStatus?.({ state, message });
  const peersList = () => {
    const list = [];
    if (role === 'host') {
      list.push({ id: peer?.id, name: myName, self: true, host: true });
      for (const [id, c] of conns) list.push({ id, name: c.name ?? 'Joining…', self: false, host: false });
    } else {
      for (const [id, name] of names) list.push({ id, name, self: id === peer?.id, host: id === ID_PREFIX + code });
    }
    return list;
  };
  const announce = () => onPeers?.(peersList());
  const safeSend = (conn, msg) => {
    try {
      if (!conn.open) return;
      if (conn.dataChannel && conn.dataChannel.bufferedAmount > MAX_BUFFERED && msg.t === 'pose') return;
      conn.send(msg);
      const entry = conns.get(conn.peer);
      if (entry) entry.sent = (entry.sent ?? 0) + 1;
    } catch (err) { console.warn('send failed', err); }
  };

  function hostRoster() {
    const peers = peersList().map(({ id, name, host }) => ({ id, name, host }));
    for (const { conn } of conns.values()) safeSend(conn, { t: 'roster', peers });
    announce();
  }

  function wireHostConnection(conn) {
    if (conns.size >= MAX_GUESTS) {
      conn.on('open', () => { conn.send({ t: 'full' }); setTimeout(() => conn.close(), 500); });
      return;
    }
    const entry = { conn, name: null };
    conns.set(conn.peer, entry);
    conn.on('data', msg => {
      if (!msg || typeof msg !== 'object') return;
      if (msg.t === 'hello') { entry.name = cleanName(msg.name); hostRoster(); return; }
      if (msg.t === 'pose' && msg.b) {
        onPose?.(conn.peer, unpackPoses(msg.b), Number(msg.a) || 16 / 9);
        const relay = { t: 'pose', id: conn.peer, a: msg.a, b: msg.b }; // id set by the host, so it can't be spoofed
        for (const [id, other] of conns) if (id !== conn.peer) safeSend(other.conn, relay);
      }
    });
    const drop = () => { if (conns.delete(conn.peer)) { onPose?.(conn.peer, null); hostRoster(); } };
    conn.on('close', drop);
    conn.on('error', drop);
    announce();
  }

  function wireGuestConnection(conn) {
    conns.set(conn.peer, { conn, name: null });
    conn.on('open', () => {
      clearTimeout(connectTimer);
      safeSend(conn, { t: 'hello', name: myName, a: myAspect });
      status('joined');
    });
    conn.on('data', msg => {
      if (!msg || typeof msg !== 'object') return;
      if (msg.t === 'full') { leave('This session is full (up to 6 dancers).'); return; }
      if (msg.t === 'roster' && Array.isArray(msg.peers)) {
        const before = new Set(names.keys());
        names.clear();
        for (const p of msg.peers.slice(0, MAX_GUESTS + 1)) names.set(String(p.id), cleanName(p.name));
        for (const id of before) if (!names.has(id)) onPose?.(id, null);
        announce();
        return;
      }
      if (msg.t === 'pose' && msg.b) onPose?.(String(msg.id || conn.peer), unpackPoses(msg.b), Number(msg.a) || 16 / 9);
    });
    conn.on('close', () => { if (!ended) leave('The host ended the session.'); });
    conn.on('error', () => { if (!ended) leave('The connection to the host was lost.'); });
  }

  function onPeerError(err) {
    const type = err?.type || '';
    if (type === 'peer-unavailable') leave('This session isn’t available. The host may have ended it, or the link is incomplete.');
    else if (type === 'unavailable-id') leave('That room code is already in use. Try again.');
    else if (['network', 'server-error', 'socket-error', 'socket-closed'].includes(type)) leave('Could not reach the connection server. Check your network and try again.');
    else if (type === 'browser-incompatible') leave('This browser cannot make peer-to-peer connections.');
    else console.warn('PeerJS', type, err);
  }

  async function makePeer(id) {
    const Peer = await loadPeerJS();
    return new Promise((resolve, reject) => {
      const p = new Peer(id, { debug: 0, ...signalOptions() }); // errors are reported through onStatus
      const fail = err => reject(err);
      p.once('open', () => { p.off('error', fail); resolve(p); });
      p.once('error', fail);
    });
  }

  async function host(name, aspect) {
    if (peer) leave();
    ended = false; role = 'host'; myName = cleanName(name); myAspect = aspect || myAspect;
    code = makeRoomCode();
    status('connecting', 'Creating a link…');
    try {
      peer = await makePeer(ID_PREFIX + code);
    } catch (err) {
      onPeerError(err);
      throw err;
    }
    peer.on('connection', wireHostConnection);
    peer.on('error', onPeerError);
    peer.on('disconnected', () => { if (!ended) peer.reconnect(); }); // lost the signalling server; existing links keep working
    status('hosting');
    announce();
    return code;
  }

  async function join(roomCode, name, aspect) {
    if (peer) leave();
    ended = false; role = 'guest'; code = roomCode; myName = cleanName(name); myAspect = aspect || myAspect;
    status('connecting', 'Connecting…');
    try {
      peer = await makePeer(`${ID_PREFIX}g-${makeRoomCode()}`); // our own random id: no round trip to the server
    } catch (err) {
      onPeerError(err);
      throw err;
    }
    peer.on('error', onPeerError);
    names.set(peer.id, myName);
    const conn = peer.connect(ID_PREFIX + roomCode, { reliable: true, metadata: { name: myName } });
    wireGuestConnection(conn);
    connectTimer = setTimeout(() => { if (!conn.open) leave('Could not connect to the host. Their network may block direct connections; try again, or from another network.'); }, CONNECT_TIMEOUT_MS);
  }

  /** Send my current poses (throttled); call with every detection. */
  function sendPoses(poses, now = performance.now()) {
    if (!peer || ended || !conns.size) return;
    const quiet = !poses.length;
    if (now - lastSent < (quiet ? KEEPALIVE_MS : SEND_INTERVAL_MS)) return;
    lastSent = now;
    const msg = { t: 'pose', id: peer.id, a: myAspect, b: packPoses(poses.slice(0, 6)) };
    for (const { conn } of conns.values()) safeSend(conn, msg);
  }

  function leave(message = '') {
    if (ended && !peer) return;
    ended = true;
    clearTimeout(connectTimer);
    for (const { conn } of conns.values()) { try { conn.close(); } catch { /* closed */ } }
    for (const id of [...conns.keys(), ...names.keys()]) onPose?.(id, null);
    conns.clear(); names.clear();
    try { peer?.destroy(); } catch { /* gone */ }
    peer = null; role = null; code = null;
    status('ended', message);
    onPeers?.([]);
  }

  return {
    host, join, sendPoses, leave,
    /** Per-connection state, for diagnostics. */
    stats() { return [...conns].map(([id, c]) => ({ id, name: c.name, open: c.conn.open, buffered: c.conn.dataChannel?.bufferedAmount ?? null, sent: c.sent ?? 0 })); },
    setName(n) { myName = cleanName(n); if (role === 'host') hostRoster(); },
    setAspect(a) { if (a > 0) myAspect = a; },
    get role() { return role; },
    get code() { return code; },
    get active() { return Boolean(peer) && !ended; },
    get myId() { return peer?.id ?? null; },
    get count() { return peersList().length; },
  };
}
