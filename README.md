# Kinesphere

A browser app that watches you dance through your webcam, tracks your body with
[MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)
(or, optionally, [MoveNet](https://www.tensorflow.org/hub/tutorials/movenet) via TensorFlow.js),
and turns a recorded session into a dashboard of movement metrics: how big your kinesphere was,
where in the body the activity was concentrated, how fast you moved over time, and a bird's eye view of
your path across the floor. A Just Dance-style **game mode** scores you against an animated coach.

Everything runs in the browser. Video never leaves your device and is never stored; only the
detected pose landmarks are recorded. Sessions are persisted in the browser (IndexedDB, with room
for recordings of 15 minutes and more) and can be exported to a file so you can share them or move them between browsers.

The app is a plain static site (no build step) and is hosted with GitHub Pages. The dashboard and
tracking demo were conceived and designed by Danielle Lottridge, Andrew Wilson, Nick Young, Becca
Weber, Brenda San Germán Bravo and Maxine Yang (see [Credits](#credits)).

## Using the app

1. **Start camera.** Allow camera access. The pose runtime and model (about 18 MB for MediaPipe
   Lite) are downloaded on first use with a progress bar in the preview. They are then kept in
   the browser's Cache Storage, so later visits load them instantly, even offline; the **?** help
   panel shows how much is cached and can clear it. Then your webcam is shown mirrored with the detected skeleton drawn on top
   (left side orange, right side blue). Stand back so your whole body is in the frame. While less
   than head to hips is in view (or few joints are visible), a small grey *Body partially out of
   frame* note appears above the status line, because pose models degrade when the body is very
   close or only partly in view; the same note is shown in the dashboard replay and the game.
   The buttons in the corner of the preview toggle the pose overlay (<kbd>O</kbd>), hide the camera
   feed so only the overlay is shown on black (<kbd>V</kbd>), and make the camera view fullscreen
   (<kbd>F</kbd>); recording controls stay available in fullscreen. Hiding the feed does not stop
   the camera: detection and recording carry on.
   The **Overlay** selector (or <kbd>S</kbd> to cycle) picks how the pose is drawn: plain
   skeleton, a cartoon stick figure, a neon glow, light trails that follow the hands and feet,
   sparks that fly off fast-moving joints, a constellation, **Sandman**, **Fuzzy blur**,
   **Origami** or **Pencil sketch**. *Origami* folds the body from paper: each limb is a kite
   creased along the bone, with its two facets shaded by a light from the upper left, a
   four-facet torso and a faceted head, casting a soft shadow. *Pencil sketch* draws a loose
   graphite figure (doubled limb contours, sketched joints, a hatched torso) whose lines redraw
   slightly about seven times a second, like hand-drawn animation; with the camera feed hidden
   (and in the replay) it is drawn on a sheet of paper.
   *Sandman* draws a body made of sand that sheds grains as you move: a trickle when still, a
   stream from fast limbs and a burst when a limb speeds up or stops sharply. Grains leave with
   the momentum of the body part that shed them (its velocity, nudged along its acceleration),
   so a sweeping arm flings sand onwards and a sudden stop throws it forwards; then they fall,
   settle on the bottom of the picture and fade. When Sandman is selected, sliders under the
   controls set the particle **Colour**, **Gravity** (0 to 3 times normal; below zero the grains
   float upwards), **Decay time** (0.3 to 8 s before a grain fades) and **Momentum** (how much
   of the body's motion the grains carry: 0 drops them in place, 2 exaggerates it). They take
   effect immediately, are remembered, and **Reset** restores the defaults. *Sparks* also carry
   the joints' momentum. *Fuzzy blur* draws the body as a soft, out-of-focus glow (left side
   orange, right side blue) that lingers for a fraction of a second, so movement smears into a
   motion blur. The same style and settings
   are used in the dashboard replay. Styles are purely visual; the recorded data is identical.
   If you have more than one camera a **Camera** selector appears next to the model selector;
   your choice is remembered. Rear-facing cameras are shown un-mirrored, and **Mirror view** in
   the controls flips the feed, overlay and 3D figure either way.
2. **Record.** Either press **Record**, press <kbd>Space</kbd>, or use *pose control*:
   raise both hands above your head and hold them there for 1.5 s. A 3-second countdown
   follows so you can get into position. The same gesture (or **Stop** / <kbd>Space</kbd>)
   ends the recording; the frames spent holding the stop gesture are trimmed off.
3. **Dashboard.** When the recording stops the session is analysed, saved in the browser and the
   dashboard opens. Recordings use about 0.75–1 MB per minute, so a 15-minute session is about
   11–15 MB.
4. **Sessions.** The *Sessions* tab lists everything saved in this browser, with **Open**,
   **Export JSON**, **CSV** and **Delete** for each session, plus an indicator of how much
   storage the sessions use out of what the browser makes available to the site.
5. **Share.** **Export JSON** downloads a `*.kinesphere.json` file. Anyone can drop that file
   into **Import…** to see the same dashboard. **Export CSV** produces one row per frame with
   `x, y, z, visibility` for each of the 33 landmarks, for analysis elsewhere.

### Model and people settings

The **Model** selector chooses the pose model, and **People** how many dancers to track at once
(up to six). A line under the controls explains the consequences of the current choice, and the
**?** button opens a fuller explanation. Both choices are remembered.

| Model | Landmarks | People | Notes |
| --- | --- | --- | --- |
| MediaPipe Lite / Full / Heavy | 33, with face, hands, feet and an estimated depth per joint | up to 6 | Lite is fastest, Heavy most accurate (30 MB download). With several people, MediaPipe returns poses without identities, so the app keeps people apart by position; identities can swap when dancers cross or leave the frame. |
| MediaPipe Holistic | 33 pose landmarks plus 21 per hand (recorded) and a 478-point face mesh (drawn live only) | 1 | One model for body, hands and face. Fingers show in the overlay and replay; sessions are about twice the size. |
| MoveNet Lightning / Thunder | 17 (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles), no depth | 1 | TensorFlow.js models, often the smoothest on phones and laptops without a usable GPU. Thunder is the more accurate of the two. |
| MoveNet MultiPose | 17, no depth | up to 6 | Built-in tracker keeps identities stable across frames. |

The **?** panel in the app also shows a feature matrix (landmarks, depth, hands, face, people,
identity tracking, speed, download size) generated from the same model list.

Whatever the model, the dashboard metrics use the same 13 major joints, so results are comparable.
MoveNet sessions simply leave the other MediaPipe landmark slots empty (visibility 0), which makes
the overlay simpler (no hands, feet or face detail). Tracking more people lowers the frame rate;
pose control fires when *anyone* raises both hands. The GPU delegate (MediaPipe) or WebGL backend
(MoveNet) is used when available. The theme button in the header cycles between system / light / dark.

### 3D view and VR (optional)

Both the live view and the replay have a **3D view** button that shows the tracked skeleton as a
life-size figure on a floor grid, which you can orbit with the mouse or a finger. Depth comes from
MediaPipe's per-joint z estimate (so MoveNet sessions appear flat). That estimate is noisy and
exaggerated, so a **Depth** slider (remembered across visits) controls how much of it is shown:
35% by default, 0% for a flat figure, 100% for the raw estimate. Depth is also clamped to a
plausible reach around the hips. The figure is scaled so a torso is about 0.5 m with the feet on
the floor. Annotated axes at the dancer's feet show x (red, the dancer's left/right), y (green,
up) and z (blue, towards the camera), and help text under the view explains the orbit, zoom and
pan controls. The **Mirror view** checkbox in the live controls flips the camera feed, overlay and 3D
figure together (on by default for front cameras, off for rear ones, remembered per kind of
camera); the replay has its own Mirror view toggle that flips the 2D replay and its 3D figure.
Untick it to see the dancer as someone facing them would.

On a WebXR-capable browser with a headset (Meta Quest Browser, or Chrome/Edge with a PC headset)
**Enter VR** places the figure about 1.8 m in front of you at life size; **Enter AR** does the same
over passthrough where the browser supports `immersive-ar`. A controller trigger toggles play/pause
in the replay. The buttons only appear when the browser reports headset support; everything else
in the app works without VR. three.js is loaded from the CDN the first time a 3D view is opened.

### Game mode

The **Game** tab turns the camera preview into a dance game in the spirit of Just Dance. Pick a
song, follow the glowing coach, and every move is scored on its beat.

* **Songs.** Three original tracks, synthesised live in the browser with the Web Audio API (no
  audio files are downloaded), each with its own routine:

  | Song | Tempo | Moves | Notes |
  | --- | --- | --- | --- |
  | Sunrise Warm-up (easy) | 96 BPM, one move per bar | 24 | Arms only |
  | Neon Nights (medium) | 116 BPM, a move every 2 beats | 64 | Disco points, egypt arms, knee lifts |
  | Kinesphere Rush (hard) | 128 BPM, a move every 2 beats | 64 | Kicks, squats and stars; keep your legs in view |

  A one-bar count-in (3, 2, 1, Dance!) starts every song.
* **Screen.**
  * The coach, a neon stick figure, is on the left and turns gold for gold moves.
  * You are shown in a mirrored picture-in-picture whose frame glows with how well you match right now.
  * Pictograms of the next moves scroll along the bottom lane into the target box, where each rating pops up.
  * The score, combo and a five-star meter are shown around them.
* **Follow the coach like a mirror.** Move the arm on the same side of the screen as the coach's.
  You are always matched as a mirror image, whatever the *Mirror view* setting.
* **Scoring.**
  * Each move compares the direction of your upper arms, forearms, thighs and shins with the coach's.
    Arms count most; legs count less and only when they are in view.
  * Your best match within a short window around the beat earns **Perfect** (100 points),
    **Super** (70), **Good** (45), **OK** (20) or **X** (0).
  * The window runs from 0.4 s before the beat to 0.3 s after, and the scoring allows about 0.1 s
    for camera and model latency.
  * Gold moves score double. Perfect, Super and Good build a combo.
  * Stars come at 20, 35, 50, 65 and 80% of the song's maximum score, and Superstar at 90%.
* **Groups.** With *People* set above 1, everyone tracked is scored separately and the results
  show one card per player, coloured like the overlay.
* **Best scores** are kept per song in this browser. The music volume and the
  **Record my dance for the dashboard** option are in the song menu. When recording is on, the
  dance is saved to *Sessions* when the song ends, named after the song, and can be opened in the
  dashboard from the results screen. A stopped song is not saved.
* **Controls.** <kbd>Esc</kbd> or ✕ stops the song, or goes back to the song list. ✕ on the song
  list leaves the game. <kbd>F</kbd> or ⛶ toggles fullscreen.

## The dashboard

All distances are divided by the dancer's torso length (median distance from mid-shoulders to
mid-hips over the session), so the metrics do not depend on how far from the camera you stood.
Frames where the shoulders or hips are not visible are ignored. Left and right always refer to
the dancer's own left and right, shown as they appeared in the mirrored preview.

| Section | What it shows |
| --- | --- |
| **Replay** | The recorded skeleton(s), with a scrubber. Moving it also moves the cursor on the time charts. **Metrics overlay** draws the kinesphere spheres around the centre of the torso (the same spheres throughout; the one reached is highlighted), the standing shoulder and knee lines with the current level shaded, markers sized by each body part's current speed, and a readout of the current kinesphere size and limb, level, left/right split over the last second and most active body part. Useful for understanding the dashboard and for checking the classifiers. |
| **Person tabs** | When several people were tracked, one tab per person (coloured like the overlay) switches every metric below, and the overlay, to that dancer. |
| **Kinesphere** | Share of time in a small, medium or large kinesphere, shown as three figures (the most typical pose in each class, drawn inside the three spheres) and a stacked bar. The spheres are concentric around the centre of the torso (midway between the mid-hip and the mid-shoulder) and sized from the dancer's own limb lengths, measured from the skeleton. With MediaPipe models the kinesphere is measured in **3D** from MediaPipe's world landmarks (metric positions in metres): limb lengths are each segment's median 3D length (also shown in cm) and reaches towards or away from the camera count fully. MoveNet sessions and recordings made before 3D support fall back to 2D screen positions, where limb lengths are the 90th percentile of each segment's on-screen length and reaches towards the camera are under-counted. The panel says which mode a session used. Spheres: **small** at the distance of the proximal joints (centre to shoulder), **medium** at the mid-limb joints when outstretched (+ upper arm), **large** at the hands when outstretched (+ forearm). The spheres are sized from the arms (legs measured from the torso centre are longer, so leg-sized spheres would put large out of the arms' reach) and stay fixed for the whole session. Each hand, and each lifted foot, is classed by the sphere it is nearest to, with the boundaries halfway between neighbouring spheres, and the frame takes the largest of those; relaxed arms hanging at the sides therefore count as small. A leg only counts while lifted off the floor, since a standing leg is support rather than reach. Based on the multi-scale kinesphere of Kubota et al., *Emotion Estimation Using Laban Feature Values Based on a Multi-scale Kinesphere* (SUI '24). "Show limb lengths and sphere sizes" lists every estimate. |
| **Body** | **Body part activity**: 13 markers (head, shoulders, elbows, hands, hips, knees, feet) sized and coloured by each part's share of all movement; *Show values* prints the percentages on the figure, and "Show body part data" lists share of movement, active time (share of the session the part moved faster than 0.5 torso lengths/s) and mean speed. **Left / right**: the figure split down the middle with each side's share of movement (head excluded). **Level: where was the activity?**: share of time at a high, medium or low level. This is about position, not which limb moved: *high* when at least one limb is above standing shoulder height, *low* when a hand or the hips are below standing knee height, *medium* otherwise; the head is not counted, standing legs are ignored, and standing heights are measured from the session with the lowest visible joint taken as the floor. The floor is only measured while a foot (ankle, heel or toe) is inside the frame; when the feet are cropped out, the dancer is assumed to be standing and the floor is placed at the session's standing hip height below the hips, so *low* is then only detected from the hands (the overlay says "feet out of view"). |
| **Speed** | Mean speed of the 13 major joints over time (smoothed over 1 s). Numbers are hidden by default, showing the shape of the curve, the fastest moment and the average as a share of the peak; **Show values** adds the numbers in approximate metres per second (torso lengths per second converted with the dancer's torso length measured in 3D, or a typical 50 cm for MoveNet sessions). Speeds use the body scale of the moment, so stepping towards the camera does not inflate them. Tracking glitches are removed first: a Hampel filter drops any landmark that jumps more than 0.35 torso lengths away from the median of its neighbours within ±3 frames (smooth movement, however fast, stays near that median; a glitch does not), then joint speeds that are extreme outliers for that joint (above Q3 + 3 IQR on a log scale, or faster than 20 torso lengths/s, about 10 m/s) and frames where several joints jump at once are left out. The removed share is reported under the chart, and the same clean-up applies to body part activity. |
| **Path on the floor** | A bird's eye view of where the dancer moved in the room, with the camera at the top and its field of view shaded. Distance from the camera is estimated from the apparent size of the torso (pinhole camera model with an assumed 65° horizontal field of view, typical of webcams, and the torso length in metres from MediaPipe's world landmarks, or 50 cm for MoveNet); the sideways position comes from where the hips are across the picture. Positions are median-smoothed over ±0.5 s. Shaded 25 cm squares show where the dancer spent the most time. Stats: floor area used (side to side × front to back, 2nd–98th percentile), distance travelled (the smoothed path sampled every 0.25 s) and typical distance from the camera. These are estimates, roughly ±20%, and a camera with a very different field of view scales them. "Show floor position data" lists the path in metres. |
| **Top shapes** | The three most common body shapes, found by k-means clustering of the pose in every frame (experimental). |

## Data and privacy

* Nothing is uploaded anywhere. The only network requests are for the pose libraries and model
  files (jsDelivr, Google's model storage and, for MoveNet, TF Hub / Kaggle) and the app's own
  static files.
* Downloaded model files (MediaPipe's WebAssembly runtime and `.task` models, MoveNet weights)
  are stored in the browser's Cache Storage under `kinesphere-models-v1`, between 5 and 45 MB
  depending on which models you have used. Clear them from the **?** help panel or by clearing
  the site's data.
* Game best scores and game settings are kept in `localStorage` too (`kinesphere:gameBest`,
  `kinesphere:gameVolume`, `kinesphere:gameRecord`).
* Sessions live in the browser's IndexedDB (database `kinesphere`, store `sessions`), with a
  small index of names and sizes in `localStorage` (`kinesphere:sessions`). IndexedDB is allowed
  far more space than `localStorage`'s 5 MB: typically hundreds of MB or more, depending on the
  browser and free disk space, so recordings of 10–15 minutes (about 1 MB per minute with
  MediaPipe at 30 frames per second) fit comfortably. The app asks the browser to keep the data
  persistent, so it is not evicted when the disk runs low. Sessions saved by earlier versions in
  `localStorage` are moved to IndexedDB automatically. If storage is ever full, the dashboard
  still shows, with a warning that the session was not saved; export it to a file or delete
  older sessions to make room. Clearing the site's data deletes the sessions.
* Only landmarks are recorded, at up to 30 frames per second. The stored frame rate is limited by
  how fast MediaPipe runs on your device (shown in the corner of the preview).

### Session file format

An exported session is JSON:

```jsonc
{
  "format": "kinesphere-session",
  "version": 4,
  "id": "…", "name": "…", "createdAt": "2026-09-02T10:00:00.000Z",
  "durationMs": 220000, "width": 1280, "height": 720,
  "mirrored": true,              // the preview was mirrored; landmarks are in raw camera coordinates
  "model": "lite", "engine": "mediapipe", "people": 1, "trigger": "pose", "frameCount": 6600,
  "landmarkNames": ["nose", "left_eye_inner", …],   // MediaPipe's 33 landmarks, in index order
  "encoding": { … },             // human-readable description of the fields below
  "times": [0, 33, 66, …],       // ms since the start of the recording, one per frame
  "positions": "base64…",        // little-endian int16 [x, y, z] per landmark, ordered frame → person → landmark; divide by 10000
  "visibility": "base64…",       // uint8 per landmark in the same order; divide by 255
  "worldJoints": [0, 11, 12, …], // MediaPipe sessions only, with:
  "worldPositions": "base64…",   //   metric 3D [x, y, z] in metres (origin at the hip midpoint) for those 13 joints,
                                 //   little-endian int16 per value, frame → person → joint; divide by 10000
  "handLandmarks": 21,           // Holistic sessions only, with:
  "handPositions": "base64…",    //   42 landmarks per person per frame (left hand, then right), packed like positions
  "handVisibility": "base64…"    //   1 = hand detected, 0 = not detected
}
```

`people` is the number of tracked slots per frame (1 unless several people were tracked); a
person who was not detected in a frame has visibility 0 for every landmark. `engine` is
`mediapipe`, `holistic` or `movenet`; MoveNet files fill only the 17 matching landmark slots and
always have `z = 0`. `trigger` says how the recording started: `button`, `pose` (the hands-up
gesture) or `game` (recorded during a game song). Holistic files add the hand tracks above (the face mesh is never stored).
World landmarks are stored for the 13 analysed joints only (head, shoulders, elbows, wrists, hips,
knees, ankles), which adds about a third to a session's size. Version 1 (single person, no
`engine`/`people`), version 2 (no hands) and version 3 (no world landmarks) files are still
imported; they are analysed in 2D.

`x` and `y` are MediaPipe's normalised image coordinates (0–1 across the frame, y down);
`z` is depth relative to the hips in roughly the same scale as `x`. Decoding in Python:

```python
import base64, json, numpy as np
s = json.load(open("session.kinesphere.json"))
people = s.get("people", 1)
pos = np.frombuffer(base64.b64decode(s["positions"]), "<i2").reshape(s["frameCount"], people, 33, 3) / 10000
vis = np.frombuffer(base64.b64decode(s["visibility"]), "u1").reshape(s["frameCount"], people, 33) / 255
```

The CSV export contains the same data with one row per frame and columns
`t_ms, nose_x, nose_y, nose_z, nose_visibility, left_eye_inner_x, …`. For multi-person sessions
a leading `person` column (1-based) is added and there is one row per frame per detected person.
Holistic sessions append `left_hand_0_x … right_hand_20_visibility` columns, and MediaPipe sessions
append `nose_world_x_m … right_ankle_world_z_m` columns with the world landmarks in metres.

## Development

There is no build step. Because the app uses ES modules and the camera, it must be served over
HTTP(S) rather than opened as a file:

```sh
python3 -m http.server 8000
# then open http://localhost:8000/
```

Camera access requires a secure context, which browsers grant to `localhost` and to `https://`
pages (GitHub Pages is served over https).

```
index.html        page shell
css/style.css     styles, light and dark themes
js/app.js         UI state: camera, recording, gesture control, dashboard, session library
js/pose.js        detection engines (MediaPipe, MoveNet), person tracker, skeleton drawing
js/effects.js     overlay styles (skeleton, stick figure, neon, trails, sparks, constellation)
js/smoothing.js   One Euro display smoothing for the overlay and replay
js/viewer3d.js    three.js 3D figure viewer with WebXR (VR / AR) sessions
js/game.js        game mode: song menu, Web Audio music, coach and lane rendering, scoring loop
js/game-core.js   game logic without DOM or audio: moves, songs, choreography, pose matching
js/analysis.js    metrics (kinesphere, body part activity, level, speed, path, shapes)
js/charts.js      dependency-free SVG/HTML charts
js/session.js     session model, compact JSON format, IndexedDB storage, JSON/CSV export
js/util.js        formatting and statistics helpers
```

Dependencies are loaded from the CDN at runtime: `@mediapipe/tasks-vision` 1.0.1 with the official
pose landmarker models; only when a MoveNet model is selected, `@tensorflow/tfjs` 4.22 with
`@tensorflow-models/pose-detection` 2.1.3 and the MoveNet weights from TF Hub / Kaggle; and only
when a 3D view is opened, `three` 0.170 (via the import map in `index.html`).

### Hosting on GitHub Pages

The site is served from the root of the `main` branch (Settings → Pages → Deploy from a branch →
`main` / `/ (root)`). All paths are relative, so it also works from a project sub-path such as
`https://<org>.github.io/kinesphere/`.

## Credits

The ideas and design of the dashboard and of this tracking demo, and the Kinesphere project, are
by:

| Name | Email |
| --- | --- |
| Danielle Lottridge | [d.lottridge@auckland.ac.nz](mailto:d.lottridge@auckland.ac.nz) |
| Andrew Wilson | [andrew.wilson@auckland.ac.nz](mailto:andrew.wilson@auckland.ac.nz) |
| Nick Young | [nick.young@auckland.ac.nz](mailto:nick.young@auckland.ac.nz) |
| Becca Weber | [b.weber@auckland.ac.nz](mailto:b.weber@auckland.ac.nz) |
| Brenda San Germán Bravo | [brenda.san.german.bravo@auckland.ac.nz](mailto:brenda.san.german.bravo@auckland.ac.nz) |
| Maxine Yang | [myan565@aucklanduni.ac.nz](mailto:myan565@aucklanduni.ac.nz) |

All at the University of Auckland. The app is built by the University of Auckland Centre for
eResearch. The kinesphere measure follows Kubota et al., *Emotion Estimation Using Laban Feature
Values Based on a Multi-scale Kinesphere* (SUI '24).

## Contributing and license

Kinesphere is open source under the MIT licence (see [LICENSE](LICENSE)). The app links back to
this repository from its header and footer; issues and pull requests are welcome at
<https://github.com/UoA-eResearch/kinesphere>.
