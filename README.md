# Kinesphere

A browser app that watches you dance through your webcam, tracks your body with
[MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker)
(or, optionally, [MoveNet](https://www.tensorflow.org/hub/tutorials/movenet) via TensorFlow.js),
and turns a recorded session into a dashboard of movement metrics: how big your kinesphere was,
where in the body the activity was concentrated, how fast you moved over time, and how you moved
around the frame.

Everything runs in the browser. Video never leaves your device and is never stored; only the
detected pose landmarks are recorded. Sessions are persisted in the browser's `localStorage`
and can be exported to a file so you can share them or move them between browsers.

The app is a plain static site (no build step) and is hosted with GitHub Pages.

## Using the app

1. **Start camera.** Allow camera access. The lite pose model (about 6 MB) is downloaded on first
   use, then your webcam is shown mirrored with the detected skeleton drawn on top
   (left side orange, right side blue). Stand back so your whole body is in the frame.
   The buttons in the corner of the preview toggle the pose overlay (<kbd>O</kbd>), hide the camera
   feed so only the overlay is shown on black (<kbd>V</kbd>), and make the camera view fullscreen
   (<kbd>F</kbd>); recording controls stay available in fullscreen. Hiding the feed does not stop
   the camera: detection and recording carry on.
   The **Overlay** selector (or <kbd>S</kbd> to cycle) picks how the pose is drawn: plain
   skeleton, a cartoon stick figure, a neon glow, light trails that follow the hands and feet,
   sparks that fly off fast-moving joints, or a constellation. The same style is used in the
   dashboard replay. Styles are purely visual; the recorded data is identical.
   If you have more than one camera a **Camera** selector appears next to the model selector;
   your choice is remembered. Rear-facing cameras are shown un-mirrored.
2. **Record.** Either press **Record**, press <kbd>Space</kbd>, or use *pose control*:
   raise both hands above your head and hold them there for 1.5 s. A 3-second countdown
   follows so you can get into position. The same gesture (or **Stop** / <kbd>Space</kbd>)
   ends the recording; the frames spent holding the stop gesture are trimmed off.
3. **Dashboard.** When the recording stops the session is analysed, saved to `localStorage`
   and the dashboard opens.
4. **Sessions.** The *Sessions* tab lists everything saved in this browser, with **Open**,
   **Export JSON**, **CSV** and **Delete** for each session, plus an indicator of how much of
   the roughly 5 MB storage budget is used.
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
| MoveNet Lightning / Thunder | 17 (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles), no depth | 1 | TensorFlow.js models, often the smoothest on phones and laptops without a usable GPU. Thunder is the more accurate of the two. |
| MoveNet MultiPose | 17, no depth | up to 6 | Built-in tracker keeps identities stable across frames. |

Whatever the model, the dashboard metrics use the same 13 major joints, so results are comparable.
MoveNet sessions simply leave the other MediaPipe landmark slots empty (visibility 0), which makes
the overlay simpler (no hands, feet or face detail). Tracking more people lowers the frame rate;
pose control fires when *anyone* raises both hands. The GPU delegate (MediaPipe) or WebGL backend
(MoveNet) is used when available. The theme button in the header cycles between system / light / dark.

## The dashboard

All distances are divided by the dancer's torso length (median distance from mid-shoulders to
mid-hips over the session), so the metrics do not depend on how far from the camera you stood.
Frames where the shoulders or hips are not visible are ignored. Left and right always refer to
the dancer's own left and right, shown as they appeared in the mirrored preview.

| Section | What it shows |
| --- | --- |
| **Replay** | The recorded skeleton(s), with a scrubber. Moving it also moves the cursor on the time charts. |
| **Person tabs** | When several people were tracked, one tab per person (coloured like the overlay) switches every metric below to that dancer. |
| **Kinesphere** | Share of time spent with a *small*, *medium* or *large* kinesphere, an "openness over time" chart and a joint angle table. Openness averages the shoulder, elbow, hip and knee angles (0° folded, 180° fully extended), weighting arms 60% and legs 40%. Below 0.55 counts as small, 0.55–0.80 medium, above 0.80 large. The angle table gives the mean and the 5th–95th percentile range (a proxy for range of motion) for each joint on each side. |
| **Where was the activity?** | A heatmap of movement share by body region: rows upper (elbows, wrists) / mid (shoulders, hips) / lower (knees, ankles) and columns left / right. Next to it, a body diagram whose joints are sized and coloured by their mean speed. Activity is the mean speed of the joints in each region. |
| **Speed** | Mean joint speed over time, in torso lengths per second (smoothed over 1 s), with the mean and peak. |
| **Space** | The path of the hips through the camera frame, coloured from start to end, over shaded cells showing where you spent the most time. Coverage is the 2nd–98th percentile spread of the hip position; path length is the distance the hips travelled. |
| **Top shapes** | The three most common body shapes, found by k-means clustering of the pose in every frame (experimental). |

## Data and privacy

* Nothing is uploaded anywhere. The only network requests are for the pose libraries and model
  files (jsDelivr, Google's model storage and, for MoveNet, TF Hub / Kaggle) and the app's own
  static files.
* Sessions live in `localStorage` under keys starting with `kinesphere:`. Browsers allow about
  5 MB per site, which is roughly 6–8 minutes of recording at 30 frames per second. When the
  quota is exceeded the dashboard still shows, with a warning that the session was not saved;
  export it to a file or delete older sessions to make room.
* Only landmarks are recorded, at up to 30 frames per second. The stored frame rate is limited by
  how fast MediaPipe runs on your device (shown in the corner of the preview).

### Session file format

An exported session is JSON:

```jsonc
{
  "format": "kinesphere-session",
  "version": 2,
  "id": "…", "name": "…", "createdAt": "2026-09-02T10:00:00.000Z",
  "durationMs": 220000, "width": 1280, "height": 720,
  "mirrored": true,              // the preview was mirrored; landmarks are in raw camera coordinates
  "model": "lite", "engine": "mediapipe", "people": 1, "trigger": "pose", "frameCount": 6600,
  "landmarkNames": ["nose", "left_eye_inner", …],   // MediaPipe's 33 landmarks, in index order
  "encoding": { … },             // human-readable description of the fields below
  "times": [0, 33, 66, …],       // ms since the start of the recording, one per frame
  "positions": "base64…",        // little-endian int16 [x, y, z] per landmark, ordered frame → person → landmark; divide by 10000
  "visibility": "base64…"        // uint8 per landmark in the same order; divide by 255
}
```

`people` is the number of tracked slots per frame (1 unless several people were tracked); a
person who was not detected in a frame has visibility 0 for every landmark. `engine` is
`mediapipe` or `movenet`; MoveNet files fill only the 17 matching landmark slots and always have
`z = 0`. Version 1 files (single person, no `engine`/`people`) are still imported.

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
js/analysis.js    metrics (kinesphere, activity by region, speed, space, shapes)
js/charts.js      dependency-free SVG/HTML charts
js/session.js     session model, compact JSON format, localStorage, JSON/CSV export
js/util.js        formatting and statistics helpers
```

Dependencies are loaded from the CDN at runtime: `@mediapipe/tasks-vision` 1.0.1 with the official
pose landmarker models, and, only when a MoveNet model is selected, `@tensorflow/tfjs` 4.22 with
`@tensorflow-models/pose-detection` 2.1.3 and the MoveNet weights from TF Hub / Kaggle.

### Hosting on GitHub Pages

The site is served from the root of the `main` branch (Settings → Pages → Deploy from a branch →
`main` / `/ (root)`). All paths are relative, so it also works from a project sub-path such as
`https://<org>.github.io/kinesphere/`.

## Contributing and license

Kinesphere is open source under the MIT licence (see [LICENSE](LICENSE)). The app links back to
this repository from its header and footer; issues and pull requests are welcome at
<https://github.com/UoA-eResearch/kinesphere>.
