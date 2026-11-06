# Kinesphere

[![Tests](https://github.com/UoA-eResearch/kinesphere/actions/workflows/tests.yml/badge.svg)](https://github.com/UoA-eResearch/kinesphere/actions/workflows/tests.yml)

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
   **Origami**, **Origami pleats**, **Pencil sketch**, **Animator sketch** or **Hoops**.
   The **?** help panel has an *Overlay styles* gallery: every style with a small preview drawn
   by the style itself, how it looks, how it reacts to movement, its settings and the voice
   command that selects it; click a card to switch to that style.
   The pencil, animator and origami-pleats styles draw a fitted human body rather than lines
   between joints (`js/body-model.js`): each arm and leg is a tube along a smooth curve through
   its joints, from the shoulder to the fingertips and the hip to the toes, with an anatomical
   width profile (deltoid, bicep, elbow, forearm, wrist, hand; thigh, knee, calf, ankle, foot);
   the torso is a curved outline over the neck, shoulders, chest, waist and hips; the head is an
   egg shape with a narrower jaw on a neck. It works from MoveNet's 17 keypoints (hands and feet
   are inferred from the forearm and shin) and uses MediaPipe's hand and foot points when it has
   them. A spring-damped rig smooths the joints, with a little follow-through at the hands, feet
   and head, so the body moves fluidly and briefly lost joints don't make it flicker. Its springs
   are solved exactly each frame, so they stay steady however slowly the page draws. A soft-body deformer
   (also in `js/body-model.js`) then bends, stretches and squashes the fitted body according to
   how hard each part is accelerating (strongest in *Animator sketch*, gentler in *Pencil
   sketch* and *Origami pleats*). A **Body bend** slider under the controls (shown for those three
   styles, and remembered) scales the deformation from 0, a rigid body, through 1, the default, to
   3, a rubbery cartoon body; voice control can set it too (“more bend”, “less bend”, “bend off”,
   “bend 2”, “rubbery”, “normal bend”). *Origami* folds the body from paper: each limb is a kite
   creased along the bone, with its two facets shaded by a light from the upper left, a
   four-facet torso and a faceted head, casting a soft shadow. *Pencil sketch* draws a loose
   graphite figure (doubled limb contours, sketched joints, a hatched torso) whose lines redraw
   slightly about seven times a second, like hand-drawn animation; with the camera feed hidden
   (and in the replay) it is drawn on a sheet of paper. *Hoops* draws the body as rings: three
   around each arm and leg segment (wider bangles at the wrists and ankles), four around the
   torso, one round the neck and a halo above the head, each with a dim far half and a bright near
   half. A striped hula hoop circles the waist: it spins faster the more the hips move, wobbles
   round the waist as it spins, tilts with the hip line, and hangs on a springy link so it swings
   behind quick hip movements. *Animator sketch* is a loose construction drawing in the manner of
   hand-drawn character animation: each limb is a pair of contour lines that swell over the
   muscle, the torso carries a centre line, a chest line and a waist ellipse, shoulders get ovals,
   hands are mittens with a thumb and the head is a boxy sketch with an ear and a stroke of hair,
   all redrawn slightly every eighth of a second, with a soft shadow cast to one side. Fast moves
   *smear*, as in hand-drawn animation, by deforming the body itself with the force of the
   movement: each arm and leg is a soft chain of point masses on springs that are stiff at the
   shoulder or hip and loose towards the hand or foot, so when a limb is swung hard its outer
   part lags behind and the whole limb bends into a curve, stretches long and thins (its width
   shrinks as it stretches, keeping its volume), then overshoots slightly and snaps back over
   about a quarter of a second when the move stops. The torso squashes and stretches along the
   direction it accelerates, and the head and neck lag a little. The harder the move, the
   stronger the deformation; faint echoes of the bent limb and speed lines along it mark the
   fastest moments. Jumps (scrubbing
   a replay, a limb lost and found) don't count as movement. Dancers
   alternate between graphite and white chalk, and with the camera feed hidden they are drawn on
   warm toned paper. *Origami pleats* folds each dancer from cream paper shaded with amber in the
   folds: a pyramid-shaped head over a pointed face, a torso of fan pleats spreading from the
   neck to sharp shoulder flaps and the waist, long sleeves that taper to a point, wide creased
   trousers flaring at the hem, and folded waist flaps that hang on a spring and swing behind the
   movement. With the camera feed hidden it is shown in a soft studio with a wooden floor.
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
**?** button opens a fuller explanation. Both choices are remembered. Choosing more than one
person switches the model to **MoveNet MultiPose** (with a short notice), because its built-in
tracker is the best at keeping people apart; the other models tell people apart only by position
and can mix them up when dancers cross. You can still pick a MediaPipe model afterwards if you
need 3D depth for a group.

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

### Dance together online

Press **👥 Dance together** under the preview, enter your name and **Create invite link**, then
**Copy link** (or **Share…** on phones) and send it to up to five people. When someone opens the
link, the app asks for their name and **Start camera and join**; from then on everyone sees
everyone's avatar on their own screen, with a name tag, drawn in the overlay style *they* have
chosen. A badge in the corner of the preview shows how many are dancing, and the panel lists
them. **Leave** (a guest) or **End session** (the host) disconnects; when the host ends it,
everyone is told. Voice commands: “invite” and “leave session”.

* **Side by side.** Every dancer gets a lane across the picture. You stay where you are (your
  avatar still follows your camera), and the others are placed in the remaining lanes, so two
  dancers appear side by side. Each remote dancer is scaled so their torso matches yours and
  stands with their hips at your hip height, whatever their camera distance or picture shape;
  moving around their room moves them around their lane.
* **How it connects.** There is no Kinesphere server. Browsers connect directly to each other
  with WebRTC data channels via [PeerJS](https://peerjs.com/) 1.5.5 (loaded from the CDN only when
  you use this). PeerJS's free public signalling server introduces the browsers using the room
  code in the link; if a direct connection is impossible (strict firewalls), PeerJS's TURN relay
  servers carry the encrypted data. The host's browser passes each guest's poses on to the other
  guests, so the session lasts as long as the host stays. To use your own PeerJS server, add
  `?signal=wss://your.server[:port]/path` to the page address; invite links keep it.
* **What is sent.** Only pose landmarks (x, y and visibility of the 33 points for each tracked
  person, plus the 13 major joints in 3D when the model gives them), up to 20 times a second:
  about 10 KB/s per dancer. Never video or audio. WebRTC data is
  always encrypted end to end. As in any video call, participants' browsers learn each other's IP
  addresses, and the signalling server sees the room code and IP addresses. Anyone with the link
  can join while the session is open; room codes are 10 random characters.
* **Recording together.** Press Record during a session and the online dancers in view are recorded
  too, as extra people after you (up to 12 people in one recording), with their names. The
  dashboard then has a tab per dancer (“Ben (online)”), the replay shows them beside you scaled
  to your size as in the live view (the metrics overlay follows whoever is selected), and
  *Compare two dancers* can put you side by side with any of them. Their metrics come from their
  own camera's view: their poses are stored as their camera saw them (x rescaled to your
  recording's aspect ratio), with their 3D world landmarks when their model provides them, so
  kinesphere, body activity, level and speed are measured just as for you, and their floor path
  is their own room. Someone who joins after the recording started is not added to it. The
  session file lists every slot's name and whether they were online (`slots`).
* The dashboard and the game still work during a session; the others keep seeing you while you
  look at the dashboard.

### AI dance partner

Press **🤖 AI partner** under the preview and dance: an AI dancer watches the last 3.7 seconds of
your movement and dances a reply beside you, again and again as you go on. Nothing to install: the
models are bundled with the site and run **in your browser** with ONNX Runtime Web (on the GPU
through WebGPU when the browser has it, otherwise on the CPU with WebAssembly), in a Web Worker so
the page stays smooth; nothing is sent anywhere, and no server is needed.

Two models are bundled; **Model** in the panel switches between them (remembered for next time),
and the panel links to [Jay Zhou's thesis](https://hdl.handle.net/2292/74331) and the
[Bailando paper](https://arxiv.org/abs/2203.13055):

* **Altlando** (the default) is the model from Jiazhi (Jay) Zhou's thesis, *Interactive AI Dancer*
  (University of Auckland, 2025): Bailando retrained **without music**, so it answers movement
  alone, with 2048-pose vocabularies and more varied training data (AIST++ and FineDance dance plus
  the CMU motion capture database of everyday movement). These are the research team's trained
  weights (`vqvae_lb.pt`, VQ-VAE epoch 500, and `gpt_lb.pt`, GPT epoch 180).
* **Bailando** (Siyao et al., CVPR 2022) is the model Altlando grew from, with its **original
  released weights** (trained on AIST++ street and jazz dance) and **Altlando's music removal
  re-implemented** on top: Bailando's GPT reads three streams of tokens side by side (music, upper
  body, lower body: an attention input of B × 3T × C) and Altlando's only the two body streams
  (B × 2T × C), and the original weights fit that shape directly. The music projection is dropped,
  the position table keeps its upper- and lower-body rows, and every other weight carries over
  unchanged (`ai-partner/export_onnx.py --bailando`). It was never trained without music, though.

In our tests with synthetic dancers, **Altlando** answered steady movement (walking on the spot)
with a lively reply, reading it as many different poses, and answered a mostly still dancer with
calm, moderate movement. **Bailando** answered the still dancer with big, energetic movement, but
let its upper body settle into one held pose when replying to steady walking, likely the cost of
running a GPT without the music it was trained with. The comparison below gives the details.

* **How it works.** Your pose becomes the model's input format: the 24 joints of the SMPL body
  model in metres (y up), 60 frames a second, like the AIST++ dataset. The tracked joints come from
  MediaPipe's 3D world landmarks (or, with MoveNet, the 2D picture lifted onto a plane, scaled to a
  0.5 m torso); the spine, collarbones, head, hands and feet are placed from them in SMPL's
  proportions, and the pelvis gets a height from your legs and a position on the floor from where
  you are in the picture. The model's VQ-VAE turns every 8 frames into a pose code for the upper
  and one for the lower body; its GPT continues your 28 codes with 28 of its own (choosing the most
  likely code each time, as Bailando and Altlando do); the VQ-VAE decodes them back to joints. The partner
  then dances that reply, cross-fading from one reply into the next, and asks for the next reply
  before the current one runs out, so it keeps answering what you did most recently.
* **Show the model's reading of my movement** also plays the first half of each reply: the model
  re-reading your last 3.7 s through its pose vocabulary (the "mimicry" phase in the thesis).
* **On screen** the partner is drawn like an online dancer: in its own lane beside you, scaled to
  your size, in the overlay style you chose, with its model's name (*Altlando* or *Bailando*) over
  its head. It is **recorded** with you (as a person with that name, with 3D joints), so the
  dashboard analyses its dancing and can compare it with yours.
* **Show the model's reading of my movement** also plays the first half of each reply: the model
  re-reading your last 3.7 s through its pose vocabulary (the "mimicry" phase in the thesis).
* **Speed.** Each model has about 150–160 million weights (Altlando 185 MB, Bailando 160 MB,
  stored as 8-bit, downloaded the first time and kept in the browser). Nearly all the work is the
  GPT re-reading its 29-code window for every new code, 28 times per reply; with WebGPU (Chrome or
  Edge on a computer, recent Safari and Firefox) that runs on the graphics card. On the CPU alone
  (WebAssembly, one thread, as on GitHub Pages) a reply took 12–13 s on a 4-core virtual machine,
  slower than the 3.7 s it lasts, so the partner pauses between replies, and the panel says when
  it is slow.
* **Voice:** “AI partner” or “Dance with me” starts it, “Use Altlando” / “Use Bailando” switch
  model, “AI partner off” stops it.
* **The model files** are in `models/altlando/` and `models/bailando/` (see their READMEs and
  `ai-partner/README.md`). **Model files** in the panel can load another export instead, from a
  folder or from files on this computer; `?ai=<folder>` in the address does the same.
* **Licence: non-commercial research use.** Bailando's weights are released under the S-Lab
  License 1.0, which allows redistribution for non-commercial use only; the licence ships with them
  (`models/bailando/LICENSE-BAILANDO`). Altlando's weights were trained by the research team with
  Bailando's code on AIST++, FineDance and CMU data, and are shared here for this research project;
  ask the team before any other use. Kinesphere's own code stays MIT.
* **Export for Altlando** on the dashboard saves a dancer as the model's input format (a JSON list
  of 60 fps frames of 24 SMPL joints, as `main_v1.py --eval_all` in dance-ai-research-project
  reads), for testing the model offline or building training data.

#### How Altlando's weights differ from Bailando's

The two share one architecture and input format: a VQ-VAE (convolutions 512 wide and 3 deep, one
pose code per 8 frames, separate upper- and lower-body halves plus a root-velocity decoder) feeding
a GPT (6 base + 6 head layers, 12 heads, 768 dimensions, a 29-code window), on SMPL-24 joint
positions in metres at 60 frames a second. That is why one export script and one browser runtime
handle both. What differs is how the weights were trained (from the thesis, §6.2.3–6.3, pp. 43–47,
and the checkpoints and the configs stored in them):

| | Altlando (default) | Bailando (music removed by Kinesphere) |
| --- | --- | --- |
| **Music** | GPT trained *without* music: two token streams (upper, lower body) from the start | GPT trained *with* music: three token streams (music, upper, lower body), the music made from 438 audio features. Kinesphere removes the music stream afterwards, so the GPT attends over a two-stream input it never saw in training |
| **Pose vocabulary** | 2048 codes for each half of the body (512 were "not enough for contemporary dance") | 512 codes for each half |
| **Training data** | AIST++ + FineDance (about 7 hours, 22 genres) + the CMU motion-capture database (walking, jumping, climbing and other everyday movement), all fitted with SMPL and converted to the same joint format | AIST++ only: about 5 hours of 10 street and jazz genres (break, pop, lock, middle and LA hip-hop, house, waack, krump, street jazz, ballet jazz) |
| **Training** | VQ-VAE 500 epochs on the joints (learning rate 10⁻⁵; the stored config says batch 256, the thesis 128), then the root (locomotion) branch, then the GPT (learning rate 2 × 10⁻⁴; stored config batch 64, the thesis 128). The thesis describes about 300 GPT epochs in 24 hours on two H100 GPUs; the bundled checkpoint is **epoch 180** | VQ-VAE 500 epochs; GPT 400 epochs (batch 32, learning rate 3 × 10⁻⁴), followed in Bailando by actor-critic fine-tuning on a music beat-alignment reward. Kinesphere uses the GPT from *before* that music-specific stage (`cc_motion_gpt` epoch 400; VQ-VAE `sep_vqvae_root` epoch 500) |
| **Context** | 29 codes (3.7 s). The thesis also made a 45-code (6 s) version for the Unity tool; that one isn't bundled | 29 codes (3.7 s) |
| **Weights** | the research team's (`vqvae_lb.pt`, `gpt_lb.pt`) | public (Bailando's README), S-Lab License, non-commercial |

What the thesis found when comparing Altlando (29 codes) with the original Bailando, measuring how
well each reply correlates with the dancer's movement over 13 overlapping windows (Figures
6.1–6.3):

* **Reading the dancer** (the first windows, the "mimicry" phase): much the same overall, about 0.3.
* **The reply** stays related to the dancer's movement for longer with Altlando: around 0.1 out to
  the 12th window, where Bailando's falls to about zero by the 4th.
* **By kind of movement:** Altlando does better with slow, low, floor-level and still movement
  (clusters 3, 4, 5 and 8), which the thesis puts down to the CMU data, and worse with fast or stiff
  movement (clusters 2 and 7).
* **The 6 s version** sometimes fell back to CMU-like movement (it once started "swimming") and
  occasionally turned the torso a full 360°.

The thesis doesn't say whether the Bailando in that comparison was given music. Its continuous
and VAE prototypes (§6.2.1–6.2.2), which fed the encoder's output to the GPT without quantising
it, drifted back to a neutral pose and were set aside; Altlando keeps Bailando's quantised pose
codes.

The two models' codes can't be mixed: a code number means a different pose in each codebook, so
each folder holds a VQ-VAE and a GPT exported together.

As the thesis notes, Bailando and Altlando output joint positions rather than SMPL rotations (that
was the point of Bailando++), so a 3D mesh fitted to their output can bend unnaturally; Kinesphere
draws the partner with its own overlay styles, which don't need a mesh.

### Voice control (hands-free)

Press **🎤 Voice control** under the preview (or **🎤 Voice** in its corner, or <kbd>M</kbd>) and
change settings by speaking, without going back to the keyboard. Say **“help”** (or “what can I
say”) to show every command over the preview, and **“stop listening”** to turn it off. Each
phrase you say is shown briefly at the top of the preview with what it did, so you can see
whether you were understood. Examples:

| Say | Does |
| --- | --- |
| “Start recording” / “Record”, “Stop”, “Cancel” | start after the 3-second countdown, stop (the dashboard opens), cancel the countdown |
| “Neon”, “Sandman”, “Pencil”, “Animator”, “Origami”, “Pleats”, “Fuzzy blur”, “Hoops”, “Next style” | overlay style |
| “More bend”, “Less bend”, “Bend off”, “Bend 2”, “Rubbery”, “Normal bend” | how much the pencil, animator and origami-pleats bodies bend with movement (switches to Animator sketch if another style is showing) |
| “AI partner”, “Dance with me”, “AI partner off” | start or stop the AI dance partner |
| “Overlay off”, “Hide video”, “Mirror off”, “Flip”, “3D view”, “Fullscreen”, “Smoothing strong” | display options |
| “Blue sand”, “Gravity up / off / 2”, “Float”, “Longer trails”, “Decay 5 seconds”, “More momentum”, “Reset sand” | Sandman particles (switches to Sandman) |
| “Two people”, “Just me”, “Use MultiPose / Lite / Heavy / Holistic” | tracking settings |
| “Game”, “Play Sunrise / Neon / Rush”, “Stop song”, “Louder”, “Quieter”, “Mute” | dance game |
| “Sessions”, “Dashboard”, “Live view” | change page |

Phrases are matched loosely (“please”, “OK Kinesphere” and similar are ignored; the recogniser's
alternative transcripts are tried in turn), but ordinary conversation is not mistaken for a
command. Browsers only allow *entering* fullscreen from a click or key press, so “fullscreen”
may ask you to press <kbd>F</kbd>; leaving it works by voice. Voice control uses the browser's
Web Speech API: Chrome, Edge and Safari support it, Firefox does not (the button is hidden).
Listening restarts automatically when the browser pauses it, and stops with a message if the
microphone is blocked or recognition keeps failing.

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

Each panel has a short explanation under its title and notes on how it is measured. Untick
**Explanations** at the top of the dashboard to hide them all for a cleaner view; the choice is
remembered.

| Section | What it shows |
| --- | --- |
| **Replay** | The recorded skeleton(s), with a scrubber. Moving it also moves the cursor on the time charts. **Metrics overlay** draws the kinesphere spheres around the centre of the torso (the same spheres throughout; the one reached is highlighted), the standing shoulder and knee lines with the current level shaded, markers sized by each body part's current speed, and a readout of the current kinesphere class and limb, level, left/right and upper/lower split over the last second and most active body part. Useful for understanding the dashboard and for checking the classifiers. |
| **Person tabs** | When several people were tracked, one tab per person (coloured like the overlay) switches every metric below, and the overlay, to that dancer. |
| **Compare two dancers** | The last tab puts two dancers side by side, labelled **A** (blue) and **B** (orange): two people tracked in this session, or anyone from another session saved in the browser, chosen from two menus. It shows an *At a glance* table (duration, pose tracked, kinesphere, left/right, upper/lower, level, most active body part, mean and peak speed in m/s, floor area, distance travelled in total and per minute, distance from the camera) with a B − A column; the two kinesphere bars; body part activity on one shared colour scale, left/right, upper/lower and level figures for each; both speed curves on one chart; and both floor paths on one bird's eye view. Every metric is computed for each dancer separately, relative to their own body size, so people of different heights and recordings of different lengths compare fairly. |
| **Kinesphere** | Near, medium and far are how far from the body the limbs reached (stored files keep the keys `small`, `medium` and `large`). Share of time in a near, medium or far kinesphere, shown as three figures (the most typical pose in each class, drawn inside the three spheres) and a stacked bar. The spheres are concentric around the centre of the torso (midway between the mid-hip and the mid-shoulder) and sized from the dancer's own limb lengths, measured from the skeleton. With MediaPipe models the kinesphere is measured in **3D** from MediaPipe's world landmarks (metric positions in metres): limb lengths are each segment's median 3D length (also shown in cm) and reaches towards or away from the camera count fully. MoveNet sessions and recordings made before 3D support fall back to 2D screen positions, where limb lengths are the 90th percentile of each segment's on-screen length and reaches towards the camera are under-counted. The panel says which mode a session used. Spheres: **near** at the distance of the proximal joints (centre to shoulder), **medium** at the mid-limb joints when outstretched (+ upper arm), **far** at the hands when outstretched (+ forearm). The spheres are sized from the arms (legs measured from the torso centre are longer, so leg-sized spheres would put far out of the arms' reach) and stay fixed for the whole session. Each hand, and each lifted foot, is classed by the sphere it is nearest to, with the boundaries halfway between neighbouring spheres, and the frame takes the farthest of those; relaxed arms hanging at the sides therefore count as near. A leg only counts while lifted off the floor, since a standing leg is support rather than reach. Based on the multi-scale kinesphere of Kubota et al., *Emotion Estimation Using Laban Feature Values Based on a Multi-scale Kinesphere* (SUI '24).

The AI dance partner is Jiazhi Zhou's Altlando model, with its trained weights, and alternatively
Bailando's released weights with the music condition removed as in Altlando: J. Zhou, *Interactive AI Dancer:
Exploring the Possibilities of Real-Time Full-Body Interactive Dance AI Agent for Creative Support
and Public Installations*, University of Auckland, 2025, <https://hdl.handle.net/2292/74331>
(code: [dance-ai-research-project](https://github.com/jzho987/dance-ai-research-project) and the
`feat/remove-music-condition` branch of [jzho987/Bailando](https://github.com/jzho987/Bailando/tree/feat/remove-music-condition)),
with notes on the server code by Maxine Yang. It builds on:

* L. Siyao et al., "Bailando: 3D Dance Generation by Actor-Critic GPT with Choreographic Memory,"
  CVPR 2022, pp. 11040–11049, doi:[10.1109/CVPR52688.2022.01077](https://doi.org/10.1109/CVPR52688.2022.01077).
  Altlando uses the original Bailando rather than Bailando++, whose rotation-matrix output exists
  to fit an SMPL mesh, which an avatar drawn from joints doesn't need.
* L. Siyao et al., "Bailando++: 3D Dance GPT With Choreographic Memory," IEEE TPAMI 45(12),
  pp. 14192–14207, 2023, doi:[10.1109/TPAMI.2023.3319435](https://doi.org/10.1109/TPAMI.2023.3319435).
* M. Loper, N. Mahmood, J. Romero, G. Pons-Moll and M. J. Black, "SMPL: A Skinned Multi-Person
  Linear Model," ACM Trans. Graph. 34(6), article 248, 2015, doi:[10.1145/2816795.2818013](https://doi.org/10.1145/2816795.2818013);
  reprinted in *Seminal Graphics Papers*, vol. 2, article 88, 2023, doi:[10.1145/3596711.3596800](https://doi.org/10.1145/3596711.3596800).
  The partner's joints follow SMPL's 24-joint skeleton. "Show limb lengths and sphere sizes" lists every estimate. |
| **Body** | **Body part activity**: 13 markers (head, shoulders, elbows, hands, hips, knees, feet) sized and coloured by each part's share of all movement; *Show values* prints the percentages on the figure, and "Show body part data" lists share of movement, active time (share of the session the part moved faster than 0.5 torso lengths/s) and mean speed. **Left / right**: the figure split down the middle with each side's share of movement (head excluded). **Upper / lower**: the figure split at the waist with the share of movement in the upper body (shoulders, elbows, hands) and the lower body (hips, knees, feet), head excluded. **Level: where was the activity?**: share of time at a high, medium or low level. This is about position, not which limb moved: *high* when at least one limb is above standing shoulder height, *low* when a hand or the hips are below standing knee height, *medium* otherwise; the head is not counted, standing legs are ignored, and standing heights are measured from the session with the lowest visible joint taken as the floor. The floor is only measured while a foot (ankle, heel or toe) is inside the frame; when the feet are cropped out, the dancer is assumed to be standing and the floor is placed at the session's standing hip height below the hips, so *low* is then only detected from the hands (the overlay says "feet out of view"). |
| **Speed** | Mean speed of the 13 major joints over time (smoothed over 1 s). Numbers are hidden by default, showing the shape of the curve, the fastest moment and the average as a share of the peak; **Show values** adds the numbers in approximate metres per second (torso lengths per second converted with the dancer's torso length measured in 3D, or a typical 50 cm for MoveNet sessions). Speeds use the body scale of the moment, so stepping towards the camera does not inflate them. Tracking glitches are removed first: a Hampel filter drops any landmark that jumps more than 0.35 torso lengths away from the median of its neighbours within ±3 frames (smooth movement, however fast, stays near that median; a glitch does not), then joint speeds that are extreme outliers for that joint (above Q3 + 3 IQR on a log scale, or faster than 20 torso lengths/s, about 10 m/s) and frames where several joints jump at once are left out. The removed share is reported under the chart, and the same clean-up applies to body part activity. |
| **Path on the floor** | A bird's eye view of where the dancer moved in the room, with the camera at the top and its field of view shaded. Distance from the camera is estimated from the apparent size of the torso (pinhole camera model with an assumed 65° horizontal field of view, typical of webcams, and the torso length in metres from MediaPipe's world landmarks, or 50 cm for MoveNet); the sideways position comes from where the hips are across the picture. Positions are median-smoothed over ±0.5 s. Shaded 25 cm squares show where the dancer spent the most time. Stats: floor area used (side to side × front to back, 2nd–98th percentile), distance travelled (the smoothed path sampled every 0.25 s) and typical distance from the camera. These are estimates, roughly ±20%, and a camera with a very different field of view scales them. "Show floor position data" lists the path in metres. |
| **Top shapes** | The three most common body shapes, found by k-means clustering of the pose in every frame (experimental). |

## Data and privacy

* Nothing is uploaded anywhere unless you choose to dance together online or use voice control (below). The only network requests are for the pose libraries and model
  files (jsDelivr, Google's model storage and, for MoveNet, TF Hub / Kaggle) and the app's own
  static files.
* Downloaded model files (MediaPipe's WebAssembly runtime and `.task` models, MoveNet weights)
  are stored in the browser's Cache Storage under `kinesphere-models-v1`, between 5 and 45 MB
  depending on which models you have used. Clear them from the **?** help panel or by clearing
  the site's data.
* **Dancing together online** sends your pose landmarks (never video) to the people in the
  session, directly over encrypted WebRTC; see *Dance together online* above for what the PeerJS
  signalling and relay servers see.
* **Voice control** is off until you turn it on, and is not remembered between visits. It uses the
  browser's own speech recognition. Where the browser can transcribe on the device (recent Chrome
  with its on-device speech pack) the app uses that, and starts downloading the pack in the
  background when it is available. Otherwise your voice is sent to the browser maker's speech
  service (Google for Chrome, Apple for Safari, Microsoft for Edge) to be turned into text; the
  app asks before turning voice control on for the first time in that case. Only the
  recognised text is used, to pick a command; nothing is recorded or stored. Video never leaves
  your device either way.
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
js/body-model.js  fitted 2D human body (curved limbs, torso, head) and motion rig for the drawn styles
js/smoothing.js   One Euro display smoothing for the overlay and replay
js/viewer3d.js    three.js 3D figure viewer with WebXR (VR / AR) sessions
js/game.js        game mode: song menu, Web Audio music, coach and lane rendering, scoring loop
js/game-core.js   game logic without DOM or audio: moves, songs, choreography, pose matching
js/voice.js       voice control: spoken-phrase parser and Web Speech API listener
js/multiplayer.js dance together online: PeerJS (WebRTC) session, pose packing, invite links
js/ai-partner.js  AI partner: your pose <-> SMPL-24 joints, replies scheduled and cross-faded
js/altlando.js    the dance model in the browser (promise API over the worker)
js/altlando-worker.js  ONNX Runtime Web inference: encode, generate, decode (WebGPU / WebAssembly)
ai-partner/       export_onnx.py (Bailando / Altlando -> ONNX), safe checkpoint loading, the test-model script
models/altlando/  the default AI partner model: Altlando, int8 ONNX
models/bailando/  the alternative: Bailando's released weights with the music removed, int8 ONNX (S-Lab License)
js/analysis.js    metrics (kinesphere, body part activity, level, speed, path, shapes)
js/charts.js      dependency-free SVG/HTML charts
js/session.js     session model, compact JSON format, IndexedDB storage, JSON/CSV export
js/util.js        formatting and statistics helpers
tests/unit/       Node tests of the modules without a browser
tests/browser/    Playwright tests of the app in Chromium
tests/helpers/    synthetic sessions with known ground truth, and the browser harness
```

Dependencies are loaded from the CDN at runtime: `@mediapipe/tasks-vision` 1.0.1 with the official
pose landmarker models; only when a MoveNet model is selected, `@tensorflow/tfjs` 4.22 with
`@tensorflow-models/pose-detection` 2.1.3 and the MoveNet weights from TF Hub / Kaggle; and only
when a 3D view is opened, `three` 0.170 (via the import map in `index.html`); and only when the AI
partner starts, `onnxruntime-web` 1.22.0.

### Tests

The tests run on every push and pull request in GitHub Actions (`.github/workflows/tests.yml`).
Running them locally needs Node 22 (only for the tests; the app has no build step):

```sh
npm ci
npx playwright install chromium
npm test                # everything (about two minutes)
npm run test:unit       # just the Node tests (a couple of seconds)
npm run test:browser    # just the browser tests
```

The AI partner tests need a small model: the real Altlando architecture with random weights,
exported to ONNX. Build it once with Python and PyTorch (CPU is enough):

```sh
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r ai-partner/requirements-export.txt
npm run build:test-model   # clones dance-ai-research-project, writes test-results/altlando-tiny
```

- **Unit tests** (`tests/unit/`) check the analysis against synthetic sessions with known ground
  truth: the kinesphere class (near / medium / far) of each of ten phases, measured in 3D and in
  2D, the space level (high / medium / low), the floor path recovered from a dancer walking a
  known square, and that tracking glitches don't create speed spikes. They also cover the voice
  command parser, the multiplayer wire format and invite links, game-mode scoring for every move
  and song, the fitted body model, and the soft-body deformer (a still body is not deformed; a
  fast swing bends the moving limb, scaled by strength; it springs back; teleports don't smear).
- **Browser tests** (`tests/browser/`) serve the working tree to headless Chromium. They check
  that every overlay style draws over the camera feed and on its own background, under a
  per-frame time limit (`KINESPHERE_MAX_FRAME_MS`, default 250 ms), and that every style copes
  with the poses a camera gives when someone is partly out of frame or the tracking glitches
  (walking up to the camera, flickering visibility, a collapsed torso, all joints on one point,
  a joint jumping away) at 60, 30 and 15 frames a second; and the help panel and style
  gallery (including at phone width). On the dashboard they check the kinesphere figures, value
  toggles, hiding the explanations, the replay overlay, exports, the floor plot, IndexedDB storage
  (including a 15-minute session) and comparing dancers. On the live view, with Chromium's fake
  camera and a scripted dancer in place of the pose model's output, they check the styles, the
  "Body partially out of frame" hint, recording and voice control (with a fake speech
  recogniser). Multiplayer is tested with a host and two guests in separate browser contexts and
  a local PeerJS signalling server. The AI partner tests check that the model in the browser gives
  exactly what PyTorch gives for the same input (the same pose codes and decoded joints), and that
  each bundled model (Altlando and Bailando) loads and gives exactly what onnxruntime gives, and
  replies to a dancer with human bone lengths; that Altlando is the default and Bailando can be
  chosen and is remembered; then
  they dance with the partner (including a dancer drifting out of frame on a slowed-down CPU, in
  the fitted-body styles), record it, stop and restart it, load model files chosen from disk,
  report a folder without a model and export a dancer for Altlando. Unit tests cover the pose conversions
  to and from SMPL joints (bone lengths, left and right, y up), resampling to 60 fps and the
  reply scheduling (when it asks, cross-fades, errors and back-off).

The live and multiplayer tests load MediaPipe and PeerJS from the CDN. Offline or behind a
restrictive proxy, set `KINESPHERE_CDN_CACHE` to a directory of copies laid out as
`<dir>/<host>/<path>` (e.g. `<dir>/cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/...`).

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
