# AI dance partner: Altlando and a music-free Bailando in the browser

Kinesphere's AI partner is **Altlando** (Jiazhi Zhou, *Interactive AI Dancer*, University of
Auckland, 2025): Bailando (Siyao et al., CVPR 2022) with the music condition removed from the GPT
(its attention input goes from `B × (3T) × C`, music + upper + lower body, to `B × (2T) × C`),
2048-pose vocabularies for each half of the body, and training on AIST++, FineDance and the CMU
motion capture database. The model code is in
[dance-ai-research-project](https://github.com/jzho987/dance-ai-research-project)
(`src/bailandopp/app/models_v1`, served by `server_v1.py`) and in the
[`feat/remove-music-condition`](https://github.com/jzho987/Bailando/tree/feat/remove-music-condition)
branch of jzho987/Bailando (`models/cross_cond_gpt2_nm.py`).

**What Kinesphere bundles.** Two exports of the same architecture, chosen with **Model** in the
panel:

* `models/altlando/` (the default): **Altlando's trained weights** (`vqvae_lb.pt`, `gpt_lb.pt`),
  2048 pose codes per half of the body, trained without music on AIST++, FineDance and CMU.
* `models/bailando/`: the **original Bailando weights** (released with the paper, trained on
  AIST++, 512 pose codes per half) with **Altlando's music removal re-implemented**: the weights are
  loaded into Altlando's music-free GPT (`CrossCondGPT2NoMusic`). Bailando's GPT input is three
  token streams, music then upper then lower body, each with its own block of positions (`pos_emb`
  has 3 × 29 rows) and the music tokens made from audio features by `cond_emb`. Without music the
  input is just the two body streams, so `cond_emb` is dropped and `pos_emb` keeps rows 29–57
  (upper body) and 58–86 (lower body); the attention masks go from a 3 × 3 to a 2 × 2 grid of
  causal blocks, and all other weights (embeddings, attention, MLPs, output heads, the whole
  VQ-VAE) carry over unchanged (`bailando_gpt_without_music` in `export_onnx.py`). Unlike
  Altlando, this GPT was never trained without music.

The main README's *How Altlando's weights differ from Bailando's* compares the two
(vocabulary, training data, training, context length, and the thesis's results).

The Bailando weights are © 2022 S-Lab under the S-Lab License 1.0 (`LICENSE-BAILANDO`, copied next
to the model by the export): non-commercial use and redistribution only. Altlando's weights are
the research team's, shared for this project (non-commercial research use).

Kinesphere runs the model in the browser, with no server:

| File | What it does |
| --- | --- |
| `ai-partner/export_onnx.py` | exports the trained model to three ONNX graphs plus a manifest and a reference |
| `js/altlando-worker.js` | runs them with ONNX Runtime Web 1.22 in a Web Worker: the GPT on WebGPU when available, the VQ-VAE on WebAssembly |
| `js/altlando.js` | promise API over the worker |
| `js/ai-partner.js` | converts your pose to SMPL joints and back, schedules and cross-fades replies |
| `models/altlando/`, `models/bailando/` | the bundled, exported models (Altlando is the default) |

## Exporting the model

Bailando (original weights, music removed; `models/bailando/`):

```sh
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r ai-partner/requirements-export.txt
git clone https://github.com/jzho987/dance-ai-research-project
# experiments.zip from Bailando's README (Google Drive), unzipped
python3 ai-partner/export_onnx.py \
    --altlando-app dance-ai-research-project/src/bailandopp/app \
    --bailando experiments --quantize --out models/bailando
```

It uses `experiments/sep_vqvae_root/ckpt/epoch_500.pt` and `experiments/cc_motion_gpt/ckpt/epoch_400.pt`
(the GPT before Bailando's music-specific actor-critic fine-tuning). Checkpoints are read with
`safe_load.py`, which refuses anything in the pickle but tensors, storages and plain containers.

Altlando's trained weights (`models/altlando/`):

```sh
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r ai-partner/requirements-export.txt
git clone https://github.com/jzho987/dance-ai-research-project
python3 ai-partner/export_onnx.py \
    --altlando-app dance-ai-research-project/src/bailandopp/app \
    --vqvae path/to/vqvae_lb.pt --gpt path/to/gpt_lb.pt \
    --quantize --out models/altlando
```

The script builds the networks from Altlando's own code and configs (`config/vqvae_config_lb.py`,
`config/gpt_config_lb.py`), loads the checkpoints and writes:

* `encoder.onnx`: 225 frames × 72 numbers → 28 upper-body and 28 lower-body pose codes;
* `gpt.onnx`: 29 codes of each → the most likely next code after every position (the attention is
  causal, so a shorter context is padded at the end and the code at its last real position used);
* `decoder.onnx`: 56 codes of each → 448 frames of joints relative to the root, with the root's
  velocity in its channels;
* `altlando.json`: sizes, file names (big files split into ≤ 40 MB parts), provenance;
* `reference.json`: a test input with PyTorch's codes and decoded joints for it.

Before writing, it runs the exported graphs with onnxruntime and checks they give exactly what
PyTorch gives (the same codes, decoded joints within 10⁻³). `--quantize` stores the convolution and
matrix weights as 8-bit integers with a per-channel scale (`DequantizeLinear`; the arithmetic stays
32-bit float, so every backend can run it): Bailando's 612 MB becomes 160 MB, and the decoded joints
move by 0.5 % of their size on average. For a quantised export, `reference.json` holds what the
int8 graphs give (computed with onnxruntime's integer fusions off, as the browser runs them).

`--random --tiny` exports the same architecture, small and with random weights, for testing:
`npm run build:test-model` does that into `test-results/altlando-tiny`, and the browser tests
check that the page reproduces PyTorch's output for `reference.json` exactly.

## Generation, as Altlando's server does it

1. The last 224 frames (3.7 s) of your movement, as 24 SMPL joints in metres with y up at 60 fps,
   the joints made relative to the root (the root itself stays absolute), padded to 225 frames
   (the server pads to the next multiple of 5).
2. Encode; keep the first 28 codes (`shift`).
3. 28 times, run the GPT on the last 29 codes and append the most likely next code.
4. Decode the 56 codes; integrate the root velocity from the origin; add the root back to the joints.

The first 224 frames of the result are the model's reading of your movement (the "mimicry"
phase); the partner dances the second 224 unless **Show the model's reading of my movement** is on.

## Data format, and training data

Altlando's input and output, and Kinesphere's **Export for Altlando** button on the dashboard,
are AIST++-style keypoint sequences: a JSON list of frames, each 72 numbers (x, y, z of the 24
SMPL joints in SMPL order: pelvis, left hip, right hip, spine1, left knee, right knee, spine2, left
ankle, right ankle, spine3, left foot, right foot, neck, left collar, right collar, head, left
shoulder, right shoulder, left elbow, right elbow, left wrist, right wrist, left hand, right hand),
metres, y up, 60 frames a second. Bailando's VQ-VAE splits them into the lower body
(`[0, 1, 2, 4, 5, 7, 8, 10, 11]`) and the upper body (the rest).

Training data from other sources (for example cleaned OptiTrack captures exported as CSV) has to
be brought into the same form: SMPL joint positions in metres, y up, at 60 fps, in arrays of
frames like the above (Altlando's own training data was produced by fitting SMPL to AIST++,
FineDance and CMU and taking its joints). Kinesphere's export does this for webcam recordings,
placing the joints the camera doesn't track (spine, collars, head, hands, feet) in SMPL's
proportions.

## Known limits

* The Bailando GPT was trained with music; it runs without it here (see above). Altlando's was
  trained without, which is why it is the default.
* Only Altlando's 29-code (3.7 s) version is bundled; the thesis's 45-code (6 s) version would need
  the export and the browser runtime to take a longer window (they assume 28 + 1 codes).
* Without WebGPU the CPU is slower than real time: about 12 s per reply on a 4-core virtual
  machine with one thread (GitHub Pages can't enable the multi-threaded WebAssembly build, which
  needs cross-origin isolation headers), so the partner pauses between replies.
* ONNX Runtime Web 1.22's WebGPU backend computes the VQ-VAE's `ScatterND` wrongly, so the encoder
  and decoder always run on WebAssembly (they run once per reply; the GPT is the heavy part). Its
  graph optimiser also fails on the VQ-VAE graphs when their shapes are dynamic, so the export uses
  fixed shapes.
* Bailando and Altlando output joint positions rather than SMPL rotations, so a mesh fitted to them
  can bend unnaturally; Kinesphere's overlay styles draw from joints and don't need one.
