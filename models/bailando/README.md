# The bundled Bailando model (the AI partner's alternative)

When **Bailando** is chosen as the 🤖 AI partner's model, it loads from this folder: `altlando.json` plus `encoder.onnx`,
`gpt.onnx` and `decoder.onnx`, with big files split into `.part0`, `.part1`, … of at most 40 MB
(GitHub refuses files over 100 MB; the browser joins them again).

**What it is:** Bailando (Siyao et al., CVPR 2022) with its original released weights (trained on
AIST++: the VQ-VAE `sep_vqvae_root` epoch 500 and the GPT `cc_motion_gpt` epoch 400 from the
`experiments.zip` linked in [Bailando's README](https://github.com/lisiyao21/Bailando)), with the
music condition removed from the GPT as in Jiazhi Zhou's Altlando: the music projection is dropped and
the position table keeps its upper- and lower-body rows; every other weight is unchanged. The
weights are stored as 8-bit integers with a per-channel scale (the arithmetic is 32-bit float).

**Licence:** the weights are © 2022 S-Lab, under the S-Lab License 1.0 (`LICENSE-BAILANDO`):
non-commercial use and redistribution only.

**Rebuilding it:**

```sh
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r ai-partner/requirements-export.txt
git clone https://github.com/jzho987/dance-ai-research-project
# download experiments.zip from Bailando's README and unzip it
python3 ai-partner/export_onnx.py \
    --altlando-app dance-ai-research-project/src/bailandopp/app \
    --bailando experiments --quantize --out models/bailando
```

Altlando, the default model, is in `models/altlando/`
(see its README).

`reference.json` is a test input with the codes and joints these files give for it; the browser
tests check the page reproduces them exactly.
