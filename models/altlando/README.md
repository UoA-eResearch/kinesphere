# The bundled Altlando model (the AI partner's default)

The 🤖 AI partner loads its default model from this folder: `altlando.json` plus `encoder.onnx`,
`gpt.onnx` and `decoder.onnx`, with big files split into `.part0`, `.part1`, … of at most 40 MB
(GitHub refuses files over 100 MB; the browser joins them again).

**What it is:** Altlando, from Jiazhi (Jay) Zhou, *Interactive AI Dancer* (University of Auckland,
2025): Bailando with the music condition removed from the GPT, 2048 pose codes for each half of the
body, trained on AIST++, FineDance and the CMU motion capture database. These are the research
team's trained weights, `vqvae_lb.pt` (VQ-VAE, epoch 500) and `gpt_lb.pt` (GPT, epoch 180), run
with the model code in [dance-ai-research-project](https://github.com/jzho987/dance-ai-research-project)
(`src/bailandopp/app/models_v1`). The weights are stored as 8-bit integers with a per-channel scale
(the arithmetic is 32-bit float); the decoded joints move by under 1 % of their size.

**Use:** shared for this research project; non-commercial research use. Ask the research team
before any other use. (Altlando was trained with Bailando's code, whose S-Lab License allows
non-commercial use; see `models/bailando/LICENSE-BAILANDO`.)

**Rebuilding it** from the checkpoints:

```sh
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r ai-partner/requirements-export.txt
git clone https://github.com/jzho987/dance-ai-research-project
python3 ai-partner/export_onnx.py \
    --altlando-app dance-ai-research-project/src/bailandopp/app \
    --vqvae path/to/vqvae_lb.pt --gpt path/to/gpt_lb.pt \
    --quantize --out models/altlando
```

`reference.json` is a test input with the codes and joints these files give for it; the browser
tests check the page reproduces them exactly.
