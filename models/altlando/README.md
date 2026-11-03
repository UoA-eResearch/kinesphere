# Altlando model files (bundled)

The AI dance partner loads its model from this folder: `altlando.json` plus `encoder.onnx`,
`gpt.onnx` and `decoder.onnx` (big files are split into `.part0`, `.part1`, … of at most 40 MB,
because GitHub refuses files over 100 MB; the browser joins them again).

They are exported from the trained Altlando weights (`vqvae_lb.pt` and `gpt_lb_l_240.pt` in
dance-ai-research-project's `src/bailandopp/app/weight/`), which are not public yet. With them:

```sh
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r ai-partner/requirements-export.txt
git clone https://github.com/jzho987/dance-ai-research-project
python3 ai-partner/export_onnx.py \
    --altlando-app dance-ai-research-project/src/bailandopp/app \
    --vqvae path/to/vqvae_lb.pt --gpt path/to/gpt_lb_l_240.pt \
    --quantize --out models/altlando
```

Then commit the files in this folder: once they are on GitHub Pages, the 🤖 AI partner button works
for everyone with nothing to install. See `ai-partner/README.md`.
