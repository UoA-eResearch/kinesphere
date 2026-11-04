#!/usr/bin/env python3
"""
Export a music-free Bailando dance model to ONNX so Kinesphere can run it in the browser.

Altlando (Zhou 2025) is Bailando (Siyao et al., CVPR 2022) with the music condition removed from
the GPT, 2048-code vocabularies and more training data. Its code lives in dance-ai-research-project
(src/bailandopp/app/models_v1). This script builds the networks from that code and either

  --bailando   loads the ORIGINAL Bailando weights (AIST++, 512 codes; public, S-Lab License) into
               Altlando's music-free GPT: Jay's fix applied to the released model, without retraining
               (the music projection is dropped and the position table keeps its upper- and
               lower-body thirds; every other weight carries over unchanged), or
  --vqvae/--gpt  loads trained Altlando checkpoints,

and writes:

    altlando.json   manifest read by the browser (sizes, file names, provenance)
    encoder.onnx    pose [1, 225, 72] float32 -> codes_up, codes_down [1, 28] int64
    gpt.onnx        codes_up, codes_down [1, 29] int64 -> next_up, next_down [1, 29] int64
                    (the most likely next code after every position; the causal attention means
                    position i only sees positions <= i, so a shorter context is padded at the end)
    decoder.onnx    codes_up, codes_down [1, 56] int64 -> pose [1, 448, 72] float32
                    (joints relative to the root; the root channels hold its velocity)
    reference.json  a test input and PyTorch's output for it, to check the browser matches

Poses are AIST++-style SMPL-24 joint positions (metres, y up, 60 frames a second).

Usage
    pip install torch onnx onnxruntime numpy easydict
    git clone https://github.com/jzho987/dance-ai-research-project

    # what Kinesphere bundles: the original Bailando weights (experiments.zip from Bailando's README),
    # music removed, int8, split for GitHub
    python3 ai-partner/export_onnx.py \\
        --altlando-app dance-ai-research-project/src/bailandopp/app \\
        --bailando experiments --quantize --out models/bailando

    # trained Altlando weights
    python3 ai-partner/export_onnx.py \\
        --altlando-app dance-ai-research-project/src/bailandopp/app \\
        --vqvae weight/vqvae_lb.pt --gpt weight/gpt_lb.pt \\
        --out models/altlando

    # Altlando as Kinesphere bundles it in models/altlando (int8 weights, files split for GitHub)
    python3 ai-partner/export_onnx.py ... --quantize --out models/altlando

    # the same architecture with random weights, small, for testing the pipeline
    python3 ai-partner/export_onnx.py --altlando-app ... --random --tiny --out test-results/altlando-tiny
"""
import argparse
import copy
import hashlib
import json
import os
import sys

CODE_FRAMES = 8
SHIFT = 28  # codes of the dancer's movement the model starts from (224 frames, 3.7 s)


def load_configs(app_dir, tiny, bins=None):
    sys.path.insert(0, os.path.abspath(app_dir))
    from easydict import EasyDict
    import config.gpt_config_lb as gpt_cf
    import config.vqvae_config_lb as vq_cf

    vq = EasyDict({k: copy.deepcopy(getattr(vq_cf, k)) for k in dir(vq_cf) if not k.startswith('_')})
    vq.device = 'cpu'
    for half in (vq.up_half, vq.down_half):
        half['device'] = 'cpu'
    gpt = EasyDict({k: copy.deepcopy(getattr(gpt_cf, k)) for k in dir(gpt_cf) if not k.startswith('_')})
    if bins:  # the original Bailando: 512 codes for each half of the body
        for half in (vq.up_half, vq.down_half):
            half['l_bins'] = bins
        vq.l_bins = bins
        gpt.base.update(vocab_size_up=bins, vocab_size_down=bins)
        gpt.head.update(vocab_size=bins, vocab_size_up=bins, vocab_size_down=bins)
    if tiny:  # same structure (8 frames per code, 2048 codes, 29-code context), far fewer weights
        for half in (vq.up_half, vq.down_half):
            half.update(width=32, emb_width=32, depth=1)
        for part in (gpt.base, gpt.head):
            part.update(n_embd=64, n_head=4, n_layer=1, n_music=64, n_music_emb=64)
    return vq, gpt


def quantize_weights(path, min_elements=4096):
    """
    Weight-only int8: every Conv, ConvTranspose, MatMul and Gemm weight stored as int8 with a
    per-output-channel scale, behind a DequantizeLinear node. The arithmetic stays float32, so it
    runs on every ONNX Runtime backend (WebGPU, WebAssembly); the file shrinks about four times.
    """
    import numpy as np
    import onnx
    from onnx import numpy_helper, helper
    model = onnx.load(path)
    g = model.graph
    inits = {i.name: i for i in g.initializer}
    users = {}
    for node in g.node:
        for k, name in enumerate(node.input):
            users.setdefault(name, []).append((node, k))
    new_nodes, done = [], set()
    for node in g.node:
        if node.op_type not in ('Conv', 'ConvTranspose', 'MatMul', 'Gemm') or len(node.input) < 2:
            continue
        name = node.input[1]
        init = inits.get(name)
        if init is None or name in done or init.data_type != onnx.TensorProto.FLOAT:
            continue
        if any(n.op_type not in ('Conv', 'ConvTranspose', 'MatMul', 'Gemm') or k != 1 for n, k in users.get(name, [])):
            continue  # shared with something else (e.g. a codebook): leave it alone
        w = numpy_helper.to_array(init)
        if w.size < min_elements or w.ndim < 2:
            continue
        if node.op_type == 'Conv':
            axis = 0
        elif node.op_type == 'ConvTranspose':
            axis = 1
        elif node.op_type == 'Gemm':
            axis = 0 if any(a.name == 'transB' and a.i for a in node.attribute) else 1
        else:
            axis = w.ndim - 1
        red = tuple(i for i in range(w.ndim) if i != axis)
        scale = (np.abs(w).max(axis=red) / 127.0).astype(np.float32)
        scale[scale == 0] = 1e-8
        shape = [1] * w.ndim
        shape[axis] = -1
        q = np.clip(np.round(w / scale.reshape(shape)), -127, 127).astype(np.int8)
        g.initializer.remove(init)
        g.initializer.extend([numpy_helper.from_array(q, name + '_q'), numpy_helper.from_array(scale, name + '_scale'),
                              numpy_helper.from_array(np.zeros_like(scale, dtype=np.int8), name + '_zero')])
        new_nodes.append(helper.make_node('DequantizeLinear', [name + '_q', name + '_scale', name + '_zero'], [name], axis=axis, name=name + '_dq'))
        done.add(name)
    for n in reversed(new_nodes):
        g.node.insert(0, n)
    onnx.save(model, path)
    return len(done)


def strip_module(state):
    """Checkpoints were saved from nn.DataParallel: drop the 'module.' prefix."""
    return {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}


def bailando_gpt_without_music(state, block):
    """
    Jay's fix (Altlando), applied to the original Bailando GPT's weights: the music tokens are taken
    out of the sequence, so the attention input goes from B x (3T) x C (music, upper, lower body) to
    B x (2T) x C. The music projection (cond_emb) goes, and the position table keeps the positions of
    the upper- and lower-body tokens (rows block..2*block and 2*block..3*block); the attention, MLP and
    output weights are the same shapes and carry over unchanged.
    """
    import torch
    out = {k: v for k, v in state.items() if not k.startswith('gpt_base.cond_emb.')}
    pos = state['gpt_base.pos_emb']
    assert pos.shape[1] == 3 * block, f'expected {3 * block} positions (music, up, down), got {pos.shape[1]}'
    out['gpt_base.pos_emb'] = torch.cat([pos[:, block:2 * block], pos[:, 2 * block:3 * block]], dim=1).clone()
    return out


def build(args):
    import torch
    from safe_load import load_checkpoint
    torch.manual_seed(args.seed)
    vq_cfg, gpt_cfg = load_configs(args.altlando_app, args.tiny, bins=512 if args.bailando else None)
    from models_v1.sep_vqvae_root import SepVQVAER
    from models_v1.cross_cond_gpt2_nm import CrossCondGPT2NoMusic

    vqvae = SepVQVAER(vq_cfg)
    gpt = CrossCondGPT2NoMusic(gpt_cfg)
    if args.random:
        # random codebooks too, so different poses get different codes
        for m in vqvae.modules():
            if hasattr(m, 'k') and isinstance(getattr(m, 'k'), torch.Tensor) and m.k.dim() == 2:
                m.k.data.normal_(0, 1)
                if hasattr(m, 'init'):
                    m.init = True
    else:
        vq_path, gpt_path = (os.path.join(args.bailando, 'sep_vqvae_root', 'ckpt', 'epoch_500.pt'),
                             os.path.join(args.bailando, 'cc_motion_gpt', 'ckpt', 'epoch_400.pt')) if args.bailando else (args.vqvae, args.gpt)
        args.vqvae, args.gpt = vq_path, gpt_path
        vq_ck = load_checkpoint(vq_path)
        vq_state = strip_module(vq_ck['model'])
        missing, unexpected = vqvae.load_state_dict(vq_state, strict=False)
        assert not missing, f'VQ-VAE weights missing from {vq_path}: {missing[:5]}'
        gpt_ck = load_checkpoint(gpt_path)
        gpt_state = strip_module(gpt_ck['model'])
        if args.bailando:
            gpt_state = bailando_gpt_without_music(gpt_state, gpt_cfg.block_size)
        gpt.load_state_dict(gpt_state, strict=True)
        args.epochs = {'vqvae': vq_ck.get('epoch'), 'gpt': gpt_ck.get('epoch')}
    return vqvae.eval(), gpt.eval(), vq_cfg, gpt_cfg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--altlando-app', required=True, help='dance-ai-research-project/src/bailandopp/app')
    p.add_argument('--vqvae', help='VQ-VAE checkpoint (e.g. weight/vqvae_lb.pt)')
    p.add_argument('--gpt', help='GPT checkpoint (e.g. weight/gpt_lb.pt)')
    p.add_argument('--bailando', metavar='EXPERIMENTS', help="the original Bailando weights: the unzipped 'experiments' folder from Bailando's README")
    p.add_argument('--random', action='store_true', help='random weights instead of checkpoints (for testing)')
    p.add_argument('--tiny', action='store_true', help='with --random: a small version of the architecture')
    p.add_argument('--out', required=True, help='e.g. models/altlando or models/bailando')
    p.add_argument('--quantize', action='store_true', help='int8 weights (about a third of the size; what the site bundles)')
    p.add_argument('--part-mb', type=float, default=40, help='split files bigger than this into parts (GitHub refuses files over 100 MB)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--opset', type=int, default=17)
    args = p.parse_args(argv)
    if not args.random and not args.bailando and not (args.vqvae and args.gpt):
        p.error('give --bailando, --vqvae and --gpt, or --random')
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    import numpy as np
    import torch
    import torch.nn as nn

    vqvae, gpt, vq_cfg, gpt_cfg = build(args)
    block = gpt_cfg.block_size
    os.makedirs(args.out, exist_ok=True)

    class Encode(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m

        def forward(self, pose):
            up, down = self.m.encode(pose.clone())
            return up[0], down[0]

    class Step(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m

        def forward(self, up, down):
            (lu, ld), _ = self.m((up, down))
            return lu.argmax(-1), ld.argmax(-1)  # = topk(softmax, 1), as Altlando's sample()

    class Decode(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m

        def forward(self, up, down):
            return self.m.decode(([up], [down]))

    T = SHIFT * CODE_FRAMES + 1  # 225: Altlando's server pads 224 frames to the next multiple of 5
    pose = torch.zeros(1, T, 72)
    codes = torch.zeros(1, block, dtype=torch.long)
    n = 2 * SHIFT

    def export(module, inputs, path, names_in, names_out, axes):
        kw = dict(input_names=names_in, output_names=names_out, dynamic_axes=axes, opset_version=args.opset, do_constant_folding=True)
        module.eval()
        try:
            torch.onnx.export(module, inputs, path, dynamo=False, **kw)
        except TypeError:  # older PyTorch without the dynamo switch
            torch.onnx.export(module, inputs, path, **kw)
        module.eval()  # export restores the wrapper's mode, which would switch dropout on in the models
        vqvae.eval(); gpt.eval()

    with torch.no_grad():
        # fixed sizes (225 frames in, 56 codes out): the shape arithmetic folds away, which keeps ONNX
        # Runtime Web's graph optimiser happy and the graphs fast
        export(Encode(vqvae), (pose,), os.path.join(args.out, 'encoder.onnx'), ['pose'], ['codes_up', 'codes_down'], {})
        export(Step(gpt), (codes, codes), os.path.join(args.out, 'gpt.onnx'), ['codes_up', 'codes_down'], ['next_up', 'next_down'], {})
        export(Decode(vqvae), (torch.zeros(1, n, dtype=torch.long), torch.zeros(1, n, dtype=torch.long)), os.path.join(args.out, 'decoder.onnx'),
               ['codes_up', 'codes_down'], ['pose'], {})

    # ---- reference: PyTorch end to end on a test input (as Altlando's eval_single_epoch) ----------
    t = np.arange(SHIFT * CODE_FRAMES) / 60.0
    rng = np.random.default_rng(args.seed)
    base = rng.normal(0, 0.3, size=72)
    dance = np.stack([base[j] + 0.2 * np.sin(2 * np.pi * (0.5 + 0.05 * j) * t + j) for j in range(72)], axis=1)
    dance[:, 1] += 0.95
    payload = dance.astype(np.float32)
    x = payload.astype(np.float64).copy()
    root = x[:, :3].copy()
    x = x - np.tile(root, (1, 24)); x[:, :3] = root
    x = np.concatenate([x, x[-1:]], axis=0)  # pad 224 -> 225
    with torch.no_grad():
        up, down = vqvae.encode(torch.tensor(x, dtype=torch.float32).unsqueeze(0))
        xu, xd = up[0][:, :SHIFT], down[0][:, :SHIFT]
        for _ in range(SHIFT):
            (lu, ld), _ = gpt((xu[:, -block:], xd[:, -block:]))
            xu = torch.cat([xu, lu[:, -1].argmax(-1, keepdim=True)], 1)
            xd = torch.cat([xd, ld[:, -1].argmax(-1, keepdim=True)], 1)

    # ---- check onnxruntime agrees with PyTorch -------------------------------------------------
    import onnxruntime as ort
    enc = ort.InferenceSession(os.path.join(args.out, 'encoder.onnx'))
    cu, cd = enc.run(None, {'pose': x[None].astype(np.float32)})
    assert (cu[:, :SHIFT] == up[0][:, :SHIFT].numpy()).all() and (cd[:, :SHIFT] == down[0][:, :SHIFT].numpy()).all(), 'encoder codes differ from PyTorch'
    step = ort.InferenceSession(os.path.join(args.out, 'gpt.onnx'))
    seq_u, seq_d = [int(v) for v in cu[0, :SHIFT]], [int(v) for v in cd[0, :SHIFT]]
    for _ in range(SHIFT):
        wu, wd = seq_u[-block:], seq_d[-block:]
        k = len(wu)
        pad = [0] * (block - k)
        nu, nd = step.run(None, {'codes_up': np.array([wu + pad], dtype=np.int64), 'codes_down': np.array([wd + pad], dtype=np.int64)})
        seq_u.append(int(nu[0, k - 1])); seq_d.append(int(nd[0, k - 1]))
    if seq_u != xu[0].tolist() or seq_d != xd[0].tolist():
        bad = next(i for i in range(len(seq_u)) if seq_u[i] != xu[0, i] or seq_d[i] != xd[0, i])
        raise AssertionError(f'generated codes differ from PyTorch at code {bad}: {seq_u[bad]},{seq_d[bad]} vs {int(xu[0, bad])},{int(xd[0, bad])}')
    dec = ort.InferenceSession(os.path.join(args.out, 'decoder.onnx'))
    (pose_out,) = dec.run(None, {'codes_up': np.array([seq_u], dtype=np.int64), 'codes_down': np.array([seq_d], dtype=np.int64)})
    with torch.no_grad():
        ref_pose = vqvae.decode(([xu], [xd]))[0].numpy()
    err = float(np.abs(pose_out[0] - ref_pose).max())
    assert err < 1e-3, f'decoder differs from PyTorch by {err}'

    # ---- int8 weights, then split big files ------------------------------------------------------
    names = ('encoder', 'gpt', 'decoder')
    quant_note = None
    reference_from = 'PyTorch (= onnxruntime)'
    if args.quantize:
        for name in names:
            quantize_weights(os.path.join(args.out, f'{name}.onnx'))
        # as the browser runs them: weights dequantised, arithmetic in float32 (onnxruntime's CPU build
        # would otherwise fuse DequantizeLinear + MatMul into integer kernels that round differently)
        so = ort.SessionOptions()
        so.add_session_config_entry('session.disable_quant_qdq', '1')
        dec8 = ort.InferenceSession(os.path.join(args.out, 'decoder.onnx'), so)
        (p8,) = dec8.run(None, {'codes_up': np.array([seq_u], dtype=np.int64), 'codes_down': np.array([seq_d], dtype=np.int64)})
        quant_note = float(np.abs(p8[0] - ref_pose).mean() / max(1e-9, np.abs(ref_pose).mean()))
        # the reference the browser is checked against: what the int8 graphs themselves give
        enc8 = ort.InferenceSession(os.path.join(args.out, 'encoder.onnx'), so)
        step8 = ort.InferenceSession(os.path.join(args.out, 'gpt.onnx'), so)
        cu8, cd8 = enc8.run(None, {'pose': x[None].astype(np.float32)})
        seq_u, seq_d = [int(v) for v in cu8[0, :SHIFT]], [int(v) for v in cd8[0, :SHIFT]]
        for _ in range(SHIFT):
            wu, wd = seq_u[-block:], seq_d[-block:]
            k = len(wu)
            pad = [0] * (block - k)
            nu, nd = step8.run(None, {'codes_up': np.array([wu + pad], dtype=np.int64), 'codes_down': np.array([wd + pad], dtype=np.int64)})
            seq_u.append(int(nu[0, k - 1])); seq_d.append(int(nd[0, k - 1]))
        (ref_pose,) = dec8.run(None, {'codes_up': np.array([seq_u], dtype=np.int64), 'codes_down': np.array([seq_d], dtype=np.int64)})
        ref_pose = ref_pose[0]
        reference_from = 'onnxruntime (int8 weights, float32 arithmetic)'
    sizes, files, hashes = {}, {}, {}
    part = int(args.part_mb * 1e6)
    for name in names:
        path = os.path.join(args.out, f'{name}.onnx')
        for old in os.listdir(args.out):
            if old.startswith(f'{name}.onnx.part'):
                os.remove(os.path.join(args.out, old))
        sizes[name] = os.path.getsize(path)
        with open(path, 'rb') as f:
            hashes[name] = hashlib.sha256(f.read()).hexdigest()
        if sizes[name] > part:
            with open(path, 'rb') as f:
                data = f.read()
            parts = []
            for k in range(0, len(data), part):
                pn = f'{name}.onnx.part{len(parts)}'
                with open(os.path.join(args.out, pn), 'wb') as g:
                    g.write(data[k:k + part])
                parts.append(pn)
            os.remove(path)
            files[name] = parts
        else:
            files[name] = f'{name}.onnx'
    epochs = getattr(args, 'epochs', None)
    manifest = {
        'format': 'kinesphere-altlando-onnx', 'version': 1,
        'model': ('Altlando architecture (random weights, for testing)' if args.random
                  else 'Bailando (original AIST++ weights) with Altlando\'s music-free GPT' if args.bailando else 'Altlando (Zhou 2025)'),
        'licence': 'Bailando weights: S-Lab License 1.0, non-commercial use (see LICENSE-BAILANDO)' if args.bailando else None,
        'tiny': bool(args.tiny), 'precision': 'int8' if args.quantize else 'fp32',
        'fps': 60, 'joints': 24, 'codeFrames': CODE_FRAMES, 'blockSize': block, 'shift': SHIFT, 'length': SHIFT, 'padTo': 5,
        'vocabUp': gpt_cfg.base['vocab_size_up'], 'vocabDown': gpt_cfg.base['vocab_size_down'],
        'files': files, 'bytes': sizes, 'sha256': hashes,
        'source': 'https://github.com/jzho987/dance-ai-research-project (src/bailandopp/app/models_v1)'
                  + ('; weights https://github.com/lisiyao21/Bailando (experiments/sep_vqvae_root epoch 500, cc_motion_gpt epoch 400)' if args.bailando else ''),
        'checkpoints': None if args.random else (
            [os.path.relpath(args.vqvae, args.bailando), os.path.relpath(args.gpt, args.bailando)] if args.bailando
            else [os.path.basename(args.vqvae), os.path.basename(args.gpt)]),
        'epochs': epochs,
    }
    if args.bailando:  # the S-Lab License asks binary redistributions to carry it
        lic = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LICENSE-BAILANDO')
        with open(lic) as src, open(os.path.join(args.out, 'LICENSE-BAILANDO'), 'w') as dst:
            dst.write(src.read())
    with open(os.path.join(args.out, 'altlando.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(args.out, 'reference.json'), 'w') as f:
        json.dump({'from': reference_from, 'payload': np.round(payload, 6).tolist(), 'codesUp': seq_u, 'codesDown': seq_d,
                   'rawPose': np.round(ref_pose, 6).tolist()}, f)
    total = sum(sizes.values()) / 1e6
    if quant_note is not None:
        print(f'int8: the decoder output moves by {quant_note:.1%} of its size on average')
    print(f'wrote {args.out}: encoder {sizes["encoder"] / 1e6:.1f} MB, gpt {sizes["gpt"] / 1e6:.1f} MB, decoder {sizes["decoder"] / 1e6:.1f} MB ({total:.1f} MB); ONNX matches PyTorch (decoder within {err:.1e})')


if __name__ == '__main__':
    main()
