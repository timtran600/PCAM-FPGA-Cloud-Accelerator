"""
export.py — turn a Brevitas QAT checkpoint into integer artifacts.

This is the handoff from the GPU box to the FPGA. Everything downstream —
golden.py, the HLS testbenches, the XRT host — reads what this writes, so
a mistake here is invisible until hardware disagrees with software and
you cannot tell which is wrong.

Writes to artifacts/weights/:
    params.npz   int8 weights, int32 biases, int32 M0 and shift per layer
    weights.bin  the same arrays, flat, in the order the kernel streams
    params.h     dimensions and array declarations for the HLS build

All three come from one set of arrays so they cannot drift apart.

-------------------------------------------------------------------------
The arithmetic
-------------------------------------------------------------------------
Brevitas stores a float scale per tensor. Hardware cannot use a float, so
each layer's real-valued requantization multiplier

    M = s_in * s_w / s_out

is rewritten as an integer pair, M = M0 * 2^-shift, with M0 chosen in
[2^30, 2^31) so that all 31 available bits of precision are used. The
kernel then computes (acc * M0 + (1 << (shift-1))) >> shift, which is a
multiply and an arithmetic shift — no division, no float.

Bias is folded into the accumulator domain:

    bias_int[co] = round( bias_float[co] / (s_in * s_w[co]) )

so it is added directly to the int32 accumulator before requantization,
exactly as golden.py does. The BatchNorm affine term is already inside
bias_float because qat.py folded BN before quantizing.

GAP: the float model computes a mean over 6x6. Hardware sums and never
divides, so the 1/36 is folded into the GAP requantization multiplier:

    M_gap = s_in / (36 * s_out)

FC: emits int32 logits with no requantization. The host thresholds
logit[1] - logit[0], so no output scale exists.

-------------------------------------------------------------------------
What --verify does, and what it does not prove
-------------------------------------------------------------------------
--verify runs golden.py and the Brevitas model on the same tiles and
compares predictions. They will not agree bit-for-bit: Brevitas simulates
quantization with float arithmetic, so its intermediate values carry
rounding this integer model does not have. Agreement on predictions plus
matching accuracy is the strongest check available at this boundary.

True bit-exactness is a property of golden.py against the HLS kernel, and
that is checked in Part 2 with these same vectors.

    python phase1_software/export.py --ckpt models/qat_int8_best.pt --verify 512
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import PCamH5                                  # noqa: E402
from qat import QuantPCamCNN                             # noqa: E402
import golden                                            # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GAP_DIVISOR = 36          # 6x6 spatial, folded into M_gap
M0_MIN = 1 << 30          # keep M0 in [2^30, 2^31) for full precision
M0_MAX = (1 << 31) - 1


# -------------------------------------------------------------------------
# Reading quantities out of Brevitas
# -------------------------------------------------------------------------

def act_scale(module) -> float:
    """Scalar activation scale from a Quant{ReLU,Identity}."""
    s = module.act_quant.scale()
    return float(np.asarray(s.detach().cpu()).reshape(-1)[0])


def weight_int_scale(qlayer):
    """(int8 weights, per-output-channel float scales).

    Brevitas gives a QuantTensor whose .value is the dequantized float
    and .scale the per-channel step, so value/scale recovers the integer
    grid. Rounded, then asserted to be in int8 range rather than clipped:
    a weight outside [-127,127] means the quantizer is not configured the
    way this file assumes, and silently clipping would hide that.
    """
    qt = qlayer.quant_weight()
    w = qt.value.detach().cpu()
    s = qt.scale.detach().cpu()
    w_int = torch.round(w / s)
    lo, hi = int(w_int.min()), int(w_int.max())
    assert -128 <= lo and hi <= 127, f"weights outside int8 range: [{lo},{hi}]"
    scales = s.reshape(s.shape[0] if s.numel() > 1 else 1, -1)[:, 0]
    if scales.numel() == 1:
        scales = scales.repeat(w.shape[0])
    return w_int.numpy().astype(np.int8), scales.numpy().astype(np.float64)


def quantize_multiplier(m: np.ndarray):
    """Real multiplier M -> (M0 int32, shift int32), per channel.

    Picks the smallest shift that puts M0 in [2^30, 2^31). Returns both
    as int32 so the arrays can be written straight to params.npz.
    """
    m = np.asarray(m, dtype=np.float64)
    assert np.all(m > 0), "non-positive requantization multiplier"
    m0 = np.empty(m.shape, dtype=np.int64)
    shift = np.empty(m.shape, dtype=np.int64)

    for i, mi in enumerate(m.reshape(-1)):
        s = 0
        while mi * (2.0 ** s) < M0_MIN:
            s += 1
            assert s < 62, f"multiplier {mi} too small to represent"
        cand = int(round(mi * (2.0 ** s)))
        if cand > M0_MAX:                       # rounding pushed it over
            cand = int(round(mi * (2.0 ** (s - 1))))
            s -= 1
        m0.reshape(-1)[i] = cand
        shift.reshape(-1)[i] = s

    assert np.all(m0 >= M0_MIN) and np.all(m0 <= M0_MAX), "M0 out of range"
    assert np.all(shift >= 1), "shift must be >= 1 for round-half-up"
    return m0.astype(np.int32), shift.astype(np.int32)


def bias_to_int(bias_f: np.ndarray, s_in: float, s_w: np.ndarray):
    """bias_float -> int32 in the accumulator domain."""
    b = np.round(bias_f / (s_in * s_w))
    assert np.all(np.abs(b) < 2**30), "bias too large for int32 accumulator"
    return b.astype(np.int32)


# -------------------------------------------------------------------------
# Export
# -------------------------------------------------------------------------

def export(qmodel: QuantPCamCNN, out_dir: str, verbose=True):
    params = {}
    rows = []

    s_in = act_scale(qmodel.input_quant)
    expected = 1.0 / 256.0
    assert abs(s_in - expected) < 1e-9, (
        f"input scale is {s_in}, expected {expected} (2^-8). The kernel "
        "cannot consume raw tile bytes; resolve before exporting.")

    for i, block in enumerate(qmodel.features, start=1):
        name = f"conv{i}"
        w_int, s_w = weight_int_scale(block.conv)
        s_out = act_scale(block.relu)
        b_int = bias_to_int(block.conv.bias.detach().cpu().numpy().astype(np.float64),
                            s_in, s_w)
        m0, shift = quantize_multiplier(s_in * s_w / s_out)

        params[f"{name}_w"] = w_int
        params[f"{name}_bias"] = b_int
        params[f"{name}_m0"] = m0
        params[f"{name}_shift"] = shift
        rows.append((name, w_int.shape, s_in, float(s_w.mean()), s_out,
                     int(shift.min()), int(shift.max())))
        s_in = s_out          # this layer's output feeds the next

    # GAP: sum over 6x6, 1/36 folded into the multiplier so hardware
    # never divides.
    s_gap_out = act_scale(qmodel.gap_quant)
    m0, shift = quantize_multiplier(np.full(128, s_in / (GAP_DIVISOR * s_gap_out)))
    params["gap_bias"] = np.zeros(128, dtype=np.int32)
    params["gap_m0"] = m0
    params["gap_shift"] = shift
    rows.append(("gap", (128,), s_in, 1.0 / GAP_DIVISOR, s_gap_out,
                 int(shift.min()), int(shift.max())))

    # FC: int32 logits out, no requantization.
    fc_w, fc_s = weight_int_scale(qmodel.fc)
    params["fc_w"] = fc_w
    params["fc_bias"] = bias_to_int(
        qmodel.fc.bias.detach().cpu().numpy().astype(np.float64), s_gap_out, fc_s)
    rows.append(("fc", fc_w.shape, s_gap_out, float(fc_s.mean()), None, None, None))

    golden.check_param_dtypes(params)

    wdir = os.path.join(out_dir, "weights")
    os.makedirs(wdir, exist_ok=True)
    np.savez(os.path.join(wdir, "params.npz"), **params)

    if verbose:
        print(f"{'layer':<8} {'shape':<20} {'s_in':>10} {'s_w':>10} "
              f"{'s_out':>10} {'shift':>9}")
        print("-" * 72)
        for n, shp, si, sw, so, lo, hi in rows:
            so_s = f"{so:.3e}" if so else "    (int32)"
            sh = f"{lo}-{hi}" if lo is not None else "   -"
            print(f"{n:<8} {str(shp):<20} {si:>10.3e} {sw:>10.3e} "
                  f"{so_s:>10} {sh:>9}")
        total = sum(v.nbytes for k, v in params.items() if k.endswith("_w"))
        print(f"\nint8 weights: {total:,} B ({total/1024:.1f} KB) — fits on-chip")
        print(f"wrote {os.path.join(wdir, 'params.npz')}")

    write_bin(params, wdir, verbose)
    write_header(params, wdir, verbose)
    return params


def write_bin(params, wdir, verbose=True):
    """Flat binary in the order the kernel loads: per layer, weights then
    bias then M0 then shift. The HLS testbench reads this with a single
    fread per array, so the order here IS the contract."""
    path = os.path.join(wdir, "weights.bin")
    order = []
    with open(path, "wb") as f:
        for name in golden.CONV_STAGES + ["gap", "fc"]:
            for suffix, dt in (("_w", np.int8), ("_bias", np.int32),
                               ("_m0", np.int32), ("_shift", np.int32)):
                key = name + suffix
                if key not in params:
                    continue
                arr = np.ascontiguousarray(params[key], dtype=dt)
                f.write(arr.tobytes())
                order.append((key, arr.dtype.name, arr.size, arr.nbytes))
    if verbose:
        print(f"wrote {path} ({os.path.getsize(path):,} B)")
    return order


def write_header(params, wdir, verbose=True):
    """params.h — dimensions only. The kernel reads values from
    weights.bin; duplicating them as C initializers would create a second
    source of truth."""
    path = os.path.join(wdir, "params.h")
    lines = [
        "// Generated by export.py. Do not edit.",
        "// Values live in weights.bin; this file declares sizes only.",
        "#ifndef PCAM_PARAMS_H",
        "#define PCAM_PARAMS_H",
        "",
        "#define TILE_H 96",
        "#define TILE_W 96",
        "#define TILE_C 3",
        "#define TILE_BYTES (TILE_C * TILE_H * TILE_W)",
        f"#define GAP_DIVISOR {GAP_DIVISOR}",
        "#define ACT_MIN 0",
        "#define ACT_MAX 255",
        "",
    ]
    for name in golden.CONV_STAGES:
        w = params[f"{name}_w"]
        u = name.upper()
        lines += [f"#define {u}_OUT {w.shape[0]}",
                  f"#define {u}_IN  {w.shape[1]}",
                  f"#define {u}_K   {w.shape[2]}"]
    lines += ["", f"#define GAP_CH {params['gap_m0'].size}",
              f"#define FC_IN  {params['fc_w'].shape[1]}",
              f"#define FC_OUT {params['fc_w'].shape[0]}",
              "", "#endif  // PCAM_PARAMS_H", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    if verbose:
        print(f"wrote {path}")


# -------------------------------------------------------------------------
# Verification
# -------------------------------------------------------------------------

def verify(qmodel, params, data_root, n, device):
    """Compare golden (integer) against Brevitas (float-simulated) on the
    same tiles. See the docstring on what this does and does not prove."""
    from data import to_float01

    ds = PCamH5(data_root, "test")
    tiles = np.stack([ds[i][0].numpy() for i in range(n)])
    labels = np.array([ds[i][1] for i in range(n)], dtype=np.int64)

    qmodel.eval()
    with torch.no_grad():
        x = torch.from_numpy(tiles).to(device)
        bf = qmodel(to_float01(x)).cpu().numpy()
    bpred = bf.argmax(1)

    gl = golden.forward_batch(tiles, params)
    gpred = (golden.tumor_score(gl) > 0).astype(np.int64)

    agree = (bpred == gpred).mean()
    print(f"\nverification on {n} test tiles")
    print(f"  brevitas accuracy : {(bpred == labels).mean():.4f}")
    print(f"  golden   accuracy : {(gpred == labels).mean():.4f}")
    print(f"  prediction agreement: {agree:.4f} "
          f"({int((bpred != gpred).sum())} disagreements)")

    # Correlation of the decision variable is the more sensitive check:
    # predictions can agree while the underlying scores drift.
    bscore = bf[:, 1] - bf[:, 0]
    gscore = golden.tumor_score(gl).astype(np.float64)
    r = np.corrcoef(bscore, gscore)[0, 1]
    print(f"  score correlation : {r:.6f}")

    if agree < 0.99 or r < 0.999:
        print("\n  Disagreement is above what float-vs-integer rounding "
              "explains. Check scale extraction and the bias fold before "
              "using these artifacts.")
    else:
        print("\n  Consistent with rounding differences alone.")
    return agree, r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/qat_int8_best.pt")
    p.add_argument("--out", default="artifacts")
    p.add_argument("--root", default="./data")
    p.add_argument("--verify", type=int, default=0,
                   help="verify against Brevitas on N test tiles")
    a = p.parse_args()

    ckpt_path = a.ckpt if os.path.isabs(a.ckpt) else os.path.join(REPO_ROOT, a.ckpt)
    out_dir = a.out if os.path.isabs(a.out) else os.path.join(REPO_ROOT, a.out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    qmodel = QuantPCamCNN()
    qmodel.load_state_dict(ck["model"])
    qmodel = qmodel.to(device).eval()
    print(f"checkpoint: {a.ckpt}  (epoch {ck['epoch']}, val AUC {ck['val_auc']:.4f})\n")

    params = export(qmodel, out_dir)

    if a.verify:
        verify(qmodel, params, a.root, a.verify, device)


if __name__ == "__main__":
    main()