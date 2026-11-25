"""
golden.py — pure-integer reference implementation of the quantized PCamCNN.

This file defines correctness for the whole project. The HLS kernel must
match it bit-for-bit on all 32,768 test tiles; when hardware and golden
disagree, the kernel is wrong, not this.

HARD RULE: no floating point below this docstring. No float dtypes, no
true division, no np.mean. Every arithmetic operation here must have a
direct integer equivalent in HLS C++. If a line cannot be written with
int32 adds, multiplies, shifts and comparisons, it does not belong here.

-------------------------------------------------------------------------
Quantization scheme
-------------------------------------------------------------------------
Weights      int8, symmetric, per-output-channel. zero-point 0.
Activations  uint8, zero-point 0. Legal because every activation this
             model stores is post-ReLU and therefore non-negative. No
             zero-point arithmetic anywhere, which removes a correction
             term from every HLS accumulator.
Input        the raw tile byte. data.py scales by 1/256, so s_in = 2^-8
             exactly and the kernel consumes HBM bytes with no conversion.

Per layer:

    acc[co] = sum_ci sum_kh sum_kw  w[co,ci,kh,kw] * a[ci,h+kh,w+kw]
            + bias[co]                                          (int32)

    out     = clamp( (acc * M0[co] + round_add) >> shift[co], 0, 255 )

M0 and shift encode the real-valued requantization multiplier
M = s_w[co] * s_in / s_out as a fixed-point integer, per output channel.
The BatchNorm affine term and the conv bias are folded into bias[co] at
export time; there is no separate BN stage here or in hardware.

round_add = 1 << (shift-1) gives round-half-up. The HLS kernel must use
the same, and the testbench checks it: half-even rounding here and
half-up there is a one-LSB mismatch on a few tiles out of 32,768, which
is the single most annoying bug class in this project.

ReLU is the lower clamp bound. It is not a separate operation.

-------------------------------------------------------------------------
Layer sequence
-------------------------------------------------------------------------
    conv1 3->16   96x96   requant -> uint8   maxpool 2x2 -> 48x48
    conv2 16->32  48x48   requant -> uint8   maxpool 2x2 -> 24x24
    conv3 32->64  24x24   requant -> uint8   maxpool 2x2 -> 12x12
    conv4 64->64  12x12   requant -> uint8   maxpool 2x2 -> 6x6
    conv5 64->128  6x6    requant -> uint8
    gap           sum over 6x6 (no divide; 1/36 folded into the requant
                  multiplier) -> requant -> uint8[128]
    fc    128->2  int32 logits, NOT requantized. The host thresholds
                  logit[1] - logit[0].

Padding is zero. Since the activation zero-point is 0, a zero pad is a
literal integer zero and needs no correction term.

-------------------------------------------------------------------------
Artifact format (written by export.py, read here and by the HLS testbench)
-------------------------------------------------------------------------
artifacts/weights/params.npz, with for each layer L in
{conv1..conv5, gap, fc}:

    L_w       int8    [out, in, 3, 3]   (conv)  /  [2, 128] (fc)
    L_bias    int32   [out]
    L_m0      int32   [out]             (absent for fc)
    L_shift   int32   [out]             (absent for fc)

export.py also emits the same numbers as weights.bin and params.h for
the kernel. Both come from one array set so they cannot diverge.

    python phase1_software/golden.py --self-test
    python phase1_software/golden.py --artifacts ./artifacts --split test
    python phase1_software/golden.py --artifacts ./artifacts --split test --dump-layers 8
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ACT_MIN, ACT_MAX = 0, 255          # uint8 activations, zero-point 0
CONV_STAGES = ["conv1", "conv2", "conv3", "conv4", "conv5"]
POOL_AFTER = {"conv1", "conv2", "conv3", "conv4"}


# -------------------------------------------------------------------------
# Primitive integer ops. Each maps to a few lines of HLS C++.
# -------------------------------------------------------------------------

def requantize(acc, m0, shift):
    """int32 accumulator -> uint8 activation, per output channel.

    acc    int32 [C, H, W] or [C]
    m0     int32 [C]
    shift  int32 [C]

    (acc * m0 + (1 << (shift-1))) >> shift, then clamp to [0, 255].

    The product is computed in int64 because acc and m0 are both int32
    and their product does not fit in 32 bits. In HLS this is an
    ap_int<64> intermediate, or a DSP48 cascade. It is NOT a float
    multiply, and the >> is an arithmetic shift, not a divide.
    """
    acc = np.asarray(acc, dtype=np.int64)
    m0 = np.asarray(m0, dtype=np.int64)
    shift = np.asarray(shift, dtype=np.int64)

    # Broadcast per-channel params across spatial dims.
    while m0.ndim < acc.ndim:
        m0 = m0[..., None]
        shift = shift[..., None]

    round_add = np.left_shift(np.int64(1), shift - 1)
    scaled = np.right_shift(acc * m0 + round_add, shift)
    return np.clip(scaled, ACT_MIN, ACT_MAX).astype(np.uint8)


def conv3x3(x_u8, w_i8, bias_i32):
    """3x3 stride-1 pad-1 convolution, integer accumulate.

    x_u8      uint8 [Cin, H, W]
    w_i8      int8  [Cout, Cin, 3, 3]
    bias_i32  int32 [Cout]
    returns   int32 [Cout, H, W]

    Accumulator bound: the largest layer is conv5 with Cin=64, so the
    worst case is 64*3*3 = 576 terms, each at most 127*255 = 32,385.
    576 * 32,385 = 18,653,760, well inside int32. The HLS kernel can
    therefore use ap_int<32> accumulators without saturation logic, and
    the testbench asserts this bound rather than assuming it.
    """
    cin, h, w = x_u8.shape
    cout = w_i8.shape[0]
    assert w_i8.shape[1:] == (cin, 3, 3), (w_i8.shape, x_u8.shape)

    # Zero pad. Activation zero-point is 0, so this is a true zero.
    xp = np.zeros((cin, h + 2, w + 2), dtype=np.int32)
    xp[:, 1:h + 1, 1:w + 1] = x_u8.astype(np.int32)

    acc = np.empty((cout, h, w), dtype=np.int32)
    acc[:] = bias_i32.astype(np.int32)[:, None, None]

    for kh in range(3):
        for kw in range(3):
            # Window for this kernel tap across the whole feature map.
            win = xp[:, kh:kh + h, kw:kw + w]                 # [Cin,H,W]
            tap = w_i8[:, :, kh, kw].astype(np.int32)         # [Cout,Cin]
            acc += np.einsum("oc,chw->ohw", tap, win, dtype=np.int32)

    return acc


def maxpool2x2(x_u8):
    """2x2 stride-2 max pool. A comparator tree in hardware."""
    c, h, w = x_u8.shape
    assert h % 2 == 0 and w % 2 == 0, f"odd spatial dim {h}x{w}"
    r = x_u8.reshape(c, h // 2, 2, w // 2, 2)
    return r.max(axis=(2, 4))


def global_sum(x_u8):
    """Sum over spatial dims. No divide — the 1/36 is folded into the
    GAP requantization multiplier at export time, so hardware never
    performs a division.

    Bound: 36 * 255 = 9,180 per channel. Fits int32 with room to spare.
    """
    return x_u8.astype(np.int32).sum(axis=(1, 2))


def dense(x_u8, w_i8, bias_i32):
    """Fully connected, int32 out, no requantization.

    The two class scores leave the kernel as int32. The host computes
    logit[1] - logit[0] and thresholds at 0, so no output scale is
    needed and nothing about the decision depends on a float.
    """
    return w_i8.astype(np.int32) @ x_u8.astype(np.int32) + bias_i32.astype(np.int32)


# -------------------------------------------------------------------------
# Whole-network forward
# -------------------------------------------------------------------------

def forward_tile(tile_u8, params, dump=None):
    """One 96x96x3 CHW uint8 tile -> int32 [2] logits.

    dump: optional dict; if given, each stage's output is stored under
    its name. The per-layer HLS testbenches compare against these, so a
    mismatch is localized to one layer instead of requiring a full
    network bisect.
    """
    assert tile_u8.dtype == np.uint8 and tile_u8.shape == (3, 96, 96), tile_u8.shape
    x = tile_u8

    for name in CONV_STAGES:
        acc = conv3x3(x, params[f"{name}_w"], params[f"{name}_bias"])
        if dump is not None:
            dump[f"{name}_acc"] = acc
        x = requantize(acc, params[f"{name}_m0"], params[f"{name}_shift"])
        if dump is not None:
            dump[name] = x
        if name in POOL_AFTER:
            x = maxpool2x2(x)
            if dump is not None:
                dump[f"{name}_pool"] = x

    s = global_sum(x)
    if dump is not None:
        dump["gap_acc"] = s + params["gap_bias"]
    x = requantize(s + params["gap_bias"], params["gap_m0"], params["gap_shift"])
    if dump is not None:
        dump["gap"] = x

    logits = dense(x, params["fc_w"], params["fc_bias"])
    if dump is not None:
        dump["fc"] = logits
    return logits


def forward_batch(tiles_u8, params):
    """[N,3,96,96] uint8 -> [N,2] int32. Loops; clarity over speed."""
    out = np.empty((tiles_u8.shape[0], 2), dtype=np.int32)
    for i in range(tiles_u8.shape[0]):
        out[i] = forward_tile(tiles_u8[i], params)
    return out


def tumor_score(logits_i32):
    """The single integer the kernel reports per tile. Threshold at 0."""
    return logits_i32[..., 1] - logits_i32[..., 0]


# -------------------------------------------------------------------------
# Loading and checking
# -------------------------------------------------------------------------

def load_params(artifacts_dir):
    path = os.path.join(artifacts_dir, "weights", "params.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run export.py on a QAT checkpoint first.")
    z = np.load(path)
    params = {k: z[k] for k in z.files}
    check_param_dtypes(params)
    return params


def check_param_dtypes(params):
    """Fail loudly if export.py emitted a float. This is the guard that
    keeps the no-float rule from silently eroding."""
    for k, v in params.items():
        if v.dtype.kind == "f":
            raise TypeError(f"{k} is {v.dtype}; artifacts must be integer only")
        if k.endswith("_w") and v.dtype != np.int8:
            raise TypeError(f"{k} is {v.dtype}, expected int8")
        if k.endswith(("_bias", "_m0", "_shift")) and v.dtype != np.int32:
            raise TypeError(f"{k} is {v.dtype}, expected int32")


def random_params(seed=0):
    """Structurally valid random parameters for testing the plumbing
    before export.py exists. These produce meaningless predictions —
    they exist only to exercise shapes, dtypes and accumulator bounds."""
    rng = np.random.default_rng(seed)
    shapes = [("conv1", 16, 3), ("conv2", 32, 16), ("conv3", 64, 32),
              ("conv4", 64, 64), ("conv5", 128, 64)]
    p = {}
    for name, co, ci in shapes:
        p[f"{name}_w"] = rng.integers(-127, 128, (co, ci, 3, 3), dtype=np.int8)
        p[f"{name}_bias"] = rng.integers(-1000, 1000, co).astype(np.int32)
        p[f"{name}_m0"] = rng.integers(2**29, 2**30, co).astype(np.int32)
        p[f"{name}_shift"] = np.full(co, 38, dtype=np.int32)
    p["gap_bias"] = np.zeros(128, dtype=np.int32)
    p["gap_m0"] = np.full(128, 1 << 25, dtype=np.int32)      # includes 1/36
    p["gap_shift"] = np.full(128, 30, dtype=np.int32)
    p["fc_w"] = rng.integers(-127, 128, (2, 128), dtype=np.int8)
    p["fc_bias"] = rng.integers(-1000, 1000, 2).astype(np.int32)
    return p


def self_test(seed=0):
    """Exercise the integer path end to end with random parameters.

    Checks the three things that actually break: dtypes stay integer,
    accumulators stay inside int32, and the forward pass is
    deterministic. Does not check accuracy — random weights have none.
    """
    p = random_params(seed)
    check_param_dtypes(p)
    rng = np.random.default_rng(seed + 1)
    tile = rng.integers(0, 256, (3, 96, 96), dtype=np.uint8)

    dump = {}
    logits = forward_tile(tile, p, dump=dump)

    print(f"{'stage':<12} {'shape':<18} {'dtype':<8} {'min':>8} {'max':>8}")
    print("-" * 58)
    for k, v in dump.items():
        print(f"{k:<12} {str(v.shape):<18} {str(v.dtype):<8} "
              f"{v.min():>8} {v.max():>8}")
        assert v.dtype.kind in "iu", f"{k} is not integer"
        if k.endswith("_acc"):
            lo, hi = int(v.min()), int(v.max())
            assert -2**31 <= lo and hi < 2**31, f"{k} overflows int32"

    print(f"\nlogits      : {logits}  (int32)")
    print(f"tumor score : {tumor_score(logits)}")

    again = forward_tile(tile, p)
    assert np.array_equal(logits, again), "forward is not deterministic"
    print("\nself-test OK: integer-only, int32-safe, deterministic")
    print("NOTE: random weights — this says nothing about accuracy.")


def run_split(artifacts_dir, data_root, split, limit):
    """Run the golden model over packed tiles and report accuracy.

    Reads the same flat binary the XRT host will read, so the bytes under
    test are the bytes the hardware sees.
    """
    from data import TILE_BYTES

    params = load_params(artifacts_dir)
    bin_path = os.path.join(artifacts_dir, "vectors", f"{split}_tiles.bin")
    lab_path = bin_path + ".labels.u8"
    if not os.path.exists(bin_path):
        raise FileNotFoundError(
            f"{bin_path} not found. Run:\n"
            f"  python phase1_software/data.py --root {data_root} "
            f"--dump-bin {bin_path} --split {split}")

    raw = np.fromfile(bin_path, dtype=np.uint8)
    labels = np.fromfile(lab_path, dtype=np.uint8)
    n = raw.size // TILE_BYTES
    if limit:
        n = min(n, limit)
    tiles = raw[:n * TILE_BYTES].reshape(n, 3, 96, 96)
    labels = labels[:n].astype(np.int32)

    scores = np.empty(n, dtype=np.int32)
    for i in range(n):
        scores[i] = tumor_score(forward_tile(tiles[i], params))
        if (i + 1) % 256 == 0:
            print(f"  {i+1}/{n}", end="\r", flush=True)

    pred = (scores > 0).astype(np.int32)
    acc = (pred == labels).mean()
    print(f"\n{split}: {n} tiles, integer accuracy {acc:.4f}")
    print("Compare against the Brevitas INT8 model on the same tiles; they "
          "must agree on every tile, not merely in aggregate accuracy.")

    out = os.path.join(artifacts_dir, "vectors", f"{split}_golden_scores.i32")
    scores.tofile(out)
    print(f"wrote {out}")
    return scores


# (name, dtype, C, H, W) in file and manifest order.
DUMP_STAGES = [
    ("input",      np.uint8,   3, 96, 96),
    ("conv1_acc",  np.int32,  16, 96, 96),
    ("conv1",      np.uint8,  16, 96, 96),
    ("conv1_pool", np.uint8,  16, 48, 48),
    ("conv2_acc",  np.int32,  32, 48, 48),
    ("conv2",      np.uint8,  32, 48, 48),
    ("conv2_pool", np.uint8,  32, 24, 24),
    ("conv3_acc",  np.int32,  64, 24, 24),
    ("conv3",      np.uint8,  64, 24, 24),
    ("conv3_pool", np.uint8,  64, 12, 12),
    ("conv4_acc",  np.int32,  64, 12, 12),
    ("conv4",      np.uint8,  64, 12, 12),
    ("conv4_pool", np.uint8,  64,  6,  6),
    ("conv5_acc",  np.int32, 128,  6,  6),
    ("conv5",      np.uint8, 128,  6,  6),
    ("gap_acc",    np.int32, 128,  1,  1),
    ("gap",        np.uint8, 128,  1,  1),
    ("fc",         np.int32,   2,  1,  1),
]
FLAT_STAGES = {"gap_acc", "gap", "fc"}     # stored in the dump dict as [C]


def dump_layers(artifacts_dir, split, n):
    """Write per-layer reference outputs for the first n tiles of a split
    to artifacts/layer_dumps/, one raw little-endian C-order file per
    stage, tile-major, plus manifest.txt (written last). Never overwrites.

    Stage files: <name>.bin holds tile 0, then tile 1, ... (tile-major).
    Each tile's block is the stage tensor in C (row-major, CHW) order,
    little-endian, with the dtype given in the manifest.

    manifest.txt format (ASCII, one record per line, space-separated):

        pcam_layer_dump_v1                      magic, line 1
        split <split>                           8 header lines,
        num_tiles <n>                           always in this order
        tile_first 0
        tile_last <n-1>
        tile_bytes 27648
        byte_order little
        memory_order c_chw
        num_stages <S>
        <name> <file> <dtype> <C> <H> <W> <elems_per_tile> <bytes_per_tile>
        ...                                     S stage lines, file order

    1-D stages (gap_acc, gap, fc) are written with H = W = 1, i.e. as
    "C 1 1". manifest.txt is written only after every stage file has been
    closed and size-checked, so a directory with stage .bin files but no
    manifest.txt holds an incomplete (interrupted) dump and must not be
    read.
    """
    tile_bytes = 3 * 96 * 96
    if tile_bytes != 27648:
        raise RuntimeError(f"tile_bytes is {tile_bytes}, expected 27648")
    # Stage files are written with tobytes() in native order and the
    # manifest declares little-endian, so refuse to run on a big-endian host.
    if sys.byteorder != "little":
        raise RuntimeError(f"host byte order is {sys.byteorder}; layer dumps "
                           f"are declared little-endian")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")

    bin_path = os.path.join(artifacts_dir, "vectors", f"{split}_tiles.bin")
    raw = np.fromfile(bin_path, dtype=np.uint8, count=n * tile_bytes)
    if raw.size != n * tile_bytes:
        raise ValueError(f"{bin_path} holds fewer than {n} tiles "
                         f"({raw.size} bytes read)")
    tiles = raw.reshape(n, 3, 96, 96)
    params = load_params(artifacts_dir)

    out_dir = os.path.join(artifacts_dir, "layer_dumps")
    names = ["manifest.txt"] + [f"{s[0]}.bin" for s in DUMP_STAGES]
    if os.path.isdir(out_dir):
        present = [f for f in names if os.path.exists(os.path.join(out_dir, f))]
        if present and "manifest.txt" not in present:
            raise FileExistsError(
                f"{out_dir} contains {', '.join(present)} but no "
                f"manifest.txt: a previous dump was interrupted and is "
                f"incomplete. Refusing to overwrite. Decide whether to "
                f"delete these files yourself before rerunning "
                f"(CLAUDE.md: ask before deleting anything in artifacts/).")
        if present:
            raise FileExistsError(
                f"{out_dir} already contains {', '.join(present)}; "
                f"refusing to overwrite")
    os.makedirs(out_dir, exist_ok=True)

    files = {}
    meta = {}
    try:
        # Opens are inside the try so a failure on any one of them still
        # closes the handles already opened.
        for name, *_ in DUMP_STAGES:
            files[name] = open(os.path.join(out_dir, f"{name}.bin"), "xb")
        for i in range(n):
            d = {}
            forward_tile(tiles[i], params, dump=d)
            d["input"] = tiles[i]
            for name, dt, c, h, w in DUMP_STAGES:
                if name not in d:
                    raise KeyError(f"forward_tile dump has no '{name}'")
                arr = d[name]
                if name == "gap_acc":
                    # Checked narrowing, gap_acc only. Under NumPy 2, int32
                    # .sum() promotes to int64 on every platform, so
                    # forward_tile holds this as int64 (under NumPy 1.x on
                    # Windows it stayed int32, hence both are accepted).
                    # Values are bounded by 36*255 + |bias|, and the kernel
                    # accumulates GAP in int32, so the dump is int32 once
                    # the range check proves nothing is lost.
                    if arr.dtype not in (np.int64, np.int32):
                        raise ValueError(
                            f"stage {name}, tile {i}: dtype {arr.dtype}, "
                            f"expected int64 or int32")
                    lo, hi = np.iinfo(np.int32).min, np.iinfo(np.int32).max
                    if not (int(arr.min()) >= lo and int(arr.max()) <= hi):
                        raise ValueError(
                            f"stage {name}, tile {i}: range "
                            f"[{int(arr.min())}, {int(arr.max())}] does not "
                            f"fit int32")
                    narrowed = arr.astype(np.int32)
                    if not np.array_equal(narrowed.astype(np.int64),
                                          arr.astype(np.int64)):
                        raise ValueError(
                            f"stage {name}, tile {i}: int32 narrowing does "
                            f"not round-trip")
                    arr = narrowed
                if arr.dtype != dt:
                    raise ValueError(
                        f"stage {name}, tile {i}: dtype {arr.dtype}, "
                        f"expected {np.dtype(dt)}")
                want = (c,) if name in FLAT_STAGES else (c, h, w)
                if not (arr.shape == want and arr.size == c * h * w):
                    raise ValueError(
                        f"stage {name}, tile {i}: shape {arr.shape}, "
                        f"expected {want}")
                b = np.ascontiguousarray(arr).tobytes()
                if len(b) != c * h * w * arr.itemsize:
                    raise RuntimeError(
                        f"stage {name}, tile {i}: serialized {len(b)} bytes, "
                        f"expected {c * h * w * arr.itemsize}")
                files[name].write(b)
                meta[name] = (arr.dtype.name, c, h, w, c * h * w, len(b))
    finally:
        for f in files.values():
            f.close()

    for name, *_ in DUMP_STAGES:
        p = os.path.join(out_dir, f"{name}.bin")
        if os.path.getsize(p) != n * meta[name][5]:
            raise RuntimeError(
                f"stage {name}: {p} is {os.path.getsize(p)} bytes, expected "
                f"{n * meta[name][5]} ({n} tiles); manifest not written")

    lines = ["pcam_layer_dump_v1",
             f"split {split}",
             f"num_tiles {n}",
             "tile_first 0",
             f"tile_last {n - 1}",
             f"tile_bytes {tile_bytes}",
             "byte_order little",
             "memory_order c_chw",
             f"num_stages {len(DUMP_STAGES)}"]
    for name, *_ in DUMP_STAGES:
        dtn, c, h, w, e, bpt = meta[name]
        lines.append(f"{name} {name}.bin {dtn} {c} {h} {w} {e} {bpt}")
    man = os.path.join(out_dir, "manifest.txt")
    with open(man, "xb") as f:
        f.write(("\n".join(lines) + "\n").encode("ascii"))

    print(f"read {bin_path} tiles 0..{n - 1}, params from "
          f"{os.path.join(artifacts_dir, 'weights', 'params.npz')}")
    print(f"wrote {len(DUMP_STAGES)} stage files + manifest to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--artifacts", default="./artifacts")
    ap.add_argument("--root", default="./data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dump-layers", type=int, nargs="?", const=8,
                    default=None, metavar="N")
    a = ap.parse_args()

    if a.dump_layers is not None:
        if a.self_test or a.limit is not None:
            ap.error("--dump-layers cannot be combined with --self-test or --limit")
        if a.dump_layers < 1:
            ap.error("--dump-layers N requires N >= 1")
        dump_layers(a.artifacts, a.split, a.dump_layers)
        sys.exit(0)

    if a.self_test:
        self_test()
    else:
        run_split(a.artifacts, a.root, a.split, a.limit)