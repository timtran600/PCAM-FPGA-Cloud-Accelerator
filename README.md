# pcam_accel — INT8 PCam CNN inference on Alveo U280

EECE5698 cloud FPGA project. Trains a small CNN on PatchCamelyon in
PyTorch, quantizes it to INT8 with Brevitas, reimplements inference as a
Vitis HLS dataflow kernel, and measures throughput on one, two, and four
U280s on the Open Cloud Testbed.

## Layout

```
pcam_accel/
  data/                # PCam .h5 files (gitignored, ~8 GB)
  models/              # trained checkpoints (gitignored except notes)
  artifacts/           # handoff boundary — phase1 writes, phase2/3 read
    weights/           #   weights.bin, scales.bin, params.h
    vectors/           #   test_tiles.bin, expected_logits.bin, labels.u8
    layer_dumps/       #   per-layer INT activations for a few tiles
  phase1_software/
    data.py            # PCam HDF5 loader, sanity check, packed-tile dumper [done]
    model.py           # FP32 PCamCNN                                       [next]
    train.py           # FP32 training loop
    qat.py             # Brevitas INT8 QAT
    export.py          # dump INT8 weights, scales, zero-points
    golden.py          # pure-integer NumPy reference model
    bench_baseline.py  # GPU/CPU latency + throughput baseline
    tests/
  phase2_hls/          # src/ tb/ scripts/ cfg/ reports/
  phase3_cloud/        # host/ multi_fpga/ xclbin/ results/
  phase4_stretch/      # opennic/ vck5000/
  report/              # proposal, checkpoints, final, figures/
```

Phase 1 runs on the GPU box, not the OCT build VM. Only `artifacts/` and
`phase2_hls/` cross over to the FPGA side.

## Setup

```bash
conda create -n pcam python=3.10 -y && conda activate pcam
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Data

`torchvision.datasets.PCAM(..., download=True)` pulls from Google Drive and
fails on quota often enough to be unreliable. Use the direct HDF5 release
linked from <https://github.com/basveeling/pcam> — six `.h5.gz` files,
about 8 GB gunzipped — and put them in `data/`.

```bash
gunzip data/*.gz
python phase1_software/data.py --root ./data --check
```

Expect 262,144 / 32,768 / 32,768 tiles, shape `(3, 96, 96)`, dtype
`uint8`, and a positive fraction near 0.50 in all three splits. A length
mismatch means a partial download.

## Conventions that the hardware depends on

- Tiles stay `uint8` until `to_float01`, which divides by **256**, not 255,
  so the input quantizer scale is 2^-8 and the kernel shifts instead of
  multiplying.
- No mean/std normalization — it would become an affine stage in HLS.
- `dump_bin` writes tiles CHW, 27,648 B each, contiguous. The golden model
  (Part 1) and the XRT host program (Part 3) read this same file, so a
  mismatch on hardware is a kernel bug and not a data bug.
- Known caveat for the report: PCam labels refer only to the centre 32×32
  region of each 96×96 tile, though the standard practice is to train on
  the whole tile. Worth citing when discussing the accuracy ceiling.