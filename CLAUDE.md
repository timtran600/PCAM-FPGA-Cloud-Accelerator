# CLAUDE.md

INT8 PCam CNN inference accelerator for Alveo U280 on the Open Cloud Testbed.
EECE 5698, Fall 2026. Solo project.

## 1. Hard Rules

- Never change the topology without being asked: 5x conv 3x3 at 16/32/64/64/128, pool 2x2 after the first four, GAP, FC->2. 134,320 weights, 33.2 MMAC/tile.
- Every conv is 3x3, stride 1, pad 1, `bias=False`. BatchNorm supplies the affine term.
- Layer order is always Conv -> BN -> ReLU. BN after ReLU does not fold.
- Tiles stay `uint8` until `to_float01`, which divides by **256**, not 255. Input quantizer scale must stay 2^-8.
- No mean/std normalization anywhere in the data path.
- Packed tile layout is CHW, 27,648 B/tile, contiguous. `phase1_software/golden.py` and `phase3_cloud/host/` read the same bytes.
- `golden.py` is pure integer NumPy. No float ops, no `np.float*` dtypes, no division that is not an arithmetic right shift.
- Correctness is defined against the golden model, not against dataset labels.
- Bit-exact means bit-exact. Never relax a comparison to `allclose` or add a tolerance to make a test pass.
- `artifacts/` is the only directory that crosses from the GPU box to the OCT build VM. Nothing else is a handoff.
- Never commit `data/`, `models/`, `checkpoints/`, `*.xclbin`, or `*.xo`.
- Do not add a dependency without being asked.
- One pragma change per synthesis run, logged in `phase2_hls/reports/DSE_LOG.md`.

## 2. Authority & Links

- Proposal (authoritative scope + success criteria): `report/proposal.md`
- Master to-do list and phase breakdown: `README.md`
- DSE results table: `phase2_hls/reports/DSE_LOG.md`
- Trained-model results: `models/RESULTS.md`
- PCam dataset: https://github.com/basveeling/pcam
- Brevitas: https://github.com/Xilinx/brevitas
- OpenNIC shell: https://github.com/Xilinx/open-nic-shell
- Vitis HLS User Guide: UG1399
- Alveo U280 data sheet: DS963

## 3. Setup / Test

- Phase 1 runs on the GPU box. Phases 2-3 run on the OCT build VM with Vitis 2023.2.
- Environment:
  - `conda create -n pcam python=3.10 -y && conda activate pcam`
  - `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`
  - `pip install -r requirements.txt`
- Data: download the PCam HDF5 release, gunzip into `data/`, then `python phase1_software/data.py --root ./data --check`
- Expect 262,144 / 32,768 / 32,768 tiles, `(3, 96, 96)`, `uint8`, positive fraction near 0.50.
- Tests: `pytest phase1_software/tests/ -q`

- Phase 2 runs locally on Vitis HLS 2023.2
  (`C:\Xilinx\Vitis_HLS\2023.2\bin\vitis_hls.bat`), the same version as
  the OCT build machine. 2022.1 and 2024.1 are also installed locally and
  must not be used for this project — tool version changes move latency,
  resource and Fmax numbers, and the .xclbin must match the card's
  platform shell.
- All DSE_LOG.md entries and every number quoted in the report come from
  2023.2. Phase 3 (v++ link, .xclbin, hardware runs) still happens on the
  OCT build machine.
- Run HLS from the Vitis-provided shell or by full path, not from the
  activated pcam_env PowerShell — Vitis ships its own Python and library
  paths that conflict with the venv.

## 4. Workflow

- `python phase1_software/model.py` — shape walk, parameter and MAC counts
- `python phase1_software/train.py --epochs 30 --batch 256 --seed 0 --out models/fp32_best.pt`
- `python phase1_software/qat.py --init models/fp32_best.pt --out models/qat_int8_best.pt`
- `python phase1_software/export.py --ckpt models/qat_int8_best.pt --out artifacts/`
- `python phase1_software/golden.py --artifacts ./artifacts --split test` — must match Brevitas on all 32,768 tiles

## 5. Stop Conditions

- Stop and ask before changing the topology, channel counts, kernel size, padding, or pooling.
- Stop and ask before changing the input scale, the tile byte layout, or anything in `artifacts/`.
- Stop if a bit-exactness check fails. Report which layer and which tile index diverged. Do not widen the tolerance, do not regenerate the golden model to match the kernel.
- Stop before regenerating `artifacts/weights/` from a different checkpoint than the one named in `models/RESULTS.md`.
- Stop before deleting or overwriting anything in `artifacts/`, `models/`, or `phase2_hls/reports/`.
- Stop before running a v++ link or any build expected to exceed a few minutes; ask first.
- Stop if a requested change would make the golden model depend on PyTorch, or on float arithmetic.
- Ask rather than guess when a number is needed that is not in `report/proposal.md` or measured in the repo. Do not invent accuracy, throughput, resource, or link-rate figures.