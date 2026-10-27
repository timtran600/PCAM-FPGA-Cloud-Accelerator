"""
PCam data access for the EECE5698 FPGA accelerator project.

Design rule that matters for the FPGA: samples come out of the Dataset as
raw uint8 CHW tensors. Any scaling happens in exactly one place
(`to_float01`), which divides by 256 rather than 255, so the input
quantizer scale stays a power of two and the HLS kernel can consume raw
tile bytes with a shift instead of a multiply. There is no mean/std
normalization anywhere, because that would become an affine
pre-processing stage to reproduce in hardware.

Usage:
    python phase1_software/data.py --root ./data --check
    python phase1_software/data.py --root ./data \
        --dump-bin ./artifacts/vectors/test_tiles.bin
"""

import argparse
import os

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# Official PCam release file names (the same ones torchvision downloads).
SPLITS = {
    "train": ("camelyonpatch_level_2_split_train_x.h5",
              "camelyonpatch_level_2_split_train_y.h5"),
    "valid": ("camelyonpatch_level_2_split_valid_x.h5",
              "camelyonpatch_level_2_split_valid_y.h5"),
    "test":  ("camelyonpatch_level_2_split_test_x.h5",
              "camelyonpatch_level_2_split_test_y.h5"),
}

EXPECTED_LEN = {"train": 262144, "valid": 32768, "test": 32768}

TILE_BYTES = 3 * 96 * 96  # 27,648 B per tile, CHW


def _resolve(root: str, fname: str) -> str:
    """Accept either <root>/<file> or torchvision's <root>/pcam/<file>."""
    for cand in (os.path.join(root, fname),
                 os.path.join(root, "pcam", fname)):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"Could not find {fname} under {root} (or {root}/pcam). "
        "Download the PCam HDF5 release and gunzip it."
    )


class PCamH5(Dataset):
    """Returns (uint8 tensor [3,96,96], int label).

    The HDF5 handles are opened lazily, per worker process, because an
    h5py.File object cannot be shared across a fork.
    """

    def __init__(self, root: str, split: str, augment: bool = False):
        assert split in SPLITS, f"split must be one of {list(SPLITS)}"
        self.split = split
        self.augment = augment
        self.x_path = _resolve(root, SPLITS[split][0])
        self.y_path = _resolve(root, SPLITS[split][1])
        self._x = None
        self._y = None
        with h5py.File(self.y_path, "r") as f:
            self._len = int(f["y"].shape[0])

    def _ensure_open(self):
        if self._x is None:
            self._x = h5py.File(self.x_path, "r")["x"]
            self._y = h5py.File(self.y_path, "r")["y"]

    def __len__(self):
        return self._len

    def __getitem__(self, i: int):
        self._ensure_open()
        img = np.asarray(self._x[i])               # HWC uint8, (96,96,3)
        label = int(np.asarray(self._y[i]).reshape(-1)[0])

        if self.augment:
            # No canonical orientation for a slide patch, so the dihedral
            # subgroup of rotations + horizontal flip is label-preserving.
            k = np.random.randint(4)
            if k:
                img = np.rot90(img, k)
            if np.random.rand() < 0.5:
                img = img[:, ::-1]

        chw = np.ascontiguousarray(img.transpose(2, 0, 1))  # CHW uint8
        return torch.from_numpy(chw), label

    def labels(self) -> np.ndarray:
        with h5py.File(self.y_path, "r") as f:
            return np.asarray(f["y"]).reshape(-1).astype(np.int64)


def to_float01(x_uint8: torch.Tensor) -> torch.Tensor:
    """The ONLY place a tile becomes float. /256 keeps the scale 2^-8."""
    return x_uint8.float() / 256.0


def make_loaders(root, batch_size=256, workers=8, augment_train=True):
    train = PCamH5(root, "train", augment=augment_train)
    valid = PCamH5(root, "valid")
    test = PCamH5(root, "test")
    kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, **kw),
        DataLoader(valid, batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test, batch_size=batch_size, shuffle=False, **kw),
    )


def check(root: str):
    print(f"{'split':>7} {'len':>9} {'expected':>9} {'pos frac':>9}")
    for split in SPLITS:
        ds = PCamH5(root, split)
        y = ds.labels()
        ok = "" if len(ds) == EXPECTED_LEN[split] else "   <-- MISMATCH"
        print(f"{split:>7} {len(ds):>9} {EXPECTED_LEN[split]:>9} {y.mean():>9.4f}{ok}")
        x, lab = ds[0]
        assert x.dtype == torch.uint8 and tuple(x.shape) == (3, 96, 96), (x.dtype, x.shape)
        assert set(np.unique(y)).issubset({0, 1})

    # Per-channel statistics on a sample of train, for the report only.
    ds = PCamH5(root, "train")
    idx = np.random.default_rng(0).choice(len(ds), 2000, replace=False)
    sample = np.stack([ds[int(i)][0].numpy() for i in sorted(idx)]).astype(np.float32) / 256.0
    print("\ntrain mean (RGB):", sample.mean(axis=(0, 2, 3)).round(4))
    print("train std  (RGB):", sample.std(axis=(0, 2, 3)).round(4))
    print("\nNOTE: stats are for the report only — training uses x/256 with no "
          "mean/std normalization so the FPGA can eat raw bytes.")


def dump_bin(root, out_path, split="test", limit=None):
    """Flat binary of packed tiles for hardware timing runs.

    Layout: tile i occupies bytes [i*27648, (i+1)*27648) in CHW order,
    matching what the HLS kernel will stream from HBM. Labels go to a
    sibling .labels.u8 file, one byte per tile.
    """
    ds = PCamH5(root, split)
    n = len(ds) if limit is None else min(limit, len(ds))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as fx, open(out_path + ".labels.u8", "wb") as fy:
        for i in range(n):
            x, y = ds[i]
            fx.write(x.numpy().tobytes())
            fy.write(bytes([y]))
            if (i + 1) % 4096 == 0:
                print(f"  {i+1}/{n}", end="\r")
    print(f"\nwrote {out_path}: {n} tiles x {TILE_BYTES} B = "
          f"{n * TILE_BYTES / 2**20:.1f} MiB")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./data")
    p.add_argument("--check", action="store_true")
    p.add_argument("--dump-bin", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    if a.check:
        check(a.root)
    if a.dump_bin:
        dump_bin(a.root, a.dump_bin, a.split, a.limit)
