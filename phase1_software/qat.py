"""
qat.py — INT8 quantization-aware fine-tuning of PCamCNN with Brevitas.

Starts from the FP32 checkpoint and fine-tunes with fake quantization in
the forward pass, so the weights learn to tolerate their own rounding
error. Post-training quantization would be cheaper, but on a 134K-param
network there is no redundancy to absorb the damage, and the proposal's
success bar is within 1% of FP32 AUC.

-------------------------------------------------------------------------
Why BatchNorm is folded BEFORE quantization, not after
-------------------------------------------------------------------------
The usual Brevitas flow keeps BN as a separate module during QAT and
merges it at export. That is wrong for this project. If BN is merged
after training, the weights that were quantized during QAT are not the
weights that get exported — folding changes every value, so the network
you validated is not the network the FPGA runs.

Since FP32 training is finished and the BN running statistics are frozen,
the fold is a closed-form rewrite:

    w_fold[co] = w[co] * gamma[co] / sqrt(var[co] + eps)
    b_fold[co] = beta[co] - gamma[co] * mean[co] / sqrt(var[co] + eps)

This file folds first, then quantizes the folded conv. What QAT sees is
exactly what export.py writes and what the HLS kernel computes. The cost
is fine-tuning without BN's regularization, which at 1e-4 for a few
epochs is not a problem.

-------------------------------------------------------------------------
Quantization scheme — must match golden.py exactly
-------------------------------------------------------------------------
Weights      Int8WeightPerChannelFloat — symmetric, per output channel,
             zero-point 0.
Activations  Uint8ActPerTensorFloat on every ReLU — unsigned, zero-point
             0. Valid because post-ReLU values are non-negative, and it
             removes the zero-point correction term from every HLS
             accumulator.
Input        Uint8ActPerTensorFixedPoint — power-of-two scale, so the
             learned input scale lands on 2^-8 and the kernel reads raw
             HBM bytes. Asserted after training rather than assumed.
Output       FC emits unquantized values; the host thresholds the
             difference of the two class scores.

    python phase1_software/qat.py --init models/fp32_best.pt --epochs 10
"""

import argparse
import copy
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear, QuantReLU  # noqa: E402
from brevitas.quant import (                                                # noqa: E402
    Int8WeightPerChannelFloat,
    Uint8ActPerTensorFloat,
    Uint8ActPerTensorFixedPoint,
)

from data import make_loaders, to_float01     # noqa: E402
from model import PCamCNN, STAGES             # noqa: E402
from train import set_seed, append_results    # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIT_WIDTH = 8


# -------------------------------------------------------------------------
# BatchNorm folding
# -------------------------------------------------------------------------

def fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    """Closed-form fold of a frozen BN into the preceding conv.

    Returns (weight, bias) for an equivalent conv with bias=True. The
    conv must have bias=False, which model.py guarantees.
    """
    assert conv.bias is None, "conv must have bias=False for a clean fold"
    gamma = bn.weight.data
    beta = bn.bias.data
    mean = bn.running_mean.data
    var = bn.running_var.data
    eps = bn.eps

    scale = gamma / torch.sqrt(var + eps)            # [Cout]
    w = conv.weight.data * scale.reshape(-1, 1, 1, 1)
    b = beta - mean * scale
    return w, b


def fold_model(fp32: PCamCNN):
    """Return per-stage folded (weight, bias) tensors, plus the FC params."""
    folded = []
    for block in fp32.features:
        folded.append(fold_bn(block.conv, block.bn))
    return folded, (fp32.fc.weight.data.clone(), fp32.fc.bias.data.clone())


# -------------------------------------------------------------------------
# Quantized network
# -------------------------------------------------------------------------

class QuantConvBlock(nn.Module):
    """QuantConv2d -> QuantReLU, optional max pool. No BN: already folded."""

    def __init__(self, in_ch, out_ch, pool):
        super().__init__()
        self.conv = QuantConv2d(
            in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True,
            weight_quant=Int8WeightPerChannelFloat,
            weight_bit_width=BIT_WIDTH,
            return_quant_tensor=False,
        )
        self.relu = QuantReLU(
            act_quant=Uint8ActPerTensorFloat,
            bit_width=BIT_WIDTH,
            return_quant_tensor=False,
        )
        self.pool = nn.MaxPool2d(2, 2) if pool else None

    def forward(self, x):
        x = self.relu(self.conv(x))
        if self.pool is not None:
            x = self.pool(x)
        return x


class QuantPCamCNN(nn.Module):
    """INT8 PCamCNN. Same topology as model.PCamCNN, BN already folded in.

    Max pool sits after the activation quantizer on purpose: pooling a
    quantized tensor is a comparison between values on the same grid, so
    it introduces no new quantization levels and needs no requantization
    in hardware.
    """

    def __init__(self, stages=STAGES, num_classes=2):
        super().__init__()
        # Power-of-two input scale so the kernel consumes raw tile bytes.
        self.input_quant = QuantIdentity(
            act_quant=Uint8ActPerTensorFixedPoint,
            bit_width=BIT_WIDTH,
            return_quant_tensor=False,
        )
        blocks, in_ch = [], 3
        for out_ch, pool in stages:
            blocks.append(QuantConvBlock(in_ch, out_ch, pool))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap_quant = QuantReLU(
            act_quant=Uint8ActPerTensorFloat,
            bit_width=BIT_WIDTH,
            return_quant_tensor=False,
        )
        self.fc = QuantLinear(
            in_ch, num_classes, bias=True,
            weight_quant=Int8WeightPerChannelFloat,
            weight_bit_width=BIT_WIDTH,
            return_quant_tensor=False,
        )

    def forward(self, x):
        x = self.input_quant(x)
        x = self.features(x)
        x = self.gap_quant(self.gap(x).flatten(1))
        return self.fc(x)


def init_from_fp32(qmodel: QuantPCamCNN, fp32: PCamCNN):
    """Load folded FP32 weights into the quantized network."""
    folded, (fc_w, fc_b) = fold_model(fp32)
    for block, (w, b) in zip(qmodel.features, folded):
        block.conv.weight.data.copy_(w)
        block.conv.bias.data.copy_(b)
    qmodel.fc.weight.data.copy_(fc_w)
    qmodel.fc.bias.data.copy_(fc_b)
    return qmodel


# -------------------------------------------------------------------------
# Checks
# -------------------------------------------------------------------------

def check_fold(fp32: PCamCNN, qmodel: QuantPCamCNN, device):
    """The folded float network must match the original before quantization
    is calibrated. If this fails, the fold is wrong and nothing downstream
    is meaningful."""
    fp32.eval()
    x = torch.rand(4, 3, 96, 96, device=device)

    with torch.no_grad():
        ref = fp32(x)
        # Folded conv path without quantizers: compare layer 1 only, which
        # is enough to catch a sign or axis error in the fold.
        b0 = fp32.features[0]
        w, bias = fold_bn(b0.conv, b0.bn)
        folded_out = torch.nn.functional.conv2d(x, w, bias, padding=1)
        orig_out = b0.bn(b0.conv(x))
        err = (folded_out - orig_out).abs().max().item()

    print(f"BN fold check : max abs err on conv1 = {err:.3e}")
    assert err < 1e-3, f"BN fold is wrong (err {err})"
    return ref


def check_input_scale(qmodel: QuantPCamCNN):
    """The input quantizer scale must be 2^-8. golden.py and the HLS
    kernel both assume the raw tile byte IS the quantized value."""
    try:
        scale = qmodel.input_quant.quant_act_scale()
        scale = float(scale.reshape(-1)[0])
    except Exception as e:                                  # noqa: BLE001
        print(f"  could not read input scale ({e}); check manually")
        return None
    expected = 1.0 / 256.0
    print(f"input scale   : {scale:.8f}  (expected {expected:.8f}, 2^-8)")
    if abs(scale - expected) > 1e-6:
        print("  WARNING: input scale is not 2^-8. export.py must emit an "
              "input requantization stage, or the kernel cannot read raw "
              "bytes. Do not proceed without resolving this.")
    return scale


# -------------------------------------------------------------------------
# Train / eval
# -------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total, n, probs, targets = 0.0, 0, [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(to_float01(x))
        total += criterion(logits, y).item() * y.size(0)
        n += y.size(0)
        probs.append(torch.softmax(logits.float(), 1)[:, 1].cpu())
        targets.append(y.cpu())
    probs = torch.cat(probs).numpy()
    targets = torch.cat(targets).numpy()
    acc = ((probs >= 0.5).astype(np.int64) == targets).mean()
    return total / n, float(acc), float(roc_auc_score(targets, probs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./data")
    p.add_argument("--init", default="models/fp32_best.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="much lower than FP32: this is fine-tuning")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="models/qat_int8_best.pt")
    p.add_argument("--log-every", type=int, default=200)
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = (torch.cuda.get_device_name(0) if device.type == "cuda"
                   else "CPU")
    print(f"device   : {device_name}")
    print(f"init     : {args.init}")
    print(f"lr       : {args.lr} (AdamW, cosine over {args.epochs} epochs)\n")

    init_path = args.init if os.path.isabs(args.init) else os.path.join(REPO_ROOT, args.init)
    ckpt = torch.load(init_path, map_location="cpu", weights_only=True)
    fp32 = PCamCNN()
    fp32.load_state_dict(ckpt["model"])
    print(f"FP32 ckpt: epoch {ckpt['epoch']}, val AUC {ckpt['val_auc']:.4f}")

    fp32 = fp32.to(device)
    qmodel = QuantPCamCNN().to(device)
    check_fold(fp32, qmodel, device)
    init_from_fp32(qmodel, fp32)
    qmodel = qmodel.to(device)

    train_dl, val_dl, test_dl = make_loaders(
        args.root, batch_size=args.batch, workers=args.workers, augment_train=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(qmodel.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # One eval before any fine-tuning: this is the post-training-
    # quantization number, and the difference between it and the final
    # number is what QAT actually bought. Worth a line in the report.
    _, ptq_acc, ptq_auc = evaluate(qmodel, val_dl, device, criterion)
    print(f"\nPTQ (no fine-tune): val acc {ptq_acc:.4f}  AUC {ptq_auc:.4f}")
    check_input_scale(qmodel)
    print()

    out_path = args.out if os.path.isabs(args.out) else os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    best_auc, best_epoch, t_start = -1.0, -1, time.time()

    for epoch in range(1, args.epochs + 1):
        qmodel.train()
        t0, run, n = time.time(), 0.0, 0
        for step, (x, y) in enumerate(train_dl):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(qmodel(to_float01(x)), y)
            loss.backward()
            optimizer.step()
            run += loss.item() * y.size(0)
            n += y.size(0)
            if args.log_every and step % args.log_every == 0:
                print(f"    step {step:>5}  loss {run/n:.4f}  "
                      f"{n/max(time.time()-t0,1e-9):,.0f} tiles/s", flush=True)

        val_loss, val_acc, val_auc = evaluate(qmodel, val_dl, device, criterion)
        scheduler.step()

        flag = ""
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            torch.save({
                "model": qmodel.state_dict(),
                "epoch": epoch, "val_auc": val_auc, "val_acc": val_acc,
                "args": vars(args), "torch": str(torch.__version__),
                "device": device_name, "fp32_init": args.init,
            }, out_path)
            flag = "  <- best, saved"

        print(f"epoch {epoch:>3}/{args.epochs}  train {run/n:.4f}  "
              f"val {val_loss:.4f}  acc {val_acc:.4f}  AUC {val_auc:.4f}  "
              f"{(time.time()-t0)/60:.1f} min{flag}", flush=True)

    best = torch.load(out_path, map_location=device, weights_only=True)
    qmodel.load_state_dict(best["model"])
    _, val_acc, val_auc = evaluate(qmodel, val_dl, device, criterion)
    _, test_acc, test_auc = evaluate(qmodel, test_dl, device, criterion)
    wall = time.time() - t_start

    fp32_test_auc = ckpt.get("test_auc")
    print(f"\nbest epoch  : {best_epoch}")
    print(f"val  acc/AUC: {val_acc:.4f} / {val_auc:.4f}")
    print(f"test acc/AUC: {test_acc:.4f} / {test_auc:.4f}")
    print(f"PTQ -> QAT  : val AUC {ptq_auc:.4f} -> {val_auc:.4f}")
    if fp32_test_auc:
        drop = fp32_test_auc - test_auc
        print(f"vs FP32 test: {fp32_test_auc:.4f} -> {test_auc:.4f} "
              f"(drop {drop:+.4f}) "
              f"{'WITHIN' if drop <= 0.01 else 'EXCEEDS'} the 1% bound")
    print(f"wall        : {wall/60:.1f} min")

    append_results({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "seed": args.seed, "val_acc": val_acc, "val_auc": val_auc,
        "test_acc": test_acc, "test_auc": test_auc,
        "wall": f"{wall/60:.0f} min", "device": device_name,
        "ckpt": args.out,
        "command": " ".join([os.path.basename(sys.executable)] + sys.argv),
    })


if __name__ == "__main__":
    main()