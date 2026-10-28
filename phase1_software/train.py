"""
FP32 training loop for PCamCNN — the accuracy baseline the INT8 model
must land within 1% of.

The FP32 number this produces is the reference for the whole project, so
every run is seeded and appends its own row to models/RESULTS.md with the
command that produced it. If you cannot reproduce a number from that
file, it does not go in the report.

Selection is by validation AUC, not accuracy. PCam is balanced, so
accuracy is meaningful, but AUC is threshold-independent and the
quantized model will shift the operating point slightly — comparing
AUC keeps the FP32/INT8 comparison honest.

    python phase1_software/train.py --epochs 30 --seed 0 --out models/fp32_best.pt
"""

import argparse
import json
import os
import platform
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

# Allow running as `python phase1_software/train.py` from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data import make_loaders, to_float01          # noqa: E402
from model import PCamCNN                          # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_MD = os.path.join(REPO_ROOT, "models", "RESULTS.md")

RESULTS_HEADER = """# Trained model results

One row per run. Never overwrite — append. A number that is not in this
table with the command that produced it does not go in the report.

| date | model | epochs | batch | lr | seed | val acc | val AUC | test acc | test AUC | wall | device | ckpt |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
"""


def set_seed(seed: int):
    """Seed every RNG the training loop touches.

    cudnn.benchmark stays on: it picks algorithms by timing, which makes
    runs non-bit-identical but roughly 10-20% faster. Bit-exactness is a
    property of the INT8 inference path, not of FP32 training, so the
    speed is worth more here.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    """Returns (mean loss, accuracy, AUC).

    AUC is computed on the softmax probability of the positive class, not
    on argmax predictions. Passing hard labels to roc_auc_score silently
    returns a plausible-looking number that is not an AUC.
    """
    model.eval()
    total_loss, n = 0.0, 0
    probs, targets = [], []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(to_float01(x))
            loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        n += y.size(0)
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu())
        targets.append(y.cpu())

    probs = torch.cat(probs).numpy()
    targets = torch.cat(targets).numpy()
    acc = ((probs >= 0.5).astype(np.int64) == targets).mean()
    auc = roc_auc_score(targets, probs)
    return total_loss / n, float(acc), float(auc)


def train_one_epoch(model, loader, device, criterion, optimizer, scaler, log_every):
    model.train()
    total_loss, n, t0 = 0.0, 0, time.time()

    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            loss = criterion(model(to_float01(x)), y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * y.size(0)
        n += y.size(0)

        if log_every and step % log_every == 0:
            rate = n / max(time.time() - t0, 1e-9)
            print(f"    step {step:>5}  loss {total_loss/n:.4f}  "
                  f"{rate:,.0f} tiles/s", flush=True)

    return total_loss / n


def append_results(row: dict):
    os.makedirs(os.path.dirname(RESULTS_MD), exist_ok=True)
    if not os.path.exists(RESULTS_MD):
        with open(RESULTS_MD, "w", encoding="utf-8") as f:
            f.write(RESULTS_HEADER)
    with open(RESULTS_MD, "a", encoding="utf-8") as f:
        f.write(
            f"| {row['date']} | PCamCNN FP32 | {row['epochs']} | {row['batch']} | "
            f"{row['lr']} | {row['seed']} | {row['val_acc']:.4f} | {row['val_auc']:.4f} | "
            f"{row['test_acc']:.4f} | {row['test_auc']:.4f} | {row['wall']} | "
            f"{row['device']} | `{row['ckpt']}` |\n"
        )
        f.write(f"\n<!-- command: {row['command']} -->\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="./data")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="models/fp32_best.pt")
    p.add_argument("--log-every", type=int, default=100,
                   help="print a within-epoch line every N steps; 0 disables")
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = (torch.cuda.get_device_name(0) if device.type == "cuda"
                   else platform.processor() or "CPU")

    print(f"device      : {device_name}")
    print(f"torch       : {torch.__version__}")
    print(f"seed        : {args.seed}")
    print(f"batch/workers: {args.batch}/{args.workers}")
    print(f"lr          : {args.lr} (AdamW, cosine to ~0 over {args.epochs} epochs)")
    print()

    train_dl, val_dl, test_dl = make_loaders(
        args.root, batch_size=args.batch, workers=args.workers, augment_train=True)

    model = PCamCNN().to(device)
    model = model.to(memory_format=torch.channels_last)
    n_params = sum(p_.numel() for p_ in model.parameters())
    print(f"parameters  : {n_params:,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    use_amp = (device.type == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_path = os.path.join(REPO_ROOT, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    best_auc, best_epoch = -1.0, -1
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_dl, device, criterion,
                                     optimizer, scaler, args.log_every)
        val_loss, val_acc, val_auc = evaluate(model, val_dl, device, criterion)
        scheduler.step()
        dt = time.time() - t0

        flag = ""
        if val_auc > best_auc:
            best_auc, best_epoch = val_auc, epoch
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_auc": val_auc,
                "val_acc": val_acc,
                "args": vars(args),
                "torch": str(torch.__version__),
                "device": device_name,
            }, out_path)
            flag = "  <- best, saved"

        print(f"epoch {epoch:>3}/{args.epochs}  "
              f"train {train_loss:.4f}  val {val_loss:.4f}  "
              f"acc {val_acc:.4f}  AUC {val_auc:.4f}  "
              f"lr {scheduler.get_last_lr()[0]:.2e}  {dt/60:.1f} min{flag}",
              flush=True)

    # Final test-set numbers come from the best checkpoint, not the last
    # epoch — otherwise "best val AUC" and the reported test score come
    # from different models.
    ckpt = torch.load(out_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    _, test_acc, test_auc = evaluate(model, test_dl, device, criterion)
    _, val_acc, val_auc = evaluate(model, val_dl, device, criterion)

    wall = time.time() - t_start
    print(f"\nbest epoch  : {best_epoch}")
    print(f"val  acc/AUC: {val_acc:.4f} / {val_auc:.4f}")
    print(f"test acc/AUC: {test_acc:.4f} / {test_auc:.4f}")
    print(f"wall        : {wall/60:.1f} min")
    print(f"checkpoint  : {out_path}")

    append_results({
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
        "seed": args.seed, "val_acc": val_acc, "val_auc": val_auc,
        "test_acc": test_acc, "test_auc": test_auc,
        "wall": f"{wall/60:.0f} min", "device": device_name,
        "ckpt": args.out,
        "command": " ".join([os.path.basename(sys.executable)] + sys.argv),
    })
    print(f"appended    : {RESULTS_MD}")


if __name__ == "__main__":
    # Required on Windows: DataLoader workers spawn rather than fork, so
    # each worker re-imports this module. Without the guard you get
    # recursive process spawning.
    main()