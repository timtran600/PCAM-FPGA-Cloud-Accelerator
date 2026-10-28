"""
PCamCNN — FP32 reference topology for the EECE5698 FPGA accelerator.

Five 3x3 convolutions at 16/32/64/64/128 channels, max-pool after the
first four, global average pool, then a 128->2 fully-connected layer.
134,320 weights and 33.2 MMAC per 96x96x3 tile, so the whole INT8 weight
set (~131 KB) sits on-chip and the kernel never streams weights.

Hardware-driven constraints baked into this file — change them only with
a reason, because each one costs something in HLS:

1. Every conv is 3x3, stride 1, pad 1. One line-buffer/sliding-window
   implementation is reused for all five layers.
2. Convs have bias=False. BatchNorm supplies the affine term, and at
   export the BN folds into the preceding conv as conv -> bias ->
   requantize -> ReLU. A separate conv bias would just be a second
   constant to fold.
3. Order is always Conv -> BN -> ReLU. BN after ReLU does not fold.
4. Pooling is max 2x2 stride 2, which is a comparator tree in hardware.
   Average pooling would need a divide.
5. Classifier head is GAP -> FC. Flattening 6x6x128 into a dense layer
   would be 294,912 weights by itself and would force a store of the
   whole final feature map.
6. Input is x/256 with no mean/std normalization (see data.py), so the
   input quantizer scale is 2^-8 and the kernel consumes raw tile bytes.

Note for the report: the proposal described folding a mean/std
normalization into conv1. That normalization was dropped in data.py, so
conv1 folds only the BN and the input scale. The fold is simpler than
proposed, not more complex.

    python model.py          # shape walk, parameter and MAC counts
"""

import torch
import torch.nn as nn

TILE_HW = 96
IN_CH = 3
NUM_CLASSES = 2

# (out_channels, pool_after)
STAGES = [
    (16, True),
    (32, True),
    (64, True),
    (64, True),
    (128, False),
]


class ConvBNReLU(nn.Module):
    """3x3 conv (no bias) -> BN -> ReLU, optionally followed by max 2x2.

    Kept as one module so that export.py can walk the model and find
    (conv, bn) pairs positionally without guessing.
    """

    def __init__(self, in_ch, out_ch, pool):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1,
                              padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2) if pool else None

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        if self.pool is not None:
            x = self.pool(x)
        return x


class PCamCNN(nn.Module):
    """FP32 baseline. Input: float tensor [N,3,96,96] already in [0,1)."""

    def __init__(self, num_classes=NUM_CLASSES, stages=STAGES):
        super().__init__()
        blocks = []
        in_ch = IN_CH
        for out_ch, pool in stages:
            blocks.append(ConvBNReLU(in_ch, out_ch, pool))
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_ch, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.fc(x)


# --------------------------------------------------------------------------
# Analysis helpers. These numbers go straight into the report, so they are
# computed rather than quoted.
# --------------------------------------------------------------------------

def layer_table(model=None, verbose=True):
    """Walk the network, printing per-layer shape, weights, and MACs."""
    model = model or PCamCNN()
    rows = []
    h = w = TILE_HW
    in_ch = IN_CH

    for i, block in enumerate(model.features, start=1):
        out_ch = block.conv.out_channels
        weights = block.conv.weight.numel()
        macs = weights * h * w                      # pad 1, stride 1
        rows.append((f"Conv{i}", f"{h}x{w}x{out_ch}", weights, macs))
        if block.pool is not None:
            h, w = h // 2, w // 2
            rows.append((f"Pool{i}", f"{h}x{w}x{out_ch}", 0, 0))
        in_ch = out_ch

    rows.append(("GAP", f"{in_ch}", 0, in_ch * h * w))
    fc_w = model.fc.weight.numel()
    rows.append(("FC", f"{model.fc.out_features}", fc_w, fc_w))

    # BN parameters fold away at export but are trained, so count them too.
    bn_params = sum(p.numel() for m in model.modules()
                    if isinstance(m, nn.BatchNorm2d) for p in m.parameters())
    conv_fc_w = sum(r[2] for r in rows)
    total_macs = sum(r[3] for r in rows)

    if verbose:
        print(f"{'Layer':<8} {'Output':<14} {'Weights':>10} {'MACs':>12}")
        print("-" * 48)
        for name, shape, wts, macs in rows:
            print(f"{name:<8} {shape:<14} {wts:>10,} {macs:>12,}")
        print("-" * 48)
        print(f"{'Total':<8} {'':<14} {conv_fc_w:>10,} {total_macs:>12,}")
        print(f"\nconv/fc weights : {conv_fc_w:,}  ({conv_fc_w/1024:.1f} KB at INT8)")
        print(f"BN params       : {bn_params:,}  (folded away at export)")
        print(f"trainable total : {sum(p.numel() for p in model.parameters()):,}")
        print(f"MACs per tile   : {total_macs/1e6:.2f} M")
        print(f"\nAt 10,000 tiles/s -> {total_macs*1e4/1e9:.0f} GMAC/s "
              f"= {2*total_macs*1e4/1e9:.0f} GOP/s")
        print(f"Input bandwidth  -> {IN_CH*TILE_HW*TILE_HW*1e4/1e6:.0f} MB/s "
              f"(~2% of PCIe Gen3 x16): compute-bound, not I/O-bound")

    return rows, conv_fc_w, total_macs


def shape_walk(model=None):
    """Push one tile through and print the real tensor shapes.

    Catches the classic failure where a stride/pool choice leaves an odd
    spatial dimension that the 2x2 pool silently truncates.
    """
    model = (model or PCamCNN()).eval()
    x = torch.zeros(1, IN_CH, TILE_HW, TILE_HW)
    print(f"{'input':<10} {tuple(x.shape)}")
    with torch.no_grad():
        for i, block in enumerate(model.features, start=1):
            x = block(x)
            print(f"{'block'+str(i):<10} {tuple(x.shape)}")
            assert x.shape[-1] == x.shape[-2], "non-square feature map"
        g = model.gap(x).flatten(1)
        print(f"{'gap':<10} {tuple(g.shape)}")
        out = model.fc(g)
        print(f"{'logits':<10} {tuple(out.shape)}")
    assert out.shape == (1, NUM_CLASSES)
    return out


if __name__ == "__main__":
    print("=== shape walk ===")
    shape_walk()
    print("\n=== layer table ===")
    layer_table()