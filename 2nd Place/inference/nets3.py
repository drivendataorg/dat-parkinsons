"""
nets3.py — 3D architectures for the DaT striatal crops.

Shared by training and inference, so the submission container rebuilds a model
from `build(arch, **kw)` + a state_dict without importing any training code.

Three families, deliberately different so their errors decorrelate in the blend:
  * `res`   — residual GroupNorm CNN (the v2 workhorse, re-tuned and leaner).
  * `se`    — squeeze-excitation residual net with a wider head.
  * `mil`   — hemisphere multiple-instance model: a shared encoder scores the
              left and right striatum separately and the scan probability is the
              noisy-OR of the two. That matches the clinical fact that PD starts
              unilaterally, and it doubles the effective sample count seen by the
              encoder (1362 scans -> 2724 hemispheres).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int, groups: int = 8) -> nn.GroupNorm:
    g = groups
    while g > 1 and c % g:
        g //= 2
    return nn.GroupNorm(g, c)


class SE(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        h = max(c // r, 4)
        self.fc1 = nn.Conv3d(c, h, 1)
        self.fc2 = nn.Conv3d(h, c, 1)

    def forward(self, x):
        s = F.adaptive_avg_pool3d(x, 1)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s))))
        return x * s


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, pool: bool = True, se: bool = False):
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1, bias=False)
        self.n1 = _gn(cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1, bias=False)
        self.n2 = _gn(cout)
        self.se = SE(cout) if se else nn.Identity()
        self.skip = nn.Conv3d(cin, cout, 1, bias=False) if cin != cout else nn.Identity()
        self.pool = nn.MaxPool3d(2) if pool else nn.Identity()

    def forward(self, x):
        h = F.relu(self.n1(self.conv1(x)))
        h = self.se(self.n2(self.conv2(h)))
        return self.pool(F.relu(h + self.skip(x)))


class Encoder(nn.Module):
    def __init__(self, in_ch=1, stem=24, widths=(32, 64, 128), se=False, stem_stride=2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, stem, 3, stride=stem_stride, padding=1, bias=False),
            _gn(stem), nn.ReLU(inplace=True))
        blocks, c = [], stem
        for w in widths:
            blocks.append(Block(c, w, pool=True, se=se))
            c = w
        self.blocks = nn.Sequential(*blocks)
        self.out_ch = c

    def forward(self, x):
        return self.blocks(self.stem(x))


class VolNet(nn.Module):
    """Whole-crop classifier. Concatenated avg+max pooling keeps both the mean
    uptake level and the peak, which carry different clinical information."""

    def __init__(self, in_ch=1, stem=24, widths=(32, 64, 128), drop=0.3, se=False,
                 stem_stride=2):
        super().__init__()
        self.enc = Encoder(in_ch, stem, widths, se, stem_stride)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(2 * self.enc.out_ch, 1)

    def forward(self, x):
        h = self.enc(x).flatten(2)      # (B,C,V); flatten+amax stays on the GPU,
        a, m = h.mean(2), h.amax(2)     # adaptive_max_pool3d falls back to CPU on MPS
        return self.fc(self.drop(torch.cat([a, m], 1))).squeeze(1)


class HemiMIL(nn.Module):
    """Score each hemisphere with a shared encoder, combine by noisy-OR.

    The crop is split at the mid-sagittal plane (axis -3 after the canonical
    alignment) and the left half is mirrored so both hemispheres reach the
    encoder in the same handedness. `forward` returns the scan-level logit;
    `hemi_logits` exposes the per-side scores for analysis.
    """

    def __init__(self, in_ch=1, stem=24, widths=(32, 64, 128), drop=0.3, se=True,
                 stem_stride=2, aux_dim=0):
        super().__init__()
        self.enc = Encoder(in_ch, stem, widths, se, stem_stride)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(2 * self.enc.out_ch, 1)
        # Auxiliary regression head: predicts a handful of the clinical ROI
        # measurements from the same embedding. It exists only during training —
        # a shape-supervision signal that tells the encoder WHICH structure to
        # look at, which is the sort of prior a 1362-sample dataset cannot supply
        # from labels alone. aux_dim=0 adds no parameters, so an existing
        # checkpoint's state_dict is unchanged.
        self.aux = nn.Linear(2 * self.enc.out_ch, aux_dim) if aux_dim else None

    def _embed(self, half):
        h = self.enc(half).flatten(2)
        return torch.cat([h.mean(2), h.amax(2)], 1)

    def _score(self, half):
        return self.fc(self.drop(self._embed(half))).squeeze(1)

    def aux_out(self, x):
        """Auxiliary prediction from the mean of the two hemisphere embeddings."""
        n = x.shape[-3]
        h = n // 2
        e = 0.5 * (self._embed(torch.flip(x[..., :h, :, :], dims=[-3]))
                   + self._embed(x[..., n - h:, :, :]))
        return self.aux(self.drop(e))

    def hemi_logits(self, x):
        """Per-hemisphere logits. Each is the log-odds that THAT hemisphere is
        intact; the head learns the sign, and `forward` combines them so that the
        scan reads abnormal as soon as either side is not intact."""
        n = x.shape[-3]
        h = n // 2
        left = torch.flip(x[..., :h, :, :], dims=[-3])      # mirror to right-handed
        right = x[..., n - h:, :, :]
        return self._score(left), self._score(right)

    def forward(self, x):
        zl, zr = self.hemi_logits(x)
        # Noisy-OR over the two hemispheres, in log space for stability.
        # With qL = sigmoid(zl) = P(left intact),  P(abnormal) = 1 - qL*qR.
        #   log(qL*qR) = -softplus(-zl) - softplus(-zr)      [log sigmoid(z) = -softplus(-z)]
        neg = -(F.softplus(-zl) + F.softplus(-zr))          # log P(both intact)
        pos = torch.log1p(-torch.exp(neg.clamp(max=-1e-6)))  # log P(abnormal)
        return pos - neg                                     # logit P(abnormal)


class HemiCropMIL(nn.Module):
    """Multiple-instance model over PRE-CROPPED hemispheres.

    `HemiMIL` above crops the whole head at the midpoint of the two striata and
    only then splits at the mid-sagittal plane, so half of every forward pass is
    discarded and — in exactly the asymmetric cases that carry the disease
    signature — *both* striata sit off-centre. Here the instances arrive already
    sampled by `preprocessing3.hemi_crops`: one tight 40x60x40 mm box per
    striatum at 1.25 mm, both in the same handedness. Same voxel budget, 2.5x the
    linear resolution on target, and each side properly centred.

    Input is (B, K*C, D, H, W), channels instance-major: K boxes of C channels.
    Aggregation is the same noisy-OR as `HemiMIL` — the scan reads abnormal as
    soon as either side is not intact, which is what unilateral onset looks like.
    """

    def __init__(self, in_ch=2, n_inst=2, stem=24, widths=(32, 64, 128), drop=0.3,
                 se=True, stem_stride=2):
        super().__init__()
        if in_ch % n_inst:
            raise ValueError(f"in_ch {in_ch} not divisible by n_inst {n_inst}")
        self.n_inst = n_inst
        self.enc = Encoder(in_ch // n_inst, stem, widths, se, stem_stride)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(2 * self.enc.out_ch, 1)

    def inst_logits(self, x):
        b, c = x.shape[0], x.shape[1] // self.n_inst
        h = self.enc(x.reshape(b * self.n_inst, c, *x.shape[2:])).flatten(2)
        a, m = h.mean(2), h.amax(2)
        z = self.fc(self.drop(torch.cat([a, m], 1))).squeeze(1)
        return z.reshape(b, self.n_inst)

    def forward(self, x):
        z = self.inst_logits(x)                              # log-odds "intact"
        neg = -F.softplus(-z).sum(1)                         # log P(all intact)
        pos = torch.log1p(-torch.exp(neg.clamp(max=-1e-6)))  # log P(abnormal)
        return pos - neg


class HemiPairNet(nn.Module):
    """Same per-hemisphere encoder as `HemiCropMIL`, different aggregation.

    Noisy-OR can only combine two SCALARS, so everything the encoder learned
    about a hemisphere has to be squeezed through one number before the two
    sides ever meet. Here the two embeddings are pooled elementwise by max and
    min and read by a small head. That is still exactly permutation-invariant —
    so the instance-swap symmetry holds — but the head can use "worse side looks
    like X while the better side looks like Y", which is precisely the
    comparison the worse-side-first hemisphere features are worth 0.011 for in
    the linear model.
    """

    def __init__(self, in_ch=2, n_inst=2, stem=24, widths=(32, 64, 128), drop=0.3,
                 se=True, stem_stride=2, hidden=64):
        super().__init__()
        if in_ch % n_inst:
            raise ValueError(f"in_ch {in_ch} not divisible by n_inst {n_inst}")
        self.n_inst = n_inst
        self.enc = Encoder(in_ch // n_inst, stem, widths, se, stem_stride)
        e = 2 * self.enc.out_ch
        self.drop = nn.Dropout(drop)
        self.head = nn.Sequential(nn.Linear(2 * e, hidden), nn.ReLU(inplace=True),
                                  nn.Dropout(drop), nn.Linear(hidden, 1))

    def forward(self, x):
        b, c = x.shape[0], x.shape[1] // self.n_inst
        h = self.enc(x.reshape(b * self.n_inst, c, *x.shape[2:])).flatten(2)
        e = torch.cat([h.mean(2), h.amax(2)], 1).reshape(b, self.n_inst, -1)
        pair = torch.cat([e.amax(1), e.amin(1)], 1)      # symmetric in the sides
        return self.head(self.drop(pair)).squeeze(1)


class MedResEncoder(nn.Module):
    """MONAI 3D ResNet in the MedicalNet layout (Tencent Med3D, MIT licence),
    truncated before the classifier so it drops into `HemiMIL._embed` exactly
    like `Encoder` (returns a (B, 512, d, h, w) feature map). `pretrained` is a
    path to a MedicalNet state_dict (keys prefixed `module.`); None = scratch.
    Only the encoder init differs from the shipped MIL recipe — the head, the
    noisy-OR and every training knob are HemiMIL's."""

    def __init__(self, in_ch=1, depth=10, pretrained=None, stem_stride=1, max_pool=True):
        super().__init__()
        from monai.networks.nets import resnet as _mr
        bias_ds, short = _mr.get_medicalnet_pretrained_resnet_args(depth)
        fn = {10: _mr.resnet10, 18: _mr.resnet18, 34: _mr.resnet34}[depth]
        # stem_stride 2 runs every layer at 1/8 of the voxels (the pretrained
        # weights are stride-agnostic); MPS is ~8x too slow at stride 1.
        self.net = fn(pretrained=False, spatial_dims=3, n_input_channels=in_ch,
                      feed_forward=False, shortcut_type=short,
                      bias_downsample=bias_ds, num_classes=1,
                      conv1_t_stride=stem_stride, no_max_pool=not max_pool)
        self.out_ch = 512
        if pretrained:
            sd = torch.load(pretrained, map_location="cpu", weights_only=False)
            sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
            res = self.net.load_state_dict(sd, strict=False)
            bad = [k for k in res.missing_keys if not k.startswith("fc")]
            if bad:
                raise RuntimeError(f"MedicalNet init mismatch: {bad[:4]}")

    def forward(self, x):
        n = self.net
        x = n.act(n.bn1(n.conv1(x)))
        if not n.no_max_pool:
            x = n.maxpool(x)
        return n.layer4(n.layer3(n.layer2(n.layer1(x))))


class HemiMILMed(HemiMIL):
    """HemiMIL with a MedicalNet ResNet encoder (see MedResEncoder)."""

    def __init__(self, in_ch=1, drop=0.3, depth=10, pretrained=None, aux_dim=0,
                 stem_stride=1, max_pool=True, **_ignored):
        nn.Module.__init__(self)
        self.enc = MedResEncoder(in_ch, depth, pretrained, stem_stride, max_pool)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(2 * self.enc.out_ch, 1)
        self.aux = nn.Linear(2 * self.enc.out_ch, aux_dim) if aux_dim else None


ARCHS = {
    "res":  dict(cls="VolNet",  kw=dict(stem=24, widths=(32, 64, 128), se=False)),
    "resw": dict(cls="VolNet",  kw=dict(stem=32, widths=(48, 96, 160), se=False)),
    "se":   dict(cls="VolNet",  kw=dict(stem=24, widths=(32, 64, 128), se=True)),
    "sed":  dict(cls="VolNet",  kw=dict(stem=24, widths=(32, 48, 96, 160), se=True)),
    "mil":  dict(cls="HemiMIL", kw=dict(stem=24, widths=(32, 64, 128), se=True)),
    "milw": dict(cls="HemiMIL", kw=dict(stem=32, widths=(48, 96, 160), se=True)),
    # per-hemisphere-crop MIL (v4 input); stem_stride 1 because the boxes are
    # already tight — striding the stem at 1.25 mm would throw the resolution
    # gain away before the first block sees it.
    "hmil":  dict(cls="HemiCropMIL",
                  kw=dict(stem=24, widths=(32, 64, 128), se=True, stem_stride=1)),
    "hmilw": dict(cls="HemiCropMIL",
                  kw=dict(stem=32, widths=(48, 96, 160), se=True, stem_stride=1)),
    "hmils": dict(cls="HemiCropMIL",
                  kw=dict(stem=24, widths=(32, 64, 128), se=True, stem_stride=2)),
    "hmild": dict(cls="HemiCropMIL",
                  kw=dict(stem=24, widths=(32, 48, 96, 160), se=True, stem_stride=1)),
    "hpair": dict(cls="HemiPairNet",
                  kw=dict(stem=24, widths=(32, 64, 128), se=True, stem_stride=2)),
    "hpairw": dict(cls="HemiPairNet",
                   kw=dict(stem=32, widths=(48, 96, 160), se=True, stem_stride=2)),
    # MedicalNet (Med3D) 3D-ResNet encoders inside the HemiMIL head. `pretrained`
    # is resolved against <project>/external/ at build time and silently becomes
    # None when the file is absent (inference: the trained state_dict is loaded
    # afterwards, so the init file is never shipped).
    "med10":  dict(cls="HemiMILMed", kw=dict(depth=10, pretrained="medicalnet/resnet_10_23dataset.pth")),
    "med10s": dict(cls="HemiMILMed", kw=dict(depth=10, pretrained=None)),
    "med18":  dict(cls="HemiMILMed", kw=dict(depth=18, pretrained="medicalnet/resnet_18_23dataset.pth")),
    "med18s": dict(cls="HemiMILMed", kw=dict(depth=18, pretrained=None)),
    # stride-2 stem variants (8x cheaper; the ones that are trainable on MPS)
    "med10p2":  dict(cls="HemiMILMed", kw=dict(depth=10, pretrained="medicalnet/resnet_10_23dataset.pth", stem_stride=2, max_pool=True)),
    "med10s2":  dict(cls="HemiMILMed", kw=dict(depth=10, pretrained=None, stem_stride=2, max_pool=True)),
    "med10p2n": dict(cls="HemiMILMed", kw=dict(depth=10, pretrained="medicalnet/resnet_10_23dataset.pth", stem_stride=2, max_pool=False)),
    "med18p2":  dict(cls="HemiMILMed", kw=dict(depth=18, pretrained="medicalnet/resnet_18_23dataset.pth", stem_stride=2, max_pool=True)),
}


def build(arch: str, in_ch: int = 1, drop: float = 0.3, **over) -> nn.Module:
    spec = ARCHS[arch]
    kw = dict(spec["kw"])
    kw.update(over)
    cls = {"VolNet": VolNet, "HemiMIL": HemiMIL, "HemiMILMed": HemiMILMed,
           "HemiCropMIL": HemiCropMIL, "HemiPairNet": HemiPairNet}[spec["cls"]]
    if cls is HemiMILMed:
        import os
        from pathlib import Path
        p = kw.get("pretrained")
        if p and os.environ.get("DAT_NO_PRETRAINED") != "1":
            cand = Path(__file__).resolve().parent.parent / "external" / p
            kw["pretrained"] = str(cand) if cand.exists() else None
        else:
            kw["pretrained"] = None
        kw.pop("n_inst", None)
        kw.pop("stem", None); kw.pop("widths", None); kw.pop("se", None)
    if cls in (VolNet, HemiMIL):
        kw.pop("n_inst", None)          # whole-crop models take no instance dim
    return cls(in_ch=in_ch, drop=drop, **kw)
