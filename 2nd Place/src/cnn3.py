"""
cnn3.py — 3D CNN training with an HONEST out-of-fold protocol.

THE BUG THIS REPLACES
---------------------
`cnn_train.py` checkpointed on the *validation* fold's log loss and then reported
that same fold's predictions as OOF. Selecting the epoch on the data you then
score is leakage: it makes CV look better than the model really is, and — worse
for a log-loss competition — the blend weights and the temperature are fitted on
those over-sharp probabilities. That is how the shipped model ended up with
T = 0.891, i.e. *sharpening* predictions that were already too confident for the
real test set. CV said 0.3049; the leaderboard said 0.3364.

Here the schedule is fixed (no validation-driven stopping), evaluation uses an
EMA of the weights, and the OOF prediction is simply the final model's output on
its held-out fold. Nothing about the validation fold touches training.

REGULARISATION CHOSEN FOR THIS DOMAIN
-------------------------------------
  * reference jitter  — multiply the whole SBR volume by exp(N(0, s)). The
    reference level still carries ~23% residual CV after the preprocessing3 fix,
    so this teaches the model not to trust absolute scale.
  * resolution jitter — random Gaussian blur, simulating the reconstruction /
    collimator differences that separate the acquisition sites.
  * count noise       — signal-dependent (sqrt) noise, the SPECT noise model.
  * mixup             — strong calibration regulariser, which is what a log-loss
    metric actually rewards.
  * L/R flip          — an exact symmetry of the label.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, log_loss

from nets3 import build

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def device():
    # DAT_DEVICE=cpu forces the CPU (2026-09-03: lets a slow architecture train
    # on the idle CPU while MPS is busy with another job).
    if os.environ.get("DAT_DEVICE"):
        return torch.device(os.environ["DAT_DEVICE"])
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# input representation
# --------------------------------------------------------------------------- #
_BG = None


def _bg_mask(n: int) -> np.ndarray:
    """Posterior background region of the canonical crop (occipital stand-in)."""
    global _BG
    if _BG is None or _BG.shape[0] != n:
        ax = (np.arange(n, dtype=np.float32) - (n - 1) / 2.0) * (96.0 / n)
        X, Y, Zc = np.meshgrid(ax, ax, ax, indexing="ij")
        _BG = (Y < -34.0) & (np.abs(X) < 26.0) & (np.abs(Zc) < 16.0)
    return _BG


def to_input(sbr: np.ndarray, channels: str = "log") -> np.ndarray:
    """SBR volume(s) -> network input channels.

    `selfnorm` re-normalises each crop by its OWN posterior background instead of
    the whole-head reference. The head reference still carries ~23% residual
    spread after the preprocessing3 fix, so a model built this way depends on a
    completely different normalisation and decorrelates from the rest of the
    ensemble.
    """
    s = np.clip(sbr.astype(np.float32), 0.0, None)
    if channels == "log":
        return np.log1p(s)[:, None]
    if channels == "log+spec":
        a = np.log1p(s)
        b = np.log1p(np.clip(s - 1.0, 0.0, None))
        return np.stack([a, b], axis=1)
    if channels in ("selfnorm", "log+selfnorm"):
        m = _bg_mask(s.shape[-1])
        bg = s[:, m].mean(axis=1)
        bg = np.clip(bg, 1e-3, None)[:, None, None, None]
        b = np.log1p(s / bg)
        if channels == "selfnorm":
            return b[:, None]
        return np.stack([np.log1p(s), b], axis=1)
    raise ValueError(channels)


# --------------------------------------------------------------------------- #
# augmentation (batched, on device)
# --------------------------------------------------------------------------- #
def rand_affine(x, rot_deg, shift_frac, scale, dev):
    B = x.shape[0]
    rot = np.deg2rad(np.random.uniform(-rot_deg, rot_deg, size=(B, 3)))
    sc = np.random.uniform(1 - scale, 1 + scale, size=B)
    t = np.random.uniform(-shift_frac, shift_frac, size=(B, 3))
    cos, sin = np.cos(rot), np.sin(rot)
    R = np.zeros((B, 3, 3))
    for b in range(B):
        rx, ry, rz = rot[b]
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R[b] = (Rx @ Ry @ Rz) * sc[b]
    theta = torch.zeros(B, 3, 4, device=dev)
    theta[:, :, :3] = torch.tensor(R, dtype=torch.float32, device=dev)
    theta[:, :, 3] = torch.tensor(t, dtype=torch.float32, device=dev)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="border")


def _blur(x, sigma_vox, dev):
    """Separable Gaussian blur with one shared sigma per batch."""
    if sigma_vox <= 0.05:
        return x
    r = max(1, int(math.ceil(2.5 * sigma_vox)))
    k = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    k = torch.exp(-0.5 * (k / sigma_vox) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    for ax in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + ax] = -1
        w = k.view(shape).expand(c, 1, *[k.numel() if i == ax else 1 for i in range(3)])
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - ax)] = pad[2 * (2 - ax) + 1] = r
        x = F.conv3d(F.pad(x, pad, mode="replicate"), w, groups=c)
    return x


def _blur_persample(x, sigmas, dev):
    """Per-sample separable Gaussian blur as ONE grouped conv per axis: the
    batch is folded into the channel dim (groups = B*C) and each sample gets its
    own kernel. Cost is independent of how many distinct sigmas are drawn."""
    B, C = x.shape[0], x.shape[1]
    sig = torch.as_tensor(np.maximum(sigmas, 1e-3), dtype=torch.float32, device=dev)
    r = max(1, int(math.ceil(2.5 * float(sigmas.max()))))
    t = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    k = torch.exp(-0.5 * (t[None, :] / sig[:, None]) ** 2)          # (B, K)
    k = k / k.sum(1, keepdim=True)
    # samples with sigma ~ 0 get a delta kernel
    delta = torch.zeros_like(k); delta[:, r] = 1.0
    k = torch.where(torch.as_tensor(sigmas <= 0.05, device=dev)[:, None], delta, k)
    k = k.repeat_interleave(C, dim=0)                                 # (B*C, K)
    y = x.reshape(1, B * C, *x.shape[2:])
    for ax in range(3):
        shape = [B * C, 1, 1, 1, 1]
        shape[2 + ax] = -1
        w = k.view(shape)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - ax)] = pad[2 * (2 - ax) + 1] = r
        y = F.conv3d(F.pad(y, pad, mode="replicate"), w, groups=B * C)
    return y.reshape(B, C, *x.shape[2:])


ROT_DEG = 7.0   # max rotation (deg) per axis in rand_affine; --rot-deg overrides (2026-09-03 pitch probe)


def augment(x, dev, strength=1.0, ref_jitter=0.12, mode="batch"):
    """`ref_jitter` is the sd of the multiplicative gain augmentation.

    `mode="sample"` (v12 probe): blur sigma, gain and count-noise amplitude are
    drawn PER SAMPLE instead of once per batch, so a batch of 32 sees 32
    different acquisition conditions rather than one. The default reproduces
    the shipped recipe exactly.

    It was set to teach the network not to trust absolute scale, because the
    reference level still carries ~23% residual spread. But RESEARCH_NOTES §29
    established that for THIS label — abnormal DaT scan, not Parkinson's — the
    dominant discriminator is exactly specific-binding MAGNITUDE, and a linear
    model on 18 radial magnitude numbers (0.2460) beats this network (0.2611)
    while seeing strictly less. So the augmentation may be destroying the main
    signal. Exposed as a flag so that is a measurement rather than a guess.
    """
    if np.random.rand() < 0.5:                                    # exact symmetry
        x = torch.flip(x, dims=[2])
    if np.random.rand() < 0.8:
        x = rand_affine(x, ROT_DEG * strength, 0.05 * strength, 0.07 * strength, dev)
    if mode == "sample":
        B = x.shape[0]
        sig = np.where(np.random.rand(B) < 0.5,
                       np.random.uniform(0.0, 1.1 * strength, size=B), 0.0).astype(np.float32)
        if sig.max() > 0.05:
            x = _blur_persample(x, sig, dev)
        if ref_jitter > 0:
            g = np.where(np.random.rand(B) < 0.7,
                         np.exp(np.random.normal(0.0, ref_jitter * strength, size=B)), 1.0)
            x = x * torch.as_tensor(g, dtype=x.dtype, device=dev).view(B, 1, 1, 1, 1)
        amp = np.where(np.random.rand(B) < 0.5, np.random.uniform(0.0, 0.1 * strength, size=B), 0.0)
        amp_t = torch.as_tensor(amp, dtype=x.dtype, device=dev).view(B, 1, 1, 1, 1)
        x = x + torch.randn_like(x) * amp_t * x.clamp(min=0).sqrt()
        amp2 = np.where(np.random.rand(B) < 0.3, np.random.uniform(0.0, 0.04 * strength, size=B), 0.0)
        x = x + torch.randn_like(x) * torch.as_tensor(amp2, dtype=x.dtype, device=dev).view(B, 1, 1, 1, 1)
        return x
    if np.random.rand() < 0.5:                                    # resolution jitter
        x = _blur(x, float(np.random.uniform(0.0, 1.1 * strength)), dev)
    if ref_jitter > 0 and np.random.rand() < 0.7:                 # reference jitter
        x = x * float(np.exp(np.random.normal(0.0, ref_jitter * strength)))
    if np.random.rand() < 0.5:                                    # count noise
        x = x + torch.randn_like(x) * (0.05 * strength) * x.clamp(min=0).sqrt()
    if np.random.rand() < 0.3:
        x = x + torch.randn_like(x) * 0.02 * strength
    return x


# --------------------------------------------------------------------------- #
class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone().float()

    def state(self, model):
        ref = model.state_dict()
        return {k: self.shadow[k].to(ref[k].dtype) for k in ref}


@torch.no_grad()
def predict(net, X, bs=128, tta=True, off=None):
    """`off` is a per-row logit offset: the model is then a BOOSTER on top of
    whatever produced it (see make_offsets.py), not a standalone classifier."""
    net.eval()
    if len(X) == 0:                 # full-data training has no held-out rows
        return np.zeros(0)
    out = []
    for j in range(0, len(X), bs):
        xb = X[j:j + bs]
        ob = 0.0 if off is None else off[j:j + bs]
        z = torch.sigmoid(net(xb) + ob)
        if tta:
            z = 0.5 * (z + torch.sigmoid(net(torch.flip(xb, dims=[2])) + ob))
        out.append(z.float().cpu().numpy())
    return np.concatenate(out)


# Auxiliary regression targets: clinical ROI measurements computed from the SAME
# crop the network sees, so this is supervision on a deterministic function of the
# input and carries no label information. It tells the encoder which structure
# matters — a prior 1362 binary labels cannot supply on their own.
AUX_COLS = ["ppu_min", "ppu_mean", "put_min", "cau_max", "pc_ratio_min",
            "stri_asym", "h_spec_worse", "h_spec_better", "h_ap_extent_worse",
            "blob_elong"]


def run_fold(x, y, tr, va, dev, args, seed, tag, off=None, aux=None, fold_id=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu = float(x[tr].mean())
    sd = float(x[tr].std() + 1e-6)
    Xtr = torch.from_numpy((x[tr] - mu) / sd).float().to(dev)
    Xva = torch.from_numpy((x[va] - mu) / sd).float().to(dev)
    ytr = torch.from_numpy(y[tr]).float().to(dev)
    otr = None if off is None else torch.from_numpy(off[tr]).float().to(dev)
    ova = None if off is None else torch.from_numpy(off[va]).float().to(dev)

    aux_dim = 0 if aux is None else aux.shape[1]
    if aux is not None:
        mu_a = aux[tr].mean(0)
        sd_a = aux[tr].std(0) + 1e-6
        Atr = torch.from_numpy(((aux[tr] - mu_a) / sd_a).astype(np.float32)).to(dev)
    net = build(args.arch, in_ch=x.shape[1], drop=args.drop,
                **({"aux_dim": aux_dim} if aux_dim else {})).to(dev)
    if getattr(args, "init", None):
        # encoder weights from self-supervised pretraining on THIS fold's training
        # rows only (ssl_pretrain.py). The head stays random.
        ck = torch.load(OUT / f"{args.init}_fold{fold_id}.pt", map_location="cpu",
                        weights_only=False)
        missing = net.enc.load_state_dict(ck["enc"], strict=False)
        if missing.missing_keys:
            raise SystemExit(f"SSL init mismatch: {missing.missing_keys[:4]}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    ema = EMA(net, decay=args.ema)
    n = len(tr)
    steps_per_epoch = max(1, math.ceil(n / args.bs))
    total = args.epochs * steps_per_epoch
    warm = args.warmup * steps_per_epoch
    step = 0
    ls = args.label_smooth

    for ep in range(args.epochs):
        net.train()
        perm = np.random.permutation(n)
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            xb = augment(Xtr[idx], dev, args.aug, args.ref_jitter,
                         mode=getattr(args, "aug_mode", "batch"))
            tb = ytr[idx]
            ob = None if otr is None else otr[idx]
            ab = None if aux_dim == 0 else Atr[idx]
            if ls > 0:
                tb = tb * (1 - ls) + 0.5 * ls
            if args.mixup > 0 and np.random.rand() < getattr(args, "mixup_p", 0.5):
                lam = float(np.random.beta(args.mixup, args.mixup))
                lam = max(lam, 1 - lam)
                p = torch.randperm(len(idx), device=dev)
                xb = lam * xb + (1 - lam) * xb[p]
                tb = lam * tb + (1 - lam) * tb[p]
                if ob is not None:                  # mix the offset with the image
                    ob = lam * ob + (1 - lam) * ob[p]
                if ab is not None:                  # and the auxiliary targets
                    ab = lam * ab + (1 - lam) * ab[p]
            lr = (step + 1) / max(warm, 1) if step < warm else \
                0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
            for g in opt.param_groups:
                g["lr"] = args.lr * max(lr, 1e-3)
            opt.zero_grad()
            z = net(xb)
            loss = F.binary_cross_entropy_with_logits(z if ob is None else z + ob, tb)
            if ab is not None:
                loss = loss + args.aux_weight * F.mse_loss(net.aux_out(xb), ab)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            ema.update(net)
            step += 1
        if args.verbose and (ep % 10 == 0 or ep == args.epochs - 1):
            ev = build(args.arch, in_ch=x.shape[1], drop=args.drop,
                       **({"aux_dim": aux_dim} if aux_dim else {})).to(dev)
            ev.load_state_dict(ema.state(net))
            pv = predict(ev, Xva, off=ova)
            print(f"  [{tag}] ep{ep:03d} val_ll {log_loss(y[va], np.clip(pv,1e-6,1-1e-6)):.4f} "
                  f"auc {roc_auc_score(y[va], pv):.4f}", flush=True)

    final = build(args.arch, in_ch=x.shape[1], drop=args.drop,
                  **({"aux_dim": aux_dim} if aux_dim else {})).to(dev)
    final.load_state_dict(ema.state(net))
    pv = predict(final, Xva, off=ova)
    # The auxiliary head is a TRAINING-time regulariser only. Strip it, so the
    # shipped checkpoint is structurally identical to any other and the inference
    # path needs no knowledge of it at all.
    state = {k: v.cpu() for k, v in final.state_dict().items()
             if not k.startswith("aux.")}
    return pv, state, (mu, sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v3a")
    ap.add_argument("--arch", default="res")
    ap.add_argument("--crops", default="crops_v3.npz")
    ap.add_argument("--crops2", default=None,
                    help="second cache, added as extra input channels at the same "
                         "grid but a DIFFERENT field of view — a foveated input. "
                         "RESEARCH_NOTES 5b's negative result about second "
                         "channels was for log(specific), which is a "
                         "deterministic function of channel 1 and so adds "
                         "nothing; a different FOV is not redundant.")
    ap.add_argument("--key", default="aligned", choices=["aligned", "raw", "aligned_reg1"],
                    help="aligned_reg1 = registered crop resampled once from native (v13, §73p); "
                         "the checkpoint records it so main.py builds the matching input")
    ap.add_argument("--channels", default="log")
    ap.add_argument("--size", type=int, default=48, help="resize crops to this cube")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=3e-2)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--mixup", type=float, default=0.4)
    ap.add_argument("--label-smooth", type=float, default=0.02)
    ap.add_argument("--aug", type=float, default=1.0)
    ap.add_argument("--rot-deg", type=float, default=7.0, help="max affine rotation per axis (deg)")
    ap.add_argument("--aug-mode", default="batch", choices=["batch", "sample"],
                    help="'sample' = per-sample blur/gain/noise draws (v12 probe)")
    ap.add_argument("--mixup-p", type=float, default=0.5,
                    help="probability a batch is mixed (shipped recipe: 0.5)")
    ap.add_argument("--ref-jitter", type=float, default=0.12)
    ap.add_argument("--init", default=None,
                    help="ssl_<tag> prefix: initialise the encoder from "
                         "self-supervised pretraining on this fold's "
                         "training rows (ssl_pretrain.py)")
    ap.add_argument("--offset", default=None,
                    help="offsets_<name>.npz from make_offsets.py: train this "
                         "network as a BOOSTER on top of that model's logit "
                         "instead of as a standalone classifier")
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--only-fold", type=int, default=-1)
    ap.add_argument("--full", action="store_true",
                    help="train ONE model on 100%% of the rows instead of the "
                         "fold models. The schedule has no validation-dependent "
                         "step, so this is legitimate; it sees 25%% more data than "
                         "any fold model and costs a fifth of a 5-fold run. It "
                         "cannot contribute to the OOF, so build3 only ships it "
                         "under --use-full.")
    ap.add_argument("--full-tag", default="full0")
    ap.add_argument("--aux", default=None,
                    help="feature table to use as AUXILIARY regression targets "
                         "during training (e.g. features_v6_c.csv). The head is "
                         "stripped before saving, so nothing about inference "
                         "changes.")
    ap.add_argument("--aux-weight", type=float, default=0.1)
    ap.add_argument("--drop-train-folds", type=int, nargs="*", default=[],
                    help="withhold these folds from TRAINING as well as from "
                         "scoring. Used only to measure the learning-curve slope: "
                         "training on 3 folds vs 4 and scoring the same held-out "
                         "fold says how much a 100%%-data model would gain, which "
                         "is otherwise unmeasurable from a 5-fold OOF.")
    ap.add_argument("--folds", default="folds.csv")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()
    global ROT_DEG
    ROT_DEG = float(args.rot_deg)

    dev = device()
    z = np.load(OUT / args.crops, allow_pickle=True)
    sbr = z[args.key].astype(np.float32)
    y = z["y"].astype(np.float32)
    uids = z["uids"].astype(str)
    # crop extent is no longer fixed at 96 mm (cache_fov.py), so record which box
    # this cache holds; inference has to rebuild the same one
    if "geom" in z.files:
        crop_mm, out_vox = float(z["geom"][0]), int(z["geom"][1])
    else:
        from preprocessing3 import CROP_MM, OUT_VOX
        crop_mm, out_vox = float(CROP_MM), int(OUT_VOX)
    x = to_input(sbr, args.channels)
    if args.size != x.shape[-1]:
        t = torch.from_numpy(x)
        x = F.interpolate(t, size=(args.size,) * 3, mode="trilinear",
                          align_corners=False).numpy()
    crops2_geom = None
    if args.crops2:
        z2 = np.load(OUT / args.crops2, allow_pickle=True)
        if not np.array_equal(z2["uids"].astype(str), uids):
            raise SystemExit("crops2 is in a different row order — refusing to train")
        x2 = to_input(z2[args.key].astype(np.float32), args.channels)
        if args.size != x2.shape[-1]:
            x2 = F.interpolate(torch.from_numpy(x2), size=(args.size,) * 3,
                               mode="trilinear", align_corners=False).numpy()
        x = np.concatenate([x, x2], axis=1)
        crops2_geom = ([float(z2["geom"][0]), int(z2["geom"][1])]
                       if "geom" in z2.files else [96.0, 48])
        del x2, z2
    folds = pd.read_csv(OUT / args.folds).set_index("uid").loc[uids].reset_index()
    fold = folds["fold"].values
    auxmat = None
    if args.aux:
        fa = pd.read_csv(OUT / args.aux).set_index("uid").loc[uids]
        auxmat = fa[AUX_COLS].to_numpy(np.float64)
        auxmat = np.where(np.isfinite(auxmat), auxmat, 0.0)
        print(f"[{args.tag}] auxiliary targets: {len(AUX_COLS)} features, "
              f"weight {args.aux_weight}", flush=True)
    offmat = None
    if args.offset:
        zo = np.load(OUT / args.offset, allow_pickle=True)
        order = pd.Series(range(len(zo["uids"])), index=zo["uids"].astype(str))
        offmat = np.log(zo["off"] / (1.0 - zo["off"]))[order.loc[uids].to_numpy()]
    print(f"[{args.tag}] dev={dev} arch={args.arch} x{x.shape} key={args.key} "
          f"size={args.size} epochs={args.epochs}", flush=True)

    oof = np.zeros(len(y))
    t0 = time.time()
    if args.full:
        allidx = np.arange(len(y))
        _, state, (mu, sd) = run_fold(x, y, allidx, allidx[:0], dev, args,
                                      args.seed_base + 99, f"{args.tag}full",
                                      None if offmat is None else offmat[:, 0])
        torch.save({"state": state, "mu": mu, "sd": sd, "arch": args.arch,
                    "in_ch": x.shape[1], "size": args.size, "key": args.key,
                    "channels": args.channels, "drop": args.drop,
                    "crops": args.crops, "offset": args.offset,
                    "crop_mm": crop_mm, "out_vox": out_vox,
                    "crops2": args.crops2, "crops2_geom": crops2_geom},
                   OUT / f"{args.tag}_{args.full_tag}.pt")
        print(f"[{args.tag}] full-data model -> {args.tag}_{args.full_tag}.pt "
              f"({time.time()-t0:.0f}s)", flush=True)
        return
    fold_ids = sorted(np.unique(fold))
    if args.only_fold >= 0:
        fold_ids = [args.only_fold]
    for k in fold_ids:
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
        if args.drop_train_folds:
            tr = tr[~np.isin(fold[tr], args.drop_train_folds)]
        pv, state, (mu, sd) = run_fold(x, y, tr, va, dev, args, args.seed_base + k,
                                       f"{args.tag}f{k}",
                                       None if offmat is None else offmat[:, k],
                                       auxmat, k)
        oof[va] = pv
        if not args.no_save:
            torch.save({"state": state, "mu": mu, "sd": sd, "arch": args.arch,
                        "in_ch": x.shape[1], "size": args.size, "key": args.key,
                        "channels": args.channels, "drop": args.drop,
                        # which cache this was trained from: crop extent is no
                        # longer fixed at 96 mm, so the verification gate has to
                        # know which reference tensor is the right one
                        "crops": args.crops, "offset": args.offset,
                        "crop_mm": crop_mm, "out_vox": out_vox,
                        "crops2": args.crops2, "crops2_geom": crops2_geom},
                       OUT / f"{args.tag}_fold{k}.pt")
        print(f"[{args.tag}] fold {k}: ll {log_loss(y[va], np.clip(pv,1e-6,1-1e-6)):.4f} "
              f"auc {roc_auc_score(y[va], pv):.4f}  ({time.time()-t0:.0f}s)", flush=True)

    if args.only_fold >= 0:
        return
    oof = np.clip(oof, 1e-6, 1 - 1e-6)
    LL, AUC = log_loss(y, oof), roc_auc_score(y, oof)
    print(f"\n[{args.tag}] OOF logloss {LL:.4f} auroc {AUC:.4f} ({time.time()-t0:.0f}s)", flush=True)
    if not args.no_save:
        pd.DataFrame({"uid": uids, "y": y.astype(int), "fold": fold, args.tag: oof}).to_csv(
            OUT / f"oof_{args.tag}.csv", index=False)
        (OUT / f"cv_{args.tag}.json").write_text(json.dumps(
            {"logloss": LL, "auroc": AUC, "args": vars(args)}, indent=2, default=str))


if __name__ == "__main__":
    main()
