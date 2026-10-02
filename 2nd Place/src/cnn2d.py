"""
cnn2d.py — ImageNet-pretrained 2D backbone on the clinical axial read (v12).

WHY THIS AND NOT tl25d.py
-------------------------
tl25d fed 28 slice-triplets per scan through attention-MIL at batch 6, and its
validation curve oscillated 0.41 <-> 1.9 between consecutive epochs while train
loss fell monotonically — a recipe failure, not a negative result. It also
checkpointed on the fold it reported (the leak cnn3's header exists to prevent).

Here the input is ONE image per scan: three overlapping axial slabs through the
striatum (the summed axial slab IS the clinical read), the backbone sees a
normal batch, the schedule is fixed (no validation-driven stopping), evaluation
uses the EMA weights at the END of training, and the OOF row is that model's
output on its held-out fold. Same honesty contract as cnn3.

Axes of the cached crop: (L->R, P->A, I->S). Image dims are (L->R, P->A), so a
flip along image dim 1 is the exact left-right symmetry of the label.

Container notes: weights are loaded with timm `pretrained=False` from a bundled
state_dict (no internet at inference). The checkpoint stores the backbone name,
the slab definition and the normalisation constants.
"""
from __future__ import annotations

import argparse, json, math, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, log_loss

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

SBR_SCALE = float(np.log1p(12.0))
IM_MEAN = (0.485, 0.456, 0.406)
IM_STD = (0.229, 0.224, 0.225)
# three overlapping axial slabs (voxel z-ranges of the 48^3 / 2 mm crop):
# central 24 mm summed slab, plus inferior and superior slabs.
SLABS = ((18, 30), (12, 22), (26, 36))


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_image(crop: np.ndarray, slabs=SLABS, views: str = "slabs") -> np.ndarray:
    """48^3 SBR crop -> (3, 48, 48) float32 in ~[0, 1].

    views="slabs": three overlapping axial slabs (dims L-R x P-A).
    views="mip3" : axial MIP, plus the coronal and sagittal MIPs re-embedded so
                   that image dim 1 stays L-R for the axial and coronal views
                   (the sagittal view has no L-R axis; its flip is harmless).
    """
    v = np.log1p(np.clip(crop.astype(np.float32), 0, None)) / SBR_SCALE
    if views == "slabs":
        chans = [v[:, :, a:b].mean(axis=2) for a, b in slabs]
    elif views == "mip3":
        ax = v.max(axis=2)                     # (L-R, P-A)
        co = v.max(axis=1)                     # (L-R, I-S)
        sa = v.max(axis=0)                     # (P-A, I-S)
        chans = [ax, co, sa]
    elif views == "ax3":
        # co-located axial channels (2026-09-07, external review member E): peak
        # appearance, broad slab mean, centrally weighted mean — all three refer to
        # the same (L-R, P-A) ray, unlike mip3 whose channels are different planes.
        zc = v.shape[2] // 2
        w = np.exp(-0.5 * ((np.arange(v.shape[2]) - zc) / 3.0) ** 2).astype(np.float32)
        ax = v.max(axis=2)
        mn = v[:, :, zc - 12:zc + 12].mean(axis=2)
        wm = (v * w[None, None, :]).sum(axis=2) / w.sum()
        chans = [ax, mn, wm]
    else:
        raise ValueError(views)
    return np.stack(chans, axis=0)


def make_images(crops: np.ndarray, slabs=SLABS, views: str = "slabs") -> np.ndarray:
    return np.stack([make_image(c, slabs, views) for c in crops]).astype(np.float32)


class HemiNet(nn.Module):
    """2D analogue of nets3.HemiMIL: split the image at the mid-sagittal line
    (dim 1 = L-R), mirror the left half, score both halves with the shared
    backbone and combine by noisy-OR. Each half is upsampled to res x res/2."""

    def __init__(self, backbone: str, pretrained: bool, drop: float):
        super().__init__()
        self.net = build(backbone, pretrained, drop)

    def hemi_logits(self, x):
        n = x.shape[2]
        h = n // 2
        left = torch.flip(x[:, :, :h, :], dims=[2])
        right = x[:, :, n - h:, :]
        both = torch.cat([left, right], 0)
        z = self.net(both).squeeze(1)
        return z[: x.shape[0]], z[x.shape[0]:]

    def forward(self, x):
        zl, zr = self.hemi_logits(x)
        neg = -(F.softplus(-zl) + F.softplus(-zr))
        pos = torch.log1p(-torch.exp(neg.clamp(max=-1e-6)))
        return (pos - neg).unsqueeze(1)


# --------------------------------------------------------------------------- #
# per-sample augmentation, on device, on the 48x48 image (before upsampling)
# --------------------------------------------------------------------------- #
def _blur2d(x, sigma, dev):
    if sigma <= 0.05:
        return x
    r = max(1, int(math.ceil(2.5 * sigma)))
    k = torch.arange(-r, r + 1, device=dev, dtype=torch.float32)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    c = x.shape[1]
    wx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    wy = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), wx, groups=c)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), wy, groups=c)
    return x


def augment(x, dev, strength=1.0, ref_jitter=0.12):
    """x: (B, 3, H, W). Every draw is per sample."""
    B = x.shape[0]
    flip = torch.as_tensor(np.random.rand(B) < 0.5, device=dev)
    x = torch.where(flip.view(B, 1, 1, 1), torch.flip(x, dims=[2]), x)
    # affine: rotation about the axial axis, isotropic scale, shift
    ang = np.deg2rad(np.random.uniform(-12, 12, B) * strength)
    sc = np.random.uniform(1 - 0.08 * strength, 1 + 0.08 * strength, B)
    tx = np.random.uniform(-0.06, 0.06, B) * strength
    ty = np.random.uniform(-0.06, 0.06, B) * strength
    on = np.random.rand(B) < 0.85
    th = np.zeros((B, 2, 3), dtype=np.float32)
    th[:, 0, 0] = np.where(on, np.cos(ang) * sc, 1.0)
    th[:, 0, 1] = np.where(on, -np.sin(ang) * sc, 0.0)
    th[:, 1, 0] = np.where(on, np.sin(ang) * sc, 0.0)
    th[:, 1, 1] = np.where(on, np.cos(ang) * sc, 1.0)
    th[:, 0, 2] = np.where(on, tx, 0.0)
    th[:, 1, 2] = np.where(on, ty, 0.0)
    grid = F.affine_grid(torch.from_numpy(th).to(dev), x.shape, align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="border")
    # per-sample blur, quantised to 3 levels so each level is one conv
    sig = np.where(np.random.rand(B) < 0.5, np.random.uniform(0.0, 1.0 * strength, B), 0.0)
    for lo, hi, s in ((0.05, 0.45, 0.3), (0.45, 0.8, 0.6), (0.8, 9.0, 0.95)):
        idx = np.where((sig > lo) & (sig <= hi))[0]
        if len(idx):
            it = torch.as_tensor(idx, device=dev)
            x[it] = _blur2d(x[it], s * strength, dev)
    # per-sample gain (reference jitter) and count-like noise
    g = np.where(np.random.rand(B) < 0.7, np.exp(np.random.normal(0.0, ref_jitter * strength, B)), 1.0)
    x = x * torch.as_tensor(g, dtype=x.dtype, device=dev).view(B, 1, 1, 1)
    amp = np.where(np.random.rand(B) < 0.5, np.random.uniform(0.0, 0.06 * strength, B), 0.0)
    x = x + torch.randn_like(x) * torch.as_tensor(amp, dtype=x.dtype, device=dev).view(B, 1, 1, 1) * x.clamp(min=0).sqrt()
    return x


def prep(x, res, dev):
    """(B,3,H,W) -> upsampled (same factor on both dims) + ImageNet-normalised."""
    if res != x.shape[-1]:
        f = res / x.shape[-1]
        x = F.interpolate(x, size=(int(round(x.shape[-2] * f)), res), mode="bilinear",
                          align_corners=False)
    m = torch.tensor(IM_MEAN, device=dev).view(1, 3, 1, 1)
    s = torch.tensor(IM_STD, device=dev).view(1, 3, 1, 1)
    return (x - m) / s


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        # warmed-up decay (timm ModelEmaV3 style): early on the shadow tracks the
        # live weights closely, so a ~1000-step schedule is not dominated by the
        # random initial head
        self.n = getattr(self, "n", 0) + 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                self.shadow[k] = v.detach().clone().float()

    def state(self, model):
        ref = model.state_dict()
        return {k: self.shadow[k].to(ref[k].dtype) for k in ref}


def build(backbone: str, pretrained: bool, drop: float):
    """timm initialises a 1-output Linear head with uniform(-1/sqrt(fan_out), ..)
    = uniform(-1, 1) — initial logits have sd ~5 and the first steps wreck the
    pretrained features (measured: train loss 3-7 for 40 steps). Zero the head
    so training starts at p = 0.5 and the backbone is touched gently."""
    import timm
    net = timm.create_model(backbone, pretrained=pretrained, num_classes=1,
                            drop_rate=drop, in_chans=3)
    head = net.get_classifier()
    nn.init.zeros_(head.weight)
    if head.bias is not None:
        nn.init.zeros_(head.bias)
    return net


def build_model(args_or_dict, pretrained: bool):
    a = args_or_dict
    get = (lambda k, d=None: getattr(a, k, d)) if not isinstance(a, dict) else (lambda k, d=None: a.get(k, d))
    if get("hemi", False):
        return HemiNet(get("backbone"), pretrained, get("drop", 0.3))
    return build(get("backbone"), pretrained, get("drop", 0.3))


@torch.no_grad()
def predict(net, X, res, dev, bs=64):
    net.eval()
    if len(X) == 0:                 # full-data training has no held-out rows
        return np.zeros(0)
    out = []
    for j in range(0, len(X), bs):
        xb = X[j:j + bs].to(dev)
        z = net(prep(xb, res, dev)).squeeze(1)
        z2 = net(prep(torch.flip(xb, dims=[2]), res, dev)).squeeze(1)
        out.append((0.5 * (torch.sigmoid(z) + torch.sigmoid(z2))).float().cpu().numpy())
    return np.concatenate(out)


def run_fold(X, y, tr, va, dev, args, seed, tag, fold_id):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr = torch.from_numpy(X[tr])
    Xva = torch.from_numpy(X[va])
    ytr = torch.from_numpy(y[tr]).float().to(dev)
    net = build_model(args, pretrained=not args.scratch).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    ema = EMA(net, args.ema)
    n = len(tr)
    spe = max(1, math.ceil(n / args.bs))
    total = args.epochs * spe
    warm = args.warmup * spe
    step = 0
    ls = args.label_smooth
    t0 = time.time()
    for ep in range(args.epochs):
        net.train()
        perm = np.random.permutation(n)
        tot, nb = 0.0, 0
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            xb = augment(Xtr[idx].to(dev), dev, args.aug, args.ref_jitter)
            tb = ytr[idx]
            if ls > 0:
                tb = tb * (1 - ls) + 0.5 * ls
            if args.mixup > 0 and np.random.rand() < args.mixup_p:
                lam = float(np.random.beta(args.mixup, args.mixup))
                lam = max(lam, 1 - lam)
                p = torch.randperm(len(idx), device=dev)
                xb = lam * xb + (1 - lam) * xb[p]
                tb = lam * tb + (1 - lam) * tb[p]
            lr = (step + 1) / max(warm, 1) if step < warm else \
                0.5 * (1 + math.cos(math.pi * (step - warm) / max(total - warm, 1)))
            for g in opt.param_groups:
                g["lr"] = args.lr * max(lr, 1e-3)
            opt.zero_grad()
            z = net(prep(xb, args.res, dev)).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(z, tb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            ema.update(net)
            step += 1
            tot += float(loss.detach()); nb += 1
        if args.verbose and (ep % 5 == 0 or ep == args.epochs - 1):
            ev = build_model(args, pretrained=False).to(dev)
            ev.load_state_dict(ema.state(net))
            pv = predict(ev, Xva, args.res, dev)
            print(f"  [{tag}] ep{ep:03d} train {tot / nb:.4f} val_ll "
                  f"{log_loss(y[va], np.clip(pv, 1e-6, 1 - 1e-6)):.4f} auc "
                  f"{roc_auc_score(y[va], pv):.4f} ({time.time() - t0:.0f}s)", flush=True)
    final = build_model(args, pretrained=False).to(dev)
    final.load_state_dict(ema.state(net))
    pv = predict(final, Xva, args.res, dev)
    state = {k: v.cpu() for k, v in final.state_dict().items()}
    return pv, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="e2d")
    ap.add_argument("--backbone", default="efficientnet_b0.ra_in1k")
    ap.add_argument("--scratch", action="store_true", help="no ImageNet init")
    ap.add_argument("--crops", default="crops_v3_b.npz")
    ap.add_argument("--key", default="aligned")
    ap.add_argument("--res", type=int, default=192)
    ap.add_argument("--views", default="slabs", choices=["slabs", "mip3", "ax3"])
    ap.add_argument("--hemi", action="store_true", help="2D hemisphere-MIL (noisy-OR)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--mixup", type=float, default=0.4)
    ap.add_argument("--mixup-p", type=float, default=1.0)
    ap.add_argument("--label-smooth", type=float, default=0.02)
    ap.add_argument("--aug", type=float, default=1.0)
    ap.add_argument("--ref-jitter", type=float, default=0.12)
    ap.add_argument("--seed-base", type=int, default=7000)
    ap.add_argument("--only-fold", type=int, default=-1)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--full-tag", default="full0")
    ap.add_argument("--folds", default="folds.csv")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    dev = device()
    z = np.load(OUT / args.crops, allow_pickle=True)
    crops = z[args.key]
    uids = z["uids"].astype(str)
    y = z["y"].astype(np.float32)
    X = make_images(crops, views=args.views)
    folds = pd.read_csv(OUT / args.folds)
    fmap = dict(zip(folds["uid"].astype(str), folds["fold"]))
    fold = np.array([fmap[u] for u in uids])
    print(f"[{args.tag}] dev={dev} backbone={args.backbone} X{X.shape} res={args.res} "
          f"epochs={args.epochs} lr={args.lr} aug={args.aug}", flush=True)
    t0 = time.time()

    if args.full:
        tr = np.arange(len(uids)); va = np.zeros(0, dtype=int)
        _, state = run_fold(X, y, tr, va, dev, args, args.seed_base + 99, args.tag, -1)
        if not args.no_save:
            torch.save(dict(kind="cnn2d", backbone=args.backbone, res=args.res,
                            slabs=SLABS, views=args.views, hemi=bool(args.hemi),
                            state=state, drop=args.drop),
                       OUT / f"{args.tag}_{args.full_tag}.pt")
        print(f"[{args.tag}] full-data model saved ({time.time() - t0:.0f}s)", flush=True)
        return

    oof = np.full(len(uids), np.nan)
    for k in range(5):
        if args.only_fold >= 0 and k != args.only_fold:
            continue
        tr = np.where(fold != k)[0]; va = np.where(fold == k)[0]
        pv, state = run_fold(X, y, tr, va, dev, args, args.seed_base + k, args.tag, k)
        oof[va] = pv
        ll = log_loss(y[va], np.clip(pv, 1e-6, 1 - 1e-6)); auc = roc_auc_score(y[va], pv)
        print(f"[{args.tag}] fold {k}: ll {ll:.4f} auc {auc:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if not args.no_save:
            torch.save(dict(kind="cnn2d", backbone=args.backbone, res=args.res,
                            slabs=SLABS, views=args.views, hemi=bool(args.hemi),
                            state=state, drop=args.drop),
                       OUT / f"{args.tag}_fold{k}.pt")
        # kill-bar readout against the incumbents on this fold
        try:
            f3 = pd.read_csv(ROOT / "champion_v7_CV0.2139" / "oof_feat3.csv").set_index("uid")
            m = pd.read_csv(OUT / "oof_mil48y.csv").set_index("uid")
            lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / np.clip(1 - q, 1e-6, 1 - 1e-6))
            pl = f3.loc[uids[va], "lin"].to_numpy(float); pm = m.loc[uids[va], "mil48y"].to_numpy(float)
            print(f"[{args.tag}] fold {k}: lin ll {log_loss(y[va], np.clip(pl,1e-6,1-1e-6)):.4f} "
                  f"mil48y ll {log_loss(y[va], np.clip(pm,1e-6,1-1e-6)):.4f} | "
                  f"corr(lin) {np.corrcoef(lg(pv), lg(pl))[0,1]:.3f} corr(mil48y) "
                  f"{np.corrcoef(lg(pv), lg(pm))[0,1]:.3f}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{args.tag}] readout skipped: {e!r}")
    if args.only_fold < 0:
        ok = ~np.isnan(oof)
        print(f"\n[{args.tag}] OOF logloss {log_loss(y[ok], np.clip(oof[ok],1e-6,1-1e-6)):.4f} "
              f"auroc {roc_auc_score(y[ok], oof[ok]):.4f} ({time.time() - t0:.0f}s)", flush=True)
        pd.DataFrame(dict(uid=uids, y=y.astype(int), fold=fold, **{args.tag: oof})).to_csv(
            OUT / f"oof_{args.tag}.csv", index=False)
        json.dump(dict(logloss=float(log_loss(y[ok], np.clip(oof[ok],1e-6,1-1e-6))),
                       auroc=float(roc_auc_score(y[ok], oof[ok])), args=vars(args)),
                  open(OUT / f"cv_{args.tag}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
