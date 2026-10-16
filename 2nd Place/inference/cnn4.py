"""
cnn4.py — trainer for the per-hemisphere-crop MIL models (`crops_hemi.npz`).

Same honest protocol as `cnn3.py` — fixed schedule, no validation-driven
stopping, EMA weights, OOF = the final model's output on its held-out fold — but
the input is now (N, K, D, H, W): K per-hemisphere boxes already sampled in a
common handedness, not one whole-head cube to be split later.

Three things had to change with the geometry, and each of them is a correctness
issue rather than a tuning choice:

1. **The L/R flip is no longer an augmentation.** `cnn3.augment` flips the x axis
   of the whole-head cube, which is an exact symmetry of the label. Here the two
   boxes have already been mirrored into a common handedness, so flipping x
   inside a box produces an anatomically impossible left-handed striatum. The
   symmetry that survives is *swapping the two instances*, and that is what is
   used — for augmentation and for TTA.

2. **The affine has to be built in physical units.** `F.affine_grid` works in
   normalised coordinates, so on a cube a rotation matrix can be passed straight
   through. On a 32x48x32 box the same matrix is a shear. The rotation is
   therefore conjugated by the axis-length scaling, and expressed in the
   (W, H, D) order `affine_grid` expects.

3. **Which nuisances are shared and which are not.** Blur, gain and count noise
   are properties of the acquisition, so they are drawn once per scan and applied
   to both hemispheres. Localisation error is independent per striatum, so the
   affine is drawn per instance.

    python cnn4.py --tag h1 --arch hmils --only-fold 0 --no-save     # probe
    python cnn4.py --tag h1 --arch hmils                             # 5 folds
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

from nets3 import build

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# input representation
# --------------------------------------------------------------------------- #
def _rank(v: np.ndarray) -> np.ndarray:
    """Voxel percentile within its own box, in [0, 1].

    Invariant to ANY monotone intensity transform, so it is immune to reference
    error, scanner gain and reconstruction non-linearity at once. It discards
    absolute contrast, so it is weaker alone — the point is that it shares no
    intensity assumption with the log-SBR members (ROADMAP §1).
    """
    flat = v.reshape(v.shape[0], -1)
    out = np.empty_like(flat)
    n = flat.shape[1]
    idx = np.argsort(flat, axis=1, kind="stable")
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    np.put_along_axis(out, idx, np.broadcast_to(ramp, flat.shape), axis=1)
    return out.reshape(v.shape)


def to_input_hemi(sbr: np.ndarray, channels: str = "log") -> np.ndarray:
    """(N, K, D, H, W) SBR boxes -> (N, K*C, D, H, W), channels instance-major.

    Instance-major so `HemiCropMIL` can reshape straight to (N*K, C, D, H, W).
    """
    s = np.clip(np.asarray(sbr, dtype=np.float32), 0.0, None)
    n, k = s.shape[0], s.shape[1]
    flat = s.reshape(n * k, *s.shape[2:])
    if channels == "log":
        ch = [np.log1p(flat)]
    elif channels == "rank":
        ch = [_rank(flat)]
    elif channels == "log+rank":
        ch = [np.log1p(flat), _rank(flat)]
    elif channels == "log+spec":
        ch = [np.log1p(flat), np.log1p(np.clip(flat - 1.0, 0.0, None))]
    elif channels == "selfmax":
        # normalise each box by its own robust maximum: removes the head
        # reference from the loop entirely, keeping only the SHAPE of the uptake
        m = np.percentile(flat.reshape(len(flat), -1), 99.5, axis=1)
        m = np.clip(m, 1e-3, None)[:, None, None, None]
        ch = [np.log1p(flat / m * 4.0)]
    else:
        raise ValueError(channels)
    x = np.stack(ch, axis=1)                       # (N*K, C, D, H, W)
    return x.reshape(n, k * len(ch), *s.shape[2:])


# --------------------------------------------------------------------------- #
# augmentation
# --------------------------------------------------------------------------- #
def _theta(batch: int, shape, rot_deg: float, shift: float, scale: float, dev):
    """Affine matrices for `F.affine_grid` that are true rigid+scale transforms
    in millimetres, not in normalised coordinates.

    `shape` is (D, H, W) of the volume. affine_grid orders coordinates
    (x, y, z) = (W, H, D), so the physical rotation is reversed into that order
    and conjugated by diag(axis lengths).
    """
    rot = np.deg2rad(np.random.uniform(-rot_deg, rot_deg, size=(batch, 3)))
    sc = np.random.uniform(1 - scale, 1 + scale, size=batch)
    t = np.random.uniform(-shift, shift, size=(batch, 3))
    s = np.asarray(shape[::-1], dtype=np.float64)          # (W, H, D) lengths
    A = np.zeros((batch, 3, 4))
    for b in range(batch):
        rx, ry, rz = rot[b]
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R = (Rx @ Ry @ Rz) * sc[b]
        R = R[::-1, ::-1]                                   # (x,y,z) -> (W,H,D)
        A[b, :, :3] = (R * s[None, :]) / s[:, None]         # diag(1/s) R diag(s)
        A[b, :, 3] = t[b]
    return torch.tensor(A, dtype=torch.float32, device=dev)


def rand_affine(x, rot_deg, shift, scale, dev):
    th = _theta(x.shape[0], tuple(x.shape[2:]), rot_deg, shift, scale, dev)
    grid = F.affine_grid(th, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="border")


def _blur(x, sigma_vox, dev):
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


def augment(x, n_inst, dev, strength=1.0):
    """x: (B, K*C, D, H, W). Scan-level nuisances shared, per-striatum jitter not."""
    b, ch = x.shape[0], x.shape[1]
    c = ch // n_inst
    if np.random.rand() < 0.5:                       # exact symmetry: swap sides
        x = x.reshape(b, n_inst, c, *x.shape[2:]).flip(1).reshape(b, ch, *x.shape[2:])
    if np.random.rand() < 0.85:                      # localisation error: per side
        v = x.reshape(b * n_inst, c, *x.shape[2:])
        v = rand_affine(v, 7 * strength, 0.06 * strength, 0.07 * strength, dev)
        x = v.reshape(b, ch, *x.shape[2:])
    if np.random.rand() < 0.5:                       # reconstruction / collimator
        x = _blur(x, float(np.random.uniform(0.0, 1.5 * strength)), dev)
    if np.random.rand() < 0.7:                       # residual reference error
        x = x * float(np.exp(np.random.normal(0.0, 0.12 * strength)))
    if np.random.rand() < 0.5:                       # SPECT count statistics
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
def predict(net, X, n_inst=2, bs=64, tta=True):
    """Mean probability over the instance-swap symmetry."""
    net.eval()
    out = []
    for j in range(0, len(X), bs):
        xb = X[j:j + bs]
        z = torch.sigmoid(net(xb))
        if tta:
            b, ch = xb.shape[0], xb.shape[1]
            c = ch // n_inst
            sw = xb.reshape(b, n_inst, c, *xb.shape[2:]).flip(1).reshape(b, ch, *xb.shape[2:])
            z = 0.5 * (z + torch.sigmoid(net(sw)))
        out.append(z.float().cpu().numpy())
    return np.concatenate(out)


def run_fold(x, y, tr, va, dev, args, seed, n_inst):
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu = float(x[tr].mean())
    sd = float(x[tr].std() + 1e-6)
    Xtr = torch.from_numpy((x[tr] - mu) / sd).float().to(dev)
    Xva = torch.from_numpy((x[va] - mu) / sd).float().to(dev)
    ytr = torch.from_numpy(y[tr]).float().to(dev)

    net = build(args.arch, in_ch=x.shape[1], drop=args.drop, n_inst=n_inst).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    ema = EMA(net, decay=args.ema)
    n = len(tr)
    total = args.epochs * max(1, math.ceil(n / args.bs))
    warm = args.warmup * max(1, math.ceil(n / args.bs))
    step = 0

    for ep in range(args.epochs):
        net.train()
        perm = np.random.permutation(n)
        for i in range(0, n, args.bs):
            idx = perm[i:i + args.bs]
            xb = augment(Xtr[idx], n_inst, dev, args.aug)
            tb = ytr[idx]
            if args.label_smooth > 0:
                tb = tb * (1 - args.label_smooth) + 0.5 * args.label_smooth
            if args.mixup > 0 and np.random.rand() < 0.5:
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
            loss = F.binary_cross_entropy_with_logits(net(xb), tb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            ema.update(net)
            step += 1
        if args.verbose and (ep % 5 == 0 or ep == args.epochs - 1) and len(va):
            ev = build(args.arch, in_ch=x.shape[1], drop=args.drop, n_inst=n_inst).to(dev)
            ev.load_state_dict(ema.state(net))
            pv = predict(ev, Xva, n_inst)
            print(f"    ep{ep:03d} val_ll {log_loss(y[va], np.clip(pv,1e-6,1-1e-6)):.4f} "
                  f"auc {roc_auc_score(y[va], pv):.4f}", flush=True)

    final = build(args.arch, in_ch=x.shape[1], drop=args.drop, n_inst=n_inst).to(dev)
    final.load_state_dict(ema.state(net))
    pv = predict(final, Xva, n_inst) if len(va) else np.zeros(0)
    return pv, {k: v.cpu() for k, v in final.state_dict().items()}, (mu, sd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="h1")
    ap.add_argument("--arch", default="hmils")
    ap.add_argument("--crops", default="crops_hemi.npz")
    ap.add_argument("--key", default="hemi", choices=["hemi", "hemi_raw"])
    ap.add_argument("--channels", default="log")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=3e-2)
    ap.add_argument("--drop", type=float, default=0.3)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--mixup", type=float, default=0.4)
    ap.add_argument("--label-smooth", type=float, default=0.02)
    ap.add_argument("--aug", type=float, default=1.0)
    ap.add_argument("--seed-base", type=int, default=20000)
    ap.add_argument("--only-fold", type=int, default=-1)
    ap.add_argument("--folds", default="folds.csv")
    ap.add_argument("--full", action="store_true",
                    help="also train one model on 100%% of the rows (no held-out "
                         "fold). The schedule has no validation-dependent step, so "
                         "this is legitimate; it ships as an extra source of the "
                         "same member and sees 25%% more data than a fold model.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    dev = device()
    z = np.load(OUT / args.crops, allow_pickle=True)
    sbr = z[args.key].astype(np.float32)
    y = z["y"].astype(np.float32)
    uids = z["uids"].astype(str)
    n_inst = sbr.shape[1]
    # Record the sampling geometry in the checkpoint: inference has to rebuild
    # exactly this box, and the shipped stack mixes more than one.
    if "geom" in z.files:
        g = z["geom"]
        hemi_spacing, hemi_shape = float(g[0]), [int(v) for v in g[1:]]
    else:                       # the first cache predates the geom field
        from preprocessing3 import HEMI_SPACING, HEMI_SHAPE
        hemi_spacing, hemi_shape = float(HEMI_SPACING), [int(v) for v in HEMI_SHAPE]
    if tuple(hemi_shape) != tuple(sbr.shape[2:]):
        raise SystemExit(f"cache geometry {hemi_shape} does not match array "
                         f"{tuple(sbr.shape[2:])} — refusing to train")
    x = to_input_hemi(sbr, args.channels)
    del sbr
    folds = pd.read_csv(OUT / args.folds).set_index("uid").loc[uids].reset_index()
    fold = folds["fold"].values
    print(f"[{args.tag}] dev={dev} arch={args.arch} x{x.shape} key={args.key} "
          f"ch={args.channels} inst={n_inst} epochs={args.epochs}", flush=True)

    oof = np.zeros(len(y))
    t0 = time.time()
    fold_ids = sorted(np.unique(fold))
    if args.only_fold >= 0:
        fold_ids = [args.only_fold]
    for k in fold_ids:
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
        pv, state, (mu, sd) = run_fold(x, y, tr, va, dev, args, args.seed_base + k, n_inst)
        oof[va] = pv
        if not args.no_save:
            torch.save({"state": state, "mu": mu, "sd": sd, "arch": args.arch,
                        "in_ch": x.shape[1], "n_inst": n_inst, "input": "hemi",
                        "key": args.key, "channels": args.channels,
                        "size": 0, "drop": args.drop, "crops": args.crops,
                        "hemi_spacing": hemi_spacing, "hemi_shape": hemi_shape},
                       OUT / f"{args.tag}_fold{k}.pt")
        print(f"[{args.tag}] fold {k}: ll {log_loss(y[va], np.clip(pv,1e-6,1-1e-6)):.4f} "
              f"auc {roc_auc_score(y[va], pv):.4f}  ({time.time()-t0:.0f}s)", flush=True)

    if args.full and not args.no_save:
        allidx = np.arange(len(y))
        _, state, (mu, sd) = run_fold(x, y, allidx, allidx[:0], dev, args,
                                      args.seed_base + 99, n_inst)
        torch.save({"state": state, "mu": mu, "sd": sd, "arch": args.arch,
                    "in_ch": x.shape[1], "n_inst": n_inst, "input": "hemi",
                    "key": args.key, "channels": args.channels,
                    "size": 0, "drop": args.drop, "crops": args.crops,
                    "hemi_spacing": hemi_spacing, "hemi_shape": hemi_shape},
                   OUT / f"{args.tag}_full0.pt")
        print(f"[{args.tag}] full-data model written ({time.time()-t0:.0f}s)", flush=True)

    if args.only_fold >= 0:
        return
    oof = np.clip(oof, 1e-6, 1 - 1e-6)
    LL, AUC = log_loss(y, oof), roc_auc_score(y, oof)
    print(f"\n[{args.tag}] OOF logloss {LL:.4f} auroc {AUC:.4f} ({time.time()-t0:.0f}s)",
          flush=True)
    if not args.no_save:
        pd.DataFrame({"uid": uids, "y": y.astype(int), "fold": fold,
                      args.tag: oof}).to_csv(OUT / f"oof_{args.tag}.csv", index=False)
        (OUT / f"cv_{args.tag}.json").write_text(json.dumps(
            {"logloss": LL, "auroc": AUC, "args": vars(args)}, indent=2, default=str))


if __name__ == "__main__":
    main()
