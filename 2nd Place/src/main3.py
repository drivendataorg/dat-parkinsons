"""
main3.py — INFERENCE ENTRYPOINT for the DrivenData DaT-Park code-execution container.

Reads the test set mounted at /code_execution/data:
    data/submission_format.csv   (index=uid, column: is_pathologic)
    data/niftis/<uid>.nii.gz
Writes ./submission.csv.

Per scan, independently (per-sample rule): preprocessing3 gives a
reference-normalised striatal crop in a canonical frame; features6 turns it into
clinical measurements for the linear/GBM models; the same crop feeds the 3D CNNs.
Both of those are the exact modules `cache3.py` used to build the training
tables, so the two cannot drift. Member probabilities are combined by a stacker
whose non-negative coefficients were fitted out-of-fold. Any per-scan failure
falls back to the training prior, and a CNN failure degrades to the feature
models alone.

LOGGING RULE — do not regress this. The platform filters submission logs and
treats any line containing test-set-derived content as a rules violation. Every
logger call below is a STATIC string: no counts, no uids, no per-scan values, no
paths. Failure warnings fire at most once so even the number of warning lines
cannot reveal how many scans failed. Verify with:
    grep -nE "logger\\.[a-z]+\\(f" main.py     # must print nothing
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

SRC = Path(__file__).parent.resolve()
ASSETS = SRC / "assets"
sys.path.insert(0, str(SRC))

import preprocessing3 as pp                        # noqa: E402
import features13                                  # noqa: E402
import portable3 as P                              # noqa: E402

DATA_ROOT = Path(os.environ.get("DATA_DIR", "/code_execution/data"))
NIFTI_DIR = DATA_ROOT / "niftis"
SUBMISSION_FORMAT = DATA_ROOT / "submission_format.csv"
OUT_PATH = Path("submission.csv")
BLOCK = 128
CNN_BATCH = 32


def _load():
    cfg = json.loads((ASSETS / "inference_config.json").read_text())
    linears = {name: P.load_linear(ASSETS / f"linear_{name}.json")
               for name in cfg.get("linears", [])}
    gbms = {}
    for name, seeds in cfg.get("gbms", {}).items():
        gbms[name] = [P.load_lgbm(ASSETS / f"lgbm_{name}_s{s}.txt") for s in seeds]
    mlps = {}
    for name, seeds in cfg.get("mlps", {}).items():
        mlps[name] = [P.load_mlp(ASSETS / f"mlp_{name}_s{s}.json") for s in seeds]
    cnns = []
    if cfg.get("cnns"):
        import torch
        from nets3 import build
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for spec in cfg["cnns"]:
            models = []
            # `sources` is a list of [checkpoint-prefix, fold]; one member may span
            # several training runs, whose predictions are averaged.
            sources = spec.get("sources") or [[spec["tag"], k] for k in spec["folds"]]
            if spec.get("kind") == "cnn2d":
                # v12: ImageNet-initialised 2D backbone on the axial read (cnn2d.py).
                # Weights come entirely from the bundled checkpoint; timm builds the
                # architecture offline (pretrained=False).
                import cnn2d
                for src, k in sources:
                    part = k if isinstance(k, str) else f"fold{k}"
                    if spec.get("ts"):
                        # TorchScript export (tools/ts_export.py): the container
                        # needs no timm; image geometry comes from the spec
                        net = torch.jit.load(str(ASSETS / f"{src}_{part}.ts.pt"), map_location=dev)
                        net.eval()
                        models.append((net, dict(res=int(spec["res"]), slabs=spec["slabs"],
                                                 views=spec.get("views", "slabs"))))
                        continue
                    ck = torch.load(ASSETS / f"{src}_{part}.pt",
                                    map_location=dev, weights_only=False)
                    net = cnn2d.build_model(dict(backbone=ck["backbone"], drop=ck.get("drop", 0.3),
                                                 hemi=bool(ck.get("hemi", False))),
                                            pretrained=False).to(dev)
                    net.load_state_dict(ck["state"])
                    net.eval()
                    models.append((net, ck))
                cnns.append(dict(spec=spec, models=models, dev=dev, torch=torch))
                continue
            for src, k in sources:
                # `k` is a checkpoint suffix ("fold0" .. "fold4", "full0"). Older
                # configs wrote a bare fold index; accept both.
                part = k if isinstance(k, str) else f"fold{k}"
                # weights_only=False: our own checkpoints; the torch>=2.6 default
                # would otherwise depend on the runtime's exact torch version.
                ck = torch.load(ASSETS / f"{src}_{part}.pt",
                                map_location=dev, weights_only=False)
                over = {"n_inst": ck["n_inst"]} if "n_inst" in ck else {}
                net = build(ck["arch"], in_ch=ck["in_ch"],
                            drop=ck.get("drop", 0.3), **over).to(dev)
                net.load_state_dict(ck["state"])
                net.eval()
                models.append((net, float(ck["mu"]), float(ck["sd"])))
            cnns.append(dict(spec=spec, models=models, dev=dev, torch=torch))
    return cfg, linears, gbms, mlps, cnns


def _box_key(spec):
    """Whole-crop geometry of one member. 96 mm / 48 voxels is the v3 default and
    the fallback for configs written before crop extent became a variable."""
    return f"{float(spec.get('crop_mm', 96.0)):g}:{int(spec.get('out_vox', 48))}"


def _tta_views(torch, xb, hemi, n_inst, level):
    """Views to average over at inference.

    View 0 is always the image itself. View 1 is the label's exact symmetry — a
    mid-sagittal flip for a whole-head cube, an instance swap for the
    pre-cropped hemispheres, where flipping inside a box would instead produce
    an anatomically impossible left-handed striatum. Higher levels add the
    resolution jitter the models were explicitly trained to be invariant to, so
    it is variance reduction rather than a new assumption.
    """
    views = [xb]
    if level >= 1:
        if hemi:
            b, ch = xb.shape[0], xb.shape[1]
            c = ch // n_inst
            views.append(xb.reshape(b, n_inst, c, *xb.shape[2:])
                         .flip(1).reshape(b, ch, *xb.shape[2:]))
        else:
            views.append(torch.flip(xb, dims=[2]))
    if level >= 2:
        import cnn4
        extra = []
        for v in list(views):
            for s in (0.6, 1.1):
                extra.append(cnn4._blur(v, s, v.device))
        views += extra
    return views


def _cnn2d_prob(cnn, crops):
    """2D member: images from the 96 mm / 48-voxel aligned crop, flip-TTA, mean
    over checkpoints (cnn2d.predict is the training-time evaluation path)."""
    import cnn2d
    torch = cnn["torch"]
    dev = cnn["dev"]
    spec = cnn["spec"]
    arr = np.asarray(crops[f"{spec.get('key', 'aligned')}@96:48"], dtype=np.float32)
    probs = np.zeros(len(arr))
    for net, ck in cnn["models"]:
        X = torch.from_numpy(cnn2d.make_images(arr, slabs=tuple(tuple(v) for v in ck["slabs"]),
                                               views=ck.get("views", "slabs")))
        probs += cnn2d.predict(net, X, int(ck["res"]), dev, bs=CNN_BATCH)
    return probs / max(len(cnn["models"]), 1)


def _cnn_prob(cnn, crops, comp=None):
    """crops: key -> stacked SBR array. Mean prob over checkpoints and TTA views.

    A member with `offset_member` is a BOOSTER: it was trained to predict the
    residual of that member's logit, so its own logit has to be added back here.
    `comp` therefore has to already contain that member — the feature models are
    evaluated before the networks for exactly this reason.
    """
    torch = cnn["torch"]
    dev = cnn["dev"]
    spec = cnn["spec"]
    import torch.nn.functional as Fn
    hemi = spec.get("input") == "hemi"
    key = spec.get("key", "aligned")
    if hemi:
        key = f"{key}@{pp.geom_key(spec['hemi_spacing'], spec['hemi_shape'])}"
    else:
        key = f"{key}@{_box_key(spec)}"
    arr = np.asarray(crops[key], dtype=np.float32)
    if hemi:
        from cnn4 import to_input_hemi
        x = to_input_hemi(arr, spec.get("channels", "log"))
        n_inst = int(spec.get("n_inst", arr.shape[1]))
    else:
        from cnn3 import to_input
        x = to_input(arr, spec.get("channels", "log"))
        n_inst = 1
    t = torch.from_numpy(x).float()
    size = int(spec.get("size", 48))
    if not hemi and size != t.shape[-1]:
        t = Fn.interpolate(t, size=(size,) * 3, mode="trilinear", align_corners=False)
    if spec.get("crops2_mm"):
        # foveated member: a second field of view on the same grid, appended as
        # extra channels in the order the trainer concatenated them
        from cnn3 import to_input as _ti
        k2 = (f"{spec.get('key', 'aligned')}@{float(spec['crops2_mm']):g}"
              f":{int(spec['crops2_vox'])}")
        t2 = torch.from_numpy(_ti(np.asarray(crops[k2], dtype=np.float32),
                                  spec.get("channels", "log"))).float()
        if size != t2.shape[-1]:
            t2 = Fn.interpolate(t2, size=(size,) * 3, mode="trilinear",
                                align_corners=False)
        t = torch.cat([t, t2], dim=1)
    level = int(spec.get("tta", 1))
    off = None
    om = spec.get("offset_member")
    if om:
        q = np.clip(np.asarray(comp[om], dtype=np.float64), 1e-6, 1 - 1e-6)
        off = torch.from_numpy(np.log(q / (1.0 - q))).float()
    probs = np.zeros(len(arr))
    with torch.no_grad():
        for net, mu, sd in cnn["models"]:
            xs = (t - mu) / sd
            for j in range(0, len(xs), CNN_BATCH):
                xb = xs[j:j + CNN_BATCH].to(dev)
                ob = 0.0 if off is None else off[j:j + CNN_BATCH].to(dev)
                views = _tta_views(torch, xb, hemi, n_inst, level)
                p = sum(torch.sigmoid(net(v) + ob) for v in views) / len(views)
                probs[j:j + CNN_BATCH] += p.float().cpu().numpy()
    return probs / max(len(cnn["models"]), 1)


def main():
    logger.info("Reading submission format")
    sub = pd.read_csv(SUBMISSION_FORMAT, index_col=0)
    sub["is_pathologic"] = sub["is_pathologic"].astype(float)
    cfg, linears, gbms, mlps, cnns = _load()
    order = cfg["feature_order"]
    members = cfg["members"]
    prior = float(cfg.get("prior", 0.5))
    coef = np.array(cfg["stacker"]["coef"], dtype=np.float64)
    intercept = float(cfg["stacker"]["intercept"])
    # Safety calibration fitted on the leave-site-out surface rather than random
    # splits (shift_cal2.py). The stack is not over-confident under site shift, so
    # T is 1.0; the clip bounds the one thing a log-loss metric punishes hardest,
    # a confident miss on an unfamiliar scanner. Both were checked to also help
    # in-distribution, so this is not a premium paid against a hypothetical.
    cal_T = float(cfg.get("calibration", {}).get("T", 1.0))
    cal_clip = float(cfg.get("calibration", {}).get("clip", 0.0))
    uids = list(sub.index)
    logger.info("Models loaded; running inference")

    # Fail loudly if the feature module cannot produce every name the models were
    # fitted on. Without this a version skew shows up as silently NaN (then
    # median-imputed) columns — a large, invisible accuracy loss rather than an
    # error. This exact skew happened once during development.
    # The probe's synthetic meta must name every key `preprocess_full` really
    # produces, or a genuinely-present feature reads as missing and the guard
    # below refuses to ship a correct package. Built from the real thing rather
    # than a fourth hand-maintained list: the meta dict, features4's allowlist,
    # train_final3.SCANNER and this probe are four separate places a key has to
    # be named, which is how `head_ml` stayed invisible to every model until v8.
    probe = set(features13.extract(
        np.ones((48, 48, 48), np.float32), 26.0,
        {k: 1.0 for k in ("head_ml", "inner_ml", "hole_ml", "two_blob",
                          "native_spacing", "ref")}))
    absent = [k for k in order if k not in probe]
    if absent:
        raise RuntimeError(f"feature module is missing {len(absent)} of "
                           f"{len(order)} expected features")

    idx_of = {k: j for j, k in enumerate(order)}

    # Registration to the packaged template (regfit.py). The canonical frame pins
    # only the L-R and I-S axes, leaving ~10 degrees of subject-dependent pose
    # error that every fixed-ROI measurement is read through; this estimates it
    # per scan and folds it into the crop's rotation. Failure is not fatal — the
    # refine callable rejects implausible fits itself, and if the whole thing is
    # unavailable the canonical frame is used, which is what v3 shipped.
    register = None
    rcfg = cfg.get("registration")
    if rcfg:
        try:
            import torch
            import regfit
            rdev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            _tpl = regfit.load_template(ASSETS / rcfg.get("template", "template_v6.npy"),
                                        rdev)
            _steps = tuple(rcfg.get("steps", (120, 80)))
            _wt = bool(rcfg.get("weighted", False))
            _md = float(rcfg.get("max_deg", regfit.MAX_DEG))
            _mn = float(rcfg.get("min_ncc", regfit.MIN_NCC))

            def register(arr):
                w, rot, sh, _q, bad = regfit.register(
                    arr, _tpl, steps=_steps, weighted=_wt,
                    max_deg=_md, min_ncc=_mn)
                return w, np.asarray(bad, dtype=bool), rot, sh
            logger.info("Registration template loaded")
        except Exception:                                       # noqa: BLE001
            logger.warning("Registration was unavailable; the canonical frame "
                           "is used instead.")
            register = None

    def select(X, names):
        return X[:, [idx_of[n] for n in names]]

    # Each per-hemisphere geometry costs two extra resamplings per scan, so only
    # the geometries the shipped stack actually uses are computed — and each is
    # computed once even if several members share it.
    hemi_geoms = sorted({(float(c["spec"]["hemi_spacing"]),
                          tuple(int(v) for v in c["spec"]["hemi_shape"]))
                         for c in cnns if c["spec"].get("input") == "hemi"})
    # Whole-crop members may also disagree about extent now (96 mm vs 128 mm).
    # 96 mm / 48 voxels always has to be produced: the feature models read it.
    box_geoms = sorted({(96.0, 48)}
                       | {(float(c["spec"].get("crop_mm", 96.0)),
                           int(c["spec"].get("out_vox", 48)))
                          for c in cnns if c["spec"].get("input") != "hemi"}
                       | {(float(c["spec"]["crops2_mm"]), int(c["spec"]["crops2_vox"]))
                          for c in cnns if c["spec"].get("crops2_mm")})

    need_reg1 = any(c["spec"].get("key") == "aligned_reg1" for c in cnns)
    need_f2 = any(c["spec"].get("key") == "aligned_f2" for c in cnns)

    preds = {}
    warned_scan = warned_cnn = warned_reg = False
    for i in range(0, len(uids), BLOCK):
        block = uids[i:i + BLOCK]
        rows, boxes, hm, hr, ok, main, f2 = [], [], [], [], [], [], []
        for uid in block:
            try:
                per_box = {}
                for cm, nv in box_geoms:
                    main_box = (cm, nv) == (96.0, 48)
                    dd = pp.preprocess_full(
                        NIFTI_DIR / f"{uid}.nii.gz", crop_mm=cm, out_vox=nv,
                        out_spacing=cm / nv,
                        hemi_geoms=(hemi_geoms or None) if main_box else None,
                        want_f2=bool(need_f2 and main_box))
                    per_box[f"{cm:g}:{nv}"] = dd
                    if main_box:
                        d = dd
                # features are extracted after the block is assembled, because
                # the registration is estimated for the whole block at once
                main.append((d["aligned"], d["sep"], d["meta"]))
                if need_f2:
                    f2.append(np.asarray(d.get("aligned_f2", d["aligned"]), dtype=np.float32))
                boxes.append(per_box)
                if hemi_geoms:
                    hm.append(d["hemi"])
                    hr.append(d["hemi_raw"])
                ok.append(uid)
            except Exception:                                   # noqa: BLE001
                if not warned_scan:
                    logger.warning("One or more scans could not be processed; "
                                   "the prior probability is used for those.")
                    warned_scan = True
                preds[uid] = prior
        if not ok:
            continue

        # Registration: estimate each scan's residual pose against the frozen
        # template and read the features in that frame. Per-sample independent —
        # the fit is a sum of per-scan terms, so batching is only vectorisation.
        # If it fails for any reason the canonical frame is used, which is what
        # v3 and v5 shipped, so this degrades rather than breaks.
        canon = np.stack([m[0] for m in main])
        feat_src = canon
        reg_bad = np.zeros(len(ok), dtype=bool)
        reg_rot = reg_sh = None
        if register is not None:
            try:
                feat_src, reg_bad, reg_rot, reg_sh = register(canon)
            except Exception:                                   # noqa: BLE001
                if not warned_reg:
                    logger.warning("Registration failed for a block; the "
                                   "canonical frame is used there.")
                    warned_reg = True
                feat_src = canon
        for j in range(len(ok)):
            f = features13.extract(np.asarray(feat_src[j], dtype=np.float32),
                                   sep_mm=main[j][1], extra=main[j][2])
            rows.append([f.get(k, np.nan) for k in order])

        X = np.asarray(rows, dtype=np.float64)
        comp = {}
        for name, lin in linears.items():
            comp[name] = P.predict_linear(lin, select(X, lin["features"]))
        for name, bs in gbms.items():
            comp[name] = np.mean([P.predict_lgbm(b, med, select(X, fn))
                                  for (b, med, fn) in bs], axis=0)
        for name, ns in mlps.items():
            comp[name] = np.mean([P.predict_mlp(n, select(X, n["features"]))
                                  for n in ns], axis=0)
        crops = {}
        for cm, nv in box_geoms:
            bk = f"{cm:g}:{nv}"
            crops[f"aligned@{bk}"] = np.stack([b[bk]["aligned"] for b in boxes])
            crops[f"raw@{bk}"] = np.stack([b[bk]["raw"] for b in boxes])
        for sp_, shape_ in hemi_geoms:
            k = pp.geom_key(sp_, shape_)
            crops[f"hemi@{k}"] = np.stack([h[k] for h in hm])
            crops[f"hemi_raw@{k}"] = np.stack([h[k] for h in hr])
        # v13 (RESEARCH_NOTES §73p): a member reading `aligned_reg1` gets the
        # registered crop resampled ONCE from the native volume with the pose
        # the feature path just fitted (rejected fits keep the canonical crop,
        # exactly as the training cache did). Same call as cache_rg1.py.
        if need_reg1 and reg_rot is not None:
            reg1 = []
            for j, uid in enumerate(ok):
                try:
                    if reg_bad[j]:
                        reg1.append(np.asarray(main[j][0], dtype=np.float32))
                    else:
                        dd = pp.preprocess_full(
                            NIFTI_DIR / f"{uid}.nii.gz",
                            refine=lambda _c, r=reg_rot[j], t=reg_sh[j]: (r, t))
                        reg1.append(np.asarray(dd["aligned_reg1"], dtype=np.float32))
                except Exception:                               # noqa: BLE001
                    reg1.append(np.asarray(main[j][0], dtype=np.float32))
            crops["aligned_reg1@96:48"] = np.stack(reg1)
        elif need_reg1:
            crops["aligned_reg1@96:48"] = crops["aligned@96:48"]
        if need_f2:
            # v14 (§76g): the 2D member trained on guarded-frame crops reads them
            crops["aligned_f2@96:48"] = np.stack(f2) if f2 else crops["aligned@96:48"]
        if cnns:
            try:
                for cnn in cnns:
                    if cnn["spec"].get("kind") == "cnn2d":
                        comp[cnn["spec"]["tag"]] = _cnn2d_prob(cnn, crops)
                    else:
                        comp[cnn["spec"]["tag"]] = _cnn_prob(cnn, crops, comp)
            except Exception:                                   # noqa: BLE001
                if not warned_cnn:
                    logger.warning("Neural-network inference was unavailable; "
                                   "the feature models are used alone.")
                    warned_cnn = True

        missing = [m for m in members if m not in comp]
        if missing:
            # renormalise the stacker over the members we actually have
            keep = [j for j, m in enumerate(members) if m in comp]
            Z = np.column_stack([P.to_logit(comp[members[j]]) for j in keep])
            w = coef[keep]
            s = float(np.abs(coef).sum())
            w = w * (s / max(float(np.abs(w).sum()), 1e-9))
            p = P.stack(Z, w, intercept)
        else:
            Z = np.column_stack([P.to_logit(comp[m]) for m in members])
            p = P.stack(Z, coef, intercept)

        if cal_T != 1.0:
            p = 1.0 / (1.0 + np.exp(-np.clip(P.to_logit(p) / cal_T, -30, 30)))
        # governor (v11, RESEARCH_NOTES 69a): a scan whose registration was
        # rejected fed canonical-frame crops to registered-frame models; on every
        # battery cell the prior beats that mismatched prediction
        if bool(cfg.get("governor", {}).get("reject_to_prior")):
            p = np.where(reg_bad, prior, p)
        # a non-finite probability would be scored as a total miss; fall back to
        # the prior for those rows rather than shipping a NaN
        p = np.where(np.isfinite(p), p, prior)
        lo = max(cal_clip, 1e-6)
        for uid, v in zip(ok, p):
            preds[uid] = float(np.clip(v, lo, 1 - lo))

    sub["is_pathologic"] = [preds.get(uid, prior) for uid in uids]
    sub.to_csv(OUT_PATH)
    logger.success("Wrote submission.csv")


if __name__ == "__main__":
    main()
