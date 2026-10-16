"""
build3.py — assemble submission_src/ and pack submission.zip for the runtime.

Copies only the modules the inference path actually imports, writes
`inference_config.json` from the fitted stacker, copies the CNN checkpoints of
the members the stacker kept, and packs `main.py` at the archive root as the
runtime requires.
"""
from __future__ import annotations
import argparse, json, shutil, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
# Competition data (niftis/ + labels.csv). Patched for the prize repo; was ROOT/"data-demo".
import os  # noqa: E402  (prize-repo patch)
DATA = Path(os.environ.get("DAT_DATA_DIR", str(ROOT / "data" / "raw")))
SRC = ROOT / "src"
OUT = ROOT / "outputs"
DEST = ROOT / "submission_src"
ASSETS = DEST / "assets"

# Every module main.py imports, transitively. features6 imports features5 which
# imports features4, so all three must ship even though main.py names only one.
# roigeom owns the ROI-placement bounds that features4 and features5 both clip
# through; if it is ever dropped from this list the package will not import.
CODE = ["preprocessing3.py", "roigeom.py", "features4.py", "features5.py",
        "features6.py",
        "features13.py", "portable3.py", "nets3.py", "cnn3.py", "cnn4.py",
        # registration to the study template: regfit is what main.py calls,
        # register3 owns the transform and the objective it calls into
        "register3.py", "regfit.py",
        # v12: 2D pretrained-backbone member (imports timm lazily; timm ships in
        # the runtime image)
        "cnn2d.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ensemble", default="ensemble_v3.json")
    ap.add_argument("--cnn-tags", nargs="*", default=[],
                    help="member tags; use NAME=tagA+tagB to ship several training "
                         "runs of one architecture as a single averaged member")
    ap.add_argument("--zip", default="submission.zip")
    ap.add_argument("--fp16", action="store_true", help="store CNN weights as fp16")
    ap.add_argument("--key2d", nargs="*", default=[], help="cnn2d member crop key, NAME=KEY (v14: e2dmipf2=aligned_f2)")
    ap.add_argument("--crops2d", nargs="*", default=[], help="cnn2d member verify cache, NAME=FILE (v14: e2dmipf2=crops_v3_f2.npz)")
    ap.add_argument("--use-full", action="store_true",
                    help="also ship `<tag>_full0.pt`, a model trained on 100%% of "
                         "the rows. Legitimate (the schedule has no "
                         "validation-dependent step) and it sees 25%% more data, "
                         "but it CANNOT appear in the OOF the stacker was fitted "
                         "on — so it is opt-in, never automatic.")
    ap.add_argument("--tta", type=int, default=1,
                    help="inference TTA level: 0 none, 1 the label's exact "
                         "symmetry, 2 also two blur levels")
    ap.add_argument("--template", default=None,
                    help="ship this template (outputs/<name>.npy) and register "
                         "every scan to it before reading features. Omit to keep "
                         "the v3/v5 canonical frame exactly as it was.")
    ap.add_argument("--reg-steps", type=int, nargs=2, default=[120, 80])
    args = ap.parse_args()

    ens = json.loads((OUT / args.ensemble).read_text())
    members = ens["members"]
    coef = np.array(ens["params"]["coef"], dtype=float)
    intercept = float(ens["params"]["intercept"])
    if ens["params"]["kind"] in ("mean", "weights"):
        # a plain weighted average of logits has no intercept term to learn;
        # keep the same interface so main.py has a single code path
        coef = coef / max(coef.sum(), 1e-9)
    if len(coef) != len(members):
        raise SystemExit(f"stacker emitted {len(coef)} coefficients for "
                         f"{len(members)} members — refusing to build")

    # drop members the stacker zeroed out, so we do not ship dead weight
    keep = [m for m, c in zip(members, coef) if abs(c) > 1e-6]
    if len(keep) != len(members):
        idx = [j for j, m in enumerate(members) if m in keep]
        members, coef = keep, coef[idx]

    DEST.mkdir(exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    # start from a clean tree: stale modules from an earlier build would still be
    # importable inside the container and are a silent source of drift
    keep_top = set(CODE) | {"main.py", "assets"}
    for p in DEST.iterdir():
        if p.name not in keep_top:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    for f in CODE:
        shutil.copy2(SRC / f, DEST / f)
    shutil.copy2(SRC / "main3.py", DEST / "main.py")

    feat_order = json.loads((OUT / "feature_order.json").read_text())
    union = json.loads((OUT / "feature_union.json").read_text())

    cnn_specs = []
    import torch
    for entry in args.cnn_tags:
        name, _, rhs = entry.partition("=")
        sources = rhs.split("+") if rhs else [name]
        if name not in members:
            continue
        # One member, possibly several training runs. Averaging the checkpoints'
        # predictions cuts seed variance without giving the stacker another weight
        # to fit — separate members of the same architecture correlate ~0.99 and
        # measurably hurt the stack (RESEARCH_NOTES 5b).
        pairs, ck = [], None
        for src in sources:
            # `full0` is a model trained on 100% of the rows: the schedule has no
            # validation-dependent step, so it is legitimate, and it sees 25% more
            # data than any fold model. It contributes to the shipped member even
            # though it cannot contribute to the OOF the stacker was fitted on.
            suffixes = [f"fold{j}" for j in range(5)]
            if args.use_full:
                suffixes.append("full0")
            for k in suffixes:
                q = OUT / f"{src}_{k}.pt"
                if not q.exists():
                    continue
                ck = torch.load(q, map_location="cpu", weights_only=False)
                if args.fp16:
                    ck["state"] = {kk: (v.half() if v.dtype == torch.float32 else v)
                                   for kk, v in ck["state"].items()}
                torch.save(ck, ASSETS / f"{src}_{k}.pt")
                pairs.append([src, k])
        if ck is None:
            raise SystemExit(f"no checkpoints found for member {name}")
        if ck.get("kind") == "cnn2d":
            _k2 = dict(x.split("=", 1) for x in args.key2d); _c2 = dict(x.split("=", 1) for x in args.crops2d)
            spec2 = dict(tag=name, sources=pairs, kind="cnn2d", key=_k2.get(name, "aligned"),
                         backbone=ck["backbone"], res=int(ck["res"]),
                         slabs=[list(v) for v in ck["slabs"]],
                         views=ck.get("views", "slabs"), hemi=bool(ck.get("hemi", False)),
                         crops=_c2.get(name, "crops_v3_b.npz"), crop_mm=96.0, out_vox=48)
            # TorchScript exports (tools/ts_export.py) replace the state-dict
            # checkpoints so the runtime needs no timm (unpinned in the image)
            ts = [OUT / f"{src}_{k}.ts.pt" for src, k in pairs]
            if ts and all(q.exists() for q in ts):
                for (src, k), q in zip(pairs, ts):
                    shutil.copy(q, ASSETS / q.name)
                    (ASSETS / f"{src}_{k}.pt").unlink(missing_ok=True)
                spec2["ts"] = True
                print(f"[build3] {name}: {len(ts)} TorchScript checkpoints (no timm at inference)")
            cnn_specs.append(spec2)
            continue
        spec = dict(tag=name, sources=pairs, arch=ck["arch"], size=ck["size"],
                    key=ck["key"], channels=ck["channels"], tta=args.tta,
                    crops=ck.get("crops", "crops_v3_b.npz"),
                    crop_mm=float(ck.get("crop_mm", 96.0)),
                    out_vox=int(ck.get("out_vox", 48)))
        if ck.get("offset"):
            # trained as a booster on a feature member's logit; main.py must add
            # that member's logit back at inference
            spec["offset_member"] = Path(ck["offset"]).stem.replace("offsets_", "")
        if ck.get("crops2_geom"):
            spec.update(crops2=ck["crops2"],
                        crops2_mm=float(ck["crops2_geom"][0]),
                        crops2_vox=int(ck["crops2_geom"][1]))
        if ck.get("input") == "hemi":
            spec.update(input="hemi", n_inst=int(ck["n_inst"]),
                        hemi_spacing=float(ck["hemi_spacing"]),
                        hemi_shape=[int(v) for v in ck["hemi_shape"]],
                        # unused at inference; it tells verify_submission which
                        # training cache is the right reference for this member
                        crops=ck.get("crops", "crops_hemi.npz"))
        cnn_specs.append(spec)

    prior = float(pd.read_csv(DATA / "labels.csv")["is_pathologic"].mean())
    cal = ens.get("calibration", {"T": 1.0, "clip": 0.0})
    cfg = dict(feature_order=union, members=members,
               stacker=dict(coef=coef.tolist(), intercept=intercept),
               calibration=cal,
               # v11 governor (RESEARCH_NOTES 69a): registration-rejected scans -> prior
               governor=ens.get("governor", {"reject_to_prior": False}),
               linears=[m for m in members if m in ("lin", "lin4")],
               gbms={m: [0, 1, 2, 3, 4] for m in members if m in ("gbm",)},
               mlps={m: [0, 1, 2] for m in members if m in ("mlp",)},
               cnns=cnn_specs, prior=prior,
               feature_sets=feat_order)
    if args.template:
        tpl_src = OUT / args.template
        if not tpl_src.exists():
            raise SystemExit(f"template {args.template} not found — run register3.py")
        shutil.copy2(tpl_src, ASSETS / args.template)
        # Take the rejection bounds from regfit itself rather than repeating them
        # here. They are not free parameters: RESEARCH_NOTES §37 measures that
        # tightening max_deg costs up to 0.0066, and the training tables were
        # built through regfit's defaults, so a second copy of these numbers in
        # this file is a train/test mismatch waiting to happen. It already
        # happened once.
        import regfit
        cfg["registration"] = dict(template=args.template,
                                   steps=list(args.reg_steps),
                                   max_deg=float(regfit.MAX_DEG),
                                   min_ncc=float(regfit.MIN_NCC))
    (ASSETS / "inference_config.json").write_text(json.dumps(cfg, indent=2))

    # Decide which assets this submission needs. They are pruned from the ZIP, not
    # from disk: deleting them would break a later build that ships a different
    # member set (train_final3.py --export would have to be re-run first).
    referenced = {"inference_config.json"}
    if args.template:
        referenced.add(args.template)
    referenced |= {f"linear_{m}.json" for m in cfg["linears"]}
    for m, seeds in cfg["gbms"].items():
        referenced |= {f"lgbm_{m}_s{s}.txt" for s in seeds}
        referenced |= {f"lgbm_{m}_s{s}.txt.meta.json" for s in seeds}
    for m, seeds in cfg["mlps"].items():
        referenced |= {f"mlp_{m}_s{s}.json" for s in seeds}
    for s in cnn_specs:
        ext = ".ts.pt" if s.get("ts") else ".pt"
        referenced |= {f"{src}_{k}{ext}" for src, k in s["sources"]}
    missing = [n for n in sorted(referenced) if not (ASSETS / n).exists()]
    if missing:
        raise SystemExit("missing assets (re-run train_final3.py --export): "
                         + ", ".join(missing))

    zpath = ROOT / args.zip
    if zpath.exists():
        zpath.unlink()
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in CODE + ["main.py"]:
            z.write(DEST / f, f)
        for n in sorted(referenced):
            z.write(ASSETS / n, f"assets/{n}")
    size = zpath.stat().st_size / 1e6
    print(f"[build3] members={members}")
    print(f"[build3] coef={np.round(coef,4).tolist()} intercept={intercept:.4f}")
    print(f"[build3] calibration T={cal['T']} clip={cal['clip']}")
    print(f"[build3] cnns={[s['tag'] for s in cnn_specs]}")
    print(f"[build3] {zpath.name}: {size:.1f} MB")


if __name__ == "__main__":
    main()
