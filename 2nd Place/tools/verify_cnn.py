"""verify_cnn.py -- fidelity checks for the CNN stage (caches, retrains, export).

Prints and stores AGGREGATES ONLY (counts, hashes, max/mean absolute
differences, log loss, AUROC, correlations); never a per-scan value or uid.
Results are also written to outputs/chk_verify_<command>.json.

Commands
--------
cache-b   Rebuilt outputs/crops_v3_b.npz (pipeline/04_caches.sh, DAT_MASK_V2=0)
          1. structure (keys, shapes, dtypes, row order vs labels.csv/folds.csv);
          2. per-fold training mean/sd of log1p(clip(aligned, 0)), computed
             exactly as src/cnn3.py run_fold() does, vs the mu/sd stored in the
             shipped checkpoints <tag>_fold{k}.pt and mf*_full0.pt;
          3. replay: each shipped fold checkpoint of the six 5-fold runs predicts
             its held-out fold on the rebuilt cache (CPU, no flip-TTA, which is
             exact for HemiMIL), compared with the archived oof_<tag>.csv
             (--ref-dir). Mirrors experiments/20260914/replay_cnn_oof.py of the
             original project, which matched the original cache to <=6e-7.
          4. side outputs qc_v3_b.csv / features_v4_b.csv vs archived copies.
cache-f2  Rebuilt outputs/crops_v3_f2.npz (DAT_MASK_V2=1 DAT_FRAME_V2=1)
          replay of the shipped TorchScript fold models of e2dmipf2, e2dmipf22
          and e2db3f2 on their held-out folds (flip-TTA, as cnn2d.predict), vs
          the archived oof_<tag>.csv. Mirrors experiments/v16/replay_2d.py of the
          original project (which matched to <=1.4e-5). Side outputs
          features_v{4,5,6}_f2.csv / qc_v3_f2.csv vs archived copies.
spot      Fold-0 spot-check retrains (outputs/chk_mil48a_fold0.pt,
          outputs/chk_e2dmipf2_fold0.pt): their fold-0 log loss / AUROC vs the
          original run's fold-0 rows and vs the other seeds of the same recipe
          (seed spread), plus agreement (correlation, mean |dp|) with the
          original fold-0 predictions and the stored mu/sd of the 3D model.
ts        TorchScript exports outputs/<tag>_<part>.ts.pt vs the shipped
          <shipped_tag>_<part>.ts.pt: max |logit difference| on cached images.
          Pairs are given as TAG=SHIPPED_TAG (e.g. chk_orig_e2dmipf2=e2dmipf2).

Arguments
---------
--assets DIR   directory with the shipped v17 assets (*.pt, *.ts.pt,
               inference_config.json); default inference/assets
--ref-dir DIR  directory with the archived original outputs (oof_<tag>.csv,
               qc/feature tables); comparisons needing it are skipped if absent
--device       cpu | mps (default: cpu for 3D replay, mps if available for 2D)

Usage (repo root):
    python tools/verify_cnn.py cache-b  --assets A --ref-dir R
    python tools/verify_cnn.py cache-f2 --assets A --ref-dir R
    python tools/verify_cnn.py spot     --assets A --ref-dir R
    python tools/verify_cnn.py ts --assets A --pairs e2dmipf2=e2dmipf2 ...
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))

RUNS3D = ["mil48a", "mil48b", "m31000", "m33000", "m35000", "m37000"]
FULL3D = ["mf21000", "mf23000", "mf25000", "mf27000"]
RUNS2D = {"e2dmipf2": 192, "e2dmipf22": 192, "e2db3f2": 224}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def met(y, p, clip=1e-6):
    q = np.clip(np.asarray(p, float), clip, 1 - clip)
    return {"logloss": float(log_loss(y, q)), "auroc": float(roc_auc_score(y, q))}


def diff(a, b, tol=1e-4):
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    return {"max_abs": float(d.max()), "mean_abs": float(d.mean()), f"n_gt_{tol:g}": int((d > tol).sum())}


def labels_path() -> Path:
    data = Path(os.environ.get("DAT_DATA_DIR", str(ROOT / "data" / "raw")))
    return data / "labels.csv"


def load_cache(name: str):
    z = np.load(OUT / name, allow_pickle=True)
    info = {"file": name, "sha256": sha(OUT / name), "keys": sorted(z.files)}
    for k in z.files:
        info[f"{k}_shape"] = list(z[k].shape)
        info[f"{k}_dtype"] = str(z[k].dtype)
    uids = z["uids"].astype(str)
    lab = pd.read_csv(labels_path())
    info["row_order_equals_labels_csv"] = bool(np.array_equal(uids, lab["uid"].astype(str).to_numpy()))
    info["y_equals_labels_csv"] = bool(np.array_equal(z["y"].astype(int), lab["is_pathologic"].astype(int).to_numpy()))
    folds = pd.read_csv(OUT / "folds.csv").set_index("uid").loc[uids].reset_index()
    info["folds_cover_all_rows"] = bool(len(folds) == len(uids))
    return z, uids, folds["fold"].to_numpy(), z["y"].astype(np.float32), info


def ref_oof(ref: Path | None, tag: str, uids):
    if ref is None or not (ref / f"oof_{tag}.csv").exists():
        return None
    d = pd.read_csv(ref / f"oof_{tag}.csv").set_index("uid")
    return d.loc[uids, tag].to_numpy(float)


def compare_tables(ref: Path | None, names):
    """sha256 + numeric column diffs (aggregate) of cache3 side outputs."""
    res = {}
    for n in names:
        a, b = OUT / n, (ref / n if ref else None)
        if not a.exists() or b is None or not b.exists():
            res[n] = "skipped (missing)"
            continue
        r = {"sha256_equal": sha(a) == sha(b)}
        da, db = pd.read_csv(a).set_index("uid"), pd.read_csv(b).set_index("uid")
        r["n_rows_new_ref"] = [len(da), len(db)]
        r["same_uid_set"] = bool(set(da.index) == set(db.index))
        common = [c for c in db.columns if c in da.columns]
        r["n_cols_new_ref_common"] = [da.shape[1], db.shape[1], len(common)]
        r["cols_only_in_new"] = [c for c in da.columns if c not in db.columns][:20]
        r["cols_only_in_ref"] = [c for c in db.columns if c not in da.columns][:20]
        db = db.loc[da.index.intersection(db.index)]
        da = da.loc[db.index]
        worst, ndiff, maxd = [], 0, 0.0
        for c in common:
            x = pd.to_numeric(da[c], errors="coerce").to_numpy(float)
            y = pd.to_numeric(db[c], errors="coerce").to_numpy(float)
            both_nan = np.isnan(x) & np.isnan(y)
            d = np.where(both_nan, 0.0, np.abs(x - y))
            d = np.where(np.isnan(d), np.inf, d)
            m = float(d.max()) if len(d) else 0.0
            if m > 1e-6:
                ndiff += 1
                worst.append((m, c, int((d > 1e-6).sum())))
            maxd = max(maxd, m)
        worst.sort(reverse=True)
        r["n_common_cols_with_diff_gt_1e-6"] = ndiff
        r["max_abs_diff_all_cols"] = maxd
        r["worst_cols(max_abs, name, n_rows_diff)"] = [(float(m), c, k) for m, c, k in worst[:8]]
        res[n] = r
    return res


# --------------------------------------------------------------------------- #
def cmd_cache_b(a):
    import torch
    import cnn3
    from nets3 import build
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    t0 = time.time()
    assets, ref = Path(a.assets), (Path(a.ref_dir) if a.ref_dir else None)
    z, uids, fold, y, info = load_cache("crops_v3_b.npz")
    out = {"structure": info}
    sbr = z["aligned"].astype(np.float32)                     # cnn3.main
    x = cnn3.to_input(sbr, "log")
    del sbr, z
    gc.collect()

    # --- normalisation constants, exactly as cnn3.run_fold -------------------
    norm, worst = {}, 0.0
    for k in range(5):
        tr = np.where(fold != k)[0]
        mu, sd = float(x[tr].mean()), float(x[tr].std() + 1e-6)
        row = {"mu_rebuilt": mu, "sd_rebuilt": sd}
        for tag in RUNS3D:
            ck = torch.load(assets / f"{tag}_fold{k}.pt", map_location="cpu", weights_only=False)
            row[f"{tag}_dmu"] = ck["mu"] - mu
            row[f"{tag}_dsd"] = ck["sd"] - sd
            worst = max(worst, abs(ck["mu"] - mu), abs(ck["sd"] - sd))
        row["shipped_mu"], row["shipped_sd"] = ck["mu"], ck["sd"]
        norm[f"fold{k}"] = row
    allidx = np.arange(len(y))
    mu, sd = float(x[allidx].mean()), float(x[allidx].std() + 1e-6)
    row = {"mu_rebuilt": mu, "sd_rebuilt": sd}
    for tag in FULL3D:
        ck = torch.load(assets / f"{tag}_full0.pt", map_location="cpu", weights_only=False)
        row[f"{tag}_dmu"], row[f"{tag}_dsd"] = ck["mu"] - mu, ck["sd"] - sd
        worst = max(worst, abs(ck["mu"] - mu), abs(ck["sd"] - sd))
    row["shipped_mu"], row["shipped_sd"] = ck["mu"], ck["sd"]
    norm["full"] = row
    norm["max_abs_diff_over_34_checkpoints"] = worst
    out["normalisation"] = norm
    print(f"[cache-b] mu/sd: max |shipped - rebuilt| over 34 checkpoints = {worst:.3e}")
    for key in [f"fold{k}" for k in range(5)] + ["full"]:
        r = norm[key]
        print(f"   {key:<6} rebuilt mu {r['mu_rebuilt']:.6f} sd {r['sd_rebuilt']:.6f} | "
              f"shipped mu {r['shipped_mu']:.6f} sd {r['shipped_sd']:.6f}")

    # --- replay of the shipped fold checkpoints ------------------------------
    if not a.no_replay:
        rep, preds = {}, {}
        for tag in RUNS3D:
            p = np.zeros(len(y))
            for k in range(5):
                va = np.where(fold == k)[0]
                ck = torch.load(assets / f"{tag}_fold{k}.pt", map_location="cpu", weights_only=False)
                net = build(ck["arch"], in_ch=ck["in_ch"], drop=ck.get("drop", 0.3))
                net.load_state_dict(ck["state"])
                net.eval()
                xb = torch.from_numpy((x[va] - ck["mu"]) / ck["sd"]).float()
                p[va] = cnn3.predict(net, xb, bs=16, tta=False)
                del net, xb
            preds[tag] = p
            r = {"replayed": met(y, p)}
            rec = ref_oof(ref, tag, uids)
            if rec is not None:
                r["recorded"] = met(y, rec)
                r["replay_vs_recorded"] = diff(p, rec)
            rep[tag] = r
            print(f"[cache-b] replay {tag}: ll {r['replayed']['logloss']:.5f} auc {r['replayed']['auroc']:.5f}"
                  + (f" | recorded ll {r['recorded']['logloss']:.5f} | max|dp| "
                     f"{r['replay_vs_recorded']['max_abs']:.2e} mean {r['replay_vs_recorded']['mean_abs']:.2e} "
                     f"n>1e-4 {r['replay_vs_recorded']['n_gt_0.0001']}" if "recorded" in r else ""), flush=True)
        for n, name in ((4, "mil48y"), (6, "mil48y6")):
            m = np.clip(np.mean([preds[t] for t in RUNS3D[:n]], axis=0), 1e-6, 1 - 1e-6)
            r = {"replayed": met(y, m)}
            rec = ref_oof(ref, name, uids)
            if rec is not None:
                r["recorded"] = met(y, rec)
                r["replay_vs_recorded"] = diff(m, rec)
            rep[f"{n}-run mean"] = r
            print(f"[cache-b] {n}-run mean: ll {r['replayed']['logloss']:.5f} auc {r['replayed']['auroc']:.5f}"
                  + (f" | vs archived oof_{name}.csv max|dp| {r['replay_vs_recorded']['max_abs']:.2e}"
                     if "recorded" in r else ""))
        out["replay"] = rep
    out["side_outputs"] = compare_tables(ref, ["qc_v3_b.csv", "features_v4_b.csv",
                                              "features_v5_b.csv", "features_v6_b.csv"])
    for n, r in out["side_outputs"].items():
        print(f"[cache-b] {n}: " + (r if isinstance(r, str) else
              f"sha256_equal {r['sha256_equal']} common cols {r['n_cols_new_ref_common']} "
              f"cols with diff {r['n_common_cols_with_diff_gt_1e-6']} max {r['max_abs_diff_all_cols']:.3e}"))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def cmd_cache_f2(a):
    import torch
    import cnn2d
    t0 = time.time()
    assets, ref = Path(a.assets), (Path(a.ref_dir) if a.ref_dir else None)
    dev = torch.device(a.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    if dev.type == "mps":
        torch.mps.set_per_process_memory_fraction(0.35)
    z, uids, fold, y, info = load_cache("crops_v3_f2.npz")
    out = {"structure": info, "device": str(dev)}
    crops = z["aligned"].astype(np.float32)
    del z
    cfg = {}
    if (assets / "inference_config.json").exists():
        cfg = {s["tag"]: s for s in json.loads((assets / "inference_config.json").read_text())["cnns"]}
    rep, preds = {}, {}
    for tag, res in RUNS2D.items():
        spec = cfg.get("e2dmipf2" if tag == "e2dmipf22" else tag, {})
        res = int(spec.get("res", res))
        slabs = tuple(map(tuple, spec.get("slabs", cnn2d.SLABS)))
        p = np.zeros(len(y))
        for k in range(5):
            ids = np.where(fold == k)[0]
            net = torch.jit.load(str(assets / f"{tag}_fold{k}.ts.pt"), map_location=dev).eval()
            X = torch.from_numpy(cnn2d.make_images(crops[ids], slabs=slabs, views=spec.get("views", "mip3")))
            p[ids] = cnn2d.predict(net, X, res, dev, bs=24)
            del net, X
            gc.collect()
            if dev.type == "mps":
                torch.mps.empty_cache()
        preds[tag] = p
        r = {"replayed": met(y, p), "replayed_clip0.005": met(y, p, 0.005)}
        rec = ref_oof(ref, tag, uids)
        if rec is not None:
            r["recorded"] = met(y, rec)
            r["recorded_clip0.005"] = met(y, rec, 0.005)
            r["replay_vs_recorded"] = diff(p, rec)
        rep[tag] = r
        print(f"[cache-f2] replay {tag}: ll {r['replayed']['logloss']:.5f} auc {r['replayed']['auroc']:.5f}"
              + (f" | recorded ll {r['recorded']['logloss']:.5f} auc {r['recorded']['auroc']:.5f} | max|dp| "
                 f"{r['replay_vs_recorded']['max_abs']:.2e} mean {r['replay_vs_recorded']['mean_abs']:.2e} "
                 f"n>1e-4 {r['replay_vs_recorded']['n_gt_0.0001']}" if "recorded" in r else ""), flush=True)
    m = 0.5 * (preds["e2dmipf2"] + preds["e2dmipf22"])
    r = {"replayed": met(y, m)}
    rec = ref_oof(ref, "e2dmipf2x", uids)
    if rec is not None:
        r["recorded"] = met(y, rec)
        r["replay_vs_recorded"] = diff(m, rec)
    rep["e2dmipf2x (2-seed mean)"] = r
    print(f"[cache-f2] e2dmipf2x: ll {r['replayed']['logloss']:.5f} auc {r['replayed']['auroc']:.5f}"
          + (f" | vs archived max|dp| {r['replay_vs_recorded']['max_abs']:.2e}" if "recorded" in r else ""))
    out["replay"] = rep
    out["side_outputs"] = compare_tables(ref, ["qc_v3_f2.csv", "features_v4_f2.csv",
                                              "features_v5_f2.csv", "features_v6_f2.csv"])
    for n, r in out["side_outputs"].items():
        print(f"[cache-f2] {n}: " + (r if isinstance(r, str) else
              f"sha256_equal {r['sha256_equal']} common cols {r['n_cols_new_ref_common']} "
              f"cols with diff {r['n_common_cols_with_diff_gt_1e-6']} max {r['max_abs_diff_all_cols']:.3e}"))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cmd_spot(a):
    import torch
    t0 = time.time()
    assets, ref = Path(a.assets), (Path(a.ref_dir) if a.ref_dir else None)
    out = {}
    # ---- 3D: chk_mil48a fold 0 vs mil48a fold 0 (seed 3000) -----------------
    tag3 = a.tag3d
    if (OUT / f"{tag3}_fold0.pt").exists():
        import cnn3
        from nets3 import build
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
        z, uids, fold, y, _ = load_cache("crops_v3_b.npz")
        va = np.where(fold == 0)[0]
        x = cnn3.to_input(z["aligned"][va].astype(np.float32), "log")
        del z
        ck = torch.load(OUT / f"{tag3}_fold0.pt", map_location="cpu", weights_only=False)
        sh = torch.load(assets / "mil48a_fold0.pt", map_location="cpu", weights_only=False)
        net = build(ck["arch"], in_ch=ck["in_ch"], drop=ck.get("drop", 0.3))
        net.load_state_dict(ck["state"])
        p = cnn3.predict(net, torch.from_numpy((x - ck["mu"]) / ck["sd"]).float(), bs=16, tta=False)
        r = {"retrained_fold0": met(y[va], p), "mu_sd_retrained": [ck["mu"], ck["sd"]],
             "mu_sd_shipped": [sh["mu"], sh["sd"]],
             "mu_sd_equal": bool(ck["mu"] == sh["mu"] and ck["sd"] == sh["sd"]),
             "state_tensors_equal": bool(all(torch.equal(ck["state"][k], sh["state"][k]) for k in sh["state"]))}
        seeds = {}
        for t in RUNS3D:
            rec = ref_oof(ref, t, uids)
            if rec is not None:
                seeds[t] = met(y[va], rec[va])
        r["original_fold0_by_seed"] = seeds
        rec = ref_oof(ref, "mil48a", uids)
        if rec is not None:
            r["agreement_with_original_mil48a_fold0"] = {
                "logit_corr": float(np.corrcoef(_logit(p), _logit(rec[va]))[0, 1]), **diff(p, rec[va], 0.05)}
        if seeds:
            lls = [v["logloss"] for v in seeds.values()]
            r["seed_spread_fold0_logloss"] = [min(lls), max(lls)]
        out["3d"] = r
        print(f"[spot] 3D {tag3} fold0: ll {r['retrained_fold0']['logloss']:.4f} auc {r['retrained_fold0']['auroc']:.4f}"
              f" | original mil48a fold0 " + (f"{seeds['mil48a']['logloss']:.4f}/{seeds['mil48a']['auroc']:.4f}" if 'mil48a' in seeds else "n/a")
              + (f" | seed spread {r['seed_spread_fold0_logloss'][0]:.4f}-{r['seed_spread_fold0_logloss'][1]:.4f}" if seeds else "")
              + f" | mu/sd equal {r['mu_sd_equal']} | weights identical {r['state_tensors_equal']}")
        if "agreement_with_original_mil48a_fold0" in r:
            g = r["agreement_with_original_mil48a_fold0"]
            print(f"[spot] 3D agreement with original fold-0 predictions: logit corr {g['logit_corr']:.4f} "
                  f"mean|dp| {g['mean_abs']:.4f} max|dp| {g['max_abs']:.4f}")
        del net, x
        gc.collect()
    else:
        print(f"[spot] 3D: outputs/{tag3}_fold0.pt not found -- skipped")
    # ---- 2D: chk_e2dmipf2 fold 0 vs e2dmipf2 fold 0 (seed 9000) -------------
    tag2 = a.tag2d
    if (OUT / f"{tag2}_fold0.pt").exists():
        import cnn2d
        dev = torch.device(a.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        z, uids, fold, y, _ = load_cache("crops_v3_f2.npz")
        va = np.where(fold == 0)[0]
        X = torch.from_numpy(cnn2d.make_images(z["aligned"][va], views="mip3"))
        del z
        ck = torch.load(OUT / f"{tag2}_fold0.pt", map_location="cpu", weights_only=False)
        net = cnn2d.build_model(dict(backbone=ck["backbone"], drop=ck.get("drop", 0.3),
                                     hemi=bool(ck.get("hemi", False))), pretrained=False)
        net.load_state_dict(ck["state"])
        net = net.to(dev)
        p = cnn2d.predict(net, X, int(ck["res"]), dev, bs=24)
        r = {"retrained_fold0": met(y[va], p), "device": str(dev)}
        seeds = {}
        for t in RUNS2D:
            rec = ref_oof(ref, t, uids)
            if rec is not None:
                seeds[t] = met(y[va], rec[va])
        r["original_fold0_by_run"] = seeds
        rec = ref_oof(ref, "e2dmipf2", uids)
        if rec is not None:
            r["agreement_with_original_e2dmipf2_fold0"] = {
                "logit_corr": float(np.corrcoef(_logit(p), _logit(rec[va]))[0, 1]), **diff(p, rec[va], 0.05)}
        out["2d"] = r
        print(f"[spot] 2D {tag2} fold0: ll {r['retrained_fold0']['logloss']:.4f} auc {r['retrained_fold0']['auroc']:.4f}"
              + "".join(f" | {t} fold0 {v['logloss']:.4f}/{v['auroc']:.4f}" for t, v in seeds.items()))
        if "agreement_with_original_e2dmipf2_fold0" in r:
            g = r["agreement_with_original_e2dmipf2_fold0"]
            print(f"[spot] 2D agreement with original fold-0 predictions: logit corr {g['logit_corr']:.4f} "
                  f"mean|dp| {g['mean_abs']:.4f} max|dp| {g['max_abs']:.4f}")
    else:
        print(f"[spot] 2D: outputs/{tag2}_fold0.pt not found -- skipped")
    out["seconds"] = round(time.time() - t0, 1)
    return out


def cmd_ts(a):
    import torch
    import cnn2d
    t0 = time.time()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    assets = Path(a.assets)
    z = np.load(OUT / "crops_v3_f2.npz", allow_pickle=True)
    n = min(a.n_images, len(z["uids"]))
    X = torch.from_numpy(cnn2d.make_images(z["aligned"][:n], views="mip3"))
    del z
    dev = torch.device("cpu")
    out = {"n_images": n}
    for pair in a.pairs:
        tag, shipped = pair.split("=")
        res = RUNS2D.get(shipped, 192)
        for part in ["fold0", "fold1", "fold2", "fold3", "fold4", "full0"]:
            fa, fb = OUT / f"{tag}_{part}.ts.pt", assets / f"{shipped}_{part}.ts.pt"
            if not fa.exists():
                out[f"{tag}_{part}"] = "missing"
                continue
            ma = torch.jit.load(str(fa), map_location="cpu").eval()
            mb = torch.jit.load(str(fb), map_location="cpu").eval()
            with torch.no_grad():
                xa = cnn2d.prep(X, res, dev)
                la, lb = ma(xa).squeeze(1).numpy(), mb(xa).squeeze(1).numpy()
            pa = {k: v for k, v in ma.state_dict().items()}
            r = {"sha256_equal": sha(fa) == sha(fb), "max_abs_logit_diff": float(np.abs(la - lb).max()),
                 "mean_abs_logit_diff": float(np.abs(la - lb).mean()), "n_state_tensors": len(pa)}
            out[f"{tag}_{part}"] = r
            print(f"[ts] {tag}_{part} vs shipped {shipped}_{part}: sha256_equal {r['sha256_equal']} "
                  f"max|dlogit| {r['max_abs_logit_diff']:.2e} mean {r['mean_abs_logit_diff']:.2e}", flush=True)
            del ma, mb
    out["seconds"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=["cache-b", "cache-f2", "spot", "ts"])
    ap.add_argument("--assets", default=str(ROOT / "inference" / "assets"))
    ap.add_argument("--ref-dir", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-replay", action="store_true")
    ap.add_argument("--tag3d", default="chk_mil48a")
    ap.add_argument("--tag2d", default="chk_e2dmipf2")
    ap.add_argument("--pairs", nargs="*", default=["e2dmipf2=e2dmipf2", "e2dmipf22=e2dmipf22", "e2db3f2=e2db3f2"])
    ap.add_argument("--n-images", type=int, default=256)
    ap.add_argument("--out-name", default=None, help="JSON name (default chk_verify_<command>.json)")
    a = ap.parse_args()
    fn = {"cache-b": cmd_cache_b, "cache-f2": cmd_cache_f2, "spot": cmd_spot, "ts": cmd_ts}[a.command]
    res = fn(a)
    name = a.out_name or f"chk_verify_{a.command.replace('-', '_')}.json"
    (OUT / name).write_text(json.dumps(res, indent=2, default=str))
    print(f"[verify_cnn] {a.command} done in {res.get('seconds')} s -> outputs/{name}")


if __name__ == "__main__":
    main()
