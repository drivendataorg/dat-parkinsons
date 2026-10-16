"""
cache3.py — single pass over the training scans producing everything v3 needs:
aligned + unaligned SBR crops for the CNNs, the v4 clinical features, and QC.

One pass because preprocessing3 localisation is the expensive part (~180 ms/scan)
and all three products share it.
"""
from __future__ import annotations
import re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing3 import preprocess_full, OUT_VOX
import features4
import features5
import features6
import features13

ROOT = Path(__file__).resolve().parent.parent
# Competition data (niftis/ + labels.csv). Patched for the prize repo; was ROOT/"data-demo".
import os  # noqa: E402  (prize-repo patch)
DATA = Path(os.environ.get("DAT_DATA_DIR", str(ROOT / "data" / "raw")))
NIFTI = DATA / "niftis"
OUT = ROOT / "outputs"
Z = OUT_VOX


def process(uid: str) -> dict:
    try:
        d = preprocess_full(NIFTI / f"{uid}.nii.gz")
        # features come from the FULL-PRECISION crop, matching what main.py sees.
        # (The cached crop tensor is float16 for size; deriving features from that
        # instead would train the linear model on a slightly different input
        # distribution than inference produces.)
        feats = features13.extract(d["aligned"], sep_mm=d["sep"], extra=d["meta"])
        return dict(uid=uid, ok=True, aligned=d["aligned"].astype(np.float16),
                    raw=d["raw"].astype(np.float16), sep=d["sep"], feats=feats,
                    **d["meta"])
    except Exception as e:                                  # noqa: BLE001
        zeros = np.zeros((Z, Z, Z), np.float16)
        return dict(uid=uid, ok=False, aligned=zeros, raw=zeros, ref=1.0, sep=26.0,
                    feats=features13.extract(np.ones((Z, Z, Z), np.float32), 26.0),
                    head_ml=np.nan, inner_ml=np.nan, two_blob=0.0,
                    native_spacing=np.nan, err=repr(e)[:160])


def main(tag=""):
    labels = pd.read_csv(DATA / "labels.csv")
    uids = labels["uid"].tolist()
    y = labels["is_pathologic"].values.astype(np.float32)
    t0 = time.time()
    res = Parallel(n_jobs=3, batch_size=4, verbose=1)(delayed(process)(u) for u in uids)
    bad = [r for r in res if not r["ok"]]
    print(f"[cache3] {len(res)} scans in {time.time()-t0:.0f}s | failures: {len(bad)}")
    for r in bad[:10]:
        print("   FAIL", r["uid"], r.get("err"))

    np.savez_compressed(
        OUT / f"crops_v3{tag}.npz",
        aligned=np.stack([r["aligned"] for r in res]),
        raw=np.stack([r["raw"] for r in res]),
        refs=np.array([r["ref"] for r in res], np.float32),
        sep=np.array([r["sep"] for r in res], np.float32),
        uids=np.array(uids), y=y)

    feat_df = pd.DataFrame([r["feats"] for r in res])
    feat_df.insert(0, "uid", uids)
    feat_df["is_pathologic"] = y
    feat_df.to_csv(OUT / f"features_v6{tag}.csv", index=False)
    # nested subsets, all with the same full-precision provenance:
    #   v4 = clinical ROI measurements only
    #   v5 = v4 + the striatal grid profile
    #   v6 = v5 + per-hemisphere shape descriptors
    grid = [c for c in feat_df.columns if re.fullmatch(r"g\d\d_(worse|better|rel)", c)
            or c.startswith("grid_")]
    hemi = [c for c in feat_df.columns if c.startswith("h_")]
    # `s_*` are the v13 radial shells: part of the rich table only, so v4/v5 keep
    # exactly the columns they have always had and `lin4` is unchanged.
    shell = [c for c in feat_df.columns if c.startswith("s_r")]
    feat_df.drop(columns=hemi + shell).to_csv(OUT / f"features_v5{tag}.csv", index=False)
    feat_df.drop(columns=grid + hemi + shell).to_csv(OUT / f"features_v4{tag}.csv", index=False)

    qc = pd.DataFrame([{k: r.get(k) for k in
                        ("uid", "ok", "ref", "sep", "head_ml", "inner_ml",
                         "two_blob", "native_spacing")} for r in res])
    qc["y"] = y
    qc.to_csv(OUT / f"qc_v3{tag}.csv", index=False)
    print(f"[cache3] wrote crops_v3.npz, features_v4.csv ({feat_df.shape[1]-2} feats), qc_v3.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
