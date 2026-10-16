"""
eda_real.py — characterize the REAL DaT-SPECT training data.

Scans image geometry (shape, voxel spacing, orientation), intensity ranges,
and label linkage/integrity across all 1362 scans. Writes a summary parquet/csv
so downstream preprocessing (resampling target, crop size) is data-driven.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parent.parent
# Competition data (niftis/ + labels.csv). Patched for the prize repo; was ROOT/"data-demo".
import os  # noqa: E402  (prize-repo patch)
DATA = Path(os.environ.get("DAT_DATA_DIR", str(ROOT / "data" / "raw")))
NIFTI = DATA / "niftis"
LABELS = DATA / "labels.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def probe(uid: str) -> dict:
    p = NIFTI / f"{uid}.nii.gz"
    try:
        img = nib.load(str(p))
        hdr = img.header
        shape = tuple(int(s) for s in img.shape[:3])
        zooms = tuple(round(float(z), 4) for z in hdr.get_zooms()[:3])
        # Load data lazily but compute robust intensity stats
        data = np.asarray(img.dataobj, dtype=np.float32)
        data = np.squeeze(data)
        finite = np.isfinite(data)
        dv = data[finite]
        pos = dv[dv > 0]
        ax = nib.aff2axcodes(img.affine)
        return {
            "uid": uid,
            "ndim": int(data.ndim),
            "nx": shape[0], "ny": shape[1], "nz": shape[2],
            "sx": zooms[0], "sy": zooms[1], "sz": zooms[2],
            "orient": "".join(ax),
            "dtype": str(img.get_data_dtype()),
            "vmin": float(dv.min()) if dv.size else 0.0,
            "vmax": float(dv.max()) if dv.size else 0.0,
            "vmean": float(dv.mean()) if dv.size else 0.0,
            "p99": float(np.percentile(pos, 99)) if pos.size else 0.0,
            "nvox": int(data.size),
            "frac_pos": float(pos.size / max(data.size, 1)),
            "n_nonfinite": int((~finite).sum()),
            "ok": True,
        }
    except Exception as e:
        return {"uid": uid, "ok": False, "err": repr(e)}


def main():
    labels = pd.read_csv(LABELS)
    labels = labels[labels["uid"] != "is_pathologic"].copy()  # drop any stray header row
    labels["is_pathologic"] = labels["is_pathologic"].astype(float)
    uids = labels["uid"].tolist()
    print(f"[labels] {len(uids)} rows | prevalence={labels.is_pathologic.mean():.4f} "
          f"| pathologic={int(labels.is_pathologic.sum())} normal={int((1-labels.is_pathologic).sum())}")

    # linkage check
    on_disk = {p.name[:-7] for p in NIFTI.glob("*.nii.gz")}
    missing = [u for u in uids if u not in on_disk]
    extra = [u for u in on_disk if u not in set(uids)]
    print(f"[linkage] missing images for labels: {len(missing)} | images without labels: {len(extra)}")

    rows = Parallel(n_jobs=-1, verbose=1)(delayed(probe)(u) for u in uids)
    df = pd.DataFrame(rows)
    bad = df[~df["ok"]] if "ok" in df else df
    print(f"[probe] failed: {len(bad)}")
    ok = df[df["ok"]].copy()
    ok = ok.merge(labels, on="uid", how="left")
    ok.to_csv(OUT / "eda_real_geometry.csv", index=False)

    print("\n=== SHAPE distribution (nx,ny,nz) ===")
    print(ok.groupby(["nx", "ny", "nz"]).size().sort_values(ascending=False).head(15))
    print("\n=== VOXEL SPACING distribution (sx,sy,sz) ===")
    print(ok.groupby(["sx", "sy", "sz"]).size().sort_values(ascending=False).head(15))
    print("\n=== ORIENTATION codes ===")
    print(ok["orient"].value_counts())
    print("\n=== NDIM ===")
    print(ok["ndim"].value_counts())
    print("\n=== DTYPE ===")
    print(ok["dtype"].value_counts())
    print("\n=== physical extent mm (n*spacing) ===")
    ok["ext_x"] = ok.nx * ok.sx; ok["ext_y"] = ok.ny * ok.sy; ok["ext_z"] = ok.nz * ok.sz
    print(ok[["ext_x", "ext_y", "ext_z"]].describe().round(1))
    print("\n=== intensity: vmax, p99, frac_pos by class ===")
    print(ok.groupby("is_pathologic")[["vmax", "p99", "vmean", "frac_pos"]].median().round(2))
    print("\n=== nonfinite voxels: scans affected ===", int((ok.n_nonfinite > 0).sum()))

    summary = {
        "n": int(len(ok)),
        "prevalence": float(labels.is_pathologic.mean()),
        "shapes_unique": int(ok.groupby(["nx", "ny", "nz"]).ngroups),
        "spacings_unique": int(ok.groupby(["sx", "sy", "sz"]).ngroups),
        "orient_codes": ok["orient"].value_counts().to_dict(),
        "median_spacing": [float(ok.sx.median()), float(ok.sy.median()), float(ok.sz.median())],
        "median_extent_mm": [float(ok.ext_x.median()), float(ok.ext_y.median()), float(ok.ext_z.median())],
        "missing_images": len(missing), "extra_images": len(extra), "probe_failed": int(len(bad)),
    }
    (OUT / "eda_real_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n[saved] outputs/eda_real_geometry.csv + eda_real_summary.json")


if __name__ == "__main__":
    main()
