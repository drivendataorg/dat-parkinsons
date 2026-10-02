"""Predict P(abnormal DaT scan) for any folder of NIfTI files, e.g. a single new scan.

The inference entry point expects the competition test layout (a submission_format.csv
plus niftis/<uid>.nii.gz). This helper builds that layout in a temporary directory
(symlinks, no copies), runs scripts/predict.sh, and writes a CSV with one row per input
file: uid (file name without .nii.gz / .nii), is_pathologic (probability).

On the normal path each scan's prediction uses only that scan's voxels and the frozen
model files. Scans are registered and run through the CNNs in blocks of up to 128, so
the scans that share a block change the floating-point summation order and move the
probabilities slightly. Predicting five scans one at a time changed them by at most
3.1e-4, and shuffling the order of 150 scans changed them by 9e-6 on average and at
most 1.3e-3. The error fallbacks for a failed registration or CNN act on the whole
block. None of them fired in the local runs. See README, "Run inference".

Usage (from the repository root):
    python scripts/predict_folder.py /path/to/folder_with_niftis predictions.csv
    python scripts/predict_folder.py /path/to/one_scan.nii.gz prediction.csv
"""
from __future__ import annotations

import csv
import gzip
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def uid_of(p: Path) -> str:
    name = p.name
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    raise ValueError(f"not a NIfTI file: {p}")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    files = [src] if src.is_file() else sorted(
        p for p in src.iterdir() if p.name.endswith((".nii.gz", ".nii")))
    if not files:
        print("no .nii/.nii.gz files found")
        return 1
    with tempfile.TemporaryDirectory() as td:
        data = Path(td)
        (data / "niftis").mkdir()
        uids = []
        for p in files:
            uid = uid_of(p)
            dst = data / "niftis" / f"{uid}.nii.gz"
            if p.name.endswith(".nii.gz"):
                os.symlink(p, dst)
            else:
                # the pipeline reads niftis/<uid>.nii.gz and nibabel picks the decompressor
                # from the extension, so an uncompressed .nii is gzipped into the temp dir
                with open(p, "rb") as fi, gzip.open(dst, "wb") as fo:
                    shutil.copyfileobj(fi, fo)
            uids.append(uid)
        with open(data / "submission_format.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["uid", "is_pathologic"])
            w.writerows([u, 0.5] for u in uids)
        cmd = ["bash", str(ROOT / "scripts" / "predict.sh"), str(data), str(out)]
        return subprocess.call(cmd, env={**os.environ, "PYTHON": os.environ.get("PYTHON", sys.executable)})


if __name__ == "__main__":
    sys.exit(main())
