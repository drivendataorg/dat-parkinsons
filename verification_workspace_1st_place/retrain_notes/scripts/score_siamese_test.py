"""Fast test-set scoring for the 1st-place Siamese ensemble.

Reproduces the submitted inference path (main.py, FAMILY_STACKING with the
single family "Mashup_Internal") but only builds the Siamese inputs, which skips
the slow radiomics, 2D and 3D preprocessing that main.py computes and then
discards.

  cache   : preprocess each test scan once -> <cache_dir>/<uid>.pt
  probs   : per-model sigmoid probs for every .pth in a weights dir -> csv (uid x model)
  score   : combine per-model prob csvs, optionally swapping some models, then
            p = clip(sigmoid(logit(mean_k p_k) / 1.094), 0.01, 0.99) and log loss

Run from the unpacked submission dir so `src` and `src/libs` resolve:
  cd ~/sfmn1/submitted && ../code/.venv/bin/python <this> cache  --nifti-dir data/niftis --cache-dir ~/sfmn1/test_siamese_cache
  cd ~/sfmn1/submitted && ../code/.venv/bin/python <this> probs  --cache-dir ... --weights-dir weights --out probs_submitted.csv
  ../code/.venv/bin/python <this> score --base probs_submitted.csv [--swap probs_retrained.csv] --labels data/test_labels.csv
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

TEMP = 1.094
CLIP = (0.01, 0.99)


def _cache_one(task):
    nifti_path, out_path = task
    if os.path.exists(out_path):
        return out_path
    import torch
    from src.preprocessing import preprocess_patient_numpy, extract_siamese_patches
    vol = preprocess_patient_numpy(nifti_path)
    sd = extract_siamese_patches(vol, crop_size=32, global_crop_size=48, stride=2)
    torch.save({"patch_L": sd["patch_L"], "patch_R": sd["patch_R"],
                "patch_Global": sd["striatums"]["patch_global"]}, out_path)
    return out_path


def cmd_cache(a):
    os.makedirs(a.cache_dir, exist_ok=True)
    uids = pd.read_csv(Path(a.nifti_dir).parent / "submission_format.csv")["uid"]
    tasks = [(os.path.join(a.nifti_dir, f"{u}.nii.gz"), os.path.join(a.cache_dir, f"{u}.pt")) for u in uids]
    with ProcessPoolExecutor(a.workers) as ex:
        for i, _ in enumerate(ex.map(_cache_one, tasks), 1):
            if i % 50 == 0:
                print(f"cached {i}/{len(tasks)}", flush=True)


def cmd_probs(a):
    import torch
    from src.inference_2d_avg import SiameseMashupInternalEnsemble
    ens = SiameseMashupInternalEnsemble(a.weights_dir, n_exp=0, n_folds=0)  # load nothing; reuse its preprocessing
    names, models = [], []
    from src.builder import build_model
    for p in sorted(Path(a.weights_dir).glob(f"{a.prefix}_exp_*_fold_*.pth")):
        m = build_model("siamese_internal", use_dwt=False)
        sd = torch.load(p, map_location=ens.device)
        m.load_state_dict(sd.get("state_dict", sd))
        models.append(m.to(ens.device).eval())
        names.append(p.stem.replace(f"{a.prefix}_", ""))
    print(f"{len(models)} models from {a.weights_dir}")
    rows = []
    uids = sorted(f[:-3] for f in os.listdir(a.cache_dir) if f.endswith(".pt"))
    with torch.no_grad():
        for u in uids:
            t = ens._prepare_tensors(torch.load(os.path.join(a.cache_dir, f"{u}.pt")))
            rows.append([u] + [torch.sigmoid(m(*t)[0]).item() for m in models])
    pd.DataFrame(rows, columns=["uid"] + names).to_csv(a.out, index=False)
    print(f"wrote {a.out}")


def final_prob(p_models):
    from scipy.special import logit, expit
    p = np.clip(p_models.mean(axis=1), 1e-7, 1 - 1e-7)
    return np.clip(expit(logit(p) / TEMP), *CLIP)


def cmd_score(a):
    from sklearn.metrics import log_loss, roc_auc_score
    base = pd.read_csv(a.base).set_index("uid")
    for s in a.swap or []:
        sw = pd.read_csv(s).set_index("uid")
        cols = [c for c in sw.columns if c in base.columns]
        base[cols] = sw.loc[base.index, cols]
        print(f"swapped {len(cols)} models from {s}: {cols}")
    if a.only:
        base = base[[c for c in base.columns if c in a.only]]
    y = pd.read_csv(a.labels).set_index("uid").loc[base.index, "is_pathologic"].values
    p = final_prob(base.values)
    print(f"n_models={base.shape[1]}  test log loss={log_loss(y, p):.6f}  AUC={roc_auc_score(y, p):.4f}")
    if a.out:
        pd.DataFrame({"uid": base.index, "is_pathologic": p}).to_csv(a.out, index=False)


if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    sys.path.insert(0, os.path.join(os.getcwd(), "src", "libs"))
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("cache"); c.add_argument("--nifti-dir", required=True); c.add_argument("--cache-dir", required=True); c.add_argument("--workers", type=int, default=4)
    c = sp.add_parser("probs"); c.add_argument("--cache-dir", required=True); c.add_argument("--weights-dir", required=True); c.add_argument("--out", required=True); c.add_argument("--prefix", default="mashup_ensemble")
    c = sp.add_parser("score"); c.add_argument("--base", required=True); c.add_argument("--swap", nargs="*"); c.add_argument("--only", nargs="*"); c.add_argument("--labels", required=True); c.add_argument("--out")
    a = ap.parse_args()
    {"cache": cmd_cache, "probs": cmd_probs, "score": cmd_score}[a.cmd](a)
