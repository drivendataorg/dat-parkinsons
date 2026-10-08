"""Rebuild every member's out-of-fold predictions from the SUBMITTED fold traces, so build/calibrate.py and
build/build_roster.py can be exercised without a full 50-fold retrain.

For each recipe member (cv_round split from the recipe) and fold k, the submitted trace <member>_f<k>.ts.pt is
scored on its own held-out rows (split fold == k, in meta/uids.csv order -- the same rows and order
datscan/train.py dumps; checked against the real fu_a1_dnet fold-0 dump) with main.py's flip-TTA forward,
and written as <member>__best_oof_{idx,p}_fold<k>.npy (int64 / float32, as train.py writes them).

  usage (from the repo root, with DAT_WORK set):
    python ../verification_workspace_3rd_place/retrain_notes/scripts/oof_from_traces.py --recipe recipes/fusion10_s078.json \
        --assets ~/sfmn3/submitted/assets --out ~/sfmn3/work/oof_from_traces
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
from sklearn.metrics import log_loss

sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "inference"))
from datscan import paths as P   # noqa: E402
import canonize as CZ            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--recipe", required=True)
ap.add_argument("--assets", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=16)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

torch.set_grad_enabled(False)
torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
torch._C._jit_set_profiling_executor(False); torch._C._jit_set_profiling_mode(False)
torch._C._jit_override_can_fuse_on_cpu(False); torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)

uids = pd.read_csv(P.META / "uids.csv").uid.values
y = np.load(P.META / "labels.npy")
comp = np.load(str(P.BOX_CANONV2), mmap_mode="r"); cm = np.load(str(P.MASK_CANONV2), mmap_mode="r")

for mem in json.load(open(a.recipe))["members"]:
    m = mem["member"]
    fold_of = dict(pd.read_csv(mem["splits"])[["uid", "fold"]].values)
    folds = np.array([fold_of[u] for u in uids])
    for k in range(5):
        idx = np.where(folds == k)[0].astype(np.int64)
        ts = torch.jit.load(f"{a.assets}/{m}_f{k}.ts.pt", map_location="cuda").eval()
        dt = next(ts.parameters()).dtype
        zs = []
        for s in range(0, len(idx), a.bs):
            ii = idx[s:s + a.bs]
            x = torch.from_numpy(np.asarray(comp[ii], np.float32))[:, None].cuda()
            an = CZ.with_slabs(torch.from_numpy(np.asarray(cm[ii], np.float32))[:, None].cuda())
            z1 = ts(x.to(dt), an.to(dt)).float()
            z2 = ts(torch.flip(x, dims=[2]).to(dt), torch.flip(an, dims=[2]).to(dt)).float()
            zs.append(((z1 + z2) / 2).cpu().numpy().ravel())
        p = (1 / (1 + np.exp(-np.concatenate(zs).astype(np.float64)))).astype(np.float32)
        np.save(f"{a.out}/{m}__best_oof_idx_fold{k}.npy", idx)
        np.save(f"{a.out}/{m}__best_oof_p_fold{k}.npy", p)
        print(f"OOF {m} f{k} ({mem['splits']}): {len(idx)} rows | val ll "
              f"{log_loss(y[idx], np.clip(p, 1e-6, 1 - 1e-6)):.4f}", flush=True)
        del ts; torch.cuda.empty_cache()
