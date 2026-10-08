"""Compare a retrained (member, fold) against the submitted TorchScript trace of the same (member, fold).

Both models held out the same rows (the member's cv_round split, fold k), so on those rows we compare:
  * the retrained run's dumped best-epoch OOF probabilities (flip-TTA, as written by datscan/train.py)
  * the submitted trace re-scored here with the same flip-TTA forward main.py uses.

  usage (from the repo root, with DAT_WORK set):
    python ../verification_workspace_3rd_place/retrain_notes/scripts/compare_fold.py --runs $DAT_WORK/runs_fusion10 --member fu_a1_dnet --fold 0 \
        --assets ~/sfmn3/submitted/assets
"""
import argparse, os, sys
import numpy as np, torch
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "inference"))
from datscan import paths as P   # noqa: E402
import canonize as CZ            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--member", default="fu_a1_dnet")
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--assets", required=True)
ap.add_argument("--bs", type=int, default=16)
a = ap.parse_args()

torch.set_grad_enabled(False)
torch._C._jit_set_profiling_executor(False); torch._C._jit_set_profiling_mode(False)
torch._C._jit_override_can_fuse_on_cpu(False); torch._C._jit_override_can_fuse_on_gpu(False)
torch._C._jit_set_texpr_fuser_enabled(False)

y = np.load(P.META / "labels.npy")
idx = np.load(f"{a.runs}/{a.member}__best_oof_idx_fold{a.fold}.npy")
p_new = np.clip(np.load(f"{a.runs}/{a.member}__best_oof_p_fold{a.fold}.npy"), 1e-6, 1 - 1e-6)

comp = np.load(str(P.BOX_CANONV2), mmap_mode="r"); cm = np.load(str(P.MASK_CANONV2), mmap_mode="r")
ts = torch.jit.load(f"{a.assets}/{a.member}_f{a.fold}.ts.pt", map_location="cuda").eval()
dt = next(ts.parameters()).dtype
z_old = []
for s in range(0, len(idx), a.bs):
    ii = idx[s:s + a.bs]
    x = torch.from_numpy(np.asarray(comp[ii], np.float32))[:, None].cuda()
    an = CZ.with_slabs(torch.from_numpy(np.asarray(cm[ii], np.float32))[:, None].cuda())
    z1 = ts(x.to(dt), an.to(dt)).float()
    z2 = ts(torch.flip(x, dims=[2]).to(dt), torch.flip(an, dims=[2]).to(dt)).float()
    z_old.append(((z1 + z2) / 2).cpu().numpy().ravel())
z_old = np.concatenate(z_old)
p_old = np.clip(1 / (1 + np.exp(-z_old)), 1e-6, 1 - 1e-6)
z_new = np.log(p_new / (1 - p_new))
yv = y[idx]

print(f"{a.member} fold {a.fold}: {len(idx)} held-out training scans")
print(f"  submitted trace : val ll {log_loss(yv, p_old):.4f}  auc {roc_auc_score(yv, p_old):.4f}")
print(f"  retrained here  : val ll {log_loss(yv, p_new):.4f}  auc {roc_auc_score(yv, p_new):.4f}")
print(f"  agreement       : logit corr {np.corrcoef(z_old, z_new)[0, 1]:.4f} | prob corr {np.corrcoef(p_old, p_new)[0, 1]:.4f}"
      f" | mean |dp| {np.abs(p_old - p_new).mean():.4f} | max |dp| {np.abs(p_old - p_new).max():.4f}"
      f" | same 0.5-side {np.mean((p_old > .5) == (p_new > .5)):.3f}")
