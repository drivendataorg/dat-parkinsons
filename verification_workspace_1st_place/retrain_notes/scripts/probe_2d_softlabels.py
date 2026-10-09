"""Probe why ResNet2D/ViT2D synthetic soft labels differ from the HF csv.

Usage: python probe_2d_softlabels.py <src_root> <label>
  src_root: dir containing `src/` (released code: ~/sfmn1/code; submission-era: ~/sfmn1/submitted)
Prints, for 20 synthetic scans, |our prob - HF csv prob| for ResNet2D and ViT2D (HF weights), and
for 20 fold-0 training scans, ResNet2D fold-0-only prob vs HF oof_resnet2d.csv.
"""
import os, sys
import numpy as np, pandas as pd
root, label = sys.argv[1], sys.argv[2]
sys.path.insert(0, root); sys.path.insert(0, os.path.join(root, "src", "libs"))
from src.preprocessing import prepare_all_inputs
from src import inference_2d_avg as m2d

W = "/home/ec2-user/sfmn1/code/1_initial_training/results"
hf = pd.read_csv("/home/ec2-user/sfmn1/code/3_soft_labeling/results/synthetic_soft_labels.csv").set_index("uid")
uids = hf.index[:20]
for name, cls, d, col in [("ResNet2D", m2d.ResNet2DEnsemble, "resnet2d", "prob_ResNet2D"), ("Vit2D", m2d.ViT2DEnsemble, "vit2d", "prob_Vit2D")]:
    try:
        ens = cls(os.path.join(W, d))
    except Exception as e:
        print(f"[{label}] {name}: LOAD FAILED: {type(e).__name__}: {str(e)[:200]}"); continue
    ps = [ens.predict(prepare_all_inputs(f"/home/ec2-user/sfmn1/data/external/PET_GENERATED/{u}.nii.gz")["slices_2d"]) for u in uids]
    d_ = np.abs(np.array(ps) - hf.loc[uids, col].values)
    print(f"[{label}] {name} synthetic vs HF csv: mean|d| {d_.mean():.4f} max|d| {d_.max():.4f}")
    if name == "ResNet2D":
        ens.models = ens.models[:1]  # fold 0 only
        f = pd.read_csv("/home/ec2-user/sfmn1/data/train_labels_with_folds.csv")
        oof = pd.read_csv(os.path.join(W, "resnet2d", "oof_resnet2d.csv")).set_index("uid")
        tu = [u for u in f[f.fold == 0].uid if u in oof.index][:20]
        ps = [ens.predict(prepare_all_inputs(f"/home/ec2-user/sfmn1/data/niftis/{u}.nii.gz")["slices_2d"]) for u in tu]
        d_ = np.abs(np.array(ps) - oof.loc[tu, "prob_2d"].values)
        print(f"[{label}] ResNet2D fold0 on fold-0 train scans vs HF oof: mean|d| {d_.mean():.4f} max|d| {d_.max():.4f}")
