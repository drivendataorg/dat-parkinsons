"""Evaluate Siamese .pth files on their own (exp, fold) validation split.

Uses train_siamese.py's own split, dataset and model code, so the val set is exactly
the one each worker early-stopped on. Prints the same metric the worker logs
(log loss of prob_tta, which equals prob_normal because USE_TTA=False).

Run from a dir whose 0_configs/default.yaml points at the real patch datasets
(train_siamese creates its output dirs at import time, relative to that config):
  cd ~/sfmn1/runs/siamese_nosynth && ~/sfmn1/code/.venv/bin/python <this> E:F:path.pth [E:F:path.pth ...]
"""
import importlib.util
import sys

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch.utils.data import DataLoader

specs = sys.argv[1:]
sys.argv = [sys.argv[0], "--no-synth"]  # val split does not depend on synth; avoids needing synth patches
spec = importlib.util.spec_from_file_location("ts", "/home/ec2-user/sfmn1/code/4_siamese_training/train_siamese.py")
ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)

for s in specs:
    e, f, path = s.split(":", 2)
    _, val_df = ts.get_deterministic_split(int(e), int(f))
    dl = DataLoader(ts.DaTMashupDataset(val_df, is_train=False), batch_size=64, shuffle=False)
    m = ts.MashupInternalEnsembleCNN(dropout_rate=ts.DROPOUT).cuda().eval()
    m.load_state_dict(torch.load(path, map_location="cuda"))
    ps, ys = [], []
    with torch.no_grad():
        for b in dl:
            t = [x.cuda() for x in b[:6]]
            ps.append(torch.sigmoid(m(*t)[0]).cpu().numpy().ravel())
            ys.append(b[8].numpy().ravel())
    p, y = np.concatenate(ps), np.concatenate(ys)
    print(f"exp {e} fold {f}  n={len(y)}  val log loss={log_loss(y, p):.4f}  AUC={roc_auc_score(y, p):.4f}  {path}")
