"""Canonical cross-validation fold assignment, shared by ALL models so that
out-of-fold predictions can be ensembled without leakage.

One patient per scan (per the competition), so StratifiedKFold on the label is
valid — no group leakage. Written once to outputs/folds.csv and reused."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parent.parent
# Competition data (niftis/ + labels.csv). Patched for the prize repo; was ROOT/"data-demo".
import os  # noqa: E402  (prize-repo patch)
DATA = Path(os.environ.get("DAT_DATA_DIR", str(ROOT / "data" / "raw")))
OUT = ROOT / "outputs"
N_SPLITS = 5
SEED = 42


def get_folds(uids, y, n_splits=N_SPLITS, seed=SEED) -> pd.DataFrame:
    path = OUT / "folds.csv"
    uids = list(uids)
    if path.exists():
        df = pd.read_csv(path)
        if set(df["uid"]) == set(uids) and df["fold"].nunique() == n_splits:
            return df
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.zeros(len(uids), dtype=int)
    y = np.asarray(y)
    for k, (_, va) in enumerate(skf.split(np.zeros(len(uids)), y)):
        fold[va] = k
    df = pd.DataFrame({"uid": uids, "fold": fold, "y": np.asarray(y).astype(int)})
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    labels = pd.read_csv(DATA / "labels.csv")
    df = get_folds(labels["uid"], labels["is_pathologic"].astype(int))
    print("folds written:", (OUT / "folds.csv"))
    print(df.groupby("fold")["y"].agg(["size", "mean"]).round(3))
