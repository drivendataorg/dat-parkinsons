# python train_medicalnet.py --run_name medicalnet10 --model medicalnet --medicalnet_backbone resnet --medicalnet_depth 10 --resnet_shortcut B
# python train_medicalnet.py --run_name test --model medicalnet --medicalnet_backbone resnext --medicalnet_depth 50 --resnet_shortcut B --resnext_cardinality 32

import os
import concurrent.futures
import multiprocessing
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset as TorchDataset
from scipy import ndimage
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from tqdm import tqdm

from monai.data import DataLoader
import monai.transforms as mt
from monai.utils import set_determinism
from src.models.builder import build_model
import random
import time
from functools import wraps
from datetime import datetime

import argparse
from omegaconf import OmegaConf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--model", required=True)
    args, unknown = parser.parse_known_args()
    RUN_NAME = args.run_name
    MODEL = args.model
else:
    RUN_NAME = "medicalnet"
    MODEL = "medicalnet"

# ==========================================
# 1. CONFIGURATION DYNAMIQUE (via YAML)
# ==========================================
cfg = OmegaConf.merge(OmegaConf.load("0_configs/default.yaml"))

DATA_DIR    = cfg.paths.niftis
LABELS_PATH = cfg.paths.train_labels
OUTPUT_DIR  = Path(cfg.paths.phase1_out) / RUN_NAME
# Préfixe des poids / nom OOF depuis le YAML (ex: medicalnextB -> mednext_b, oof_mednextB.csv), sinon RUN_NAME
MODEL_CFG   = next((m for m in cfg.models.values() if m.get("folder") == RUN_NAME), {})
CACHE_DIR   = Path(cfg.paths.root_dir) / "vol_cache"
SEED = 0

TARGET_SHAPE       = (128, 128, 128)
BATCH_SIZE         = 4
ACCUMULATION_STEPS = 4
EPOCHS             = 50
LEARNING_RATE      = 5e-4
ETA_MIN            = 1e-4
WEIGHT_DECAY       = 1e-4
WARMUP_EPOCHS      = 5
N_SPLITS           = 5
NUM_WORKERS        = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# ==========================================
# UTILS
# ==========================================
def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Human-readable timestamps
        start_datetime = datetime.now()
        start_perf = time.perf_counter()
        
        print(f"[{func.__name__}] Starting at {start_datetime.strftime('%A %Y-%m-%d %H:%M:%S')}")
        
        result = func(*args, **kwargs)
        
        end_perf = time.perf_counter()
        end_datetime = datetime.now()
        
        elapsed = end_perf - start_perf
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        print(f"[{func.__name__}] Ending at   {end_datetime.strftime('%A %Y-%m-%d %H:%M:%S')}")
        print(f"[{func.__name__}] Total time:  {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        
        return result
    return wrapper


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_determinism(seed=seed)

# ==========================================
# 3b. VOLUME PRECOMPUTATION (deterministic transforms -> .npy cache)
# ==========================================
def _precompute_volume_worker(args):
    """Top-level worker (picklable) — preprocesses one volume and saves .npy."""
    uid, nifti_path, cache_dir_str = args
    out_path = Path(cache_dir_str) / f"{uid}.npy"
    if out_path.exists():
        return uid, True
    try:
        import nibabel as _nib
        import numpy as _np
        from scipy import ndimage as _ndi

        _TS  = TARGET_SHAPE
        _TSP = (2.0, 2.0, 2.0)

        img  = _nib.load(nifti_path)
        vol  = img.get_fdata(dtype=_np.float32)
        sp   = _np.array(img.header.get_zooms()[:3], dtype=float)
        zoom = sp / _np.array(_TSP, dtype=float)
        res  = _ndi.zoom(vol, zoom, order=1)

        p95    = float(_np.percentile(res, 95))
        bright = res > p95
        try:
            center = _np.array(_ndi.center_of_mass(bright), dtype=int)
        except Exception:
            center = _np.array(res.shape) // 2

        half = _np.array(_TS) // 2
        s0, e0 = center - half, center + half
        bef = _np.maximum(0, -s0)
        aft = _np.maximum(0, e0 - _np.array(res.shape))
        pad = list(zip(bef.tolist(), aft.tolist()))
        padded = _np.pad(res, pad, mode="constant", constant_values=float(res.min()))
        s1 = s0 + bef; e1 = e0 + bef
        crop = padded[s1[0]:e1[0], s1[1]:e1[1], s1[2]:e1[2]]

        cmin, cmax = float(crop.min()), float(crop.max())
        sc = ((crop - cmin) / (cmax - cmin + 1e-8)).astype(_np.float32)

        _np.save(str(out_path), sc)
        return uid, True
    except Exception as e:
        print(f"  WARNING {uid}: {e}")
        return uid, False


def precompute_volumes(df: pd.DataFrame, data_dir: str, cache_dir: str) -> None:
    """
    Run Spacing + CenterCrop + ScaleIntensity once per volume, save as .npy.
    Uses multiprocessing for speed. Resumable: skips already-cached volumes.
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    to_process = [(row.uid, os.path.join(data_dir, f"{row.uid}.nii.gz"), cache_dir)
                  for _, row in df.iterrows()
                  if not (cache_path / f"{row.uid}.npy").exists()]
    if not to_process:
        print(f"Volume cache complete ({len(df)} volumes in {cache_dir})")
        return
    print(f"Precomputing {len(to_process)}/{len(df)} volumes ({NUM_WORKERS} workers)...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        results = list(tqdm(
            pool.map(_precompute_volume_worker, to_process),
            total=len(to_process), desc="Preprocessing volumes"
        ))
    failed = [uid for uid, ok in results if not ok]
    if failed:
        print(f"WARNING: {len(failed)} volumes failed precomputation")


class CachedVolumeDataset(TorchDataset):
    """
    Loads preprocessed volumes from .npy cache and optionally applies
    random augments (train) or nothing (val). Replaces the MONAI Dataset
    + full transform pipeline — skipping the deterministic Spacing/Crop/Scale
    that are now pre-cached makes each batch ~10x faster to load.
    """
    def __init__(self, file_dicts, cache_dir, rand_transforms=None):
        self.dicts      = file_dicts
        self.cache_dir  = Path(cache_dir)
        self.rand_trans = rand_transforms

    def __len__(self):
        return len(self.dicts)

    def __getitem__(self, idx):
        d   = self.dicts[idx]
        uid = Path(d["image"]).name.replace(".nii.gz", "").replace(".nii", "")
        vol = np.load(self.cache_dir / f"{uid}.npy")
        sample = {
            "image":    torch.tensor(vol, dtype=torch.float32).unsqueeze(0),
            "label":    d["label"],
            "uid":      uid,
        }
        if self.rand_trans:
            sample = self.rand_trans(sample)
        return sample


def get_rand_transforms():
    """Random augments only — deterministic preprocessing is pre-cached."""
    return mt.Compose([
        mt.RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
        mt.RandRotated(keys=["image"], range_x=0.1, range_y=0.1, range_z=0.1, prob=0.4),
        mt.RandAffined(keys=["image"], translate_range=(5, 5, 5), prob=0.3),
        mt.RandGaussianSmoothd(keys=["image"], prob=0.2,
                               sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        mt.RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
        mt.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.4),
    ])


# ==========================================
# 5. BOUCLE D'ENTRAÎNEMENT K-FOLD
# ==========================================
@time_it
def main():
    seed_everything(SEED)

    effective_batch = BATCH_SIZE * ACCUMULATION_STEPS
    print(f"\n{'='*60}")
    print(f"🚀 LANCEMENT ENSEMBLE K-FOLD 🚀")
    print(f"Machine : {DEVICE} | Résolution : {TARGET_SHAPE}")
    print(f"Batch physique : {BATCH_SIZE} | Accumulation : {ACCUMULATION_STEPS} | Effectif : {effective_batch}")
    print(f"Run name: {RUN_NAME} | Model : {MODEL}")
    print(f"{'='*60}\n")

    # --- Données ---
    print("📥 Lecture des labels…")
    df_real = pd.read_csv(LABELS_PATH)
    y_real = df_real['is_pathologic'].values

    # Step 2 — volume cache (deterministic transforms, resumable)
    precompute_volumes(df_real, DATA_DIR, CACHE_DIR)

    df = df_real.copy()
    y  = df['is_pathologic'].values

    # Dicts de base 
    base_dicts = [
        {
            "image":    os.path.join(DATA_DIR, f"{row['uid']}.nii.gz"),
            "label":    row['is_pathologic'],
            "uid":      row['uid'],
        }
        for i, (_, row) in enumerate(df.iterrows())
    ]
    X = np.array(base_dicts)

    criterion = nn.BCEWithLogitsLoss()
    rand_trans = get_rand_transforms()

    oof_predictions = np.zeros(len(df_real))
    
    ## Pre-defined folds
    N_SPLITS = df["fold"].nunique()
    uid_to_fold = dict(zip(df["uid"], df["fold"]))

    total_steps = N_SPLITS * EPOCHS
    for fold in sorted(df["fold"].unique()):  
        seed_everything(SEED + int(fold))
        print(f"\n{'*'*20} FOLD {fold + 1}/{N_SPLITS} {'*'*20}")

        train_files = [
            d for d in base_dicts
            if uid_to_fold[d["uid"]] != fold
        ]

        val_files = [
            d for d in base_dicts
            if uid_to_fold[d["uid"]] == fold and 'NMZ_' not in d['uid']
        ]
        val_idx = df_real.loc[df_real.fold == fold].index.tolist()
        print(f"📊 Données : {len(train_files)} Train | {len(val_files)} Validation")

        train_ds = CachedVolumeDataset(train_files, CACHE_DIR, rand_transforms=rand_trans)
        val_ds   = CachedVolumeDataset(val_files,   CACHE_DIR, rand_transforms=None)
        print(f"📏 Input size: {train_ds[0]['image'].shape}")

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        model     = build_model(MODEL, VERBOSE=fold == 0).to(DEVICE)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=LEARNING_RATE, 
            weight_decay=WEIGHT_DECAY
        )

        ### CosineAnnealing with restarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=ETA_MIN)
        
        best_val_loss   = float('inf')
        model_save_path = os.path.join(OUTPUT_DIR, f"{MODEL_CFG.get('prefix', RUN_NAME)}_fold_{fold}.pth")

        for epoch in range(EPOCHS):
            # ── Train ──
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()

            pbar = tqdm(train_loader, desc=f"Fold {fold+1}/{N_SPLITS} | Ep {epoch+1:02d}/{EPOCHS}", leave=False)
            for step, batch in enumerate(pbar):
                inputs = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE).float().unsqueeze(1)

                outputs    = model(inputs)
                loss       = criterion(outputs, labels)
                (loss / ACCUMULATION_STEPS).backward()
                train_loss += loss.item() * inputs.size(0)

                if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
                    optimizer.step()
                    optimizer.zero_grad()

                # Live update during training: step number + current batch loss
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    best_val_loss=f"{best_val_loss:.4f}" if best_val_loss != float('inf') else "N/A"
                )

            train_loss /= len(train_loader.dataset)

            # ── Val ──
            model.eval()
            val_loss   = 0.0
            fold_preds = []
            fold_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["image"].to(DEVICE)
                    labels = batch["label"].to(DEVICE).float().unsqueeze(1)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)

                    probs = torch.sigmoid(outputs)

                    fold_preds.extend(probs.cpu().numpy().flatten())
                    fold_labels.extend(labels.cpu().numpy().flatten())

            val_loss /= len(val_loader.dataset)

            # Compute AUC
            val_auc = roc_auc_score(fold_labels, fold_preds)

            scheduler.step()
            lr = optimizer.param_groups[0]["lr"]

            is_best = False
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"state_dict": model.state_dict()}, model_save_path)
                oof_predictions[val_idx] = np.array(fold_preds)
                is_best = True

        # Compute best predictions for this fold across saved OOF
        fold_oof_preds = oof_predictions[val_idx]
        fold_oof_labels = y_real[val_idx]
        
        fold_best_auc = roc_auc_score(fold_oof_labels, fold_oof_preds)
        fold_best_loss = log_loss(fold_oof_labels, fold_oof_preds)

        # Permanent terminal log for this fold
        tqdm.write(
            f"✅ [FOLD {fold + 1}/{N_SPLITS} COMPLETE] "
            f"Best Val Loss: {fold_best_loss:.4f} | "
            f"Best Val AUC: {fold_best_auc:.4f}\n"
        )
        
    # ==========================================
    # GÉNÉRATION ET SAUVEGARDE DU CSV OOF
    # ==========================================
    print(f"\n{'='*60}")
    print("📈 RÉSULTATS GLOBAUX OUT-OF-FOLD (OOF)")
    print(f"{'='*60}")
    print(f"🏆 OOF Log Loss : {log_loss(y_real, oof_predictions):.4f}")
    print(f"🏆 OOF AUC      : {roc_auc_score(y_real, oof_predictions):.4f}")
    print(f"\n✅ {N_SPLITS} modèles sauvegardés dans {OUTPUT_DIR}/")

    # Création du DataFrame pour le Stacking
    df_oof = pd.DataFrame({
        'uid': df_real['uid'],
        'label': y_real,
        'prob': oof_predictions
    })
    
    # Nom du fichier OOF aligné avec ton YAML (ex: oof_medicalnet18.csv)
    oof_csv_path = OUTPUT_DIR / MODEL_CFG.get("oof_file", f"oof_{RUN_NAME}.csv")
    df_oof.to_csv(oof_csv_path, index=False)
    print(f"✅ Fichier OOF sauvegardé pour le Stacking : {oof_csv_path}")

if __name__ == "__main__":
    main()