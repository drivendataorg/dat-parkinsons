import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from pathlib import Path
from scipy import ndimage
from sklearn.metrics import log_loss, roc_auc_score
from monai.data import DataLoader, CacheDataset, Dataset as MonaiDataset
import monai.transforms as mt
import random
from monai.utils import set_determinism
from monai.transforms import MapTransform
from monai.networks.nets import DenseNet121

from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================

cfg = OmegaConf.merge(
    OmegaConf.load("0_configs/default.yaml"), 
    OmegaConf.from_cli()
)

DATA_DIR    = cfg.paths.niftis
LABELS_PATH = cfg.paths.train_labels
OUTPUT_DIR = Path(cfg.paths.phase1_out) / cfg.models['DenseNet3D121']['folder']

TARGET_SHAPE       = (128, 128, 128)
BATCH_SIZE         = 16
ACCUMULATION_STEPS = 1
EPOCHS             = 50  # 10 (Cycle 1) + 20 (Cycle 2) = 30
LEARNING_RATE      = 5e-4 # Aligné avec tes autres modèles
ETA_MIN = 1e-4
WEIGHT_DECAY       = 1e-4
NUM_WORKERS        = 16

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. TRANSFORMS MONAI
# ==========================================
class StriatumCenterCropd(MapTransform):
    """Centre le crop sur le centre de masse du striatum (P95)."""
    def __init__(self, keys, spatial_size):
        super().__init__(keys)
        self.spatial_size = np.array(spatial_size)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            vol = d[key][0].numpy()
            p95 = np.percentile(vol, 95)
            mask = vol > p95
            try:
                cz, cy, cx = ndimage.center_of_mass(mask)
                center = np.array([int(cz), int(cy), int(cx)])
            except ValueError:
                center = np.array(vol.shape) // 2

            half  = self.spatial_size // 2
            start = center - half
            end   = center + half
            pb    = np.maximum(0, -start)
            pa    = np.maximum(0, end - np.array(vol.shape))
            
            if np.any(pb > 0) or np.any(pa > 0):
                vol   = np.pad(vol, list(zip(pb, pa)), mode='constant', constant_values=vol.min())
                start += pb
                end += pb
                
            cropped = vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            d[key] = torch.tensor(cropped, dtype=torch.float32).unsqueeze(0)
        return d

def get_transforms():
    # 1. Partie Déterministe (Ce qui va dans la RAM)
    det_transforms = mt.Compose([
        mt.LoadImaged(keys=["image"]),
        mt.EnsureChannelFirstd(keys=["image"]),
        mt.Spacingd(keys=["image"], pixdim=(2.0, 2.0, 2.0), mode="bilinear"),
        StriatumCenterCropd(keys=["image"], spatial_size=TARGET_SHAPE),
        mt.ScaleIntensityd(keys=["image"]),
    ])

    # 2. Partie Aléatoire (Calculée à la volée sur le CPU)
    rand_transforms = mt.Compose([
        mt.RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
        mt.RandRotated(keys=["image"], range_x=0.1, range_y=0.1, range_z=0.1, prob=0.4),
        mt.RandAffined(keys=["image"], translate_range=(5, 5, 5), prob=0.3),
        mt.RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        mt.RandGaussianNoised(keys=["image"], prob=0.2, std=0.05),
        mt.RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.4),
    ])

    val_transforms = det_transforms # La validation n'a pas de random
    
    return det_transforms, rand_transforms, val_transforms


def seed_everything(seed=42):
    """Fixe la seed pour Python, Numpy, PyTorch et MONAI."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Si multi-GPU
    
    # CuDNN (pour la reproductibilité des convolutions)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # MONAI (Très important pour les RandTransforms)
    set_determinism(seed=seed)
    
    print(f"🌱 Seed globale fixée à {seed}")


# ==========================================
# 3. BOUCLE D'ENTRAÎNEMENT (FUSION OOF)
# ==========================================
def main():

    seed_everything(seed=42)

    effective_batch = BATCH_SIZE * ACCUMULATION_STEPS
    print(f"\n{'='*60}")
    print(f"🚀 LANCEMENT ENSEMBLE DENSENET 3D (MONAI) 🚀")
    print(f"Machine : {DEVICE} | Résolution : {TARGET_SHAPE}")
    print(f"Batch physique : {BATCH_SIZE} | Accumulation : {ACCUMULATION_STEPS} | Effectif : {effective_batch}")
    print(f"{'='*60}\n")

    print("📥 Lecture des labels et des folds…")
    df = pd.read_csv(LABELS_PATH)
    
    if 'fold' not in df.columns:
        raise ValueError(f"⚠️ La colonne 'fold' est introuvable dans {LABELS_PATH}")

    y = df['is_pathologic'].values
    df_folds = df['fold']
    uids = df['uid'].values

    # Création des dictionnaires pour MONAI
    base_dicts = [
        {"image": os.path.join(DATA_DIR, f"{uid}.nii.gz"), "label": label}
        for uid, label in zip(uids, y)
    ]
    X = np.array(base_dicts)

    det_transforms, rand_transforms, val_transforms = get_transforms()

    df_oof_global = pd.DataFrame(columns=['uid', 'label', 'prob_3d'])

    # Boucle sur les folds officiels
    for fold in sorted(df_folds.unique()):
        print(f"\n{'*'*20} FOLD {fold} {'*'*20}")

        val_idx = (df_folds == fold).values
        train_idx = ~val_idx

        train_files = X[train_idx].tolist()
        val_files   = X[val_idx].tolist()
        val_uids    = uids[val_idx]
        val_labels  = y[val_idx]
        
        print(f"📊 Données : {len(train_files)} Train | {len(val_files)} Validation")

        print("\n🔥 Mise en cache dans la RAM...")
        train_cache = CacheDataset(data=train_files, transform=det_transforms, cache_num=len(train_files), num_workers=NUM_WORKERS)
        val_ds      = CacheDataset(data=val_files,   transform=val_transforms, cache_num=len(val_files), num_workers=NUM_WORKERS)

        train_ds = MonaiDataset(data=train_cache, transform=rand_transforms)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
        val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        # Modèle 3D Natif
        model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1, dropout_prob=0.2).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss()
        
        # NOUVEAU : Stratégie CosineAnnealingWarmRestarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=ETA_MIN)

        best_val_loss   = float('inf')
        model_save_path = os.path.join(OUTPUT_DIR, f"densenet3d_fold_{fold}.pth")
        best_fold_preds = []

        for epoch in range(EPOCHS):
            # ── Train ──
            model.train()
            train_loss = 0.0
            optimizer.zero_grad()

            pbar_train = tqdm(train_loader, desc=f"Ep {epoch+1:02d}/{EPOCHS} [Train]", leave=False)
            for step, batch in enumerate(pbar_train):
                inputs = batch["image"].to(DEVICE)
                labels = batch["label"].to(DEVICE).float().unsqueeze(1)

                outputs = model(inputs)
                loss    = criterion(outputs, labels)
                
                (loss / ACCUMULATION_STEPS).backward()
                train_loss += loss.item() * inputs.size(0)

                if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

                pbar_train.set_postfix(loss=f"{loss.item():.4f}")

            train_loss /= len(train_loader.dataset)

            # ── Val ──
            model.eval()
            val_loss   = 0.0
            fold_preds = []

            pbar_val = tqdm(val_loader, desc=f"Ep {epoch+1:02d}/{EPOCHS} [Val]  ", leave=False)
            with torch.no_grad():
                for batch in pbar_val:
                    inputs = batch["image"].to(DEVICE)
                    labels = batch["label"].to(DEVICE).float().unsqueeze(1)

                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    
                    probs = torch.sigmoid(outputs).detach().cpu().numpy().flatten()
                    fold_preds.extend(probs.tolist())
                    
                    pbar_val.set_postfix(loss=f"{loss.item():.4f}")

            val_loss /= len(val_loader.dataset)
            
            # Mise à jour du scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Ep {epoch+1:02d}/{EPOCHS} | LR: {current_lr:.1e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", end="")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_fold_preds = fold_preds.copy()
                torch.save({"state_dict": model.state_dict()}, model_save_path)
                print(" 🌟 Nouveau best!")
            else:
                print("")

        # ── Fin du Fold -> Agrégation dans le DataFrame global ──
        fold_df = pd.DataFrame({
            'uid': val_uids,
            'label': val_labels,
            'prob_3d': best_fold_preds
        })
        df_oof_global = pd.concat([df_oof_global, fold_df], ignore_index=True)

    # ==========================================
    # RÉCAPITULATIF FINAL ET EXPORT OOF
    # ==========================================
    print(f"\n{'='*60}")
    print("📈 RÉSULTATS GLOBAUX OUT-OF-FOLD (DENSENET 3D)")
    print(f"{'='*60}")
    
    final_auc = roc_auc_score(df_oof_global['label'].astype(int), df_oof_global['prob_3d'])
    final_ll  = log_loss(df_oof_global['label'].astype(int), df_oof_global['prob_3d'])
    
    print(f"🏆 OOF Score AUC      : {final_auc:.4f}")
    print(f"🏆 OOF Score Log Loss : {final_ll:.4f}")
    
    oof_path = os.path.join(OUTPUT_DIR, 'oof_densenet3d.csv')
    df_oof_global.to_csv(oof_path, index=False)
    
    print(f"\n✅ {len(df_folds.unique())} modèles sauvegardés dans {OUTPUT_DIR}")
    print(f"✅ Fichier OOF prêt pour le Stacking : {oof_path}")


if __name__ == "__main__":
    main()