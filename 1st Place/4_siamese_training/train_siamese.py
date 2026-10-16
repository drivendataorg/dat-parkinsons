import os
import sys
import random
import argparse
import json
import subprocess
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, log_loss
import ptwt

# ==========================================
# 0. PARSING GLOBAL DES ARGUMENTS
# ==========================================
# On centralise tous les arguments ici pour éviter les conflits
parser = argparse.ArgumentParser()
parser.add_argument("--no-synth", action="store_true", help="Désactive l'usage des données synthétiques")
parser.add_argument('--worker', action='store_true', help="Flag interne pour lancer un sous-processus")
parser.add_argument('--exp', type=int, default=0)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--device', type=str, default="cpu")
args, unknown = parser.parse_known_args()

USE_SYNTHETIC = not args.no_synth
print(f"[{'MODE CLASSIQUE' if USE_SYNTHETIC else 'MODE SANS SYNTHÉTIQUE'}] - USE_SYNTHETIC = {USE_SYNTHETIC}")

# ==========================================
# 1. SEED ABSOLU
# ==========================================
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

cfg = OmegaConf.merge(OmegaConf.load("0_configs/default.yaml"), OmegaConf.from_cli())

# ==========================================
# 2. CONFIGURATION GLOBALE (DYNAMIQUE)
# ==========================================
REAL_CSV_PATH = os.path.join(cfg.paths.dataset_siamese, 'folds_siamese.csv')
REAL_SIAM_DIR = os.path.join(cfg.paths.dataset_siamese, 'patches_pt')
REAL_GLOB_DIR = os.path.join(cfg.paths.dataset_striatums, 'patches_pt')

SYNTH_CSV_PATH = os.path.join(cfg.paths.dataset_synthetic_siamese, 'folds_siamese.csv')
SYNTH_SIAM_DIR = os.path.join(cfg.paths.dataset_synthetic_siamese, 'patches_pt')
SYNTH_GLOB_DIR = os.path.join(cfg.paths.dataset_synthetic_striatums, 'patches_pt')

SYNTH_LABELS = None # Used to override labels

RUN_NAME    = 'siamese_network'
OUTPUT_DIR  = os.path.join(cfg.paths.phase4_out, RUN_NAME)
PLOT_DIR    = os.path.join(OUTPUT_DIR, 'loss_curves')
LOGS_DIR    = os.path.join(OUTPUT_DIR, 'logs_workers')
PREDS_DIR   = os.path.join(OUTPUT_DIR, 'tmp_preds')

for d in [OUTPUT_DIR, PLOT_DIR, LOGS_DIR, PREDS_DIR]:
    os.makedirs(d, exist_ok=True)

N_EXP       = 20   
K_SPLITS    = 5    
BATCH_SIZE  = 32 
EPOCHS      = 150 

MAX_CONCURRENT_FOLDS = 16  
DATALOADER_WORKERS   = 2  

LR_BACKBONE    = 2.45e-4
LR_HEADS       = 4.24e-4
WEIGHT_DECAY   = 1.79e-6
ETA_MIN        = 4.30e-6

DROPOUT        = 0.3
SOFT_LABEL_COL = 'label' 
PATIENCE       = 60 
AUX_WEIGHT     = 0.3  

USE_DWT        = False  
USE_SYNCHRO_FLIP = False  
USE_TTA          = False  

USE_RELABEL      = False  
RELABEL_CSV_PATH = None

# ==========================================
# 3. TRANSFORMATIONS SPÉCIFIQUES & DWT
# ==========================================
class SpatialMedialFade(object):
    def __init__(self, size=32, fade_start=22):
        self.mask = torch.ones((3, size, size), dtype=torch.float32)
        for i in range(fade_start, size):
            self.mask[:, :, i] = 1.0 - (i - fade_start) / (size - 1 - fade_start)
    def __call__(self, tensor): return tensor * self.mask

base_aug = transforms.Compose([
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
])

tight_crop = transforms.CenterCrop(24)
fade_transform = SpatialMedialFade(size=32, fade_start=22)

def compute_dwt_LL_tensor(x, wave='db1'):
    LL, _ = ptwt.wavedec2(x, wavelet=wave, level=1)
    return LL

# ==========================================
# 4. DATASET
# ==========================================
class DaTMashupDataset(Dataset):
    def __init__(self, df, is_train=False):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid, label = row['uid'], row['label']
        
        siam_pt = torch.load(row['path_siamese'])
        img_L, img_R = siam_pt[0:1], siam_pt[12:13] 
        
        glob_pt = torch.load(row['path_global'])
        if glob_pt.ndim == 4: glob_pt = glob_pt.squeeze(0)
        img_G = glob_pt[0:1] 
        if img_G.shape[-1] != 48: img_G = transforms.functional.resize(img_G, [48, 48], antialias=True)
            
        if self.is_train:
            img_L, img_R, img_G = base_aug(img_L), base_aug(img_R), base_aug(img_G)
            img_L += torch.randn_like(img_L) * 0.02
            img_R += torch.randn_like(img_R) * 0.02
            img_G += torch.randn_like(img_G) * 0.02

            if USE_SYNCHRO_FLIP and random.random() < 0.5:
                img_G = torch.flip(img_G, dims=[1])
                img_L, img_R = img_R, img_L        

        img_L, img_R, img_G = img_L.expand(3, -1, -1), img_R.expand(3, -1, -1), img_G.expand(3, -1, -1)
        
        img_G_flip = torch.flip(img_G, dims=[1])
        img_L_fade, img_R_fade = fade_transform(img_L), fade_transform(img_R)
        img_L_tight, img_R_tight = tight_crop(img_L), tight_crop(img_R)
        
        mean, std = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        def norm(t): return (t - mean) / std
        
        return (norm(img_G), norm(img_G_flip), norm(img_L_fade), norm(img_R_fade), 
                norm(img_L_tight), norm(img_R_tight), norm(img_L), norm(img_R), 
                torch.tensor([label], dtype=torch.float32), uid)

# ==========================================
# 5. ARCHITECTURE : INTERNAL ENSEMBLE
# ==========================================
def create_backbone():
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2), 
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2), 
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2), 
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten()
    )

class MashupInternalEnsembleCNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.enc_global, self.enc_fade, self.enc_tight = create_backbone(), create_backbone(), create_backbone()
        self.proj_global, self.proj_fade, self.proj_tight = nn.Linear(1024, 256), nn.Linear(1024, 256), nn.Linear(1024, 256)
        self.head_global = nn.Sequential(nn.LayerNorm(256), nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(256, 1))
        self.head_fade   = nn.Sequential(nn.LayerNorm(256), nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(256, 1))
        self.head_tight  = nn.Sequential(nn.LayerNorm(256), nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(256, 1))

        self.use_dwt = USE_DWT
        if self.use_dwt:
            self.enc_dwt = create_backbone()
            self.proj_dwt = nn.Linear(1024, 256)
            self.head_dwt = nn.Sequential(nn.LayerNorm(256), nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(256, 1))

    def _siamese_fusion(self, feat_A, feat_B, projector):
        return projector(torch.cat([feat_A, feat_B, torch.abs(feat_A - feat_B), feat_A * feat_B], dim=1))

    def forward(self, img_G, img_G_flip, img_L_fade, img_R_fade, img_L_tight, img_R_tight, img_L=None, img_R=None):
        v_glob  = self._siamese_fusion(self.enc_global(img_G), self.enc_global(img_G_flip), self.proj_global)
        v_fade  = self._siamese_fusion(self.enc_fade(img_L_fade), self.enc_fade(img_R_fade), self.proj_fade)
        v_tight = self._siamese_fusion(self.enc_tight(img_L_tight), self.enc_tight(img_R_tight), self.proj_tight)
        
        logit_glob, logit_fade, logit_tight = self.head_global(v_glob), self.head_fade(v_fade), self.head_tight(v_tight)
        
        if self.use_dwt and img_L is not None and img_R is not None:
            dwt_L = compute_dwt_LL_tensor(img_L)
            dwt_R = compute_dwt_LL_tensor(img_R)
            v_dwt = self._siamese_fusion(self.enc_dwt(dwt_L), self.enc_dwt(dwt_R), self.proj_dwt)
            logit_dwt = self.head_dwt(v_dwt)
            
            mean_logit = (logit_glob + logit_fade + logit_tight + logit_dwt) / 4.0
            return mean_logit, logit_glob, logit_fade, logit_tight, logit_dwt
        else:
            mean_logit = (logit_glob + logit_fade + logit_tight) / 3.0
            return mean_logit, logit_glob, logit_fade, logit_tight

# ==========================================
# 6. DÉTERMINISME DES SPLITS
# ==========================================
def get_deterministic_split(exp, fold):
    real_df = pd.read_csv(REAL_CSV_PATH)
    real_df = real_df.sort_values('uid').reset_index(drop=True)
    real_df['path_siamese'] = real_df['uid'].apply(lambda u: os.path.join(REAL_SIAM_DIR, f"{u}.pt"))
    real_df['path_global']  = real_df['uid'].apply(lambda u: os.path.join(REAL_GLOB_DIR, f"{u}.pt"))
    
    train_val_pool = real_df.copy()
    
    unique_ids = np.sort(train_val_pool['uid'].unique())
    step_size = len(unique_ids) // N_EXP
    shift_amount = exp * step_size
    shifted_ids = np.roll(unique_ids, shift=shift_amount)
    
    chunks = np.array_split(shifted_ids, K_SPLITS)
    id_to_fold = {uid: k for k, chunk in enumerate(chunks) for uid in chunk}
    
    train_val_pool['fold'] = train_val_pool['uid'].map(id_to_fold)
    
    fold_train_real = train_val_pool[train_val_pool['fold'] != fold].copy()
    val_df          = train_val_pool[train_val_pool['fold'] == fold].copy()
    
    if USE_RELABEL:
        relabel_df = pd.read_csv(RELABEL_CSV_PATH)
        relabel_map = dict(zip(relabel_df['uid'], relabel_df['is_pathologic_fixed_soft']))
        fold_train_real['label'] = fold_train_real['uid'].map(relabel_map).fillna(fold_train_real['label'])

    if USE_SYNTHETIC:
        synth_df_raw = pd.read_csv(SYNTH_CSV_PATH)

        if SYNTH_LABELS and os.path.exists(SYNTH_LABELS):
            synth_override = pd.read_csv(SYNTH_LABELS)
            synth_map = dict(zip(synth_override['uid'], synth_override['is_pathologic']))
            synth_df_raw[SOFT_LABEL_COL] = synth_df_raw['uid'].map(synth_map).fillna(synth_df_raw[SOFT_LABEL_COL])

        synth_df = pd.DataFrame({
            'uid': synth_df_raw['uid'],
            'path_siamese': synth_df_raw['uid'].apply(lambda u: os.path.join(SYNTH_SIAM_DIR, f"{u}.pt")),
            'path_global': synth_df_raw['uid'].apply(lambda u: os.path.join(SYNTH_GLOB_DIR, f"{u}.pt")),
            'label': synth_df_raw[SOFT_LABEL_COL],
            'fold': -1 
        })

        train_df = pd.concat([fold_train_real, synth_df], ignore_index=True)
    else:
        train_df = fold_train_real.copy()
        
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return train_df, val_df

# ==========================================
# 7. LOGIQUE DU WORKER
# ==========================================
def worker_main(args_worker):
    exp = args_worker.exp
    fold = args_worker.fold
    device_str = args_worker.device
    
    print(f"▶️ Démarrage Worker [Exp {exp} | Fold {fold}] sur {device_str}")
    seed_everything(42 + exp*10 + fold) 
    
    train_df, val_df = get_deterministic_split(exp, fold)
    device = torch.device(device_str)

    train_loader = DataLoader(DaTMashupDataset(train_df, is_train=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=DATALOADER_WORKERS)
    val_loader   = DataLoader(DaTMashupDataset(val_df, is_train=False), batch_size=BATCH_SIZE, shuffle=False, num_workers=DATALOADER_WORKERS)
    
    model = MashupInternalEnsembleCNN(dropout_rate=DROPOUT).to(device)
    
    backbone_params = list(model.enc_global.parameters()) + list(model.enc_fade.parameters()) + list(model.enc_tight.parameters())
    head_params = list(model.proj_global.parameters()) + list(model.proj_fade.parameters()) + list(model.proj_tight.parameters()) + \
                  list(model.head_global.parameters()) + list(model.head_fade.parameters()) + list(model.head_tight.parameters())
    
    if USE_DWT:
        backbone_params += list(model.enc_dwt.parameters())
        head_params += list(model.proj_dwt.parameters()) + list(model.head_dwt.parameters())
        
    optimizer = optim.AdamW([{'params': backbone_params, 'lr': LR_BACKBONE}, {'params': head_params, 'lr': LR_HEADS}], weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss() 
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(3, EPOCHS//4), eta_min=ETA_MIN)
    
    best_val_loss = float('inf')
    best_preds = []
    patience_counter = 0
    history_train_loss, history_val_loss = [], []
    
    model_save_path = os.path.join(OUTPUT_DIR, f"mashup_ensemble_exp_{exp}_fold_{fold}.pth")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_train_main_loss = 0.0
        
        for img_G, img_G_flip, img_L_fade, img_R_fade, img_L_tight, img_R_tight, img_L, img_R, labels, _ in train_loader:
            img_G, img_G_flip = img_G.to(device), img_G_flip.to(device)
            img_L_fade, img_R_fade = img_L_fade.to(device), img_R_fade.to(device)
            img_L_tight, img_R_tight = img_L_tight.to(device), img_R_tight.to(device)
            img_L, img_R = img_L.to(device), img_R.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            if USE_DWT:
                mean_logit, logit_glob, logit_fade, logit_tight, logit_dwt = model(img_G, img_G_flip, img_L_fade, img_R_fade, img_L_tight, img_R_tight, img_L, img_R)
                loss_mean  = criterion(mean_logit, labels)
                total_loss = loss_mean + AUX_WEIGHT * (criterion(logit_glob, labels) + criterion(logit_fade, labels) + criterion(logit_tight, labels) + criterion(logit_dwt, labels))
            else:
                mean_logit, logit_glob, logit_fade, logit_tight = model(img_G, img_G_flip, img_L_fade, img_R_fade, img_L_tight, img_R_tight)
                loss_mean  = criterion(mean_logit, labels)
                total_loss = loss_mean + AUX_WEIGHT * (criterion(logit_glob, labels) + criterion(logit_fade, labels) + criterion(logit_tight, labels))
            
            total_loss.backward()
            optimizer.step()
            epoch_train_main_loss += loss_mean.item()
        
        avg_train_loss = epoch_train_main_loss / len(train_loader)
        history_train_loss.append(avg_train_loss)
            
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for img_G, img_G_flip, img_L_fade, img_R_fade, img_L_tight, img_R_tight, img_L, img_R, labels, uids in val_loader:
                img_G, img_G_flip = img_G.to(device), img_G_flip.to(device)
                img_L_fade, img_R_fade = img_L_fade.to(device), img_R_fade.to(device)
                img_L_tight, img_R_tight = img_L_tight.to(device), img_R_tight.to(device)
                img_L, img_R = img_L.to(device), img_R.to(device)
                
                # === PASSE 1 : Image normale ===
                if USE_DWT:
                    mean_logit_1, _, _, _, _ = model(img_G, img_G_flip, img_L_fade, img_R_fade, img_L_tight, img_R_tight, img_L, img_R)
                else:
                    mean_logit_1, _, _, _ = model(img_G, img_G_flip, img_L_fade, img_R_fade, img_L_tight, img_R_tight)
                    
                probs_1 = torch.sigmoid(mean_logit_1).cpu().numpy().flatten()
                
                # === PASSE 2 : TTA ===
                if USE_TTA:
                    if USE_DWT:
                        mean_logit_2, _, _, _, _ = model(img_G_flip, img_G, img_R_fade, img_L_fade, img_R_tight, img_L_tight, img_R, img_L)
                    else:
                        mean_logit_2, _, _, _ = model(img_G_flip, img_G, img_R_fade, img_L_fade, img_R_tight, img_L_tight)
                    probs_2 = torch.sigmoid(mean_logit_2).cpu().numpy().flatten()
                else:
                    probs_2 = probs_1.copy()
                
                probs_tta = (probs_1 + probs_2) / 2.0
                labels_flat = labels.cpu().numpy().flatten()
                
                for uid_val, label_val, p_norm, p_flip, p_tta in zip(uids, labels_flat, probs_1, probs_2, probs_tta):
                    fold_preds.append({
                        'uid': str(uid_val), # Cast en string pour être sûr (validate_pipeline.py l'exige)
                        'label': float(label_val), 
                        'prob_normal': float(p_norm),
                        'prob_flip': float(p_flip),
                        'prob_tta': float(p_tta),
                        'exp': exp, 
                        'fold': fold
                    })
        
        df_preds = pd.DataFrame(fold_preds)
        val_loss = log_loss(df_preds['label'], df_preds['prob_tta'])
        history_val_loss.append(val_loss)
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_preds = df_preds.copy()
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0  
        else:
            patience_counter += 1 

        if PATIENCE is not None and patience_counter >= PATIENCE:
            break

    plt.figure(figsize=(10, 5))
    plt.plot(history_train_loss, label='Train Loss (Mean Logit)', color='blue')
    plt.plot(history_val_loss, label='Validation Loss (Log Loss)', color='orange')
    plt.title(f"Loss Curve - Exp {exp} / Fold {fold} (DWT: {USE_DWT})")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, f"loss_curve_exp_{exp}_fold_{fold}.png"))
    plt.close()

    fold_auc = roc_auc_score(best_preds['label'], best_preds['prob_tta'])
    
    best_preds.to_csv(os.path.join(PREDS_DIR, f"preds_exp_{exp}_fold_{fold}.csv"), index=False)
    with open(os.path.join(PREDS_DIR, f"stats_exp_{exp}_fold_{fold}.json"), "w") as f:
        json.dump({'exp': exp, 'fold': fold, 'val_loss': best_val_loss, 'val_auc': fold_auc, 'stopped_epoch': epoch+1}, f)

    print(f"✅ Worker Terminé | Best Loss: {best_val_loss:.4f} | AUC: {fold_auc:.4f}")

# ==========================================
# 8. ORCHESTRATEUR PRINCIPAL
# ==========================================
def run_orchestrator():
    print(f"\n{'='*80}")
    print(f"🚀 ENTRAÎNEMENT SUBPROCESS INTERNAL ENSEMBLE : {N_EXP} EXPs x {K_SPLITS}-FOLDS")
    print(f"🌲 BRANCHE ONDELETTE (DWT) : {'ACTIVÉE ✅' if USE_DWT else 'DÉSACTIVÉE ❌'}")
    print(f"🧪 DONNÉES SYNTHÉTIQUES  : {'ACTIVÉES ✅' if USE_SYNTHETIC else 'DÉSACTIVÉES ❌'}")
    print(f"🖥️  Workers max : {MAX_CONCURRENT_FOLDS}")
    
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0: print("⚠️ Aucun GPU détecté, entraînement sur CPU.")
    
    tasks = []
    task_idx = 0
    for exp in range(N_EXP):
        for fold in range(K_SPLITS):
            device_str = f"cuda:{task_idx % num_gpus}" if num_gpus > 0 else "cpu"
            tasks.append((exp, fold, device_str))
            task_idx += 1

    def run_worker_cmd(task):
        exp, fold, device_str = task
        log_file = os.path.join(LOGS_DIR, f"worker_exp_{exp}_fold_{fold}.log")
        
        # --- VITAL : On transmet le flag --no-synth au sous-processus si nécessaire ---
        cmd = [sys.executable, __file__, '--worker', '--exp', str(exp), '--fold', str(fold), '--device', device_str]
        if not USE_SYNTHETIC:
            cmd.append("--no-synth")
            
        with open(log_file, "w") as f:
            res = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            
        if res.returncode != 0:
            return False, f"❌ Erreur critique sur Exp {exp} Fold {fold}. Voir le log : {log_file}"
        
        try:
            with open(os.path.join(PREDS_DIR, f"stats_exp_{exp}_fold_{fold}.json"), "r") as f:
                stats = json.load(f)
            return True, stats
        except Exception as e:
            return False, f"❌ Impossible de lire les résultats de Exp {exp} Fold {fold}: {e}"

    master_oof = pd.DataFrame()
    fold_stats = []
    
    pbar = tqdm(total=len(tasks), desc="⏳ Folds complétés (Orchestrateur)", position=0)
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FOLDS) as executor:
        futures = {executor.submit(run_worker_cmd, t): t for t in tasks}
        
        for future in as_completed(futures):
            success, result = future.result()
            if success:
                stats = result
                preds_df = pd.read_csv(os.path.join(PREDS_DIR, f"preds_exp_{stats['exp']}_fold_{stats['fold']}.csv"))
                master_oof = pd.concat([master_oof, preds_df], ignore_index=True)
                fold_stats.append(stats)
                
                pbar.write(f"✅ Exp {stats['exp']} Fold {stats['fold']} complété | Loss: {stats['val_loss']:.4f} | AUC: {stats['val_auc']:.4f}")
            else:
                pbar.write(result) 
            pbar.update(1)

    pbar.close()

    # --- SAUVEGARDE STRICTE POUR LE VALIDATE_PIPELINE.PY ---
    master_oof.to_csv(os.path.join(OUTPUT_DIR, 'master_oof_complet.csv'), index=False)
    
    # Validation attend : uid, label, prob, exp, fold
    oof_normal = master_oof[['uid', 'label', 'prob_normal', 'exp', 'fold']].rename(columns={'prob_normal': 'prob'})
    oof_normal.to_csv(os.path.join(OUTPUT_DIR, 'master_oof_predictions.csv'), index=False)

    oof_flip = master_oof[['uid', 'label', 'prob_flip', 'exp', 'fold']].rename(columns={'prob_flip': 'prob'})
    oof_flip.to_csv(os.path.join(OUTPUT_DIR, 'master_oof_predictions_flip.csv'), index=False)

    pd.DataFrame(fold_stats).to_csv(os.path.join(OUTPUT_DIR, 'fold_statistics.csv'), index=False)
    
    print(f"\n✅ Terminé ! Résultats OOF et graphiques sauvegardés dans : {OUTPUT_DIR}")

# ==========================================
# ENTRY POINT CLI
# ==========================================
if __name__ == '__main__':
    if args.worker:
        worker_main(args)
    else:
        run_orchestrator()