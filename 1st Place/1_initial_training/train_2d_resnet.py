import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from omegaconf import OmegaConf
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from sklearn.metrics import roc_auc_score, log_loss

# ==========================================
# CONFIGURATION ET HYPERPARAMÈTRES OPTIMISÉS
# ==========================================

cfg = OmegaConf.merge(
    OmegaConf.load("0_configs/default.yaml"), 
    OmegaConf.from_cli()
)

CSV_PATH    = cfg.paths.folds_2d_csv
OUTPUT_DIR = Path(cfg.paths.phase1_out) / 'resnet2d'

# --- SWITCH DE MODE ---
# 'holdout' : Entraîne uniquement sur 4 folds et valide sur VAL_FOLD (Rapide)
# 'kfold'   : Entraîne les 5 modèles et génère le fichier OOF global (Pour le Stacking)
RUN_MODE    = 'kfold'  
VAL_FOLD    = 4        # Ignoré si RUN_MODE = 'kfold'
N_SPLITS    = 5

# --- HYPERPARAMÈTRES GAGNANTS ---
LR              = 5e-4
ETA_MIN         = 1e-4
WEIGHT_DECAY    = 1e-4
DROPOUT         = 0.0
UNFREEZE_BLOCKS = 4

BATCH_SIZE  = 32
EPOCHS      = 30  # 10 (Cycle 1) + 20 (Cycle 2) = 30 époques parfaites

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# AUGMENTATIONS ET DATASET
# ==========================================
CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR_STD  = [0.2675, 0.2565, 0.2761]

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    transforms.RandomErasing(
        p=0.3,
        scale=(0.02, 0.1),
        ratio=(0.3, 3.3),
        value=0
    )
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

class DaT2DDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(row['slice_path'], cv2.IMREAD_GRAYSCALE)
        img = cv2.merge([img, img, img]) 
        img = Image.fromarray(img)
        if self.transform: img = self.transform(img)
        return img, torch.tensor([row['label']], dtype=torch.float32), row['uid']

# ==========================================
# ARCHITECTURE OPTIMISÉE EN ENTONNOIR
# ==========================================
class MedicalResNet(nn.Module):
    def __init__(self, dropout_rate=0.0, unfreeze_blocks=4):
        super().__init__()

        # ==========================================================
        # ResNet-50 CIFAR-100
        # ==========================================================
        backbone = models.resnet50(
            weights=None,
            num_classes=100
        )

        # CIFAR-style stem
        backbone.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # Pas de maxpool initial
        backbone.maxpool = nn.Identity()

        # ==========================================================
        # Charger les poids HF
        # ==========================================================

        checkpoint_path = hf_hub_download(
            repo_id="edadaltocg/resnet50_cifar100",
            filename="model.safetensors"
        )

        state_dict = load_file(checkpoint_path)

        # Le checkpoint contient normalement :
        # conv1, bn1, layer1, layer2, layer3, layer4, fc
        backbone.load_state_dict(state_dict, strict=True)

        # ==========================================================
        # On garde uniquement le backbone
        # ==========================================================
        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )

        # ==========================================================
        # FREEZE / UNFREEZE
        #
        # Même logique que ton script original :
        #
        # 0 = tout freeze
        # 1 = layer4
        # 2 = layer3 + layer4
        # 3 = layer2 + layer3 + layer4
        # 4 = tout
        # ==========================================================
        for param in self.features.parameters():
            param.requires_grad = False

        if unfreeze_blocks >= 1:
            for param in backbone.layer4.parameters():
                param.requires_grad = True

        if unfreeze_blocks >= 2:
            for param in backbone.layer3.parameters():
                param.requires_grad = True

        if unfreeze_blocks >= 3:
            for param in backbone.layer2.parameters():
                param.requires_grad = True

        if unfreeze_blocks >= 4:
            for param in self.features.parameters():
                param.requires_grad = True

        # ==========================================================
        # Pooling
        # ==========================================================
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # ResNet-50 = 2048 features
        # avg + max = 4096
        self.head = nn.Sequential(
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.features(x)

        avg_x = self.avg_pool(x).flatten(1)
        max_x = self.max_pool(x).flatten(1)

        x = torch.cat([avg_x, max_x], dim=1)

        return self.head(x)

# ==========================================
# BOUCLE PRINCIPALE (HOLD-OUT OU K-FOLD)
# ==========================================
def main():
    print(f"\n{'='*60}")
    print(f"🚀 ENTRAÎNEMENT RESNET-50 FINAL | Mode: {RUN_MODE.upper()}")
    print(f"{'='*60}\n")
    
    df = pd.read_csv(CSV_PATH)
    
    # Détermination des folds à traiter
    if RUN_MODE == 'holdout':
        folds_to_run = [VAL_FOLD]
    else:
        folds_to_run = range(N_SPLITS)
        
    df_oof_global = pd.DataFrame(columns=['uid', 'label', 'prob_2d'])
    
    for fold in folds_to_run:
        print(f"\n{'*'*18} FOLD {fold} {'*'*18}")
        
        train_df = df[df['fold'] != fold]
        val_df   = df[df['fold'] == fold]
        
        train_loader = DataLoader(DaT2DDataset(train_df, transform=train_transforms), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        val_loader   = DataLoader(DaT2DDataset(val_df, transform=val_transforms), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        
        # Réinitialisation stricte du modèle et de l'optimiseur par fold
        model = MedicalResNet(dropout_rate=DROPOUT, unfreeze_blocks=UNFREEZE_BLOCKS).to(DEVICE)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss()
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=ETA_MIN)
        
        best_val_loss = float('inf')
        best_patient_preds = None
        model_save_path = os.path.join(OUTPUT_DIR, f"resnet2d_fold_{fold}.pth")
        
        for epoch in range(EPOCHS):
            model.train()
            train_loss = 0.0
            
            pbar_train = tqdm(train_loader, desc=f"Ep {epoch+1:02d}/{EPOCHS} [Train]", leave=False)
            for imgs, labels, _ in pbar_train:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)
            
            model.eval()
            fold_preds = []
            
            pbar_val = tqdm(val_loader, desc=f"Ep {epoch+1:02d}/{EPOCHS} [Val]  ", leave=False)
            with torch.no_grad():
                for imgs, labels, uids in pbar_val:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(imgs)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    for i in range(len(uids)):
                        fold_preds.append({'uid': uids[i], 'label': labels[i].item(), 'prob': probs[i][0]})
            
            # Agrégation Patient
            df_preds = pd.DataFrame(fold_preds)
            patient_preds = df_preds.groupby('uid').mean().reset_index()
            val_loss = log_loss(patient_preds['label'], patient_preds['prob'])
            val_auc  = roc_auc_score(patient_preds['label'], patient_preds['prob'])
            
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            print(f"Ep {epoch+1:02d}/{EPOCHS} | LR: {current_lr:.1e} | Train Loss: {train_loss/len(train_df):.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}", end="")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_patient_preds = patient_preds.copy()
                torch.save(model.state_dict(), model_save_path)
                print(" 🌟 Nouveau best!")
            else:
                print("")
                
        # Fin du Fold -> Sauvegarde OOF
        best_patient_preds.rename(columns={'prob': 'prob_2d'}, inplace=True)
        df_oof_global = pd.concat([df_oof_global, best_patient_preds], ignore_index=True)

    # ==========================================
    # RÉCAPITULATIF FINAL
    # ==========================================
    print(f"\n{'='*60}")
    print("📈 RÉSULTATS GLOBAUX PATIENT")
    print(f"{'='*60}")
    
    final_auc = roc_auc_score(df_oof_global['label'].astype(int), df_oof_global['prob_2d'])
    final_ll  = log_loss(df_oof_global['label'].astype(int), df_oof_global['prob_2d'])
    
    print(f"🏆 Score AUC      : {final_auc:.4f}")
    print(f"🏆 Score Log Loss : {final_ll:.4f}")
    
    if RUN_MODE == 'kfold':
        oof_path = os.path.join(OUTPUT_DIR, 'oof_resnet2d.csv')
        df_oof_global.to_csv(oof_path, index=False)
        print(f"\n✅ Fichier prêt pour le Stacking : {oof_path}")
    else:
        print(f"\n✅ Modèle Hold-Out sauvegardé dans : {OUTPUT_DIR}")

if __name__ == '__main__':
    main()