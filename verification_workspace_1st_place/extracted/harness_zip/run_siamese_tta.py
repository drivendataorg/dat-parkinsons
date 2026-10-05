import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. CONFIGURATION
# ==========================================
OOF_CSV_PATH = '/data/HOME/sansel/dat-park/runs_siamois/mashup_attention/oof_mashup_attention_RAW.csv'
REAL_SIAM_DIR = '/data2/iceberg/dat_parkinson/dataset_siamese_patches/patches_pt'
REAL_GLOB_DIR = '/data2/iceberg/dat_parkinson/dataset_striatums_patches/patches_pt'
REAL_CSV_PATH = '/data2/iceberg/dat_parkinson/dataset_siamese_patches/folds_siamese.csv'

OUTPUT_TTA_CSV = 'tta_siamese_predictions.csv'
MODELS_DIR = '/data/HOME/sansel/dat-park/runs_siamois/mashup_attention'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. CLASSES D'ARCHITECTURE & FADE
# ==========================================
class MedialFade(object):
    def __init__(self, size=32, fade_len=10, side='left'):
        self.mask = torch.ones((3, size, size), dtype=torch.float32)
        if side == 'left':
            fade_start = size - fade_len
            for i in range(fade_start, size):
                self.mask[:, :, i] = 1.0 - (i - fade_start) / (size - 1 - fade_start)
        elif side == 'right':
            fade_end = fade_len
            for i in range(fade_end):
                self.mask[:, :, i] = i / fade_end
    def __call__(self, tensor): return tensor * self.mask

fade_left = MedialFade(size=32, fade_len=10, side='left')
fade_right = MedialFade(size=32, fade_len=10, side='right')
tight_crop = transforms.CenterCrop(24)

def create_backbone():
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2), 
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2), 
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2), 
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten()
    )

class MashupAttentionCNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.enc_global, self.enc_fade, self.enc_tight = create_backbone(), create_backbone(), create_backbone()
        self.proj_global, self.proj_fade, self.proj_tight = nn.Linear(1024, 256), nn.Linear(1024, 256), nn.Linear(1024, 256)
        self.attention = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        self.head = nn.Sequential(nn.LayerNorm(256), nn.SiLU(), nn.Dropout(dropout_rate), nn.Linear(256, 1))

    def _siamese_fusion(self, feat_A, feat_B, projector):
        return projector(torch.cat([feat_A, feat_B, torch.abs(feat_A - feat_B), feat_A * feat_B], dim=1))

    def forward(self, img_G, img_G_flip, img_L_fade, img_R_fade, img_L_tight, img_R_tight):
        vec_global = self._siamese_fusion(self.enc_global(img_G), self.enc_global(img_G_flip), self.proj_global)
        vec_fade   = self._siamese_fusion(self.enc_fade(img_L_fade), self.enc_fade(img_R_fade), self.proj_fade)
        vec_tight  = self._siamese_fusion(self.enc_tight(img_L_tight), self.enc_tight(img_R_tight), self.proj_tight)
        stacked_views = torch.stack([vec_global, vec_fade, vec_tight], dim=1)
        attn_weights = torch.softmax(self.attention(stacked_views), dim=1)
        return self.head(torch.sum(stacked_views * attn_weights, dim=1)), attn_weights

# ==========================================
# 3. DÉFINITION DES TTA PRÉDÉFINIES
# ==========================================
def _sync_geom(L, R, G, t_func):
    """Applique une transformation géométrique synchronisée sur les patchs L et R."""
    combo = torch.cat([L, R], dim=0)
    combo = t_func(combo)
    return combo[0:1], combo[1:2], t_func(G)

TTA_CONFIGS = [
    ("00_Baseline", lambda L, R, G: (L, R, G)),
    ("01_FlipSwap", lambda L, R, G: (R, L, TF.hflip(G))),
    ("02_Rot_Plus5", lambda L, R, G: _sync_geom(L, R, G, lambda x: TF.rotate(x, 5))),
    ("03_Rot_Minus5", lambda L, R, G: _sync_geom(L, R, G, lambda x: TF.rotate(x, -5))),
    ("04_Shift_Up", lambda L, R, G: _sync_geom(L, R, G, lambda x: TF.affine(x, 0, [0,-2], 1, 0))),
    ("05_Shift_Down", lambda L, R, G: _sync_geom(L, R, G, lambda x: TF.affine(x, 0, [0,2], 1, 0))),
    ("06_Shift_Left", lambda L, R, G: _sync_geom(L, R, G, lambda x: TF.affine(x, 0, [-2,0], 1, 0))),
    ("07_Shift_Right", lambda L, R, G: _sync_geom(L, R, G, lambda x: TF.affine(x, 0, [2,0], 1, 0))),
    ("08_Scale_Up", lambda L, R, G: _sync_geom(L, R, G, lambda x: TF.affine(x, 0, [0,0], 1.05, 0))),
    ("09_Scale_Down", lambda L, R, G: _sync_geom(L, R, G, lambda x: TF.affine(x, 0, [0,0], 0.95, 0))),
    ("10_Brightness", lambda L, R, G: (TF.adjust_brightness(L, 1.15), TF.adjust_brightness(R, 1.15), TF.adjust_brightness(G, 1.15))),
    ("11_Blur", lambda L, R, G: (TF.gaussian_blur(L, 3, 0.5), TF.gaussian_blur(R, 3, 0.5), TF.gaussian_blur(G, 3, 0.5)))
]

# ==========================================
# 4. DATASET PATIENT TTA
# ==========================================
class PatientTTADataset(Dataset):
    """Génère un Batch contenant les N augmentations pour UN seul patient."""
    def __init__(self, uid):
        siam_pt = torch.load(os.path.join(REAL_SIAM_DIR, f"{uid}.pt"))
        self.raw_L = siam_pt[0:1]
        self.raw_R = siam_pt[12:13]
        
        glob_pt = torch.load(os.path.join(REAL_GLOB_DIR, f"{uid}.pt"))
        if glob_pt.ndim == 4: glob_pt = glob_pt.squeeze(0)
        self.img_G = glob_pt[0:1]
        if self.img_G.shape[-1] != 48:
            self.img_G = TF.resize(self.img_G, [48, 48], antialias=True)

    def __len__(self): return len(TTA_CONFIGS)

    def __getitem__(self, idx):
        tta_name, tta_func = TTA_CONFIGS[idx]
        L, R, G = self.raw_L.clone(), self.raw_R.clone(), self.img_G.clone()

        is_swapped = ("FlipSwap" in tta_name)
        L, R, G = tta_func(L, R, G)

        L, R, G = L.expand(3, -1, -1), R.expand(3, -1, -1), G.expand(3, -1, -1)
        G_flip = torch.flip(G, dims=[1])

        # Respect Anatomique pour le Fade
        if is_swapped:
            L_fade, R_fade = fade_right(L), fade_left(R)
        else:
            L_fade, R_fade = fade_left(L), fade_right(R)

        L_tight, R_tight = tight_crop(L), tight_crop(R)

        mean, std = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        def norm(t): return (t - mean) / std

        return (norm(G), norm(G_flip), norm(L_fade), norm(R_fade), norm(L_tight), norm(R_tight), tta_name)

# ==========================================
# 5. ORCHESTRATEUR TTA
# ==========================================
def main():
    print(f"\n{'='*80}\n🚀 LANCEMENT TEST TIME AUGMENTATION (SIAMESE MASHUP)\n{'='*80}")
    
    # 1. Chargement de l'OOF
    df_oof = pd.read_csv(OOF_CSV_PATH)
    
    # 2. 🌟 CORRECTION : Récupération des folds depuis le fichier source
    df_folds = pd.read_csv(REAL_CSV_PATH)[['uid', 'fold']]
    
    # On merge pour récupérer la colonne 'fold' de manière sécurisée
    df_oof = pd.merge(df_oof, df_folds, on='uid', how='inner')

    # Chargement en mémoire des 5 modèles pour inférence rapide
    models_by_fold = {}
    print("📥 Chargement des modèles...")
    for fold in range(5):
        m = MashupAttentionCNN().to(DEVICE)
        model_path = os.path.join(MODELS_DIR, f"mashup_attn_fold_{fold}.pth")
        if not os.path.exists(model_path):
            print(f"❌ ERREUR FATALE : Modèle introuvable : {model_path}")
            return
            
        m.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
        m.eval()
        models_by_fold[fold] = m

    all_results = []
    
    for _, row in tqdm(df_oof.iterrows(), total=len(df_oof), desc="Inférence TTA par patient"):
        uid = row['uid']
        fold = int(row['fold'])
        true_label = row['label']
        prob_baseline_oof = row['prob'] # La proba originale sans TTA pour comparer

        model = models_by_fold[fold]
        
        # Le dataset génère exactement len(TTA_CONFIGS) images.
        dataset = PatientTTADataset(uid)
        loader = DataLoader(dataset, batch_size=len(TTA_CONFIGS), shuffle=False)
        
        # Extraction du seul et unique batch contenant toutes les TTA du patient
        img_G, img_G_flip, img_L_fade, img_R_fade, img_L_tight, img_R_tight, tta_names = next(iter(loader))
        
        with torch.no_grad():
            outputs, attns = model(
                img_G.to(DEVICE), img_G_flip.to(DEVICE), 
                img_L_fade.to(DEVICE), img_R_fade.to(DEVICE), 
                img_L_tight.to(DEVICE), img_R_tight.to(DEVICE)
            )
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            attns = attns.cpu().numpy()
            
        for i in range(len(tta_names)):
            all_results.append({
                'uid': uid,
                'label': true_label,
                'fold': fold,
                'tta_name': tta_names[i],
                'prob_tta': probs[i],
                'prob_original_oof': prob_baseline_oof,
                'attn_global': attns[i, 0, 0],
                'attn_fade': attns[i, 1, 0],
                'attn_tight': attns[i, 2, 0]
            })

    # Sauvegarde
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(OUTPUT_TTA_CSV, index=False)
    print(f"\n✅ Étude TTA terminée ! {len(df_results)} prédictions générées.")
    print(f"📁 Sauvegarde dans : {OUTPUT_TTA_CSV}")

if __name__ == '__main__':
    main()