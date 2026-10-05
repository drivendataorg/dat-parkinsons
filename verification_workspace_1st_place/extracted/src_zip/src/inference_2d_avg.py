"""
inference_2d.py
===============
Classes d'inférence orientées objet pour les modèles 2D (ResNet et ViT).
Elles encapsulent l'architecture, le chargement des 5 folds, et la prédiction.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from scipy.ndimage import label, gaussian_filter
from skimage.morphology import dilation, disk

from .builder import build_model

# ==========================================
# ARCHITECTURES
# ==========================================
class MedicalResNet(nn.Module):
    def __init__(self, dropout_rate=0.0):
        super().__init__()
        backbone = models.resnet50(weights=None) # Pas besoin des poids ImageNet à l'inférence
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),                     
            nn.Dropout(dropout_rate),      
            nn.Linear(512, 1)      
        )

    def forward(self, x):
        x = self.features(x)                                 
        avg_x = self.avg_pool(x).view(x.size(0), -1)         
        max_x = self.max_pool(x).view(x.size(0), -1)         
        x = torch.cat([avg_x, max_x], dim=1)                 
        return self.head(x)

class MedicalViT(nn.Module):
    def __init__(self, dropout_rate=0.4):
        super().__init__()
        self.backbone = models.vit_b_16(weights=None)
        in_features = self.backbone.heads.head.in_features
        self.backbone.heads = nn.Identity() 
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),                     
            nn.Dropout(dropout_rate),      
            nn.Linear(256, 1)      
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


# ==========================================
# CLASSES D'INFÉRENCE ENSEMBLE (5 FOLDS)
# ==========================================
class ResNet2DEnsemble:
    def __init__(self, models_dir, num_folds=5, device=None):
        """
        Initialise l'ensemble ResNet50 et charge les 5 folds en mémoire.
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models = []
        
        print(f"⚙️ Chargement de l'ensemble ResNet2D (5 folds)")
        for fold in range(num_folds):
            model_path = os.path.join(models_dir, f"resnet2d_fold_{fold}.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Poids introuvables : {model_path}")
            
            model = MedicalResNet(dropout_rate=0.0) # Doit correspondre à l'entraînement
            
            # Chargement sécurisé (map_location permet de charger sur CPU si pas de GPU)
            state_dict = torch.load(model_path, map_location=self.device)
            # Gestion de MONAI/PyTorch save formats
            if "state_dict" in state_dict: state_dict = state_dict["state_dict"]
                
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.models.append(model)
            
        # Transforms de validation (identique à l'entraînement)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, slices_2d):
        """
        Prend la liste `slices_2d` du preprocessing et renvoie la proba finale.
        """
        if not slices_2d:
            return 0.5 # Sécurité si aucune coupe n'est valide
            
        # 1. Préparation du batch (Toutes les coupes du patient)
        batch_tensors = []
        for slice_dict in slices_2d:
            img_gray = slice_dict['image']
            img_rgb = cv2.merge([img_gray, img_gray, img_gray]) 
            img_pil = Image.fromarray(img_rgb)
            tensor = self.transform(img_pil)
            batch_tensors.append(tensor)
            
        batch = torch.stack(batch_tensors).to(self.device)
        
        # 2. Inférence sur les 5 folds
        fold_probs = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(batch)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                
                # Agrégation Patient : on moyenne les probas des 5 coupes
                patient_prob_for_this_fold = np.mean(probs)
                fold_probs.append(patient_prob_for_this_fold)
                
        # 3. Agrégation Ensemble : on moyenne les avis des 5 folds
        final_ensemble_prob = np.mean(fold_probs)
        return float(final_ensemble_prob)


class ViT2DEnsemble:
    def __init__(self, models_dir, num_folds=5, device=None):
        """
        Initialise l'ensemble ViT-B/16 et charge les 5 folds en mémoire.
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models = []
        
        print(f"⚙️ Chargement de l'ensemble ViT2D (5 folds)")
        for fold in range(num_folds):
            model_path = os.path.join(models_dir, f"vit2d_fold_{fold}.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Poids introuvables : {model_path}")
            
            model = MedicalViT(dropout_rate=0.4)
            
            state_dict = torch.load(model_path, map_location=self.device)
            if "state_dict" in state_dict: state_dict = state_dict["state_dict"]
                
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.models.append(model)
            
        # Transforms de validation Spécifique ViT (Resize 224x224 obligatoire)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, slices_2d):
        """
        Prend la liste `slices_2d` du preprocessing et renvoie la proba finale.
        """
        if not slices_2d:
            return 0.5 
            
        batch_tensors = []
        for slice_dict in slices_2d:
            img_gray = slice_dict['image']
            img_rgb = cv2.merge([img_gray, img_gray, img_gray]) 
            img_pil = Image.fromarray(img_rgb)
            tensor = self.transform(img_pil)
            batch_tensors.append(tensor)
            
        batch = torch.stack(batch_tensors).to(self.device)
        
        fold_probs = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(batch)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                
                patient_prob_for_this_fold = np.mean(probs)
                fold_probs.append(patient_prob_for_this_fold)
                
        final_ensemble_prob = np.mean(fold_probs)
        return float(final_ensemble_prob)

# ==========================================
# CLASSE D'INFÉRENCE SIAMOISE (5 FOLDS)
# ==========================================
class SimpleSiameseEnsemble:
    # 🔑 Ajout du paramètre "prefix"
    def __init__(self, models_dir, num_folds=5, device=None, prefix="simple_siamese"):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models = []
        
        print(f"⚙️ Chargement de l'ensemble {prefix} (MIP Axial) - {num_folds} folds")
        for fold in range(num_folds):
            # 🔑 Utilisation du préfixe pour trouver le bon fichier de poids
            model_path = os.path.join(models_dir, f"{prefix}_fold_{fold}.pth")
            if not os.path.exists(model_path):
                print(f"⚠️ Poids introuvables ignorés : {model_path}")
                continue
            
            model = build_model("simple_siamese") 
            
            state_dict = torch.load(model_path, map_location=self.device)
            if "state_dict" in state_dict: 
                state_dict = state_dict["state_dict"]
                
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.models.append(model)
            
        self.transform = transforms.Compose([
            transforms.Resize((128, 128), antialias=True),
        ])
            
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def _prepare_tensors(self, patch_L, patch_R):
        mip_L = patch_L[0:1] # (1, 32, 32)
        mip_R = patch_R[0:1] # (1, 32, 32)
        
        # 1. On recrée le tenseur "combo" exactement comme dans le Dataset
        combo = torch.cat([mip_L, mip_R], dim=0)
        
        # 2. Application du Resize avec antialias
        combo = self.transform(combo)
        
        # 3. Expansion en 3 canaux et ajout de la dimension Batch
        img_L = combo[0:1].expand(3, -1, -1).unsqueeze(0).to(self.device)
        img_R = combo[1:2].expand(3, -1, -1).unsqueeze(0).to(self.device)
        
        # 4. Normalisation ImageNet
        img_L = (img_L - self.mean) / self.std
        img_R = (img_R - self.mean) / self.std
        
        return img_L, img_R

    def predict(self, siamese_data):
        if not siamese_data or 'patch_L' not in siamese_data or 'patch_R' not in siamese_data:
            return 0.5
            
        img_L, img_R = self._prepare_tensors(siamese_data['patch_L'], siamese_data['patch_R'])
        
        fold_probs = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(img_L, img_R)
                prob = torch.sigmoid(outputs).item()
                fold_probs.append(prob)
                
        return float(np.mean(fold_probs))


# ==========================================
# CLASSE D'INFÉRENCE MULTI-VIEW DUAL-STREAM (5 FOLDS)
# ==========================================
class DualStreamSiameseEnsemble:
    def __init__(self, models_dir, num_folds=5, device=None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models = []
        
        print(f"⚙️ Chargement de l'ensemble Dual-Stream Siamese 2.5D ({num_folds} folds)")
        for fold in range(num_folds):
            model_path = os.path.join(models_dir, f"multiview_siamese_fold_{fold}.pth")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Poids introuvables : {model_path}")
            
            model = build_model("multiview_siamese") 
            
            state_dict = torch.load(model_path, map_location=self.device)
            if "state_dict" in state_dict: 
                state_dict = state_dict["state_dict"]
                
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.models.append(model)

        # 🎯 CORRECTION : Utilisation stricte du même Transform qu'à l'entraînement
        self.transform = transforms.Compose([
            transforms.Resize((128, 128), antialias=True),
        ])

    def _prepare_tensors(self, patch_L, patch_R):
        idx_ax = [0, 2, 3, 4, 5, 6]
        idx_cor = [1, 7, 8, 9, 10, 11]
        
        ax_L = patch_L[idx_ax]  # (6, 32, 32)
        ax_R = patch_R[idx_ax]  
        cor_L = patch_L[idx_cor]
        cor_R = patch_R[idx_cor]
        
        # 1. On recrée le tenseur "combo_subset" exactement dans le même ordre que le Dataset
        combo_subset = torch.cat([ax_L, ax_R, cor_L, cor_R], dim=0) # (24, 32, 32)
        
        # 2. Application du Resize global avec antialias
        combo_subset = self.transform(combo_subset) # (24, 128, 128)
        
        # 3. Redécoupage et ajout de la dimension batch
        ax_L = combo_subset[0:6].unsqueeze(0).to(self.device)
        ax_R = combo_subset[6:12].unsqueeze(0).to(self.device)
        cor_L = combo_subset[12:18].unsqueeze(0).to(self.device)
        cor_R = combo_subset[18:24].unsqueeze(0).to(self.device)

        # 4. Normalisation de base [-1, 1] identique à l'entraînement
        ax_L = (ax_L - 0.5) / 0.5
        ax_R = (ax_R - 0.5) / 0.5
        cor_L = (cor_L - 0.5) / 0.5
        cor_R = (cor_R - 0.5) / 0.5

        return (ax_L, ax_R), (cor_L, cor_R)

    def predict(self, siamese_data):
        if not siamese_data or 'patch_L' not in siamese_data or 'patch_R' not in siamese_data:
            return 0.5 
            
        ax_pair, cor_pair = self._prepare_tensors(siamese_data['patch_L'], siamese_data['patch_R'])
        
        fold_probs = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(ax_pair, cor_pair)
                prob = torch.sigmoid(outputs).item()
                fold_probs.append(prob)
                
        return float(np.mean(fold_probs))
    
# --- TRANSFORMATIONS COMMUNES ---
class SpatialMedialFadeInference(object):
    def __init__(self, size=32, fade_start=22):
        self.mask = torch.ones((3, size, size), dtype=torch.float32)
        for i in range(fade_start, size):
            self.mask[:, :, i] = 1.0 - (i - fade_start) / (size - 1 - fade_start)
    def __call__(self, tensor):
        # Gardé sur le même device que le tenseur d'entrée
        return tensor * self.mask.to(tensor.device)

# ==========================================
# CLASSE D'INFÉRENCE : MASHUP CLASSIQUE (3 BRANCHES)
# ==========================================
class SiameseMashupEnsemble:
    def __init__(self, models_dir, num_folds=5, device=None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models = []
        
        print(f"⚙️ Chargement de l'ensemble Siamese Mashup (3 Branches) - {num_folds} folds")
        for fold in range(num_folds):
            model_path = os.path.join(models_dir, f"mashup_attn_fold_{fold}.pth")
            if not os.path.exists(model_path): continue
            
            model = build_model("siamese_mashup") 
            state_dict = torch.load(model_path, map_location=self.device)
            if "state_dict" in state_dict: state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict)
            model.to(self.device); model.eval()
            self.models.append(model)
            
        self.fade_transform = SpatialMedialFadeInference(size=32, fade_start=22)
        self.tight_crop = transforms.CenterCrop(24)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def _prepare_tensors(self, siamese_data):
        img_L = siamese_data['patch_L'][0:1].clone().expand(3, -1, -1)
        img_R = siamese_data['patch_R'][0:1].clone().expand(3, -1, -1)
        img_G = siamese_data['patch_Global'][0:1].clone()
        
        # Resize CPU
        if img_G.shape[-1] != 48: 
            img_G = transforms.functional.resize(img_G, [48, 48], antialias=True)
        img_G = img_G.expand(3, -1, -1)
        
        # Flip (dims=[1] car entraîné comme ça)
        img_G_flip = torch.flip(img_G, dims=[1])
        
        img_L_fade = self.fade_transform(img_L)
        img_R_fade = self.fade_transform(img_R)
        img_L_tight = self.tight_crop(img_L)
        img_R_tight = self.tight_crop(img_R)
        
        # 🔑 Normalisation 100% CPU (comme dans le __getitem__ de l'entraînement)
        mean_cpu = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std_cpu  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        def norm(t): 
            # Opération mathématique sur CPU, PUIS ajout du batch et envoi GPU
            t_norm = (t - mean_cpu) / std_cpu
            return t_norm.unsqueeze(0).to(self.device)
        
        return (norm(img_G), norm(img_G_flip), norm(img_L_fade), norm(img_R_fade), norm(img_L_tight), norm(img_R_tight))

    def predict(self, siamese_data):
        if not all(k in siamese_data for k in ('patch_L', 'patch_R', 'patch_Global')): return 0.5 
        tensors = self._prepare_tensors(siamese_data)
        
        fold_probs = []
        with torch.no_grad():
            for model in self.models:
                fold_probs.append(torch.sigmoid(model(*tensors)).item())
        return float(np.mean(fold_probs)) if fold_probs else 0.5


# ==========================================
# CLASSE D'INFÉRENCE : MASHUP ALIGN (4 BRANCHES)
# ==========================================
class MaskedECCCoupledAlignmentInference(object):
    def __init__(self, dilation_radius=1, blur_sigma=1.0):
        self.dilation_radius = dilation_radius
        self.blur_sigma = blur_sigma

    def _isolate_feathered_intensity(self, img_np):
        H, W = img_np.shape
        cy, cx = H // 2, W // 2
        img_max = img_np.max()
        if img_max == 0: return np.zeros_like(img_np)
            
        best_mask = None
        for t in np.arange(0.50, 0.95, 0.05):
            binary = img_np > (img_max * t)
            labeled, num_features = label(binary)
            if num_features == 0: continue
            min_dist, central_label = float('inf'), 0
            for i in range(1, num_features + 1):
                coords = np.argwhere(labeled == i)
                if len(coords) == 0: continue
                my, mx = coords.mean(axis=0)
                dist = (my - cy)**2 + (mx - cx)**2
                if dist < min_dist:
                    min_dist, central_label = dist, i
            if central_label == 0: continue
            mask = (labeled == central_label)
            if not (mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()):
                best_mask = mask
                break 
                
        if best_mask is None:
            binary = img_np > (img_max * 0.9)
            labeled, _ = label(binary)
            if labeled.max() > 0:
                sizes = np.bincount(labeled.flat)[1:]
                best_mask = (labeled == (sizes.argmax() + 1))
            else:
                best_mask = binary

        dilated_mask = dilation(best_mask, disk(self.dilation_radius)).astype(float) if self.dilation_radius > 0 else best_mask.astype(float)
        soft_mask = gaussian_filter(dilated_mask, sigma=self.blur_sigma)
        return img_np * soft_mask

    def __call__(self, img_L, img_R):
        device = img_L.device
        np_L = img_L[0].detach().cpu().numpy().astype(np.float32)
        np_R = img_R[0].detach().cpu().numpy().astype(np.float32)
        rows, cols = np_L.shape

        mask_L = self._isolate_feathered_intensity(np_L)
        mask_R = self._isolate_feathered_intensity(np_R)

        img_ref_norm = cv2.normalize(mask_L, None, 0, 1, cv2.NORM_MINMAX)
        img_moving_norm = cv2.normalize(mask_R, None, 0, 1, cv2.NORM_MINMAX)
        
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-3)
        
        try:
            _, warp_matrix = cv2.findTransformECC(img_ref_norm, img_moving_norm, warp_matrix, warp_mode, criteria, None, 1)
            warp_matrix_correct = cv2.invertAffineTransform(warp_matrix)
        except cv2.error:
            warp_matrix_correct = np.eye(2, 3, dtype=np.float32)

        aligned_channels = []
        for c in range(img_R.shape[0]):
            channel_np = img_R[c].detach().cpu().numpy().astype(np.float32)
            aligned_c = cv2.warpAffine(channel_np, warp_matrix_correct, (cols, rows), 
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            aligned_channels.append(torch.tensor(aligned_c, dtype=torch.float32, device=device))
            
        img_R_aligned = torch.stack(aligned_channels, dim=0)
        return img_L, img_R_aligned

class SiameseMashupAlignEnsemble:
    def __init__(self, models_dir, num_folds=5, device=None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models = []
        
        print(f"⚙️ Chargement de l'ensemble Siamese Mashup ALIGN (4 Branches) - {num_folds} folds")
        for fold in range(num_folds):
            model_path = os.path.join(models_dir, f"mashup_attn_4br_fold_{fold}.pth") 
            if not os.path.exists(model_path): continue
            
            model = build_model("siamese_mashup_align") 
            state_dict = torch.load(model_path, map_location=self.device)
            if "state_dict" in state_dict: state_dict = state_dict["state_dict"]
            model.load_state_dict(state_dict)
            model.to(self.device); model.eval()
            self.models.append(model)
            
        self.align_transform = MaskedECCCoupledAlignmentInference()
        self.fade_transform = SpatialMedialFadeInference(size=32, fade_start=22)
        self.tight_crop = transforms.CenterCrop(24)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

    def _prepare_tensors(self, siamese_data):
        # 🔑 CPU
        img_L_base = siamese_data['patch_L'][0:1].clone()
        img_R_base = siamese_data['patch_R'][0:1].clone()
        img_G = siamese_data['patch_Global'][0:1].clone()
        
        if img_G.shape[-1] != 48: 
            img_G = transforms.functional.resize(img_G, [48, 48], antialias=True)
            
        img_L_align, img_R_align = self.align_transform(img_L_base.clone(), img_R_base.clone())
        
        img_L = img_L_base.expand(3, -1, -1)
        img_R = img_R_base.expand(3, -1, -1)
        img_G = img_G.expand(3, -1, -1)
        img_L_align = img_L_align.expand(3, -1, -1)
        img_R_align = img_R_align.expand(3, -1, -1)
        
        # 🔑 CORRECTION: dims=[1]
        img_G_flip = torch.flip(img_G, dims=[1])
        
        img_L_fade, img_R_fade = self.fade_transform(img_L), self.fade_transform(img_R)
        img_L_tight, img_R_tight = self.tight_crop(img_L), self.tight_crop(img_R)
        img_L_align_tight, img_R_align_tight = self.tight_crop(img_L_align), self.tight_crop(img_R_align)
        
        def norm(t): 
            t_gpu = t.unsqueeze(0).to(self.device)
            return (t_gpu - self.mean) / self.std
        
        return (norm(img_G), norm(img_G_flip), norm(img_L_fade), norm(img_R_fade), 
                norm(img_L_tight), norm(img_R_tight), norm(img_L_align_tight), norm(img_R_align_tight))

    def predict(self, siamese_data):
        if not all(k in siamese_data for k in ('patch_L', 'patch_R', 'patch_Global')): return 0.5 
        tensors = self._prepare_tensors(siamese_data)
        
        fold_probs = []
        with torch.no_grad():
            for model in self.models:
                fold_probs.append(torch.sigmoid(model(*tensors)).item())
        return float(np.mean(fold_probs)) if fold_probs else 0.5

# ==============================================================================
# 1. MASHUP WAVELET LINEAR ENSEMBLE (MashupAttentionCNNLinear)
# ==============================================================================
class SiameseMashupWaveletLinearEnsemble:
    """
    Classe d'inférence pour le modèle Mashup avec fusion par Attention Linéaire Simple.
    Utilise 4 vues : Global, Fade, Tight et DWT (Wavelet).
    """
    def __init__(self, models_dir, num_folds=5, device=None, prefix="mashup_attention_dwt_linear"):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models = []
        
        print(f"⚙️ Chargement de l'ensemble Mashup Wavelet Linear ({prefix}) - {num_folds} folds")
        for fold in range(num_folds):
            model_path = os.path.join(models_dir, f"{prefix}_fold_{fold}.pth")
            if not os.path.exists(model_path):
                print(f"⚠️ Poids introuvables ignorés : {model_path}")
                continue
            
            # Instanciation de l'architecture avec attention linéaire
            model = build_model("siamese_mashup_wavelet_linear")
            state_dict = torch.load(model_path, map_location=self.device)
            
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
                
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.models.append(model)
            
        self.fade_transform = SpatialMedialFadeInference(size=32, fade_start=22)
        self.tight_crop = transforms.CenterCrop(24)
        
        # Transforms TTA identiques au script d'évaluation
        self.tta_transforms = [
            lambda x: x,
            transforms.RandomRotation(degrees=(3, 3)),
            transforms.RandomRotation(degrees=(-3, -3)),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
        ]

    def _prepare_tensors(self, siamese_data):
        img_L_raw = siamese_data['patch_L'][0:1].clone()
        img_R_raw = siamese_data['patch_R'][0:1].clone()
        img_G_raw = siamese_data['patch_Global'][0:1].clone()
        
        # 1. Resize Global à 48x48
        if img_G_raw.shape[-1] != 48:
            img_G_raw = transforms.functional.resize(img_G_raw, [48, 48], antialias=True)
            
        # 2. Expansion sur 3 canaux
        img_L = img_L_raw.expand(3, -1, -1)
        img_R = img_R_raw.expand(3, -1, -1)
        img_G = img_G_raw.expand(3, -1, -1)
        
        # 3. Génération des sous-vues
        img_G_flip = torch.flip(img_G, dims=[1])
        img_L_fade = self.fade_transform(img_L)
        img_R_fade = self.fade_transform(img_R)
        img_L_tight = self.tight_crop(img_L)
        img_R_tight = self.tight_crop(img_R)
        
        # 4. Normalisation ImageNet (sur CPU avant transfert GPU)
        mean_cpu = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std_cpu  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        def norm(t):
            t_norm = (t - mean_cpu) / std_cpu
            return t_norm.unsqueeze(0).to(self.device)
            
        return (
            norm(img_G), norm(img_G_flip),
            norm(img_L_fade), norm(img_R_fade),
            norm(img_L_tight), norm(img_R_tight),
            norm(img_L), norm(img_R)
        )

    def predict(self, siamese_data, use_tta=False):
        if not all(k in siamese_data for k in ('patch_L', 'patch_R', 'patch_Global')):
            return 0.5
            
        if not self.models:
            return 0.5

        tensors = self._prepare_tensors(siamese_data)
        
        fold_probs = []
        with torch.no_grad():
            for model in self.models:
                if use_tta:
                    tta_outputs = []
                    for t_func in self.tta_transforms:
                        transformed_tensors = tuple(t_func(t) for t in tensors)
                        logits = model(*transformed_tensors)
                        tta_outputs.append(logits)
                    outputs = torch.stack(tta_outputs, dim=0).mean(dim=0)
                else:
                    outputs = model(*tensors)
                
                prob = torch.sigmoid(outputs).item()
                fold_probs.append(prob)

        return float(np.mean(fold_probs)) if fold_probs else 0.5


# ==============================================================================
# 2. MASHUP WAVELET ATTN ENSEMBLE (MashupAttentionCNNWavelet)
# ==============================================================================
class SiameseMashupWaveletAttnEnsemble:
    """
    Classe d'inférence pour le modèle Mashup avec fusion par Transformer Encoder (CLS Token).
    Utilise 4 vues : Global, Fade, Tight et DWT (Wavelet).
    """
    def __init__(self, models_dir, num_folds=5, device=None, prefix="mashup_attention_dwt_attn"):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.models = []
        
        print(f"⚙️ Chargement de l'ensemble Mashup Wavelet Transformer Attention ({prefix}) - {num_folds} folds")
        for fold in range(num_folds):
            model_path = os.path.join(models_dir, f"{prefix}_fold_{fold}.pth")
            if not os.path.exists(model_path):
                print(f"⚠️ Poids introuvables ignorés : {model_path}")
                continue
            
            # Instanciation de l'architecture basée Transformer
            model = build_model("siamese_mashup_wavelet_attn")
            state_dict = torch.load(model_path, map_location=self.device)
            
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
                
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            self.models.append(model)
            
        self.fade_transform = SpatialMedialFadeInference(size=32, fade_start=22)
        self.tight_crop = transforms.CenterCrop(24)
        
        # Transforms TTA identiques au script d'évaluation
        self.tta_transforms = [
            lambda x: x,
            transforms.RandomRotation(degrees=(3, 3)),
            transforms.RandomRotation(degrees=(-3, -3)),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
        ]

    def _prepare_tensors(self, siamese_data):
        img_L_raw = siamese_data['patch_L'][0:1].clone()
        img_R_raw = siamese_data['patch_R'][0:1].clone()
        img_G_raw = siamese_data['patch_Global'][0:1].clone()
        
        # 1. Resize Global à 48x48
        if img_G_raw.shape[-1] != 48:
            img_G_raw = transforms.functional.resize(img_G_raw, [48, 48], antialias=True)
            
        # 2. Expansion sur 3 canaux
        img_L = img_L_raw.expand(3, -1, -1)
        img_R = img_R_raw.expand(3, -1, -1)
        img_G = img_G_raw.expand(3, -1, -1)
        
        # 3. Génération des sous-vues
        img_G_flip = torch.flip(img_G, dims=[1])
        img_L_fade = self.fade_transform(img_L)
        img_R_fade = self.fade_transform(img_R)
        img_L_tight = self.tight_crop(img_L)
        img_R_tight = self.tight_crop(img_R)
        
        # 4. Normalisation ImageNet
        mean_cpu = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std_cpu  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        def norm(t):
            t_norm = (t - mean_cpu) / std_cpu
            return t_norm.unsqueeze(0).to(self.device)
            
        return (
            norm(img_G), norm(img_G_flip),
            norm(img_L_fade), norm(img_R_fade),
            norm(img_L_tight), norm(img_R_tight),
            norm(img_L), norm(img_R)
        )

    def predict(self, siamese_data, use_tta=False):
        if not all(k in siamese_data for k in ('patch_L', 'patch_R', 'patch_Global')):
            return 0.5
            
        if not self.models:
            return 0.5

        tensors = self._prepare_tensors(siamese_data)
        
        fold_probs = []
        with torch.no_grad():
            for model in self.models:
                if use_tta:
                    tta_outputs = []
                    for t_func in self.tta_transforms:
                        transformed_tensors = tuple(t_func(t) for t in tensors)
                        logits = model(*transformed_tensors)
                        tta_outputs.append(logits)
                    outputs = torch.stack(tta_outputs, dim=0).mean(dim=0)
                else:
                    outputs = model(*tensors)
                    
                prob = torch.sigmoid(outputs).item()
                fold_probs.append(prob)

        return float(np.mean(fold_probs)) if fold_probs else 0.5

# ==============================================================================
# 3. ENSEMBLE DEUX PHASES (MeanAgg, 6 Branches, Masques inclus)
# ==============================================================================
import ptwt
from .striatum_preproc import init_preproc

def _dwt(x, wave='db1'):
    LL, (LH, HL, HH) = ptwt.wavedec2(x, wavelet=wave, level=1)
    return torch.cat([LL, LH, HL, HH], dim=1)
 
class TwoPhaseMeanEnsemble:
    def __init__(self, models_dir, num_folds=5, device=None, prefix="twophase"):
        self.device = torch.device(device if device else
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"⚙️ Chargement de l'ensemble Two-Phase (MeanAgg) - {num_folds} folds")
        self.preproc = init_preproc(models_dir)
        n_scalars = len(self.preproc.feature_cols)
 
        self.models = []
        for fold in range(num_folds):
            p = os.path.join(models_dir, f"{prefix}_fold{fold}.pth")
            if not os.path.exists(p):
                print(f"⚠️ Poids introuvables ignorés : {p}")
                continue
            model = build_model("twophase_mean", n_scalars=n_scalars)
            sd = torch.load(p, map_location=self.device)
            if "state_dict" in sd:
                sd = sd["state_dict"]
            missing, _ = model.load_state_dict(sd, strict=False)
            if missing:
                print(f"  ⚠️ fold {fold} : {len(missing)} clés manquantes (ex. {missing[:2]})")
            model.to(self.device).eval()
            self.models.append(model)
 
        if not self.models:
            raise FileNotFoundError(f"Aucun poids Two-Phase dans {models_dir}")
        print(f"  ✅ {len(self.models)} folds | {n_scalars} scalaires")
 
        self.fade = SpatialMedialFadeInference(32, 22)
        self.tight_crop = transforms.CenterCrop(24)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
 
    def _norm_mask(self, m):
        """Comme _load_masks à l'entraînement : uint8 ou 0-255 -> [0,1]."""
        m = m.clone().float().to(self.device)
        if m.max() > 1.5:
            m = m / 255.0
        return m.clamp(0, 1)
 
    def _prepare_batch(self, d):
        L = d['patch_L'][0:1].clone().to(self.device).expand(3, -1, -1)
        R = d['patch_R'][0:1].clone().to(self.device).expand(3, -1, -1)
        G = d['patch_Global'][0:1].clone().to(self.device)
        if G.shape[-1] != 48:
            G = transforms.functional.resize(G, [48, 48], antialias=True)
        G = G.expand(3, -1, -1)
 
        Gn = (G - self.mean) / self.std
        Ln = (L - self.mean) / self.std
        Rn = (R - self.mean) / self.std
        hL, tL = self._norm_mask(d['mask_hL']), self._norm_mask(d['mask_tL'])
        hR, tR = self._norm_mask(d['mask_hR']), self._norm_mask(d['mask_tR'])
 
        b = {}
        b['G'] = Gn.unsqueeze(0)
        b['G_flip'] = torch.flip(Gn, dims=[1]).unsqueeze(0)
        lf, rf = self.fade(Ln), self.fade(Rn)
        b['L_fade'] = torch.cat([lf, hL, tL], 0).unsqueeze(0)
        b['R_fade'] = torch.cat([rf, hR, tR], 0).unsqueeze(0)
        lt, rt = self.tight_crop(Ln), self.tight_crop(Rn)
        b['L_tight'] = torch.cat([lt, self.tight_crop(hL), self.tight_crop(tL)], 0).unsqueeze(0)
        b['R_tight'] = torch.cat([rt, self.tight_crop(hR), self.tight_crop(tR)], 0).unsqueeze(0)
        b['L_dwt'] = _dwt(Ln.unsqueeze(0))
        b['R_dwt'] = _dwt(Rn.unsqueeze(0))
        b['scalars'] = d['scalars'].unsqueeze(0).float().to(self.device)
        b['spine_L'] = d['spine_L'].unsqueeze(0).float().to(self.device)
        b['spine_R'] = d['spine_R'].unsqueeze(0).float().to(self.device)
        return b
 
    def predict(self, data):
        required = ['patch_L', 'patch_R', 'patch_Global', 'mask_hL', 'mask_tL',
                    'mask_hR', 'mask_tR', 'spine_L', 'spine_R', 'scalars']
        missing = [k for k in required if k not in data]
        if missing:
            print(f"⚠️ Clés manquantes pour Two-Phase : {missing}")
            return 0.5
        batch = self._prepare_batch(data)
        with torch.no_grad():
            probs = [torch.sigmoid(m(batch)).item() for m in self.models]
        return float(np.mean(probs))
    
# ==============================================================================
# CLASSE D'INFÉRENCE : MASHUP INTERNAL ENSEMBLE (LATE FUSION)
# ==============================================================================
class SiameseMashupInternalEnsemble:
    # 🔑 Ajout du paramètre use_dwt (False par défaut pour rétrocompatibilité)
    def __init__(self, models_dir, n_exp=20, n_folds=5, device=None, use_dwt=False, prefix='mashup_ensemble'):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.use_dwt = use_dwt
        self.models = []
        
        print(f"⚙️ Chargement de l'ensemble Siamese Internal - {n_exp} exps × {n_folds} folds | DWT: {self.use_dwt}")
        
        loaded_count = 0
        for exp in range(n_exp):
            for fold in range(n_folds):
                model_path = os.path.join(models_dir, f"{prefix}_exp_{exp}_fold_{fold}.pth")
                if not os.path.exists(model_path): 
                    continue
                
                # 🔑 On passe l'argument use_dwt au builder
                model = build_model("siamese_internal", use_dwt=self.use_dwt) 
                state_dict = torch.load(model_path, map_location=self.device)
                if "state_dict" in state_dict: state_dict = state_dict["state_dict"]
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.models.append(model)
                loaded_count += 1
                
        print(f"✅ {loaded_count} modèles chargés avec succès en VRAM.")
        
        # Outils de preprocessing
        self.fade_transform = SpatialMedialFadeInference(size=32, fade_start=22)
        self.tight_crop = transforms.CenterCrop(24)

    def _prepare_tensors(self, siamese_data):
        img_L = siamese_data['patch_L'][0:1].clone().expand(3, -1, -1)
        img_R = siamese_data['patch_R'][0:1].clone().expand(3, -1, -1)
        img_G = siamese_data['patch_Global'][0:1].clone()
        
        if img_G.shape[-1] != 48: 
            img_G = transforms.functional.resize(img_G, [48, 48], antialias=True)
        img_G = img_G.expand(3, -1, -1)
        img_G_flip = torch.flip(img_G, dims=[1])
        
        img_L_fade = self.fade_transform(img_L)
        img_R_fade = self.fade_transform(img_R)
        img_L_tight = self.tight_crop(img_L)
        img_R_tight = self.tight_crop(img_R)
        
        mean_cpu = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std_cpu  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        def norm(t): 
            t_norm = (t - mean_cpu) / std_cpu
            return t_norm.unsqueeze(0).to(self.device)
        
        # 🔑 Si DWT est activé, on retourne également les images brutes
        if self.use_dwt:
            return (norm(img_G), norm(img_G_flip), norm(img_L_fade), norm(img_R_fade), 
                    norm(img_L_tight), norm(img_R_tight), norm(img_L), norm(img_R))
        else:
            return (norm(img_G), norm(img_G_flip), norm(img_L_fade), norm(img_R_fade), 
                    norm(img_L_tight), norm(img_R_tight))

    def _apply_recipe(self, p_array):
        # # ... Reste inchangé ...
        # EPS = 1e-7
        # T_SCALING = 0.884
        # CLIP_VAL  = 0.002
        
        # if len(p_array) == 0:
        #     return 0.5
            
        # p_mean = np.mean(p_array)
        # p_clip_temp = np.clip(p_mean, EPS, 1.0 - EPS)
        # p_logit = np.log(p_clip_temp / (1.0 - p_clip_temp))
        # p_calibrated = 1.0 / (1.0 + np.exp(-p_logit / T_SCALING))
        # p_final = np.clip(p_calibrated, CLIP_VAL, 1.0 - CLIP_VAL)
        
        p_final = np.mean(p_array)

        return float(p_final)

    def predict(self, siamese_data):
        if not all(k in siamese_data for k in ('patch_L', 'patch_R', 'patch_Global')): 
            return 0.5 
            
        tensors = self._prepare_tensors(siamese_data)
        
        fold_probs = []
        with torch.no_grad():
            for model in self.models:
                # 🔑 Unpacking dynamique sécurisé (fonctionne si renvoie 4 éléments sans DWT ou 5 avec)
                outputs = model(*tensors)
                mean_logit = outputs[0]
                
                prob = torch.sigmoid(mean_logit).item()
                fold_probs.append(prob)
                
        return self._apply_recipe(fold_probs)