import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Imports de ta production ---
from src.preprocessing import prepare_all_inputs
from src.inference_2d_avg import init_preproc

# ==========================================
# 1. CONFIGURATION
# ==========================================
UID = "05xt54fn"
NIFTI_PATH = f"/data2/iceberg/dat_parkinson/niftis/{UID}.nii.gz"
WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights')

TRAIN_MASK_BUNDLE = "/data2/iceberg/dat_parkinson/dataset_siamese_masks/all_masks.pt"
TRAIN_SPINE_CSV = "/data/HOME/sansel/dat-park/notebooks/spine_profiles_real.csv"

def compare_tensors(name, t_prod, t_train):
    diff = np.abs(t_prod - t_train)
    mae = diff.mean()
    max_err = diff.max()
    status = "✅ PARFAIT" if max_err < 1e-5 else "⚠️ ECART"
    print(f"{status} | {name:<15} | MAE: {mae:.6f} | Erreur Max: {max_err:.6f}")
    return diff

def main():
    print(f"🔍 INVESTIGATION (SPINES & MASQUES) - PATIENT : {UID}\n" + "="*65)
    
    init_preproc(WEIGHTS_DIR)
    
    print("⏳ Extraction à la volée (Production)...")
    inputs_prod = prepare_all_inputs(NIFTI_PATH)
    tp_prod = inputs_prod['twophase_data']
    
    # 1. Spines Production
    prod_spine_L = tp_prod['spine_L'].numpy() # Shape (24, 3)
    prod_spine_R = tp_prod['spine_R'].numpy()
    
    # 2. Masques Production
    prod_mask_hL = tp_prod['mask_hL'][0].numpy()
    prod_mask_tL = tp_prod['mask_tL'][0].numpy()

    # --- C. CHARGEMENT ENTRAÎNEMENT ---
    print("⏳ Chargement des Masques et Spines d'entraînement...")
    
    # Masques
    train_masks = torch.load(TRAIN_MASK_BUNDLE)
    train_mask_uid = train_masks[UID].numpy().astype(np.float32) # Shape (4, 32, 32)
    train_mask_hL = train_mask_uid[0]
    train_mask_tL = train_mask_uid[1]
    
    # Spines
    spine_csv = pd.read_csv(TRAIN_SPINE_CSV)
    spine_row = spine_csv[spine_csv['uid'] == UID].iloc[0]
    
    cl = [f"L_spine_w_{i:02d}" for i in range(24)]
    cr = [f"R_spine_w_{i:02d}" for i in range(24)]
    
    vl = spine_row[cl].values.astype(np.float32)
    vr = spine_row[cr].values.astype(np.float32)
    u = np.linspace(-1, 1, 24).astype(np.float32)
    
    # Reproduction du "pack" de l'entrainement: [intensite, validité, coord_u]
    train_spine_L = np.stack([np.nan_to_num(vl, nan=0.), np.isfinite(vl).astype(np.float32), u], axis=-1)
    train_spine_R = np.stack([np.nan_to_num(vr, nan=0.), np.isfinite(vr).astype(np.float32), u], axis=-1)

    # --- D. COMPARAISON MATHÉMATIQUE ---
    print("\n" + "="*65 + "\nCOMPARAISON DES ENTRÉES\n" + "="*65)
    compare_tensors("Masque Tête L", prod_mask_hL, train_mask_hL)
    compare_tensors("Masque Queue L", prod_mask_tL, train_mask_tL)
    compare_tensors("Spine_L (Total)", prod_spine_L, train_spine_L)
    compare_tensors("Spine_R (Total)", prod_spine_R, train_spine_R)
    
    # --- E. TRACÉ DES SPINES ---
    print("\n📸 Génération du tracé 'comparison_spines.png'...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Spines 1D: Entraînement vs Production - {UID}", fontweight='bold')
    
    # Tracé Spine L (On trace la dimension 0 = l'intensité normalisée)
    axes[0].plot(u, train_spine_L[:, 0], 'o-', label="Train", linewidth=2)
    axes[0].plot(u, prod_spine_L[:, 0], 'x--', label="Prod", linewidth=2)
    axes[0].set_title("Spine Gauche (L)")
    axes[0].legend()
    
    # Tracé Spine R
    axes[1].plot(u, train_spine_R[:, 0], 'o-', label="Train", linewidth=2, color='green')
    axes[1].plot(u, prod_spine_R[:, 0], 'x--', label="Prod", linewidth=2, color='red')
    axes[1].set_title("Spine Droite (R)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("comparison_spines.png")
    print("✅ Terminé. Si un écart apparaît, le graphique vous montrera exactement à quel endroit de la courbe il se situe.")

if __name__ == '__main__':
    main()