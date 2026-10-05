import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

import sys
# On ajoute le dossier parent pour que Python trouve 'src'
sys.path.append(os.path.dirname(__file__))
from src.preprocessing import prepare_all_inputs

# ==========================================
# CONFIGURATION
# ==========================================
CSV_PATH = '/data2/iceberg/dat_parkinson/dataset_siamese_patches/folds_siamese.csv'
NIFTI_DIR = Path('/data2/iceberg/dat_parkinson/niftis')
OUTPUT_DIR = Path(os.path.join(os.path.dirname(__file__), 'debug_orientations'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NB_PATIENTS_TO_CHECK = 3

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.squeeze().cpu().numpy()
    return tensor

# ==========================================
# CORRECTIONS VISUELLES (AFFICHAGE UNIQUEMENT)
# ==========================================
# k=1 : Rotation 90° Anti-horaire | k=-1 : 90° Horaire | k=2 : 180°
def view_fix_3d_ax(img): return np.rot90(img, k=1)   # Annule la rotation 90° droite
def view_fix_3d_cor(img): return np.rot90(img, k=1)  # Annule la rotation 90° droite
def view_fix_3d_sag(img): return np.rot90(img, k=1)  # Tentative de redressement (nez en bas)

def view_fix_2d_ax(img): return np.rot90(img, k=2)   # 180° pour ramener le nez en haut
def view_fix_2d_cor(img): return img                 # Déjà OK selon vous
def view_fix_2d_sag(img): return img                 # Nez à droite, mais à l'endroit a priori

def view_fix_siam(img): return np.rot90(img, k=1)    # Annule la rotation 90° droite

# ==========================================

def plot_patient_raw_data(inputs, uid):
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    fig.suptitle(f"Patient {uid} - Orientations Redressées (Affichage Uniquement)", fontsize=16, fontweight='bold')
    
    # ---------------------------------------------------------
    # LIGNE 1 : Volume 3D (MedicalNet)
    # ---------------------------------------------------------
    vol = to_numpy(inputs['volume_3d_numpy'])
    z_mid, y_mid, x_mid = vol.shape[0]//2, vol.shape[1]//2, vol.shape[2]//2
    
    axes[0, 0].imshow(view_fix_3d_ax(vol[z_mid, :, :]), cmap='gray')
    axes[0, 0].set_title(f"3D: Coupe Axiale\n(Affichage pivoté -90°)")
    
    axes[0, 1].imshow(view_fix_3d_cor(vol[:, y_mid, :]), cmap='gray')
    axes[0, 1].set_title("3D: Coupe Coronale\n(Affichage pivoté -90°)")
    
    axes[0, 2].imshow(view_fix_3d_sag(vol[:, :, x_mid]), cmap='gray')
    axes[0, 2].set_title("3D: Coupe Sagittale\n(Affichage redressé)")
    
    # ---------------------------------------------------------
    # LIGNE 2 : Slices 2D (SagCorAx / ResNet2D)
    # ---------------------------------------------------------
    sagcorax = inputs['slices_2d_sagcorax']
    ax = sagcorax['slices_2d_axial'][len(sagcorax['slices_2d_axial'])//2]['image']
    cor = sagcorax['slices_2d_coronal'][len(sagcorax['slices_2d_coronal'])//2]['image']
    sag = sagcorax['slices_2d_sagittal'][len(sagcorax['slices_2d_sagittal'])//2]['image']
    
    axes[1, 0].imshow(view_fix_2d_ax(ax), cmap='gray')
    axes[1, 0].set_title("2D: Axial\n(Affichage pivoté 180°)")
    
    axes[1, 1].imshow(view_fix_2d_cor(cor), cmap='gray')
    axes[1, 1].set_title("2D: Coronal\n(Brut - OK)")
    
    axes[1, 2].imshow(view_fix_2d_sag(sag), cmap='gray')
    axes[1, 2].set_title("2D: Sagittal\n(Brut - OK)")
    
    # ---------------------------------------------------------
    # LIGNE 3 : Siamois L & R (Canal 0 = MIP Ax, Canal 1 = MIP Cor)
    # ---------------------------------------------------------
    pL = to_numpy(inputs['siamese_patches']['patch_L'])
    pR = to_numpy(inputs['siamese_patches']['patch_R'])
    
    axes[2, 0].imshow(view_fix_siam(pL[0]), cmap='gray')
    axes[2, 0].set_title("Siamois Left: MIP Axial\n(Affichage pivoté -90°)")
    
    axes[2, 1].imshow(view_fix_siam(pL[1]), cmap='gray')
    axes[2, 1].set_title("Siamois Left: MIP Coronal\n(Affichage pivoté -90°)")
    
    axes[2, 2].imshow(view_fix_siam(pR[0]), cmap='gray')
    axes[2, 2].set_title("Siamois Right: MIP Axial\n(Affichage pivoté -90°)")
    
    # ---------------------------------------------------------
    # LIGNE 4 : Siamois Global + Siamois R (MIP Coronal)
    # ---------------------------------------------------------
    pG = to_numpy(inputs['siamese_mashup_data']['patch_Global'])
    
    axes[3, 0].imshow(view_fix_siam(pG[0]), cmap='gray')
    axes[3, 0].set_title("Siamois Global: MIP Axial\n(Affichage pivoté -90°)")
    
    axes[3, 1].imshow(view_fix_siam(pG[1]), cmap='gray')
    axes[3, 1].set_title("Siamois Global: MIP Coronal\n(Affichage pivoté -90°)")
    
    axes[3, 2].imshow(view_fix_siam(pR[1]), cmap='gray')
    axes[3, 2].set_title("Siamois Right: MIP Coronal\n(Affichage pivoté -90°)")

    for ax_row in axes:
        for ax_item in ax_row: ax_item.axis('off')

    plt.tight_layout()
    save_path = OUTPUT_DIR / f"{uid}_raw_orientations_fixed.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvegardé : {save_path}")

def main():
    df = pd.read_csv(CSV_PATH)
    processed = 0
    print(f"Génération des planches redressées dans {OUTPUT_DIR}...")
    for _, row in df.iterrows():
        uid = row['uid']
        nifti_path = NIFTI_DIR / f"{uid}.nii.gz"
        if not nifti_path.exists(): continue
        inputs = prepare_all_inputs(str(nifti_path))
        plot_patient_raw_data(inputs, uid)
        processed += 1
        if processed >= NB_PATIENTS_TO_CHECK: break

if __name__ == '__main__':
    main()