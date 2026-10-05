import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from scipy.ndimage import rotate as rotate3d
from pathlib import Path

import sys
sys.path.append(os.path.dirname(__file__))
from src.preprocessing import prepare_all_inputs

# ==========================================
# CONFIGURATION
# ==========================================
CSV_PATH = '/data2/iceberg/dat_parkinson/dataset_siamese_patches/folds_siamese.csv'
NIFTI_DIR = Path('/data2/iceberg/dat_parkinson/niftis')
OUTPUT_DIR = Path(os.path.join(os.path.dirname(__file__), 'debug_tta'))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NB_PATIENTS_TO_CHECK = 3

# ==========================================
# UTILITAIRES (Fix visuel & Augmentations)
# ==========================================
def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor): return tensor.squeeze().cpu().numpy()
    return tensor

def apply_tf(tensor, func, *args, **kwargs):
    return func(tensor.unsqueeze(0), *args, **kwargs).squeeze(0)

def aug_rotate_3d(volume, angle):
    is_tensor = isinstance(volume, torch.Tensor)
    vol_np = volume.numpy() if is_tensor else volume
    rotated = rotate3d(vol_np, angle, axes=(-1, -2), reshape=False, order=1, cval=0.0)
    rotated = np.ascontiguousarray(rotated)
    return torch.from_numpy(rotated) if is_tensor else rotated

def aug_rot_cv2(img_2d, angle):
    h, w = img_2d.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img_2d, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

# Redressements visuels (issus de la Partie 1)
def view_fix_siam(img): return np.rot90(img, k=1)
def view_fix_2d_ax(img): return np.rot90(img, k=2)
def view_fix_3d_ax(img): return np.rot90(img, k=1)

# ==========================================
# GÉNÉRATION DES COMPARAISONS
# ==========================================
def plot_tta_comparisons(inputs, uid):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f"Patient {uid} - Effet Réel des TTA sur l'Anatomie", fontsize=16, fontweight='bold')
    
    # ---------------------------------------------------------
    # 1. SIAMOIS GLOBAL (MIP Axial) - Tensor
    # ---------------------------------------------------------
    pG = inputs['siamese_mashup_data']['patch_Global'][0] # Canal Axial
    
    axes[0, 0].imshow(view_fix_siam(to_numpy(pG)), cmap='gray')
    axes[0, 0].set_title("1. Siamois: BASELINE")
    
    axes[0, 1].imshow(view_fix_siam(to_numpy(apply_tf(pG, TF.hflip))), cmap='gray')
    axes[0, 1].set_title("2. Siamois: H-FLIP\n(Inverser axe horizontal brut)")
    
    axes[0, 2].imshow(view_fix_siam(to_numpy(apply_tf(pG, TF.vflip))), cmap='gray')
    axes[0, 2].set_title("3. Siamois: V-FLIP\n(Inverser axe vertical brut)")
    
    axes[0, 3].imshow(view_fix_siam(to_numpy(apply_tf(pG, TF.rotate, 15, fill=0))), cmap='gray')
    axes[0, 3].set_title("4. Siamois: ROTATION (+15°)\n(Exagéré pour test)")

    # ---------------------------------------------------------
    # 2. RESNET 2D (Axial) - Numpy Array
    # ---------------------------------------------------------
    slices_2d = inputs['slices_2d']
    mid_slice = slices_2d[len(slices_2d) // 2]['image']
    
    axes[1, 0].imshow(view_fix_2d_ax(mid_slice), cmap='gray')
    axes[1, 0].set_title("1. 2D Ax: BASELINE")
    
    axes[1, 1].imshow(view_fix_2d_ax(cv2.flip(mid_slice, 1)), cmap='gray') # cv2 flip=1 -> horizontal
    axes[1, 1].set_title("2. 2D Ax: H-FLIP")
    
    axes[1, 2].imshow(view_fix_2d_ax(cv2.flip(mid_slice, 0)), cmap='gray') # cv2 flip=0 -> vertical
    axes[1, 2].set_title("3. 2D Ax: V-FLIP")
    
    axes[1, 3].imshow(view_fix_2d_ax(aug_rot_cv2(mid_slice, 15)), cmap='gray')
    axes[1, 3].set_title("4. 2D Ax: ROTATION (+15°)")

    # ---------------------------------------------------------
    # 3. MEDICALNET 3D (Coupe Axiale extraite) - Numpy Array
    # ---------------------------------------------------------
    vol = to_numpy(inputs['volume_3d_numpy'])
    z_mid = vol.shape[0] // 2
    
    axes[2, 0].imshow(view_fix_3d_ax(vol[z_mid, :, :]), cmap='gray')
    axes[2, 0].set_title("1. 3D Ax: BASELINE")
    
    # Flip X (axis=-1)
    vol_flipX = np.flip(vol, axis=-1)
    axes[2, 1].imshow(view_fix_3d_ax(vol_flipX[z_mid, :, :]), cmap='gray')
    axes[2, 1].set_title("2. 3D Ax: FLIP Axe X\n(axis=-1)")
    
    # Flip Y (axis=-2)
    vol_flipY = np.flip(vol, axis=-2)
    axes[2, 2].imshow(view_fix_3d_ax(vol_flipY[z_mid, :, :]), cmap='gray')
    axes[2, 2].set_title("3. 3D Ax: FLIP Axe Y\n(axis=-2)")
    
    # Rot 3D 
    vol_rot = aug_rotate_3d(vol, 15)
    axes[2, 3].imshow(view_fix_3d_ax(vol_rot[z_mid, :, :]), cmap='gray')
    axes[2, 3].set_title("4. 3D Ax: ROT Z (+15°)")

    # Nettoyage
    for ax_row in axes:
        for ax_item in ax_row:
            ax_item.axis('off')

    plt.tight_layout()
    save_path = OUTPUT_DIR / f"{uid}_tta_anatomical_check.png"
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✅ Sauvegardé : {save_path}")

def main():
    df = pd.read_csv(CSV_PATH)
    processed = 0
    print(f"Génération des planches TTA dans {OUTPUT_DIR}...")
    for _, row in df.iterrows():
        uid = row['uid']
        nifti_path = NIFTI_DIR / f"{uid}.nii.gz"
        if not nifti_path.exists(): continue
        
        inputs = prepare_all_inputs(str(nifti_path))
        plot_tta_comparisons(inputs, uid)
        
        processed += 1
        if processed >= NB_PATIENTS_TO_CHECK: break

if __name__ == '__main__':
    main()