"""
preprocessing.py
=================
Point d'entrée UNIQUE pour le prétraitement des .nii.gz dans le pipeline
d'inférence final (ensemble/stacking Volume 3D + Image 2D + Features + Siamois).
"""

import numpy as np
import nibabel as nib
import torch
import logging
import warnings
from scipy import ndimage
from scipy.stats import kurtosis, skew
from skimage.morphology import binary_closing, disk, ball
from skimage.measure import label, regionprops
from skimage.feature import peak_local_max
import monai.transforms as mt
from monai.transforms import MapTransform
import os
import joblib
import pandas as pd
import json

# Imports Pyradiomics
import SimpleITK as sitk
from radiomics import featureextractor

# Silence les logs très bavards de Pyradiomics
warnings.filterwarnings("ignore")
logging.getLogger("radiomics").setLevel(logging.ERROR)

# ==========================================
# CONFIGURATION PARTAGÉE
# ==========================================
TARGET_SHAPE   = (128, 128, 128)
TARGET_SPACING = (2.0, 2.0, 2.0)

NUM_SLICES          = 5
STEP_Z              = 2
MIN_MAX_INTENSITY   = 40
MIN_MEAN_INTENSITY  = 1.0

# TOP 59 Features Optimales (Hybride + Pyradiomics)
FEATURE_NAMES = [
    'pyr_firstorder_InterquartileRange_R', 'pyr_firstorder_Uniformity_L', 'pyr_glrlm_ShortRunLowGrayLevelEmphasis_R',
    'pyr_glcm_MaximumProbability_R', 'pyr_glrlm_LongRunHighGrayLevelEmphasis_R', 'pyr_firstorder_Kurtosis_R',
    'pyr_shape_Sphericity_R', 'pyr_glrlm_LongRunHighGrayLevelEmphasis_L', 'pyr_firstorder_Kurtosis_L',
    'pyr_firstorder_Variance_L', 'pyr_glcm_ClusterTendency_R', 'pyr_glrlm_RunPercentage_L',
    'pyr_firstorder_10Percentile_L', 'pyr_shape_SurfaceVolumeRatio_R', 'r50_mm',
    'pyr_glcm_Contrast_R', 'pyr_firstorder_10Percentile_R', 'pyr_glcm_DifferenceEntropy_L',
    'fwhm', 'pyr_shape_Maximum3DDiameter_L', 'pyr_glcm_Idn_R',
    'pyr_glcm_Imc2_Asym', 'skewness', 'pyr_glcm_Idn_L',
    'pyr_shape_LeastAxisLength_R', 'pyr_shape_MinorAxisLength_R', 'pyr_shape_Maximum2DDiameterColumn_R',
    'pyr_shape_MajorAxisLength_R', 'pyr_shape_Maximum2DDiameterRow_R', 'pyr_shape_Maximum2DDiameterSlice_R',
    'pyr_glrlm_RunEntropy_Asym', 'pyr_glcm_SumEntropy_Asym', 'pyr_shape_MeshVolume_Asym',
    'pyr_glrlm_RunLengthNonUniformity_Asym', 'pyr_shape_Maximum2DDiameterRow_L', 'pyr_glcm_ClusterProminence_Asym',
    'pyr_glcm_Correlation_Asym', 'pyr_shape_Maximum2DDiameterColumn_L', 'pyr_glrlm_RunEntropy_R',
    'pyr_shape_SurfaceVolumeRatio_Asym', 'pyr_glrlm_LongRunHighGrayLevelEmphasis_Asym', 'pyr_glrlm_ShortRunEmphasis_Asym',
    'pyr_glrlm_RunEntropy_L', 'entropy', 'pyr_shape_SurfaceArea_Asym',
    'pyr_firstorder_RobustMeanAbsoluteDeviation_Asym', 'pyr_firstorder_Kurtosis_Asym', 'pyr_glcm_Imc2_R',
    'pyr_glcm_DifferenceEntropy_Asym', 'ap_gradient', 'pyr_glrlm_GrayLevelNonUniformity_Asym',
    'pyr_glcm_Imc1_Asym', 'pyr_glcm_Imc2_L', 'pyr_glcm_ClusterProminence_L',
    'pyr_firstorder_Minimum_R', 'pyr_firstorder_Minimum_L', 'pyr_firstorder_Minimum_Asym',
    'pyr_firstorder_Uniformity_Asym', 'pyr_glrlm_LongRunEmphasis_Asym', 'pyr_shape_Sphericity_Asym',
]

# # --- 1. CHARGEMENT GLOBAL DES ARTEFACTS DU NOTEBOOK ---
# WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'weights')
# try:
#     SCALER = joblib.load(os.path.join(WEIGHTS_DIR, 'scaler_production.joblib'))
#     MEDIANS = pd.read_json(os.path.join(WEIGHTS_DIR, 'medians_production.json'), typ='series')
#     with open(os.path.join(WEIGHTS_DIR, 'spine_refs.json'), 'r') as f:
#         SPINE_REFS = json.load(f)
# except Exception as e:
#     print(f"⚠️ Artefacts de production introuvables : {e}")
#     SCALER, MEDIANS, SPINE_REFS = None, None, None

# --- 2. REPRODUCTION DE LA LOGIQUE DU NOTEBOOK ---
def apply_notebook_preprocessing(raw_feats_dict):
    if SCALER is None:
        return np.zeros(14, dtype=np.float32) # Fallback sécurité
        
    feats = raw_feats_dict.copy()
    
    # A. Normalisation des centroïdes [-1, 1]
    for side in ['L_', 'R_']:
        for region in ['head_', 'tail_']:
            for axis, dim in zip(['x', 'y'], [32, 32]): 
                col = f'{side}{region}centroid_{axis}'
                if col in feats and not np.isnan(feats[col]):
                    feats[col] = (feats[col] - (dim / 2.0)) / (dim / 2.0)

    # B. Log1p sur les queues lourdes
    heavy_tail_keywords = ['area', 'sum_int', 'complexity']
    cols_to_log = [c for c in feats.keys() if any(k in c for k in heavy_tail_keywords) 
                   and not any(exc in c for exc in ['ratio', 'frac', 'AI', 'detected'])]
                   
    for col in cols_to_log:
        val = feats[col]
        # On log1p(max(0, val)) pour sécuriser l'inférence contre d'éventuelles valeurs aberrantes négatives
        feats[f'{col}_log'] = np.log1p(max(0, val)) if not np.isnan(val) else np.nan
        del feats[col]

    # C. Imputation & Standardisation
    df_new = pd.DataFrame([feats])
    feature_cols = list(MEDIANS.index) # Garantit l'ordre strict du scaler
    
    # Sécurité si une feature n'a pas été extraite du tout
    for c in feature_cols:
        if c not in df_new.columns:
            df_new[c] = np.nan
            
    X_scaled = SCALER.transform(df_new[feature_cols].fillna(MEDIANS))
    
    # D. Remplace les NaN finaux par 0 (la moyenne d'une variable standardisée)
    X_final = np.nan_to_num(X_scaled, nan=0., posinf=0., neginf=0.).astype(np.float32)
    
    return X_final[0] # Renvoie le vecteur 1D

# ==========================================
# 1. TRANSFORM PARTAGÉ (MONAI)
# ==========================================
class _StriatumCenterCropd(MapTransform):
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

            half = self.spatial_size // 2
            start = center - half
            end = center + half
            pb = np.maximum(0, -start)
            pa = np.maximum(0, end - np.array(vol.shape))

            if np.any(pb > 0) or np.any(pa > 0):
                vol = np.pad(vol, list(zip(pb, pa)), mode='constant', constant_values=vol.min())
                mask = np.pad(mask, list(zip(pb, pa)), mode='constant')
                start += pb
                end += pb

            cropped = vol[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
            cropped_mask = mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

            d[key] = torch.tensor(cropped, dtype=torch.float32).unsqueeze(0)
            d[f"{key}_bright_mask"] = cropped_mask
            d[f"{key}_center"] = center
        return d


def _build_transforms(target_shape, target_spacing):
    return mt.Compose([
        mt.LoadImaged(keys=["image"]),
        mt.EnsureChannelFirstd(keys=["image"]),
        mt.Spacingd(keys=["image"], pixdim=target_spacing, mode="bilinear"),
        _StriatumCenterCropd(keys=["image"], spatial_size=target_shape),
        mt.ScaleIntensityd(keys=["image"]),
    ])

_DEFAULT_TRANSFORMS = _build_transforms(TARGET_SHAPE, TARGET_SPACING)

def preprocess_patient(nifti_path, target_shape=TARGET_SHAPE, target_spacing=TARGET_SPACING):
    if (target_shape, target_spacing) == (TARGET_SHAPE, TARGET_SPACING):
        transforms = _DEFAULT_TRANSFORMS
    else:
        transforms = _build_transforms(target_shape, target_spacing)

    data = transforms({"image": nifti_path})
    return {
        "volume": data["image"][0].numpy(),
        "bright_mask": data["image_bright_mask"],
        "center": data["image_center"],
        "tensor": data["image"], 
    }

# ==========================================
# 1.B. TRANSFORM PARTAGÉ (NUMPY PURE POUR MEDICALNET)
# ==========================================
def preprocess_patient_numpy(nifti_path, target_shape=TARGET_SHAPE, target_spacing=TARGET_SPACING):
    img = nib.load(nifti_path)
    vol = img.get_fdata(dtype=np.float32)
    sp = np.array(img.header.get_zooms()[:3], dtype=float)
    zoom = sp / np.array(target_spacing, dtype=float)
    res = ndimage.zoom(vol, zoom, order=1)

    p95 = float(np.percentile(res, 95))
    bright = res > p95
    try:
        center = np.array(ndimage.center_of_mass(bright), dtype=int)
    except Exception:
        center = np.array(res.shape) // 2

    half = np.array(target_shape) // 2
    s0, e0 = center - half, center + half
    bef = np.maximum(0, -s0)
    aft = np.maximum(0, e0 - np.array(res.shape))
    pad = list(zip(bef.tolist(), aft.tolist()))
    padded = np.pad(res, pad, mode="constant", constant_values=float(res.min()))
    s1 = s0 + bef; e1 = e0 + bef
    crop = padded[s1[0]:e1[0], s1[1]:e1[1], s1[2]:e1[2]]

    cmin, cmax = float(crop.min()), float(crop.max())
    sc = ((crop - cmin) / (cmax - cmin + 1e-8)).astype(np.float32)

    return sc


# ==========================================
# 2. BRANCHES 2D (Séquentielles, Corax, Sagcorax)
# ==========================================
def extract_2d_slices(volume, num_slices=NUM_SLICES, step_z=STEP_Z, min_max_intensity=MIN_MAX_INTENSITY, min_mean_intensity=MIN_MEAN_INTENSITY):
    sx, sy, _ = volume.shape
    x, y = np.ogrid[:sx, :sy]
    sigma = sx / 8.0
    center_weight = np.exp(-((x - sx//2)**2 + (y - sy//2)**2) / (2 * sigma**2))
    
    weighted_volume = volume * center_weight[:, :, np.newaxis]
    z_profile = weighted_volume.sum(axis=(0, 1))
    z_center = int(np.argmax(z_profile))

    half_num = num_slices // 2
    z_indices = [z_center + (i * step_z) for i in range(-half_num, half_num + 1)]

    slices = []
    for z in z_indices:
        z_safe = int(np.clip(z, 0, volume.shape[2] - 1))
        slice_2d = volume[:, :, z_safe].T
        slice_img = (slice_2d * 255).astype(np.uint8)

        if slice_img.max() < min_max_intensity or slice_img.mean() < min_mean_intensity:
            continue
        slices.append({"z_index": z_safe, "image": slice_img})

    return slices

def extract_2d_slices_coronal(volume, num_slices=NUM_SLICES, step_y=1, min_max_intensity=MIN_MAX_INTENSITY, min_mean_intensity=MIN_MEAN_INTENSITY):
    sx, _, sz = volume.shape
    x, z = np.ogrid[:sx, :sz]
    sigma = sx / 8.0
    center_weight = np.exp(-((x - sx//2)**2 + (z - sz//2)**2) / (2 * sigma**2))
    
    weighted_volume = volume * center_weight[:, np.newaxis, :]
    y_profile = weighted_volume.sum(axis=(0, 2))
    y_center = int(np.argmax(y_profile))

    half_num = num_slices // 2
    y_indices = [y_center + (i * step_y) for i in range(-half_num, half_num + 1)]

    slices = []
    for y in y_indices:
        y_safe = int(np.clip(y, 0, volume.shape[1] - 1))
        slice_2d = volume[:, y_safe, :].T
        slice_img = (slice_2d * 255).astype(np.uint8)
        slice_img_corrected = np.flip(slice_img, axis=(0, 1))

        if (slice_img_corrected.max() < min_max_intensity or slice_img_corrected.mean() < min_mean_intensity):
            continue

        slices.append({"y_index": y_safe, "image": slice_img_corrected})

    return slices

def extract_2d_slices_sagittal(volume, num_slices=NUM_SLICES, step_x=1, min_max_intensity=MIN_MAX_INTENSITY, min_mean_intensity=MIN_MEAN_INTENSITY):
    sx, sy, sz = volume.shape
    y, z = np.ogrid[:sy, :sz]
    sigma = sy / 8.0
    center_weight = np.exp(-((y - sy//2)**2 + (z - sz//2)**2) / (2 * sigma**2))
    
    weighted_volume = volume * center_weight[np.newaxis, :, :]
    x_profile = weighted_volume.sum(axis=(1, 2))
    x_center = int(np.argmax(x_profile))

    half_num = num_slices // 2
    x_indices = [x_center + (i * step_x) for i in range(-half_num, half_num + 1)]

    slices = []
    for x in x_indices:
        x_safe = int(np.clip(x, 0, sx - 1))
        slice_2d = volume[x_safe, :, :].T
        slice_img = (slice_2d * 255).astype(np.uint8)
        slice_img_corrected = slice_img[::-1, :]

        if (slice_img_corrected.max() < min_max_intensity or slice_img_corrected.mean() < min_mean_intensity):
            continue

        slices.append({"x_index": x_safe, "image": slice_img_corrected})

    return slices


def extract_siamese_patches(vol, crop_size=32, global_crop_size=64, stride=2):
    sx, sy, sz = vol.shape
    x, y, z = np.ogrid[:sx, :sy, :sz]
    cx, cy, cz = sx // 2, sy // 2, sz // 2
    sigma = sx / 6.0 
    dist_sq = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
    center_weight_3d = np.exp(-dist_sq / (2 * sigma**2))

    def get_best_striatum_pair(coords, intensity_map):
        if len(coords) < 2: return coords
        best_pair, best_score = None, -1
        ideal_dist = 21.0
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                p1, p2 = coords[i], coords[j]
                dist = np.linalg.norm(p1 - p2)
                dist_penalty = np.exp(-((dist - ideal_dist)**2) / (2 * 6.0**2))
                dy, dz = p1[1] - p2[1], p1[2] - p2[2]
                align_penalty = np.exp(-(dy**2 + dz**2) / (2 * 5.0**2))
                score = (intensity_map[tuple(p1)] + intensity_map[tuple(p2)]) * dist_penalty * align_penalty
                if score > best_score:
                    best_score, best_pair = score, [p1, p2]
        if best_pair is not None:
            final_dist = np.linalg.norm(best_pair[0] - best_pair[1])
            if 12 < final_dist < 32: return best_pair
        return [coords[0]]

    thresh = np.percentile(vol, 98)
    binary = vol > thresh
    binary_closed = binary_closing(binary, ball(2)) 
    vol_smooth = ndimage.gaussian_filter(vol, sigma=1.0)
    weighted_vol_smooth = vol_smooth * center_weight_3d
    coords_raw = peak_local_max(weighted_vol_smooth, min_distance=10, labels=binary_closed, num_peaks=10)
    coords = get_best_striatum_pair(coords_raw, weighted_vol_smooth)
    
    left_center, right_center = None, None
    angle_z_deg, angle_y_deg = 0.0, 0.0 
    
    if len(coords) == 2:
        x1, y1, z1 = coords[0]
        x2, y2, z2 = coords[1]
        if x1 > x2:
            x1, y1, z1, x2, y2, z2 = x2, y2, z2, x1, y1, z1
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
        angle_z_deg = np.clip(-np.degrees(np.arctan2(dy, dx)), -30, 30)
        angle_y_deg = np.clip(-np.degrees(np.arctan2(dz, dx)), -30, 30)
    elif len(coords) == 1:
        x, y, z = coords[0]
        if x < cx:
            left_center, right_center = (x, y, z), (cx + (cx - x), y, z) 
        else:
            right_center, left_center = (x, y, z), (cx - (x - cx), y, z)
    else:
        offset = 20
        left_center, right_center = (cx - offset, cy, cz), (cx + offset, cy, cz)

    vol_rot_z = ndimage.rotate(vol, angle_z_deg, axes=(0, 1), reshape=False, order=3)
    vol_rot_3d = ndimage.rotate(vol_rot_z, angle_y_deg, axes=(0, 2), reshape=False, order=3)
    
    bin_rot = vol_rot_3d > np.percentile(vol_rot_3d, 98)
    bin_rot_closed = binary_closing(bin_rot, ball(2))
    vol_rot_smooth = ndimage.gaussian_filter(vol_rot_3d, sigma=1.0)
    weighted_rot_smooth = vol_rot_smooth * center_weight_3d
    coords_rot_raw = peak_local_max(weighted_rot_smooth, min_distance=10, labels=bin_rot_closed, num_peaks=10)
    coords_rot = get_best_striatum_pair(coords_rot_raw, weighted_rot_smooth)
    
    if len(coords_rot) >= 2:
        x1, y1, z1 = coords_rot[0]
        x2, y2, z2 = coords_rot[1]
        if x1 < x2: left_center, right_center = (x1, y1, z1), (x2, y2, z2)
        else: left_center, right_center = (x2, y2, z2), (x1, y1, z1)
    elif len(coords_rot) == 1:
        x, y, z = coords_rot[0]
        if x < cx: left_center, right_center = (x, y, z), (cx + (cx - x), y, z)
        else: right_center, left_center = (x, y, z), (cx - (x - cx), y, z)
    else:
        offset = 20
        left_center, right_center = (cx - offset, cy, cz), (cx + offset, cy, cz)
        
    hc = crop_size // 2
    hc_global = global_crop_size // 2

    # --- CALCUL DU VOLUME SBR ---
    mid_x, mid_y, mid_z = int((left_center[0] + right_center[0]) / 2), int((left_center[1] + right_center[1]) / 2), int((left_center[2] + right_center[2]) / 2)
    non_zero = vol_rot_3d[vol_rot_3d > 0]
    if len(non_zero) > 0:
        bg_mask = (vol_rot_3d > np.percentile(non_zero, 15)).copy()
        sx_r, sy_r, sz_r = vol_rot_3d.shape
        x_min, x_max = max(0, mid_x - 20), min(sx_r, mid_x + 20)
        y_min, y_max = max(0, mid_y - 20), min(sy_r, mid_y + 20)
        z_min, z_max = max(0, mid_z - 20), min(sz_r, mid_z + 20)
        bg_mask[x_min:x_max, y_min:y_max, z_min:z_max] = False
        bg_pixels = vol_rot_3d[bg_mask]
        B = np.median(bg_pixels) if len(bg_pixels) > 100 else np.median(non_zero)
    else: B = 1e-6
    if B <= 0: B = 1e-6

    vol_sbr_3d = np.clip((vol_rot_3d - B) / B, 0, None)
    
    def extract_slices_3d(img, cx_, cy_, cz_, half_size):
        cx_, cy_, cz_ = int(cx_), int(cy_), int(cz_)
        padded = np.pad(img, pad_width=half_size, mode='constant', constant_values=0)
        cx_p, cy_p, cz_p = cx_ + half_size, cy_ + half_size, cz_ + half_size
        cube = padded[cx_p - half_size : cx_p + half_size, cy_p - half_size : cy_p + half_size, cz_p - half_size : cz_p + half_size]
        
        z_offsets = [-2 * stride, -1 * stride, 0, 1 * stride, 2 * stride]
        axial = np.stack([cube[:, :, half_size + z_off] for z_off in z_offsets], axis=0) 
        y_offsets = [-2 * stride, -1 * stride, 0, 1 * stride, 2 * stride]
        coronal = np.stack([cube[:, half_size + y_off, :] for y_off in y_offsets], axis=0) 
        
        mip_axial = cube.max(axis=2)   
        mip_coronal = cube.max(axis=1) 
        
        return mip_axial, mip_coronal, axial, coronal, cube
    
    # Extraction individuelle
    mip_ax_L, mip_cor_L, axial_L, coronal_L, cube_L = extract_slices_3d(vol_rot_3d, *left_center, hc)
    mip_ax_R, mip_cor_R, axial_R, coronal_R, cube_R = extract_slices_3d(vol_rot_3d, *right_center, hc)
    mip_ax_L_sbr, mip_cor_L_sbr, axial_L_sbr, coronal_L_sbr, cube_L_sbr = extract_slices_3d(vol_sbr_3d, *left_center, hc)
    mip_ax_R_sbr, mip_cor_R_sbr, axial_R_sbr, coronal_R_sbr, cube_R_sbr = extract_slices_3d(vol_sbr_3d, *right_center, hc)
    
    # Extraction Globale (Striatums)
    mip_ax_G, mip_cor_G, axial_G, coronal_G, cube_G = extract_slices_3d(vol_rot_3d, mid_x, mid_y, mid_z, hc_global)
    mip_ax_G_sbr, mip_cor_G_sbr, axial_G_sbr, coronal_G_sbr, cube_G_sbr = extract_slices_3d(vol_sbr_3d, mid_x, mid_y, mid_z, hc_global)

    def canonicalize_right(mip_a, mip_c, ax, cor, cb):
        return (np.flip(mip_a, axis=0), np.flip(mip_c, axis=0), 
                np.flip(ax, axis=1), np.flip(cor, axis=1), 
                np.flip(cb, axis=0))

    mip_ax_R, mip_cor_R, axial_R, coronal_R, cube_R = canonicalize_right(mip_ax_R, mip_cor_R, axial_R, coronal_R, cube_R)
    mip_ax_R_sbr, mip_cor_R_sbr, axial_R_sbr, coronal_R_sbr, cube_R_sbr = canonicalize_right(mip_ax_R_sbr, mip_cor_R_sbr, axial_R_sbr, coronal_R_sbr, cube_R_sbr)

    # Concaténations
    patch_L_np = np.concatenate([mip_ax_L[np.newaxis], mip_cor_L[np.newaxis], axial_L, coronal_L], axis=0) 
    patch_R_np = np.concatenate([mip_ax_R[np.newaxis], mip_cor_R[np.newaxis], axial_R, coronal_R], axis=0) 
    patch_L_sbr_np = np.concatenate([mip_ax_L_sbr[np.newaxis], mip_cor_L_sbr[np.newaxis], axial_L_sbr, coronal_L_sbr], axis=0) 
    patch_R_sbr_np = np.concatenate([mip_ax_R_sbr[np.newaxis], mip_cor_R_sbr[np.newaxis], axial_R_sbr, coronal_R_sbr], axis=0) 
    
    patch_G_np = np.concatenate([mip_ax_G[np.newaxis], mip_cor_G[np.newaxis], axial_G, coronal_G], axis=0)
    patch_G_sbr_np = np.concatenate([mip_ax_G_sbr[np.newaxis], mip_cor_G_sbr[np.newaxis], axial_G_sbr, coronal_G_sbr], axis=0)

    return {
        "patch_L": torch.tensor(patch_L_np.copy(), dtype=torch.float32),
        "patch_R": torch.tensor(patch_R_np.copy(), dtype=torch.float32),
        "patch_L_sbr": torch.tensor(patch_L_sbr_np.copy(), dtype=torch.float32),
        "patch_R_sbr": torch.tensor(patch_R_sbr_np.copy(), dtype=torch.float32),
        
        "cube_L": torch.tensor(cube_L.copy(), dtype=torch.float32).unsqueeze(0),
        "cube_R": torch.tensor(cube_R.copy(), dtype=torch.float32).unsqueeze(0),
        "cube_L_sbr": torch.tensor(cube_L_sbr.copy(), dtype=torch.float32).unsqueeze(0),
        "cube_R_sbr": torch.tensor(cube_R_sbr.copy(), dtype=torch.float32).unsqueeze(0),
        
        # 🌟 Clé dédiée "striatums" pour le bilatéral englobant
        "striatums": {
            "patch_global": torch.tensor(patch_G_np.copy(), dtype=torch.float32),
            "cube_global": torch.tensor(cube_G.copy(), dtype=torch.float32).unsqueeze(0)
        },
        "striatums_sbr": {
            "patch_global": torch.tensor(patch_G_sbr_np.copy(), dtype=torch.float32),
            "cube_global": torch.tensor(cube_G_sbr.copy(), dtype=torch.float32).unsqueeze(0)
        },
        
        "angle_z": angle_z_deg,
        "angle_y": angle_y_deg
    }

# ==========================================
# 3. BRANCHE FEATURES (Hybride + Pyradiomics)
# ==========================================
def extract_features(volume, bright_mask, target_spacing=TARGET_SPACING):
    sc = volume
    cbright = bright_mask
    TS = sc.shape
    SYM_AXIS = TS[0] // 2

    # --- 1. SÉLECTION DE LA COUPE Z (Masque Gaussien) ---
    sx, sy, _ = sc.shape
    x, y = np.ogrid[:sx, :sy]
    sigma = sx / 8.0
    center_weight = np.exp(-((x - sx//2)**2 + (y - sy//2)**2) / (2 * sigma**2))
    
    weighted_sc = np.where(cbright, sc, 0.0) * center_weight[:, :, np.newaxis]
    sl = int(np.argmax(weighted_sc.sum(axis=(0, 1))))
    
    # --- 2. CALCUL DES FEATURES MANUELLES NÉCESSAIRES ---
    hy = TS[1] // 2
    ant = cbright.copy(); ant[:, hy:, :] = False
    post = cbright.copy(); post[:, :hy, :] = False
    ma = float(sc[ant].mean()) if ant.any() else 1.0
    mpv = float(sc[post].mean()) if post.any() else 1.0
    ap_gradient = ma / (mpv + 1e-6)

    vals = sc[cbright] if cbright.any() else np.array([0.0], dtype=np.float32)
    hist_, _ = np.histogram(vals, bins=50, range=(0.0, 1.0))
    ph = hist_[hist_ > 0].astype(float); ph /= ph.sum()
    entropy = float(-np.sum(ph * np.log(ph))) if len(ph) else 0.0
    skew_v = float(skew(vals)) if len(vals) >= 3 else 0.0

    b2d = cbright[:, :, sl]
    b2d_c = binary_closing(b2d, disk(2))
    cm_ = ndimage.center_of_mass(b2d_c) if b2d_c.any() else (TS[0] // 2, TS[1] // 2)

    rp = int(np.clip(cm_[1] if not np.any(np.isnan(cm_)) else TS[1] // 2, 0, TS[1] - 1))
    prof = sc[:, rp, sl]

    def _get_fwhm(p):
        h = float(p.max()) / 2.0
        ab = np.where(p > h)[0]
        return float(ab[-1] - ab[0]) if len(ab) >= 2 else 0.0
    
    fwhm = _get_fwhm(prof)

    cx0_, cy0_ = cm_
    xx, yy = np.ogrid[:TS[0], :TS[1]]
    dist_r = np.sqrt((xx - cx0_) ** 2 + (yy - cy0_) ** 2)
    slice_2d = sc[:, :, sl]
    intens_r = []
    for r in range(1, 51):
        mask_r = (dist_r >= r - 0.5) & (dist_r < r + 0.5)
        intens_r.append(float(slice_2d[mask_r].mean()) if mask_r.any() else 0.0)

    intens_r = np.array(intens_r)
    pk = intens_r.max() if len(intens_r) > 0 else 0.0
    b50 = np.where(intens_r < pk * 0.5)[0]
    r50_mm = float((b50[0] + 1) * target_spacing[0]) if len(b50) else float(50 * target_spacing[0])

    # --- 3. CALCUL DES FEATURES PYRADIOMICS ---
    strict_mask = sc > 0.40
    strict_mask = ndimage.binary_opening(strict_mask, structure=np.ones((3, 3, 3)))
    
    left_mask = np.zeros_like(strict_mask)
    left_mask[:SYM_AXIS, :, :] = strict_mask[:SYM_AXIS, :, :]
    
    right_mask = np.zeros_like(strict_mask)
    right_mask[SYM_AXIS:, :, :] = strict_mask[SYM_AXIS:, :, :]

    label_mask = np.zeros_like(sc, dtype=np.uint8)
    label_mask[left_mask] = 1
    label_mask[right_mask] = 2

    sitk_img = sitk.GetImageFromArray(sc.T)
    sitk_img.SetSpacing(target_spacing)
    sitk_mask = sitk.GetImageFromArray(label_mask.T)
    sitk_mask.SetSpacing(target_spacing)

    settings = {'binWidth': 0.05, 'resampledPixelSpacing': None, 'interpolator': sitk.sitkBSpline, 'verbose': False}
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('shape')
    extractor.enableFeatureClassByName('firstorder')
    extractor.enableFeatureClassByName('glcm')
    extractor.enableFeatureClassByName('glrlm')

    feat_L = extractor.execute(sitk_img, sitk_mask, label=1) if left_mask.any() else {}
    feat_R = extractor.execute(sitk_img, sitk_mask, label=2) if right_mask.any() else {}

    # --- 4. ASSEMBLAGE GLOBAL DES FEATURES ---
    result_dict = {
        'fwhm': fwhm,
        'entropy': entropy,
        'ap_gradient': ap_gradient,
        'r50_mm': r50_mm,
        'skewness': skew_v
    }

    valid_keys = [k for k in feat_L.keys() if k.startswith('original_')]
    for k in valid_keys:
        name = k.replace('original_', 'pyr_')
        v_L = float(feat_L.get(k, 0.0))
        v_R = float(feat_R.get(k, 0.0))
        result_dict[f"{name}_L"] = v_L
        result_dict[f"{name}_R"] = v_R
        result_dict[f"{name}_Asym"] = abs(v_L - v_R) / (v_L + v_R + 1e-6)

    features = np.array([result_dict.get(f, 0.0) for f in FEATURE_NAMES], dtype=np.float32)
    
    bad_mask = np.isnan(features) | np.isinf(features)
    if np.any(bad_mask):
        try:
            from loguru import logger
            bad_indices = np.where(bad_mask)[0]
            bad_names = [FEATURE_NAMES[i] for i in bad_indices]
            logger.warning(f"⚠️ Valeurs NaN/Inf détectées et corrigées pour les features : {bad_names}")
        except ImportError:
            pass

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

from .striatum_preproc import init_preproc, get_preproc
from .striatum_features import pair_features, spine_profile, head_tail_masks

CHANNEL_MIP = 0
 
 
class SpatialMedialFadeInference:
    def __init__(self, size=32, fade_start=22):
        m = torch.ones((1, size, size), dtype=torch.float32)
        for i in range(fade_start, size):
            m[:, :, i] = 1.0 - (i - fade_start) / (size - 1 - fade_start)
        self.mask = m
    def __call__(self, t):
        return t * self.mask.to(t.device)
 
 
def build_twophase_data(siamese_mashup_data, preproc=None):
    """Tout le preprocessing lourd, une seule fois par patient.
 
    Appele depuis prepare_all_inputs(). `preproc=None` recupere le singleton
    initialise par TwoPhaseMeanEnsemble ; renvoie None si Two-Phase n'est pas
    charge, pour ne pas penaliser les autres modeles.
    """
    if preproc is None:
        preproc = get_preproc(required=False)
        if preproc is None:
            return None
    pL, pR = siamese_mashup_data['patch_L'], siamese_mashup_data['patch_R']
    mip_L = pL[CHANNEL_MIP].detach().cpu().numpy().astype(np.float32)
    mip_R = pR[CHANNEL_MIP].detach().cpu().numpy().astype(np.float32)
    hs = preproc.head_side
 
    hL, tL = head_tail_masks(mip_L, hs)
    hR, tR = head_tail_masks(mip_R, hs)
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    return {
        'patch_L': pL, 'patch_R': pR,
        'patch_Global': siamese_mashup_data['patch_Global'],
        'mask_hL': t(hL).unsqueeze(0), 'mask_tL': t(tL).unsqueeze(0),
        'mask_hR': t(hR).unsqueeze(0), 'mask_tR': t(tR).unsqueeze(0),
        'spine_L': t(spine_profile(mip_L, preproc.spine_refs, preproc.spine_len,
                                   hs, preproc.normalize_spine_by_bg)),
        'spine_R': t(spine_profile(mip_R, preproc.spine_refs, preproc.spine_len,
                                   hs, preproc.normalize_spine_by_bg)),
        'scalars': t(preproc.scalars(pair_features(mip_L, mip_R, hs))),
    }

def build_tabular_data(siamese_mashup_data, head_side="right"):
    """
    Extraction directe des features tabulaires (Shape & Intensité) pour XGBoost.
    Remplace l'ancienne dépendance au singleton TwoPhase/preproc.
    """
    pL = siamese_mashup_data['patch_L']
    pR = siamese_mashup_data['patch_R']
    
    # Extraction des MIPs sur le canal approprié
    mip_L = pL[CHANNEL_MIP].detach().cpu().numpy().astype(np.float32)
    mip_R = pR[CHANNEL_MIP].detach().cpu().numpy().astype(np.float32)
    
    # Appel direct de la logique d'extraction 
    # (Remplace "pair_features" par le nom exact de ta fonction si différent)
    tabular_dict = pair_features(mip_L, mip_R, head_side)
    
    # On ajoute directement le dictionnaire au payload patient
    siamese_mashup_data['tabular_features'] = tabular_dict
    
    return siamese_mashup_data

# ==========================================
# 4. POINT D'ENTRÉE UNIQUE POUR L'INFÉRENCE
# ==========================================
def prepare_all_inputs(nifti_path, invert_y=False):
    base = preprocess_patient(nifti_path)
    vol_numpy = preprocess_patient_numpy(nifti_path)

    if invert_y:
        base["volume"] = np.flip(base["volume"], axis=1).copy()
        vol_numpy = np.flip(vol_numpy, axis=1).copy()
        base["tensor"] = torch.flip(base["tensor"], dims=[2])
        
    tensor_numpy = torch.tensor(vol_numpy, dtype=torch.float32).unsqueeze(0) 

    siamese_data = extract_siamese_patches(vol_numpy, crop_size=32, global_crop_size=48, stride=2)

    # two_phase_data = build_twophase_data(
    #     {"patch_L": siamese_data["patch_L"],
    #     "patch_R": siamese_data["patch_R"],
    #     "patch_Global": siamese_data["striatums"]["patch_global"]},
    # ) 

    # scalar_features = build_tabular_data(
    #     {"patch_L": siamese_data["patch_L"],
    #     "patch_R": siamese_data["patch_R"],
    #     "patch_Global": siamese_data["striatums"]["patch_global"]},
    # ) 

    return {
        "volume_3d": base["tensor"],                
        "volume_3d_numpy": tensor_numpy,            
        "slices_2d": extract_2d_slices(base["volume"]),
        "features": extract_features(vol_numpy, vol_numpy > np.percentile(vol_numpy, 95)),
        "slices_2d_seq": extract_2d_slices(base["volume"], num_slices=20, step_z=1),
        "slices_2d_corax": {
            "slices_2d_axial": extract_2d_slices(base["volume"], num_slices=20, step_z=1),
            "slices_2d_coronal": extract_2d_slices_coronal(base["volume"], num_slices=20, step_y=1),
        },
        "slices_2d_sagcorax": {
            "slices_2d_axial": extract_2d_slices(base["volume"], num_slices=20, step_z=1),
            "slices_2d_coronal": extract_2d_slices_coronal(base["volume"], num_slices=20, step_y=1),
            "slices_2d_sagittal": extract_2d_slices_sagittal(base["volume"], num_slices=20, step_x=2),
        },
        "siamese_patches": {
            "patch_L": siamese_data["patch_L"],
            "patch_R": siamese_data["patch_R"],
            "angle_z": siamese_data["angle_z"],
            "angle_y": siamese_data["angle_y"]
        },
        "siamese_patches_sbr": {
            "patch_L": siamese_data["patch_L_sbr"],
            "patch_R": siamese_data["patch_R_sbr"],
            "angle_z": siamese_data["angle_z"],
            "angle_y": siamese_data["angle_y"]
        },
        "siamese_cubes": {
            "cube_L": siamese_data["cube_L"],
            "cube_R": siamese_data["cube_R"],
            "angle_z": siamese_data["angle_z"],
            "angle_y": siamese_data["angle_y"]
        },
        "siamese_cubes_sbr": {
            "cube_L": siamese_data["cube_L_sbr"],
            "cube_R": siamese_data["cube_R_sbr"],
            "angle_z": siamese_data["angle_z"],
            "angle_y": siamese_data["angle_y"]
        },
        # 🌟 Clé "striatums" propre, indépendante des siamese_patches d'origine
        "striatums": {
            "patch_global": siamese_data["striatums"]["patch_global"],
            "cube_global": siamese_data["striatums"]["cube_global"]
        },
        "striatums_sbr": {
            "patch_global": siamese_data["striatums_sbr"]["patch_global"],
            "cube_global": siamese_data["striatums_sbr"]["cube_global"]
        },
        "siamese_mashup_data": {
            "patch_L": siamese_data["patch_L"],
            "patch_R": siamese_data["patch_R"],
            "patch_Global": siamese_data["striatums"]["patch_global"]
        },
        # "scalars" : scalar_features["tabular_features"]
        # "twophase_data": two_phase_data
    }