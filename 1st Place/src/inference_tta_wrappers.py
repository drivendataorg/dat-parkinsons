"""
inference_tta_wrappers.py
=========================
Wrappers de Test Time Augmentation (TTA).
Retournent un dictionnaire {nom_tta: prediction} pour le stacking.
"""

import numpy as np
import cv2
import torch
import torchvision.transforms.functional as TF
from scipy.ndimage import rotate as rotate3d

from src.inference_2d_avg import ResNet2DEnsemble, SimpleSiameseEnsemble, SiameseMashupEnsemble, SiameseMashupAlignEnsemble
from src.inference_2d_sagcorax import SeResNext26DLSTMSagcoraxEnsemble
from src.inference_3d import MedicalNetEnsemble

# ==========================================
# 1. FONCTIONS D'AUGMENTATION 
# ==========================================

def aug_hflip_cv2(img_2d): return cv2.flip(img_2d, 1)

def aug_rot_cv2(img_2d, angle):
    h, w = img_2d.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img_2d, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def aug_shift_cv2(img_2d, dx, dy):
    h, w = img_2d.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img_2d, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def apply_to_slice_list(slices_list, aug_func, *args):
    augmented = []
    for s in slices_list:
        new_s = s.copy()
        new_s['image'] = aug_func(s['image'], *args)
        augmented.append(new_s)
    return augmented

def aug_rotate_3d(volume, angle):
    is_tensor = isinstance(volume, torch.Tensor)
    vol_np = volume.numpy() if is_tensor else volume
    rotated = rotate3d(vol_np, angle, axes=(-3, -2), reshape=False, order=1, cval=0.0)
    rotated = np.ascontiguousarray(rotated)
    return torch.from_numpy(rotated) if is_tensor else rotated

# ==========================================
# 2. WRAPPERS TTA PAR ARCHITECTURE
# ==========================================

class SiameseSynthTTA(SimpleSiameseEnsemble):
    def predict(self, siamese_data):
        if not siamese_data or 'patch_L' not in siamese_data: return {"00_Baseline": 0.5}
        oL, oR = siamese_data['patch_L'], siamese_data['patch_R']
        res = {}
        res["00_Baseline"] = super().predict(siamese_data)
        res["01_Swap_LR"] = super().predict({'patch_L': oR, 'patch_R': oL})
        res["02_Rot_Plus5"] = super().predict({'patch_L': TF.rotate(oL, 5, fill=0), 'patch_R': TF.rotate(oR, 5, fill=0)})
        res["03_Rot_Minus5"] = super().predict({'patch_L': TF.rotate(oL, -5, fill=0), 'patch_R': TF.rotate(oR, -5, fill=0)})
        res["04_Shift_P5_P5"] = super().predict({'patch_L': TF.affine(oL, 0, [5, 5], 1.0, 0, fill=0), 'patch_R': TF.affine(oR, 0, [5, 5], 1.0, 0, fill=0)})
        res["05_Shift_M5_M5"] = super().predict({'patch_L': TF.affine(oL, 0, [-5, -5], 1.0, 0, fill=0), 'patch_R': TF.affine(oR, 0, [-5, -5], 1.0, 0, fill=0)})
        return res

class SiameseMashupTTA(SiameseMashupEnsemble):
    def predict(self, siamese_data):
        if not siamese_data or 'patch_Global' not in siamese_data: return {"00_Baseline": 0.5}
        oL, oR, oG = siamese_data['patch_L'], siamese_data['patch_R'], siamese_data['patch_Global']
        res = {}
        res["00_Baseline"] = super().predict(siamese_data)
        res["01_Swap_VFlip"] = super().predict({'patch_L': oR, 'patch_R': oL, 'patch_Global': TF.vflip(oG)})
        res["02_Rot_Plus5"] = super().predict({'patch_L': TF.rotate(oL, 5, fill=0), 'patch_R': TF.rotate(oR, 5, fill=0), 'patch_Global': TF.rotate(oG, 5, fill=0)})
        res["03_Rot_Minus5"] = super().predict({'patch_L': TF.rotate(oL, -5, fill=0), 'patch_R': TF.rotate(oR, -5, fill=0), 'patch_Global': TF.rotate(oG, -5, fill=0)})
        res["04_Shift_P5_P5"] = super().predict({'patch_L': TF.affine(oL, 0, [5, 5], 1, 0, fill=0), 'patch_R': TF.affine(oR, 0, [5, 5], 1, 0, fill=0), 'patch_Global': TF.affine(oG, 0, [5, 5], 1, 0, fill=0)})
        res["05_Shift_M5_M5"] = super().predict({'patch_L': TF.affine(oL, 0, [-5, -5], 1, 0, fill=0), 'patch_R': TF.affine(oR, 0, [-5, -5], 1, 0, fill=0), 'patch_Global': TF.affine(oG, 0, [-5, -5], 1, 0, fill=0)})
        return res

class SiameseMashupAlignTTA(SiameseMashupAlignEnsemble):
    def predict(self, siamese_data):
        if not siamese_data or 'patch_Global' not in siamese_data: return {"00_Baseline": 0.5}
        oL, oR, oG = siamese_data['patch_L'], siamese_data['patch_R'], siamese_data['patch_Global']
        res = {}
        res["00_Baseline"] = super().predict(siamese_data)
        res["01_Swap_VFlip"] = super().predict({'patch_L': oR, 'patch_R': oL, 'patch_Global': TF.vflip(oG)})
        res["02_Rot_Plus5"] = super().predict({'patch_L': TF.rotate(oL, 5, fill=0), 'patch_R': TF.rotate(oR, 5, fill=0), 'patch_Global': TF.rotate(oG, 5, fill=0)})
        res["03_Rot_Minus5"] = super().predict({'patch_L': TF.rotate(oL, -5, fill=0), 'patch_R': TF.rotate(oR, -5, fill=0), 'patch_Global': TF.rotate(oG, -5, fill=0)})
        res["04_Shift_P5_P5"] = super().predict({'patch_L': TF.affine(oL, 0, [5, 5], 1, 0, fill=0), 'patch_R': TF.affine(oR, 0, [5, 5], 1, 0, fill=0), 'patch_Global': TF.affine(oG, 0, [5, 5], 1, 0, fill=0)})
        res["05_Shift_M5_M5"] = super().predict({'patch_L': TF.affine(oL, 0, [-5, -5], 1, 0, fill=0), 'patch_R': TF.affine(oR, 0, [-5, -5], 1, 0, fill=0), 'patch_Global': TF.affine(oG, 0, [-5, -5], 1, 0, fill=0)})
        return res

class ResNet2DTTA(ResNet2DEnsemble):
    def predict(self, slices_2d):
        if not slices_2d: return {"00_Baseline": 0.5}
        res = {}
        res["00_Baseline"] = super().predict(slices_2d)
        res["01_HFlip"] = super().predict(apply_to_slice_list(slices_2d, aug_hflip_cv2))
        res["02_Rot_Plus5"] = super().predict(apply_to_slice_list(slices_2d, aug_rot_cv2, 5))
        res["03_Rot_Minus5"] = super().predict(apply_to_slice_list(slices_2d, aug_rot_cv2, -5))
        res["04_Shift_P5_P5"] = super().predict(apply_to_slice_list(slices_2d, aug_shift_cv2, 5, 5))
        res["05_Shift_M5_M5"] = super().predict(apply_to_slice_list(slices_2d, aug_shift_cv2, -5, -5))
        return res

class SeResNext26DLSTMSagcoraxTTA(SeResNext26DLSTMSagcoraxEnsemble):
    def predict(self, slices_input, *args, **kwargs):
        if not isinstance(slices_input, dict): 
            return {"00_Baseline": super().predict(slices_input, *args, **kwargs)}
        
        ax, cor, sag = slices_input.get("slices_2d_axial"), slices_input.get("slices_2d_coronal"), slices_input.get("slices_2d_sagittal")
        if ax is None or cor is None or sag is None: return {"00_Baseline": 0.5}
        
        def apply_triview(func, *fargs):
            return super(SeResNext26DLSTMSagcoraxTTA, self).predict({
                "slices_2d_axial": apply_to_slice_list(ax, func, *fargs),
                "slices_2d_coronal": apply_to_slice_list(cor, func, *fargs),
                "slices_2d_sagittal": apply_to_slice_list(sag, func, *fargs)
            })
            
        res = {}
        res["00_Baseline"] = super().predict(slices_input)
        res["01_HFlip"] = apply_triview(aug_hflip_cv2)
        res["02_Rot_Plus5"] = apply_triview(aug_rot_cv2, 5)
        res["03_Rot_Minus5"] = apply_triview(aug_rot_cv2, -5)
        res["04_Shift_P5_P5"] = apply_triview(aug_shift_cv2, 5, 5)
        return res

class MedicalNet34TTA(MedicalNetEnsemble):
    def predict(self, volume):
        res = {}
        res["00_Baseline"] = super().predict(volume)
        
        if isinstance(volume, torch.Tensor):
            res["01_Flip_X"] = super().predict(torch.flip(volume, dims=[-3]))
        else:
            res["01_Flip_X"] = super().predict(np.ascontiguousarray(np.flip(volume, axis=-3)))
            
        res["02_Rot_Plus5"] = super().predict(aug_rotate_3d(volume, 5))
        res["03_Rot_Minus5"] = super().predict(aug_rotate_3d(volume, -5))
        return res