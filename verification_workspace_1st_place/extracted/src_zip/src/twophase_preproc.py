"""
twophase_preproc.py
===================
Construit la cle "twophase_data" attendue par TwoPhaseMeanEnsemble.

A appeler depuis prepare_all_inputs() dans preprocessing.py :

    from .twophase_preproc import build_twophase_data
    ...
    "twophase_data": build_twophase_data(siamese_data, feature_extractor=extract_pair_features_from_patches)

Les masques et la spine sont calcules ici, a partir des MIP axiaux (canal 0)
des patchs gauche et droit -- exactement les memes entrees qu'a l'entrainement.
"""
import numpy as np
import torch

from .striatum_geometry import head_tail_masks, spine_profile

CHANNEL_MIP = 0   # canal du MIP axial dans patch_L / patch_R


def build_twophase_data(siamese_mashup_data, scalar_values=None,
                        spine_refs=None, spine_len=24, head_side='right',
                        normalize_spine_by_bg=True):
    """
    Parametres
    ----------
    siamese_mashup_data : dict avec patch_L, patch_R, patch_Global
    scalar_values : np.ndarray (n_scalars,) BRUTE, non standardisee.
                    La standardisation est faite par fold dans l'ensemble,
                    car chaque fold a son propre scaler.
    spine_refs : {'head': float, 'tail': float} du fold. Si None, la spine est
                 renvoyee nulle (le modele la verra comme une extraction ratee).

    Retourne un dict pret pour TwoPhaseMeanEnsemble.
    """
    pL = siamese_mashup_data['patch_L']
    pR = siamese_mashup_data['patch_R']
    mip_L = pL[CHANNEL_MIP].detach().cpu().numpy().astype(np.float32)
    mip_R = pR[CHANNEL_MIP].detach().cpu().numpy().astype(np.float32)

    hL, tL = head_tail_masks(mip_L, head_side)
    hR, tR = head_tail_masks(mip_R, head_side)

    if spine_refs is not None:
        sL = spine_profile(mip_L, spine_refs, spine_len, head_side, normalize_spine_by_bg)
        sR = spine_profile(mip_R, spine_refs, spine_len, head_side, normalize_spine_by_bg)
    else:
        u = np.linspace(-1, 1, spine_len).astype(np.float32)
        z = np.zeros(spine_len, np.float32)
        sL = sR = np.stack([z, z, u], -1)

    t = lambda a: torch.tensor(a, dtype=torch.float32)
    return {
        'patch_L': pL,
        'patch_R': pR,
        'patch_Global': siamese_mashup_data['patch_Global'],
        'mask_hL': t(hL).unsqueeze(0), 'mask_tL': t(tL).unsqueeze(0),
        'mask_hR': t(hR).unsqueeze(0), 'mask_tR': t(tR).unsqueeze(0),
        'spine_L': t(sL), 'spine_R': t(sR),
        # BRUTE : la standardisation depend du fold, elle est faite dans l'ensemble
        'scalars_raw': None if scalar_values is None else np.asarray(scalar_values, np.float32),
    }


def standardize_scalars(raw, artifacts):
    """Applique le scaler DU FOLD. Reproduit exactement l'entrainement :
    imputation par la mediane du train, puis z-score, puis nan_to_num."""
    cols = artifacts['scalar_cols']
    if raw is None:
        return torch.zeros(len(cols), dtype=torch.float32)
    x = np.asarray(raw, dtype=np.float64).copy()
    if x.shape[0] != len(cols):
        raise ValueError(f"{x.shape[0]} scalaires recus, {len(cols)} attendus "
                         f"(fold {artifacts['fold']}). Verifiez l'ordre des colonnes.")
    med = np.array([artifacts['medians'].get(c, 0.0) for c in cols], dtype=np.float64)
    bad = ~np.isfinite(x)
    x[bad] = med[bad]
    z = (x - artifacts['scaler_mean']) / artifacts['scaler_scale']
    return torch.tensor(np.nan_to_num(z, nan=0., posinf=0., neginf=0.), dtype=torch.float32)
