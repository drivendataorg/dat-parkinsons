import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import torch
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import warnings
import logging

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger('nibabel').setLevel(logging.ERROR)

import sys
sys.path.append(os.path.dirname(__file__))

from src.preprocessing import prepare_all_inputs

from src.inference_tta_wrappers import (
    SiameseSynthTTA,
    SiameseMashupTTA,
    SiameseMashupAlignTTA,
    ResNet2DTTA,
    SeResNext26DLSTMSagcoraxTTA,
    MedicalNet34TTA,
    aug_hflip_cv2,
    aug_rot_cv2,
    aug_shift_cv2,
    aug_rotate_3d
)

# ==========================================
# 1. CONFIGURATION
# ==========================================
CSV_PATH = '/data2/iceberg/dat_parkinson/dataset_siamese_patches/folds_siamese.csv'
NIFTI_DIR = Path('/data2/iceberg/dat_parkinson/niftis')
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), 'weights')
OUTPUT_DIR = Path(os.path.join(os.path.dirname(__file__), 'oofs_tta'))
VISUAL_DIR = OUTPUT_DIR / 'tta_visualizations'

SAVE_FREQ = 50 
DEBUG_PATIENTS_LIMIT = 10 

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VISUAL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_ROUTING = {
    'Siamese_Synth': "siamese_patches",
    'Siamese_Mashup': "siamese_mashup_data",
    'Siamese_Mashup_Align': "siamese_mashup_data",
    'ResNet2D': "slices_2d",
    'SeResNext26DLSTMSagcorax': "slices_2d_sagcorax",
    'MedicalNet34': "volume_3d_numpy" 
}

# ==========================================
# 2. OUTILS DE DEBUG VISUEL
# ==========================================
def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.squeeze().cpu().numpy()
    return tensor

def view_fix_siam(img): return np.rot90(img, k=1)
def view_fix_2d_ax(img): return np.rot90(img, k=2)
def view_fix_2d_cor(img): return img
def view_fix_2d_sag(img): return img
def view_fix_3d_ax(img): return np.rot90(img, k=1)
def view_fix_3d_cor(img): return np.rot90(img, k=1)
def view_fix_3d_sag(img): return np.rot90(img, k=1)

def plot_grid(images_dict, title, filename, view_fixes=None):
    n_rows = len(images_dict)
    n_cols = len(list(images_dict.values())[0])
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols + 1, 3 * n_rows))
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    if n_rows == 1: axes = [axes]
    if n_cols == 1: axes = [[ax] for ax in axes]
        
    for r, (tta_name, img_list) in enumerate(images_dict.items()):
        for c, img in enumerate(img_list):
            ax = axes[r][c]
            img_np = to_numpy(img)
            if view_fixes is not None and c < len(view_fixes) and view_fixes[c] is not None:
                img_np = view_fixes[c](img_np)
                
            ax.imshow(img_np, cmap='gray')
            if r == 0: ax.set_title(f"Vue {c+1}", fontsize=11)
            if c == 0: ax.set_ylabel(tta_name, fontsize=10, fontweight='bold', rotation=0, labelpad=50, ha='right')
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.tight_layout()
    plt.savefig(VISUAL_DIR / filename, bbox_inches='tight')
    plt.close()

def generate_patient_tta_debug(inputs, uid):
    logger.info(f"📸 Debug visuel : Génération des planches TTA pour le patient {uid}...")
    def apply_tf(tensor, func, *args, **kwargs):
        return func(tensor.unsqueeze(0), *args, **kwargs).squeeze(0)

    # 1. Siamese Synth
    oL, oR = inputs['siamese_patches']['patch_L'][0], inputs['siamese_patches']['patch_R'][0]
    plot_grid({
        "1. Baseline (Origine)": [oL, oR],
        "2. Swap L/R": [oR, oL],
        "3. Rotation +5° (Fond Noir)": [apply_tf(oL, TF.rotate, 5, fill=0), apply_tf(oR, TF.rotate, 5, fill=0)],
        "4. Shift (+5, +5)px": [apply_tf(oL, TF.affine, 0, [5, 5], 1, 0, fill=0), apply_tf(oR, TF.affine, 0, [5, 5], 1, 0, fill=0)]
    }, f"Siamese Synth : Origine vs TTA (Patient {uid})", f"{uid}_01_siamese_synth.png", view_fixes=[view_fix_siam, view_fix_siam]) 

    # 2. Siamese Mashup
    oG = inputs['siamese_mashup_data']['patch_Global'][0]
    plot_grid({
        "1. Baseline (Origine)": [oL, oR, oG],
        "2. Swap L/R + Miroir Global": [oR, oL, apply_tf(oG, TF.vflip)],
        "3. Rotation +5°": [apply_tf(oL, TF.rotate, 5, fill=0), apply_tf(oR, TF.rotate, 5, fill=0), apply_tf(oG, TF.rotate, 5, fill=0)],
        "4. Shift (+5, +5)px": [apply_tf(oL, TF.affine, 0, [5, 5], 1, 0, fill=0), apply_tf(oR, TF.affine, 0, [5, 5], 1, 0, fill=0), apply_tf(oG, TF.affine, 0, [5, 5], 1, 0, fill=0)]
    }, f"Siamese Mashup : Origine vs TTA (Patient {uid})", f"{uid}_02_siamese_mashup.png", view_fixes=[view_fix_siam, view_fix_siam, view_fix_siam]) 

    # 3. ResNet2D
    slices_2d = inputs['slices_2d']
    mid_slice = slices_2d[len(slices_2d) // 2]['image']
    plot_grid({
        "1. Baseline (Origine)": [mid_slice],
        "2. Horizontal Flip": [aug_hflip_cv2(mid_slice)],
        "3. Rotation +5°": [aug_rot_cv2(mid_slice, 5)],
        "4. Shift (+5, +5)px": [aug_shift_cv2(mid_slice, 5, 5)]
    }, f"ResNet2D Coupe Médiane : Origine vs TTA (Patient {uid})", f"{uid}_03_resnet2d.png", view_fixes=[view_fix_2d_ax])

    # 4. SagCorAx
    sagcorax = inputs['slices_2d_sagcorax']
    ax, cor, sag = sagcorax['slices_2d_axial'][len(sagcorax['slices_2d_axial'])//2]['image'], sagcorax['slices_2d_coronal'][len(sagcorax['slices_2d_coronal'])//2]['image'], sagcorax['slices_2d_sagittal'][len(sagcorax['slices_2d_sagittal'])//2]['image']
    plot_grid({
        "1. Baseline (Origine)": [ax, cor, sag],
        "2. Horizontal Flip": [aug_hflip_cv2(ax), aug_hflip_cv2(cor), aug_hflip_cv2(sag)],
        "3. Rotation +5°": [aug_rot_cv2(ax, 5), aug_rot_cv2(cor, 5), aug_rot_cv2(sag, 5)],
        "4. Shift (+5, +5)px": [aug_shift_cv2(ax, 5, 5), aug_shift_cv2(cor, 5, 5), aug_shift_cv2(sag, 5, 5)]
    }, f"SagCorAx : Origine vs TTA (Patient {uid})", f"{uid}_04_sagcorax.png", view_fixes=[view_fix_2d_ax, view_fix_2d_cor, view_fix_2d_sag])

    # 5. MedicalNet34
    vol = inputs['volume_3d_numpy']
    if isinstance(vol, torch.Tensor): vol = vol.numpy()
    z_mid, y_mid, x_mid = vol.shape[-3]//2, vol.shape[-2]//2, vol.shape[-1]//2
    vol_flip_anatomique = np.flip(vol, axis=-3)
    vol_rot5 = aug_rotate_3d(vol, 5) 
    plot_grid({
        "1. Baseline (Coupes Dim -3, -2, -1)": [vol[0, z_mid, :, :], vol[0, :, y_mid, :], vol[0, :, :, x_mid]],
        "2. Flip G/D (Axe -3)": [vol_flip_anatomique[0, z_mid, :, :], vol_flip_anatomique[0, :, y_mid, :], vol_flip_anatomique[0, :, :, x_mid]],
        "3. Rotation Axiale 3D +5°": [vol_rot5[0, z_mid, :, :], vol_rot5[0, :, y_mid, :], vol_rot5[0, :, :, x_mid]]
    }, f"MedicalNet34 3D : Origine vs TTA (Patient {uid})", f"{uid}_05_medicalnet34.png", view_fixes=[view_fix_3d_ax, view_fix_3d_cor, view_fix_3d_sag]) 

# ==========================================
# 3. INITIALISATION & UTILITAIRES
# ==========================================
def load_tta_models():
    logger.info("Chargement en mémoire des 6 architectures TTA (5 folds chacune)...")
    return {
        'Siamese_Synth': SiameseSynthTTA(WEIGHTS_DIR, prefix="siamois_final_pseudo"),
        'Siamese_Mashup': SiameseMashupTTA(WEIGHTS_DIR),
        'Siamese_Mashup_Align': SiameseMashupAlignTTA(WEIGHTS_DIR),
        'ResNet2D': ResNet2DTTA(WEIGHTS_DIR),
        'SeResNext26DLSTMSagcorax': SeResNext26DLSTMSagcoraxTTA(WEIGHTS_DIR, prefix="seresnext26d_lstm_sagcorax"),
        'MedicalNet34': MedicalNet34TTA(WEIGHTS_DIR, prefix="medicalnet34-ft")
    }

def save_checkpoints(results_by_model, output_dir, is_final=False):
    suffix = "" if is_final else "_IN_PROGRESS"
    for model_name, results in results_by_model.items():
        if len(results) > 0:
            pd.DataFrame(results).to_csv(output_dir / f"oof_tta_{model_name.lower()}{suffix}.csv", index=False)

# ==========================================
# 4. GÉNÉRATION DES OOF
# ==========================================
def main():
    print(f"\n{'='*80}\n🚀 GÉNÉRATION DES OOF (UN FICHIER PAR MODÈLE) AVEC TTA ET DEBUG VISUEL\n{'='*80}\n")

    df = pd.read_csv(CSV_PATH)
    if 'fold' not in df.columns: raise ValueError("Le CSV doit contenir une colonne 'fold'.")

    modeles_actifs = load_tta_models()
    results_by_model = {model_name: [] for model_name in modeles_actifs.keys()}
    
    patients_processed = 0
    debug_patients_count = 0 

    for fold in range(5):
        logger.info(f"Traitement du FOLD Validation {fold}...")
        val_df = df[df['fold'] == fold]
        
        for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc=f"Patients Fold {fold}"):
            uid, label = row['uid'], row['label']
            nifti_path = NIFTI_DIR / f"{uid}.nii.gz"
            
            if not nifti_path.exists(): continue
                
            inputs = prepare_all_inputs(str(nifti_path))
            
            if debug_patients_count < DEBUG_PATIENTS_LIMIT:
                generate_patient_tta_debug(inputs, uid)
                debug_patients_count += 1
            
            for model_name, model_instance in modeles_actifs.items():
                original_models = model_instance.models
                
                if len(original_models) > fold:
                    model_instance.models = [original_models[fold]]
                else:
                    logger.error(f"Modèle manquant pour {model_name} au fold {fold}")
                    continue
                
                try:
                    # 🌟 Les wrappers renvoient maintenant un dict {tta_name: prediction_output}
                    preds_dict = model_instance.predict(inputs[INPUT_ROUTING[model_name]])
                except Exception as e:
                    logger.error(f"Erreur avec {model_name} sur {uid} : {e}")
                    preds_dict = {"00_Baseline": 0.5}
                finally:
                    model_instance.models = original_models
                
                # Récupération de la probabilité OOF d'origine (Baseline)
                baseline_pred = preds_dict.get("00_Baseline", 0.5)
                prob_original_oof = baseline_pred[0] if isinstance(baseline_pred, (list, tuple)) else baseline_pred
                
                # Formatage du dictionnaire en lignes CSV
                for tta_name, pred_val in preds_dict.items():
                    if isinstance(pred_val, (list, tuple)):
                        prob_tta = pred_val[0]
                        extras = pred_val[1:]
                    else:
                        prob_tta = pred_val
                        extras = []
                        
                    row_data = {
                        'uid': uid, 'label': label, 'fold': fold,
                        'tta_name': tta_name,
                        'prob_tta': prob_tta,
                        'prob_original_oof': prob_original_oof
                    }
                    
                    # Intégration des features d'attention (si présentes dans l'output du modèle)
                    if len(extras) == 3:
                        row_data['attn_global'] = extras[0]
                        row_data['attn_fade'] = extras[1]
                        row_data['attn_tight'] = extras[2]
                    else:
                        for i, ext in enumerate(extras):
                            row_data[f'extra_feat_{i}'] = ext
                            
                    results_by_model[model_name].append(row_data)
                
            patients_processed += 1
            if patients_processed % SAVE_FREQ == 0:
                save_checkpoints(results_by_model, OUTPUT_DIR, is_final=False)

    for f in OUTPUT_DIR.glob("*_IN_PROGRESS.csv"): f.unlink()
    save_checkpoints(results_by_model, OUTPUT_DIR, is_final=True)

    print("\n" + "="*80 + "\n🏆 RÉCAPITULATIF DES PERFORMANCES OOF (Sur la Baseline Uniquement)\n" + "="*80)
    from sklearn.metrics import log_loss, roc_auc_score
    
    for model_name, results in results_by_model.items():
        if not results: continue
        df_model = pd.DataFrame(results)
        
        # 🌟 Le récapitulatif ne se base QUE sur la 00_Baseline pour ne pas fausser le calcul AUC/Loss
        df_baseline = df_model[df_model['tta_name'] == '00_Baseline']
        if len(df_baseline) == 0: continue
            
        y_true, y_pred = df_baseline['label'].astype(int), df_baseline['prob_original_oof']
        
        ll = log_loss(y_true, y_pred) if len(y_true.unique()) > 1 else float('inf')
        auc = roc_auc_score(y_true, y_pred) if len(y_true.unique()) > 1 else 0.5
            
        print(f" ➔ {model_name:<30} | Log Loss : {ll:.4f} | AUC : {auc:.4f}")
            
    print(f"\n✅ Terminé ! Fichiers OOF dans : {OUTPUT_DIR}/\n📸 Images de debug dans : {VISUAL_DIR}/")

if __name__ == '__main__':
    main()