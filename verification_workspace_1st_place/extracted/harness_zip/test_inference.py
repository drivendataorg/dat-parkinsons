import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from loguru import logger
import warnings
import logging

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger('nibabel').setLevel(logging.ERROR)

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = Path("/data2/iceberg/dat_parkinson/")
if not DATA_DIR.exists():
    DATA_DIR = Path("data") 
    
NIFTI_DIR = DATA_DIR / "niftis"

# ⚠️ Chemin vers l'OOF du nouveau modèle Internal Ensemble
OOF_CSV_PATH = Path("/data/HOME/sansel/dat-park/runs_siamois/mashup_internal_ensemble_full/master_oof_predictions.csv")

# ==========================================
# 2. IMPORTATIONS DEPUIS MAIN.PY
# ==========================================
import main 
from main import (
    init_pipeline, 
    predict_for_model, 
    TabularInference
)
from src.preprocessing import prepare_all_inputs

# ==========================================
# 3. DIAGNOSTIC CHIRURGICAL SUR UN SEUL MODÈLE
# ==========================================
def diagnostiquer_un_fold(nifti_path, arbitre, modeles_actifs, target_idx):
    """
    Exécute l'inférence en forçant les modèles d'ensemble à n'utiliser que 
    le réseau précis ciblé par 'target_idx'. Désactive le post-processing 
    pour simuler les probabilités brutes d'entraînement.
    """
    inputs = prepare_all_inputs(nifti_path)
    avis_experts = {}
    
    for nom_modele, instance_modele in modeles_actifs.items():
        if isinstance(instance_modele, TabularInference):
            avis_experts[nom_modele] = predict_for_model(instance_modele, inputs)
        else:
            original_models = getattr(instance_modele, 'models', None)
            original_apply = getattr(instance_modele, '_apply_recipe', None)
            
            if original_models is not None and target_idx < len(original_models):
                try:
                    # 1. On remplace temporairement la liste par le seul modèle visé
                    instance_modele.models = [original_models[target_idx]]
                    
                    # 2. On désactive temporairement la recette post-processing 
                    # pour matcher les probas brutes enregistrées dans le CSV d'OOF
                    if original_apply is not None:
                        instance_modele._apply_recipe = lambda p_array: float(np.mean(p_array)) if p_array else 0.5
                        
                    avis_experts[nom_modele] = predict_for_model(instance_modele, inputs)
                    
                finally:
                    # 3. On restaure la liste complète et la recette d'origine
                    instance_modele.models = original_models
                    if original_apply is not None:
                        instance_modele._apply_recipe = original_apply
            else:
                avis_experts[nom_modele] = predict_for_model(instance_modele, inputs)
            
    # Verdict de l'arbitre (si applicable)
    verdict_final = arbitre.predict(avis_experts)
    verdict_final = arbitre.post_process(verdict_final, inputs.get('features', None))
    
    return avis_experts, verdict_final

# ==========================================
# 4. BOUCLE DE COMPARAISON
# ==========================================
def main_test(num_samples=10):
    if not OOF_CSV_PATH.exists():
        logger.error(f"Fichier OOF introuvable : {OOF_CSV_PATH}")
        return
        
    df_ref = pd.read_csv(OOF_CSV_PATH)
    
    # Déduction dynamique du nombre de folds pour calculer l'index global (Ex: 5 folds -> index = exp*5 + fold)
    n_folds = int(df_ref['fold'].max() + 1) if 'fold' in df_ref.columns else 5
    
    arbitre, modeles_actifs = init_pipeline()
    
    sample_df = df_ref.head(num_samples)
    
    print(f"\n{'='*90}")
    print("🔍 VALIDATION CHIRURGICALE : OOF CSV vs INFÉRENCE TEMPS RÉEL (Modèle Brut)")
    print(f"{'='*90}")

    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        uid = row['uid']
        
        target_fold = int(row['fold']) if 'fold' in row else 0
        target_exp = int(row['exp']) if 'exp' in row else None
        
        # Calcul de l'index dans la liste `self.models` qui contient les 100 réseaux
        if target_exp is not None:
            target_idx = (target_exp * n_folds) + target_fold
            eval_str = f"EXP: {target_exp} / FOLD: {target_fold} (Index liste: {target_idx})"
        else:
            target_idx = target_fold
            eval_str = f"FOLD: {target_fold}"
            
        img_path = NIFTI_DIR / f"{uid}.nii.gz"
        
        if not img_path.exists():
            print(f"⚠️ Image manquante pour {uid}, ignoré.")
            continue
            
        label_val = row.get('label', row.get('is_pathologic', 'N/A'))
            
        print(f"\n👨‍⚕️ Patient: {uid} | Label: {label_val} | Evalué sur {eval_str}")
        
        # Extraction chirurgicale
        avis_experts_rt, verdict_rt = diagnostiquer_un_fold(str(img_path), arbitre, modeles_actifs, target_idx)
        
        print("   --- Sous-modèles ---")
        for model_name, prob_rt in avis_experts_rt.items():
            prob_ref = row.get('prob', None)
            
            if prob_ref is not None:
                diff = abs(prob_rt - prob_ref)
                status = "✅ EXACT" if diff < 1e-4 else "⚠️ ECART"
                print(f"   {status} {model_name:<28} | OOF Brut: {prob_ref:.4f} | Tps réel: {prob_rt:.4f} | Diff: {diff:.6f}")
            else:
                print(f"   ⚠️ Colonne 'prob' introuvable dans le CSV.")
                
        print("   --- Arbitre Final ---")
        ref_final = row.get('prob', None)
        
        if ref_final is not None:
            diff_final = abs(verdict_rt - ref_final)
            status_final = "✅ OK" if diff_final < 1e-4 else "⚠️ ECART"
            print(f"   {status_final} {'SOLO PIPELINE':<20} | OOF CSV: {ref_final:.4f} | Tps réel: {verdict_rt:.4f} | Diff: {diff_final:.4f}")
        else:
            print(f"   ⚠️ Impossible de vérifier : la colonne 'prob' n'existe pas dans le CSV.")
            
        print("-" * 90)

if __name__ == '__main__':
    main_test(num_samples=10)