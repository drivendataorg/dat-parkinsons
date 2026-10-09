import os
import sys
import numpy as np
import nibabel
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from loguru import logger
import warnings
import logging
from scipy.special import logit, expit
import concurrent.futures
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger('nibabel').setLevel(logging.ERROR)

# ==========================================
# 1. 📂 CONFIGURATION CENTRALISÉE (YAML)
# ==========================================
cfg = OmegaConf.merge(OmegaConf.load("0_configs/default.yaml"), OmegaConf.from_cli())

SYNTHETIC_NIFTI_DIR = Path(cfg.paths.synthetic_niftis)
OUTPUT_DIR          = Path(cfg.paths.phase3_out)
OUTPUT_CSV_PATH     = OUTPUT_DIR / "synthetic_soft_labels.csv"

PHASE1_DIR = Path(cfg.paths.phase1_out)
PHASE2_DIR = Path(cfg.paths.phase2_out)
META_MODEL_PATH = PHASE2_DIR / 'meta_model_stacking.pkl'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TQDM = True

# ==========================================
# IMPORTATIONS DES MODULES D'INFÉRENCE
# ==========================================
from src.preprocessing import prepare_all_inputs
from src.inference_2d_avg import ResNet2DEnsemble, ViT2DEnsemble
from src.inference_3d import (
    DenseNet121Ensemble, 
    DenseNet201Ensemble,
    AttentionDenseNet121Ensemble, 
    AttentionDenseNet201Ensemble,
    MedicalNetEnsemble, 
    MedNeXtEnsemble
)
from src.inference_feats import TabularInference, StackingInference

class DummyInference:
    def __init__(self, name):
        self.name = name
    def predict(self, x):
        return 0.5

# Modèles dont le chargement a échoué (remplacés par DummyInference) / models that failed to load
ECHECS_CHARGEMENT = []

def avertir_echecs_chargement():
    if ECHECS_CHARGEMENT:
        noms = sorted(ECHECS_CHARGEMENT)
        logger.warning(f"⚠️ ATTENTION : {len(noms)} modèle(s) n'ont pas pu être chargés et sont remplacés par une constante 0.5 : {noms}. Les soft labels seront faux. Voir la/les trace(s) d'erreur ci-dessus.")
        logger.warning(f"⚠️ WARNING: {len(noms)} model(s) failed to load and were replaced by a constant 0.5: {noms}. Soft labels will be wrong. See the traceback(s) above.")

# ==========================================
# FONCTION WORKER POUR LE MULTITHREADING
# ==========================================
def _charger_un_modele(clean_name, model_info, phase1_dir):
    """Charge un seul modèle depuis son dossier de poids spécifique (Phase 1)."""
    try:
        # Le dossier contenant les .pth pour ce modèle précis
        model_weights_dir = str(phase1_dir / model_info.folder)
        prefix = model_info.get("prefix", clean_name.lower())
        
        if clean_name == 'ResNet2D':
            modele = ResNet2DEnsemble(model_weights_dir)
        elif clean_name == 'Vit2D':
            modele = ViT2DEnsemble(model_weights_dir)
            
        elif clean_name == 'DenseNet3D121':
            modele = DenseNet121Ensemble(model_weights_dir, prefix=prefix)
            
        elif clean_name == 'DenseNet3D201':
            modele = DenseNet201Ensemble(model_weights_dir, prefix=prefix)
            
        elif clean_name == 'DenseNetAttention':
            modele = AttentionDenseNet121Ensemble(model_weights_dir, prefix=prefix)
            
        elif clean_name == 'MedNextB':
            modele = MedNeXtEnsemble(model_weights_dir, prefix=prefix)
            
        elif 'Medical' in clean_name:
            modele = MedicalNetEnsemble(model_weights_dir, prefix=prefix)
            
        else:
            modele = TabularInference(clean_name, model_weights_dir)
            
        return clean_name, modele
        
    except Exception as e:
        # logger.exception va imprimer toute la trace rouge (stack trace)
        logger.exception(f"⚠️ Erreur détaillée pour {clean_name} :")
        ECHECS_CHARGEMENT.append(clean_name)
        return clean_name, DummyInference(clean_name)

# ==========================================
# INITIALISATION DU PIPELINE (EN PARALLÈLE)
# ==========================================
def init_pipeline():
    logger.info("🚀 DÉMARRAGE DU PIPELINE D'INFÉRENCE SOFT-LABELS (SYNTHÉTIQUE)...")
    
    if not META_MODEL_PATH.exists():
        raise FileNotFoundError(f"🚨 Méta-modèle introuvable ! Lancez la Phase 2 d'abord. ({META_MODEL_PATH})")
        
    arbitre = StackingInference(model_path=str(META_MODEL_PATH))
    
    features_requises = arbitre.model_names 
    logger.info(f"Modèles retenus par l'arbitre : {features_requises}")
    modeles_actifs = {}

    modeles_a_charger = []

    for feat in features_requises:
        clean_name = feat.replace('prob_', '')
        
        # On vérifie si l'arbitre utilise ce modèle (optimisation)
        idx = arbitre.model_names.index(clean_name)
        poids_A = abs(arbitre.meta_lr.coef_[0][idx]) if hasattr(arbitre.meta_lr, 'coef_') else 0.0
        poids_B = arbitre.ensemble_weights.get(clean_name, 0.0)
        
        if poids_A <= 1e-6 and poids_B <= 1e-6:
            logger.info(f" 🗑️ {clean_name} totalement ignoré par l'arbitre. Bypass du chargement.")
            modeles_actifs[clean_name] = DummyInference(clean_name)
        else:
            if clean_name not in cfg.models:
                logger.warning(f"⚠️ Le modèle {clean_name} est demandé par l'arbitre mais n'est pas dans paths.yaml !")
                modeles_actifs[clean_name] = DummyInference(clean_name)
            else:
                modeles_a_charger.append(clean_name)

    # ⚡ CHARGEMENT MULTI-THREADÉ
    logger.info(f"⚡ Chargement en parallèle de {len(modeles_a_charger)} sous-modèles sur le GPU...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(_charger_un_modele, nom, cfg.models[nom], PHASE1_DIR) 
            for nom in modeles_a_charger
        ]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Chargement GPU"):
            nom_modele, instance_modele = future.result()
            modeles_actifs[nom_modele] = instance_modele

    logger.info("✅ Pipeline prêt ! Tous les modèles sont en VRAM.")
    avertir_echecs_chargement()
    return arbitre, modeles_actifs


# ==========================================
# DIAGNOSTIC DÉTAILLÉ
# ==========================================
def diagnostiquer_patient_detaille(nifti_path, arbitre, modeles_actifs):
    inputs = prepare_all_inputs(nifti_path)

    avis_experts = {}
    for nom_modele, instance_modele in modeles_actifs.items():
        if isinstance(instance_modele, DummyInference):
            avis_experts[nom_modele] = instance_modele.predict(None)
            
        elif isinstance(instance_modele, (ResNet2DEnsemble, ViT2DEnsemble)):
            avis_experts[nom_modele] = instance_modele.predict(inputs["slices_2d"])
            
        elif isinstance(instance_modele, TabularInference):
            avis_experts[nom_modele] = instance_modele.predict(inputs["features"])

        elif isinstance(instance_modele, (DenseNet121Ensemble, DenseNet201Ensemble, AttentionDenseNet121Ensemble, AttentionDenseNet201Ensemble)):
            avis_experts[nom_modele] = instance_modele.predict(inputs["volume_3d"])
            
        else:
            avis_experts[nom_modele] = instance_modele.predict(inputs["volume_3d_numpy"])

    probas_sub_list = list(avis_experts.values())
    votes_gt_50 = sum(1 for p in probas_sub_list if p > 0.5)
    total_models = len(probas_sub_list)
    vote_ratio = votes_gt_50 / total_models if total_models > 0 else 0.0

    clean_probs = {k.replace('prob_', ''): v for k, v in avis_experts.items()}
    X_raw = np.array([clean_probs[name] for name in arbitre.model_names])
    
    X_calibrated = np.zeros_like(X_raw)
    for i, name in enumerate(arbitre.model_names):
        cal = arbitre.base_calibrators.get(name)
        X_calibrated[i] = arbitre._apply_calibrator(cal, X_raw[i])
        
    if arbitre.blend_space_logit:
        X_logit = arbitre._safe_logit(X_calibrated)
    else:
        X_logit = X_calibrated
        
    X_logit_2d = X_logit.reshape(1, -1)
    prob_meta_A = arbitre.meta_lr.predict_proba(X_logit_2d)[0, 1]
    prob_meta_B = expit(np.dot(X_logit, arbitre.weights_B_array))
    verdict_final = arbitre.predict(avis_experts)

    resultats = {}
    for nom_modele, prob_val in avis_experts.items():
        resultats[f"prob_{nom_modele}"] = float(prob_val)
        
    resultats["vote_count_gt_0.5"] = int(votes_gt_50)
    resultats["total_models"] = int(total_models)
    resultats["vote_ratio_gt_0.5"] = float(vote_ratio)
    resultats["prob_meta_A"] = float(prob_meta_A)
    resultats["prob_meta_B"] = float(prob_meta_B)
    resultats["is_pathologic"] = float(verdict_final)
    resultats["hard_label_0.5"] = 1 if verdict_final >= 0.5 else 0

    return resultats


# ==========================================
# BOUCLE PRINCIPALE
# ==========================================
def main() -> None:
    if not SYNTHETIC_NIFTI_DIR.exists():
        logger.error(f"Dossier de données synthétiques introuvable : {SYNTHETIC_NIFTI_DIR}")
        return

    nifti_files = sorted(list(SYNTHETIC_NIFTI_DIR.glob("*.nii.gz")))
    logger.info(f"📁 {len(nifti_files)} fichiers NIfTI trouvés dans {SYNTHETIC_NIFTI_DIR}")

    if len(nifti_files) == 0:
        logger.warning("Aucun fichier .nii.gz à traiter.")
        return

    arbitre, modeles_actifs = init_pipeline()

    records = []
    it = tqdm(nifti_files, desc="Labellisation Synthétique") if TQDM else nifti_files

    for img_path in it:
        filename = img_path.name
        uid = filename.replace('.nii.gz', '').replace('.nii', '')
        
        try:
            res_dict = diagnostiquer_patient_detaille(str(img_path), arbitre, modeles_actifs)
            res_dict["uid"] = uid
            res_dict["filename"] = filename
            records.append(res_dict)
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de {filename} : {e}")

    df_out = pd.DataFrame(records)

    first_cols = ["uid", "filename"]
    sub_model_cols = [c for c in df_out.columns if c.startswith("prob_") and c not in ["prob_meta_A", "prob_meta_B"]]
    stat_cols = ["vote_count_gt_0.5", "total_models", "vote_ratio_gt_0.5"]
    meta_cols = ["prob_meta_A", "prob_meta_B", "is_pathologic", "hard_label_0.5"]

    ordered_cols = first_cols + sub_model_cols + stat_cols + meta_cols
    remaining_cols = [c for c in df_out.columns if c not in ordered_cols]
    final_cols = ordered_cols + remaining_cols

    df_out = df_out[final_cols]
    df_out.to_csv(OUTPUT_CSV_PATH, index=False)
    
    logger.success(f"🎉 Terminé ! Soft labels sauvegardés dans : {OUTPUT_CSV_PATH.resolve()}")
    logger.info(f"📊 Aperçu des 3 premières lignes :\n{df_out.head(3)}")
    avertir_echecs_chargement()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception("💥 ERREUR FATALE PENDANT LA LABELLISATION SYNTHÉTIQUE :")
        raise