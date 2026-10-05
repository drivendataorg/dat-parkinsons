# main.py
from pathlib import Path
import os
import sys
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from loguru import logger
import warnings
import logging
import subprocess

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger('nibabel').setLevel(logging.ERROR)

# ==========================================
# 1. CONFIGURATION ET CHEMINS
# ==========================================
DATA_DIR = Path("/data2/iceberg/dat_parkinson/data-demo")
if not DATA_DIR.exists():
    DATA_DIR = Path("data") # Fallback local
    
NIFTI_DIR = DATA_DIR / "niftis"
SUBMISSION_FORMAT_PATH = DATA_DIR / "submission_format.csv"
WRITE_SUBMISSION_PATH = Path("submission.csv")
TQDM = False

# 🚀 NOUVEAU SYSTÈME DE CONFIGURATION RÉTROCOMPATIBLE
# Options : "FAMILY_STACKING" | "STACKING_SKLEARN" | "SOLO"
MODE_INFERENCE = "FAMILY_STACKING" 

# Configurations spécifiques selon le mode choisi :
SOLO_MODEL_NAME = "Mashup_Internal" 

# Définit les 4 familles exactes à moyenner en mode FAMILY_STACKING
FAMILY_MODELS = [
    # "Mashup_Internal_DWT",  # 4br
    "Mashup_Internal",      # 3br
    # "MiniCNN3D",            # 3d (à ajuster selon le modèle exact que tu veux utiliser)
    # "XGBoost"               # xgb
]
TEMP = 1.094

SRC_ROOT = Path(__file__).parent.resolve()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, 'weights')
WEIGHTS_DIR_ZIP = os.path.join(BASE_DIR, 'weights.zip')

def unzip_folder(zip_path, extract_to, path_to_check):
    if os.path.exists(path_to_check):
        return
    logger.info(f"Unzipping {zip_path}...")
    subprocess.run(
        ["unzip", "-q", zip_path, "-d", extract_to],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True,
    )


# >>>> START: Unzipping for submission src/ and weights/ <<<<
if __name__ == '__main__':
    unzip_folder(
        zip_path=os.path.join(BASE_DIR, 'src.zip'),
        extract_to=BASE_DIR,
        path_to_check=os.path.join(BASE_DIR, 'src'),
    )
    unzip_folder(WEIGHTS_DIR_ZIP, BASE_DIR, WEIGHTS_DIR)
# >>>> END: Unzipping <<<<


import sys
sys.path.insert(0, "./src/libs")

# ==========================================
# IMPORTATIONS DES MODULES D'INFÉRENCE
# ==========================================
from src.preprocessing import prepare_all_inputs

from src.inference_2d_avg import (
    ResNet2DEnsemble, 
    ViT2DEnsemble,
    SimpleSiameseEnsemble,
    DualStreamSiameseEnsemble,
    SiameseMashupEnsemble,
    SiameseMashupAlignEnsemble,
    SiameseMashupWaveletLinearEnsemble,
    SiameseMashupWaveletAttnEnsemble,
    TwoPhaseMeanEnsemble, # 🚀 NOUVEAU
    SiameseMashupInternalEnsemble
)

from src.inference_2d_rnn import (
    Resnet18DGRUEnsemble, 
    Resnet34DGRUEnsemble, 
    SeResNext26DGRUEnsemble, 
    Resnet18DLSTMEnsemble
)
from src.inference_2d_channel import (
    SeResNext26TS2DChannelEnsemble,
    SeResNext26D2DChannelEnsemble, 
    RegnetY0022DChannelEnsemble
)

from src.inference_3d import (
    DenseNet121Ensemble, 
    DenseNet201Ensemble, 
    AttentionDenseNet121Ensemble, 
    AttentionDenseNet201Ensemble,
    MedicalNetEnsemble, 
    MedNeXtEnsemble,
    MiniCNN3DSiameseEnsemble
)

from src.inference_2d_corax import (
    SeResNext26DLSTMCoraxEnsemble,
    SeResNext26TSGRUCoraxEnsemble,
    SeResNext26DGRUCoraxEnsemble,
    Resnet18DLSTMCoraxEnsemble,
)

from src.inference_2d_sagcorax import (
    SeResNext26DLSTMSagcoraxEnsemble,
    SeResNext26TSGRUSagcoraxEnsemble,
    SeResNext26TSLSTMSagcoraxEnsemble,
    SeResNext26DGRUSagcoraxEnsemble,
    Resnet18DLSTMSagcoraxEnsemble,
)

from src.inference_tta_wrappers import (
    SiameseSynthTTA,
    SiameseMashupTTA,
    SiameseMashupAlignTTA,
    ResNet2DTTA,
    SeResNext26DLSTMSagcoraxTTA,
    MedicalNet34TTA
)

from src.inference_feats import (
    TabularInference, 
    StackingInference, 
    SoloModelInference, 
    XGBoostEnsembleInference, 
    LogitFamilyStackingInference
)
from constants import prefix_map

# ==========================================
# ROUTAGE DES INPUTS
# ==========================================
INPUT_ROUTING = {
    TabularInference: "features",
    
    (
        SimpleSiameseEnsemble,
        DualStreamSiameseEnsemble,
        SiameseSynthTTA
    ): "siamese_patches",

    (
        SiameseMashupEnsemble,
        SiameseMashupAlignEnsemble,
        SiameseMashupWaveletLinearEnsemble,
        SiameseMashupWaveletAttnEnsemble,
        SiameseMashupInternalEnsemble
    ): "siamese_mashup_data",
    
    # 🚀 NOUVEAU : Routage des données pour le modèle Two-Phase
    (TwoPhaseMeanEnsemble,): "twophase_data",

    (
        ResNet2DEnsemble, 
        ViT2DEnsemble, 
        SeResNext26TS2DChannelEnsemble, 
        RegnetY0022DChannelEnsemble, 
        SeResNext26D2DChannelEnsemble,
        ResNet2DTTA
    ): "slices_2d",
    
    (
        DenseNet121Ensemble, 
        DenseNet201Ensemble, 
        AttentionDenseNet121Ensemble, 
        AttentionDenseNet201Ensemble
    ): "volume_3d",

    (
        Resnet18DGRUEnsemble, 
        Resnet34DGRUEnsemble, 
        Resnet18DLSTMEnsemble, 
        SeResNext26DGRUEnsemble
    ): "slices_2d_seq",

    (
        SeResNext26DLSTMCoraxEnsemble,
        SeResNext26TSGRUCoraxEnsemble,
        SeResNext26DGRUCoraxEnsemble,
        Resnet18DLSTMCoraxEnsemble,
    ): "slices_2d_corax",

    (
        SeResNext26DLSTMSagcoraxEnsemble,
        SeResNext26TSGRUSagcoraxEnsemble,
        SeResNext26TSLSTMSagcoraxEnsemble,
        SeResNext26DGRUSagcoraxEnsemble,
        Resnet18DLSTMSagcoraxEnsemble,
        SeResNext26DLSTMSagcoraxTTA
    ): "slices_2d_sagcorax",

    (MiniCNN3DSiameseEnsemble,): "siamese_cubes",

    (XGBoostEnsembleInference,): "scalars",
}

def predict_for_model(instance_modele, inputs):
    """Retrieves the right input key from dictionary and calls predict."""
    for model_types, input_key in INPUT_ROUTING.items():
        if isinstance(instance_modele, model_types):
            return instance_modele.predict(inputs[input_key])
            
    return instance_modele.predict(inputs["volume_3d_numpy"])


# ==========================================
# INITIALISATION DE LA PIPELINE
# ==========================================
def init_pipeline():
    logger.info(f"DÉMARRAGE DU PIPELINE D'INFÉRENCE (MODE : {'STACKING' if MODE_INFERENCE else 'MODÈLE SOLO -> ' + SOLO_MODEL_NAME})...")
    
    if MODE_INFERENCE == "FAMILY_STACKING":
        # A. Mode Nouvelle Moyenne Logit (4 Familles)
        arbitre = LogitFamilyStackingInference(model_names=FAMILY_MODELS, temperature=TEMP)
        features_requises = arbitre.model_names 
        
    elif MODE_INFERENCE == "STACKING_SKLEARN":
        # B. Mode Ancien Stacking (Joblib Meta-model)
        meta_model_path = os.path.join(WEIGHTS_DIR, 'meta_model_stacking.pkl')
        arbitre = StackingInference(model_path=meta_model_path)
        features_requises = arbitre.model_names 
        
    else:
        # C. Mode Modèle Unique
        arbitre = SoloModelInference(SOLO_MODEL_NAME, use_clipping=False)
        features_requises = arbitre.model_names
        
    print("Modèles requis :", features_requises)
    modeles_actifs = {}

    # B. Chargement dynamique des sous-modèles
    for feat in features_requises:
        clean_name = feat.replace('prob_', '')
        
        # ── Modèles Siamois (2.5D) ──
        if clean_name == 'Siamese':
            modeles_actifs[clean_name] = SimpleSiameseEnsemble(WEIGHTS_DIR, prefix="simple_siamese")
        elif clean_name == 'Siamese2.5d':
            modeles_actifs[clean_name] = DualStreamSiameseEnsemble(WEIGHTS_DIR)
            
        elif clean_name == 'Siamese_Synth':
            modeles_actifs[clean_name] = SimpleSiameseEnsemble(WEIGHTS_DIR, prefix="siamois_final_pseudo")

        elif clean_name == 'Siamese_Mashup':
            modeles_actifs[clean_name] = SiameseMashupEnsemble(WEIGHTS_DIR)
        elif clean_name == 'Siamese_Mashup_Align':
            modeles_actifs[clean_name] = SiameseMashupAlignEnsemble(WEIGHTS_DIR)
        elif clean_name == 'Siamese_Mashup_DTW_Attn':
            modeles_actifs[clean_name] = SiameseMashupWaveletAttnEnsemble(WEIGHTS_DIR)
        elif clean_name == 'Siamese_Mashup_DTW_Linear':
            modeles_actifs[clean_name] = SiameseMashupWaveletLinearEnsemble(WEIGHTS_DIR)

        elif clean_name == 'MiniCNN3D':
            # Le prefix doit correspondre au nom sauvegardé (ex: minicnn3d_fold_0.pth)
            modeles_actifs[clean_name] = MiniCNN3DSiameseEnsemble(
                WEIGHTS_DIR, 
                prefix="minicnn3d", 
                n_exp=20,
                num_folds=5
            )

        # 🚀 NOUVEAU : Chargement du modèle Two-Phase
        elif clean_name == 'TwoPhase_Mean':
            modeles_actifs[clean_name] = TwoPhaseMeanEnsemble(WEIGHTS_DIR)

        # 🔑 Rétrocompatibilité : Modèle original sans DWT
        elif clean_name == 'Mashup_Internal':
            modeles_actifs[clean_name] = SiameseMashupInternalEnsemble(
                WEIGHTS_DIR, n_exp=20, n_folds=5, use_dwt=False
            )

        # 🚀 NOUVEAU : Modèle avec branche Ondelettes activée
        elif clean_name == 'Mashup_Internal_DWT':
            modeles_actifs[clean_name] = SiameseMashupInternalEnsemble(
                WEIGHTS_DIR, n_exp=20, n_folds=5, use_dwt=True, prefix='mashup_ensemble_dwt'
            )

        # 🚀 NOUVEAU : Chargement de l'ensemble XGBoost
        elif clean_name == 'XGBoost':
            features_path = os.path.join(WEIGHTS_DIR, 'top_k_features.json')
            modeles_actifs[clean_name] = XGBoostEnsembleInference(
                models_dir=WEIGHTS_DIR, 
                features_list_path=features_path, 
                n_exp=20,   # Correspond à tes 100 modèles (20 * 5)
                k_splits=5
            )

        # ── Modèles 2D ──
        elif clean_name == 'ResNet2D':
            modeles_actifs[clean_name] = ResNet2DEnsemble(WEIGHTS_DIR)
        elif clean_name == 'Vit2D':
            modeles_actifs[clean_name] = ViT2DEnsemble(WEIGHTS_DIR)

        # ── Modèles 2D + RNN ──
        elif clean_name == 'Resnet18DGRU':
            prefix = prefix_map[clean_name]
            modeles_actifs[clean_name] = Resnet18DGRUEnsemble(WEIGHTS_DIR, prefix=prefix)
        elif clean_name == 'Resnet18DLSTM':
            prefix = prefix_map[clean_name]
            modeles_actifs[clean_name] = Resnet18DLSTMEnsemble(WEIGHTS_DIR, prefix=prefix)
        elif clean_name == 'Resnet34DGRU':
            prefix = prefix_map[clean_name]
            modeles_actifs[clean_name] = Resnet34DGRUEnsemble(WEIGHTS_DIR, prefix=prefix)
        elif clean_name == 'SeResNext26DGRU':
            prefix = prefix_map[clean_name]
            modeles_actifs[clean_name] = SeResNext26DGRUEnsemble(WEIGHTS_DIR, prefix=prefix)

        # ── Modèles 2D + RNN + Corax ──
        elif clean_name == 'SeResNext26DLSTMCorax':
            prefix = prefix_map[clean_name]
            modeles_actifs[clean_name] = SeResNext26DLSTMCoraxEnsemble(WEIGHTS_DIR, prefix=prefix)
        elif clean_name == 'SeResNext26TSGRUCorax':
            prefix = prefix_map[clean_name]
            modeles_actifs[clean_name] = SeResNext26TSGRUCoraxEnsemble(WEIGHTS_DIR, prefix=prefix)
        elif clean_name == 'SeResNext26DGRUCorax':
            prefix = prefix_map[clean_name]
            modeles_actifs[clean_name] = SeResNext26DGRUCoraxEnsemble(WEIGHTS_DIR, prefix=prefix)

        # ── Modèles 2D + RNN + Sagcorax ──
        elif clean_name == 'SeResNext26DGRUSagcorax':
            prefix = prefix_map[clean_name]
            modeles_actifs[clean_name] = SeResNext26DGRUSagcoraxEnsemble(WEIGHTS_DIR, prefix=prefix)
        elif clean_name == 'SeResNext26DLSTMSagcorax':
            prefix = prefix_map[clean_name]
            modeles_actifs[clean_name] = SeResNext26DLSTMSagcoraxEnsemble(WEIGHTS_DIR, prefix=prefix)
        elif clean_name == 'SeResNext26TSLSTMSagcorax':
            prefix = prefix_map[clean_name]
            modeles_actifs[clean_name] = SeResNext26TSLSTMSagcoraxEnsemble(WEIGHTS_DIR, prefix=prefix)

        # ── Modèles 2D + Channel ──
        elif clean_name == 'SeResNext26TS2DChannel':
            prefix = prefix_map[clean_name]
            modeles_actifs[clean_name] = SeResNext26TS2DChannelEnsemble(WEIGHTS_DIR, prefix=prefix)
        elif clean_name == 'SeResNext26D2DChannel':
            prefix = prefix_map[clean_name]
            modeles_actifs[clean_name] = SeResNext26D2DChannelEnsemble(WEIGHTS_DIR, prefix=prefix)
        elif clean_name == 'RegnetY0022DChannel':
            prefix = prefix_map[clean_name]
            modeles_actifs[clean_name] = RegnetY0022DChannelEnsemble(WEIGHTS_DIR, prefix=prefix)
            
        # ── Modèles 3D ──
        elif clean_name == 'DenseNet3D121':
            prefix = prefix_map[clean_name]
            modeles_actifs[clean_name] = DenseNet121Ensemble(WEIGHTS_DIR, prefix=prefix)
            
        elif clean_name == 'DenseNet3D201':
            prefix = prefix_map[clean_name]
            modeles_actifs[clean_name] = DenseNet201Ensemble(WEIGHTS_DIR, prefix=prefix)
            
        elif clean_name == 'DenseNetAttention':
            prefix = prefix_map[clean_name]
            modeles_actifs[clean_name] = AttentionDenseNet121Ensemble(WEIGHTS_DIR, prefix=prefix)
            
        elif clean_name == 'MedNextB':
            prefix = prefix_map[clean_name]
            modeles_actifs[clean_name] = MedNeXtEnsemble(WEIGHTS_DIR, prefix=prefix)
        elif clean_name == 'MedNextM':
            prefix = prefix_map[clean_name]
            modeles_actifs[clean_name] = MedNeXtEnsemble(WEIGHTS_DIR, prefix=prefix)
            
        elif 'Medical' in clean_name:
            prefix = prefix_map.get(clean_name, clean_name.lower())
            modeles_actifs[clean_name] = MedicalNetEnsemble(WEIGHTS_DIR, prefix=prefix)
            
        # ── Modèles Tabulaires (SVM, etc.) ──
        else:
            modeles_actifs[clean_name] = TabularInference(clean_name, WEIGHTS_DIR)

    logger.info(f"✅ Pipeline prêt ! {len(modeles_actifs)} modèles de base chargés en mémoire.")
    return arbitre, modeles_actifs


# ==========================================
# 2. PRÉDICTION POUR UN NOUVEAU PATIENT
# ==========================================
def diagnostiquer_patient(nifti_path, arbitre, modeles_actifs):
    # 1. Preprocessing (Extraction Slices 2D, Volume 3D, Features Tabulaires, et Patchs Siamois)
    inputs = prepare_all_inputs(nifti_path)
    
    # 2. Collecte des avis
    avis_experts = {}
    for nom_modele, instance_modele in modeles_actifs.items():
        avis_experts[nom_modele] = predict_for_model(instance_modele, inputs)
            
    # 3. Verdict de l'arbitre (Blend Logit L1 + Ensemble Selection)
    verdict = arbitre.predict(avis_experts)
    verdict = arbitre.post_process(verdict, inputs['features'])

    return verdict


# ==========================================
# 3. BOUCLE PRINCIPALE
# ==========================================
def main() -> None:
    if not SUBMISSION_FORMAT_PATH.exists():
        logger.error(f"Fichier de soumission introuvable : {SUBMISSION_FORMAT_PATH}")
        return
        
    submission_format = pd.read_csv(SUBMISSION_FORMAT_PATH)
    logger.info(f"Loaded submission_format.csv.")

    logger.info("Initializing the full hybrid stacking pipeline (2D, 3D, Tabular, Siamese)...")
    arbitre, modeles_actifs = init_pipeline()

    logger.info("Generating predictions for nifti images.")

    if TQDM:
        it = tqdm(submission_format["uid"], desc="Inférence Patients")
    else:
        it = submission_format["uid"]

    for uid in it:
        img_path = NIFTI_DIR / f"{uid}.nii.gz"
        
        if img_path.exists():
            proba = diagnostiquer_patient(str(img_path), arbitre, modeles_actifs)
        else:
            logger.warning(f"Image introuvable pour le patient {uid}. Prédiction par défaut (0.5).")
            proba = 0.5
            
        submission_format.loc[submission_format["uid"] == uid, "is_pathologic"] = proba

    logger.info(f"Writing out {WRITE_SUBMISSION_PATH.name}")
    submission_format.to_csv(WRITE_SUBMISSION_PATH, index=False)
    logger.success("Pipeline done. Fichier de soumission généré avec succès !")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.exception("💥 ERREUR FATALE PENDANT L'INFÉRENCE :")
        raise