"""
generate_tabular_oof.py
=======================
1. Extraction Hybride et Mise en cache des 59 TOP features (via preprocessing.py).
2. Définition et GridSearch des modèles (XGBoost, SVM, RandomForest...).
3. Génération des prédictions OOF avec un K-Fold strict (selon CSV).
4. Sauvegarde des 5 meilleurs modèles OOF dans un fichier CSV.
5. Entraînement final sur 100% des données et sauvegarde (.pkl).
"""

import os
import joblib
import concurrent.futures
import multiprocessing
import numpy as np
import pandas as pd
import warnings
import logging
from omegaconf import OmegaConf
from pathlib import Path

# Silence global
warnings.filterwarnings("ignore")
logging.getLogger("radiomics").setLevel(logging.ERROR)

from src.preprocessing import preprocess_patient_numpy, extract_features

# Modèles
from sklearn.svm import SVC

# Outils ML
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================

cfg = OmegaConf.merge(
    OmegaConf.load("0_configs/default.yaml"), 
    OmegaConf.from_cli()
)

DATA_DIR    = cfg.paths.niftis
LABELS_PATH = cfg.paths.train_labels
OUTPUT_DIR = Path(cfg.paths.phase1_out) / cfg.models['SVM']['folder']

CACHE_FILE  = OUTPUT_DIR / 'features.csv'
MODELS_DIR  = OUTPUT_DIR

FEAT_WORKERS = min(32, multiprocessing.cpu_count())
N_WORKERS = 32

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

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ==========================================
# 2. MODULE D'EXTRACTION HYBRIDE & CACHING
# ==========================================
def _extract_features_worker(args):
    """
    Worker épuré qui délègue toute la logique complexe au fichier preprocessing.py
    """
    uid, nifti_path = args
    warnings.filterwarnings('ignore')
    
    try:
        # 1. Prétraitement numpy pour les features
        vol_numpy = preprocess_patient_numpy(nifti_path)
        
        # 2. Création du masque d'intensité > P95 (comme défini dans ton orchestrateur d'origine)
        bright_mask = vol_numpy > np.percentile(vol_numpy, 95)
        
        # 3. Extraction via le module central
        features_array = extract_features(vol_numpy, bright_mask)
        
        # 4. Formatage du résultat pour le DataFrame
        result = {'uid': uid}
        for i, f_name in enumerate(FEATURE_NAMES):
            result[f_name] = features_array[i]
            
        return result
        
    except Exception as err:
        return {'_error': uid, '_msg': str(err)}


def load_or_extract_features(df: pd.DataFrame, data_dir: str, cache_path: str) -> np.ndarray:
    if os.path.exists(cache_path):
        print(f"📦 Chargement des features depuis le cache : {cache_path}")
        df_cache = pd.read_csv(cache_path)
        df_merged = pd.merge(df[['uid']], df_cache, on='uid', how='left')
        return df_merged[FEATURE_NAMES].values
    
    args = [(row.uid, os.path.join(data_dir, f'{row.uid}.nii.gz')) for _, row in df.iterrows()]
    
    print(f'⚙️ Extraction Hybride des features ({len(args)} exams, {FEAT_WORKERS} workers)...')
    records, errors = [], []
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=FEAT_WORKERS) as pool:
        for res in tqdm(pool.map(_extract_features_worker, args), total=len(args)):
            if '_error' not in res:
                records.append(res)
            else:
                errors.append(res)
                # Fallback à Zéro en cas de corruption NIfTI
                dummy = {'uid': res['_error']}
                for f in FEATURE_NAMES: dummy[f] = 0.0
                records.append(dummy)
                
    if errors:
        print(f"⚠️ {len(errors)} patients ont échoué (remplis de zéros).")
        
    df_extracted = pd.DataFrame(records)
    # Nettoyage absolu des Infinity et NaNs
    df_extracted = df_extracted.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # On merge pour garantir l'ordre du dataset et on sauvegarde UNIQUEMENT les colonnes requises
    df_final = pd.merge(df[['uid']], df_extracted, on='uid', how='left')
    cols_to_save = ['uid'] + FEATURE_NAMES
    
    df_save = df_final[cols_to_save]
    df_save.to_csv(cache_path, index=False)
    print(f"💾 {len(FEATURE_NAMES)} Features optimales mises en cache dans : {cache_path}")
    
    return df_final[FEATURE_NAMES].values


# ==========================================
# 3. MODÈLES & ÉVALUATION
# ==========================================
def get_models_and_grids():
    return {
        "SVM": {
            "model": SVC(probability=True, random_state=42), 
            "params": {
                'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']
            }
        },
    }

def evaluate_model(name, model_info, X, y, df_folds):
    print(f"\n{'='*40}")
    print(f"🧪 Test du modèle : {name}")
    print(f"{'='*40}")
    
    base_model = clone(model_info['model'])
    param_grid = model_info['params']
    
    grid = GridSearchCV(base_model, param_grid, cv=3, scoring='neg_log_loss', n_jobs=N_WORKERS)
    X_scaled_for_grid = StandardScaler().fit_transform(X)
    grid.fit(X_scaled_for_grid, y)
    
    best_params = grid.best_params_
    print(f"✅ Meilleurs paramètres trouvés : {best_params}")
    best_model_instance = clone(base_model).set_params(**best_params)
    
    oof_preds = np.zeros(len(y))
    
    for fold in sorted(df_folds.unique()):
        val_idx = (df_folds == fold).values
        train_idx = ~val_idx
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_val_sc = scaler.transform(X_val)
        
        best_model_instance.fit(X_train_sc, y_train)
        oof_preds[val_idx] = best_model_instance.predict_proba(X_val_sc)[:, 1]
        
    model_auc = roc_auc_score(y, oof_preds)
    model_ll = log_loss(y, oof_preds)
    
    print(f"📊 {name} -> OOF AUC: {model_auc:.4f} | OOF Log Loss: {model_ll:.4f}")
    
    return {
        'model_name': name,
        'best_params': best_params,
        'oof_preds': oof_preds,
        'auc': model_auc,
        'log_loss': model_ll
    }

# ==========================================
# 4. ORCHESTRATEUR
# ==========================================
def main():
    df = pd.read_csv(LABELS_PATH)
    y = df['is_pathologic'].values.astype(int)
    
    X = load_or_extract_features(df, DATA_DIR, CACHE_FILE)
    print(f'\nMatrice de Features : {X.shape} | Pathologiques: {y.mean():.1%}\n')
    
    df_folds = df['fold']
    models_dict = get_models_and_grids()
    
    results = []
    for name, info in models_dict.items():
        res = evaluate_model(name, info, X, y, df_folds)
        results.append(res)
        
    df_results = pd.DataFrame(results).sort_values(by='log_loss')
    top_models = df_results.head(5).copy()
    
    print(f"\n{'='*65}")
    print("🏆 CLASSEMENT DES 5 MEILLEURES MÉTHODES TABULAIRES")
    print(f"{'='*65}")
    print(top_models[['model_name', 'auc', 'log_loss', 'best_params']].to_string(index=False))
    
    stack_dict = {'uid': df['uid']} 
    for _, row in top_models.iterrows():
        stack_dict[f"prob_{row['model_name']}"] = row['oof_preds']
        
    df_oof_tabular = pd.DataFrame(stack_dict)
    out_csv = os.path.join(OUTPUT_DIR, 'tabular_oof.csv')
    df_oof_tabular.to_csv(out_csv, index=False)
    
    print(f"\n✅ OOF Tabulaires générés et sauvegardés dans : {out_csv}")

    # =========================================================
    # 5. ENTRAÎNEMENT FINAL SUR TOUT LE DATASET ET SAUVEGARDE
    # =========================================================
    print(f"\n{'='*55}")
    print("💾 ENTRAÎNEMENT FINAL & SAUVEGARDE DES MODÈLES (.pkl)")
    print(f"{'='*55}")
    
    for _, row in top_models.iterrows():
        m_name = row['model_name']
        best_p = row['best_params']
        
        base_model = clone(models_dict[m_name]['model'])
        final_model = base_model.set_params(**best_p)
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', final_model)
        ])
        
        print(f"⚙️ Entraînement de {m_name:<15} sur 100% des données...")
        pipeline.fit(X, y)
        
        model_path = os.path.join(MODELS_DIR, f"{m_name}.pkl")
        joblib.dump(pipeline, model_path)
        print(f"   -> Sauvegardé : {model_path}")

if __name__ == '__main__':
    main()