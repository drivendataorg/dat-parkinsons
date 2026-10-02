"""
train_stacking.py
=================
Implémentation améliorée du stacking avec calibration nested et sélection d'ensembles :

1. Calibration nested (Platt/isotonic) de chaque modèle de base.
2. Deux méta-learners en espace LOGIT :
   A) LogisticRegressionCV pénalisée ElasticNet (L1 + L2) -- solver SAGA
      nested CV nativement.
   B) Ensemble Selection façon Caruana et al. (2004) -- DOUBLE NESTED CV PARALLÉLISÉE.
3. Le poids de blend entre A et B est recherché par grid search sur
   une grille de alpha (espace logit).
4. Calibration finale nested du stacking.
"""

import os
import warnings
import joblib
import concurrent.futures
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import cross_val_predict, PredefinedSplit, StratifiedKFold, ParameterGrid
from sklearn.metrics import log_loss, roc_auc_score
from scipy.special import logit, expit
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from omegaconf import OmegaConf

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore")

def warn(*args, **kwargs):
    pass
warnings.warn = warn

from calibrators import (
    safe_logit,
    TemperatureScaler, 
    PlattScaler, 
    IsotonicScaler,
    LabelSmoother, 
    BetaScaler, 
    SplineScaler,
    VectorTemperatureScaler,
    GPScaler,
)

# ==========================================
# 1. 📂 CHARGEMENT DE LA CONFIGURATION (YAML)
# ==========================================
cfg = OmegaConf.merge(OmegaConf.load("0_configs/default.yaml"), OmegaConf.from_cli())

LABELS_PATH      = cfg.paths.train_labels
PHASE1_DIR       = Path(cfg.paths.phase1_out)

# Création via Pathlib, puis conversion en string pour compatibilité avec os.path.join
OUTPUT_DIR_PATH  = Path(cfg.paths.phase2_out)
OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR       = str(OUTPUT_DIR_PATH)

# get() permet de renvoyer None si 'tabular_oof' vaut null dans le YAML
TABULAR_OOF_PATH = cfg.stacking.get("tabular_oof")

# ==========================================
# 2. 🔍 CONSTRUCTION ET VALIDATION DYNAMIQUE DES OOF
# ==========================================
EXTERNAL_MODELS = {}
missing_models = []

print(f"\n{'='*60}")
print("🔍 VALIDATION DES MODÈLES OOF (PHASE 1 -> PHASE 2)")
print(f"{'='*60}")

# On boucle uniquement sur les modèles activés pour le stacking
for model_name in cfg.stacking.active_models:
    if model_name not in cfg.models:
        raise KeyError(f"🚨 Modèle '{model_name}' activé dans le stacking mais introuvable dans le registre 'models' du YAML.")
    
    m_info = cfg.models[model_name]
    oof_path = PHASE1_DIR / m_info.folder / m_info.oof_file
    
    if oof_path.exists():
        EXTERNAL_MODELS[model_name] = {
            "path": str(oof_path),
            "prob_col": m_info.prob_col
        }
        print(f" ✅ {model_name:<20} : Trouvé ({oof_path.parent.name}/)")
    else:
        missing_models.append(model_name)
        print(f" ❌ {model_name:<20} : INTROUVABLE -> {oof_path}")

if not EXTERNAL_MODELS:
    raise FileNotFoundError("🚨 ERREUR CRITIQUE : Aucun modèle n'a été trouvé ! Vérifiez la Phase 1 et paths.yaml.")
elif missing_models:
    print(f"\n⚠️ Attention : {len(missing_models)} modèles configurés sont absents et seront ignorés.")
else:
    print(f"\n🎯 Tous les {len(EXTERNAL_MODELS)} modèles configurés sont prêts pour le Stacking !")

# --- REGISTRE DES CALIBRATEURS UNIFIÉS ---
CALIBRATOR_REGISTRY = {
    'temperature': TemperatureScaler,
    'vector_temp': VectorTemperatureScaler,
    'sigmoid': PlattScaler,
    'beta': BetaScaler,
    'spline': SplineScaler,
    'gp': GPScaler,
    'isotonic': IsotonicScaler,
    'smoothing': LabelSmoother,
}

# --- Réglages du stacking ---
NUM_WORKERS = 48
CALIBRATE_BASE_MODELS = True      
BASE_CALIBRATION_METHODS = []
FINAL_CALIBRATION_METHODS = ['temperature', 'sigmoid', 'beta']
BLEND_SPACE_LOGIT = True         

# --- Ensemble Selection : grille d'hyperparamètres à rechercher ---
CS_GRID = np.logspace(-3, 2, 15)
ENSEMBLE_PARAM_GRID = {
    'bag_frac': [0.5, 0.7, 0.9],
    'model_frac': [0.6, 0.8, 1.0],
    'max_models': [8, 15, 25],
}
ENSEMBLE_N_BAGS = 40          
SEARCH_N_BAGS = 15            
INNER_CV_SPLITS = 4           
SEARCH_N_JOBS = NUM_WORKERS   

BLEND_ALPHA_GRID = np.round(np.linspace(0.0, 1.0, 11), 2)  
FINAL_CALIBRATION = True         

# ==========================================
# 2. HELPERS : Calibration (ECE) & Logit
# ==========================================
def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
    return ece

def report(name, y, probs):
    auc = roc_auc_score(y, probs)
    ll = log_loss(y, probs)
    ece = compute_ece(y, probs)
    print(f" {name:<28} | AUC: {auc:.4f} | Log Loss: {ll:.4f} | ECE: {ece:.4f}")
    return auc, ll, ece

# ==========================================
# 3. CALIBRATION NESTED DES MODÈLES DE BASE
# ==========================================
def get_calibrator(method='sigmoid'):
    if method not in CALIBRATOR_REGISTRY:
        raise ValueError(f"Méthode inconnue: {method}. Choix : {list(CALIBRATOR_REGISTRY.keys())}")
    return CALIBRATOR_REGISTRY[method]()

def calibrate_oof_nested(probs, y, ps, method='sigmoid'):
    calibrated = np.zeros_like(probs, dtype=float)
    for train_idx, test_idx in ps.split():
        cal = get_calibrator(method).fit(probs[train_idx], y[train_idx])
        calibrated[test_idx] = cal.predict_proba(probs[test_idx])[:, 1]
    return calibrated

def auto_calibrate(probs, y, ps, label="", methods=['temperature', 'sigmoid', 'isotonic']):
    eps = 1e-6
    probs = np.asarray(probs, dtype=float)
    best_probs = probs.copy()
    best_loss = log_loss(y, np.clip(probs, eps, 1 - eps))
    best_method = "aucun (brut)"
    best_calibrator = None

    for m in methods:
        cal_probs = calibrate_oof_nested(probs, y, ps, method=m)
        loss = log_loss(y, np.clip(cal_probs, eps, 1 - eps))
        
        if loss < best_loss - 1e-6:
            best_loss = loss
            best_probs = cal_probs
            best_method = m

    if best_method != "aucun (brut)":
        best_calibrator = get_calibrator(best_method).fit(probs, y)
        t_info = f" (T = {best_calibrator.T:.3f})" if best_method == 'temperature' else ""
        print(f"   ✅ {label:<25} → Retenu : {best_method.upper():<11} | Log Loss : {best_loss:.4f}{t_info}")
    else:
        print(f"   ⚪ {label:<25} → Conservé : BRUT (AUCUNE AMÉLIORATION) | Log Loss : {best_loss:.4f}")

    return best_probs, best_calibrator, best_method

# ==========================================
# 4. ENSEMBLE SELECTION (Caruana et al., 2004)
# ==========================================
def _greedy_build(X_bag, y_bag, max_models):
    n_models = X_bag.shape[1]
    selected = []
    current_sum = None
    best_loss = np.inf

    for step in range(max_models):
        best_add, best_add_loss = None, best_loss
        for j in range(n_models):
            trial_sum = X_bag[:, j] if current_sum is None else current_sum + X_bag[:, j]
            trial_mean = trial_sum / (len(selected) + 1)
            loss = log_loss(y_bag, np.clip(expit(trial_mean), 1e-6, 1 - 1e-6))
            if loss < best_add_loss - 1e-7:
                best_add_loss, best_add = loss, j

        if best_add is None:
            break

        selected.append(best_add)
        current_sum = X_bag[:, best_add] if current_sum is None else current_sum + X_bag[:, best_add]
        best_loss = best_add_loss

        if len(selected) > 2:
            best_remove, best_remove_loss = None, best_loss
            for idx, candidate in enumerate(selected[:-1]): 
                temp_sum = current_sum - X_bag[:, candidate]
                temp_mean = temp_sum / (len(selected) - 1)
                loss = log_loss(y_bag, np.clip(expit(temp_mean), 1e-6, 1 - 1e-6))
                if loss < best_remove_loss - 1e-7:
                    best_remove_loss, best_remove = loss, idx

            if best_remove is not None:
                removed_model = selected.pop(best_remove)
                current_sum -= X_bag[:, removed_model]
                best_loss = best_remove_loss

    return selected

def _run_ensemble_bag(seed, X_logit, y, bag_frac, model_frac, max_models):
    n_samples, n_models = X_logit.shape
    rng = np.random.RandomState(seed)
    row_idx = rng.choice(n_samples, size=max(10, int(n_samples * bag_frac)), replace=False)
    model_idx = rng.choice(n_models, size=max(2, int(n_models * model_frac)), replace=False)
    X_bag = X_logit[np.ix_(row_idx, model_idx)]
    y_bag = y[row_idx]
    local_selected = _greedy_build(X_bag, y_bag, max_models)
    return [model_idx[s] for s in local_selected]

def ensemble_selection(X_logit, y, n_bags=ENSEMBLE_N_BAGS, bag_frac=0.7,
                        model_frac=0.8, max_models=15,
                        random_state=42, n_jobs=1):
    n_samples, n_models = X_logit.shape
    counts = np.zeros(n_models)

    for i in range(n_bags):
        selected = _run_ensemble_bag(random_state + i, X_logit, y, bag_frac, model_frac, max_models)
        for idx in selected:
            counts[idx] += 1

    if counts.sum() == 0:
        return np.ones(n_models) / n_models
    return counts / counts.sum()

def ensemble_predict(X_logit, weights):
    return np.clip(expit(X_logit @ weights), 1e-6, 1 - 1e-6)

# ==========================================
# 4.5 RECHERCHE D'HYPERPARAMÈTRES — Ensemble Selection
# ==========================================
def _ensemble_inner_cv_score(X, y, params, n_splits=INNER_CV_SPLITS, random_state=42, n_bags=SEARCH_N_BAGS):
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y))
    for tr_idx, te_idx in inner_cv.split(X, y):
        w = ensemble_selection(X[tr_idx], y[tr_idx], n_bags=n_bags, random_state=random_state, **params)
        oof[te_idx] = ensemble_predict(X[te_idx], w)
    return log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6))

def _score_params_on_data(X, y, params, n_splits, random_state, n_bags):
    score = _ensemble_inner_cv_score(X, y, params, n_splits=n_splits, random_state=random_state, n_bags=n_bags)
    return params, score

def select_best_ensemble_params(X, y, param_grid=ENSEMBLE_PARAM_GRID, n_splits=INNER_CV_SPLITS,
                                 random_state=42, n_bags=SEARCH_N_BAGS, verbose=False, n_jobs=1):
    param_list = list(ParameterGrid(param_grid))

    if n_jobs == 1:
        results = [_score_params_on_data(X, y, p, n_splits, random_state, n_bags) for p in param_list]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [pool.submit(_score_params_on_data, X, y, p, n_splits, random_state, n_bags)
                       for p in param_list]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

    best_params, best_score = min(results, key=lambda r: r[1])
    if verbose:
        for p, s in sorted(results, key=lambda r: r[1]):
            print(f"      {p} -> Log Loss interne: {s:.4f}")
    return best_params, best_score

def _evaluate_params_on_fold(fold_i, train_idx, X_logit, y, params, n_splits, random_state, n_bags):
    X_tr, y_tr = X_logit[train_idx], y[train_idx]
    score = _ensemble_inner_cv_score(X_tr, y_tr, params, n_splits=n_splits,
                                      random_state=random_state, n_bags=n_bags)
    return fold_i, params, score

def nested_ensemble_selection_oof_tuned(X_logit, y, ps, param_grid=ENSEMBLE_PARAM_GRID,
                                         n_inner_splits=INNER_CV_SPLITS, random_state=42,
                                         search_n_bags=SEARCH_N_BAGS, final_n_bags=ENSEMBLE_N_BAGS,
                                         n_jobs=1):
    fold_splits = list(ps.split())
    param_list = list(ParameterGrid(param_grid))
    tasks = [
        (fold_i, train_idx, params)
        for fold_i, (train_idx, _) in enumerate(fold_splits)
        for params in param_list
    ]
    print(f"   {len(tasks)} tâches ({len(fold_splits)} folds x {len(param_list)} configs) -- n_jobs={n_jobs}")

    if n_jobs == 1:
        results = [
            _evaluate_params_on_fold(fi, tr, X_logit, y, p, n_inner_splits, random_state, search_n_bags)
            for fi, tr, p in tasks
        ]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [
                pool.submit(_evaluate_params_on_fold, fi, tr, X_logit, y, p, n_inner_splits, random_state, search_n_bags)
                for fi, tr, p in tasks
            ]
            results = [f.result() for f in tqdm(concurrent.futures.as_completed(futures), total=len(futures),
                                                 desc="Recherche hyperparamètres Meta B")]

    best_by_fold = {}
    for fold_i, params, score in results:
        if fold_i not in best_by_fold or score < best_by_fold[fold_i][1]:
            best_by_fold[fold_i] = (params, score)

    oof = np.zeros(len(y))
    chosen_params_per_fold = []
    for fold_i, (train_idx, test_idx) in enumerate(fold_splits):
        X_tr, y_tr = X_logit[train_idx], y[train_idx]
        X_te = X_logit[test_idx]
        best_params, best_score = best_by_fold[fold_i]
        chosen_params_per_fold.append(best_params)
        print(f"   Fold {fold_i} -> {best_params} (Log Loss interne: {best_score:.4f})")
        w = ensemble_selection(X_tr, y_tr, n_bags=final_n_bags, random_state=random_state, **best_params)
        oof[test_idx] = ensemble_predict(X_te, w)

    return oof, chosen_params_per_fold

def search_blend_alpha(oof_A, oof_B, y, alpha_grid=BLEND_ALPHA_GRID):
    best_alpha, best_loss = 0.5, np.inf
    logit_A, logit_B = safe_logit(oof_A), safe_logit(oof_B)
    for alpha in alpha_grid:
        blend = expit(alpha * logit_A + (1 - alpha) * logit_B)
        loss = log_loss(y, np.clip(blend, 1e-6, 1 - 1e-6))
        if loss < best_loss:
            best_loss, best_alpha = loss, alpha
    return float(best_alpha), best_loss


# ==========================================
# MAIN
# ==========================================
def main():
    print(f"\n{'=' * 60}")
    print("🚀 STACKING — CALIBRATION + ENSEMBLE SELECTION TUNÉ (NESTED CV)")
    print(f"{'=' * 60}")

    df_labels = pd.read_csv(LABELS_PATH)
    df_stack = df_labels[['uid', 'is_pathologic', 'fold']].copy()

    # Modifié : On vérifie si la clé TABULAR_OOF_PATH existe et est non nulle
    if TABULAR_OOF_PATH is not None:
        if os.path.exists(TABULAR_OOF_PATH):
            print("📥 Importation : OOF Tabulaires")
            df_tab = pd.read_csv(TABULAR_OOF_PATH)
            prob_cols_tab = [c for c in df_tab.columns if c.startswith('prob_') if c not in ['prob_XGBoost', 'prob_CatBoost', 'prob_LightGBM', 'prob_ExtraTrees']]
            df_stack = pd.merge(df_stack, df_tab[['uid'] + prob_cols_tab], on='uid', how='left')
        else:
            print(f"⚠️ Fichier tabulaire introuvable : {TABULAR_OOF_PATH}")
    else:
        print("⚪ OOF Tabulaires désactivés dans la config (tabular_oof: null)")

    for model_name, info in EXTERNAL_MODELS.items():
        print(f"📥 Importation : OOF {model_name}")
        df_ext = pd.read_csv(info["path"])
        df_stack = pd.merge(df_stack, df_ext[['uid', info["prob_col"]]], on='uid', how='left')
        new_col_name = f'prob_{model_name}'
        df_stack.rename(columns={info["prob_col"]: new_col_name}, inplace=True)

    all_proba_columns = [col for col in df_stack.columns if col.startswith('prob_')]
    proba_columns = []

    for col in all_proba_columns:
        proba_columns.append(col)
        
    for col in proba_columns:
        if df_stack[col].isnull().any():
            df_stack[col] = df_stack[col].fillna(df_stack[col].mean())  
            
    df_stack = pd.merge(df_labels[['uid', 'is_pathologic', 'fold']], df_stack, on=['uid', 'is_pathologic', 'fold'], how='left')

    model_names = [c.replace('prob_', '') for c in proba_columns]
    y = df_stack['is_pathologic'].values.astype(int)
    ps = PredefinedSplit(test_fold=df_stack['fold'].values)

    def _icon(name):
        if "3D" in name: return "🤖"
        elif "2DChannel" in name: return "📻"
        elif name.endswith("Sagcorax"): return "🤯"
        elif name.endswith("Corax"): return "💫"
        elif name.endswith('GRU') or name.endswith('LSTM'): return "⏰" 
        elif any(dl in name for dl in ["ResNet", "Vit", "MedicalNet", "DenseNet", "MedNext", "Resnext", "Siamese"]): return "🧠"
        return "🌳"

    # ==========================================
    # 5. PERFORMANCES INDIVIDUELLES (BRUT)
    # ==========================================
    print(f"\n{'=' * 60}")
    print("📈 PERFORMANCES INDIVIDUELLES DES MODÈLES (OOF BRUT)")
    print(f"{'=' * 60}")
    raw_scores = []
    for col, name in zip(proba_columns, model_names):
        auc, ll, ece = report(f"{_icon(name)} {name}", y, df_stack[col].values)
        raw_scores.append({'Modèle': name, 'AUC': auc, 'Log Loss': ll, 'ECE': ece})
    pd.DataFrame(raw_scores).sort_values(by='Log Loss').to_csv(
        os.path.join(OUTPUT_DIR, 'scores_individuels_bruts.csv'), index=False)

    # ==========================================
    # 6. CALIBRATION NESTED DES MODÈLES DE BASE 
    # ==========================================
    X_raw = df_stack[proba_columns].values
    base_calibrators = {}

    if CALIBRATE_BASE_MODELS:
        print(f"\n{'=' * 60}")
        print("🎯 CALIBRATION NESTED DES MODÈLES DE BASE")
        print(f"{'=' * 60}")
        X_calibrated = np.zeros_like(X_raw)
        for i, (col, name) in enumerate(zip(proba_columns, model_names)):
            calibrated_probs, calibrator, method_used = auto_calibrate(
                probs=X_raw[:, i],
                y=y,
                ps=ps,
                label=f"({name})",
                methods=BASE_CALIBRATION_METHODS
            )
            
            X_calibrated[:, i] = calibrated_probs
            base_calibrators[name] = calibrator
            report(f"{_icon(name)} {name} (final)", y, X_calibrated[:, i])
        X_final_base = X_calibrated
    else:
        print("\n⚪ Calibration des modèles de base désactivée (CALIBRATE_BASE_MODELS=False)")
        X_final_base = X_raw

    X_logit = safe_logit(X_final_base) if BLEND_SPACE_LOGIT else X_final_base

    # ==========================================
    # 7. MÉTA-LEARNER A — ElasticNet LogisticRegressionCV
    # ==========================================
    print(f"\n{'=' * 60}")
    print("🔷 MÉTA-LEARNER A — LogisticRegressionCV (pénalité ElasticNet, espace logit)")
    print(f"{'=' * 60}")

    meta_lr = LogisticRegressionCV(
        Cs=CS_GRID,
        cv=5,
        penalty='elasticnet',
        solver='saga',
        l1_ratios=[0.1, 0.5, 0.9],
        scoring='neg_log_loss',
        random_state=42,
        max_iter=3000,
        n_jobs=NUM_WORKERS,
    )
    oof_A = cross_val_predict(meta_lr, X_logit, y, cv=ps, method='predict_proba', n_jobs=NUM_WORKERS)[:, 1]
    report("Meta A (ElasticNet LR, nested OOF)", y, oof_A)

    meta_lr.fit(X_logit, y)
    coefs_A = dict(zip(model_names, meta_lr.coef_[0]))

    # ==========================================
    # 8. MÉTA-LEARNER B — Ensemble Selection TUNÉ 
    # ==========================================
    print(f"\n{'=' * 60}")
    print("🔶 MÉTA-LEARNER B — Ensemble Selection (hyperparamètres recherchés en double nested CV)")
    print(f"{'=' * 60}")

    oof_B, chosen_params_per_fold = nested_ensemble_selection_oof_tuned(
        X_logit, y, ps,
        param_grid=ENSEMBLE_PARAM_GRID, n_inner_splits=INNER_CV_SPLITS,
        random_state=42, search_n_bags=SEARCH_N_BAGS, final_n_bags=ENSEMBLE_N_BAGS,
        n_jobs=SEARCH_N_JOBS,
    )
    report("Meta B (Ensemble Selection tuné, nested OOF)", y, oof_B)

    print("\n   🏭 Recherche des hyperparamètres de production (sur toutes les données)...")
    best_params_prod, best_score_prod = select_best_ensemble_params(
        X_logit, y, param_grid=ENSEMBLE_PARAM_GRID, n_splits=INNER_CV_SPLITS,
        random_state=42, n_bags=SEARCH_N_BAGS, verbose=False, n_jobs=SEARCH_N_JOBS)
    print(f"   -> Retenus : {best_params_prod} (Log Loss interne: {best_score_prod:.4f})")

    weights_B = ensemble_selection(X_logit, y, n_bags=ENSEMBLE_N_BAGS, random_state=42, **best_params_prod)
    weights_B_dict = dict(zip(model_names, weights_B))

    # ==========================================
    # 9. STACK-OF-STACKERS — alpha recherché entre A et B
    # ==========================================
    print(f"\n{'=' * 60}")
    print("🏆 RECHERCHE DU POIDS DE BLEND ALPHA (Meta A vs Meta B)")
    print(f"{'=' * 60}")
    
    best_alpha, best_alpha_loss = search_blend_alpha(oof_A, oof_B, y, alpha_grid=BLEND_ALPHA_GRID)
    print(f"   alpha optimal : {best_alpha:.2f}  (1.0 = tout Meta A, 0.0 = tout Meta B)")
    print(f"   Log Loss au meilleur alpha : {best_alpha_loss:.4f}\n")

    oof_combined = expit(best_alpha * safe_logit(oof_A) + (1 - best_alpha) * safe_logit(oof_B))

    report("Meta A seul", y, oof_A)
    report("Meta B seul", y, oof_B)
    report(f"Combiné A+B (alpha={best_alpha:.2f})", y, oof_combined)
    
    oof_final = oof_combined

    # ==========================================
    # 10. CALIBRATION FINALE DU STACKING 
    # ==========================================
    final_calibrator = None
    if FINAL_CALIBRATION:
        print(f"\n{'=' * 60}")
        print("🎯 CALIBRATION NESTED DU STACKING FINAL")
        print(f"{'=' * 60}")

        eps = 1e-6
        original_loss = log_loss(y, np.clip(oof_final, eps, 1 - eps))
        
        best_loss = original_loss
        best_method = None
        best_oof_calibrated = None

        for method in FINAL_CALIBRATION_METHODS:
            oof_cv = np.zeros_like(oof_final)
            for train_idx, test_idx in ps.split():
                calibrator = get_calibrator(method).fit(oof_final[train_idx], y[train_idx])
                oof_cv[test_idx] = calibrator.predict_proba(oof_final[test_idx])[:, 1]

            cv_loss = log_loss(y, np.clip(oof_cv, eps, 1 - eps))

            if cv_loss < best_loss - 1e-6:
                best_loss = cv_loss
                best_method = method
                best_oof_calibrated = oof_cv

        if best_method is not None:
            print(f"   ✅ Calibration cross-validée ({best_method.upper()}) améliore le score : {original_loss:.4f} → {best_loss:.4f}")
            final_calibrator = get_calibrator(best_method).fit(oof_final, y)
            oof_final = best_oof_calibrated
        else:
            print(f"   ⚪ Calibration cross-validée ne procure pas d'amélioration : {original_loss:.4f} (conservé)")
            final_calibrator = None

        report("Stacking final (retenu)", y, oof_final)

    final_auc = roc_auc_score(y, oof_final)
    final_loss = log_loss(y, oof_final)
    final_ece = compute_ece(y, oof_final)

    # ==========================================
    # 11. POIDS, COEFFICIENTS ET ÉLAGAGE (PRUNING SANS BIAIS)
    # ==========================================
    print(f"\n{'=' * 60}")
    print("⚖️ POIDS ACCORDÉS PAR LES MÉTA-LEARNERS")
    print(f"{'=' * 60}")
    
    # Seuil en dessous duquel on considère qu'un modèle ne sert à rien
    TOLERANCE = 1e-4

    print("\n🔷 Meta A (coefficients ElasticNet, 0 = exclu) :")
    active_A = []
    for name, w in sorted(coefs_A.items(), key=lambda x: -abs(x[1])):
        if abs(w) > TOLERANCE:
            print(f"   {_icon(name)} {name:<20} : {w:+.4f}")
            active_A.append(name)

    print(f"\n🔶 Meta B (fréquence de sélection normalisée, hyperparamètres: {best_params_prod}) :")
    active_B = []
    for name, w in sorted(weights_B_dict.items(), key=lambda x: -x[1]):
        if w > TOLERANCE:
            print(f"   {_icon(name)} {name:<20} : {w:.4f}")
            active_B.append(name)

    # ✂️ Logique de Pruning Scientifique (Sans Refit)
    # Un modèle est conservé s'il est utile pour Meta A OU pour Meta B
    final_kept_models = sorted(list(set(active_A) | set(active_B)))
    rejected_models = sorted(list(set(model_names) - set(final_kept_models)))

    print(f"\n🗑️ Modèles JAMAIS retenus par l'arbitre (Poids ~0) ({len(rejected_models)}) :")
    if not rejected_models:
        print("   - Aucun ! Tous les modèles contribuent.")
    else:
        for mod in rejected_models:
            print(f"   - ❌ {mod} (Sera ignoré lors de l'inférence)")

    if len(rejected_models) > 0:
        print(f"\n{'=' * 60}")
        print("✂️ PRUNING SÉCURISÉ : ALLÈGEMENT DU MÉTA-MODÈLE POUR LA PRODUCTION")
        print(f"{'=' * 60}")
        
        # 1. On identifie les indices des colonnes à garder
        kept_indices = [model_names.index(m) for m in final_kept_models]
        
        # 2. HACK SCIENTIFIQUE SCiKit-Learn : 
        # On réduit la dimension des poids du LogisticRegression SANS faire de .fit()
        # Cela garantit la conservation stricte des scores cross-validés !
        meta_lr.coef_ = meta_lr.coef_[:, kept_indices]
        meta_lr.n_features_in_ = len(kept_indices)
        if hasattr(meta_lr, 'feature_names_in_'):
            meta_lr.feature_names_in_ = np.array(final_kept_models)
            
        # 3. Renormalisation de Meta B (pour que la somme fasse toujours exactement 1.0)
        weights_B_dict = {m: weights_B_dict.get(m, 0.0) for m in final_kept_models}
        w_sum = sum(weights_B_dict.values())
        if w_sum > 0:
            weights_B_dict = {m: w / w_sum for m, w in weights_B_dict.items()}
            
        # 4. Nettoyage des calibrateurs de base
        base_calibrators = {m: base_calibrators[m] for m in final_kept_models if m in base_calibrators}
        
        # 5. Mise à jour de la liste officielle des modèles requis
        model_names = final_kept_models
        proba_columns = [f"prob_{m}" for m in model_names]
        
        print(f"   ✅ Le fichier '.pkl' d'inférence n'exigera plus que {len(model_names)} modèles purs !")

    # ==========================================
    # 12. SAUVEGARDES ET EXPORTS
    # ==========================================
    csv_path = os.path.join(OUTPUT_DIR, 'final_hybrid_oof.csv')

    df_export = df_stack[['uid', 'is_pathologic', 'fold'] + proba_columns].copy()
    df_export['prob_meta_A'] = oof_A
    df_export['prob_meta_B'] = oof_B
    df_export['prob_final_stacking'] = oof_final
    df_export.to_csv(csv_path, index=False)
    
    print(f"\n✅ Fichier de prédictions complet sauvegardé : {csv_path}")
    print(f"🌟 SCORE FINAL (nested OOF) — AUC: {final_auc:.4f} | Log Loss: {final_loss:.4f} | ECE: {final_ece:.4f}")

    model_path = os.path.join(OUTPUT_DIR, 'meta_model_stacking.pkl')
    output_dict = {
        'model_names': model_names,
        'proba_columns': proba_columns,
        'blend_space_logit': BLEND_SPACE_LOGIT,
        'base_calibrators': base_calibrators,
        'calibration_method': [BASE_CALIBRATION_METHODS, FINAL_CALIBRATION_METHODS],
        'meta_lr': meta_lr,
        'ensemble_weights': weights_B_dict,
        'ensemble_best_params': best_params_prod,
        'blend_alpha': best_alpha,
        'final_calibrator': final_calibrator
    }
    joblib.dump(output_dict, model_path)
    print(f"✅ Méta-modèle sauvegardé : {model_path}")

    # ==========================================
    # 13. VISUALISATIONS
    # ==========================================
    print(f"\n{'=' * 60}")
    print("📊 GENERATION DES GRAPHIQUES")
    print(f"{'=' * 60}")

    plot_models = model_names
    plot_cols = proba_columns

    plt.figure(figsize=(10, 8))
    corr = df_export[plot_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=0, vmax=1, fmt=".2f",
                xticklabels=plot_models, yticklabels=plot_models)
    plt.title("Matrice de Corrélation des Modèles (Stacking)", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot_01_correlation_modeles.png'), dpi=300)
    plt.close()

    try:
        sample_neg = df_export[df_export['is_pathologic'] == 0].sample(30, random_state=42)
        sample_pos = df_export[df_export['is_pathologic'] == 1].sample(30, random_state=42)
    except ValueError:
        sample_neg = df_export[df_export['is_pathologic'] == 0]
        sample_pos = df_export[df_export['is_pathologic'] == 1]

    df_sample = pd.concat([sample_neg, sample_pos]).sort_values(by=['is_pathologic', 'prob_final_stacking'])
    heatmap_data = df_sample[plot_cols + ['prob_final_stacking']].copy()
    heatmap_data.index = df_sample.apply(
        lambda row: f"{row['uid']} [{'Patho' if row['is_pathologic'] == 1 else 'Sain'}]", axis=1)
    heatmap_data.columns = plot_models + ['ARBITRE (Stacking)']

    plt.figure(figsize=(14, 10))
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', vmin=0, vmax=1, fmt=".2f", linewidths=.5)
    plt.title("Votes individuels vs Arbitrage Final (Échantillon)", pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot_02_patients_sample.png'), dpi=300)
    plt.close()

    prob_true, prob_pred = calibration_curve(y, oof_final, n_bins=10)
    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=f'Stacking Final (ECE={final_ece:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram (Calibration Curve)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'plot_03_calibration_curve.png'), dpi=300)
    plt.close()

    print(f"✅ Visuels générés dans : {OUTPUT_DIR}")

    return final_loss, final_auc

if __name__ == '__main__':
    main()