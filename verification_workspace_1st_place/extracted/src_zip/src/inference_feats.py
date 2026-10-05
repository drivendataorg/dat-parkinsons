"""
inference_feats.py
====================
Classe d'inférence orientée objet pour le pipeline Stacking.
Prend en charge la calibration, le passage en espace logit,
et la fusion des deux Méta-Learners (ElasticNet LR + Ensemble Selection).
"""

import os
import joblib
import numpy as np
from scipy.special import logit, expit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import json
import pandas as pd
from .calibrators import *

class TabularInference:
    def __init__(self, model_name, models_dir):
        """
        Initialise un modèle tabulaire spécifique.
        Exemple : model_name = "XGBoost" cherchera "XGBoost_final.pkl"
        """
        self.model_name = model_name
        self.model_path = os.path.join(models_dir, f"{model_name}_final.pkl")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"⚠️ Poids introuvables pour {model_name} : {self.model_path}")
        
        print(f"⚙️ Chargement du modèle Tabulaire : {self.model_name}...")
        self.pipeline = joblib.load(self.model_path)

    def predict(self, features_array):
        """
        Prend le vecteur de features du preprocessing et renvoie la probabilité.
        Le StandardScaler intégré au Pipeline s'occupe de la normalisation automatiquement.
        """
        if not isinstance(features_array, np.ndarray):
            features_array = np.array(features_array)
            
        if features_array.ndim == 1:
            features_array = features_array.reshape(1, -1)
            
        # Prédiction (classe 1 = Pathologique)
        prob = self.pipeline.predict_proba(features_array)[0, 1]
        
        return float(prob)


class StackingInference:
    def __init__(self, model_path):
        """
        Initialise le méta-modèle hybride.
        Restaure les classifieurs, hyperparamètres, calibrateurs, et post-traitement.
        """
        print(f"⚙️ Chargement de l'Arbitre Hybride (Stacking) : {model_path}...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"⚠️ Méta-modèle introuvable : {model_path}")
            
        data = joblib.load(model_path)
        
        # 1. Extraction des paramètres
        self.model_names = data['model_names']
        self.blend_space_logit = data.get('blend_space_logit', True)
        self.base_calibrators = data.get('base_calibrators', {})
        
        # 2. Modèles de Stacking
        self.meta_lr = data['meta_lr']  # Meta A (ElasticNet LR)
        self.ensemble_weights = data['ensemble_weights']  # Meta B (Caruana)
        self.blend_alpha = data['blend_alpha']
        self.final_calibrator = data.get('final_calibrator', None)
        
        # 3. Restauration de la configuration de Post-Processing (Sécurisée)
        pp_config = data.get('post_processing', {})
        # self.use_clipping = pp_config.get('clipping', False)
        # clipping_data = pp_config.get('clipping_data', {}) if self.use_clipping else {}
        
        # self.t_lower = clipping_data.get('lower_threshold', 0.0)
        # self.t_upper = clipping_data.get('upper_threshold', 1.0)
        # self.eps = 1e-3  # Sécurité LogLoss stricte
        # --- Config t-SNE Correction ---
        self.use_tsne = pp_config.get('tsne', False)
        self.tsne_data = pp_config.get('tsne_data', None) if self.use_tsne else None
        
        if self.use_tsne and self.tsne_data:
            self.tsne_reducer = self.tsne_data.get('tsne')
            self.tsne_scaler = self.tsne_data.get('scaler', None)  # Scaler entraîné si disponible
            
            # Normalisation du format des bornes (liste de dictionnaires)
            raw_bounds = self.tsne_data.get('bounds', [])
            self.tsne_bounds_list = raw_bounds if isinstance(raw_bounds, list) else [raw_bounds]
        else:
            self.tsne_reducer = None
            self.tsne_scaler = None
            self.tsne_bounds_list = []

        # --- Config Threshold Clipping ---
        self.use_clipping = pp_config.get('clipping', False)
        clipping_data = pp_config.get('clipping_data', {}) if self.use_clipping else {}
        
        self.t_lower = clipping_data.get('lower_threshold', 0.0)
        self.t_upper = clipping_data.get('upper_threshold', 1.0)
        self.eps = clipping_data.get('eps', 1e-15)
        
        # Pré-calcul du vecteur de poids pour Meta B
        self.weights_B_array = np.array([self.ensemble_weights.get(name, 0.0) for name in self.model_names])

        print(f"   -> {len(self.model_names)} modèles de base attendus.")
        print(f"   -> Blend Alpha configuré à : {self.blend_alpha:.2f} (1.0 = Meta A pur, 0.0 = Meta B pur)")
        
        if self.use_clipping:
            print(f"   -> Post-processing Clipping : ACTIF (t_lower = {self.t_lower:.4f}, t_upper = {self.t_upper:.4f}, eps = {self.eps})")

    def _safe_logit(self, p, eps=1e-6):
        """Transformation logit sécurisée pour éviter les infinis."""
        return logit(np.clip(p, eps, 1 - eps))

    def _apply_calibrator(self, cal, prob):
        """
        Applique le calibrateur approprié à une probabilité.
        """
        if cal is None:
            return prob

        is_scalar = np.isscalar(prob)
        prob_arr = np.atleast_1d(prob).astype(float)

        # -------------------------------------------------------------------------
        # CAS 1 : Restauration d'un dictionnaire sérialisé
        # -------------------------------------------------------------------------
        if isinstance(cal, dict) and 'calibrator' in cal:
            cal_type = cal['calibrator']

            if cal_type == 'temperature':
                obj = TemperatureScaler()
                obj.T = cal['T']
                res = obj.predict_proba(prob_arr)[:, 1]

            elif cal_type == 'vector_temp':
                obj = VectorTemperatureScaler()
                obj.a, obj.b = cal['a'], cal['b']
                res = obj.predict_proba(prob_arr)[:, 1]

            elif cal_type == 'platt':
                obj = PlattScaler()
                obj.model = cal['model']
                res = obj.predict_proba(prob_arr)[:, 1]

            elif cal_type == 'isotonic':
                obj = IsotonicScaler()
                obj.model = cal['model']
                res = obj.predict_proba(prob_arr)[:, 1]

            elif cal_type == 'label_smoothing':
                obj = LabelSmoother(auto_tune=False)
                obj.epsilon = cal['epsilon']
                res = obj.predict_proba(prob_arr)[:, 1]

            elif cal_type == 'beta':
                obj = BetaScaler()
                obj.lr = cal['lr']
                res = obj.predict_proba(prob_arr)[:, 1]

            elif cal_type == 'spline':
                obj = SplineScaler()
                obj.spline = cal['spline']
                obj.lr = cal['lr']
                res = obj.predict_proba(prob_arr)[:, 1]

            elif cal_type == 'gp':
                obj = GPScaler()
                obj.gpc = cal['kernel_params']
                res = obj.predict_proba(prob_arr)[:, 1]

            else:
                return prob

        # -------------------------------------------------------------------------
        # CAS 2 : Instance directe
        # -------------------------------------------------------------------------
        elif hasattr(cal, 'predict_proba'):
            res = cal.predict_proba(prob_arr)[:, 1]

        elif isinstance(cal, IsotonicRegression):
            res = cal.predict(prob_arr)
            res = np.clip(res, 1e-6, 1 - 1e-6)

        elif isinstance(cal, LogisticRegression):
            logits = safe_logit(prob_arr).reshape(-1, 1)
            res = cal.predict_proba(logits)[:, 1]

        else:
            return prob

        return float(res[0]) if is_scalar else res

    def predict(self, probs_dict):
        """
        probs_dict : dictionnaire contenant les prédictions des modèles de base.
        """
        # 1. Normalisation des clés (on enlève 'prob_' si présent)
        clean_probs = {k.replace('prob_', ''): v for k, v in probs_dict.items()}
        
        # 2. Vérification de sécurité stricte
        missing_models = [name for name in self.model_names if name not in clean_probs]
        if missing_models:
            raise ValueError(f"❌ Impossible de statuer ! Il manque les prédictions de : {missing_models}")
            
        # 3. Extraction brute dans l'ORDRE EXACT du Grid Search
        X_raw = np.array([clean_probs[name] for name in self.model_names])
        
        # 4. Calibration individuelle des modèles de base
        X_calibrated = np.zeros_like(X_raw)
        for i, name in enumerate(self.model_names):
            cal = self.base_calibrators.get(name)
            X_calibrated[i] = self._apply_calibrator(cal, X_raw[i])
            
        # 5. Passage dans l'espace Logit
        if self.blend_space_logit:
            X_logit = self._safe_logit(X_calibrated)
        else:
            X_logit = X_calibrated
            
        # Reshape (1, N_models)
        X_logit_2d = X_logit.reshape(1, -1)
        
        # 6. MÉTA-LEARNER A : ElasticNet Regression
        prob_A = self.meta_lr.predict_proba(X_logit_2d)[0, 1]
        
        # 7. MÉTA-LEARNER B : Ensemble Selection (Caruana)
        prob_B = expit(np.dot(X_logit, self.weights_B_array))
        
        # 8. FUSION A + B
        alpha = self.blend_alpha
        logit_A = self._safe_logit(prob_A)
        logit_B = self._safe_logit(prob_B)
        
        logit_combined = alpha * logit_A + (1.0 - alpha) * logit_B
        prob_combined = expit(logit_combined)
        
        # 9. Calibration finale
        final_prob = self._apply_calibrator(self.final_calibrator, prob_combined)
        
        return float(final_prob)

    def post_process(self, prob: float, features=None) -> float:
        """
        Applique les règles de post-traitement (Correction UMAP + Clipping de seuils)
        pour une prédiction individuelle.

        Arguments:
            prob (float): Probabilité brute issue de `predict()`.
            features (np.ndarray | pd.DataFrame | dict | None): Vecteur de features tabulaires (30,)
                                                                nécessaire pour la projection UMAP.

        Returns:
            float: Probabilité finale post-traitée.
        """
        p = float(prob)

        # ---------------------------------------------------------------------
        # 1. CORRECTION t-SNE (Si active et si les features sont fournies)
        # ---------------------------------------------------------------------
        if self.use_tsne and self.tsne_reducer is not None and features is not None:
            # Conversion des features en tableau 2D NumPy (1, N)
            if isinstance(features, (pd.DataFrame, pd.Series)):
                feat_array = features.values
            else:
                feat_array = np.array(features)

            if feat_array.ndim == 1:
                feat_array = feat_array.reshape(1, -1)

            # Nettoyage / Imputation
            feat_array = np.nan_to_num(feat_array, nan=0.0)

            # Normalisation (Scaler entraîné prioritaire)
            if self.tsne_scaler is not None:
                X_scaled = self.tsne_scaler.transform(feat_array)
            else:
                X_scaled = StandardScaler().fit_transform(feat_array)

            # Projection t-SNE -> Coordonnées (x, y) du sample
            X_tsne = self.tsne_reducer.transform(X_scaled)
            tsne_x, tsne_y = X_tsne[0, 0], X_tsne[0, 1]

            # Vérification si le point tombe dans une zone d'incertitude/erreur
            for bounds_config in self.tsne_bounds_list:
                bounds = bounds_config['bounds'] if isinstance(bounds_config, dict) and 'bounds' in bounds_config else bounds_config

                if (bounds['tsne1_min'] <= tsne_x <= bounds['tsne1_max'] and 
                    bounds['tsne2_min'] <= tsne_y <= bounds['tsne2_max']):
                    
                    correction_val = bounds_config.get('correction_value', 0.5) if isinstance(bounds_config, dict) else 0.5
                    p = float(correction_val)
                    # print(f"[t-SNE] Point in region ({tsne_x:.2f}, {tsne_y:.2f}) -> Overridden {prob:.2f} to {p}")
                    break  # Première zone matchée = correction appliquée

        # ---------------------------------------------------------------------
        # 2. THRESHOLD CLIPPING (Dual Log-Loss Optimization)
        # ---------------------------------------------------------------------
        if self.use_clipping:
            return float(np.clip(prob, self.eps, 1.0 - self.eps))
        
        return float(prob)

# ==========================================
# WRAPPER POUR MODÈLE SOLO (Si USE_STACKING = False)
# ==========================================
class SoloModelInference:
    def __init__(self, target_model_name, use_clipping):
        self.target_model_name = target_model_name
        # On fait croire au reste du script qu'on a besoin que de ce modèle
        self.model_names = [f"prob_{target_model_name}"]
        self.eps = 1e-3  # Sécurité LogLoss stricte
        self.use_clipping = use_clipping

    def predict(self, avis_experts):
        # Récupère l'avis du modèle cible
        if self.target_model_name in avis_experts:
            return avis_experts[self.target_model_name]
        # Fallback si le nom exact diffère légèrement
        for k, v in avis_experts.items():
            if self.target_model_name.lower() in k.lower():
                return v
        raise ValueError(f"Modèle solo '{self.target_model_name}' introuvable dans les avis d'experts : {list(avis_experts.keys())}")

    def post_process(self, prob, tabular_features):
        if self.use_clipping:
            return float(np.clip(prob, self.eps, 1.0 - self.eps))
        
        return float(prob)

class XGBoostEnsembleInference:
    def __init__(self, models_dir, features_list_path, n_exp=20, k_splits=5):
        """
        Initialise l'ensemble XGBoost avec filtrage strict des features.
        
        Args:
            models_dir (str): Dossier contenant les fichiers xgb_exp_X_fold_Y.json
            features_list_path (str): Chemin vers le JSON contenant la liste des features (Top K)
            n_exp (int): Nombre d'expériences. Par défaut 20 (pour tes 100 modèles).
            k_splits (int): Nombre de folds par expérience. Par défaut 5.
        """
        self.models = []
        self.model_name = "XGBoostEnsemble"
        expected_models = n_exp * k_splits
        
        print(f"⚙️ Chargement de la famille {self.model_name} ({expected_models} modèles) depuis : {models_dir}...")
        
        # 1. Chargement de la liste stricte des features
        if not os.path.exists(features_list_path):
            raise FileNotFoundError(f"⚠️ Fichier de features introuvable : {features_list_path}")
            
        with open(features_list_path, 'r') as f:
            self.expected_features = json.load(f)
            
        print(f"  -> Filtrage strict activé : {len(self.expected_features)} features requises.")

        # 2. Chargement des modèles
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"⚠️ Dossier modèles introuvable : {models_dir}")

        for exp in range(n_exp):
            for fold in range(k_splits):
                model_path = os.path.join(models_dir, f"xgb_exp_{exp}_fold_{fold}.json")
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"⚠️ Modèle introuvable : {model_path}")
                
                model = xgb.XGBClassifier()
                model.load_model(model_path)
                model.set_params(device='cpu')  # Force le CPU pour l'inférence
                self.models.append(model)
                
        print(f"  ✅ {len(self.models)} modèles XGBoost chargés avec succès.")

    def predict(self, features):
        """
        Filtre le dictionnaire/DataFrame de features entrant, vérifie l'ordre,
        puis effectue l'inférence moyennée.
        """
        # 1. Conversion en DataFrame pour faciliter le filtrage par nom de colonne
        if isinstance(features, dict):
            df_features = pd.DataFrame([features])
        elif isinstance(features, pd.Series):
            df_features = features.to_frame().T
        elif isinstance(features, pd.DataFrame):
            df_features = features.copy()
        else:
            raise TypeError("Les features doivent être fournies sous forme de dict, pd.Series ou pd.DataFrame.")
            
        # 2. Sécurité : Vérification de la présence de toutes les features requises
        missing = [f for f in self.expected_features if f not in df_features.columns]
        if missing:
            raise ValueError(f"❌ Impossible de prédire, {len(missing)} features manquantes pour XGBoost. Ex: {missing[:3]}")
            
        # 3. Filtrage et ordonnancement strict selon l'entraînement
        model_expected_features = self.models[0].get_booster().feature_names
        df_filtered = df_features[model_expected_features]
        
        # 4. Inférence sur l'ensemble
        preds = []
        for model in self.models:
            prob = model.predict_proba(df_filtered)[0, 1]
            preds.append(prob)
            
        return float(np.mean(preds))

class LogitFamilyStackingInference:
    def __init__(self, model_names, temperature=0.9784, clip_bounds=(0.01, 0.99)):
        """
        Nouvelle méthode d'arbitrage (Stacking) par moyenne des logits, 
        température et clipping strict.
        """
        self.model_names = model_names
        self.temperature = temperature
        self.clip_min, self.clip_max = clip_bounds
        
        print(f"⚙️ Initialisation Arbitre LogitFamilyStacking : T={self.temperature}, Clip=[{self.clip_min}, {self.clip_max}]")
        print(f"   -> Familles attendues (poids égaux) : {self.model_names}")

    def predict(self, avis_experts):
        """
        avis_experts : dict contenant {nom_modele: proba_moyenne_du_modele}
        """
        logits = []
        for name in self.model_names:
            # Extraction sécurisée
            p = avis_experts.get(name)
            if p is None:
                p = avis_experts.get(f"prob_{name}")
            if p is None:
                raise ValueError(f"❌ Prédiction manquante pour la famille : {name}")
            
            # 1. Les probabilités entrantes sont DÉJÀ moyennées en amont 
            # par les classes d'inférence respectives (ex: XGBoostEnsembleInference, etc.)
            
            # Sécurité pour éviter log(0)
            p_safe = np.clip(p, 1e-7, 1.0 - 1e-7)
            
            # 2. Passage en logit
            z_i = logit(p_safe)
            logits.append(z_i)

        # 3. Moyenne des logits (poids égaux)
        z_mean = np.mean(logits)

        # 4. Application de la température
        p_temp = expit(z_mean / self.temperature)

        # 5. Clipping final EN DERNIER
        p_final = np.clip(p_temp, self.clip_min, self.clip_max)

        return float(p_final)

    def post_process(self, prob, features=None):
        """
        Le post-processing (clipping) est déjà fait à l'étape 5 dans predict().
        On renvoie la probabilité telle quelle pour respecter l'interface du pipeline.
        """
        return float(prob)