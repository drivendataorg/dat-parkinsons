import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logit, expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import SplineTransformer
from sklearn.metrics import log_loss

def safe_logit(p, eps=1e-6):
    p_clipped = np.clip(p, eps, 1 - eps)
    return logit(p_clipped)

class TemperatureScaler(BaseEstimator, ClassifierMixin):
    """Calibre des probabilités via un unique scalaire T > 0 sur l'espace logit."""
    def __init__(self):
        self.T = 1.0

    def fit(self, probs, y_true):
        logits = safe_logit(probs)
        
        def loss_func(T):
            t_val = max(T[0], 1e-3)
            scaled_logits = logits / t_val
            p = np.clip(expit(scaled_logits), 1e-15, 1 - 1e-15)
            return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

        res = minimize(loss_func, x0=[1.0], bounds=[(1e-2, 10.0)], method='L-BFGS-B')
        self.T = float(res.x[0])
        return self

    def predict_proba(self, probs):
        logits = safe_logit(probs)
        scaled_logits = logits / self.T
        p = np.clip(expit(scaled_logits), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, probs):
        return (self.predict_proba(probs)[:, 1] >= 0.5).astype(int)

    def __reduce__(self):
        return (dict, ({'calibrator': 'temperature', 'T': float(self.T)},))


class VectorTemperatureScaler(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.a = 0.0
        self.b = 1.0

    def fit(self, probs, y_true):
        logits = safe_logit(probs)
        def loss_func(params):
            a, b = params
            t_val = np.maximum(a * np.abs(logits) + b, 1e-2)
            scaled_logits = logits / t_val
            p = np.clip(expit(scaled_logits), 1e-15, 1 - 1e-15)
            return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

        res = minimize(loss_func, x0=[0.0, 1.0], bounds=[(-2.0, 2.0), (1e-2, 10.0)], method='L-BFGS-B')
        self.a, self.b = float(res.x[0]), float(res.x[1])
        return self

    def predict_proba(self, probs):
        logits = safe_logit(probs)
        t_val = np.maximum(self.a * np.abs(logits) + self.b, 1e-2)
        scaled_logits = logits / t_val
        p = np.clip(expit(scaled_logits), 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, probs):
        return (self.predict_proba(probs)[:, 1] >= 0.5).astype(int)

    def __reduce__(self):
        return (dict, ({'calibrator': 'vector_temp', 'a': float(self.a), 'b': float(self.b)},))


class PlattScaler(BaseEstimator, ClassifierMixin):
    """Régression Logistique (Platt Scaling) sur espace logit."""
    def __init__(self, max_iter=1000):
        self.max_iter = max_iter
        self.model = LogisticRegression(max_iter=self.max_iter)

    def fit(self, probs, y_true):
        self.model.fit(safe_logit(probs).reshape(-1, 1), y_true)
        return self

    def predict_proba(self, probs):
        return self.model.predict_proba(safe_logit(probs).reshape(-1, 1))

    def __reduce__(self):
        return (dict, ({'calibrator': 'platt', 'model': self.model},))


class IsotonicScaler(BaseEstimator, ClassifierMixin):
    """Wrapper Scikit-Learn IsotonicRegression pour matcher l'interface predict_proba."""
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip')

    def fit(self, probs, y_true):
        self.model.fit(probs, y_true)
        return self

    def predict_proba(self, probs):
        p = self.model.predict(probs)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def __reduce__(self):
        return (dict, ({'calibrator': 'isotonic', 'model': self.model},))


class LabelSmoother(BaseEstimator, ClassifierMixin):
    """
    Calibrateur par lissage d'extremites (Label / Boundary Smoothing).
    Compatible avec la structure Scikit-Learn (fit / predict_proba).
    """
    def __init__(self, epsilon=1e-3, auto_tune=True):
        self.epsilon = epsilon
        self.auto_tune = auto_tune

    def fit(self, probs, y_true):
        probs = np.clip(np.asarray(probs, dtype=float), 1e-15, 1 - 1e-15)
        
        if self.auto_tune:
            def loss_func(eps):
                eps_val = np.clip(eps, 1e-6, 0.1)
                p_smooth = probs * (1.0 - eps_val) + 0.5 * eps_val
                p_smooth = np.clip(p_smooth, 1e-15, 1 - 1e-15)
                return log_loss(y_true, p_smooth)

            # Optimisation scalaire de epsilon entre 1e-5 et 0.05
            res = minimize_scalar(loss_func, bounds=(1e-5, 0.05), method='bounded')
            self.epsilon = float(res.x)
            
        return self

    def predict_proba(self, probs):
        probs = np.clip(np.asarray(probs, dtype=float), 1e-15, 1 - 1e-15)
        p_smooth = probs * (1.0 - self.epsilon) + 0.5 * self.epsilon
        p_smooth = np.clip(p_smooth, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p_smooth, p_smooth])

    def predict(self, probs):
        return (self.predict_proba(probs)[:, 1] >= 0.5).astype(int)

    def __reduce__(self):
        """Pickle / Joblib compatibility."""
        return (dict, ({'calibrator': 'label_smoothing', 'epsilon': float(self.epsilon)},))

class BetaScaler(BaseEstimator, ClassifierMixin):
    """
    Beta Calibration (Kull et al., 2017).
    Ajuste une Régression Logistique sur [ln(p), -ln(1-p)].
    """
    def __init__(self, max_iter=1000):
        self.max_iter = max_iter
        self.lr = LogisticRegression(max_iter=self.max_iter, solver='lbfgs')

    def fit(self, probs, y_true):
        eps = 1e-12
        p_clipped = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
        
        # Transformation Beta : features = [ln(p), -ln(1-p)]
        X_trans = np.column_stack([
            np.log(p_clipped),
            -np.log(1 - p_clipped)
        ])
        
        self.lr.fit(X_trans, y_true)
        return self

    def predict_proba(self, probs):
        eps = 1e-12
        p_clipped = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
        
        X_trans = np.column_stack([
            np.log(p_clipped),
            -np.log(1 - p_clipped)
        ])
        
        return self.lr.predict_proba(X_trans)

    def predict(self, probs):
        return (self.predict_proba(probs)[:, 1] >= 0.5).astype(int)

    def __reduce__(self):
        return (dict, ({'calibrator': 'beta', 'lr': self.lr},))


class SplineScaler(BaseEstimator, ClassifierMixin):
    """
    Spline / GAM Calibration.
    Utilise des B-Splines suivies d'une Régression Logistique pour lisser la courbe de calibration.
    """
    def __init__(self, n_knots=5, degree=3, C=1.0, max_iter=1000):
        self.n_knots = n_knots
        self.degree = degree
        self.C = C
        self.max_iter = max_iter
        
        self.spline = SplineTransformer(n_knots=self.n_knots, degree=self.degree, include_bias=False)
        self.lr = LogisticRegression(C=self.C, max_iter=self.max_iter, solver='lbfgs')

    def fit(self, probs, y_true):
        p_vec = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6).reshape(-1, 1)
        
        # 1. Génération des bases B-splines
        X_spline = self.spline.fit_transform(p_vec)
        
        # 2. Ajustement de la régression logistique
        self.lr.fit(X_spline, y_true)
        return self

    def predict_proba(self, probs):
        p_vec = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6).reshape(-1, 1)
        X_spline = self.spline.transform(p_vec)
        return self.lr.predict_proba(X_spline)

    def predict(self, probs):
        return (self.predict_proba(probs)[:, 1] >= 0.5).astype(int)

    def __reduce__(self):
        return (dict, ({'calibrator': 'spline', 'spline': self.spline, 'lr': self.lr},))


class GPScaler(BaseEstimator, ClassifierMixin):
    def __init__(self, max_iter_predict=100):
        self.max_iter_predict = max_iter_predict
        kernel = C(1.0, (1e-2, 1e2)) * RBF(1.0, (1e-2, 1e2))
        self.gpc = GaussianProcessClassifier(kernel=kernel, max_iter_predict=self.max_iter_predict, random_state=42)

    def fit(self, probs, y_true):
        p_vec = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6).reshape(-1, 1)
        self.gpc.fit(p_vec, y_true)
        return self

    def predict_proba(self, probs):
        p_vec = np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6).reshape(-1, 1)
        p = self.gpc.predict_proba(p_vec)[:, 1]
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, probs):
        return (self.predict_proba(probs)[:, 1] >= 0.5).astype(int)

    def __reduce__(self):
        return (dict, ({
            'calibrator': 'gp',
            'kernel_params': self.gpc
        },))