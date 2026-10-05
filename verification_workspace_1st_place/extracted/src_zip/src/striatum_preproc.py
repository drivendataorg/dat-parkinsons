"""
striatum_preproc.py
===================
Applique la normalisation GLOBALE du notebook a un patient isole.

Chaine, dans l'ordre exact du notebook :
    centroides -> log1p (queues lourdes) -> imputation mediane -> StandardScaler

Artefacts attendus dans WEIGHTS_DIR (exportes par le notebook) :
    scaler_production.joblib
    medians_production.json
    feature_cols_production.json      <- a ajouter, cf. NOTEBOOK_PATCH
    spine_refs.json
"""
import os, json
import numpy as np
import joblib

IMG_H = IMG_W = 32
HEAVY_TAIL_KEYWORDS = ['area', 'sum_int', 'complexity']
LOG_EXCLUDE = ['ratio', 'frac', 'AI', 'detected']


def _normalize_centroids(feats, img_h=IMG_H, img_w=IMG_W):
    out = dict(feats)
    for side in ('L_', 'R_'):
        for region in ('head_', 'tail_'):
            for axis, dim in (('x', img_w), ('y', img_h)):
                c = f'{side}{region}centroid_{axis}'
                v = out.get(c)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    out[c] = (v - dim / 2.0) / (dim / 2.0)
    return out


def _apply_log1p(feats, feature_cols):
    """Les colonnes cibles sont celles qui portent le suffixe _log dans le CSV.
    Toutes ces grandeurs sont >= 0 par construction (comptages, sommes
    d'intensites clippees, perim^2/aire, valeurs absolues) : la branche
    `log1p(col - min_val)` du notebook n'est jamais empruntee, donc la
    transformation est SANS ETAT et reproductible patient par patient."""
    out = dict(feats)
    for col in [c for c in feature_cols if c.endswith('_log')]:
        src = col[:-4]
        v = out.pop(src, np.nan)
        out[col] = (np.nan if v is None or (isinstance(v, float) and np.isnan(v))
                    else float(np.log1p(max(float(v), 0.0))))
    return out


class StriatumPreproc:
    """Charge une fois, applique a chaque patient."""

    # Colonnes quasi constantes : leur scale_ minuscule amplifie le bruit
    # d'arrondi par plusieurs ordres de grandeur. On borne les z-scores plutot
    # que d'injecter du bruit amplifie dans le reseau.
    DEGENERATE_SCALE = 1e-6
    CLIP = 20.0

    def __init__(self, weights_dir, clip=CLIP):
        self.clip = clip
        self.scaler = joblib.load(os.path.join(weights_dir, 'scaler_production.joblib'))
        with open(os.path.join(weights_dir, 'medians_production.json')) as f:
            self.medians = json.load(f)

        cols_path = os.path.join(weights_dir, 'feature_cols_production.json')
        if os.path.exists(cols_path):
            with open(cols_path) as f:
                self.feature_cols = json.load(f)
        else:
            # Repli : l'ordre des cles du JSON des medianes est celui du fit.
            self.feature_cols = list(self.medians.keys())
            print("  ⚠️ feature_cols_production.json absent, ordre déduit des médianes.")

        n = getattr(self.scaler, 'n_features_in_', len(self.feature_cols))
        if n != len(self.feature_cols):
            raise ValueError(
                f"Le scaler attend {n} colonnes, {len(self.feature_cols)} fournies.\n"
                f"   Exportez feature_cols_production.json depuis le notebook.")

        with open(os.path.join(weights_dir, 'spine_refs.json')) as f:
            meta = json.load(f)
        self.spine_refs = meta['refs']
        self.spine_len = int(meta['spine_len'])
        self.head_side = meta.get('head_side', 'right')
        self.normalize_spine_by_bg = bool(meta.get('normalize_by_bg', True))
        degen = [c for c, s in zip(self.feature_cols, self.scaler.scale_)
                 if s < self.DEGENERATE_SCALE]
        if degen:
            print(f"  ⚠️ {len(degen)} colonnes quasi constantes (scale_ < "
                  f"{self.DEGENERATE_SCALE:g}) : {degen[:5]}")
            print("     Elles ne portent aucune information et leur z-score amplifie")
            print("     le bruit d'arrondi. Retirez-les du notebook (cf. NOTEBOOK_PATCH).")
        print(f"  ✅ preproc : {len(self.feature_cols)} colonnes | "
              f"spine_len={self.spine_len} | refs={self.spine_refs}")

    def scalars(self, raw_feats):
        """features BRUTES (dict) -> vecteur (n_features,) normalise."""
        f = _apply_log1p(_normalize_centroids(raw_feats), self.feature_cols)
        missing = [c for c in self.feature_cols if c not in f]
        if missing:
            # Les patients sans striatum detecte n'ont que la cle *_detected :
            # le notebook les remplit en NaN via pandas, on fait pareil.
            for c in missing:
                f[c] = np.nan
        x = np.array([f[c] if f[c] is not None else np.nan
                      for c in self.feature_cols], dtype=np.float64)
        med = np.array([self.medians.get(c, 0.0) for c in self.feature_cols])
        bad = ~np.isfinite(x)
        x[bad] = med[bad]
        # mean_/scale_ directement : evite le warning sklearn sur les noms de
        # colonnes et rend la transformation independante de pandas.
        z = (x - self.scaler.mean_) / self.scaler.scale_
        z = np.clip(z, -self.clip, self.clip)   # borne les colonnes degenerees
        return np.nan_to_num(z, nan=0., posinf=0., neginf=0.).astype(np.float32)


# ============================================================ singleton
# `build_twophase_data` est appele depuis preprocessing.py, qui ne connait pas
# WEIGHTS_DIR ; l'ensemble, lui, le connait. On partage donc une instance unique
# plutot que d'en charger deux (risque de divergence si les chemins different).
_PREPROC = None


def init_preproc(weights_dir, **kw):
    """Appele UNE fois au demarrage, par TwoPhaseMeanEnsemble.__init__."""
    global _PREPROC
    if _PREPROC is None:
        _PREPROC = StriatumPreproc(weights_dir, **kw)
    return _PREPROC


def get_preproc(required=True):
    """Recupere l'instance. `required=False` renvoie None si Two-Phase n'est pas
    charge, pour que preprocessing.py n'echoue pas quand on tourne un autre modele."""
    if _PREPROC is None and required:
        raise RuntimeError(
            "StriatumPreproc non initialisé.\n"
            "   init_preproc(WEIGHTS_DIR) doit être appelé avant prepare_all_inputs(),\n"
            "   ce que fait TwoPhaseMeanEnsemble.__init__.")
    return _PREPROC


def reset_preproc():
    """Utile en test uniquement."""
    global _PREPROC
    _PREPROC = None