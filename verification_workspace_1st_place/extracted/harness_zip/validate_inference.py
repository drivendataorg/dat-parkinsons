"""
Validation du pipeline d'inférence contre le master_oof_predictions.csv.

Autonome : aucune dépendance à recipe.json. Les valeurs de la recette sont
recopiées à la main depuis la sortie "RECETTE FIGEE" du notebook (bloc CONFIG
ci-dessous), et le script vérifie que le code d'inférence applique bien
CES valeurs-là.

Ce que ce script vérifie :
  1. les 100 checkpoints sont présents et indexés par (exp, fold), jamais par
     l'arithmétique exp*n_folds+fold, qui se décale dès qu'un fichier manque ;
  2. l'échantillon couvre les 100 couples (exp, fold), pas les 10 premières lignes ;
  3. l'inférence temps réel reproduit les probas BRUTES de l'OOF ;
  4. l'inférence est déterministe ;
  5. les labels du CSV correspondent à la source de vérité ;
  6. _apply_recipe applique bien la température, le clip et l'agrégation attendus.

Ce que ce script NE vérifie PAS : que la VALEUR de T soit la bonne. Ça se
décide sur l'OOF, dans le notebook de préparation.

Sortie : code retour 0 si tout passe, 1 sinon. Utilisable en garde-fou.
"""

import os
import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger('nibabel').setLevel(logging.ERROR)

# ============================== CONFIG ==============================
DATA_DIR = Path("/data2/iceberg/dat_parkinson/")
if not DATA_DIR.exists():
    DATA_DIR = Path("data")
NIFTI_DIR = DATA_DIR / "niftis"

RUN_DIR = Path("/data/HOME/sansel/dat-park/runs_siamois/mashup_internal_ensemble_full")
OOF_CSV_PATH = Path("/data/HOME/sansel/dat-park/runs_siamois/mashup_internal_ensemble_full/master_oof_predictions.csv")
MODELS_DIR = RUN_DIR
CKPT_PATTERN = "mashup_ensemble_exp_{exp}_fold_{fold}.pth"

SIAMESE_KEY = "Mashup_Internal"     # clé du siamois dans modeles_actifs
N_EXP, N_FOLDS = 20, 5

# --- recopié depuis la sortie "RECETTE FIGEE" du notebook ---
EXPECTED_AGG        = "mean"
EXPECTED_T          = 0.8835259617309265
EXPECTED_CLIP       = 0.002
EXPECTED_M_DEPLOY   = 100
EXPECTED_PREVALENCE = 0.5484581497797357
EXPECTED_LL_OOF     = 0.1983
# ------------------------------------------------------------

PER_MODEL = 2        # patients testés par couple (exp, fold) -> 200 tests

# Le modèle produit des LOGITS : c'est là que la tolérance a un sens.
# En probabilité, dp = p(1-p) * dz, donc un même écart de logit paraît 250x
# plus grand à p=0.5 qu'à p=0.99. Un seuil en probabilité fait échouer les
# patients incertains et passer les patients confiants, pour la même erreur.
#   ~1e-3  : bruit float32 (batch différent, TF32, cuDNN)  -> normal
#   ~1e-1  : divergence réelle de pipeline                 -> bug
ATOL_LOGIT = 0.02
ATOL_LOGLOSS = 1e-4          # impact toléré sur la métrique elle-même
RTOL_T = 0.01                # écart de température jugé matériel (1%)

# chemin de la source de vérité pour le contre-contrôle des labels
FOLDS_CSV_CANDIDATES = [
    DATA_DIR / "folds_siamese.csv",
    RUN_DIR / "folds_siamese.csv",
    Path("/data2/iceberg/dat_parkinson/dataset_synthetic_siamese_patches/folds_siamese.csv"),
]
# ====================================================================

import main                                                          # noqa: E402
from main import init_pipeline, predict_for_model, TabularInference  # noqa: E402
from src.preprocessing import prepare_all_inputs                     # noqa: E402


# --------------------------------------------------------------------
# 1. Indexation par (exp, fold) — jamais exp * n_folds + fold
# --------------------------------------------------------------------
def build_key_index(n_loaded):
    """Reconstruit (exp, fold) -> index dans self.models, dans le MÊME ordre
    que le constructeur, et refuse toute liste incomplète.

    Le constructeur fait `if not os.path.exists(...): continue`. Un seul
    checkpoint manquant décale tous les indices suivants et fait comparer
    silencieusement le mauvais réseau.
    """
    keys, missing = [], []
    for exp in range(N_EXP):
        for fold in range(N_FOLDS):
            p = MODELS_DIR / CKPT_PATTERN.format(exp=exp, fold=fold)
            (keys if p.exists() else missing).append((exp, fold))

    if missing:
        print(f"\n❌ {len(missing)} checkpoints manquants sur {N_EXP*N_FOLDS} : "
              f"{missing[:8]}{'...' if len(missing) > 8 else ''}")
        print("   Les indices seraient décalés. Arrêt.")
        sys.exit(1)
    if len(keys) != n_loaded:
        print(f"\n❌ {len(keys)} fichiers sur disque, {n_loaded} modèles en mémoire.")
        sys.exit(1)
    if n_loaded != EXPECTED_M_DEPLOY:
        print(f"\n❌ {n_loaded} modèles chargés mais T a été calibrée pour "
              f"M={EXPECTED_M_DEPLOY}. La température ne s'applique plus.")
        sys.exit(1)
    return {k: i for i, k in enumerate(keys)}


# --------------------------------------------------------------------
# 2. Échantillonner à travers TOUS les couples (exp, fold)
# --------------------------------------------------------------------
def stratified_sample(df, per_model, seed=0):
    """df.head(n) ne donne que exp=0/fold=0 : le seul index qui reste correct
    même quand des checkpoints manquent. C'est l'angle mort exact du trou n°1.

    Boucle explicite plutôt que groupby().apply() : depuis pandas 2.2,
    apply() exclut les colonnes de groupage du résultat (include_groups=False
    par défaut), ce qui ferait disparaître 'exp' et 'fold' de l'échantillon.
    """
    parts = [g.sample(min(per_model, len(g)), random_state=seed)
             for _, g in df.groupby(['exp', 'fold'], sort=True)]
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------
# 3. Inférence d'un seul réseau, recette désactivée
# --------------------------------------------------------------------
def predict_single_model(instance, nifti_path, idx):
    inputs = prepare_all_inputs(str(nifti_path))
    orig_models = instance.models
    orig_apply = getattr(instance, '_apply_recipe', None)
    try:
        instance.models = [orig_models[idx]]
        if orig_apply is not None:
            # np.mean explicite : `if p_array` lève une exception sur un ndarray
            instance._apply_recipe = lambda a: float(np.mean(np.asarray(a, float)))
        return float(predict_for_model(instance, inputs))
    finally:
        instance.models = orig_models
        if orig_apply is not None:
            instance._apply_recipe = orig_apply


# --------------------------------------------------------------------
# 6. La recette applique-t-elle bien les valeurs attendues ?
# --------------------------------------------------------------------
def test_recipe(instance):
    print("\n" + "=" * 78)
    print("  TEST DE LA RECETTE (agrégation + température + clip)")
    print("=" * 78)
    apply_fn = getattr(instance, '_apply_recipe', None)
    if apply_fn is None:
        print("  ⚠️  pas de _apply_recipe : test impossible")
        return False

    ok, T_deduites = True, []
    _lg = lambda p: np.log(p / (1 - p))
    for p_in in (0.90, 0.70, 0.30, 0.10):
        want = float(np.clip(1 / (1 + np.exp(-_lg(p_in) / EXPECTED_T)),
                             EXPECTED_CLIP, 1 - EXPECTED_CLIP))
        got = float(apply_fn([p_in] * EXPECTED_M_DEPLOY))
        T_deduites.append(_lg(p_in) / _lg(got))          # T réellement appliquée
        print(f"  p={p_in:.2f} x{EXPECTED_M_DEPLOY}  attendu {want:.6f}  "
              f"obtenu {got:.6f}  (T appliquée {T_deduites[-1]:.6f})")

    T_code = float(np.mean(T_deduites))
    drift = abs(T_code - EXPECTED_T) / EXPECTED_T
    print(f"\n  Température appliquée par le code : {T_code:.6f}")
    print(f"  Température de la recette         : {EXPECTED_T:.6f}")
    if drift < 1e-9:
        print("  ✓ identiques")
    elif drift < RTOL_T:
        print(f"  ⚠️  écart de {drift*100:.3f}% — sans impact mesurable sur la log loss,")
        print("     mais c'est une valeur codée en dur qui a dérivé. Remplace-la par")
        print(f"     EXPECTED_T ({EXPECTED_T!r}) pour que le code suive la recette.")
    else:
        print(f"  ✗ écart de {drift*100:.2f}% — MATÉRIEL. Le code n'applique pas la recette.")
        ok = False

    # sens de la température : T<1 doit écarter de 0.5, pas rapprocher
    if EXPECTED_T < 1 and float(apply_fn([0.90] * EXPECTED_M_DEPLOY)) <= 0.90:
        print("  ✗ T<1 doit EXTRÉMISER : 0.90 doit monter. Signe du logit inversé ?")
        ok = False

    lo = float(apply_fn([1e-9] * EXPECTED_M_DEPLOY))
    hi = float(apply_fn([1 - 1e-9] * EXPECTED_M_DEPLOY))
    good = abs(lo - EXPECTED_CLIP) < 1e-9 and abs(hi - (1 - EXPECTED_CLIP)) < 1e-9
    ok &= good
    print(f"  clip bas {lo:.6f} / haut {hi:.6f}  {'✓' if good else '✗'}")

    # l'agrégation est-elle bien une moyenne ?
    mixed = [0.2, 0.4, 0.6, 0.8] * (EXPECTED_M_DEPLOY // 4)     # moyenne = 0.5
    got = float(apply_fn(mixed))
    good = abs(got - 0.5) < 1e-6
    ok &= good
    print(f"  agrégation '{EXPECTED_AGG}'  attendu 0.500000  obtenu {got:.6f}  "
          f"{'✓' if good else '✗'}")
    if not good:
        print("     -> le code n'agrège pas comme la recette (médiane ? logit_mean ?)")
    return ok


# --------------------------------------------------------------------
def main_test():
    if not OOF_CSV_PATH.exists():
        sys.exit(f"OOF introuvable : {OOF_CSV_PATH}")

    df = pd.read_csv(OOF_CSV_PATH)
    df['uid'] = df['uid'].astype(str)
    prev = df.drop_duplicates('uid')['label'].mean()
    print(f"OOF : {len(df)} lignes | {df['uid'].nunique()} patients | "
          f"{df['exp'].nunique()} exps x {df['fold'].nunique()} folds")
    print(f"      prévalence {prev:.4f} "
          + ("✓" if abs(prev - EXPECTED_PREVALENCE) < 1e-3
             else "✗ ce CSV n'est pas celui utilisé par le notebook !"))

    arbitre, modeles_actifs = init_pipeline()
    if SIAMESE_KEY not in modeles_actifs:
        sys.exit(f"'{SIAMESE_KEY}' absent : {list(modeles_actifs)}")
    inst = modeles_actifs[SIAMESE_KEY]     # on ne teste QUE le siamois

    key2idx = build_key_index(len(inst.models))
    print(f"✓ {len(key2idx)} modèles, indexation par (exp, fold) validée")

    if any(getattr(m, 'training', False) for m in inst.models):
        sys.exit("❌ un modèle est en mode train() : BatchNorm/Dropout actifs")
    print("✓ tous les modèles en eval()")

    sample = stratified_sample(df, PER_MODEL)
    print(f"✓ échantillon : {len(sample)} lignes couvrant "
          f"{sample.groupby(['exp','fold']).ngroups} / {N_EXP*N_FOLDS} couples\n")

    res, skipped = [], []
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="validation"):
        uid, exp, fold = row['uid'], int(row['exp']), int(row['fold'])
        img = NIFTI_DIR / f"{uid}.nii.gz"
        if not img.exists():
            skipped.append(uid)                  # on compte, on n'ignore pas
            continue
        res.append(dict(uid=uid, exp=exp, fold=fold,
                        p_oof=float(row['prob']),
                        p_rt=predict_single_model(inst, img, key2idx[(exp, fold)]),
                        label=row.get('label', np.nan)))

    if not res:
        sys.exit("❌ aucun patient testé (toutes les images manquent)")

    r = pd.DataFrame(res)
    r['diff'] = r['p_rt'] - r['p_oof']
    r['adiff'] = r['diff'].abs()
    _lg = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    r['dz'] = (_lg(r['p_rt']) - _lg(r['p_oof'])).abs()

    # déterminisme
    u0 = r.iloc[0]
    idx0 = key2idx[(int(u0.exp), int(u0.fold))]
    p1 = predict_single_model(inst, NIFTI_DIR / f"{u0.uid}.nii.gz", idx0)
    p2 = predict_single_model(inst, NIFTI_DIR / f"{u0.uid}.nii.gz", idx0)
    determinist = abs(p1 - p2) < 1e-9

    # impact sur la METRIQUE, seul juge qui compte vraiment
    ll_impact = np.nan
    if r['label'].notna().all():
        y = r['label'].to_numpy().astype(int)
        f = lambda p: float(-(y * np.log(np.clip(p, 1e-15, 1)) +
                              (1 - y) * np.log(np.clip(1 - p, 1e-15, 1))).mean())
        ll_impact = f(r['p_rt'].to_numpy()) - f(r['p_oof'].to_numpy())

    print("\n" + "=" * 78)
    print("  VERDICT — PREPROCESSING")
    print("=" * 78)
    print(f"  patients testés     : {len(r)}   (images manquantes : {len(skipped)})")
    print(f"  couples couverts    : {r.groupby(['exp','fold']).ngroups} / {N_EXP*N_FOLDS}")
    print(f"  écart LOGIT max     : {r.dz.max():.3e}   (seuil {ATOL_LOGIT})")
    print(f"  écart LOGIT médian  : {r.dz.median():.3e}")
    print(f"  écart signé (logit) : {(_lg(r['p_rt']) - _lg(r['p_oof'])).mean():+.3e}"
          f"   <- un biais systématique se voit ici, pas dans le max")
    print(f"  impact log loss     : {ll_impact:+.3e}   (seuil {ATOL_LOGLOSS})")
    print(f"  déterministe        : {'OUI' if determinist else 'NON ❌'}")
    print(f"  [info] écart max en probabilité : {r.adiff.max():.3e} "
          f"— trompeur, dp = p(1-p) x dz")

    n_bad = int((r.dz > ATOL_LOGIT).sum())
    ll_bad = (not np.isnan(ll_impact)) and abs(ll_impact) > ATOL_LOGLOSS

    if n_bad or ll_bad:
        print(f"\n  ❌ {n_bad} / {len(r)} au-dessus de {ATOL_LOGIT} en logit")
        print(r.nlargest(min(8, max(n_bad, 1)), 'dz')[
            ['uid', 'exp', 'fold', 'p_oof', 'p_rt', 'dz']].to_string(index=False))
        by = r[r.dz > ATOL_LOGIT].groupby('exp').size()
        if len(by) == 1:
            print(f"\n  Écarts concentrés sur exp={by.index[0]} -> checkpoint corrompu")
            print("  ou décalage d'index. PAS un problème de preprocessing.")
        else:
            print("\n  Écarts répartis sur plusieurs exps -> preprocessing divergent.")
            print("  Pistes : sens du flip (dims=[1] vertical vs dims=[-1] horizontal),")
            print("  ordre normalisation/fade, ordre des args du forward,")
            print("  interpolation du resize du patch global.")
    else:
        print(f"\n  ✅ les {len(r)} patients concordent à {ATOL_LOGIT} près en logit.")
        if r.dz.max() > 1e-2:
            print("     (écarts ~1e-2 : plus que du float32, mais sans impact mesurable)")
        elif r.dz.max() > 1e-4:
            print("     (écarts ~1e-3 : bruit float32 normal — batch différent entre")
            print("      la validation et l'inférence, TF32, ou algorithme cuDNN)")

    # labels croisés : le contrôle qui attrape un désalignement uid/label
    truth_path = next((p for p in FOLDS_CSV_CANDIDATES if p.exists()), None)
    if truth_path is None:
        print("  labels croisés      : ❌ NON VÉRIFIÉS — aucun de ces chemins n'existe :")
        for p in FOLDS_CSV_CANDIDATES:
            print(f"                          {p}")
        print("                        Renseigne FOLDS_CSV_CANDIDATES. C'est le contrôle")
        print("                        qui attrape un désalignement uid/label.")
        labels_ok = False
    else:
        try:
            truth = pd.read_csv(truth_path)
            truth['uid'] = truth['uid'].astype(str)
            col = 'label' if 'label' in truth.columns else 'is_pathologic'
            lab = truth.set_index('uid')[col]
            seen = [(u, l) for u, l in zip(r.uid, r.label)
                    if u in lab.index and not np.isnan(l)]
            mm = [u for u, l in seen if lab[u] != l]
            labels_ok = bool(seen) and not mm
            print(f"  labels croisés      : "
                  + (f"✅ {len(seen)} cohérents ({truth_path.name})" if labels_ok
                     else f"❌ {len(mm)} incohérents {mm[:5]}" if mm
                     else "❌ aucun uid commun — mauvais fichier ?"))
        except Exception as e:
            print(f"  labels croisés      : ❌ échec ({e})")
            labels_ok = False

    ok_recipe = test_recipe(inst)

    print("\n" + "=" * 78)
    ok = (n_bad == 0) and (not ll_bad) and determinist and not skipped \
        and labels_ok and ok_recipe
    print("  " + ("✅ PIPELINE VALIDÉ — soumission possible"
                  if ok else "❌ NE PAS SOUMETTRE"))
    print(f"  rappel : LL OOF attendue {EXPECTED_LL_OOF:.4f}, "
          f"prévalence {EXPECTED_PREVALENCE:.4f}")
    print("=" * 78)
    r.to_csv('validation_report.csv', index=False)
    return ok


if __name__ == '__main__':
    sys.exit(0 if main_test() else 1)