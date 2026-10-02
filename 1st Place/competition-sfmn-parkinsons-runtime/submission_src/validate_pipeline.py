"""
validate_pipeline.py
====================
Validation complète du pipeline de soumission, de bout en bout.

Neuf phases, chacune bloquante :

  0. Configuration        — MODE_INFERENCE, familles, constantes de la recette
  1. OOF de référence     — chargement, alignement, mêmes splits, mêmes labels
  2. Modèles chargés      — 100 par famille, index (exp, fold) -> position vérifié
  3. Inputs               — clés, formes, dtypes, plages, NaN, déterminisme
  4. Comparaison appariée — chaque réseau isolé vs sa ligne d'OOF (le cœur)
  5. Arbitre              — la recette recodée en numpy doit donner le même nombre
  6. Bout en bout         — l'arbitre rejoué sur les 1362 patients de l'OOF
  7. Soumission           — format, bornes, identifiants
  8. Chronométrage        — extrapolation à 600 patients

Sortie : code retour 0 si tout passe, 1 sinon. Utilisable en garde-fou.
"""

import os
import sys
import time
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit as sp_logit, expit
from tqdm.auto import tqdm
from omegaconf import OmegaConf

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
logging.getLogger('nibabel').setLevel(logging.ERROR)

# ============================== CONFIG ==============================
cfg = OmegaConf.merge(OmegaConf.load("0_configs/default.yaml"))

DATA_DIR    = cfg.paths.niftis
if not DATA_DIR.exists():
    DATA_DIR = Path("data")
NIFTI_DIR = DATA_DIR / "niftis"
SUBMISSION_FORMAT = DATA_DIR / "submission_format.csv"     # facultatif

BASE = '.'

# nom lisible -> (fichier OOF, clé dans modeles_actifs)
FAMILIES = {
    '3br_mean_logaux':       (f'{BASE}/4_siamese_training/results/siamese_network/master_oof_predictions.csv',      'Mashup_Internal'),
}

N_EXP, N_FOLDS = 20, 5
M_DEPLOY = N_EXP * N_FOLDS                 # 100 réseaux par famille

# recette attendue — doit correspondre à LogitFamilyStackingInference
EXPECTED_T = 1.094
EXPECTED_CLIP = (0.01, 0.99)

N_PATIENTS = 100         # patients échantillonnés pour la phase 4
N_MODELS_PER_FAMILY = 5  # réseaux testés par famille et par patient -> 240 comparaisons

# Le modèle sort des LOGITS : c'est là que la tolérance a un sens.
# dp = p(1-p)*dz, donc un même écart de logit paraît 250x plus grand à p=0.5
# qu'à p=0.99. Un seuil en probabilité ferait échouer les patients incertains
# et passer les patients confiants, pour la même erreur numérique.
#   ~1e-3 : bruit float32 (batch, TF32, algo cuDNN)  -> normal
#   ~1e-1 : divergence réelle de pipeline            -> bug
ATOL_LOGIT = 0.02
ATOL_LOGLOSS = 1e-4
SEED = 42
# ====================================================================

rng = np.random.default_rng(SEED)
EPS = 1e-7
FAILURES = []


def fail(phase, msg):
    FAILURES.append(f"[{phase}] {msg}")
    print(f"  ❌ {msg}")


def ok(msg):
    print(f"  ✓ {msg}")


def head(n, title):
    print("\n" + "=" * 78)
    print(f"  PHASE {n} — {title}")
    print("=" * 78)


clip_p = lambda p: np.clip(np.asarray(p, float), EPS, 1 - EPS)
lgt = lambda p: np.log(clip_p(p) / (1 - clip_p(p)))
ll_vec = lambda y, p: -(y * np.log(clip_p(p)) + (1 - y) * np.log(1 - clip_p(p)))


# ====================================================================
# PHASE 0 — configuration
# ====================================================================
head(0, "CONFIGURATION")

import main                                                     # noqa: E402
from main import init_pipeline, predict_for_model               # noqa: E402
from src.preprocessing import prepare_all_inputs                # noqa: E402

mode = getattr(main, 'MODE_INFERENCE', None)
if mode == "FAMILY_STACKING":
    ok(f"MODE_INFERENCE = {mode}")
else:
    fail("P0", f"MODE_INFERENCE = {mode!r}, attendu 'FAMILY_STACKING'")

declared = list(getattr(main, 'FAMILY_MODELS', []))
expected_keys = [k for _, k in FAMILIES.values()]
if set(declared) == set(expected_keys):
    ok(f"FAMILY_MODELS = {declared}")
else:
    fail("P0", f"FAMILY_MODELS {declared} != familles validées {expected_keys}")

print(f"\n  Recette attendue : T = {EXPECTED_T}  |  clip = {EXPECTED_CLIP}")
print(f"  -> T(M={M_DEPLOY}) implicite = {EXPECTED_T/1.245:.4f}"
      f"  |  T(M={N_EXP}) implicite = {EXPECTED_T/1.245/0.99515:.4f}")
print("     (facteur 1.245 = correction de décalage mesurée sur le leaderboard)")


# ====================================================================
# PHASE 1 — OOF de référence
# ====================================================================
head(1, "OOF DE RÉFÉRENCE")

OOF, sets_uid = {}, {}
for name, (path, _) in FAMILIES.items():
    p = Path(path)
    if not p.exists():
        fail("P1", f"{name} : fichier introuvable {p}")
        continue
    d = pd.read_csv(p)
    d['uid'] = d['uid'].astype(str)
    missing_cols = {'uid', 'label', 'prob', 'exp', 'fold'} - set(d.columns)
    if missing_cols:
        fail("P1", f"{name} : colonnes manquantes {missing_cols}")
        continue
    OOF[name] = d
    sets_uid[name] = set(d.uid)
    print(f"  {name:<24} {len(d):>7} lignes | {d.uid.nunique():>5} patients | "
          f"{d.exp.nunique()} exps x {d.fold.nunique()} folds")

if len(OOF) != len(FAMILIES):
    print("\n  Arrêt : au moins un OOF est illisible.")
    sys.exit(1)

# --- patients communs ---------------------------------------------------
COMMON = set.intersection(*sets_uid.values())
UNION = set.union(*sets_uid.values())
print(f"\n  {len(UNION)} patients au total, {len(COMMON)} communs")
for name, sv in sets_uid.items():
    miss = UNION - sv
    if miss:
        print(f"  ⚠️  {name} : {len(miss)} absent(s) -> {sorted(miss)[:5]}")
if len(COMMON) < len(UNION):
    print("     -> comparaison restreinte aux patients communs "
          "(ce n'est PAS un problème de splits)")
OOF = {k: v[v.uid.isin(COMMON)].copy() for k, v in OOF.items()}

# --- mêmes splits, mêmes labels ----------------------------------------
ref_name = list(OOF)[0]
key_ref = (OOF[ref_name][['uid', 'exp', 'fold']].drop_duplicates()
           .sort_values(['uid', 'exp']).reset_index(drop=True))
lab_ref = (OOF[ref_name].drop_duplicates('uid').set_index('uid')['label']
           .sort_index().astype(float))

print()
for name, d in OOF.items():
    k = (d[['uid', 'exp', 'fold']].drop_duplicates()
         .sort_values(['uid', 'exp']).reset_index(drop=True))
    l = d.drop_duplicates('uid').set_index('uid')['label'].sort_index().astype(float)
    same_split = k.equals(key_ref)
    # .astype(float) des deux côtés : int64 vs float64 est un faux positif
    same_lab = bool((l == lab_ref).all()) and len(l) == len(lab_ref)
    no_dup = bool((d.groupby(['uid', 'exp']).size() == 1).all())
    in_range = bool(d.prob.between(0, 1).all()) and not d.prob.isna().any()
    if same_split and same_lab and no_dup and in_range:
        ok(f"{name:<24} splits, labels, unicité (uid,exp), probas dans [0,1]")
    else:
        n_div = 0 if same_split else int((k.fold.values != key_ref.fold.values).sum())
        fail("P1", f"{name} : splits={same_split}"
                   + (f" ({n_div} affectations divergentes)" if n_div else "")
                   + f" | labels={same_lab} | unicité={no_dup} | probas={in_range}")

Y = lab_ref.astype(int).values
UIDS = list(lab_ref.index)
PREV = float(Y.mean())
print(f"\n  {len(UIDS)} patients | prévalence {PREV:.4f}")

# --- moyennes OOF par famille (M = N_EXP membres) ----------------------
P_OOF = {}
for name, d in OOF.items():
    P_OOF[name] = clip_p(d.groupby('uid')['prob'].mean().sort_index().loc[UIDS].values)


# ====================================================================
# PHASE 2 — modèles chargés
# ====================================================================
head(2, "MODÈLES CHARGÉS")

t0 = time.time()
arbitre, modeles_actifs = init_pipeline()
print(f"\n  init_pipeline : {time.time()-t0:.1f} s")

INST = {}
for name, (_, key) in FAMILIES.items():
    if key not in modeles_actifs:
        fail("P2", f"{name} : '{key}' absent de modeles_actifs {list(modeles_actifs)}")
        continue
    inst = modeles_actifs[key]
    INST[name] = inst
    n = len(getattr(inst, 'models', []))
    if n == M_DEPLOY:
        ok(f"{name:<24} {n} réseaux "
           f"-> index (exp, fold) = exp*{N_FOLDS}+fold VALIDE")
    else:
        fail("P2", f"{name} : {n} réseaux chargés au lieu de {M_DEPLOY}. "
                   f"L'index exp*{N_FOLDS}+fold serait DÉCALÉ et la température "
                   f"(calibrée pour {M_DEPLOY}) serait fausse.")

# modèles en mode eval
import torch                                                    # noqa: E402
for name, inst in INST.items():
    torch_models = [m for m in getattr(inst, 'models', [])
                    if isinstance(m, torch.nn.Module)]
    if torch_models and any(m.training for m in torch_models):
        fail("P2", f"{name} : au moins un réseau en mode train() "
                   f"(BatchNorm/Dropout actifs)")
    elif torch_models:
        ok(f"{name:<24} {len(torch_models)} réseaux torch en eval()")


# ====================================================================
# PHASE 3 — inputs
# ====================================================================
head(3, "INPUTS DU PREPROCESSING")

sample_uids = list(rng.choice(UIDS, min(N_PATIENTS, len(UIDS)), replace=False))
probe_uid = next((u for u in sample_uids if (NIFTI_DIR / f"{u}.nii.gz").exists()), None)
if probe_uid is None:
    print("  ❌ aucun NIfTI trouvé parmi l'échantillon. Arrêt.")
    sys.exit(1)

t0 = time.time()
inputs = prepare_all_inputs(str(NIFTI_DIR / f"{probe_uid}.nii.gz"))
t_prep = time.time() - t0
print(f"  patient sonde : {probe_uid}   ({t_prep:.2f} s)\n")


def describe(v):
    if isinstance(v, torch.Tensor):
        a = v.detach().cpu().numpy()
        return (f"Tensor{tuple(v.shape)} {v.dtype} "
                f"[{np.nanmin(a):.3f}, {np.nanmax(a):.3f}]"
                + ("  ⚠️ NaN" if np.isnan(a).any() else ""))
    if isinstance(v, np.ndarray):
        return (f"ndarray{v.shape} {v.dtype} "
                f"[{np.nanmin(v):.3f}, {np.nanmax(v):.3f}]"
                + ("  ⚠️ NaN" if np.isnan(v).any() else ""))
    if isinstance(v, pd.DataFrame):
        return f"DataFrame{v.shape} colonnes nommées ✓"
    if isinstance(v, pd.Series):
        return f"Series({len(v)}) index nommé ✓"
    if isinstance(v, dict):
        return f"dict({len(v)} clés) nommé ✓"
    if isinstance(v, list):
        return f"list({len(v)})"
    return type(v).__name__

print(f"  {'clé':<24}{'contenu'}")
for k in sorted(inputs):
    print(f"  {k:<24}{describe(inputs[k])}")

# --- XGBoost a besoin de features NOMMÉES ------------------------------
print()
sc = inputs.get('scalars')
if 'xgboost' not in FAMILIES:
    print("  (XGBoost non utilisé : contrôle de la clé 'scalars' ignoré)")
elif sc is None:
    fail("P3", "clé 'scalars' absente : XGBoostEnsembleInference lèvera une exception")
elif isinstance(sc, (dict, pd.Series, pd.DataFrame)):
    ok("'scalars' est nommé (dict/Series/DataFrame) -> XGBoost pourra filtrer")
    try:
        exp_feats = INST['xgboost'].expected_features
        cols = (set(sc.keys()) if isinstance(sc, dict)
                else set(sc.index) if isinstance(sc, pd.Series) else set(sc.columns))
        miss = [f for f in exp_feats if f not in cols]
        if miss:
            fail("P3", f"{len(miss)} features XGBoost manquantes, ex. {miss[:3]}")
        else:
            ok(f"les {len(exp_feats)} features attendues par XGBoost sont présentes")
    except Exception as e:
        print(f"  ⚠️  vérification des features XGBoost impossible : {e}")
else:
    fail("P3", f"'scalars' est un {type(sc).__name__} sans noms de colonnes. "
               f"XGBoostEnsembleInference fera df[feature_names] et échouera.")

# --- XGBoost : contrôles spécifiques -----------------------------------
# C'est la seule famille qui passe par des NOMS de colonnes et une conversion
# DataFrame -> DMatrix. Trois façons de se tromper silencieusement.
xgb_inst = INST.get('xgboost')
if xgb_inst is not None and getattr(xgb_inst, 'models', None):
    print("\n  XGBoost — contrôles spécifiques :")

    # 1. les 100 boosters attendent-ils EXACTEMENT les mêmes features, dans le
    #    même ordre ? predict() n'utilise que models[0].get_booster() pour
    #    construire df_filtered, puis l'envoie aux 99 autres.
    names0 = list(xgb_inst.models[0].get_booster().feature_names or [])
    bad = [i for i, m in enumerate(xgb_inst.models)
           if list(m.get_booster().feature_names or []) != names0]
    if not bad:
        ok(f"les {len(xgb_inst.models)} boosters partagent les mêmes "
           f"{len(names0)} features, dans le même ordre")
    else:
        fail("P3", f"{len(bad)} boosters ont un ordre de features différent de "
                   f"models[0] (ex. index {bad[:5]}). predict() leur enverra des "
                   f"colonnes mal ordonnées.")

    # 2. la liste JSON top_k_features correspond-elle aux boosters ?
    js = list(getattr(xgb_inst, 'expected_features', []))
    if set(js) == set(names0):
        ok(f"top_k_features.json == features des boosters ({len(js)})")
    else:
        only_js, only_bk = set(js) - set(names0), set(names0) - set(js)
        fail("P3", f"top_k_features.json et les boosters divergent : "
                   f"{len(only_js)} seulement dans le JSON {sorted(only_js)[:3]}, "
                   f"{len(only_bk)} seulement dans les boosters {sorted(only_bk)[:3]}")

    # 3. device : un modèle entraîné sur GPU et prédit sur CPU peut dériver
    dev = {str(m.get_params().get('device', 'n/a')) for m in xgb_inst.models}
    print(f"    device des boosters : {dev}"
          + ("   (forcé CPU — un entraînement GPU peut donner de légers écarts, "
             "la phase 4 le verra)" if dev == {'cpu'} else ""))

    # 4. les valeurs réellement envoyées, pour inspection à l'œil
    try:
        df_sc = (pd.DataFrame([sc]) if isinstance(sc, dict)
                 else sc.to_frame().T if isinstance(sc, pd.Series) else sc.copy())
        vals = df_sc[names0].iloc[0]
        print(f"    5 premières features envoyées : "
              + ", ".join(f"{k}={v:.4g}" for k, v in vals.head(5).items()))
        n_nan = int(vals.isna().sum())
        if n_nan:
            print(f"    ⚠️  {n_nan} features NaN — XGBoost les traite comme "
                  f"'manquantes', ce qui est valide mais doit être voulu")
    except Exception as e:
        fail("P3", f"impossible de construire le DataFrame XGBoost : {e}")

if 'features' not in inputs:
    print("  ⚠️  clé 'features' absente : main.py fait arbitre.post_process(v, "
          "inputs['features']) -> KeyError. Passe à inputs.get('features').")

# --- déterminisme du preprocessing -------------------------------------
inputs2 = prepare_all_inputs(str(NIFTI_DIR / f"{probe_uid}.nii.gz"))
det, diffs = True, []
for k, v in inputs.items():
    w = inputs2.get(k)
    try:
        if isinstance(v, torch.Tensor):
            d = float((v - w).abs().max())
        elif isinstance(v, np.ndarray):
            d = float(np.nanmax(np.abs(v - w)))
        else:
            continue
        if d > 0:
            det = False; diffs.append((k, d))
    except Exception:
        continue
if det:
    ok("preprocessing déterministe (même patient deux fois = bit à bit identique)")
else:
    fail("P3", f"preprocessing NON déterministe : {diffs[:3]}. "
               f"Toute comparaison à l'OOF devient un tirage au sort.")


# ====================================================================
# PHASE 4 — comparaison appariée, réseau par réseau
# ====================================================================
head(4, "COMPARAISON APPARIÉE : CHAQUE RÉSEAU vs SA LIGNE D'OOF")

print(f"  {N_PATIENTS} patients x {len(FAMILIES)} familles x "
      f"{N_MODELS_PER_FAMILY} réseaux = "
      f"{N_PATIENTS*len(FAMILIES)*N_MODELS_PER_FAMILY} comparaisons\n")


def predict_single(inst, inputs_, idx):
    """Force la famille à n'utiliser QUE le réseau `idx` et renvoie sa proba brute.

    On neutralise aussi _apply_recipe : le CSV d'OOF contient des probas BRUTES,
    sans température ni clip.
    """
    orig_models = inst.models
    orig_apply = getattr(inst, '_apply_recipe', None)
    try:
        inst.models = [orig_models[idx]]
        if orig_apply is not None:
            inst._apply_recipe = lambda a: float(np.mean(np.asarray(a, dtype=float)))
        return float(predict_for_model(inst, inputs_))
    finally:
        inst.models = orig_models
        if orig_apply is not None:
            inst._apply_recipe = orig_apply


rows, skipped = [], []
for uid in tqdm(sample_uids, desc="patients"):
    img = NIFTI_DIR / f"{uid}.nii.gz"
    if not img.exists():
        skipped.append(uid); continue
    inp = prepare_all_inputs(str(img))                 # une seule fois par patient
    for fam, inst in INST.items():
        sub = OOF[fam][OOF[fam].uid == uid]
        if sub.empty:
            continue
        take = sub.sample(min(N_MODELS_PER_FAMILY, len(sub)), random_state=SEED)
        for _, r in take.iterrows():
            e, f = int(r['exp']), int(r['fold'])
            idx = e * N_FOLDS + f
            if idx >= len(inst.models):
                fail("P4", f"{fam} : index {idx} hors bornes ({len(inst.models)})")
                continue
            rows.append(dict(uid=uid, famille=fam, exp=e, fold=f,
                             p_oof=float(r['prob']),
                             p_rt=predict_single(inst, inp, idx),
                             label=float(r['label'])))

if not rows:
    print("  ❌ aucune comparaison effectuée. Arrêt.")
    sys.exit(1)

R = pd.DataFrame(rows)
R['ecart_logit'] = np.abs(lgt(R.p_rt.values) - lgt(R.p_oof.values))
R['ecart_proba'] = np.abs(R.p_rt.values - R.p_oof.values)
R['ecart_logit_signe'] = lgt(R.p_rt.values) - lgt(R.p_oof.values)
y_r = R.label.values.astype(int)
log_loss_oof = float(ll_vec(y_r, R.p_oof.values).mean())
log_loss_rt = float(ll_vec(y_r, R.p_rt.values).mean())
ll_impact = log_loss_rt - log_loss_oof

print("\n  ------------------------------------------------------------------")
print("  LOG LOSS, sur les mêmes patients et les mêmes réseaux")
print("  ------------------------------------------------------------------")
print(f"  {'famille':<24}{'comparaisons':>13}{'log loss OOF':>14}"
      f"{'log loss inférence':>20}{'écart':>10}")
for fam, g in R.groupby('famille'):
    yg = g.label.values.astype(int)
    a = float(ll_vec(yg, g.p_oof.values).mean())
    b = float(ll_vec(yg, g.p_rt.values).mean())
    flag = "" if abs(b - a) <= ATOL_LOGLOSS else "  <-- ÉCART"
    print(f"  {fam:<24}{len(g):>13}{a:>14.4f}{b:>20.4f}{b-a:>+10.4f}{flag}")
print(f"  {'TOTAL':<24}{len(R):>13}{log_loss_oof:>14.4f}"
      f"{log_loss_rt:>20.4f}{ll_impact:>+10.4f}")
print(f"\n  Lecture : 'log loss OOF' est ce que le notebook a mesuré. 'log loss")
print(f"  inférence' est ce que produit ce pipeline. Les deux colonnes doivent")
print(f"  être identiques à {ATOL_LOGLOSS} près, sinon tu ne déploies pas le")
print(f"  modèle que tu as validé.")

print("\n  ------------------------------------------------------------------")
print("  ÉCART DE LOGIT, par famille")
print("  ------------------------------------------------------------------")
print("  (le modèle produit des logits ; en probabilité un même écart paraît")
print("   250 fois plus grand à p=0.5 qu'à p=0.99, donc on mesure en logit)")
print(f"\n  {'famille':<24}{'maximum':>11}{'médian':>11}{'moyen signé':>14}"
      f"{'hors seuil':>12}")
for fam, g in R.groupby('famille'):
    nb = int((g.ecart_logit > ATOL_LOGIT).sum())
    flag = "" if nb == 0 else "  <-- ANORMAL"
    print(f"  {fam:<24}{g.ecart_logit.max():>11.2e}{g.ecart_logit.median():>11.2e}"
          f"{g.ecart_logit_signe.mean():>+14.2e}{nb:>8}/{len(g):<3}{flag}")
print(f"\n  seuil = {ATOL_LOGIT}   |   bruit float32 normal ~1e-3   |   "
      f"bug de pipeline ~1e-1")
print("  La colonne 'moyen signé' est la plus importante : un décalage")
print("  systématique s'y voit, alors que le maximum peut n'être qu'un cas isolé.")
print(f"\n  Pour information, écart maximum exprimé en probabilité : "
      f"{R.ecart_proba.max():.2e}")
if skipped:
    print(f"  {len(skipped)} patients ignorés (NIfTI absent) : {skipped[:5]}")

n_bad = int((R.ecart_logit > ATOL_LOGIT).sum())
if n_bad or abs(ll_impact) > ATOL_LOGLOSS:
    fail("P4", f"{n_bad}/{len(R)} comparaisons hors seuil, "
               f"impact log loss {ll_impact:+.4f}")
    print("\n  Les 10 plus gros écarts :")
    print(R.nlargest(10, 'ecart_logit')[
        ['uid', 'famille', 'exp', 'fold', 'p_oof', 'p_rt',
         'ecart_proba', 'ecart_logit']].to_string(index=False))
    by_fam = R[R.ecart_logit > ATOL_LOGIT].famille.value_counts()
    print()
    if len(by_fam) == 1:
        f0 = by_fam.index[0]
        print(f"  DIAGNOSTIC : tous les écarts viennent de '{f0}'.")
        print(f"  Les autres familles sont propres, donc prepare_all_inputs n'est")
        print(f"  pas en cause. Le problème est dans la classe d'inférence de")
        print(f"  '{f0}' : poids partiellement chargés, mauvaise clé d'entrée,")
        print(f"  normalisation absente, ou ordre des arguments du forward.")
    else:
        print(f"  DIAGNOSTIC : écarts sur {len(by_fam)} familles "
              f"({dict(by_fam)}).")
        print(f"  Une cause commune est probable : prepare_all_inputs, une")
        print(f"  normalisation partagée, ou un resize.")
else:
    ok(f"les {len(R)} comparaisons concordent à {ATOL_LOGIT} près en logit")
    if R.ecart_logit.max() > 1e-4:
        print("     (écarts de l'ordre de 1e-3 : bruit float32 normal — taille de")
        print("      batch différente entre validation et inférence, TF32, ou")
        print("      algorithme cuDNN sélectionné différemment)")

print("\n  Les 5 plus gros écarts, en détail :")
print(R.nlargest(5, 'ecart_logit')[
    ['uid', 'famille', 'exp', 'fold', 'p_oof', 'p_rt',
     'ecart_proba', 'ecart_logit']].to_string(index=False))

# --- 4b : les 100 réseaux d'une famille sont-ils VRAIMENT différents ? ---
# Si une boucle de chargement recharge cent fois le même checkpoint, la moyenne
# vaut un seul modèle, la diversité disparaît et la température est fausse.
# Rien ne le signalerait : le compte serait bon et les probas plausibles.
print("\n  ------------------------------------------------------------------")
print(f"  DISPERSION DES 100 RÉSEAUX sur le patient sonde ({probe_uid})")
print("  ------------------------------------------------------------------")
print("  Si une boucle recharge cent fois le même checkpoint, la moyenne vaut")
print("  un seul modèle et rien ne le signale. Si des poids ne sont que")
print("  partiellement chargés, la dispersion EXPLOSE au contraire.")
print(f"\n  {'famille':<24}{'moyenne':>10}{'écart-type':>13}"
      f"{'écart-type OOF':>17}{'rapport':>10}{'minimum':>10}{'maximum':>10}")
for fam, inst in INST.items():
    ps = np.array([predict_single(inst, inputs, i) for i in range(len(inst.models))])
    sd_oof = float(OOF[fam][OOF[fam].uid == probe_uid]['prob'].std())
    flag = ""
    if ps.std() < 1e-9:
        flag = "   ❌ 100 réseaux IDENTIQUES"
        fail("P4b", f"{fam} : les {len(ps)} réseaux donnent la même sortie — "
                    f"le même checkpoint est probablement chargé en boucle")
    ratio = ps.std() / sd_oof if sd_oof > 0 else float('nan')
    if np.isfinite(ratio) and (ratio > 2 or ratio < 0.5):
        flag = flag or "  <-- ANORMAL"
        fail("P4b", f"{fam} : dispersion en inférence {ratio:.1f}x celle de l'OOF")
    print(f"  {fam:<24}{ps.mean():>10.4f}{ps.std():>13.5f}{sd_oof:>17.5f}"
          f"{ratio:>10.2f}{ps.min():>10.4f}{ps.max():>10.4f}{flag}")
print("\n  Les deux écarts-types ne seront pas identiques — 100 réseaux contre")
print("  20 en OOF — mais le rapport doit rester entre 0.5 et 2. Au-delà de 2,")
print("  suspecte des poids partiellement chargés (strict=False). En dessous de")
print("  0.5, suspecte des modèles dupliqués. Un écart-type nul est fatal.")

# ====================================================================
# PHASE 4c — VÉRIFICATION DE L'AGRÉGATION INTERNE (Ensemble complet)
# ====================================================================
head(4, "4c — VÉRIFICATION DE L'AGRÉGATION INTERNE DES CLASSES D'INFÉRENCE")

print("  On vérifie que la méthode predict() de chaque famille calcule bien une")
print("  MOYENNE PURE des 100 réseaux, sans transformation cachée (clip, T, etc.).\n")

uids_fam_test = sample_uids[:min(3, len(sample_uids))]

rows_fam = []
for uid in tqdm(uids_fam_test, desc="Vérification de l'agrégation"):
    img = NIFTI_DIR / f"{uid}.nii.gz"
    if not img.exists(): 
        continue
    inp = prepare_all_inputs(str(img))

    for fam, inst in INST.items():
        # 1. Prédiction native de la classe (qui fait tourner les 100 modèles d'un coup)
        p_native = float(predict_for_model(inst, inp))
        
        # 2. Prédiction manuelle (on force l'inférence réseau par réseau sur les 100 et on moyenne)
        # On utilise predict_single qui est déjà validé en Phase 4
        p_manual_list = [predict_single(inst, inp, i) for i in range(len(inst.models))]
        p_manual_mean = float(np.mean(p_manual_list))

        rows_fam.append(dict(
            uid=uid, 
            famille=fam,
            p_native=p_native,
            p_manual=p_manual_mean,
            ecart_proba=abs(p_native - p_manual_mean)
        ))

if rows_fam:
    R_fam = pd.DataFrame(rows_fam)
    print(f"\n  {'famille':<24}{'écart proba (max)':>20}{'état':>10}")
    max_diff_fam = 0
    
    for fam, g in R_fam.groupby('famille'):
        max_d = g.ecart_proba.max()
        max_diff_fam = max(max_diff_fam, max_d)
        
        # On tolère 1e-5 (sommer 100 float32 dans une boucle vs en vectorisé peut donner d'infimes écarts)
        flag = "  <-- ANORMAL" if max_d > 1e-5 else "  OK"
        print(f"  {fam:<24}{max_d:>20.2e}{flag}")

    if max_diff_fam > 1e-5:
        fail("P4c", f"L'agrégation interne d'une famille n'est pas une moyenne pure (écart max {max_diff_fam:.2e})")
    else:
        ok("Toutes les classes d'inférence font bien une moyenne pure de leurs 100 réseaux.")
else:
    print("  (aucun test d'agrégation réalisé : patients introuvables)")

# ====================================================================
# PHASE 5 — l'arbitre applique-t-il bien la recette ?
# ====================================================================
head(5, "ARBITRE : LA RECETTE, RECODÉE INDÉPENDAMMENT")

KEYS = [k for _, k in FAMILIES.values()]


def recipe_numpy(probs_by_key, T=EXPECTED_T, clip=EXPECTED_CLIP):
    """Référence indépendante : agrégation -> logit -> moyenne -> T -> clip."""
    z = np.mean([lgt(probs_by_key[k]) for k in KEYS], axis=0)
    return np.clip(expit(z / T), clip[0], clip[1])


# 1. égalité exacte sur des cas tirés au hasard
worst = 0.0
for _ in range(200):
    d = {k: float(rng.uniform(0.001, 0.999)) for k in KEYS}
    worst = max(worst, abs(float(arbitre.predict(d)) - float(recipe_numpy(d))))
if worst < 1e-12:
    ok(f"arbitre.predict == référence numpy (écart max {worst:.1e} sur 200 tirages)")
else:
    fail("P5", f"arbitre.predict diffère de la référence : écart max {worst:.2e}")

# 2. température réellement appliquée, retrouvée par inversion
print("\n  Température déduite en inversant la sortie :")
T_hat = []
for p_in in (0.90, 0.70, 0.30, 0.10):
    got = float(arbitre.predict({k: p_in for k in KEYS}))
    T_hat.append(sp_logit(p_in) / sp_logit(got))
    print(f"    entrée {p_in:.2f} sur les 4 familles -> {got:.6f}   "
          f"T déduite {T_hat[-1]:.6f}")
T_code = float(np.mean(T_hat))
drift = abs(T_code - EXPECTED_T) / EXPECTED_T
if drift < 1e-6:
    ok(f"T appliquée = {T_code:.6f} = valeur attendue")
elif drift < 0.01:
    print(f"  ⚠️  T appliquée {T_code:.6f} vs attendue {EXPECTED_T:.6f} "
          f"({drift*100:.3f}%) — sans impact mesurable, mais une constante a dérivé")
else:
    fail("P5", f"T appliquée {T_code:.6f} vs attendue {EXPECTED_T:.6f} "
               f"({drift*100:.2f}%) — MATÉRIEL")

# 3. bornes du clip
lo = float(arbitre.predict({k: 1e-9 for k in KEYS}))
hi = float(arbitre.predict({k: 1 - 1e-9 for k in KEYS}))
if abs(lo - EXPECTED_CLIP[0]) < 1e-9 and abs(hi - EXPECTED_CLIP[1]) < 1e-9:
    ok(f"clip actif aux bornes : {lo} / {hi}")
else:
    fail("P5", f"clip {lo} / {hi}, attendu {EXPECTED_CLIP}")

# 4. poids réellement égaux (une famille extrême ne doit pas dominer)
base = float(arbitre.predict({k: 0.5 for k in KEYS}))
shifts = []
for k in KEYS:
    d = {kk: 0.5 for kk in KEYS}; d[k] = 0.9
    shifts.append(abs(float(arbitre.predict(d)) - base))
if max(shifts) - min(shifts) < 1e-9:
    ok(f"poids égaux : chaque famille déplace la sortie de {shifts[0]:.6f}")
else:
    fail("P5", f"poids INÉGAUX entre familles : {np.round(shifts, 6)}")

# 5. clip APRÈS température (l'ordre inverse serait défait par T)
d = {k: 1e-6 for k in KEYS}
if abs(float(arbitre.predict(d)) - EXPECTED_CLIP[0]) < 1e-9:
    ok("ordre agrégation -> température -> clip respecté")
else:
    fail("P5", "le clip ne semble pas appliqué en dernier")


# ====================================================================
# PHASE 6 — bout en bout sur les 1362 patients de l'OOF
# ====================================================================
head(6, "BOUT EN BOUT SUR L'OOF COMPLET (sans inférence)")

probs_oof = {key: P_OOF[fam] for fam, (_, key) in FAMILIES.items()}
z_oof = np.mean([lgt(probs_oof[k]) for k in KEYS], axis=0)

p_arb = np.array([float(arbitre.predict({k: probs_oof[k][i] for k in KEYS}))
                  for i in range(len(UIDS))])
p_ref = np.clip(expit(z_oof / EXPECTED_T), *EXPECTED_CLIP)
d_max = float(np.abs(p_arb - p_ref).max())
if d_max < 1e-12:
    ok(f"arbitre reproduit la référence sur les {len(UIDS)} patients "
       f"(écart max {d_max:.1e})")
else:
    fail("P6", f"divergence arbitre/référence sur l'OOF : {d_max:.2e}")

print(f"\n  Log loss OOF (M={N_EXP} membres par famille, pas {M_DEPLOY}) :")
for lab, p in [("meilleure famille seule", min(P_OOF.values(), key=lambda q: ll_vec(Y, q).mean())),
               ("mélange, T=1, sans clip", expit(z_oof)),
               (f"mélange, T={EXPECTED_T} + clip", p_arb)]:
    print(f"    {lab:<30} {ll_vec(Y, p).mean():.4f}")
print(f"\n  Note : la T soumise contient le facteur 1.245 de décalage leaderboard.")
print(f"  Elle est donc DÉLIBÉRÉMENT sous-optimale sur l'OOF. Le chiffre à comparer")
print(f"  au notebook est la ligne 'T=1, sans clip', recalibrée à T={EXPECTED_T/1.245/0.99515:.4f}.")
p_nb = np.clip(expit(z_oof / (EXPECTED_T / 1.245 / 0.99515)), *EXPECTED_CLIP)
print(f"    mélange à la T de l'OOF       {ll_vec(Y, p_nb).mean():.4f}"
      f"   <- doit être proche du notebook")


# ====================================================================
# PHASE 7 — format de soumission
# ====================================================================
head(7, "FORMAT DE SOUMISSION")

if SUBMISSION_FORMAT.exists():
    sub = pd.read_csv(SUBMISSION_FORMAT)
    print(f"  {SUBMISSION_FORMAT.name} : {len(sub)} lignes, colonnes {list(sub.columns)}")
    if 'uid' in sub.columns and 'is_pathologic' in sub.columns:
        ok("colonnes 'uid' et 'is_pathologic' présentes")
    else:
        fail("P7", f"colonnes attendues 'uid'/'is_pathologic', trouvées {list(sub.columns)}")
    n_miss = sum(1 for u in sub['uid'].astype(str)
                 if not (NIFTI_DIR / f"{u}.nii.gz").exists())
    if n_miss == 0:
        ok(f"les {len(sub)} NIfTI de la soumission existent")
    else:
        print(f"  ⚠️  {n_miss}/{len(sub)} NIfTI absents -> autant de prédictions à 0.5")
        print(f"     (coût ~{n_miss*0.693/len(sub):.4f} de log loss ; envisage la "
              f"prévalence {PREV:.4f} plutôt que 0.5)")
else:
    print(f"  (ignoré : {SUBMISSION_FORMAT} absent — normal hors conteneur)")


# ====================================================================
# PHASE 8 — chronométrage
# ====================================================================
head(8, "CHRONOMÉTRAGE")

from main import diagnostiquer_patient                          # noqa: E402

n_time = min(12, len(sample_uids))
ts = []
for uid in sample_uids[:n_time]:
    img = NIFTI_DIR / f"{uid}.nii.gz"
    if not img.exists():
        continue
    t0 = time.time()
    _ = diagnostiquer_patient(str(img), arbitre, modeles_actifs)
    ts.append(time.time() - t0)

if ts:
    per = float(np.median(ts[1:]))
    total = per * 600
    print(f"  pipeline complet ({len(INST)} familles x {M_DEPLOY} réseaux) : "
          f"{per:.2f} s/patient")
    print(f"  dont preprocessing : ~{t_prep:.2f} s ({100*t_prep/per:.0f}%)")
    print(f"\n  Extrapolation à 600 patients : {total/60:.1f} min "
          f"(+ {time.time()-t0:.0f} s de chargement des modèles)")
    if total > 3600:
        print("  ⚠️  au-delà d'une heure : vérifie le budget du conteneur.")
else:
    print("  (aucun patient chronométré)")


# ====================================================================
# VERDICT
# ====================================================================
print("\n" + "=" * 78)
if FAILURES:
    print("  ❌ NE PAS SOUMETTRE")
    print("=" * 78)
    for f_ in FAILURES:
        print(f"  {f_}")
else:
    print("  ✅ PIPELINE VALIDÉ")
    print("=" * 78)
    print(f"  {len(R)} comparaisons appariées, écart de logit maximum "
          f"{R.ecart_logit.max():.2e}")
    print(f"  log loss OOF {log_loss_oof:.4f} vs inférence {log_loss_rt:.4f} "
          f"({ll_impact:+.4f})")
    print(f"  recette : 4 familles, moyenne des logits, T={EXPECTED_T}, "
          f"clip={EXPECTED_CLIP}")
print("=" * 78)

R.to_csv('validation_report.csv', index=False)
print("\nDétail par comparaison : validation_report.csv")
sys.exit(1 if FAILURES else 0)