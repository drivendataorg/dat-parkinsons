"""
diagnose_3d.py
==============
Diagnostic ciblé de la famille 3d (MiniCNN3DSiameseEnsemble), dont la
validation a révélé un décalage systématique (+0.374 en logit) et une
variance inter-modèles quatre fois supérieure à celle observée en OOF.

Trois tests, du plus rapide au plus lent :

  TEST 1  Les poids sont-ils entièrement chargés ?         (aucune inférence)
  TEST 2  La clé d'entrée est-elle la bonne ?              (cubes vs cubes_sbr)
  TEST 3  Manque-t-il une normalisation ?                  (balayage)

Le premier qui explique l'écart est la cause. Arrête-toi là.
"""

import os
import sys
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
logging.getLogger('nibabel').setLevel(logging.ERROR)

# ============================== CONFIG ==============================
DATA_DIR = Path("/data2/iceberg/dat_parkinson/")
if not DATA_DIR.exists():
    DATA_DIR = Path("data")
NIFTI_DIR = DATA_DIR / "niftis"

OOF_3D = Path('/data/HOME/sansel/dat-park/runs_siamois/'
              'minicnn3d_mp_ensemble/master_oof_predictions.csv')

FAMILY_KEY = 'MiniCNN3D'      # clé dans modeles_actifs
N_EXP, N_FOLDS = 20, 5
N_PATIENTS = 8                # patients testés par variante
SEED = 0
# ====================================================================

rng = np.random.default_rng(SEED)
lgt = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))

import main                                                    # noqa: E402
from main import init_pipeline, predict_for_model              # noqa: E402
from src.preprocessing import prepare_all_inputs               # noqa: E402

print("Chargement du pipeline…")
arbitre, modeles_actifs = init_pipeline()
inst = modeles_actifs[FAMILY_KEY]
oof = pd.read_csv(OOF_3D); oof['uid'] = oof['uid'].astype(str)

# échantillon de lignes d'OOF à reproduire
sample = oof.sample(N_PATIENTS * 2, random_state=SEED)
sample = sample[[(NIFTI_DIR / f"{u}.nii.gz").exists() for u in sample.uid]].head(N_PATIENTS)
print(f"{len(sample)} lignes d'OOF à reproduire.\n")


def ecart_logit(probs_obtenues, probs_attendues):
    d = lgt(np.asarray(probs_obtenues)) - lgt(np.asarray(probs_attendues))
    return float(np.abs(d).max()), float(np.abs(d).mean()), float(d.mean())


# ====================================================================
# TEST 1 — les poids sont-ils entièrement chargés ?
# ====================================================================
print("=" * 74)
print("  TEST 1 — CHARGEMENT DES POIDS  (aucune inférence, ~5 secondes)")
print("=" * 74)
print("""
  MiniCNN3DSiameseEnsemble charge avec strict=False. C'est la seule famille
  dans ce cas. Si les clés du checkpoint ne correspondent pas à l'architecture
  instanciée, les couches concernées restent ALÉATOIRES — et PyTorch ne dit
  rien. C'est la seule hypothèse qui explique une variance multipliée par 4.
""")

models_dir = os.path.join(os.path.dirname(os.path.abspath(main.__file__)), 'weights')
chemin = os.path.join(models_dir, f"minicnn3d_exp0_fold0.pth")
if not os.path.exists(chemin):
    print(f"  ⚠️  {chemin} introuvable — ajuste le nom de fichier.")
else:
    checkpoint = torch.load(chemin, map_location='cpu')
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    modele_neuf = inst._instantiate_model()
    manquantes, inattendues = modele_neuf.load_state_dict(checkpoint, strict=False)

    print(f"  clés du checkpoint          : {len(checkpoint)}")
    print(f"  clés de l'architecture      : {len(modele_neuf.state_dict())}")
    print(f"  clés MANQUANTES (restent aléatoires) : {len(manquantes)}")
    print(f"  clés INATTENDUES (ignorées)          : {len(inattendues)}")

    if manquantes or inattendues:
        print("\n  ❌ CAUSE TROUVÉE.")
        if manquantes:
            print(f"     manquantes : {list(manquantes)[:8]}")
        if inattendues:
            print(f"     inattendues : {list(inattendues)[:8]}")
        print("""
  L'architecture instanciée ne correspond pas au checkpoint. Corrige
  build_model("minicnn3d_siamese") ou les arguments passés, puis repasse
  en strict=True pour que ça ne puisse plus arriver en silence.

  Inutile de lancer les tests 2 et 3 tant que celui-ci n'est pas vert.
""")
        sys.exit(1)
    else:
        print("\n  ✓ tous les poids sont chargés. La cause est ailleurs.")

# ====================================================================
# TEST 2 — la clé d'entrée est-elle la bonne ?
# ====================================================================
print("\n" + "=" * 74)
print("  TEST 2 — CLÉ D'ENTRÉE  (siamese_cubes vs siamese_cubes_sbr)")
print("=" * 74)
print("""
  Le preprocessing expose DEUX jeux de cubes. Le routage envoie
  'siamese_cubes'. Ton dossier d'entraînement s'appelle minicnn3d_mp_ensemble,
  et le suffixe 'mp' se retrouve dans d'autres runs : c'est une variante de
  prétraitement. On rejoue les mêmes lignes d'OOF avec chaque clé.
""")


def rejouer(cle_entree):
    """Rejoue les lignes échantillonnées en forçant une clé d'entrée."""
    obtenues, attendues = [], []
    for _, r in sample.iterrows():
        entrees = prepare_all_inputs(str(NIFTI_DIR / f"{r.uid}.nii.gz"))
        if cle_entree not in entrees:
            return None
        idx = int(r['exp']) * N_FOLDS + int(r['fold'])
        tous = inst.models
        try:
            inst.models = [tous[idx]]
            obtenues.append(float(inst.predict(entrees[cle_entree])))
        finally:
            inst.models = tous
        attendues.append(float(r['prob']))
    return obtenues, attendues


resultats = {}
for cle in ('siamese_cubes', 'siamese_cubes_sbr'):
    out = rejouer(cle)
    if out is None:
        print(f"  {cle:<22} absent du preprocessing")
        continue
    mx, moy, signe = ecart_logit(*out)
    resultats[cle] = mx
    verdict = "✓ CONCORDE" if mx < 0.02 else "✗"
    print(f"  {cle:<22} écart logit max {mx:>8.4f} | moyen {moy:>8.4f} | "
          f"signé {signe:>+8.4f}   {verdict}")

if resultats and min(resultats.values()) < 0.02:
    gagnante = min(resultats, key=resultats.get)
    print(f"""
  ❌ CAUSE TROUVÉE : la bonne clé est '{gagnante}'.

  Corrige le routage dans main.py :
      (MiniCNN3DSiameseEnsemble,): "{gagnante}",

  Inutile de lancer le test 3.
""")
    sys.exit(1)
else:
    print("\n  ✓ aucune des deux clés ne concorde. La cause est ailleurs.")

# ====================================================================
# TEST 3 — manque-t-il une normalisation ?
# ====================================================================
print("\n" + "=" * 74)
print("  TEST 3 — NORMALISATION DES CUBES")
print("=" * 74)
print("""
  MiniCNN3DSiameseEnsemble.predict ne fait que .to(device) et unsqueeze.
  Aucune normalisation. Toutes tes autres familles normalisent. On teste
  quelques transformations classiques sur les cubes avant le passage avant.
""")

VARIANTES = {
    'aucune (actuel)':      lambda c: c,
    'z-score par volume':   lambda c: (c - c.mean()) / (c.std() + 1e-8),
    'min-max par volume':   lambda c: (c - c.min()) / (c.max() - c.min() + 1e-8),
    'division par le max':  lambda c: c / (c.max() + 1e-8),
    'centrage [-1, 1]':     lambda c: (c - 0.5) / 0.5,
}

print(f"  {'variante':<24}{'écart logit max':>18}{'moyen':>10}{'signé':>10}")
meilleure, meilleur_score = None, np.inf
for nom, f in VARIANTES.items():
    obtenues, attendues = [], []
    for _, r in sample.iterrows():
        entrees = prepare_all_inputs(str(NIFTI_DIR / f"{r.uid}.nii.gz"))
        cubes = {k: v.clone() for k, v in entrees['siamese_cubes'].items()}
        for k in ('cube_L', 'cube_R'):
            cubes[k] = f(cubes[k])
        idx = int(r['exp']) * N_FOLDS + int(r['fold'])
        tous = inst.models
        try:
            inst.models = [tous[idx]]
            obtenues.append(float(inst.predict(cubes)))
        finally:
            inst.models = tous
        attendues.append(float(r['prob']))
    mx, moy, signe = ecart_logit(obtenues, attendues)
    flag = "   ✓ CONCORDE" if mx < 0.02 else ""
    print(f"  {nom:<24}{mx:>18.4f}{moy:>10.4f}{signe:>+10.4f}{flag}")
    if mx < meilleur_score:
        meilleure, meilleur_score = nom, mx

print()
if meilleur_score < 0.02:
    print(f"  ❌ CAUSE TROUVÉE : il manque '{meilleure}' dans _prepare/predict.")
else:
    print(f"""  Aucune variante ne concorde (meilleure : '{meilleure}' à {meilleur_score:.3f}).

  Il reste à vérifier, dans ton code d'ENTRAÎNEMENT de minicnn3d :
    - l'ordre des arguments : model(cube_L, cube_R) ou model(cube_R, cube_L) ?
    - la taille des cubes : un resize ou un crop appliqué à l'entraînement ?
    - la sortie : model() renvoie-t-il un logit simple, ou un tuple dont il
      faudrait prendre le premier élément (comme les familles Mashup) ?
    - les clés du dict : 'siamese_cubes' en contient 4, tu n'en utilises que 2.
""")