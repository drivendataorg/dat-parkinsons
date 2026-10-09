# 1st place (TheAvengers) — retrain verification: plan

Status (2026-10-09): **done.** All planned stages ran; see the Conclusion at the end of `retrain_log.md`. Results go in `retrain_log.md`, the fix list in `repro_issues.md`, and helper code, logs and small outputs in `scripts/`, `logs/` and `outputs/`.

## What the submitted model actually is

- The harness `main.py` runs `MODE_INFERENCE="FAMILY_STACKING"` with `FAMILY_MODELS=["Mashup_Internal"]` and `TEMP=1.094`.
  That means one family: **100 Siamese "mashup" CNNs** (`weights/mashup_ensemble_exp_{0..19}_fold_{0..4}.pth`, 20 exps × 5 folds), whose logits are averaged, scaled by the temperature, and clipped to [0.01, 0.99].
- The submission uses no Block 1 weights. Block 1 (12 nets + SVM → stacking → soft labels) only makes `synthetic_soft_labels.csv`, and that file becomes extra training data for Block 2.
- Official private score: 0.253233 (full 524-scan test set).

## How the pieces fit together

```
Block 1  phase1_prep   prepare_folds.py, prepare_2d_dataset.py         -> train_labels_with_folds.csv, dataset_2d/
         phase1_train  12 x train_*.py (5 folds each) + train_svm.py   -> 1_initial_training/results/<model>/{*.pth, oof_*.csv}
         phase2        2_stacking/stacking.py                          -> 2_stacking/results/meta_model_stacking.pkl
         phase3        3_soft_labeling/soft_labeling.py (synthetic PET)-> 3_soft_labeling/results/synthetic_soft_labels.csv
Block 2  phase4_prep   prepare_2d_dataset_{siamese,striatum}.py  real + synthetic -> dataset_*_patches/
         phase4_train  train_siamese.py (100 subprocess workers)       -> 4_siamese_training/results/siamese_network/*.pth
Pack     6_runtime/pack_production.py -> src.zip + weights.zip -> just pack/check/test (docker)
```

- The orchestrator is `5_pipeline/run_pipeline.py --start-at <phase>`. Every script reads `0_configs/default.yaml` relative to the cwd, so all scripts must run from the code root.
- The authors' Hugging Face repo `dinhviettoanle/dat-parkinson-models` (fetched by `scripts/download_weights.py`) holds every Block 1 fold weight, the OOF csvs, `SVM.pkl`, `tabular_oof.csv`, `synthetic_soft_labels.csv` and the 100 Siamese weights.
  It does **not** hold `meta_model_stacking.pkl`, so phase 3 needs phase 2 to run first. Phase 2 only uses the OOF csvs, so it should be fast.
- The synthetic PET data (`external/PET_GENERATED`) is only on a SharePoint/OneDrive folder linked from github.com/Krying/PD_SSL_ZOO. The Zenodo record has only the code zip. It is not in our S3 bucket.

## Training for one (exp, fold) is deterministic, so it can be checked in isolation

`train_siamese.py --worker --exp E --fold F --device cuda:0` trains one model. The split comes from `np.roll` over the sorted uids, and the seed is `42 + 10E + F`.
A retrained `exp_E_fold_F` can therefore be compared with the submitted `mashup_ensemble_exp_E_fold_F.pth` directly: same val uids and same test scans.

## Planned stages (cheapest first)

| # | Stage | What it proves | Est. cost on 1× L40S, 8 CPU |
|---|---|---|---|
| 0 | Re-run the submitted package on the 524 test scans | Inference env works; reproduce 0.253233 | ~min |
| 1 | `download_weights.py`, then phase 2 stacking on the HF OOFs | Stacking code runs; meta-model rebuilt | min |
| 2 | Phase 3 soft labels from HF Block 1 weights + rebuilt meta-model; diff against HF `synthetic_soft_labels.csv` | Block 1 → soft labels is reproducible | ~1 h (README) |
| 3 | Block 1 smoke test: phase1_prep, then ~20 min each of ResNet2D / MedicalNet50 (+ SVM if quick), then kill | Block 1 training code runs | ~20–60 min |
| 4 | phase4_prep (real + synthetic) using the **HF** soft labels | Patch prep runs | small |
| 5 | Train a subset of the Siamese (exp, fold) workers at the authors' settings; compare with the submitted weights of the same id (val loss, test logits) | Block 2 training reproduces the shipped weights | per-fold time to be measured |
| 6 | Swap the retrained weights into the package and re-score on test (all 100 if the time is reasonable) | End-to-end: training → usable inference model with ~same score | depends on #5 |

Bulky artifacts (venv, data copy, results) go in `~/sfmn1/` (outside git, like `~/sfmn3/` for 3rd place). The code is copied there because the scripts write results into the code tree.
