# 1st place — retrain log

Machine: 1× NVIDIA L40S (46 GB), 8 vCPU, 61 GB RAM. Workspace: `~/sfmn1/` (not in git). Full logs are in `~/sfmn1/logs/`, with copies in `logs/` here.

## Setup (2026-10-08)

- Code copied from `1st Place/` (without `competition-sfmn-parkinsons-runtime/`) to `~/sfmn1/code`. The only edit: `0_configs/default.yaml` `root_dir: "/home/ec2-user/sfmn1/data"`.
- `~/sfmn1/data/niftis` → symlink to repo `niftis/` (1362 scans); `train_labels.csv` copied as `train_labels_JNDlMjr.csv`.
- `uv sync` (1m23s): torch 2.12.1+cu129, CUDA OK on the L40S. No errors.
- Test set: `~/sfmn1/submitted/data` → `~/sfmn3/data/test` (524 scans + `test_labels.csv`).

## Step 1 — authors' weights (`scripts/download_weights.py`) — OK, 44 s

- Puts 9.6 GB of Block 1 weights/OOFs in `1_initial_training/results`, `synthetic_soft_labels.csv` (1500 synthetic scans) in `3_soft_labeling/results`, and 100 Siamese `.pth` in `4_siamese_training/results/siamese_network`.
- **All 100 HF Siamese weights are byte-identical to `weights/` in the submitted zip** (`cmp`). The published weights are the shipped ones.

## Step 2 — phase 1 fold prep + phase 2 stacking on HF OOFs — OK

- `prepare_folds.py`: OK (needs the `train_labels_JNDlMjr.csv` name).
- `2_stacking/stacking.py`: OK, 4m07s. Nested OOF: AUC 0.9687, log loss 0.2231, ECE 0.0183. All 13 base models get non-zero weight. Writes `meta_model_stacking.pkl` (not in HF, needed by phase 3).

## Step 4 — Block 1 smoke test

- `prepare_2d_dataset.py`: OK, 7m42s, 6782 PNG slices.
- `train_2d_resnet.py` (authors' batch 32, 30 epochs, 5 folds) under `timeout 20m`: the HF `resnet2d/` folder was first moved to `1_initial_training/results_hf_resnet2d_backup`. It has since been moved back; the partial retrain is in `~/sfmn1/retrain_resnet2d_partial/`.
  - Killed by the 20 min cap (rc=124) at epoch 24/30 of fold 0, about 50 s/epoch on the shared box. No errors.
  - Fold 0 best patient-level val log loss **0.2935** (AUC 0.9532) at epoch 23, vs the authors' HF `oof_resnet2d.csv` on the same fold: **0.2918** (AUC 0.9616). The learning curve and scale match.
  - The HF ResNet2D and ViT2D OOFs have 1361 rows; `ude358tr` is missing. Our `prepare_2d_dataset.py` run also produces no slices for it (the intensity filters drop every slice; no error is printed), so this matches the authors' behavior.
- `train_medicalnet.py` MedicalNet50 under `timeout 20m` (HF folder moved to `1_initial_training/results_hf_medicalnet50_backup`; still there, so **restore it before phase 3**).
  - Builds `data/vol_cache` first, then trains. Killed by the cap during fold 0 (the log's "FOLD 1/5") at epoch 6/50. No errors; `medicalnet50_fold_0.pth` saved.
  - Best val loss 0.3545 at epoch ~3, vs the authors' final fold-0 OOF 0.2452. Too early to compare; this shows only that the 3D path trains.

## Step 4b — Block 1 crash test: the 11 remaining training scripts (2026-10-09) — all start and train

Goal: show that every Block 1 training script runs ("does it start and train without error"), not to measure result quality.
Script: `scripts/crashtest_block1.sh`, run from `~/sfmn1/runs/crashtest`.

- **Isolated config.** `phase1_out` points at `~/sfmn1/runs/crashtest/results`, so the published weights are untouched. `train_labels` and `folds_2d_csv` point at a **stratified 100-scan subset** (20 per fold, 55% positive vs 54.8% overall; `outputs/crashtest_subset_train_labels.csv`).
- **Unchanged:** code, batch sizes, epochs and hyperparameters are as the authors wrote them. The subset only makes data loading/caching short enough that each job reaches real training steps within its cap.
- Each job got a hard 3-min cap (`timeout 180`). `OMP_NUM_THREADS=2` was set to limit CPU thread oversubscription.
- **Parallelism.** All 11 jobs were launched at once first. The GPU (46 GB) filled up (peak 45.4 GB), and the jobs that hit CUDA OOM, or stalled from CPU starvation, were relaunched in smaller parallel groups (3, then 4, then 2). Total wall time: 13 min.

| script / model | wave | outcome within 3 min | checkpoint written |
|---|---|---|---|
| `train_svm.py` (SVM) | 1 (all 11) | **ran to completion** (feature extraction, 5-fold CV, final fit) | `tabular_models/SVM.pkl`, `tabular_oof.csv` |
| `train_3d_densenet121.py` | 1 | trained 4 epochs (fold 0), capped | `densenet3d_fold_0.pth` |
| `train_medicalnet.py` MedicalNet18 | 1 | trained 18 epochs (fold 0), capped | `medicalnet18_fold_0.pth` |
| `train_medicalnet.py` MedicalNet34 | 1 | trained 14 epochs (fold 0), capped | `medicalnet34_fold_0.pth` |
| `train_3d_attention.py` (DenseNet-attention) | 2a (3 jobs) | trained 9 epochs, capped (wave 1: GPU OOM, contention) | `densenet3d_attn_fold_0.pth` |
| `train_2d_vit.py` (ViT2D) | 2a | fold 0 complete, fold 1 at epoch 27/30, capped (wave 1: GPU OOM) | `vit2d_fold_0.pth`, `vit2d_fold_1.pth` |
| `train_medicalnet.py` ResNeXt50 | 2a | trained 13 epochs, capped (wave 1: GPU OOM) | `medicalnet-resnext50_fold_0.pth` |
| `train_medicalnet.py` ResNeXt101 | 2b (4 jobs) | trained 11 epochs, capped (wave 1: GPU OOM) | `medicalnet-resnext101_fold_0.pth` |
| `train_medicalnet.py` MedNeXtB | 2b | trained 7 epochs, capped (wave 1: GPU OOM) | `medicalnextB_fold_0.pth` |
| `train_3d_densenet201.py` | 3 (2 jobs) | trained 14 epochs, capped (wave 1: stalled at RAM caching under CPU load, killed; wave 2b: GPU OOM) | `densenet3d_fold_0.pth` |
| `train_medicalnet.py` MedicalNet200 | 3 | trained 12 epochs, capped (waves 1 and 2b: GPU OOM) | `medicalnet200_fold_0.pth` |

- **No code errors in any script.** Every failure was CUDA OOM from sharing one 46 GB GPU, plus one CPU-starved cache stall, and each was resolved by running fewer jobs at once. Peak GPU use: wave 2a 40.0 GB (3 jobs), wave 3 34.1 GB (2 jobs).
- **Confirms the naming hard error** from `repro_issues.md`: DenseNet121 and DenseNet201 both save `densenet3d_fold_0.pth`, and MedNeXtB saves `medicalnextB_fold_0.pth`, none of which match the `default.yaml` prefixes that phases 2/3 load.
- Combined with steps 4 (ResNet2D, MedicalNet50), **all 13 Block 1 models (12 nets + SVM) have now trained without error**. Logs: `logs/crashtest/`.

## Step 0 — submitted package on test — OK, reproduces the official score

- `~/sfmn1/submitted` = unzipped `1st_place_submission.zip`; `main.py` run with the training venv; `data` → test set.
- 79 min on the shared box (single process; main.py computes every input type, radiomics included, though only the Siamese patches are used).
- **Test log loss 0.253234 vs official 0.253233**; max per-scan |diff| from the official csv is 2.7e-5.

## Step 5 — phase 4 prep (real) — OK

- `prepare_2d_dataset_siamese.py --mode real` and `prepare_2d_dataset_striatum.py --mode real`: 1362/1362 scans each, 56 min total (CPU-bound, run alongside the other jobs).
- Synthetic mode is blocked until the synthetic PET data arrives.

## Fast test scorer — `scripts/score_siamese_test.py`

- Caches only the Siamese inputs for the 524 test scans (6 min), writes per-model test probabilities, and applies the submission's recipe `clip(sigmoid(logit(mean p)/1.094), .01, .99)`. It can swap in retrained models.
- Submitted 100 weights through this path: **0.253234, max |diff| from main.py output = 1e-16**. The swap test is exact.
- Submitted per-model test log loss across the 100 models: mean 0.2712, min 0.2468, max 0.3190 (the noise band for single-model comparisons).

## Step 6 — Block 2 worker smoke test (no synthetic data)

- `train_siamese.py --worker --exp 0 --fold 0 --device cuda:0 --no-synth`, authors' hyperparameters (batch 32, 150 epochs, patience 60). Run from `~/sfmn1/runs/siamese_nosynth`, which has its own config with `phase4_out` redirected, so the HF/submitted weights are not overwritten.
- **2m39s**, early-stopped at epoch 70, val log loss 0.2585, AUC 0.9611. No errors.
- `scripts/eval_siamese_val.py` re-evaluates any .pth on its own (exp, fold) val split with the trainer's code. It reproduces the worker's 0.2585 exactly.
- Paired comparison with the submitted `exp_0_fold_0` (trained **with** synthetic data, so not like-for-like):

| exp 0 fold 0 | val ll (273 scans) | val AUC | test ll (single model) |
|---|---|---|---|
| submitted (synthetic) | 0.2462 | 0.9633 | 0.2821 |
| retrain, no synthetic | 0.2585 | 0.9611 | 0.2920 |

  Test logit correlation is 0.9885. The synthetic-data retrain of the same ids is the real check, pending the data.

## Synthetic data (2026-10-08, ~23:20)

- User fetched it from SharePoint as `~/sfmn1/synthetic_data.zip` (13.4 GB). Checked before extracting: 1500 flat `*.nii.gz` files, no other files or paths. The names match the HF `synthetic_soft_labels.csv` filenames exactly.
- Extracted to `~/sfmn1/data/external/PET_GENERATED/` (2 min, 13 GB).

## Step 5b — phase 4 prep (synthetic) — OK

- `prepare_2d_dataset_siamese.py --mode synthetic` + `prepare_2d_dataset_striatum.py --mode synthetic`: 1500/1500 each, 22 min, rc=0.
- Uses the **HF** `synthetic_soft_labels.csv`. Checked: all 1500 `label` values in `dataset_synthetic_siamese_patches/folds_siamese.csv` equal the HF `is_pathologic` exactly (mean 0.6175).

## Step 3 — phase 3 soft labeling (HF Block 1 weights + rebuilt meta-model) — runs; 11/13 base models reproduce

- `soft_labeling.py paths.phase3_out=/home/ec2-user/sfmn1/runs/phase3/results` (an OmegaConf CLI override, so the HF csv is not overwritten; its md5 `ead1abb5…` is unchanged afterwards).
- All 13 base models load with no `DummyInference` fallbacks and 0 per-scan errors. **1h29m** for 1500 scans (3–6 s/scan, depending on load). Output: `outputs/synthetic_soft_labels_rerun.csv`.
- Comparison with the HF `synthetic_soft_labels.csv` (`outputs/phase3_compare.txt`; the HF csv uses older column names `prob_MedicalResnext50/101`):

| column | mean \|Δ\| | max \|Δ\| | corr |
|---|---|---|---|
| SVM, MedicalNet18/34/50/200, ResNeXt50/101, MedNeXtB, DenseNet121/201/Attn (11 models) | ≤ 3e-5 | ≤ 1.5e-4 | 1.000000 |
| **ResNet2D** | 0.113 | 0.375 | 0.723 |
| **Vit2D** | 0.060 | 0.331 | 0.922 |
| prob_meta_A / prob_meta_B | 0.032 / 0.022 | 0.095 / 0.061 | 0.9955 / 0.9981 |
| **is_pathologic (final soft label)** | **0.030** | **0.085** | **0.9961** |

  Hard-label (≥0.5) agreement 98.3%; mean soft label 0.646 (rerun) vs 0.618 (HF).
- **Root cause of the 2D gap** (`scripts/probe_2d_softlabels.py`):
  - The local 2D weights are byte-identical to HF (LFS sha256 checked).
  - HF ResNet2D fold-0 weights + released `src/` reproduce HF `oof_resnet2d.csv` exactly (|Δ| = 0.0000 on 20 fold-0 scans), so the published 2D weights and OOFs go together.
  - The same weights give 2D probabilities on the synthetic scans that differ from the HF soft-label csv by about 0.1. The submission-era `src.zip` 2D classes (torchvision ResNet50 / vit_b_16) **cannot load** the HF 2D weights (state_dict mismatch).
  - So the published `synthetic_soft_labels.csv` 2D columns came from an earlier version of the 2D models/code, not from the published 2D weights. The 11 3D/tabular models are fully consistent.
- The rebuilt meta-model (our phase 2) also enters `prob_meta_*`. With 11/13 inputs identical, the meta differences are dominated by the 2D inputs.
- Block 2 was retrained on the **HF** soft labels (what the authors trained on). The sensitivity to our re-generated labels is checked below (step 6c).

## Step 6b — Block 2 retrain WITH synthetic data, exp 0 folds 0–4 (`scripts/run_exp0.sh`)

Authors' hyperparameters, run sequentially from `~/sfmn1/runs/siamese_synth` (own config, `phase4_out` redirected).

| model | train time | stop epoch | val ll retrain / submitted | val AUC retrain / submitted | test ll retrain / submitted | test logit corr | ref corr (same fold, other exps) |
|---|---|---|---|---|---|---|---|
| exp0 fold0 | 5m25s | 83 | 0.2460 / 0.2462 | 0.9631 / 0.9633 | 0.2777 / 0.2821 | **0.9995** | 0.980–0.994 |
| exp0 fold1 | 4m36s | 70 | 0.2219 / 0.2234 | 0.9715 / 0.9707 | 0.2589 / 0.2578 | **0.9996** | 0.976–0.993 |
| exp0 fold2 | 6m33s | 101 | 0.1690 / 0.1680 | 0.9844 / 0.9844 | 0.2911 / 0.2626 | 0.9884 | 0.985–0.994 |
| exp0 fold3 | 5m22s | 83 | 0.1808 / 0.1821 | 0.9803 / 0.9798 | 0.2687 / 0.2680 | **0.9994** | 0.983–0.992 |
| exp0 fold4 | 4m56s | 76 | 0.2207 / 0.2201 | 0.9704 / 0.9708 | 0.2627 / 0.2641 | **0.9997** | 0.984–0.994 |

Val numbers are from `scripts/eval_siamese_val.py` on each model's own 272/273-scan split (submitted weights' stop epochs are not shipped). Test numbers are single-model log loss on all 524 test scans.

- **4/5 folds are near-identical reproductions** (test logit corr ≥ 0.9994, mean |Δp| 0.006–0.008, far above seed-to-seed similarity).
- **Fold 2** matches on val (0.1690 vs 0.1680) but differs on test. Its val curve (`~/sfmn1/runs/siamese_synth/results/siamese_network/loss_curves/loss_curve_exp_0_fold_2.png`) has many near-tied minima (≈0.168–0.172 at epochs ~2, 12, 17, 24, 38, 40). Our run kept epoch 41 (stop 101 = 41 + patience 60). Small GPU nondeterminism (the code sets `cudnn.deterministic` but not `torch.use_deterministic_algorithms`) can make the original run keep a different one of those minima. Even so, it's within the seed-to-seed band.
- Exp-0 5-model mean on test: submitted 0.2574, retrained 0.2620.
- All 5 retrained exp-0 models swapped into the 100-model ensemble: test log loss **0.253478** (submitted 0.253234, +0.00024). Ranking unaffected (2nd place 0.273088).
- Total training time for 5 models: 26.9 min on 1× L40S, run sequentially while phase 3 shared the box. That implies ~9 h for all 100 sequentially here (README: ~30 min with 16 concurrent workers on 2 GPUs).

- (Fold 0 alone swapped in: 0.253212.)

## Step 6c — sensitivity: exp0 fold0 retrained on OUR re-generated soft labels

Separate dataset dir `~/sfmn1/runs/siamese_synth_ourlabels/synth_siamese` (patches symlinked; only `label` replaced by our phase-3 `is_pathologic`). 5m15s.

| exp 0 fold 0 | val ll | val AUC | test ll | test logit corr vs submitted | mean \|Δp\| |
|---|---|---|---|---|---|
| submitted | 0.2462 | 0.9633 | 0.2821 | — | — |
| retrain, HF soft labels | 0.2460 | 0.9631 | 0.2777 | 0.9995 | 0.0080 |
| retrain, our soft labels | 0.2435 | 0.9641 | 0.2854 | 0.9993 | 0.0107 |
| retrain, no synthetic | 0.2585 | 0.9611 | 0.2920 | 0.9885 | 0.0414 |

The 2D soft-label discrepancy has a negligible effect on the final Siamese model. Swapping in our-labels exp0 fold0 gives an ensemble test log loss of 0.253254 (submitted 0.253234).

## Conclusion (2026-10-09)

- **Inference:** the submitted package reproduces the official 0.253233 (0.253234). The published HF Siamese weights are the shipped weights, byte for byte.
- **Block 2 (what the submission consists of):** the training code, run as documented with the authors' hyperparameters and their published soft labels, reproduces the shipped models.
  - 4/5 retrained exp-0 models are near-identical (test logit corr ≥ 0.9994); one (fold 2) lands on a different, near-tied early-stopping minimum.
  - Replacing all 5 in the 100-model ensemble moves test log loss by +0.00024 (0.253478).
- **Block 1:**
  - Training code runs (ResNet2D and MedicalNet50 20-min smoke tests; ResNet2D fold-0 val loss tracks the authors' OOF). Stacking and soft labeling run end-to-end on the published Block 1 weights.
  - 11/13 models reproduce the published soft labels to ~1e-4. The two 2D models don't: the published csv was made with earlier 2D weights. Effect on the final soft labels: mean 0.03. Effect on a Block 2 model trained on them: negligible.
- **Block 1 crash test (step 4b):** all 11 remaining training scripts start and train without code errors (100-scan subset, 3-min caps).
- **Not run (by design):** full 12-model Block 1 training to convergence (~3.5 days on their hardware), the other 95 Siamese models (~8 h here), and `pack_production.py` (needs `just`).
