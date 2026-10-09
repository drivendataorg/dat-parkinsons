# 1st place — repro issues (fix list for open-source release)

## Hard errors

- **Phase-1 output names don't match what phases 2/3 look for (full 12-model from-scratch run).** Found by reading the code; not hit in our runs, because `download_weights.py` renames the HF files to the expected names.
  - `train_3d_densenet121.py` saves `densenet3d_fold_{k}.pth`; config/loader expect prefix `densenet3d_121` → phase 3 `FileNotFoundError`.
  - `train_3d_densenet201.py` saves `densenet3d_fold_{k}.pth` and `oof_densenet3d.csv`; config expects `densenet3d_201_fold_{k}.pth` and `oof_densenet3d_201.csv`.
  - `train_medicalnet.py --run_name medicalnextB` saves `medicalnextB_fold_{k}.pth` and `oof_medicalnextB.csv`; config expects prefix `mednext_b` and `oof_mednextB.csv`.
  - **Failure is silent.** `stacking.py` prints ❌ and drops a model whose OOF is missing. `soft_labeling.py` catches the load error and substitutes `DummyInference` (constant 0.5), so the run "succeeds" with wrong soft labels.
  - The README's 3-model alternative (ResNet2D, MedicalNet50, SVM) is unaffected; those names line up.
  - Fix: make the save names match `default.yaml` (or vice versa), and make phase 3 fail hard on a missing model.
  - **Confirmed by running crash test:** `train_3d_densenet121.py` and `train_3d_densenet201.py` both wrote `densenet3d_fold_0.pth`, and MedNeXtB wrote `medicalnextB_fold_0.pth`.

## Doc inaccuracies / gaps

- README says `python scripts/download_weights` (missing `.py`).
- README data layout expects `train_labels_JNDlMjr.csv`; the file we were given is `train_labels.csv` (prepare_folds.py hard-codes the README name).
- `meta_model_stacking.pkl` is not in the HF weights repo, so "start from phase3 with downloaded weights" needs phase2 to run first (README table implies phase3 can start directly).
- Synthetic data: README links Zenodo 15377683 (code zip only) and an OneDrive folder; the data itself is only on OneDrive/SharePoint.
- Submitted `main.py` takes ~79 min for 524 scans on 8 vCPU. It builds every input type (radiomics features, 2D/3D slices) even though FAMILY_STACKING uses only the Siamese patches. This is not wrong, but it is slow; a release could trim `prepare_all_inputs`.
- `main.py` hard-codes `DATA_DIR = /data2/iceberg/dat_parkinson/data-demo` (falls back to `./data`).
- All phase scripts write results inside the code tree (`phase*_out` are relative paths) and read `0_configs/default.yaml` from the cwd. To redirect outputs (e.g. not overwrite downloaded weights) you have to run from another dir with its own config copy; `train_siamese.py` also can't take OmegaConf CLI overrides because it parses `--worker` etc. from the same argv.
- `scripts/download_weights.py` moves existing `results/` dirs to timestamped backups (good), but phase-1 training overwrites the downloaded weights in place, with no warning.
- `prepare_2d_dataset.py` silently yields 0 slices for `ude358tr` (intensity filters), so 2D OOFs have 1361 rows. This matches the authors' HF OOFs, but is worth a log line.
- `6_runtime/pack_production.py` needs `just` and `uvx rpzip` (not installed here; not run).
- Siamese training is not bitwise deterministic. `seed_everything` sets `cudnn.deterministic=True` but not `torch.use_deterministic_algorithms(True)`/`CUBLAS_WORKSPACE_CONFIG`, so retrains match closely but not exactly (4/5 exp-0 folds have test logit corr ≥ 0.9994; fold 2 lands on a different near-tied early-stopping minimum). This doesn't block anything; it's worth a note in the README's reproducibility section.
- Siamese per-(exp, fold) stop epochs / val losses (`fold_statistics.csv`, `tmp_preds/`) are not published with the weights, so a retrain can only be compared by re-evaluating the shipped weights.
- **Published `synthetic_soft_labels.csv` is not reproducible from the published Block 1 weights for the two 2D models.** Rerunning phase 3 with the HF weights reproduces 11/13 model columns to ~1e-4, but ResNet2D (corr 0.72) and ViT2D (corr 0.92) differ. The HF 2D weights reproduce their own OOF csvs exactly, and the submission-era 2D inference classes can't load them, so the csv was evidently made with earlier 2D models. Final soft labels: mean |Δ| 0.030, max 0.085. Ask the authors which 2D weights produced the csv, or publish them. The README presents the csv and weights as one consistent set.
- HF `synthetic_soft_labels.csv` uses column names `prob_MedicalResnext50/101`; the current code writes `prob_MedicalNetResNext50/101` (cosmetic, but it breaks naive diffs).
- 3D Block 1 trainings are GPU-memory heavy at the authors' batch sizes (128³ volumes; DenseNet batch 16–32): at most ~3–4 of them fit on one 46 GB GPU at once. The DenseNets use 16 RAM-caching workers each and can stall on small CPUs (8 vCPU). The README's hardware (2× 96 GB, 16 cores) hides this; worth stating minimum hardware per model.
