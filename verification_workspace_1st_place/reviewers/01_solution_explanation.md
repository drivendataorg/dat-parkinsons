## Subagent: Reviewer 1 — Solution Explanation (1st place, TheAvengers)

Status: needs organizer review

No competitor code executed: yes (static reading only; no imports, no torch.load/joblib/np.load, weights.zip not extracted, no competitor URLs fetched). No images viewed (`1st Place/img/*.png` hashed only).

Read-only inspection performed:
- `verification_workspace_1st_place/notes/coordinator_context.md`, `notes/governing_rules.md`, `references/verification_comp_specific_rules.md`, skill `reviewers.md` §1 and `report.md`
- `1st Place/README.md`, `0_configs/default.yaml`, `5_pipeline/run_pipeline.py`, `6_runtime/pack_production.py`, `scripts/download_weights.py`, `pyproject.toml`
- `1st Place/1_initial_training/*` (data sources, pretrained-weight entry points), `2_stacking/stacking.py`, `3_soft_labeling/soft_labeling.py`, `4_siamese_training/*`
- `1st Place/competition-sfmn-parkinsons-runtime/submission_src/{main,constants}.py`, `justfile`, `runtime/entrypoint.sh`, `runtime/pyproject.toml`
- `extracted/harness_zip/main.py`, `constants.py`, headers/paths of all 10 helper scripts, `validate_pipeline.py` config/phase 6-7, CSV headers
- `extracted/src_zip/src/{inference_2d_avg,inference_feats,builder,preprocessing,striatum_preproc,inference_tta_wrappers}.py`; `notes/diff_*.txt`, `notes/*_listing.txt`, `notes/prediction_csv_check.txt`

Competition-specific checks applied:
- Extracted CS1–CS20 into `solution_overview.md` §1 with owners; dataset policy in §2.
- CS1/GR-D (independence), CS2 (harness vs tree), CS3 (clip consistency), CS4/GR-C (logging), CS7–CS9 (external data/pretrained disclosure), CS16/CS18/CS19 (license, weight hosting, data retention) traced for downstream reviewers.

Evidence:
- `HZ/main.py` L31, L37-43: scored config = `FAMILY_STACKING`, `FAMILY_MODELS=["Mashup_Internal"]`, `TEMP=1.094`; L20-22 DATA_DIR falls back to `data/`.
- `HZ/main.py` L286-289 → `SZ/inference_2d_avg.py` L861-957: loads `mashup_ensemble_exp_{0..19}_fold_{0..4}.pth` (exactly the 100 files in weights.zip), one forward each, mean of sigmoid(mean of 3 branch logits); no TTA.
- `SZ/inference_feats.py` L413-458: `p = clip(σ(logit(p̄)/1.094), 0.01, 0.99)`; consistent with 109 rows at 0.99 and min 0.01256 in scored CSV.
- `HZ/constants.py` `"TwoPhase_Mean": "twophase"` inert (prefix_map unused for this branch; not in FAMILY_MODELS).
- Executed-path code (`SiameseMashupInternalEnsemble`, `MashupInternalEnsembleCNN`, `preprocessing.py`, `LogitFamilyStackingInference`) identical between tree `src/` and `src.zip`; diffs (`builder.py` mednext, `inference_feats.TabularInference`, `inference_2d_avg` ResNet2D/ViT2D) not executed.
- `TREE/4_siamese_training/train_siamese.py` L171-215, L306: same architecture and checkpoint names as scored weights; trains on real training folds + all soft-labelled synthetic PET (L246-262); random init.
- `TREE/3_soft_labeling/soft_labeling.py` L24, L217: labels only `external/PET_GENERATED/*.nii.gz`; teachers/meta trained on training labels only.
- t-SNE (`SZ/inference_feats.py` L86-99, L268-301) only in `StackingInference.post_process`; not used in scored mode; `stacking.py` L657-669 never stores `post_processing`; would only `transform`, never fit on test.
- Pretrained weights: HF `edadaltocg/resnet50_cifar100` (`train_2d_resnet.py` L128-136), HF `edadaltocg/vit_base_patch16_224_in21k_ft_cifar100` (`train_2d_vit.py` L104-113) — Block-1 teachers only. MedicalNet loader requires `--pretrain_path` (default None, never passed by `run_pipeline.py`). Final Siamese has no pretrained weights.
- `HZ/validate_pipeline.py` L121-125, L698-700: "facteur 1.245 = correction de décalage mesurée sur le leaderboard"; submitted T "délibérément sous-optimale sur l'OOF".
- `TREE/scripts/download_weights.py` L53-56: `snapshot_download("dinhviettoanle/dat-parkinson-models")` (teacher weights, SVM.pkl, tabular_oof.csv, synthetic_soft_labels.csv, 100 Siamese .pth). Not fetched.
- `TREE/6_runtime/pack_production.py` L64-91 + `justfile` L179-188: zips whole `submission_src/*`, explaining extra helper scripts/CSVs in the harness.

Findings:
- Severity: medium
  Summary: Final temperature T=1.094 includes a factor 1.245 measured on the public leaderboard; a global scalar, not per-case, but undocumented in README and it bears on "do not overfit to the test set".
  Evidence: `extracted/harness_zip/validate_pipeline.py` L121-125, L698-700; `extracted/harness_zip/main.py` L43
  Recommendation: Code Integrity to assess; organizer to decide whether LB-tuned global calibration is acceptable (common practice); request the notebook that froze the recipe.
- Severity: high
  Summary: README license section declares materials confidential/not to be shared, conflicting with the MIT requirement; no solution LICENSE file.
  Evidence: `1st Place/README.md` L247-250
  Recommendation: Release Readiness: require MIT license before release.
- Severity: medium
  Summary: Weights trained on competition data (plus OOF/soft-label CSVs) are distributed via a competitor HF repo whose visibility is unknown.
  Evidence: `1st Place/scripts/download_weights.py` L53-56, L71-117
  Recommendation: Organizer to confirm the repo is private/organizer-only until dataset release.
- Severity: medium
  Summary: Harness zip bundles training-uid OOF CSVs with labels and diagnostic scripts with team-internal paths.
  Evidence: `extracted/harness_zip/tta_siamese_predictions.csv`, `validation_report.csv`; `copy_weights.py`, `validate_pipeline.py` L44
  Recommendation: Release Readiness: exclude from release; note under data-use rule E.
- Severity: medium
  Summary: External synthetic PET data enters the final model; its license and the generator's training-data provenance are unresolved; MedicalNet pretrained-weight use cannot be confirmed from code.
  Evidence: `1st Place/README.md` L16-17, L71-81, L183, L238-245; `src/models/medicalnet/setting.py` L46-47; `5_pipeline/run_pipeline.py` L43-49
  Recommendation: Data Source to verify Zenodo 15377683 license and generator training data; ask the team whether MedicalNet weights were used.
- Severity: low
  Summary: Tree `main.py` uses `DATA_DIR=Path()` and would not find data in the container; tree `src/` lacks vendored `libs/`; scored zip built from a different working copy (executed logic still identical).
  Evidence: `1st Place/competition-sfmn-parkinsons-runtime/submission_src/main.py` L20; `notes/diff_main.py.txt`; `notes/src_zip_listing.txt`
  Recommendation: Release Readiness: fix before publication; Code Integrity: no score impact.
- Severity: low
  Summary: No test-set pseudo-labelling, t-SNE, TTA or cross-case statistics found in the scored path; "siamois_final_pseudo" is an inactive earlier variant with no weights or training code.
  Evidence: `extracted/harness_zip/main.py` L260-261; `constants.py` L9; `extracted/src_zip/src/inference_feats.py` L86-99
  Recommendation: Code Integrity to confirm (CS1).

Open questions:
- License of Zenodo 15377683 / OneDrive synthetic PET and what data trained its generator?
- Were Tencent MedicalNet pretrained weights used in the original Block-1 run?
- Is HF `dinhviettoanle/dat-parkinson-models` private?
- How exactly were T=1.094 (incl. LB factor 1.245) and clip (0.01, 0.99) chosen? (notebook not delivered)
- Does `img/data_processing.png` show real competition scan crops? (not viewed; organizer to inspect)
- Was the competition runtime image modified (tree `runtime/pyproject.toml` lists opentsne, ptwt, pyradiomics, xgboost), or did the scored run rely only on the vendored `src/libs/`?
