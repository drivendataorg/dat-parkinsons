# Reviewer 5 — Release Readiness — 1st place (TheAvengers)

Static, read-only review. No competitor code executed or imported; no artifacts loaded/unpickled; `weights.zip` not extracted; no competitor-hosted URL fetched (HF repo not visited); **no images viewed**. Path conventions: `TREE/` = `1st Place/`; `HZ/` = `verification_workspace_1st_place/extracted/harness_zip/`; `SZ/` = `verification_workspace_1st_place/extracted/src_zip/src/`.

Release expectations (references/verification_comp_specific_rules.md L67-71): MIT license; all required software OSS permitting commercial use; deliver code + docs + Winning Model Documentation; release to BOAS; **code only** published at announcement; comp-trained weights may only be held in a private, organizer-only location. Governing rule E: data used solely for purpose/duration of competition.


## Summary of severities

| ID | Finding | Severity |
|---|---|---|
| RR-1 | README "License" section is confidential/no-sharing; no MIT LICENSE for the solution | **Release-blocking** |
| RR-2 | Comp-trained weights + training-label-bearing files hosted on HF `dinhviettoanle/dat-parkinson-models`; visibility unknown | **Release-blocking until organizer confirms private/organizer-only** |
| RR-3 | Harness zip ships training-uid + training-label CSVs and team diagnostic scripts | **Release-blocking for any release that reuses the harness zip**; data-retention (rule E) item |
| RR-4 | Vendored third-party code (MedicalNet, 3D-ResNets, MedNeXt, 7 libs in `src/libs/`) without license/notice files | **Release-blocking (attribution)** for Apache-2.0/BSD/EUPL notices; licenses themselves OSI/commercial-OK |
| RR-5 | Team-internal absolute paths / usernames / hostnames-like mounts | Non-blocking cleanup (no secrets) |
| RR-6 | Reproduction from tree diverges from scored zip (libs vendoring not reproducible, DATA_DIR, missing helper scripts, label filename, README command typos) | Non-blocking for release, **medium** for reproducibility |
| RR-7 | Temperature T=1.094 incl. leaderboard factor 1.245 and clip [0.01,0.99] undocumented; deriving notebook not delivered | Documentation gap (medium); integrity aspect owned by Code Integrity |
| RR-8 | Winning Model Documentation not in tree; external data / pretrained provenance incomplete | Documentation gap (medium; organizer to confirm delivery) |
| RR-9 | Packaging bloat: `__pycache__`, 190 MB libxgboost.so, unused libs, dead model families | Non-blocking cleanup |
| RR-10 | `img/*.png` not viewed; may depict real scan crops | Organizer to inspect before public release |

No secrets, tokens, credentials, API keys or emails found.

---

## RR-1 License (release-blocking)

- `1st Place/README.md:247-250`: "These materials are shared solely for the verification process ... should not be used, reproduced, distributed, disclosed, or shared with any third party ..." — directly conflicts with the MIT release requirement (rules file L67) and BOAS open-source release (L69).
- No `LICENSE` file at `1st Place/` root (file listing: only `competition-sfmn-parkinsons-runtime/LICENSE`). `1st Place/pyproject.toml:4` description is the uv placeholder "Add your description here"; no `license` field.
- `1st Place/competition-sfmn-parkinsons-runtime/LICENSE:1-3` is "MIT License / Copyright (c) 2023 DrivenData" — DrivenData's runtime-repo license, not the team's grant for their solution code. It must not be read as the team licensing their solution.
- **Required change:** team replaces README L247-250 with an MIT statement and adds a root `LICENSE` (MIT, copyright the team members), at release time. Organizers may reasonably accept the confidentiality wording as applying only to the verification period, but the release package must carry MIT.

## RR-2 Competition-trained weights hosting (release-blocking pending organizer check)

- `1st Place/scripts/download_weights.py:53-56` `snapshot_download(repo_id="dinhviettoanle/dat-parkinson-models", local_dir=tmp_dir)` — **no `token=` argument**, so the README flow (L85-96) only works if the repo is public or the user's local HF login has access. This suggests (not proves) the repo may be public. Not fetched (competitor host).
- Contents inferred from the dispatcher: `01-soft_labeling/` teacher folders (`*.pth` per fold, plus per-model OOF CSVs, e.g. `oof_mednext-B.csv` L95-98), `SVM.pkl`, `tabular_oof.csv` (L74-79; per `0_configs/default.yaml`/README L134 this is the SVM training OOF → training uids + labels), `synthetic_soft_labels.csv` (L81-84); `02-siamese/` = the 100 Siamese `.pth` (L110-115). All except `synthetic_soft_labels.csv` are trained on/derived from competition training data; OOF CSVs likely carry training labels.
- Release expectation (rules L71): weights trained on competition data may only be held privately, organizer-only. Governing rule E: data used solely for the competition.
- **Organizer action:** check visibility/access list of `huggingface.co/dinhviettoanle/dat-parkinson-models`. If public → require the team to make it private (organizer-only access) or remove it, and to remove OOF/label CSVs. For the code-only public release, `scripts/download_weights.py` and README L85-96 should be removed or repointed to an organizer-controlled private location.

## RR-3 Label-bearing / data-derived files (release-blocking for harness reuse; rule E)

- `extracted/harness_zip/tta_siamese_predictions.csv` (1,499,219 B, 16,344 rows): header `uid,label,fold,tta_name,prob_tta,prob_original_oof,attn_*` — training uids with **training labels** (row 2: `0224wk0y,1.0,...`).
- `extracted/harness_zip/validation_report.csv` (65,855 B, 500 rows): header `uid,famille,exp,fold,p_oof,p_rt,label,...` — training uids with **training labels**.
- Coordinator verified 0 test uids in either file; these are not leakage, but are competition-data derivatives that must not be published.
- Diagnostic scripts in harness also reference training-data derivatives on team servers: `compare_inputs.py:19-20` (`all_masks.pt`, `spine_profiles_real.csv`), `generate_tta_oof.py:39`, `check_tta_augmentations.py:18` (`folds_siamese.csv`).
- The delivered tree itself contains **no** CSV/PKL/PTH/PT/NIfTI/ipynb files (find over `1st Place/`), so the tree is clean in this respect; the issue is only in the harness zip and HF repo.
- **Required change:** release only the tree (after RR-1/4/5 fixes), never the harness zip; team should delete retained label-bearing outputs after the competition per rule E (organizer to decide how strictly to apply post-competition retention).

## RR-4 Third-party code and vendored libraries (attribution)

Vendored model code, no license headers or NOTICE files retained (header check of every file):
- MedicalNet (Tencent, MIT per upstream — to confirm): `src/medicalnet/{model,resnet,resnext}.py`, `src/models/medicalnet/*`, `1_initial_training/models/medicalnet/*`; `setting.py:1-3` keeps only "Written by Whalechen".
- 3D-ResNets-PyTorch (kenshohara, MIT per upstream — to confirm): ResNeXt backbone (`resnext.py`), plus team-authored `src/medicalnet/attn_resnext.py`.
- MedNeXt (MIC-DKFZ, Apache-2.0 per upstream — to confirm): `src/mednext/*`, `src/models/mednext/*`, `1_initial_training/models/mednext/*`. Apache-2.0 §4 requires retaining the license and stating changes.
- Attribution exists only as README links (L239-241). **Required change:** add upstream LICENSE copies (e.g. `THIRD_PARTY_LICENSES` or per-folder LICENSE) and headers.
- Code is triplicated (`src/medicalnet`, `src/models/medicalnet`, `1_initial_training/models/medicalnet`; same for mednext) — non-blocking cleanup.

Vendored libraries inside harness `src.zip` → `src/libs/` (216.3 MB of 217.1 MB; `notes/src_zip_listing.txt`): openTSNE (BSD-3, to confirm), ptwt (EUPL-1.2, to confirm — OSI-approved weak copyleft, commercial use OK), pykwalify (MIT), pywt/PyWavelets (MIT), radiomics/pyradiomics (BSD-3), ruamel.yaml (MIT), xgboost 3.3.0 (Apache-2.0 — confirmed `libs/xgboost-3.3.0.dist-info/METADATA`) + bundled `xgboost.libs/libgomp-*.so` (GCC runtime, GPL-3 with Runtime Library Exception; sbom `dist-info/sboms/auditwheel.cdx.json`). Only xgboost carries a dist-info; no LICENSE files for the other six. All appear OSI and commercial-use compatible (CS15 OK subject to confirmation). **Recommendation:** in the release, do not vendor; declare them as dependencies (they already appear in `competition-sfmn-parkinsons-runtime/runtime/pyproject.toml:58-64`).

Dependency notes: `1st Place/pyproject.toml:15` and runtime `pyproject.toml:62` pull pyradiomics from an unpinned git URL (non-reproducible); no root `uv.lock`; `huggingface_hub` (used by `scripts/download_weights.py:9`, `train_2d_resnet.py`, `train_2d_vit.py`) is not a direct dependency (only transitive via timm).

## RR-5 Private paths, usernames, secrets

- Secrets/tokens/keys/emails: **none** (grep for api_key/secret/token/password/`hf_…`/`ghp_`/`AKIA`/wandb/email patterns over tree, harness root, `src.zip` text). `competition-sfmn-parkinsons-runtime/.github/workflows/ci.yml:89-91` references `${{secrets.AZURE_*}}` placeholders — DrivenData's template, no values.
- Team-internal absolute paths / Linux usernames (`dvtle`, `sansel`) / server mount `/data2/iceberg`:
  - Tree: `competition-sfmn-parkinsons-runtime/submission_src/test_inference.py:28` (`/data/HOME/sansel/dat-park/runs_siamois/.../master_oof_predictions.csv`). Rest of tree clean (`0_configs/default.yaml:6` uses `<your_data_dir>` placeholder).
  - Harness: `main.py:20` (`/data2/iceberg/dat_parkinson/data-demo`), `test_inference.py:17,24`, `validate_pipeline.py:38,44`, `validate_inference.py:39,44-45`, `diagnose3d.py:31,36`, `generate_tta_oof.py:39-40`, `check_tta_augmentations.py:18-19`, `compare_inputs.py:16,19-20`, `copy_weights.py:8-60` (~45 lines of `/data/HOME/dvtle/...`, `/data/HOME/sansel/...`, incl. `runs_2d/samuel`).
  - Usernames match the public DrivenData handles (README L3) — low sensitivity; HF account `dinhviettoanle` (download_weights.py:54) is a personal account name.
- **Cleanup:** remove `test_inference.py` hard-coded path in tree (non-blocking); harness scripts are not part of the release.

## RR-6 Reproducibility for a fresh verifier

Inference entrypoint:
- Tree `submission_src/main.py:20-22`: `DATA_DIR = Path()` always exists → reads `./niftis` and `./submission_format.csv`; fallback to `data/` never triggers. README L170 ("By default it will infer nii.gz files in `data/niftis`") is wrong for the tree; in the container (cwd `/code_execution`, data under `./data`) the tree main.py would not find data. Harness `main.py:20` works only because `/data2/...` does not exist. **Fix:** `DATA_DIR = Path("data")`.
- **Vendored libs not reproducible from the tree (medium):** scored code imports `ptwt` (`SZ/builder.py:10`), `xgboost` (`SZ/inference_feats.py:16`), `radiomics` (`SZ/preprocessing.py:27`) at module level, and main.py prepends `./src/libs` (`HZ/main.py:74`, tree `main.py:67`). The official runtime (organizer copy `/Users/ishashah/DrivenData/competition-sfmn-parkinsons-runtime-private/runtime/pyproject.toml`) lacks opentsne/ptwt/pykwalify/PyWavelets/pyradiomics/ruamel.yaml/xgboost — the team's runtime copy adds them (`1st Place/competition-sfmn-parkinsons-runtime/runtime/pyproject.toml:58-64`, diffed). `6_runtime/pack_production.py:64-66,90` zips `TREE/src/`, which has no `libs/`; so a fresh rebuild would pass the team's locally-modified Docker smoke test but would `ImportError` on the official platform image. The step that populated `src/libs/` is undocumented.
- Scored zip built from a different working copy: `src.zip` has `libs/`, lacks `src/models/`; `builder.py`, `inference_feats.py`, `inference_2d_avg.py` differ from tree (not on executed path, per solution_overview §3); harness `main.py`, `constants.py`, `test_inference.py`, `validate_pipeline.py` differ from tree; 8 helper scripts (check_orientation, check_tta_augmentations, compare_inputs, copy_weights, diagnose3d, generate_tta_oof, run_siamese_tta, validate_inference) absent from tree. None needed for scored inference.
- Training:
  - Label filename: `1_initial_training/prepare_folds.py:12` hard-codes `train_labels_JNDlMjr.csv` (README L60, L80); the organizer copy is `data/final/public/train_labels.csv`. Fresh verifier must rename/copy. Minor.
  - README table L121-126 shows `python run_pipeline.py ...` but the script is at `5_pipeline/run_pipeline.py` (L112, L148 are correct). README L91 `python scripts/download_weights` omits `.py`; script loads `0_configs/default.yaml` relative to cwd (L32) so must be run from repo root — undocumented.
  - External data: README L73 points to GitHub → OneDrive; L183/L242 point to Zenodo 15377683; number of synthetic samples used not stated. Without the HF bundle, full Block 1 is ~3.5 days (README L38).
  - Known bugs intentionally kept (README L212-219): 2D `RandomHorizontalFlip` flips top-down (`train_2d_resnet.py:58`, `train_2d_vit.py`) and synthetic volumes upside-down at soft-labelling; documented as kept for reproduction. Fine for verification; release should keep this note.
  - README reproduction table (L226-232): OOF log loss 0.2000 (submission) vs 0.2010 / 0.2014 (12-/3-model reproductions), std 0.0119 — run-to-run variance acknowledged; static review cannot verify these numbers.
- `MedicalNet` pretrained loading is only wired via `--pretrain_path`, never passed by `5_pipeline/run_pipeline.py` (solution_overview §4), so the README's MedicalNet acknowledgement vs actual pretrained use is ambiguous for a re-runner.

## RR-7 Documentation gaps — temperature / clip / leaderboard factor

- Inference applies `TEMP = 1.094` (tree `main.py:36`, harness `main.py:43`) and clip [0.01, 0.99] (`SZ/inference_feats.py:414,453-456`). README L205-207 says only "averaging the predictions from all 100 models"; no mention of T or clip.
- Harness `validate_pipeline.py:121-125` "(facteur 1.245 = correction de décalage mesurée sur le leaderboard)" and L698-700 "la T soumise contient le facteur 1.245 de décalage leaderboard ... DÉLIBÉRÉMENT sous-optimale sur l'OOF ... au notebook". The notebook that derived T/clip is not delivered; the factor is not in the tree's docs.
- **Required for documentation completeness:** team to document T, clip and the leaderboard-derived 1.245 factor in README / Winning Model Documentation and deliver or describe the derivation. (Whether LB-tuned scalar is acceptable is for Code Integrity / organizers.)

## RR-8 Documentation gaps — Winning Model Documentation and provenance

- No Winning Model Documentation (`driven_data.md` or equivalent) in `1st Place/` (grep); README is the only doc. Organizer to confirm whether it was delivered separately (rules L68).
- Provenance gaps (detail owned by Data Source): license of Zenodo 15377683 / OneDrive synthetic PET and its generator's training data; MedicalNet weights actually used or not; `edadaltocg` ResNet50-CIFAR100 and ViT in21k→CIFAR100 checkpoints' licenses and pretraining data (pending cross-winner ImageNet ruling). README L245 "Block 2 - no additional used resources" is true for weights but Block 2 trains on Block-1 (pretrained-teacher) soft labels.
- `1st Place/pyproject.toml:4` placeholder description; code comments/log strings are French — fine, but an English README exists.

## RR-9 Packaging / size (non-blocking)

- Harness zip 1.007 GB: `weights.zip` 788 MB (100 × ~7.9 MB `.pth`), `src.zip` 217 MB of which 216.3 MB is `src/libs/` and 190.7 MB is `libs/xgboost/lib/libxgboost.so` alone. XGBoost / openTSNE / pykwalify / ruamel are not on the scored path (only imported); `XGBoostEnsembleInference` / t-SNE inactive.
- Stale `__pycache__/main.cpython-312.pyc`, `constants.cpython-312.pyc` in harness (tree has none).
- 8 diagnostic scripts + 2 OOF CSVs packed because `just pack-submission` zips all of `submission_src/*` (solution_overview §3 step 7).
- Large amount of dead code in `src/` (TwoPhase, DWT, MiniCNN3D, RNN/corax/sagcorax, transformer-fusion Siamese, many `prefix_map` entries). Recommend pruning to the executed path for the release.

## RR-10 Images (not viewed)

- `1st Place/img/pipeline.png` 56,377 B sha256 `2483c5b6f006265812c6d3ce8bde721a09fad7131f4d83b5534ff5166d304669`.
- `1st Place/img/data_processing.png` 146,608 B sha256 `e17b32f8f887e5288b6bed4d9ca773f0141dc26a815f2d758c43274a5c4006e9`. README L176-179 says it illustrates the striatum views (global/flip, per side, close-up) → likely rendered crops of a real competition scan (or a synthetic one). **Not viewed; organizer to inspect** before public release; if it shows a competition scan, it must be replaced (competition data not yet public; rule E).

## Process confirmation

No competitor code was executed or imported; no artifacts loaded; `weights.zip` / `src.zip` .so files not extracted; HF repo, OneDrive, Zenodo not fetched; no images opened or rendered (only file size + sha256). Read-only access to `1st Place/` and `references/`; the only file written is this report. Upstream license identifications marked "to confirm" are from general knowledge, not fetched.
