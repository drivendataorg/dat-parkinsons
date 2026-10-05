# Verification report — 1st place, DaT Parkinson's Challenge (DrivenData comp 311)

**Recommendation: `needs organizer review`.** No evidence of cheating, test-label leakage, test-set annotation or non-independent test processing. The scored CSV is consistent with the submitted harness code, and recomputed log loss matches the official score. Four items need an organizer decision or follow-up:

1. Licence and provenance ruling on the synthetic DaT PET data. It is the only external data that enters the final model directly.
2. The pending cross-winner ruling on ImageNet and CIFAR pretraining, which here applies only to the Block-1 teacher models.
3. Organizer check that the competitor's Hugging Face weights repo is private.
4. A note on the leaderboard-tuned temperature. It is not rank-determinative: §5 shows 1st place keeps the lead without it.

Release also needs the solution relicensed under MIT.

If (1) and (2) are ruled permissive and (3) is remediated, the recommendation becomes `pass with notes`.

Coordinator: Claude (submission-verification skill), 2026-10-02. Static review only.

## 1. Submission

| Item | Value |
|---|---|
| Team / usernames | TheAvengers — `dvtle`, `SamuelAnsel` (`1st Place/README.md:2-3`) |
| Official private leaderboard score (intake) | 0.253233 |
| Coordinator recomputation from scored CSV (static-review observation) | full test 0.253233 (n=524); public subset 0.224948 (n=262); private-only subset 0.281519 (n=262) |
| Source tree | `1st Place/` (90 files; hashes `hashes/source_tree.sha256`) |
| Harness zip | `references/submissions/comp_subs/1st_place/1st_place_submission.zip` sha256 `fd97854777e209067e8d499bed15c91b7b3b7951debd6ba64fa4f77e95eb2de8` (19 members; nested `src.zip` 217 MB, `weights.zip` 788 MB with 100 `.pth`) |
| Scored CSV | `references/submissions/comp_subs/1st_place/1st_place_submission.csv` sha256 `8287800024040c56dd6c63db4ff1fa0f6c88328f307fbbc15b70d4a2d0bb7ece` |

**Intake correction:** In `references/submissions/1st_place.md`, the "code-execution package" line points at the `.csv`. The harness zip above was used instead. Hashes for both are in `hashes/submitted_artifacts.sha256`.

## 2. Materials reviewed

**Required (all available):**
- The competition homepage, rules, problem/data page and code-format page. Their text is in `official_pages/`; it was fetched 2026-09-29 and reused from the 2nd/3rd-place reviews, with hashes in `hashes/official_pages.sha256`.
- `references/verification_comp_specific_rules.md` (organizer-filled) and `references/competition_materials.md`.
- The submitted source tree and README.
- The runtime `pyproject.toml` and `uv.lock` files.

**Optional, available:**
- The harness zip from the platform. Text members were extracted to `extracted/`; `weights.zip` was not extracted.
- The scored CSV.
- `submission_format.csv` (uid order).
- `test_labels.csv` and `pub_sub_idx.csv`. Only the Coordinator used these, for score recomputation and the §5 counterfactual; reviewers did not open labels.
- `train_labels.csv` (uids only).

**Optional, unavailable, and what it limited:**
- **Winning Model Documentation template:** not in the tree. The disclosure cross-check was done against the README only.
- **The team's notebook that derived T and the clip (the "RECETTE FIGÉE" notebook):** not delivered. This limited the review of leaderboard-informed tuning (§5).
- **Platform run log and submission history:** unavailable. This limited how precisely we can count the public-LB submissions behind the 1.245 factor.
- **Model weights:** present in the harness but not loaded (static review). The HF weights repo was not fetched.
- **Organizer clarifications:** none, per the rules file.

## 3. Governing rules applied

The rules come from `verification_comp_specific_rules.md` plus the verbatim rules A–F in `notes/governing_rules.md`:

- **Forbidden data and models:**
  - External data that is not freely and publicly available under a permissive licence that allows commercial use.
  - NC, CC-NC or CC-BY-NC data or models.
  - Pretraining data counts under the same rules.
  - DUA-gated data is allowed only if the DUA says the data is open.
  - No public copy of the competition data exists.
- **Allowed data:**
  - Competition training niftis and `train_labels.csv`.
  - Permissive external data and models, all disclosed to organizers.
  - No organizer-approved extra datasets.
- **Added checks:**
  - Independent per-case test processing, with no pseudo-labelling or unsupervised learning on test data (D).
  - No logging of test information (C).
  - Harness layout and offline execution.
  - Predictions in [0, 1].
  - No AI tools that retain uploaded data (A).
  - Annotations on training data only, delivered with the solution, and not overfit to the test set (F).
  - Privacy.
  - Data used only for the competition (E).
  - MIT licence and OSS dependencies.
  - Release is code only at announcement; weights trained on competition data are held privately with organizer-only access.
- **Confirmed by DrivenData (not re-checked):** winner identity and eligibility, the submission cap and multi-account activity.

Reviewer 1 turned these into owned checks CS1–CS20 (`solution_overview.md` §1).

## 4. Scope

- **Static, read-only review.** No competitor code was executed by the Coordinator or any reviewer, no competitor module was imported, and no artifact was loaded.
- **Harness archives:** `weights.zip` was neither extracted nor loaded. Only text members of `src.zip` were extracted.
- **No images were viewed** (organizer restriction), including `1st Place/img/*.png`.
- **Web fetches:** reviewers fetched only public third-party pages: the Zenodo record and API, GitHub repos and API, HF model cards for edadaltocg and timm, PyPI, and Europe PMC. URL lists are in `reviewers/02_*.md` and `04_*.md`. No competitor-hosted resource (`dinhviettoanle/*`, OneDrive) was fetched. No competition or submission content was sent anywhere.
- **Nothing modified:** no file under `1st Place/` or `references/` was changed.

**Dynamic checks were not attempted** and remain for a human: environment setup, an inference rerun with score comparison, and a training rerun.

The solution overview is in `solution_overview.md`. Reviewer handoffs are in `reviewers/01_…05_*.md`.

## 5. Code integrity findings

- **Scored path (confirmed by Coordinator).**
  - Harness `main.py:31,37-43` sets `MODE_INFERENCE="FAMILY_STACKING"`, `FAMILY_MODELS=["Mashup_Internal"]` and `TEMP=1.094`.
  - It loads the 100 weights `mashup_ensemble_exp_{0..19}_fold_{0..4}.pth` (`src/inference_2d_avg.py:861-957`) and plain-averages them.
  - It then outputs `clip(sigmoid(logit(p̄)/1.094), 0.01, 0.99)` (`src/inference_feats.py:426-458`).
  - There is no TTA. The model is a randomly initialised Siamese CNN with about 1.96M parameters. It uses no pretrained weights.
- **Code matches the leaderboard submission (pass with notes).**
  - The scored CSV has the form this code produces: 524 rows, uid order identical to `submission_format`, 0 duplicates, 109 rows at exactly 0.99, minimum 0.01256 (lower clip never hit), 416 distinct values, and no 0.5 fallback rows.
  - The weight names and count match the `train_siamese.py:306` output.
  - The harness and tree differ, but only outside the executed path. `main.py` differs in `DATA_DIR` and comments. `src/builder.py`, `inference_feats.py` and `inference_2d_avg.py` differ only in classes the scored run never instantiates (`notes/diff_*.txt`).
  - The scored zip was built from a different working copy than the delivered tree. Its `src.zip` vendors `src/libs/`, and 8 helper scripts in it are not in the tree.
- **No test data in training (confirmed).**
  - 0 of 524 test uids appear anywhere in the tree or the extracted harness, in text or binary.
  - Soft labels are generated only for the external synthetic PET directory (`3_soft_labeling/soft_labeling.py:24,217`). Teachers and the meta-model are trained on the 1,362 training scans only.
  - Synthetic rows always sit in the training split. Early stopping uses real validation folds.
  - `"siamois_final_pseudo"` is an inactive legacy model name with no weights and no training code. There is no pseudo-labelling of test data.
- **Independent per-case processing (pass).**
  - Inference runs one case at a time with batch size 1, BatchNorm in eval mode, and per-volume preprocessing statistics. Nothing is fitted at test time and nothing is cached across cases.
  - t-SNE (vendored openTSNE) is not on the scored path and would only `transform`.
- **Test logging (pass, low note).** A per-case warning prints the names of NaN/Inf radiomics features, with no uid and no values (`src/preprocessing.py:579-585`).
- **Leaderboard-informed tuning of the temperature (medium, note; Coordinator verified).**
  - The harness and tree `validate_pipeline.py:121-125,698-700` say the submitted T includes a "facteur 1.245 = correction de décalage mesurée sur le leaderboard", making it "DÉLIBÉRÉMENT sous-optimale sur l'OOF".
  - The pre-leaderboard settings are in `validate_inference.py:53-60`: T=0.8835, clip 0.002, OOF log loss 0.1983. The submitted values are 0.8835 × 1.245 × 0.99515 ≈ 1.094, with the clip widened to 0.01.
  - These are global scalars with no per-case, uid or label information, so this is not label probing and does not breach the independence rule. It is relevant to rule F's "do not overfit to the test set" and is undocumented in the README.
  - **Counterfactual:** the Coordinator recomputed the score with test labels (`notes/temperature_counterfactual.txt`). The calculation is exact because sharper temperatures keep the 0.99-clipped rows clipped.

    | T / clip | full test | public | private-only |
    |---|---|---|---|
    | 1.094 / [0.01, 0.99] (submitted) | 0.253233 | 0.224948 | 0.281519 |
    | 0.8835 / [0.01, 0.99] (pre-leaderboard T) | 0.262992 | 0.231646 | 0.294337 |
    | 0.8835 / [0.002, 0.99] (full pre-leaderboard recipe, upper clip kept) | 0.262839 | 0.231481 | 0.294197 |
    | 2nd place, as submitted | 0.273088 | 0.240858 | 0.305317 |

    Without the leaderboard factor, 1st place remains ahead of 2nd place on the full test set and on the private-only half that gave no feedback. **The ranking does not depend on this tuning.** The 2nd-place solution also tuned T and the clip on the board (2nd-place report §5), so organizers should treat both the same way.
- **INFO:** The Siamese OOF score (about 0.20) is optimistic, because the soft labels come from teachers that saw all real folds. This is not a rules issue.

## 6. Data source, provenance and leakage findings

### Synthetic DaT PET data (needs organizer ruling, high)

This is the only external data used. It is the input for all 100 final models (README L22-24, L71-82; `4_siamese_training/*`). The Coordinator verified the licence facts:

- **The download the team used:** the README (L73) points to the "One Drive/Synthetic FP-CIT-PET Samples" SharePoint folder at `liveuou-my.sharepoint.com` (University of Ulsan). It states no licence terms.
- **The GitHub repo `Krying/PD_SSL_ZOO`:** no licence (`license: null`).
- **Zenodo 15377683:** CC-BY-4.0, but its resource type is "Software": a single 27.6 MB GitHub-snapshot zip. Reviewer 4 reports it contains only 3 synthetic volumes. The README cites Zenodo as if it were the dataset.
- **The paper (Won et al., Cell Rep Med 2025, PMC12281361):** the article is CC BY-NC, which covers the article only. It says the synthetic images are available on GitHub and Zenodo.
- **The generator's training data:** 1,934 real FP-CIT PET scans from Asan Medical Center, Seoul (IRB 2022-0614). This data is private and available on request with IRB approval. It is not PPMI, and overlap with the French SPECT competition data is implausible.

**Questions for organizers:**
- (a) Is a set with no stated licence, hosted alongside a CC-BY Zenodo snapshot, "freely and publicly available … under a permissive open license"?
- (b) Is synthetic output from a generator trained on private clinical data allowed, given that "pretraining data … is subject to the same rules"?

Either ruling could be resolved by the dataset authors confirming a licence. The README notes that a `--no-synth` path exists, but it states this does not reproduce the best submission.

### Pretrained models (pending cross-winner ruling, medium)

Pretrained models are used only in Block-1 teachers, and reach the final model only through the synthetic soft labels:
- `edadaltocg/vit_base_patch16_224_in21k_ft_cifar100` (MIT weights). Its lineage is ImageNet-21k → ImageNet-1k → CIFAR-100 (`1_initial_training/train_2d_vit.py:104-113`). ImageNet's terms are non-commercial. The README omits the ImageNet-1k step.
- `edadaltocg/resnet50_cifar100` (MIT weights), trained on CIFAR-100, which has no formal licence (`train_2d_resnet.py:128-136`).
- **MedicalNet:** the code never sets `--pretrain_path`, so these teachers start from random initialisation. Tencent's code is MIT. Ask the team whether the original run used Tencent's pretrained weights.

The organizer's ImageNet ruling for 2nd place should cover this case too. It should also cover CIFAR-100 and this indirect, teacher-to-soft-label use. Under the README's "3-model" alternative (ResNet2D, MedicalNet50 and SVM), the ViT teacher drops out.

### Other data-source checks

- **Dependencies:** all are OSS and allow commercial use. ptwt is EUPL-1.2 (OSI copyleft). libgomp is GPL-3.0 with the runtime exception. CUDA is proprietary, but CPU paths exist. Vendored code ships without licence texts (Reviewer 2).
- **Network / third-party APIs (pass):**
  - Inference is offline.
  - Downloads happen only at training time (public HF weights) and in the optional `scripts/download_weights.py`.
  - No upload code, trackers, cloud SDKs or LLM clients.
- **AI tools (rule A):** no evidence either way. Open question for the team.
- **Annotations (rule F):** none. `USE_RELABEL` is off.
- **Privacy (pass):** the only NIfTI header field read is the voxel spacing (`get_zooms()`).

## 7. Release readiness

Release expectations were provided. Full detail is in `reviewers/05_release_readiness.md`.

### Blocking

- **(high) Licence.** README L247-250 says the materials are confidential and must not be shared. There is no solution LICENSE file; `competition-sfmn-parkinsons-runtime/LICENSE` is DrivenData's runtime licence. The team must add an MIT LICENSE and replace that README section.
- **(organizer action) Weights host.** `scripts/download_weights.py:53-56` calls `snapshot_download("dinhviettoanle/dat-parkinson-models")` with no token, which suggests the repo may be public. It was not checked.
  - According to the script, the repo holds teacher and Siamese weights trained on competition data, `SVM.pkl`, `tabular_oof.csv` and per-teacher OOF CSVs (training labels), and `synthetic_soft_labels.csv`.
  - Organizers should confirm the repo is private with organizer-only access, and have the label CSVs removed.
  - The script and README L85-96 should be dropped from the public code release.
- **Training labels in the harness (rule E).** `tta_siamese_predictions.csv` covers all 1,362 training uids with labels, and `validation_report.csv` covers 100. Neither contains test data, but neither may be published. Release should come from the tree, which has no data files, not from the harness zip.
- **Attribution.** Add the licence texts and notices for MedicalNet, 3D-ResNets and MedNeXt (Apache-2.0), and for any vendored libraries that are kept.

### Non-blocking

- **Reproducibility:**
  - The tree's `main.py:20` sets `DATA_DIR=Path()`.
  - The tree's `src/` lacks the `libs/` that the scored modules import at load time (ptwt, xgboost, radiomics).
  - 8 helper scripts are missing from the tree.
  - The data filename is hard-coded as `train_labels_JNDlMjr.csv`.
  - README commands have typos.
  - The README documents two flip bugs that were kept on purpose to reproduce the best submission.
- **Undocumented post-processing:** T=1.094, the leaderboard factor and the clip.
- **Private paths:** team-internal paths (`/data2/iceberg`, `/data/HOME/dvtle`, `/data/HOME/sansel`) appear in the harness scripts and in tree `test_inference.py:28`.
- **Size:** about 1 GB, including a 190 MB `libxgboost.so`, a stale `__pycache__` and dead code.
- **Images:** `img/data_processing.png` may show real scan crops. Not viewed; organizer to inspect before public release.
- **Secrets:** none found.

## 8. Blocking issues, non-blocking issues, open questions

**Blocking (organizer decision or remediation required):**
1. A ruling on the synthetic PET data: its licence status and the private data its generator was trained on (§6).
2. Remove or make private the HF repo `dinhviettoanle/dat-parkinson-models`, and remove the label-bearing CSVs from it (§7).
3. Relicense the solution under MIT and add attribution (§7).

**Policy, applies across winners:**

4. The ImageNet/CIFAR pretraining ruling. Here it affects only the teacher models that produce the soft labels.
5. Consistent treatment of leaderboard-tuned temperature and clip. It is not rank-determinative for 1st place (§5).

**Non-blocking:**
- Rebuilding from the tree needs fixes: `DATA_DIR`, the vendored libs and the missing helper scripts.
- Training-label CSVs and private paths in the harness.
- Optimistic OOF claims.
- Image check before public release.

**Open questions for the team:**
- Did they use Tencent's MedicalNet pretrained weights?
- Was the ResNet-50 CIFAR checkpoint trained from scratch?
- How was 1.245 derived, and over how many submissions? DrivenData can also check the submission history.
- Can they deliver the notebook that derived T and the clip?
- Which AI coding tools did they use, if any, and did any competition data reach a service that retains it?
- Are there public copies of the synthetic soft labels or weights elsewhere?

## 9. Checklist coverage

| Checklist item | Covered by | Result |
|---|---|---|
| Obvious disqualifiers, non-OSI models/code, clear cheating | R2, R3 | None found; licence remediation needed |
| No test data in training (pseudo- or hand-labelling) | R3, Coordinator uid grep | Pass |
| Independent test processing | R3 | Pass |
| No prohibited data sources | R4, Coordinator | Needs organizer ruling (synthetic PET) |
| Submission limit / multiple accounts | DrivenData (rules file) | Checked by DrivenData, no issues |
| Code matches leaderboard submission | R1, R3, Coordinator CSV recompute | Pass with notes (built from a different working copy; executed path identical) |
| Competition-specific restrictions | R1 CS1–CS20, all reviewers | See §5–§7 |
| External models trained on a prohibited source | R2, R4 | Pending cross-winner ImageNet/CIFAR ruling |
| External data/model blocking broad release | R2, R4, R5 | Synthetic PET licence unclear |
| Competition data uploaded to third-party APIs | R4 | None found; rule A open question |
| Release tweaks to code/docs | R5 | §7 |
| Dynamic: environment setup, inference rerun, training rerun | — | **Not attempted (out of scope)** |

## Final recommendation

**`needs organizer review`**
