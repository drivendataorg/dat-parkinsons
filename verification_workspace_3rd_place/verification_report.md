# Verification report — DaT Parkinson's Challenge (DrivenData 311), 3rd place

**Recommendation: `needs organizer review`.** No disqualifier is confirmed. The competitor's hand annotations of training scans are permitted by the official rules, and the competitor gave them to the organizers. The code-integrity review is clean. Two issues need an organizer ruling: undisclosed use of the non-open PPMI dataset for evaluation and model selection (F2), and AI-agent use with possible data retention (F3).

Date: 2026-09-29 · Coordinator: Claude (submission-verification skill) · Static review only.

## 1. Submission

| | |
|---|---|
| Team / username | paulonium (solo; LICENSE © Paul Blanc-Durand) |
| Source tree | `3rd Place/` (per-file hashes in `hashes/source_tree.sha256`) |
| Harness package (from platform) | `references/submissions/3rd_place/3rd_place_submission.zip`, sha256 `43789dc1…178395` |
| Scored predictions | `references/submissions/3rd_place/3rd_place_submission.csv`, sha256 `104184ba…9022a` |
| Official private score (intake) | 0.273213 |
| Official public score (intake) | 0.231321 |

**Score cross-check.** This is a static observation, not an official score. The Coordinator recomputed log loss from the provided CSV and the organizer's test labels. No competitor code was run.
- **Full test set (524 scans):** 0.273213, which matches the official private score exactly.
- **Public subset (262 scans):** 0.231707, which does not match the intake's 0.231321. See open question Q6.

## 2. Materials reviewed

**Required, all available:**
- The five public competition pages: homepage, rules, problem description, code submission format and about. They are saved as text in `official_pages/`, fetched 2026-09-29, with hashes in `hashes/official_pages.sha256`.
- `references/verification_comp_specific_rules.md`, filled in by the organizer.
- `references/competition_materials.md`.
- The source tree, with its README, `driven_data.md`, LICENSE and `requirements.txt`.

**Optional, available:**
- The harness zip, including all 50 member weights and the localiser.
- The scored CSV.
- The leaderboard scores (intake).
- The organizer's copy of the training data. Test UIDs were used from `submission_format.csv`. Test labels were used only by the Coordinator for the score cross-check, and no reviewer opened them.

**Optional, unavailable, and what that limited:**
- **`ellipse_annotations.tar.gz`** was not in the review materials. The organizer confirms the competitor delivered it, but it wasn't inspected here. So "no test scan was annotated" rests on the code paths, not on the actual list of 276 UIDs. See Q3.
- **The competitor's agent files** (`CLAUDE.md`, `notes/agents/LAB.md`, `dd_status.py`, which are referenced but absent). Without them, data exposure to AI tools could not be assessed. See Q4.
- **The scoring script and the platform submission ID.** Without them, the public-score mismatch cannot be resolved. See Q6.

## 3. Governing rules applied

The quotes in the rules file were re-checked against the live page text and match in substance (`notes/governing_rules.md`). These rules and organizer rulings decided the outcome:
- Competitor annotations of training data: **permitted** if they are delivered with the solution and do not overfit to the test set. Manual annotation of test data is **not permitted** (official rules, *Data use and code sharing*, rules.txt:137-138).
- Data behind a data use agreement (DUA): allowed only if the DUA says the data is open.
- No datasets were approved during the competition, and there were no organizer clarifications.
- Accounts and identity or eligibility: checked and confirmed by DrivenData.
- Release: code only at the winner announcement. Weights only in a private location with organizer-only access.

The live pages contain these rules. A–E are not yet in the rules file; F is. All are recorded in `notes/governing_rules.md`:
- **A.** No tools like Codex or ChatGPT that store or retain uploaded data.
- **B.** External data must be under a permissive open license, e.g. CC0 or CC-BY.
- **C.** The submission must not print or log information about the test set.
- **D.** Each test sample must be processed independently.
- **E.** Data may be used "solely for the purpose and duration of the Competition".
- **F.** Training data may be annotated if the annotations are included with the solution and do not overfit to the test set. Test data may not be annotated (rules.txt:137-138).

## 4. Scope

- **No competitor code was executed** by the Coordinator or by any of the five reviewers. Each reviewer confirmed this in its report.
- Nothing was installed or imported, and no `.pt` file was loaded.
- TorchScript graphs were read as text from inside the zip.
- `3rd Place/` and `references/` were not modified. All outputs are in this workspace.
- **Not attempted (dynamic, out of scope):** environment setup, running inference and checking the score is within range, and running the training code.
- **Process note on confidentiality:**
  - The Release Readiness reviewer viewed the five `docs/figures/*.png` images, which are renders of seven competition training scans, so those images went through the model API used for this review.
  - The Data Source reviewer refused to open them for that reason.
  - Image viewing is not allowed in future reviews. See "Verification handling restrictions" in `references/verification_comp_specific_rules.md`.

Solution overview: `reviewers/solution_overview.md`. Reviewer reports: `reviewers/02_eligibility.md`, `03_code_integrity.md`, `04_data_provenance.md`, `05_release_readiness.md`.

## 5. Solution overview (short)

- **Preprocessing.** Each NIfTI is resampled to 2 mm isotropic, cropped to a head box, and has its glands removed.
- **Localiser.** A small localiser network, **`ellipse.ts.pt`**, fits a per-side striatal ellipsoid. That ellipsoid drives:
  - canonical pose correction;
  - the anchor mask;
  - displacement ("NDT") slabs.
- **Projection.** The scan is projected with fixed physical operators to 4 axial channels plus sagittal views.
- **Ensemble.** A MONAI DenseNet121, trained from scratch, reads the projections. There are 10 members × 5 folds = 50 TorchScript models.
- **Scoring.** Flip-TTA; member logits capped at ±6 and averaged; temperature calibration with a = 0.78 and b = −0.16455.

## 6. Findings

### Blocking

None confirmed.

### Informational

**F1 — info — The competitor's hand annotations of training scans are load-bearing, and permitted.** (Reviewers 1, 2 and 4 found this; the Coordinator verified it.)
- **What the competitor did.** The competitor drew striatal contours by hand on 276 competition training scans in three sessions (August to 2026-08-22). They used a localhost browser tool.
- **How it reaches the scored model.** The contours trained the shipped `ellipse.ts.pt`, which runs on every test scan. The localiser also generated the training inputs and lesion masks for all 50 members.
- **Evidence:**
  - `build/provenance_ellipse/README.md:24-30,53`
  - `build/train_ellipse.py:66`
  - `build/provenance_ellipse/segment_striata_server.py:1-15`
  - `docs/repro/logs/train_ellipse.log:2`
  - `build/build_canonv2_cache.py:85,96`
  - `harness/main.py:104-131`
  - `ellipse.ts.pt` sha256 `963311…3b44`, identical in the harness zip and the tree
- **Rule applied:** rules.txt:137-138, checked condition by condition:
  1. **Training scans only.** The tool reads the training scan cache. `build_box_masks.py` and `train_ellipse.py` index only the training labels. `meta/random50_uids.csv` is 50 training UIDs out of 50. No test UID appears anywhere in the tree or harness. Test scans are processed automatically at inference.
  2. **Delivered with the solution.** The organizer confirmed that `ellipse_annotations.tar.gz` was provided.
  3. **No overfitting to the test set.** No sign that the annotated scans were picked or tuned using the test set. They were 105 training "model errors" plus 105 matched controls and 66 random scans. The competitor had no test labels, so those errors can only be errors on training scans. The localiser holdout is 25 annotated training scans.
- **Residual:** confirm condition 1 against the actual UID list in the tarball (Q3). The annotations are not disclosed in `driven_data.md`; the README discloses them.

### Needs organizer ruling (not blocking on their own)

**F2 — high — Undisclosed use of PPMI, a non-open dataset, for evaluation and model selection.** (Reviewers 1, 2 and 4; the Coordinator verified it.)
- **What the competitor did.** They held 3,839 PPMI scans locally and used them for out-of-distribution AUC checks, for ranking members, and as a "veto". PPMI shaped the shipped `anchor_guard` flag.
- **What was not found.** No training on PPMI, and no PPMI-derived constants in the shipped assets.
- **Disclosure.** It is not disclosed. `driven_data.md` §9 answers "No" to undisclosed tools and data. README:185 and 568 show the competitor believed the rules banned external DaT data.
- **PPMI terms** (public DUA v5.0, per Reviewer 4):
  - access only by application and approval;
  - purpose-limited use;
  - no redistribution;
  - models trained on the data can be shared only through methods MJFF approves;
  - no open license.
- **How it measures against the rules.** It fails the organizer's "open DUA" test and rule B.
- **Evidence:** `build/build_box_mmap.py:34-37`; `notes/REFUTED.md:25-29,517-547`; `datscan/projection.py:366-391`.
- **Questions for the organizer:** does evaluation-only use count as use, and is the missing disclosure a violation in itself?

**F3 — medium — AI coding agents may have retained competition data (rule A).**
- **What the competitor said.** `driven_data.md:111-113` says Claude Code was used "throughout", including for "model evaluation".
- **What the notes show.** `notes/REFUTED.md:715-900` describes overnight agents analysing per-scan out-of-fold results, labels and severity. The agents also made authenticated requests to DrivenData pages (lines 1004-1012).
- **Why it can't be settled here.** The agent configuration files are absent, and retention depends on the account type. A static review cannot tell whether scan data or labels reached the service.

**F4 — medium — The competitor controls competition-derived data on a third-party host, and kept the data after the close (rule E).** Delivering the annotations to the organizers meets rule F. This finding is about who holds the data afterwards.
- **What's there.** The private Hugging Face repo `paulonium/dat-challenge-winner` holds the prize tarball, which has labels and scan renders according to the README, and `ellipse_annotations.tar.gz`, which has masks for all 1,362 training scans.
- **Why it matters.** The rule permitting private storage covers weights, not data. Access is "request access", so the competitor controls it.
- **Retention.** A full retrain on 2026-09-24, after the 2026-09-16 close, shows the data was still held locally.
- **Evidence:** README:51-63; `build/provenance_ellipse/README.md:42-49`; `docs/repro/md5_weights_training_new_dat.txt`.

### Non-blocking

- **F5 — medium — The shipped calibration intercept doesn't match the docs.** The shipped b is −0.16455, but the documented and "reproducible" value is −0.1478 (`harness/assets/calibration.json`; `recipes/fusion10_s078.json:73`; README:139).
  - Both values were fitted on training out-of-fold predictions (`build/build_roster.py:45-58`), most likely from different runs. It is not test leakage.
  - The capped values in the scored CSV match −0.16455 to within float32 rounding, and not −0.1478.
- **F6 — low — The README's "exact program submitted" claim is inaccurate.** The tree's `inference/main.py` has an extra `ex_*` code path dated 2026-09-11 that doesn't fire (`notes/main_py_harness_vs_tree.diff`), so it isn't the submitted file (README:67). The other harness files are byte-identical to the tree: `canonize.py`, `datprep_box.py`, `datprep_iso.py`, `glandrm.py`, `canon_const.json` and `ellipse.ts.pt`.
- **F7 — low — Latent test-logging issue (rule C).** `harness/main.py:32-39` prints a test file name and the error text if a scan fails preprocessing. This did not fire in the scored run: the CSV has 0 fallback 0.5 values.
- **F8 — info — Other competitor annotations that don't ship.** Blinded label re-reads (60 + 14 scans) and an expert-concordance row-weight option exist but are not used by the scored model (README:523-531; `datscan/config.py:570-571`). They are permitted under rule F; the re-reads were on training scans.
- **F9 — info — Leaderboard-guided choices.** The slope a = 0.78 was picked using public-leaderboard feedback (`build/calibrate.py:9-29`). The rules don't prohibit this, so it is not a violation.

### Code integrity: clean apart from the above (Reviewer 3)

- **No test UIDs** anywhere, in any scope, including decompressed TorchScript entries, `.pyc`, `.npy` and `.png` files. The positive controls worked.
- **No leaks of test information** into the model: no hard-coded predictions, labels or UIDs, no pseudo-labelling, no batch-norm adaptation, and no statistics pooled across test scans.
- **BatchNorm and dropout are off** at inference in the traced graph: all 371 batch-norm and 174 dropout calls run in eval mode. Every preprocessing step is per scan.
- **No network calls** in the harness.
- **The scored CSV fits the harness code.**
  - It follows the `submission_format` order with no duplicate UIDs.
  - It has no exact 0 or 1 values.
  - The 51 repeated values sit exactly at the calibrated cap values.

### Licensing and eligibility (Reviewer 2)

- **License:** `LICENSE` is MIT.
- **Dependencies in the shipped path** are all permissive:
  - torch, numpy, scipy, pandas and scikit-learn: BSD-3;
  - MONAI: Apache-2.0;
  - nibabel: MIT;
  - matplotlib: PSF-style.
- **No NC or non-OSI component** is shipped.
- **Pretrained weights:** none in the shipped model. The ImageNet timm backbones were tried and rejected.
- **Identity and accounts:** confirmed by DrivenData per the rules file.

### Release readiness (Reviewer 5)

Before a public code-only release:
- **Remove or make private:**
  - `inference/assets/ellipse.ts.pt`;
  - `docs/figures/*.png`, which show 7 training scans with their UIDs, labels and model probabilities;
  - `meta/labels.npy`;
  - the label columns in `meta/*.csv` and `splits.csv`.
- **Delete** the `__pycache__` folders, which leak the username and home path.
- **Parameterise** the hard-coded `/ssd/...` paths.
- **Pin** `joblib`, `timm`, `e2cnn` and `escnn`.
- **Commit** the shipped `calibration.json`.
- **Fix the README:** the 3rd vs 6th place inconsistency (README:291, 367) and the main.py claim (F6).
- **Amend `driven_data.md`** to disclose PPMI, the hand annotations and the AI-agent use.

## 7. Checklist coverage

| Item | Result |
|---|---|
| Obvious disqualifiers | None confirmed. F1 is permitted under rule F. |
| Non-OSI models or code | None found in the shipped path |
| Clear cheating | None: no test leakage or hard-coding |
| No test data in training (pseudo- or hand-labelling) | Pass. Only training scans were annotated, which is permitted (F1). |
| Test observations processed independently | Pass (Reviewer 3). Latent logging issue F7. |
| No prohibited data sources | **F2** (PPMI, needs a ruling) |
| No multi-account submissions | Checked by DrivenData, no issues (rules file) |
| Code matches leaderboard submission | Harness zip is consistent with the scored CSV. Tree differs (F5, F6). |
| Competition-specific restrictions | C1–C14 in `solution_overview.md`, all applied |
| External model trained on a prohibited source | None: no pretrained weights shipped |
| External data or model blocks broad release | PPMI influence (F2), not in the weights |
| Competition data sent to a third-party API | Undetermined (F3) |
| Tweaks needed for release | Release readiness list above |
| Dynamic items (environment, inference score, training run) | **Not attempted**: out of scope |

## 8. Open questions for organizers

1. **F2 decides the outcome:** if evaluation-only use of PPMI, or not disclosing it, is a violation, the recommendation becomes `disqualify`. Otherwise it is `pass with notes`.
2. **F2:** Does evaluation-only and model-selection use of PPMI count as using external data? Is the missing disclosure a violation in itself?
3. Check the UID list in the delivered `ellipse_annotations.tar.gz` (`striatal_masks/`, `box_automasks/` expert flags) against `submission_format.csv` to confirm that none of the 276 annotated UIDs is a test scan.
4. **F3:** Ask for their Claude account type and data-retention settings, and for the missing agent files (`CLAUDE.md`, `notes/agents/LAB.md`, `dd_status.py`). Did agents read scan arrays, per-UID labels or renders?
5. **F4:** Move the Hugging Face contents into DrivenData custody and have the competitor delete them there. Set a deletion deadline for data kept locally after the close.
6. **Public score:** the intake's 0.231321 doesn't match the 0.231707 recomputed from this CSV. Reviewer 1 links 0.231321 to a sibling submission with a = 0.85. Confirm which submission ID was used for each leaderboard, and whether the private leaderboard uses the full test set, since 0.273213 matches the full-set log loss.
7. Employer IP: the competitor used a `paire_training` environment and co-founded PAIRE. Confirm there is no employer claim that conflicts with the MIT license and the "all rights" requirement.
8. Rule C: do the organizers treat the failure-only file name print (F7) as a concern?
