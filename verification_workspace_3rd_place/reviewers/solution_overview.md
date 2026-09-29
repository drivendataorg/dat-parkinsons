# Solution Overview — 3rd place (paulonium / Paul Blanc-Durand)

> **Note:** Findings in this file that treat the competitor's annotations of training scans as prohibited do not apply. The official rules permit annotating training data when the annotations are provided with the solution (`official_pages/rules.txt:137-138`). See F1 in `verification_report.md` for the final assessment.

Reviewer 1 (Solution Explanation). Static, read-only review. **No competitor code was executed**; no competitor module was imported; no `.pt` artifact was loaded. Data files were read only with numpy/pandas (`np.load(..., allow_pickle=False)`, CSV parsing) from a scratchpad script. `test_labels.csv` was not opened.

Paths below are relative to `3rd Place/` unless prefixed with `harness/` (= `verification_workspace_3rd_place/extracted/harness_zip/`).

---

## Competition-Specific Checks

Pulled from `references/verification_comp_specific_rules.md`, with organizer answers applied. Each item has an owner.

| # | Check | Owner | Status or pointer from this review |
|---|---|---|---|
| C1 | Each test sample is processed independently. No batch-level normalisation, test-set statistics, cross-case calibration, pseudo-labelling, clustering, or TTA pooled across test cases. | Code Integrity | Harness code is per-scan; see §Flags F7. TorchScript members are traced in `eval()`, but they are binary, so the check is only as good as the static read. |
| C2 | Zip has `main.py` at its root, writes `submission.csv`, reads `/code_execution/data/niftis/` and `submission_format.csv`, runs offline, bundles all assets. Compare the harness zip with the tree and with README claims. | Code Integrity | Harness `main.py` differs from tree `inference/main.py` (Coordinator diff). README:67 says the tree copy "is the exact program submitted", which is inaccurate. See F5. |
| C3 | Predictions are probabilities in [0, 1] (metric: log loss). Check format, UID coverage and signs of hand editing. | Code Integrity | Coordinator: 524 rows, order matches, values in [0.0078, 0.9892]. Cap values match shipped `calibration.json` (b = −0.1646), not the documented b = −0.1478. See F4. |
| C4 | Every external dataset, annotation set and pretrained model is disclosed to organizers. Cross-check `driven_data.md` against what the code loads. | Data Source | `driven_data.md` does not disclose PPMI or the competitor's own hand annotations. README:59-63 and 565-569 do mention the annotations. See F1 and F2. |
| C5 | Annotations not supplied with the competition data (masks, striatal contours) count as external data. Establish who made them, from what data and under what licence. **Organizer ruling: hand annotations made by competitors on training scans are NOT permitted.** | Data Source / Eligibility | **Triggered.** The competitor hand-drew 276 striatal contours on training scans. They trained the shipped `ellipse.ts.pt`, which the harness uses at inference. See F1 (critical). |
| C6 | No forbidden or unlicensed external data. Data behind a DUA is allowed only if the DUA says the data is open. | Data Source | PPMI (3,839 scans) was held locally and used as an evaluation and selection instrument. No evidence it was used for training. Needs an organizer ruling. See F2. |
| C7 | Pretrained models: no NC licences, and their pretraining data must also be allowed. | Data Source | The shipped roster is from scratch: MONAI `DenseNet121`, no pretrained weights (`datscan/model.py:455-456`, `assert_from_scratch` at 478-487, 562-563). timm ImageNet backbones were tried and rejected, and do not ship (notes/REFUTED.md:418-429). See F6. |
| C8 | No competition data sent to third-party APIs or services. This also covers annotation and segmentation tooling. | Code Integrity / Data Source | The annotation server binds to `127.0.0.1` only (`segment_striata_server.py:289`). The harness makes no network calls. Open question: Claude Code was used "throughout", including agents for evaluation (`driven_data.md:111-113`), so competition data may have reached Anthropic's API. See F8. |
| C9 | Privacy: no reading of patient-identifying header fields and no cross-UID linking of individuals. | Code Integrity | Only `header.get_zooms()` is read (`harness/main.py:51`). `acq_meta.csv` holds geometry and dtype only. Nothing identifying found. |
| C10 | Solution under the MIT licence; all dependencies open source and commercially usable. | Release Readiness | `LICENSE` is MIT (© Paul Blanc-Durand). Requirements: torch, numpy, scipy, nibabel, pandas, monai, scikit-learn, matplotlib, all permissive. |
| C11 | Release: code only is published at announcement. Weights may go only to a private location that organizers can access. | Release Readiness | Weights and annotations are on a **private** HF repo (`README.md:51-63`, `models/README.md:13-15`). The competition-scan-derived `ellipse.ts.pt` and `canon_const.json` are committed in the tree under `inference/assets/`, so they would be published with the code unless removed. See F9. |
| C12 | Eligibility: not a sponsor employee, annotation-panel member or contributing-centre insider. | Eligibility | Organizer: "Confirmed by DrivenData". Context: the competitor is a nuclear-medicine physician at CHU Henri Mondor (AP-HP Paris) and in private practice (`driven_data.md:8-12`). AP-HP (Paris) is a different institution from AP-HM (Marseille), which is a contributing centre. |
| C13 | Eligibility: aged 18+, permitted territory, no Russian residency for prize money. | Eligibility | Not checkable statically. |
| C14 | Submission limits and multiple accounts. | Eligibility | Organizer: "Checked by DrivenData, no issues". |

---

## Dataset Policy

**Forbidden**
- External data that is not freely and publicly available to all participants, including private clinical or institutional archives and annotations.
- Any external data or model under an NC, CC NC or CC BY-NC licence.
- Data the competitor does not hold all rights to.
- **Hand annotations made by competitors on training scans (organizer ruling: "No").**
- Per the organizers, no competition data appears in any public dataset.

**Allowed**
- Competition `niftis/{uid}.nii.gz` training scans and `train_labels.csv`.
- External data and pretrained models that are freely and publicly available, commercially licensable and disclosed.
- DUA-gated data (for example PPMI) only if the DUA says the data is open source.
- Organizers approved no further datasets.

**Unresolved / needs organizer ruling**
1. PPMI (grep aliases `ppmi`, `PPMI_iso`, `ppmi_labels.csv`) was used as an out-of-distribution evaluation and selection instrument: 3,839 scans (notes/REFUTED.md:517-547, 27-29; config.py:574-578; transforms.py:188-189). Two questions for the organizers:
   - Does the PPMI DUA count as "open"? The PPMI DUA normally forbids redistribution and requires an application.
   - Does evaluation-only use, which informed model and recipe selection, count as "use"?
2. Blinded label re-reads by the competitor on 60+14 training scans (README:523-530). They fed a `row_weights` expert-concordance arm that was refuted and does not ship (REFUTED.md:181). They are not in the winning path: `meta/labels.npy` equals `train_labels.csv` exactly. They are still competitor-made annotations and should be noted to organizers.

---

## Pipeline Summary (as it actually runs)

### Offline (training side)
1. **Iso resample and centres**: nifti → 2 mm iso (`ndi.zoom`, linear). Head centres come from `build/provenance_ellipse/build_centers.py`.
2. **Legacy box cache** `comp.f16.npy`: 128×128×92 head box with whole-brain-mean normalisation (`datprep_iso.normalize`: mean of voxels > 0.15·p99.9) (`provenance_ellipse/build_box_mmap.py`). The same script also builds a `ppmi` cache (lines 33-36).
3. **Manual annotation** (`provenance_ellipse/segment_striata_server.py`): a local browser tool.
   - The "expert", who is the competitor ("the user annotated 66 more scans", `build/train_ellipse.py:66`; "user's correction", `segment_striata_server.py:7`), placed per-side ellipsoids plus one shared uptake threshold.
   - Coverage: 276 training scans in three sessions. 105 were error-matched to the model's errors, 105 were controls, and 66 more on 2026-08-22 included the random 50 (`provenance_ellipse/README.md:26-32`).
   - Output: `meta/striatal_masks/*.npz`, mapped to box space by `build_box_masks.py`. A **lost** one-off then built `box_automasks/`: expert masks for 276 scans and an "auto rule" for the remaining 1,086 (README.md:35-40).
4. **Ellipse localiser** (`build/train_ellipse.py`): 3D CNN regressing centre, radii and 6D rotation per side, trained with soft Dice plus moment-ellipsoid shape loss against the expert masks.
   - Log: `train 251 scans (251 expert) | holdout 25` (docs/repro/logs/train_ellipse.log:2).
   - Exported to `ellipse.ts.pt` by `build/export_ellipse.py`, traced in eval mode (lines 70-81).
   - README:387 says every result used the **shipped** `ellipse.ts.pt`.
5. **Canonical caches** (`build/build_canonv2_cache.py`), with the same code path as inference:
   - box → ellipse localiser → ellipsoid mask → 8-DOF-style canonical warp (`inference/canonize.py`).
   - The warp is a rotation from the L–R centre axis, a translation to REF, and a per-axis **head-span scale** to `canon_const.tgt_spans` clipped to [0.8, 1.25] (canonize.py:49-66).
   - Outputs: `canonv2.f16.npy` and `canonv2mask_ell.u8.npy`.
   - `canon_const.json` ("population-median spans") has no builder in the tree.
6. **Glands**: `glandrm.py` does blob-level parotid clamping. It is not enabled in the shipped calibration, which has no `gland_rm` key (harness/main.py:97).
7. **Projection** (`datscan/projection.py`, `PhysShape3N`, zero parameters):
   - Axial: S-I reduced to 4 channels [tau-softmax peak, mean, aniso×uptake, NDT tail]. `tau = 4.090·mu + 1.259`, with mu the per-scan box mean.
   - Sagittal: per-hemisphere views, 4 channels each, stacked into 8 channels (`sagfuse=attnres`).
   - NDT anchor = 0.5 × max inside the striatal ellipse mask (`ndt_anchor=striatal`).
8. **Model** (`datscan/model.py` `DatNet`): MONAI DenseNet121 2D axial trunk plus a DenseNet121 sagittal branch, fused through a zero-init attention gate. From scratch, BatchNorm.
9. **Training** (`datscan/train.py`, `recipes/train_fusion10.sh`):
   - Members `fu_a1..fu_a10_dnet`, where member k uses seed k and `meta/cv_round{k+1}.csv` (5 folds each), giving 50 fold-models.
   - Schedule: 120 epochs, AdamW with OneCycle, batch size 24.
   - Augmentation: LR flip, affine, Poisson counts, gamma gain. Counterfactual lesion synthesis: p = 0.25 of normals are lesioned inside the **ellipse-derived** mask (`mask3d_cache=canonv2mask_ell`) and relabelled abnormal.
   - Checkpoint = "best" epoch by validation-fold log loss (`recipes/fusion10_s078.json:78`, `train.py:916`).
   - Splits come only from training UIDs. All 15 `cv_round*.csv` files and `splits.csv` have 1,362 train UIDs, labels identical to `train_labels.csv`, and 0 test UIDs.
10. **Calibration** (`build/calibrate.py`, `build/build_roster.py:45-72`):
    - Ensemble logit `z` = mean over members of the clipped (±6) OOF logit, computed on the **training OOF** with `meta/labels.npy` (= train labels).
    - Only `b` is fitted, on the training OOF. `a` = 0.78 is a manual choice guided by public-leaderboard results: 0.85 scored 0.2313 and 0.78 scored 0.2317 (calibrate.py:9-25).
    - The cap of 6.0 is fixed.
    - Documented b = −0.147824566 (recipe:73, README:139, make_figures.py:262). The **shipped** b is −0.16455 (harness/assets/calibration.json). See F4.

### Inference (harness zip — what actually ran)
`harness/main.py`:
1. Read `submission_format.csv` and build nifti paths (lines 83-84).
2. Load all `*.ts.pt` except the ellipse file, which gives the 50 fu members. Load `ellipse.ts.pt` and `calibration.json` (85-107).
3. Preprocess each scan in a process pool, one scan per call (`preprocess`, 42-62): iso resample → largest head component crop → box → masked-mean normalise. A failed scan gets 0.5.
4. In batches of 8 (`BATCH`, line 26):
   - ellipse localiser → per-scan ellipsoid mask → `canonize.thetas` (per scan) → `grid_sample` → mask plus constant slabs (`with_slabs`).
   - For each member: flip-TTA (LR flip of the same scan), average the two logits, clamp to ±6. Then average the logits across members.
   - `p = sigmoid(0.78·z − 0.1646)`, clipped to [1e-6, 1−1e-6] (112-142). A batch of 1 is padded by duplicating the scan (116-117).
5. Write the CSV (157-158).

Nondeterminism: GPU `grid_sample` and cuDNN (README:431-452). Inference is claimed to be deterministic, but this was not verified here.

---

## Important Files

| Path | Role | In harness zip? |
|---|---|---|
| `harness/main.py` | Submitted entry point (8,762 B, 2026-09-05) | **yes** |
| `inference/main.py` | Tree copy with extra `ex_*` mixed-roster path (10,071 B) | no (differs) |
| `inference/canonize.py` | Native → canonical warp; per-axis head scale | yes (identical) |
| `inference/datprep_box.py`, `datprep_iso.py` | Head box and masked-mean normalisation | yes (identical) |
| `inference/glandrm.py` | Parotid removal; inactive (no `gland_rm` key) | yes (identical) |
| `inference/assets/ellipse.ts.pt` | Striatal ellipse localiser trained on the competitor's hand annotations | yes (identical) |
| `inference/assets/canon_const.json` | tgt_spans / REF / sh constants (no builder in tree) | yes (identical) |
| `assets/calibration.json` | a = 0.78, b = −0.16455, cap = 6 | yes only (absent from tree) |
| `assets/fu_a{1..10}_dnet_f{0..4}.ts.pt` | 50 fold-models (f16 TorchScript) | yes only (on private HF) |
| `datscan/{config,data,transforms,projection,model,lesion,train,export}.py` | Training engine | no |
| `datscan/submission_main.py`, `datscan/infer.py` | Older, stale entry and inference variants | no |
| `build/build_canonv2_cache.py` | Training cache, same code path as inference | no |
| `build/train_ellipse.py`, `build/export_ellipse.py` | Localiser training and export from the expert masks | no |
| `build/provenance_ellipse/*` | Annotation tool and data chain (verbatim, hard-coded `/ssd/...` paths) | no |
| `build/export_members.py`, `verify_traces.py`, `calibrate.py`, `build_roster.py` | Export, calibration, packaging | no |
| `build/build_box_mmap.py` | Box cache; optional `ppmi` branch (34-37) | no |
| `recipes/fusion10_s078.json` (+`train_fusion10.sh`) | Winning roster definition | no |
| `meta/labels.npy`, `uids.csv` | 1,362 train labels in train CSV order (exact match) | no |
| `meta/cv_round2..15.csv`, `splits.csv` | 5-fold partitions over train UIDs | no |
| `meta/acq_meta.csv` | Per-train-scan voxel geometry, dtype, 8 acquisition clusters | no |
| `meta/lomoe_severity.csv` | Per-train-scan severity index and groups (no builder in tree) | no |
| `meta/random50_uids.csv` | 50 train UIDs annotated as the "unbiased" sample | no |
| `docs/repro/logs*/`, `md5_weights_training_new_dat.txt` | Training logs (original and 2026-09-24 retrain) | no |
| `notes/REFUTED.md`, `notes/design-spec.md` | Research journal (PPMI use, rejected arms) | no |
| `tests/*` | Acceptance gates (not run) | no |

---

## Claims vs Code

| Claim | Where | What code/artifacts show |
|---|---|---|
| "`inference/main.py` is the exact program submitted" | README:67 | False. The harness `main.py` lacks the 2026-09-11 `ex_*` / `ex_shared.ts.pt` mixed-roster path (tree main.py:85-104, 149-154). With only fu_* assets that path is inert, so the behaviour is equivalent. |
| `p = sigmoid(0.78·z − 0.1478)` | README:139; recipe:73 ("reproduced by calibrate.py to 1.9e-08") | The shipped b is −0.16455 and the scored CSV's cap values match −0.16455. The shipped intercept is not reproducible from the documented runs. |
| "Scale deliberately kept / never normalised" | README:127, 159; driven_data.md:33, 55-56 | `canonize.thetas` rescales each axis to the median head span, clipped to [0.8, 1.25] (canonize.py:60-63). Striatal size relative to the head is kept; absolute head scale is not. Minor wording issue. |
| "Expert-drawn contours" | README:59-63, 565-567 | The expert is the competitor ("the user annotated", train_ellipse.py:66). The contours were drawn on 276 competition **training** scans. |
| "No external DaT dataset was used for training" | README:568-569; README:185 "rules banned external DaT data" | Consistent for training. PPMI was used for evaluation and selection (REFUTED.md:517-547, 27-29, 1027). It is not disclosed in driven_data.md. |
| "Only external tool: Claude Code" | driven_data.md:109-113 | Agents ran evaluation, launched training and scraped DrivenData submission pages (REFUTED.md:826-836 `dd_status.py`, which is not in the tree). |
| Inference needs only torch/numpy/scipy/nibabel/pandas | README:381-382 | Consistent with the harness imports. |
| The 0.2317/0.2732 package is the winning configuration | recipe | Intake public score 0.231321 matches the **a = 0.85** `fusion10` package (README:292). The Coordinator's recomputed public 0.231707 matches `fusion10_s078` (0.2317). This likely explains the Coordinator's public-score mismatch: the intake public number belongs to a sibling submission. |
| "Final standing 3rd" vs "6th place" | README:6 vs README:291, 367; recipe `private_rank: 6 of ~250` | Internal inconsistency; the leaderboard is presumably reranked after disqualifications. Informational. |

**Stale or unused paths**
- Tree `inference/main.py` ex_* path.
- `datscan/submission_main.py` and `datscan/infer.py`.
- Most of the 171 `Recipe` fields (README:471-476), including `hard_weight` ("TEST-LIKE acquisition clusters {0,1,2,5,6}", config.py:567-569; data.py:85-93), `row_weights`, `dann_w`, `recon_bank`, `train_subset`, `mask_feats_ckpt`, and timm pretrained backbones. None is set in `recipes/fusion10_s078.json` or `train_fusion10.sh`.
- `build/build_mask3d.py` reads `box_predmasks/`, which is not part of the winning chain.
- The glandrm path in the harness is inactive.

---

## Flags for Downstream Reviewers

**F1: CRITICAL, Data Source / Eligibility. Competitor hand annotations on training scans are load-bearing in the shipped model (organizer ruling: not permitted).**

What happened:
- The annotation tool writes manual per-scan ellipsoid and threshold contours (`build/provenance_ellipse/segment_striata_server.py:1-15, 219-221, 265-286`). It read `/ssd/datasets/DAT_SCAN/iso/{uid}.npy` (line 30), the competition scans.
- Provenance README says: "This step is manual annotation: 276 scans over three sessions" (`build/provenance_ellipse/README.md:26-32`). It also says "The only unreproducible step is 4 — the annotating itself is human work" (line 53).
- The annotator is the competitor: "HOLDOUT REDESIGN (2026-08-22, after the user annotated 66 more scans -> 276 expert)" (`build/train_ellipse.py:66`); "user's correction" (segment_striata_server.py:7). driven_data.md:8 says the competitor is a nuclear-medicine physician.
- The annotated scans are all training scans: `random50_uids.csv` holds 50 of 50 train UIDs and 0 test UIDs, and `build_box_masks.py:17,18,28` indexes them via `train_labels_JNDlMjr.csv`. There is no evidence of annotation on test scans.
- The localiser was trained on them: `docs/repro/logs/train_ellipse.log:2` shows `train 251 scans (251 expert)`.

Why it matters:
- The resulting `ellipse.ts.pt` ships in the harness. It drives canonical pose (`harness/main.py:118-131`), the striatal NDT anchor and slabs, and, during training, the lesion-synthesis masks and the `canonv2` caches for all 50 members (`recipes/fusion10_s078.json:61-67`).
- Removing it would require retraining everything.
- The annotations themselves (`ellipse_annotations.tar.gz`) are on private HF, and the reviewer did not fetch them.

**F2: HIGH / needs organizer ruling, Data Source. PPMI held and used.**
- `build/provenance_ellipse/build_box_mmap.py:33-36` and `build/build_box_mmap.py:34-37` build a PPMI box cache from `/ssd/datasets/DAT_SCAN/PPMI/`.
- 3,839 PPMI scans were used to evaluate members and packages, and to veto or rank recipes (notes/REFUTED.md:27-29, 517-547, 570, 1027).
- PPMI shaped augmentation design (config.py:574-578, transforms.py:188-189, projection.py:108, 368, 386).
- No training on PPMI was found.
- PPMI is not disclosed in `driven_data.md`. README:350 refers only to "an external-dataset check".
- Organizer questions: does the PPMI DUA count as "open", and does evaluation or selection use count as use?

**F3: MEDIUM, Data Source. Other competitor-made annotations.**
- Blinded label re-reads of training scans (README:523-530).
- A `weights_expert.npy` row-weight arm (config.py:570-571, REFUTED.md:181), refuted and not shipped.
- Labels in the shipped path equal the official labels, with 0 mismatches across `labels.npy`, all `cv_round*`, `splits.csv`, `acq_meta.csv` and `lomoe_severity.csv`.

**F4: MEDIUM, Code Integrity. The shipped intercept does not match the documentation.**
- `calibration.json` has b = −0.16455324638116142. The recipe, README and make_figures use −0.147824566.
- build_roster.py:57-58 fits b on training OOF only, using the `--runs` directory's `__best_oof_*`.
- The difference suggests the zip was built from a different run directory than the one the recipe documents (test_t5 references `runs_fusion150`).
- No code fits anything on test data. `a` was chosen with public-leaderboard feedback (calibrate.py:13, 21-23), which is ordinary LB-driven selection with two submissions, not use of test data.
- Code Integrity should confirm that the scored CSV is consistent with b = −0.16455. The Coordinator's cap-value check says it is.

**F5: LOW, Code Integrity.** The harness and tree `main.py` differ. The README's claim that the tree copy is the exact submitted program is inaccurate. The functional path is equivalent for fu-only assets.

**F6: LOW, Data Source. Pretrained weights.**
- Shipped members use MONAI `DenseNet121(...)` with no pretrained flag. MONAI's 2D DenseNet121 has no ImageNet weights by default.
- `assert_from_scratch` guards the weights at model.py:562-563, and the logs show nothing pretrained.
- timm convnext/eva02 (ImageNet, `pretrained=True`, model.py:208) were tried and rejected, and do not ship.
- Binary traces could not be inspected. The weight-std check is the competitor's own.

**F7: LOW, Code Integrity. Per-scan independence.**
- Batches of 8 go through TorchScript members traced in `eval()` (export.py:50, export_members.py:62; export_ellipse.py:70-73), so BatchNorm uses running stats.
- The export gate asserts batch-size independence (export.py:97-105).
- All preprocessing, `thetas`, `head_span`, `tau` (per-scan mean) and flip-TTA are per scan.
- The batch-of-1 duplicate pad is harmless.
- No test-set statistics, pseudo-labels or BN adaptation were found (harness/main.py:7-8, 112-142).
- Code Integrity may want to confirm from the TorchScript graph metadata, statically, that there is no `training=True` BatchNorm and no batch-dimension reduction.

**F8: OPEN, Data Source. Third-party API.** driven_data.md:111-113 says Claude Code agents did "model evaluation" and training orchestration. Notes include per-scan analyses. Did competition data (scan content, labels, rendered images) reach Anthropic's API? This needs an organizer view. The rules ban sharing competition data with third-party APIs or services.

**F9: LOW, Release Readiness.**
- `inference/assets/ellipse.ts.pt` and `canon_const.json` are committed in the tree. Weights trained on competition data should be released only to a private location that organizers can access.
- The README links a private HF repo. Competitor-hosted URLs were not fetched.
- The calibration asset is missing from the tree, so the tree alone cannot reproduce the submission.

**F10: INFO, Data Source.**
- `meta/` contains **no test UIDs**. A token scan of every text file in `3rd Place/` against the 524 `submission_format.csv` UIDs found 0 hits.
- All `meta/*.csv` files and `splits.csv` cover exactly the 1,362 train UIDs.
- `labels.npy` (float32, 615/747) equals `train_labels.csv` in the same order.
- `acq_meta.csv` holds header geometry from the training niftis.
- `lomoe_severity.csv` and `canon_const.json` have no builder in the tree, but only train UIDs appear in them.
- "TEST-LIKE clusters {0,1,2,5,6}" (config.py:567-569) were apparently inferred from leaderboard behaviour ("the easy-centre-depleted subset reproduces the board point"). The flag is unused in the winning recipe.

---

## Assumptions / Gaps
- The 50 fu_* traces and `ellipse.ts.pt` were not loaded, so equivalence to `datscan/` code is taken from competitor gates (tests/T3, verify_traces) that were not run.
- The step that assembled `box_automasks` is **lost** (provenance README:35-40). The "auto rule" masks for 1,086 scans cannot be audited. Their rule is said to be in the header of `segment_striata_server.py` and in a pre-cleanup `train_segmenter.py` that is absent.
- Builders for `canon_const.json`, `lomoe_severity.csv`, `acq_meta.csv` clusters and `head_centers.npy` are not in the tree.
- The origin of shipped b = −0.16455 is unknown.
- The annotation archive and prize tarball on private HF were not fetched, per the rules.
- It is assumed that "the user" in the competitor's code comments means the competitor himself. That fits a solo entry and the physician background, but only the organizers can confirm it.
