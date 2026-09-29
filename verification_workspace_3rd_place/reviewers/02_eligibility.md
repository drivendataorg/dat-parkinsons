# Reviewer 2: Eligibility and Disqualifier (3rd place, paulonium)

> **Note:** Findings in this file that treat the competitor's annotations of training scans as prohibited do not apply. The official rules permit annotating training data when the annotations are provided with the solution (`official_pages/rules.txt:137-138`). See F1 in `verification_report.md` for the final assessment.

**Status: fail.** The organizers ruled that hand annotations made by competitors on training scans are not permitted. The scored submission depends on exactly such annotations, and nothing can replace them without retraining. Two further items need an organizer ruling: PPMI, and use of the Claude API.

**No competitor code executed: yes.**
- No competitor script, test, notebook or module was run or imported.
- No `.pt` file was loaded.
- My own Python stdlib `csv` snippet read only these files:
  - `meta/random50_uids.csv`
  - `train_labels.csv`
  - `submission_format.csv`
- I did not open `test_labels.csv`.
- I fetched only two third-party public licence pages: the GitHub LICENSE files for QUVA-Lab e2cnn and escnn.
- I fetched no competitor-hosted URL; the Hugging Face repo was not accessed.

Paths are relative to `3rd Place/`. `harness/` means `verification_workspace_3rd_place/extracted/harness_zip/`.

---

## 1. Disqualifier review

### D1 (CRITICAL): the shipped model was trained on the competitor's own hand annotations of training scans

**Organizer rule.** `references/verification_comp_specific_rules.md`, "Other competition-specific checks" (Data Source): hand annotations made by competitors on training scans are permitted: **"No"**.

**Evidence that the behaviour happened.** I verified each of these myself.

- **Manual annotation tool.** `build/provenance_ellipse/segment_striata_server.py:1-15` describes a browser UI in which a person places ellipsoids per side and sets one threshold.
  - Line 30 reads scans from `/ssd/datasets/DAT_SCAN/iso`.
  - Line 32 writes `meta/striatal_masks/<uid>.npz`.
  - Line 4 runs it under `conda activate paire_training`.
- **The documentation says it is manual.**
  - `build/provenance_ellipse/README.md:26-32`: "**This step is manual annotation**: 276 scans over three sessions (105 error-matched + 105 controls in Aug, 66 more incl. the 50-scan unbiased random sample on 2026-08-22)".
  - Same file, line 53: "the annotating itself is human work".
- **The annotator is the competitor.**
  - `build/train_ellipse.py:66`: "after the user annotated 66 more scans -> 276 expert".
  - `segment_striata_server.py:7`: "user's correction".
  - `driven_data.md:8,14` describes a nuclear-medicine physician entering solo.
  - No third-party annotator is named anywhere.
- **The annotated scans are competition training scans.**
  - `README.md:565-566`: "trained on 276 expert-drawn contours **from this dataset**".
  - `train_ellipse.py:56` indexes `train_labels_JNDlMjr.csv`.
  - My own check: all 50 UIDs in `meta/random50_uids.csv` are train UIDs, and none is a test UID.
- **They trained the shipped localiser.**
  - `train_ellipse.py:61-84` trains only on scans flagged `expert` (unless `--auto` is passed).
  - `docs/repro/logs/train_ellipse.log:2`: `train 251 scans (251 expert) | holdout 25`.
  - `README.md:386-390`: "Every result … was produced with the SHIPPED inference/assets/ellipse.ts.pt".

**How much the scored submission depends on it: completely.**
- `harness/assets/ellipse.ts.pt` (sha256 `963311…3b44`) is byte-identical to `inference/assets/ellipse.ts.pt`, per the Coordinator.
- `harness/main.py:118` calls `out = ell(xb…)` on every batch with no fallback path. If the file is missing, `ell` stays `None` (lines 103-105) and `predict` would crash.
- Its output drives each of these:
  - the ellipsoid mask (lines 119-129)
  - the canonical warp `CZ.thetas` and `CZ.canonize` (lines 130-131)
  - the NDT-anchor slabs fed to all 50 members (`CZ.with_slabs`)
- In training, the winning recipe uses `ndt_anchor=striatal`, `box_cache=canonv2.f16.npy` and `mask3d_cache=canonv2mask_ell.u8.npy`. Both caches are derived from the ellipse (`recipes/fusion10_s078.json` shared_recipe; `recipes/train_fusion10.sh:8-10`).
- As a result, every one of the 50 members, and the lesion-synthesis masks, depend on the annotation-trained localiser.

**Disclosure.**
- Mentioned in `README.md:59-63` and `565-568`, and in the provenance README.
- **Not mentioned** in the prize documentation `driven_data.md`. §9 says "No" other tools and names only Claude Code. §5 mentions the "per-hemisphere ellipse localiser" without saying how it was trained.

**Mitigating facts.**
- The disclosure in the README appears made in good faith.
- The annotations cover training scans only; I found no test-scan annotation.
- No labels were altered. `meta/labels.npy` equals `train_labels.csv` (Reviewer 1, F10).

**Conclusion.** The documentation and code explicitly describe behaviour the organizers have ruled prohibited, and the scored submission cannot run without it. This meets the disqualify threshold unless the organizers change their ruling.

### D2 (HIGH, needs organizer ruling): PPMI held locally and used for evaluation and model selection; not disclosed in `driven_data.md`

**Evidence.**
- PPMI box caches are built from `/ssd/datasets/DAT_SCAN/PPMI/{ppmi_labels.csv, PPMI_iso}`:
  - `build/build_box_mmap.py:34-37` (comment: "optional external cohort")
  - `build/provenance_ellipse/build_box_mmap.py:33-36`
- `notes/REFUTED.md:517-521` records "3,839 PPMI scans" used for AUC of members.
- PPMI was also used to judge candidate settings:
  - `REFUTED.md:27-29` scores candidate knobs by PPMI AUC.
  - `REFUTED.md:545-547`: "PPMI is now 0/2 as a ranker … It remains a veto for catastrophic failure only".
- PPMI shaped some design choices: `datscan/config.py:574-578`, `datscan/transforms.py:188-189`, and `datscan/projection.py:108,368,386`.

**How much the shipped model depends on it.**
- I found no training on PPMI.
- `bg_sub`, `bg_aug` and `scatter_aug` are all off, because the winning recipe does not set them.
- The shipped `ndt_anchor=striatal` was actually marked "OOD-negative" on PPMI (`REFUTED.md:27`) and shipped anyway. So PPMI did not decide the final roster.
- Its role was informal model selection only.

**Disclosure.**
- Absent from `driven_data.md`.
- `README.md:350` mentions only "an external-dataset check".
- `README.md:568` says "No external DaT dataset was used for **training**".
- The rules require that all external data be shared with the organizers.

**Organizer questions.**
1. Is the PPMI data use agreement "open"? The organizer rule allows DUA data only if the DUA says the data is open.
2. Does using PPMI for evaluation and selection count as "use"?
3. Is the missing disclosure a separate violation?

### D3 (MEDIUM, informational): blinded label re-reads and expert-concordance weights

- `README.md:523-531` describes three rounds of blinded re-reading of the competitor's own training scans: 14 errors, 60 blinded reads, and 14 repeats.
- The only way these reads could reach a model is the `row_weights` option in `datscan/config.py:570-571`, which accepts `meta/weights_expert.npy`. That code is at `datscan/train.py:236-239`.
- Status of that option:
  - It was rejected: `REFUTED.md:181` records +0.0201 log loss.
  - It is not set in the winning recipe, and its default is `""`.
  - `meta/weights_expert.npy` is not in the tree.
- **They do not reach the scored model.** They are still competitor-made annotations of training scans, so the organizers should be told, but they do not by themselves disqualify.

### D4 (OPEN, needs organizer ruling): third-party API

- `driven_data.md:111-113` says Claude Code was used "throughout", including for "model evaluation" by agents.
- The homepage rule (`governing_rules.md` A) forbids tools "that store or retain uploaded data".
- Static review cannot tell whether scan content, labels or renders reached Anthropic's API.
- `REFUTED.md:826-836` also records agents running `dd_status.py` to read DrivenData submission pages. That script is not in the tree.
- Organizers need to decide this.

### Other cheating indicators searched for

- **Test data in the tree:** Reviewer 1 and the Coordinator found no test UIDs in any text file. My random50 check agrees.
- **Test-set statistics or pseudo-labels:** none found in the harness.
  - `harness/main.py:100-102` notes that there is no test-set statistic.
  - `hard_weight` ("TEST-LIKE clusters", `config.py:567-569`) is inferred from leaderboard behaviour, is off in the winning recipe, and defaults to 1.0.
  - The calibration slope `a=0.78` was chosen with public-leaderboard feedback. That is ordinary leaderboard probing, not test-data use.
- **Network calls:** the harness has none (no http, requests, socket or urlopen). The annotation server binds to `127.0.0.1` only (`segment_striata_server.py:292`).
- **Multiple accounts and submission cap:** "Checked by DrivenData, no issues" (rules file).

---

## 2. Licence review

**`LICENSE`:** MIT License, "Copyright (c) 2026 Paul Blanc-Durand", standard MIT text. **Pass.**

### Dependencies

| Package (version) | Where used | Licence | Free commercial use | OSI |
|---|---|---|---|---|
| torch 2.10.0 (cu128 wheel) | harness, datscan, build | BSD-3-Clause; bundled NVIDIA CUDA/cuDNN libraries are under the NVIDIA EULA (free, redistributable) | yes | yes (torch); CUDA runtime is proprietary but free |
| numpy 1.26.4 | all | BSD-3-Clause | yes | yes |
| scipy 1.17.1 | harness, build | BSD-3-Clause | yes | yes |
| nibabel 5.4.2 | harness `main.py:10` | MIT | yes | yes |
| pandas 2.3.3 | harness, build | BSD-3-Clause | yes | yes |
| monai 1.6.0 | `datscan/model.py` (DenseNet121, EfficientNetBN, SEResNet50, ResNet) | Apache-2.0 | yes | yes |
| scikit-learn 1.9.0 | eval / build | BSD-3-Clause | yes | yes |
| matplotlib 3.10.9 | figures, annotation tool | Matplotlib licence (PSF-based, BSD-compatible) | yes | yes (PSF-style) |
| pytest 8.* | tests | MIT | yes | yes |
| joblib (not pinned) | `build/provenance_ellipse/build_centers.py:2` | BSD-3-Clause | yes | yes |
| timm (not pinned) | `datscan/model.py:204` (rejected arm) | Apache-2.0 | yes | yes |
| e2cnn (not pinned) | `datscan/model.py:171,225…` (rejected arms) | Qualcomm BSD-style (fetched GitHub LICENSE) | yes | BSD-style |
| escnn (not pinned) | `datscan/model.py:178,265…` (rejected arms) | Qualcomm BSD-style (fetched GitHub LICENSE) | yes | BSD-style |
| `monai_pipeline` | `datscan/train.py:323`, only when `DATSCAN_MIRROR_OLD` is set | competitor's own legacy module, absent from the tree | n/a | n/a |
| Python stdlib (`http.server`, `urllib`, `subprocess`, `concurrent`, …) | various | PSF | yes | yes |

**Licence findings.**
- No non-commercial or non-OSI licence appears in the inference path or the shipped training path.
- Minor: `joblib`, `timm`, `e2cnn` and `escnn` are imported but missing from `requirements.txt`. Only `joblib` is on the reproduction path (the provenance scripts).

### Pretrained weights

- **The shipped members start from scratch.**
  - `datscan/model.py:455-456` builds MONAI `DenseNet121(...)` with no `pretrained` argument. MONAI's DenseNet121 does not load ImageNet weights by default.
  - `assert_from_scratch` (`model.py:478-487`) is called at lines 562-563 for every backbone except timm ones.
  - The winning roster, `fu_a1..10_dnet`, is densenet121 (`recipes/train_fusion10.sh:17`).
- **`ellipse.ts.pt`:** a custom EllipseNet trained from scratch by `train_ellipse.py`, with no pretrained load. Its training data is the D1 annotations.
- **The only pretrained weights in the tree** are timm ImageNet backbones (`model.py:208`, `pretrained=True`, convnext and eva02). They were rejected, are not in the recipe, and do not ship.
- **Caveat:** the TorchScript binaries were not loaded, so "from scratch" rests on the source code and the competitor's own guard.

---

## 3. Eligibility items only organizers can confirm

- **Not a sponsor employee or officer, annotation-panel member, or contributing-centre insider:** confirmed by DrivenData per the rules file.
  - Context: CHU Henri Mondor is part of AP-HP (Paris). The contributing centre AP-HM is in Marseille, a different institution.
- **Aged 18 or over, permitted territory, not resident in Russia:** confirmed by DrivenData per the rules file. Not checkable statically.
- **Accounts and submission limits:** checked by DrivenData, no issues.
- **Open question (low):** the conda environment `paire_training` (`segment_striata_server.py:4`) and `driven_data.md:9` (co-founder of PAIRE) suggest work may have been done on company infrastructure.
  - The rules require the competitor to hold all rights and to license under MIT.
  - The organizers may want a confirmation that no employer IP claim exists.
