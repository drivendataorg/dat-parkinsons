# Reviewer 5: Release Readiness (3rd place, paulonium / Paul Blanc-Durand)

> **Note:** Findings in this file that treat the competitor's annotations of training scans as prohibited do not apply. The official rules permit annotating training data when the annotations are provided with the solution (`official_pages/rules.txt:137-138`). See F1 in `verification_report.md` for the final assessment.

Static, read-only review. **No competitor code was executed.** No competitor module was imported, and no `.pt` or `.pyc` artifact was loaded or unpickled. `.pyc` headers were read as raw bytes (16-byte header only) and scanned with `strings`. The PNG figures were viewed as images. UIDs were grepped against the organizer `train_labels.csv` and `submission_format.csv` (test labels were not opened). No competitor-hosted URL was fetched, including the Hugging Face repo. Nothing under `3rd Place/` or `references/` was modified.

Scope: all of `3rd Place/` (159 files, 8.87 MB) and `verification_workspace_3rd_place/extracted/harness_zip/`.

Release expectations applied (from `references/verification_comp_specific_rules.md`):
- **R1.** The code is released under MIT. All dependencies must be open source and allow commercial use.
- **R2.** Only the code is published, and only when winners are announced.
- **R3.** Weights trained on competition data may be held only in a private location that only the organizers can access, until the dataset is published.
- **R4 (governing rule E).** The data may be used "solely for the purpose and duration of the Competition".
- **R5 (governing rule A).** Never share, copy or publish the data. Tools that store or retain uploaded data are not allowed.

---

## 1. Competition data and data-derived artifacts in the tree

The UID grep found:
- **0 test UIDs** anywhere in the tree or the harness copy, including binaries, `.pyc` files and logs.
- Train UIDs only in `meta/*`, `splits.csv`, `visualize_dat.py` (the example uid `7875aud9`) and its `.pyc`.
- No per-UID records in `docs/repro/logs*` (aggregate epoch metrics only) or in `notes/`.

| Path | Contents | Recommendation |
|---|---|---|
| `meta/labels.npy` (5.6 KB) | The 1,362 official training labels, in `uids.csv` order. Solution overview F10: exact match to `train_labels.csv`. | **Remove from the public release.** This is a verbatim copy of competition labels (R2, R5). Keep it only in the organizer-private copy. |
| `meta/uids.csv` | 1,362 train UIDs | Remove. It is the competition index, only useful together with the data. It can be regenerated from `DAT_LABELS`. |
| `meta/cv_round{2..15}.csv`, `splits.csv` | uid, **label**, fold (rounds 4 and 5 have uid and fold only). The labels copy `train_labels.csv`. | **Remove the `label` column from the public release.** Keep uid and fold only after the dataset is published. Alternatively, ship a seeded split generator instead of the files. |
| `meta/acq_meta.csv` (178 KB) | Per-train-scan voxel spacing, matrix size, FOV, dtype, the competitor's 8 "acquisition clusters", and **label**. These are NIfTI header-derived acquisition metadata that act as a site/scanner proxy. | **Remove.** It is derived data and carries labels. It is not used by the winning recipe (only by the DG, `ccdann` and `hard_weight` arms: `datscan/train.py:423-428, 490-493`). |
| `meta/lomoe_severity.csv` (33 KB) | Per-train-scan label, cluster, a severity index `sev`, tercile and group. There is no builder in the tree. | **Remove.** It is label-derived per-scan data. It is used only by the `fuse_aux_band` arm (`train.py:466-470`), which is not in the winning recipe. |
| `meta/random50_uids.csv` | 50 train UIDs that the competitor hand-annotated | Remove from public, or keep private. It identifies the annotated subset. |
| `inference/assets/ellipse.ts.pt` (3.79 MB) | A TorchScript striatal localiser **trained on competition training scans and on the competitor's 276 hand annotations**. It is byte-identical to the harness copy. | **Blocker: remove from the public tree.** This is a weight file trained on competition data (R3). Keep it only in the organizer-private location. Note that `models/.gitignore` ignores `*.pt`, but this copy sits in `inference/assets/`, which is not ignored. |
| `inference/assets/canon_const.json` (90 B) | "Population-median" head spans `[122,124,82]`, the reference point and the shape. These are 3 aggregate statistics from the training scans. There is no builder in the tree. | Low risk as aggregate constants. It is acceptable to keep, but the organizers should decide. A builder should be added. |
| `docs/figures/*.png` (5 files, 3.5 MB) | **Renderings of real competition training scans, each annotated with its uid and label** (details below). | **Blocker: remove, or replace with synthetic or public-data renders**, until the dataset is published and the organizers approve (R2, R5). |
| `docs/repro/logs*/` (101 logs), `md5_weights_training_new_dat.txt` | Aggregate training curves only. The files contain `/ssd/datasets/DAT_SCAN/...` paths and md5 hashes of retrained checkpoints. | Acceptable content-wise, but strip the absolute paths. |
| `notes/REFUTED.md` (98 KB) | The research journal. It has no UIDs. It describes PPMI use, agent use, scraping of DrivenData leaderboard pages, and other teams' names and scores. | Acceptable for release after the announcement, since the leaderboard is public then. It is also the only place PPMI use is disclosed (see §5). |

### The PNG figures depict competition scans

I viewed each figure. All 7 UIDs shown are **train** UIDs (checked against `train_labels.csv`: 7/7 in train, 0/7 in test).

| Figure | What it shows |
|---|---|
| `preprocessing.png` | Scan `53lc58dr`: the native NIfTI axial MIP (150×150×112, 2.46 mm), then the resample, head box, normalisation, and canonical frame with the striatal ellipsoid contour. |
| `channels.png` | `3xlrfcd6` (NORMAL) and `53lc58dr` (ABNORMAL): the canonical box and the four projection channels. |
| `inputs.png` | `3xlrfcd6` (NORMAL, p=0.008), `53lc58dr` (ABNORMAL, p=0.989) and `aietdn1t` (ABNORMAL mild, p=0.653): all 4 axial channels and 4 of the 8 sagittal channels. |
| `lesion.png` | `3xlrfcd6`, a real normal scan, with synthetic lesions at 4 severities and the model's probability for each. |
| `discordant.png` | `7875aud9` and `0imjyjcm`, labelled ABNORMAL, which the model calls normal. `y08p0fod` and `gom6736z`, labelled NORMAL, which the model calls abnormal. For each: uid, stored label, OOF p, and 5 image panels. |

These are image data from the competition, published together with labels. `discordant.png` goes further: it publicly asserts that specific named training scans are likely mislabelled. That reveals information about the organizers' ground truth before the dataset is released. Recommendation: remove all five from the public release. If the organizers want figures, the competitor could regenerate them with the UIDs and labels stripped after the dataset is published, or render them from a public phantom or synthetic volume.

## 2. Secrets, personal data, paths, private references

- **Secrets and credentials.** None found. I grepped for `hf_…`, api_key, token, password, bearer, AWS/GitHub/OpenAI key patterns and emails. The only email-like or named items are the author name in `LICENSE`.
- **Username and home path in bytecode.** Every `.pyc` embeds `/home/pbd/PROJETS/DAT_challenge/...`, which reveals the username `pbd`. `build/__pycache__/build_canonv2_cache.pyc` and `__pycache__/visualize_dat.pyc` also embed `/ssd/datasets/DAT_SCAN/...` defaults. `build/provenance_ellipse/README.md:13` names `~/PROJETS/DATscan`.
- **Hard-coded absolute paths in source.** `build/provenance_ellipse/{segment_striata_server.py:30, build_centers.py, build_box_mmap.py, build_box_masks.py}` and `build/build_box_mmap.py` contain `/ssd/datasets/DAT_SCAN/{iso,boxcache,PPMI/PPMI_iso,PPMI/ppmi_labels.csv,train_labels_JNDlMjr.csv}`. They are documented as "kept verbatim as historical record". The same paths appear in all 100 `docs/repro/logs*/fu_*.log` files (`box_cache`, `mask3d_cache`) and in the md5 manifest. These reveal the local data layout and that a local PPMI copy exists. They are not secrets. Recommendation: parameterise them via `datscan/paths.py` or redact them.
- **Hostnames and IPs.** None, apart from `localhost:8766` for the annotation server, which binds to 127.0.0.1.
- **Private HF repo `huggingface.co/paulonium/dat-challenge-winner`** (README.md:51-63, models/README.md:13-15, build/provenance_ellipse/README.md:42-49). I did not visit it. According to the competitor's own docs it holds:
  - `prize_submission.tar.gz` (2.9 GB, sha256 `9752e5b3…`). This is "this repo + all trained weights". By the README's description it therefore also contains `meta/labels.npy`, the label-bearing CSVs, the scan figures and `ellipse.ts.pt`.
  - `ellipse_annotations.tar.gz` (4.9 MB, sha256 `77b4e3b4…`). It contains `box_automasks/` for **all 1,362 train scans**, the raw expert `striatal_masks/`, `head_centers.npy`, a localiser checkpoint `striatal_ellipse_sh0.5.pt` and `random50_uids.csv`.

  **Assessment against the rules:**
  - (a) R3 allows weights in a private location that *only organizers* can access. The README says "private — request access". That means the competitor controls who is granted access, and the repo is hosted by a third party (Hugging Face). Whether only organizers have access cannot be verified statically.
  - (b) The annotations, masks, head centres and labels are competition-data-derived **data, not weights**. R3 does not cover them. Under R4 and R5, keeping them on a third-party service after the competition closed (2026-09-16) is not clearly "solely for the purpose and duration of the Competition", although winner handoff and verification is arguably part of that purpose.
  - (c) The derived masks were produced by hand annotation, which the organizers have ruled not permitted (solution overview F1). Their retention and hosting is a secondary issue beside that ruling.

  **Recommendation:** the organizers should confirm the HF access list, and the competitor should keep the repo private. DrivenData should take the handoff copy into organizer-controlled storage, and the competitor should then delete the HF copy, or keep it strictly organizer-only until the dataset is published. All links to it should be removed from the public README.
- **Local retention after the close.** `docs/repro/md5_weights_training_new_dat.txt` records a full 50-fold retrain on 2026-09-24, after the 2026-09-16 close, at `/ssd/datasets/DAT_SCAN/runs_fusion10`. The competitor therefore still held the competition data and derived caches (including the PPMI cache path) locally after the close. It was apparently done to support the reproducibility deliverable. Open question for the organizers: does this fall within rule E's "purpose and duration"? When should the competitor delete the data?
- **Third-party AI tooling (for Data Source, noted here as a release and documentation issue).**
  - `driven_data.md:111-113`: Claude Code was used "throughout", including for model evaluation.
  - `notes/REFUTED.md:826-836, 1005`: multiple agents (A, B, C) ran analyses, and one "fabricated" a table.
  - Agents did authenticated GETs of DrivenData pages (`dd_status.py`, which is not in the tree, and `leaderboard_partial`).
  - Rule A bans tools "that store or retain uploaded data". Whether scan content, labels or renders reached a retaining service is unresolved.

## 3. License and attribution

- `LICENSE`: standard MIT text, "Copyright (c) 2026 Paul Blanc-Durand". This is correct and matches the solo entry.
- `README.md:564` says "Code is the authors'", which is slightly ambiguous next to MIT. Suggest: "Code is released under the MIT License (see LICENSE)".
- **Vendored third-party code.** No verbatim vendored files or copied license headers were found.
  - `datscan/optim.py` implements Muon and Lion and cites `github.com/KellerJordan/Muon` (MIT) for the recipe. It appears to be a re-implementation. It is unused by the winning recipe. Suggest adding an explicit attribution line.
  - `datscan/transforms.py:167` "ported from the old `--psfaug`" refers to the competitor's own earlier code.
- **Dependencies.** All pinned dependencies are permissive: torch BSD, numpy/scipy BSD, nibabel MIT, pandas BSD, monai Apache-2.0, scikit-learn BSD, matplotlib PSF-based, pytest MIT.
  - Optional imports that are not pinned: `timm` (Apache-2.0), `e2cnn` / `escnn` (BSD-3-Clause-Clear), and `joblib` (BSD). They are all lazy or in dead arms, so none is needed for the winning path.
  - `monai_pipeline.learnproj` (`datscan/train.py:323`) is a **missing module** from the old workspace. It is a dead arm.

## 4. Hygiene, reproducibility and packaging

**Bytecode and stale files**
- **`__pycache__` / `.pyc`**: 14 files, about 290 KB. They leak the username and home path (§2).
  - Two are stale: their embedded source size differs from the current source (`build/__pycache__/build_canonv2_cache` 6,767 vs 6,731 B; `__pycache__/visualize_dat` 9,982 vs 9,945 B).
  - `tests/__pycache__/…-pytest-9.0.3.pyc` shows the tests were run under pytest 9.0.3, while `requirements.txt` pins `pytest==8.*`.
  - **Remove all of them** and add a root `.gitignore` with `__pycache__/`, `*.pyc`, `work/`, `logs/*`.
- **Stale code**:
  - `datscan/submission_main.py` and `datscan/infer.py` are older entry points.
  - The tree's `inference/main.py` differs from the submitted harness `main.py` (the ex_* mixed-roster path), yet README:67 says it "is the exact program submitted". Fix the README, or ship the harness `main.py`.
  - Harness `main.py:1` refers to `package.py`, which does not exist (the packager is `build/build_roster.py`).
  - `visualize_dat.py` sits at the repo root, is undocumented in the README, and defaults to a train uid.
  - Most of the 171 `Recipe` fields are dead arms. README:469-476 explains why they are kept.
- **Inconsistency**: `models/.gitignore` ignores `canon_const.json`, but a tracked copy lives in `inference/assets/`.

**Reproducibility gaps (a fresh verifier cannot reproduce the scored package from the tree alone)**
1. **Calibration.**
   - `calibration.json` is not in the tree. It is only in the harness zip and on the private HF.
   - Its `b = −0.16455` differs from the documented `b = −0.1478` (README:139, recipe:73, make_figures.py:262).
   - `build/build_roster.py:57-72` refits `b` from whichever `--runs` directory it is given, so the shipped `b` came from an undocumented run directory.
   - Recommendation: commit the shipped `calibration.json` and document the runs it came from, or correct the docs.
2. **Missing builders.**
   - No builder for `canon_const.json`, `meta/lomoe_severity.csv`, the `acq_meta.csv` clusters, `meta/centers.npy` / `head_centers.npy`, `meta/cv_round*.csv` / `splits.csv`, or the 2 mm iso cache.
   - The step that assembled `box_automasks/` is recorded as **lost** (provenance README:35-40).
   - The ellipse localiser can be retrained only from the private HF annotation archive, and the annotation itself is a manual, unrepeatable step.
3. **Weights.** All 50 `fu_*` weights are only on the private HF. `tests/test_t3` and `test_t5` need them or the original run directories (`runs_fusion150`, `runs_full20`, which are not shipped) and skip otherwise. The gates therefore cannot be reproduced from the release.
4. **Hard-coded paths.** Hard-coded paths in `build/provenance_ellipse/*` block re-running that chain as-is.
5. **Environment.** `requirements.txt` pins exact versions (torch 2.10.0+cu128, and so on) and the README's setup steps are adequate otherwise. `pytest==8.*` does not match the 9.0.3 actually used. `joblib`, `timm`, `e2cnn` and `escnn` are unlisted.
6. **Minor documentation mismatches.**
   - README:433 says the retrain ran on "2026-09-25"; the md5 manifest says 2026-09-24.
   - README:6 says "Final standing 3rd" while README:291 and 367 say "6th place".
   - The README says "scale never normalised", but canonize applies a clipped per-axis head-span scale.

**Artifact size**
- Tree: 8.87 MB. Of that, figures are 3.5 MB, `ellipse.ts.pt` 3.8 MB and `.pyc` about 0.3 MB.
- External weights: about 3.2 GB (50 f16 traces), a 2.9 GB tarball.
- The public code-only release, after the removals above, would be about 1.5 MB, which is fine.

## 5. Documentation gaps (`driven_data.md`)

| Gap | Evidence | Required? |
|---|---|---|
| **PPMI use is not disclosed.** 3,839 PPMI scans were held locally and used for evaluation and selection. | `notes/REFUTED.md:27-29, 517-547, 1027`; `build/build_box_mmap.py:34-37`; `/ssd/datasets/DAT_SCAN/PPMI/*` paths | Yes. All external data "must be shared with the challenge organizers" (Homepage). |
| **The competitor's own hand annotations are not disclosed** (276 striatal contours on training scans, plus blinded label re-reads). They are load-bearing via `ellipse.ts.pt`. | README:59-63, 523-530, 565-567; `build/provenance_ellipse/README.md`; `build/train_ellipse.py:66` | Yes. These are external annotations, and the organizers have ruled them not permitted. |
| **The extent of AI-agent use is understated.** Section 9 answers "No" to extra tools, then mentions Claude Code. It does not say whether data, labels or renders were sent to the service, and it omits the agent-run scraping of DrivenData pages (`dd_status.py`, not shipped). | driven_data.md:109-113; REFUTED.md:826-836, 1005 | Needed to assess rule A. |
| Where the weights and annotations are hosted (private HF) and who has access | Only in the README | Recommended |
| The calibration intercept actually shipped, and how `a=0.78` was chosen (with public-LB feedback) | calibrate.py:9-25 | Recommended |
| Section 3 says "scale deliberately never normalized", but canonize rescales head span within [0.8, 1.25] | canonize.py:60-63 | Minor |

## 6. Release-blocking changes (before any public MIT code release)

1. Remove `inference/assets/ellipse.ts.pt`. It is a weight file trained on competition data and on the competitor's hand annotations (R3).
2. Remove `docs/figures/*.png`, which are real training scans with UIDs and labels, and the README image links (R2, R5).
3. Remove `meta/labels.npy`, `meta/acq_meta.csv` and `meta/lomoe_severity.csv`. Remove the `label` column from `meta/cv_round*.csv` and `splits.csv`, or remove the files and ship a split generator (R2, R5).
4. Remove the links to the private HF repo from the public README, or replace them with "available to organizers on request". The organizers should confirm the HF access list is organizer-only (R3), and should decide on retention of the annotation archive and prize tarball under rule E.
5. Delete all `__pycache__/` directories, which embed the username and home path.

**Non-blocking (recommended):**
- Redact the `/ssd/...` and `~/PROJETS` paths.
- Commit the shipped `calibration.json` and fix the documented `b`.
- Fix README:67 (tree vs harness `main.py`) and README:6/291 (3rd vs 6th).
- Add builders, or document the provenance, for `canon_const.json`, the splits and the centres.
- Add unlisted optional dependencies, or remove dead-arm imports.
- Fix the pytest pin.
- Add the Muon attribution.
- Complete `driven_data.md` with PPMI, the hand annotations and AI-agent data handling.

Note: whether any release happens at all depends on the organizer ruling on F1 (hand annotations, which are not permitted). If the entry is disqualified, the release items become moot. The data-retention questions remain.

## 7. Other reviewers

When this report was written, `reviewers/` contained only `solution_overview.md`, so no other reviewer outputs were incorporated. The findings here are consistent with solution overview flags F1, F2, F4, F5, F8, F9 and F10.
