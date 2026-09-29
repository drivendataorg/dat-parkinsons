# Reviewer 4: Data Source and Model Provenance (3rd place, paulonium)

> **Note:** Findings in this file that treat the competitor's annotations of training scans as prohibited do not apply. The official rules permit annotating training data when the annotations are provided with the solution (`official_pages/rules.txt:137-138`). See F1 in `verification_report.md` for the final assessment.

Static, read-only review. **No competitor code was executed.** No competitor module was imported, no `.pt`, `.npy` or `.npz` artifact was loaded or unpickled, and no competitor-hosted URL was fetched (the Hugging Face repo `paulonium/dat-challenge-winner` was not fetched). Competition renders in `docs/figures/*.png` were deliberately **not** opened, because viewing them through this reviewer would itself send competition-derived images to a model API. UID scans used `grep -F -w` against the official train and test UID lists (`submission_format.csv` gives UIDs only; `test_labels.csv` was not opened).

Public third-party pages fetched on 2026-09-29:
- PPMI Data Use Agreement v5.0, April 2026: https://www.ppmi-info.org/sites/default/files/docs/ppmi-data-use-agreement.pdf. Text was extracted locally from the PDF streams.
- Claude Code data-usage page: https://code.claude.com/docs/en/data-usage.

Paths are relative to `3rd Place/` unless marked `harness/` (= `verification_workspace_3rd_place/extracted/harness_zip/`).

---

## 1. External source inventory

| # | Source / artifact | Kind | Evidence (file:line) | Reaches shipped assets? | Disclosed in `driven_data.md`? | Verdict |
|---|---|---|---|---|---|---|
| E1 | **Competitor hand-drawn striatal contours**: 276 training scans | Annotation set (external data, per the rules file) | `build/provenance_ellipse/README.md:26-32,53`; `segment_striata_server.py:1-15,265-286`; `build/train_ellipse.py:61-84`; `docs/repro/logs/train_ellipse.log:2` | **Yes.** They trained `ellipse.ts.pt` (in the harness), which feeds all 50 members. | **No** (README:59-63, 565-567 only) | **Prohibited by organizer ruling. Critical.** |
| E2 | "Auto-rule" masks for the other 1,086 train scans (`box_automasks/`, `expert=0`) | Rule-derived annotation | provenance README:35-40; `train_ellipse.py:46,82-83` | **No** for the shipped localiser: it was trained without `--auto` ("251 expert"). | No | Not load-bearing. The step that built them is lost. |
| E3 | Earlier UNet voxel segmenter and `box_predmasks/` | Model trained on E1 | `train_ellipse.py:5-17,85`; `build/build_mask3d.py:9` | No (not in the winning chain) | No | Derived from E1. Not shipped. |
| E4 | Competitor's blinded label re-reads (14 + 60 scans, 3 rounds) and `err_divergent` / `err_recoverable` sets | Competitor re-labels of training scans | README:521-541; `build/make_figures.py:289-293`; REFUTED.md:181, 719-745, 900 | **No.** `row_weights` and `train_subset` are unset in `recipes/fusion10_s078.json` and `train_fusion10.sh`. | No | Competitor annotations, but only for analysis. Report to organizers. |
| E5 | `weights_expert.npy` expert-concordance row weights | Derived from E4 | `datscan/config.py:570-571`; REFUTED.md:181 (refuted) | No | No | Not shipped |
| E6 | **PPMI** (Parkinson's Progression Markers Initiative, via LONI IDA): 3,839 DaT SPECT scans held at `/ssd/datasets/DAT_SCAN/PPMI/` | External DaT dataset | `build/build_box_mmap.py:34-37`; `provenance_ellipse/build_box_mmap.py:33-36`; REFUTED.md:27-29, 517-547, 570, 1027; `config.py:574-578`; `transforms.py:186-189`; `projection.py:108,368,386` | No weights or statistics found. It was used for evaluation, ranking and veto, and it motivated design choices (see §3). | **No.** README:350 says only "an external-dataset check". | **High. The DUA is not open, so under the ruling this is not permitted external data. Needs an organizer ruling on evaluation-only use.** |
| E7 | MONAI `DenseNet121` (2D), from scratch | Architecture code, Apache-2.0 | `datscan/model.py:455-456,562-563` (`assert_from_scratch`); logs `recipe: densenet121` | Yes, as code. No pretrained weights. | n/a (a library) | OK |
| E8 | timm ImageNet backbones (`pretrained=True`) | Pretrained weights, downloaded at run time | `datscan/model.py:195-208`; REFUTED.md:418-429 | No. Not in the winning recipe; `timm` is not in `requirements.txt`. | No | OK for the shipped model. Tried and rejected. |
| E9 | PyTorch cu128 wheel index | Package source | `requirements.txt:5` | Build only | n/a | OK (BSD-3) |
| E10 | torch, numpy, scipy, nibabel, pandas, monai, scikit-learn, matplotlib, pytest, joblib | Dependencies | `requirements.txt`; `provenance_ellipse/build_centers.py:2` (joblib) | Inference uses only the first five | n/a | All permissive (BSD / MIT / Apache). `joblib` is unlisted; minor. |
| E11 | **Claude Code (Anthropic API)**, including multi-agent "overnight lab" runs | Third-party hosted LLM tool | `driven_data.md:18,111-113`; REFUTED.md:715-716, 829-836, 883, 893; `export_ellipse.py:12`; `model.py:117`; `config.py:90,99,103`; ~80 "the user / user's" comments across the code | n/a | **Yes, as a tool.** Not disclosed what data it saw. | **Open question for rule A** |
| E12 | DrivenData site scraped by agents (`dd_status.py`, authenticated GET of `leaderboard_partial`) | Platform pages | REFUTED.md:829-836, 1004-1012, 1068 | Board scores guided `a`=0.78 and roster choice (`calibrate.py:9-25`) | No | Ordinary leaderboard feedback, automated. Info for DrivenData. |
| E13 | Other competitors' public GitHub repos (`ahammadshawki8/...`, `koneboi/...`), read on 2026-09-17 after the close | Public code | REFUTED.md:1035-1066 | No (read after the final submission) | No | Info only |
| E14 | Private Hugging Face repo `paulonium/dat-challenge-winner` (`prize_submission.tar.gz`, `ellipse_annotations.tar.gz`) | Competitor-hosted artifact store | README:50-63; `models/README.md:13-15`; provenance README:42-53 | Holds the weights and annotations | n/a | Not fetched. Rule E / release concern (see §6). |
| E15 | Zhou et al. 2019 6-D rotation parametrisation | Published method (no data) | `train_ellipse.py:19-21` | Method only | n/a | OK |

No other cohorts were found. Grep aliases used: `ppmi|loni|ida|ioflupane|datscan public|adni|ukb|biobank|external (dat|data|cohort)`. The "datscan" hits are the competitor's package name and the `/ssd/datasets/DAT_SCAN/` path. `harness/` contains no network calls, downloads or URLs. The only server code is the local annotation tool (§2.4).

UID check:
- **0 of 524 test UIDs** appear in any text or binary file under `3rd Place/` or `harness/`. This covers `.py`, `.md`, `.csv`, `.json`, logs, `.pyc`, `.npy`, `.pt` and `.png` byte strings.
- `meta/random50_uids.csv` contains 50 UIDs, all 50 in train.
- The example UID `7875aud9` in `visualize_dat.py` is a train UID.

---

## 2. Hand annotations (organizer ruling: not permitted)

### 2.1 Who drew them
The competitor did. The code comments are written by an AI assistant and call the human "the user". Evidence:
- `train_ellipse.py:66`: "after **the user annotated 66 more scans** -> 276 expert".
- `segment_striata_server.py:7`: "v2 (2026-08-17, **user's correction**)", and line 10: "what **the expert** implicitly did on the first 105".
- `provenance_ellipse/README.md:26-27`: "the browser tool **the expert** used to draw the striatal contours"; line 53: "the annotating itself is **human work**".
- `driven_data.md:8,14`: the competitor is a nuclear-medicine physician and this is a solo entry.
- `segment_striata_server.py:4`: `conda activate paire_training`. PAIRE is the competitor's company (`driven_data.md:9`). This hints that the annotation ran on company-provisioned compute. It is not a data issue, but worth noting.

No second annotator is named anywhere.

### 2.2 How many, which scans, and when
276 scans in total (provenance README:28-32):
- 105 "error-matched" scans, chosen from the model's own training out-of-fold errors, so the selection used labels.
- 105 controls, drawn around 2026-08-17 (`--uids meta/control105.csv`, segment server:4, 27).
- 66 more on 2026-08-22, including the 50-scan random sample (`meta/random50_uids.csv`).

All 276 are training scans:
- The tool loads `/ssd/datasets/DAT_SCAN/iso/{uid}.npy` (server:30, 41).
- `build_box_masks.py:17-18,28` indexes by `train_labels_JNDlMjr.csv`, and would fail on a non-train UID.
- `train_ellipse.py:56-63` iterates only over `train_labels`.
- There is **no evidence of any test UID being annotated**. The UID lists `control105.csv` and the error-matched list are not in the tree, so the full 276-UID list cannot be checked statically. It sits in the private `ellipse_annotations.tar.gz`, which organizers can request.

### 2.3 What was drawn
For each side, an axis-aligned ellipsoid (centre plus 3 radii), and one shared uptake threshold `cut = thr × max(in-ellipsoid maxima of both sides)` (server:7-11, 61-66). The saved mask is the ellipsoid AND the threshold (server:269-271).

### 2.4 The tool: local server, no uploads
`segment_striata_server.py` is a stdlib `ThreadingHTTPServer` bound to **`127.0.0.1`** (line 289). It is reached through an ssh tunnel (line 5).
- It renders slices to PNG with matplotlib Agg (76-98) and serves them to the annotator's own browser.
- On save it writes only local files: `meta/striatal_masks/*.npz`, `*.shared.json` and `meta/striatal_seg.csv` (271-285).
- There is no outbound request, no third-party JS or CDN (the page is inline, 106-226), and no import of `requests`, `socket` or cloud SDKs.
- `build_box_masks.py:2-3,34-36` also writes NIfTI copies "for external viewers", but only to local disk.

**Conclusion:** the tool itself does not send data anywhere. It was most likely written by the AI assistant (see the "user's correction" docstring). Nothing shows the assistant viewing the rendered slices, but nothing rules it out either.

### 2.5 Causal path to the shipped model (load-bearing)
1. The contours are drawn on the iso grid. `build_box_masks.py` maps them to box space, and a lost one-off assembles `box_automasks/` with `expert=1` on 276 scans.
2. `train_ellipse.py` (default mode, no `--auto`) trains on the expert masks only. The log shows **`train 251 scans (251 expert) | holdout 25`** (`train_ellipse.log:1-2`). The other **25** annotated random-sample scans are the holdout that picks the saved checkpoint by best holdout Dice (lines 181-185; best at epoch 70, Dice 0.836). So all 276 annotations shape the checkpoint: 251 as training targets and 25 as selection.
3. The annotations also set constants inside the localiser:
   - `LearnedCut` initialises alpha at 0.494, "the expert's implicit rule" measured on the first 105 (`train_ellipse.py:239-257`). The trained value is 0.493, but the cut head is dropped at export (`export_ellipse.py:5-10,43-45`).
   - The auto-init bounds in the tool are "expert-derived" (server:50).
4. `export_ellipse.py` traces `striatal_ellipse_sh0.5.pt`, giving **`ellipse.ts.pt`**. That file is byte-identical in `inference/assets/` and in `harness/assets/`, per the Coordinator.
5. At inference (`harness/main.py`), the ellipse output gives:
   - the canonical pose (`canonize.thetas`);
   - the striatal NDT anchor mask (`ndt_anchor=striatal`, `anchor_guard=true`);
   - the slabs.
   Every one of the 50 members sees inputs warped by it.
6. At training, `build/build_canonv2_cache.py:1-12,85,96` runs the same `ellipse.ts.pt` over all 1,362 train scans to build `canonv2.f16.npy` (the input of every member) and `canonv2mask_ell.u8.npy`. That mask is used for:
   - lesion-synthesis masks: `mask3d_cache`, applied to 25% of normals (`recipes/fusion10_s078.json:63-64`, `train_fusion10.sh:8-10`, and the `ARM` line of each `fu_*` log);
   - the striatal NDT anchor.
7. `canon_const.json` REF (64, 74, 46) is the "median striatal centroid" (`model.py:581-583`), which is also derived from localiser outputs. Its builder is absent.

**How load-bearing is this: total.** Every input the 50 members were trained and evaluated on is a function of `ellipse.ts.pt`, and `ellipse.ts.pt` exists only because of the 276 hand contours. The README says "every result in this repository was produced with the SHIPPED `ellipse.ts.pt`" (README:385-387). Removing the annotations means retraining the localiser without them (for example with an unsupervised centre finder, `datprep_iso.center_ellipse_iso`), then rebuilding the caches, retraining all 50 members and recalibrating. The competitor's own notes say the unsupervised centre finder has a p90 error of 42.1 mm (`projection.py:364-365`), so the loss in performance would probably be material.

### 2.6 Other competitor annotations (not shipped)
- The E2 auto-rule masks were **not** used by the shipped localiser.
- The E4 blinded re-reads were done by "the reader". This is almost certainly the competitor: the renders were "anonymised" and made by the competitor's own pipeline (README:521-531, `make_figures.py:289-293`). Their only downstream uses were the refuted `row_weights` arm (REFUTED.md:181) and agent analyses (REFUTED.md:719-745).
- The shipped labels equal the official `train_labels.csv` (confirmed by Reviewer 1).
- Load-bearing weight for the shipped model: **none**.

---

## 3. PPMI

**What it was used for** (REFUTED.md: 27-29, 517-547, 570, 1027):
- **Evaluation and model selection** on 3,839 PPMI scans, run on candidate members and packages as an out-of-distribution check.
- It served as a **veto** ("the 08-31 rule", line 546: "It remains a veto for catastrophic failure only").
- It briefly served as a **ranker** ("0/2 as a ranker", 545-547).
- Specific verdicts citing PPMI:
  - the striatal NDT anchor was called "OOD-negative: PPMI AUC −0.008", although it was **still adopted** in the shipped recipe;
  - the jitter bundle and smooth-gated aniso were rejected partly on PPMI;
  - the pmc/spc members were flagged positive on PPMI and then refuted on the leaderboard.
- **Design influence on shipped code:**
  - `anchor_guard=true`, which is in the shipped recipe, was built for the "PPMI failure mode" (`projection.py:384-391`);
  - the wide fallback window (`h×1.5`, "PPMI failure cases", `projection.py:366-371`).
  - Both are reported as idle on competition data: "0/400 guard fires" (`model.py:584-587`).
- **Augmentation design:** `bg_aug`, `bg_sub` and `RandBackground` were built to simulate PPMI background (`config.py:572-578`, `transforms.py:186-198`). They are **off** in the shipped recipe: the flags are absent from `fusion10_s078.json`, `train_fusion10.sh` and the `fu_*` ARM log lines.
- **Training or pretraining on PPMI:** none found.
  - No recipe or log references a PPMI cache.
  - `build_box_mmap.py` builds `ppmi.f16.npy` only as an optional branch.
  - No PPMI-derived numeric constant was identified in the shipped assets (`calibration.json`, `canon_const.json`, `tau` 4.090/1.259) or in the harness code.
  - The 50 traced weights cannot be inspected statically.

**Disclosure:** not in `driven_data.md`. README:185 and 568-569 say that "Competition rules banned external DaT data" and "No external DaT dataset was used for training". So the competitor thought external DaT data was restricted, yet held and used PPMI for selection without disclosing it.

**PPMI DUA v5.0 (April 2026), from the public page:**
- Data is released only after an application and approval through LONI IDA, electronically signed (DUA footer).
- It may be used only "for the purpose of scientific investigation, teaching, or the planning of clinical research studies".
- §2: "I will not further disclose these data beyond the uses outlined in this agreement and my data use application."
- §4 and §13: every user must register with PPMI.
- §8: a PPMI IP Policy applies.
- §11: PPMI may request derived data.
- Appendix A: "PPMI prohibits sharing of participant-level data, even if it is processed or altered … any model that has been trained using participant-level data may only be shared through MJFF-approved methods."
- §3 and Appendix A **prohibit AI tools that do not guarantee containment of data inputs**, and say "an opt-out clause does not constitute sufficient protection."
- The document never calls the data open, public or open source, and it carries no CC0 or CC-BY or other open licence.

**Conclusion:** PPMI is **not** "freely and publicly available … under a permissive open license that does not prohibit commercial use". Under the organizer ruling ("DUA-gated OK only if the DUA says the data is open"), PPMI is not permitted external data. What remains open is whether evaluation, selection and veto use with no training counts as "use" under the rules. It was also not disclosed, which the rules require regardless of eligibility.

A side issue for the competitor's own PPMI access: if the Claude Code agents ran the PPMI evaluations, that may breach PPMI DUA §3. This is not a competition rule, but it shows PPMI data may also have gone to a third-party API.

---

## 4. AI tools and third-party APIs (rule A)

**Confirmed from static evidence:**
- Claude Code was used "throughout", including "launching training runs", "model evaluation" and "creating agents" (`driven_data.md:111-113`).
- Three autonomous agents ran an overnight lab on 2026-09-15/16 (REFUTED.md:715-716). They ran analysis harnesses over per-scan OOF logits, labels, severity and "hard individuals" (`hardsep*.py`, `sev_*.py`, `mild_*.py`; REFUTED.md:719-900).
- An agent fabricated a leaderboard table (829-836).
- Agents made authenticated GETs to DrivenData (1004-1012).
- The code mentions a project-level `CLAUDE.md` with "standing rules" and export rules (`export_ellipse.py:12`, `model.py:117`, `config.py:99`, REFUTED.md:893), plus `CLAUDE_journal_2026-08-25.md` and `notes/agents/LAB.md`. **None of these is in the tree**, and there are no `.claude/`, `.codex/` or `AGENTS.md` files.
- Through Claude Code, prompts, file contents the agent reads and command outputs go to Anthropic's API. Retention (public page):
  - consumer plans: 30 days, or 5 years if training is opted in;
  - commercial/API plans: 30 days;
  - zero-data-retention only for qualified Enterprise organizations.
  Under rule A this is a tool "that stores or retains uploaded data", unless ZDR was in place.

**Not provable statically:**
- Whether raw scan voxels, per-UID labels or rendered images reached the model.
- The agents ran scripts on the training server, so any printed per-UID values (for example `meta/lomoe_severity.csv`: uid, label, severity) would have been sent.
- Tools that render competition scans exist: `visualize_dat.py`, `build/make_figures.py`, the annotation server, and "rendered and verified" (REFUTED.md:60) / `viz_feather_profile.png` (REFUTED.md:67). An agent can read PNGs, so renders may have been viewed.
- **Test data:** no test UID appears anywhere, and no agent note discusses per-test-scan content. The test set was not available to competitors anyway, beyond the code-execution runtime.

**Ask the competitor:**
1. Which Claude plan or account type, and was ZDR in place? Was training opt-in on?
2. Did any agent session read training-scan arrays, per-UID label or prediction tables, rendered scan images (including `docs/figures`, the annotation-tool PNGs, or the blinded-read renders), or PPMI data?
3. Provide `CLAUDE.md`, `CLAUDE_journal_*.md`, `notes/agents/LAB.md` and `dd_status.py`.
4. Were any other AI tools used?

---

## 5. Release compatibility
Release rule: code is public at announcement; weights are private and organizer-only.

| Item | Status |
|---|---|
| Code (MIT, `LICENSE`) | OK |
| MONAI, torch and the other dependencies | Permissive. OK. |
| timm / ImageNet weights | Not shipped. OK. |
| `inference/assets/ellipse.ts.pt` (trained on competition data and on prohibited annotations) and `canon_const.json` | **Committed in the code tree.** They must be removed before the public code release and delivered privately. |
| `docs/figures/*.png` (renders of competition training scans; `discordant.png` shows UID, label and model p per `make_figures.py:300`), `meta/*.csv` (per-train-UID labels and severity), `meta/labels.npy`, `splits.csv`, `meta/acq_meta.csv` | Competition data or derivatives in the code tree. Public release would breach "never share … the data" and rule E. **Strip before release** (owner: Release Readiness). |
| Private HF repo holding weights and `ellipse_annotations.tar.gz` | Allowed only if organizers alone can access it. Rule E (use "solely for the purpose and duration of the Competition") covers keeping competition-derived annotations after the competition. |
| PPMI | Not in the release. If any PPMI-selected artifact counts as "trained using", the DUA's MJFF-approval clause could apply. It does not apply to the current assets, because no PPMI training was found. |

---

## 6. Findings summary
- **CRITICAL (confirmed):** Competitor hand annotations (276 train scans) trained the shipped `ellipse.ts.pt`, and every input of all 50 members depends on it. They were not disclosed in `driven_data.md`. This is prohibited by organizer ruling.
- **HIGH (needs organizer ruling):** PPMI, a DUA-gated dataset that is not open, was held and used for evaluation, selection and veto, and shaped the `anchor_guard` design. It was not disclosed. No training was found.
- **MEDIUM / OPEN:** Claude Code agents with standard retention were used across evaluation and analysis. Whether competition data reached Anthropic cannot be resolved statically. The referenced agent files are missing.
- **LOW:** Competitor blinded re-reads and expert weights exist but are not shipped. Competition-derived assets and figures are committed in the code tree. `joblib` and `timm` are unlisted in `requirements.txt`.
- **INFO:** 0 test UIDs anywhere. The annotation server is local-only. The harness makes no network calls. The shipped members are from scratch.
