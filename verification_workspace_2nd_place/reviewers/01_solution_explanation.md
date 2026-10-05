## Subagent: Reviewer 1 — Solution Explanation (2nd place, NavAttack)

Status: needs organizer review. The solution itself is fully explainable from the submitted code. Organizer review is needed for the policy items F9, F10 and F11 and for question Q3.

No competitor code executed: yes. No competitor script, tool, notebook or module was run or imported. No artifact was loaded or unpickled (.pt, .ts.pt, .pkl.xz, .ckpt, .npy). No image was opened or viewed, including `docs/figures/*`. No web fetch was made. Nothing under `2nd Place/` or `references/` was modified. The only inspection actions were cat/sed/grep/diff/cmp/`unzip -l`/`unzip -p` on text, `python3 -c json.load` on the extracted harness JSON configs (data files, not code), and a grep of the tree against the train and test uid lists.

Read-only inspection performed:
- `notes/coordinator_context.md`, `references/verification_comp_specific_rules.md`, `notes/governing_rules.md`, the skill's `reviewers.md` §1 and `report.md`.
- `2nd Place/`: README.md, THIRD_PARTY_NOTICES.md, LICENSE, models/README.md, docs/PROVENANCE.md, docs/REPRODUCIBILITY.md, docs/stages/08-09 (parts), docs/figures/README.md, docs/figures/make_figures.py (as code), and docs/repro logs (grep plus selected files).
- All of `inference/*.py` and `inference/TABICL_MODEL_CARD.md`.
- `src/`: folds, cache3, make_regfeat32, filter_feature_tables, linear_oof, train_final3, fixstack, avg_member, avg_2d_seeds, build3, package_submission, check_package, eda_real, and a diff of main3 against inference_main. `cmp` showed the 13 shared modules are identical.
- All of `tabular/*.py` and `tabular/_original/*.py`.
- `tools/`: hdr_fingerprint, ts_export, fetch_backbones, scrub_torchscript_debug, plus headers and greps of replay_cnn_oof and verify_cnn.
- `scripts/*`, `pipeline/*.sh` (00-03 and 05 in full; key commands of 04 and 06-09), `configs/`.
- The tabicl classifier and embedding sections inside `tabicl_lib.zip`, read via `unzip -p`.
- Harness copies `extracted/harness_zip/assets/inference_config.json`, `tabular_recipe.json` and `CANDIDATE.json`.

Competition-specific checks applied:
- All RF checks plus GR A–F were converted to the owned checks CS1–CS17 in `solution_overview.md` §1. The dataset policy is in §2.

Evidence:
- `verification_workspace_2nd_place/solution_overview.md` holds the full map: pipeline (§3), components (§4), file tables (§5), claims vs code (§6), flags (§7) and gaps (§8).
- Inference flow, `inference/inference_main.py:229-481`:
  - Blocks of 128 scans. Per-scan `preprocess_full` builds the head mask, reference mode, striatal frame and 48³ SBR crops.
  - Batched rigid registration to the frozen `template_v6.npy`.
  - Per-scan features13 plus 180 scale-space and 60 mixture features.
  - Members: lin and lin4 (L1 logistic), mil48y (3D HemiMIL, GroupNorm, 34 ckpts, no TTA), e2dmipf2 (12 TorchScript B0) and e2db3f2 (6 B3), with in-image flip TTA for the 2D models.
  - Fixed-ratio stack 0.531×[1, .4, .7, .3, .3] + 0.184, then T = 1.4.
  - Blend: 0.65·stack + 0.35·TabICL/1.2.
  - Rejected registration or non-finite output gets the prior 0.5485. Output is clipped to [0.005, 0.995].
- `template_v6.npy` is built by `src/register3.py:170-259` from the training-only `crops_v3_b.npz`. It uses log1p, a mean, then 3 register-to-mean passes, and uses no labels.
- TabICL block (`tabular/prepare_tabular.py`, `freeze_tabular.py`):
  - 64 L1-selected features, 4 per-scan acquisition descriptors and 4 linear prior scores. The prior scores are leave-acquisition-family-out for context rows and full-fit for query rows.
  - Two contexts: raw (1,362 rows) and log (1,148 acquisition-balanced rows).
  - The freeze asserts that training X/y are not saved and stores only the KV cache (`freeze_tabular.py:91-93`). Per the competitor, the labels are recoverable from that cache.
- `tabular/_original/` holds 6 reference scripts that nothing imports. Their logic does not differ materially from the cleaned ports. `mixture_features.py` is a verbatim slice of the original `mixture_shape.py`.
- No test UIDs (0/524) and no training UIDs (0/1,362) appear anywhere in `2nd Place/` or the extracted harness. The grep used `submission_format.csv` and `train_labels.csv` uid lists only. `test_labels.csv` was not opened.
- `docs/figures/make_figures.py` plots only typed constants: platform public/private scores for 9 submissions, training-OOF aggregates per inferred site, and `linear_lin.json` coefficients. It reads no scan or test data. `drift.png` is private − public per submission. The images themselves were not viewed; organizer to inspect if needed.

Findings:
- Severity: high
  Summary: The README links public Dropbox bundles of comp-trained weights, including TabICL contexts from which training labels are recoverable. This conflicts with the organizer ruling that weights may be held only privately with organizer-only access.
  Evidence: README.md:131,142,153,369-375; models/README.md:25-31; THIRD_PARTY_NOTICES.md:175
  Recommendation: Release Readiness to require the links be removed or made private, and the code-only release to omit them (GR E).
- Severity: medium
  Summary: The 2D members are fine-tuned from ImageNet-1k-pretrained timm weights. The weights are Apache-2.0, but ImageNet's own terms are non-commercial, and the rules file says pretraining data is subject to the same rules.
  Evidence: THIRD_PARTY_NOTICES.md:48-61; src/cnn2d.py:203-215,246; tools/fetch_backbones.py
  Recommendation: Organizer ruling (Q1). Data Source/Eligibility to apply it.
- Severity: medium
  Summary: Code comments show that public-leaderboard feedback guided the calibration temperature and clip (and feature inclusion). The stage doc says these were "chosen by hand". This is not test-label leakage, but it touches the "do not overfit to the test set" spirit and the claim is inconsistent with the code.
  Evidence: src/fixstack.py:4-5; src/train_final3.py:57-60; docs/stages/08-09_stack_and_package.md:23-24; docs/figures/README.md:22-32 (T=1.2 then 1.4 submissions)
  Recommendation: Organizer judgment. Code Integrity to note it. Likely acceptable as standard LB-informed tuning.
- Severity: low
  Summary: Inference runs in blocks of 128. Registration, the CNNs and TabICL are batched, and the error fallbacks (registration or CNN exception) apply to the whole block. On the normal path the math is per-sample (per-row NCC loss, GroupNorm, BN frozen at TorchScript export, frozen FeatPrep/mu/sd, TabICL KV cache against a train-only context), so only floating-point order couples scans.
  Evidence: inference_main.py:49,334,378-386,438-462; regfit.py:68-94; register3.py:102-122; tools/ts_export.py:22,26; tabular_adapter.py:53-63
  Recommendation: Code Integrity to confirm the tabicl `forward_with_cache` path has no test-to-test attention (F3) and to rule on the block-level fallback (F1).
- Severity: low
  Summary: `tools/hdr_fingerprint.py` reads free-text NIfTI header fields and fingerprints site/scanner. The fields are empty for all training scans per the shipped log, it runs on training only, and only `pixdim3` feeds the acquisition groups. No linking to individuals was found.
  Evidence: tools/hdr_fingerprint.py:11-39; docs/repro/mode_full_2026-09-28/stage_logs/hdr_fingerprint.log; tabular/derive_groups.py:76-95
  Recommendation: Data Source to record it as a privacy pass with a note.
- Severity: low
  Summary: Network use is limited to training and setup (the HF Hub for public timm backbones, and an optional Dropbox download of the competitor's own bundle). There is none in `inference/` (static), and no upload of competition data was found.
  Evidence: tools/fetch_backbones.py; src/cnn2d.py:209; scripts/install_weights.py:106-116; tabular_adapter.py:5-6; HF warnings in docs/repro stage_logs
  Recommendation: Code Integrity / Data Source to note it. No action.
- Severity: info
  Summary: No external datasets (no PPMI/LONI/DaTscan cohort references). MedicalNet code exists but is unused by the shipped model. No AI-tool disclosure or mention, apart from an ambiguous "external review member E" comment.
  Evidence: grep across the tree; src/nets3.py:225-323; src/cnn2d.py:72
  Recommendation: Data Source to ask the competitor about AI-tool use if required (GR A).

Open questions:
- Q1: Is ImageNet-1k pretraining (behind Apache-2.0 timm weights) acceptable under the "pretraining data subject to the same rules" clause?
- Q2: TabICLv2 checkpoint pretraining data (synthetic priors per its papers). Data Source to confirm from the model card or paper without fetching competitor URLs.
- Q3: Do automatically derived acquisition groups (header/geometry/intensity-based `fam2` and `scale`, training-only) count as "annotations" under rule F?
- Q4: Where do the hard-coded TabICL blend weight 0.35 / T 1.2 and the member ratios come from? They are not in code (development notes are not distributed).
- Q5: The platform job log of 322397 was not kept, so whether all CNN members ran on the A100 without a fallback cannot be confirmed statically. The recomputed CSV score does match the leaderboard.
