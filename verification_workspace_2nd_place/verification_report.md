# Verification report — 2nd place, DaT Parkinson's Challenge (DrivenData comp 311)

**Recommendation: `needs organizer review`.** No evidence of cheating, test-label leakage, or prohibited data use, and the submitted code matches the scored submission. Three items need an organizer decision or follow-up: (1) the competition-trained weights, including label-bearing TabICL contexts, are linked publicly on Dropbox; (2) a ruling on ImageNet-1k pretraining behind Apache-2.0 timm weights; (3) a competitor clarification on undisclosed outside review / AI-tool use ("external review member E"). If (1) is remediated and (2) is ruled permissive, the recommendation becomes `pass with notes`.

Coordinator: Claude (submission-verification skill), 2026-10-02. Static review only.

## 1. Submission

| Item | Value |
|---|---|
| Team / username | NavAttack (single author per `2nd Place/LICENSE`) |
| Platform submission | 322397 (per `2nd Place/README.md`) |
| Official private leaderboard score (intake) | 0.2731 |
| Coordinator recomputation from scored CSV (static-review observation) | full test 0.273088 (n=524); public subset 0.240858 (n=262); private-only subset 0.305317 (n=262) |
| Source tree | `2nd Place/` (177 files, hashes in `hashes/source_tree.sha256`) |
| Harness zip | `references/submissions/comp_subs/2nd_place/2nd_place_submission.zip` sha256 `88b5611e25170d88c2fbf21b62443dbc67b722c519b5a0391a60aab62bbec7fd` (82 members, hashes in `hashes/harness_zip_members.sha256`) |
| Scored CSV | `references/submissions/comp_subs/2nd_place/2nd_place_submission.csv` sha256 `293d7041037dd6e38b55de53efbf5838df44e7de1f0a007257ca50fd5ef8e37a` |

**Intake correction:** `references/submissions/2nd_place.md` lists the CSV and harness zip as `comp_subs/3rd_place/3rd_place_submission.csv` (copy-paste error). The correct 2nd-place files above were used. The zip's sha256 matches the hash the competitor quotes for submission 322397 in `README.md`.

## 2. Materials reviewed

**Required (all available):** competition homepage, rules, problem/data and code-format pages (text saved in `official_pages/`, reused from the 3rd-place verification, fetched 2026-09-29, hashes in `hashes/official_pages.sha256`); `references/verification_comp_specific_rules.md` (organizer-filled); `references/competition_materials.md`; submitted source tree, README, `docs/`, `THIRD_PARTY_NOTICES.md`, `models/README.md`, requirements files.

**Optional, available:** harness zip from platform, scored CSV, official `submission_format.csv` (uid order), `test_labels.csv` and `pub_sub_idx.csv` (used by the Coordinator only for score recomputation; reviewers did not open labels), train labels (uid list only).

**Optional, unavailable, and what it limited:**
- Winning Model Documentation Template (`driven_data.md` or equivalent) is not in the tree (README says it was submitted separately). This limited the Data Source disclosure cross-check, which was done against README/THIRD_PARTY_NOTICES/PROVENANCE instead.
- Platform run log for submission 322397. Without it, block-level fallbacks can't be confirmed absent from logs; the CSV, though, shows none (§5).
- The competitor's development notes (`RESEARCH_NOTES`, `HANDOFF`), the 150-scan local verification set and the driver harness are not distributed. This limited provenance of hand-set constants (TabICL blend 0.35 / T 1.2, 2D stack ratios).
- Model weights: available inside the harness zip; not loaded (static review). The public Dropbox bundles were not fetched.
- Organizer clarifications: none, per the rules file.

## 3. Governing rules applied

From `verification_comp_specific_rules.md` plus extra verbatim rules A–F in `notes/governing_rules.md`:
- **Forbidden data:** any external data not freely and publicly available under a permissive, commercial-use license. No NC/CC-NC/CC-BY-NC data or models. Pretraining data is subject to the same rules. No public copy of competition data exists. DUA-gated data is OK only if the DUA says the data is open.
- **Allowed data:** competition training niftis + `train_labels.csv`; permissive external data/models, all disclosed to organizers. No organizer-approved extra datasets.
- **Added checks:** independent per-case test processing, with no pseudo-labeling or unsupervised learning on test (D); no logging of test info (C); harness layout and offline execution; CSV in [0,1]; no AI tools that retain data (A); annotations on training data only and delivered (F); privacy (no re-identification); data used only for the competition (E); MIT license with OSS dependencies; release is code only at announcement, with comp-trained weights held privately with organizer-only access.
- **Confirmed by DrivenData (not re-checked):** winner eligibility and identity, submission caps and multi-account activity.

## 4. Scope

Static, read-only review. **No competitor code was executed** by the Coordinator or any reviewer. No competitor module was imported. No artifact was loaded, unpickled or decompressed. `tabicl_lib.zip` was read via `unzip -l`/`-p` only. JSON assets were parsed as data with stdlib `json`. **No images were viewed** (organizer restriction). Reviewer 4 made web fetches only to public third-party docs (TabICL HF/arXiv/GitHub, timm HF card, image-net.org). No competitor-hosted URL was fetched. Nothing under `2nd Place/` or `references/` was modified.

Dynamic checks were **not attempted** and remain for a human: environment setup, inference rerun and score comparison, training rerun.

Solution overview: `solution_overview.md`. Reviewer handoffs are in `reviewers/01_…05_*.md`.

## 5. Code integrity findings

- **Code matches the leaderboard submission (confirmed).** All 22 code/doc members of the harness zip are byte-identical to `2nd Place/inference/`, including `main.py` at the zip root, `inference_main.py` and `tabicl_lib.zip`. All 60 asset hashes in the zip equal `2nd Place/models/SHA256SUMS.scored.txt`. The `SHA256SUMS.public.txt` list differs only for the 18 TorchScript files, which the competitor says were scrubbed of `debug_pkl` path records.
- **Scored CSV consistent with code (confirmed).** 524 rows, uid order identical to `submission_format.csv`, 0 duplicates. Range [0.005, 0.995], with 115 rows at 0.995 and 4 at 0.005, which is the clip from `assets/inference_config.json:502` applied at `inference_main.py:253,475-477`. There are 407 distinct full-precision values. 0 rows sit at the governor prior 0.5484581… (`inference_config.json:794`), so no registration rejection or failure fallback fired in the scored run. Recomputed log loss matches the official score. Notes: `notes/prediction_csv_check.txt`.
- **No test data in training (confirmed).** 0 of 524 test uids anywhere in the tree, the extracted harness, or the text of `tabicl_lib.zip`. Training enumerates scans from `labels.csv` (`src/folds.py:38`, `src/cache3.py:52`), and all repro stage logs report 1,362 scans. No pseudo-labeling. No reads of `submission_format`/`/code_execution` outside the inference entrypoint. `template_v6.npy` is built from training crops only (`src/register3.py:170-259`).
- **Independent per-case processing (pass with notes).** Inference runs in blocks of 128, but every batched op is row-independent: per-row NCC registration loss (`register3.py:102-167`, `regfit.py:62-94`), frozen scalers and constants, GroupNorm 3D nets, TorchScript eval-mode 2D nets, and within-image flip TTA. TabICL `forward_with_cache` attends only to training positions. The Coordinator verified `tabicl/_model/layers.py:432-437`: `k = v = q[..., :train_size, :]`. Residual issues:
  - (low) Block-level exception fallbacks (`inference_main.py:378-386,438-462`): a registration or CNN exception would degrade all cases in that block. It is a failure path only, not information leakage, and the CSV shows it did not fire.
  - (info) Batch-composition float noise of about 1e-5, which the competitor discloses.
- **No test logging (confirmed).** All logger calls are static strings (`inference_main.py`, `main.py` prints only exception type and location).
- **No hard-coded predictions or uid-keyed values (confirmed)** in `inference_config.json`, `tabular_recipe.json` or the `linear_lin*.json` files.
- **Leaderboard-informed tuning (low, note).** Temperature T=1.4 and clip 0.005 were chosen "from the board T-curve" (`src/fixstack.py:3-5`; Coordinator verified). `head_ml` was withheld because "the board refuted it in v9" (`src/train_final3.py:57-60`). The TabICL blend 0.35 / T 1.2 (`tabular/prepare_tabular.py:162`) have undocumented provenance. These are a few global hyperparameters across 9 scored submissions, with no per-uid or per-site constants, so it is not label probing and no rule bans it. The docs say these were "chosen by hand" (`docs/stages/08-09_stack_and_package.md:23-24`), which understates the leaderboard's role.

## 6. Data source, provenance and leakage findings

- **External data: none (confirmed).** 0 hits for PPMI/LONI/ida.loni/DaTscan/ioflupane or other public-cohort aliases. A 150-scan "local verification set" (`run/driver.log:1`) is used only for parity checks. Its log loss of 0.098 is at in-sample level, which suggests a training subset. Ask the competitor to confirm.
- **Pretrained models:**
  - TabICLv2 checkpoint `tabicl-classifier-v2-20260212.ckpt`: BSD-3-Clause, synthetic-prior pretraining per public docs. Compliant. Hash matches across the HF reference, `SHA256SUMS.scored.txt` and the harness.
  - **timm `efficientnet_b0.ra_in1k`/`efficientnet_b3.ra2_in1k`: Apache-2.0 weights pretrained on ImageNet-1k, whose terms are non-commercial research/education.** These are the starting point for all 18 shipped 2D members (`src/cnn2d.py:203-215`, `tools/fetch_backbones.py`). The competitor disclosed this (`THIRD_PARTY_NOTICES.md:48-61`). **Organizer ruling needed (medium, policy).** Read literally ("pretraining data … subject to the same rules"), these members are non-compliant. Under the common reading (the weight license governs; ImageNet terms bind those who download the database), they pass. If the strict reading applies, the 2D members (stack ratio 0.6 of 2.7) would have to be removed or retrained with `--scratch`, and the scored submission would no longer represent the eligible solution. The 1st- and 3rd-place reviews should get the same ruling. 3rd place shipped no pretrained weights.
  - MedicalNet appears only in unused architectures, and its weights are absent. Shipped 3D models are trained from scratch (`pipeline/05_cnn3d.sh:69`).
- **Dependencies:** all OSS with commercial use allowed, and disclosed. Proprietary NVIDIA CUDA libs are an accelerator only; an all-OSS CPU path exists (`inference_main.py:67,292`). Minor gap: the `rope.py` (MIT) credit is missing from the notices.
- **Network / third-party APIs (confirmed none for data).** Downloads happen only at training/setup: HF Hub timm weights, and an optional fetch of the competitor's own bundle (`scripts/install_weights.py:106-116`). Inference is offline (`tabular_adapter.py:5-6`, `allow_auto_download=False`). No upload code, cloud SDKs, trackers or LLM clients.
- **AI tools / outside help (rule A, low, open question).** There is no disclosure and no agent config files. `src/cnn2d.py:72` (identical in `inference/`) credits an idea to "external review member E". Comments also cite undistributed "HANDOFF" docs and "development session logs" (`src/roigeom.py:7`, `tabular/derive_groups.py:4-5`). These are consistent with either human reviewers or an AI multi-agent workflow. The credited idea is in an unused code path (`ax3` view). Ask the competitor what "E" is, which AI tools were used, and whether any competition data reached a service that retains data.
- **Annotations (rule F): pass.** The acquisition groups (`tabular/derive_groups.py:76-95`) are automatic, rule-based metadata on training scans only, used only in training, with code shipped. They are not manual annotations, and rule F is satisfied either way.
- **Privacy: pass.** `tools/hdr_fingerprint.py:11-28` reads free-text NIfTI header fields on training data only. All are empty per the shipped log. Downstream it uses only geometry, and inference reads only spacing and affine (`preprocessing3.py:76-87`). No re-identification.

## 7. Release readiness

Release expectations were provided: MIT, code only at announcement, comp-trained weights private and organizer-only, BOAS.

- **Blocking (high): public Dropbox links to comp-trained weights.** `README.md:131,142,153` links `navattack_v17_weights.zip` (all 60 assets) and `navattack_v17_cnn_raw_checkpoints.zip` via `rlkey` share links (Coordinator verified). The bundle includes the TabICL contexts `tabular_raw.pkl.xz`/`tabular_log.pkl.xz`, from which the competitor says training labels are recoverable (`models/README.md:25-31`, `README.md:369-375`). It also includes `template_v6.npy`, the mean training image. This conflicts with the organizer-only ruling and with rule E. The competitor's own `THIRD_PARTY_NOTICES.md:175` says their release "waits for the organisers' agreement". **Action:** have the competitor revoke both links now, hand the bundles to DrivenData privately, confirm no other public copies exist, and remove the URLs from the README before code release. Link access level not verified (not fetched).
- **Medium:** `LICENSE`, `README.md:49,379-382` and `models/README.md:22` declare the weights MIT-released. Reword so the MIT license covers the code only.
- **Medium:** `template_v6.npy` is needed by the training pipeline (`train_all.sh:48-49`, `02_features.sh:33-52`) but should stay private. For the code-only release, wire the documented rebuild (`src/register3.py`) into the pipeline.
- **Medium:** Winning Model Documentation is not in the tree. The organizer should confirm it was received.
- **Low:** release docs carry training-cohort aggregates and per-inferred-site breakdowns (`eda_real.log`, `hdr_fingerprint.log`, `make_figures.py:117-146`). They have no per-scan values or uids. The organizer should decide whether to publish them before the dataset release. PNGs in `docs/figures/` were **not viewed; organizer to inspect** before release. `make_figures.py` plots only hard-coded aggregates and no scan imagery.
- **Clean (confirmed by grep):** no secrets beyond the Dropbox share tokens, no private paths (logs sanitized), no train or test uids in any text file.
- **Non-blocking cleanup:**
  - Remove `tabular/_original/` (archival, not runnable, CatBoost).
  - Remove the dead MedicalNet code and the unused training requirements.
  - Fix the stale `CANDIDATE.json` status and the T=1.0 comment.
  - Add the missing `.gitignore` the README lists.
  - Add the `rope.py` credit.
  - Add a Linux/CUDA lock file.
  - Keep `inference/` byte-identical to the scored archive.

## 8. Checklist coverage

| Checklist item | Result | Where |
|---|---|---|
| Obvious disqualifiers | None found | §5, §6, `reviewers/02` |
| Non-OSI models/code | Pass; ImageNet pretraining-data terms pending ruling | §6 |
| Clear cheating | None | §5 |
| No test data in training | Pass | §5 |
| Independent test processing | Pass with notes (failure-path block fallback) | §5 |
| No prohibited data sources | Pass (pending ImageNet ruling) | §6 |
| Submission limit / multiple accounts | Checked by DrivenData, no issues | §3 |
| Code matches leaderboard submission | Pass (byte-identical code, hash-matched assets, score recomputed) | §5 |
| Competition-specific restrictions | Applied; rule E/release issue open, rule A question open | §6, §7 |
| External models trained on prohibited source | ImageNet-1k question (Q1) | §6 |
| External data/model preventing broad release | ImageNet question only; TabICL and deps OK | §6 |
| Competition data to third-party API | None found; AI-tool use undisclosed, question open | §6 |
| Release tweaks | Listed | §7 |
| Dynamic: env setup, inference rerun, training rerun | **Not attempted** (out of scope) | §4 |

## 9. Issues summary

**Blocking (must be resolved before award/release):**
1. Public Dropbox links to comp-trained weights and label-recoverable TabICL contexts (§7). The competitor must revoke them and move the files to organizer-only storage.

**Needs organizer ruling:**
2. Is ImageNet-1k pretraining behind Apache-2.0 timm weights acceptable under "pretraining data subject to the same rules"? (§6) Apply the same ruling to all winners.

**Non-blocking issues:**
- Leaderboard-informed choice of T, clip and `head_ml` exclusion, which the docs present as "chosen by hand" (§5).
- Block-level exception fallback creates a failure-path cross-case dependency. It did not fire in the scored run (§5).
- The weights' MIT wording, the `template_v6.npy` dependency, the missing Winning Model Documentation, cohort aggregates in docs, and the cleanup items (§7).

**Open questions for the competitor:**
- What is "external review member E"? Which AI tools were used, and did any competition data reach a data-retaining service?
- Is the 150-scan verification set made only of training scans?
- How were the TabICL blend 0.35 / T 1.2 and the 2D stack ratios chosen?
- Have the Dropbox links been revoked, and are there no other public copies?

**Open questions for the organizer:**
- The ImageNet ruling (above).
- May cohort aggregates and per-site breakdowns be published before the dataset release?
- May the TabICL contexts or `template_v6.npy` ever be released?
- Is the Winning Model Documentation on file?
- Is the platform log for submission 322397 retained?

## 10. Final recommendation

**`needs organizer review`**
