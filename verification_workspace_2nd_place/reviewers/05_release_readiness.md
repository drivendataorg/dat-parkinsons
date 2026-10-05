## Subagent: Reviewer 5 — Release Readiness (2nd place, NavAttack, submission 322397)

Status: needs organizer review

No competitor code executed: yes. No competitor code was run or imported, and no artifact was loaded, unpickled or decompressed. No image was opened, viewed or rendered: the five PNGs were only hashed, and the SVG was only hashed and caught by a text grep for URLs. I made no web fetches, including the Dropbox links, and wrote nothing under `2nd Place/` or `references/`.

Read-only inspection performed:
- `verification_workspace_2nd_place/notes/coordinator_context.md`, `notes/governing_rules.md` (rule E), `notes/zip_listing.txt`, `solution_overview.md` §7–§8
- `references/verification_comp_specific_rules.md` (Release expectations), `references/submissions/2nd_place.md`
- `.claude/skills/submission-verification/references/reviewers.md` §5 and `report.md`
- `2nd Place/`: `README.md`, `LICENSE`, `THIRD_PARTY_NOTICES.md`, `models/README.md`, `models/SHA256SUMS.{public,scored}.txt`, `requirements-inference{,.lock}.txt`, `requirements-train.txt`, `docs/REPRODUCIBILITY.md` §0, 1, 4 and 6, the relevant parts of `docs/PROVENANCE.md` and `docs/stages/*.md`, `docs/figures/README.md` and `make_figures.py` (data blocks and docstring), `docs/repro/mode_full_2026-09-28/**` (README, run logs, stage logs, results JSON structure), `inference/CANDIDATE.json`, `configs/feature_columns/*.header.csv`, the header of `scripts/install_weights.py`, `pipeline/02_features.sh`, `07_tabular.sh`, `train_all.sh` and `smoke_cnn.sh`, `tabular/_original/*`, and the MedicalNet section of `src/nets3.py`
- `inference/tabicl_lib.zip`: listed with `unzip -l`, and the headers of 5 members read with `unzip -p`
- Extracted harness copies (`extracted/harness_zip/`, code and json only) for UID and path greps
- Tree-wide greps for secrets and credentials, absolute paths and usernames, URL domains, and all 1,362 train and 524 test UIDs

Competition-specific checks applied:
- Release expectations (rules file lines 63-71): MIT licence. Code is published only at winner announcement. Weights trained on competition data may be held only in a private location that organizers alone can access. Winning Model Documentation Template required. BOAS indexing.
- Governing rule E: "use the Data solely for the purpose and duration of the Competition". Applied to post-competition public hosting of data-derived artifacts.
- §7 flag F11 (public Dropbox weights) and gaps G5 (template) and G7 (figures) from the solution overview.

Evidence:
- `2nd Place/README.md:131,142`: a public Dropbox link to `navattack_v17_weights.zip` with share token `rlkey=fdf3jzwx…`. The bundle holds all 60 assets (`models/README.md:17`): 34 3D CNN, 18 TorchScript 2D, linear_lin/lin4, `template_v6.npy`, the TabICL checkpoint, the recipe, `tabular_raw.pkl.xz` and `tabular_log.pkl.xz`, and `inference_config.json`.
- `README.md:152-153`: a public Dropbox link to `navattack_v17_cnn_raw_checkpoints.zip` (456 MB, 18 raw 2D state dicts) with share token `rlkey=sbggsrua…`. Neither link was fetched.
- `README.md:369-375`, `models/README.md:25-31`, `THIRD_PARTY_NOTICES.md:175`: the competitor states that the two frozen TabICL contexts (all 1,362 train scans, plus a balanced subset of 1,148) store row representations with label embeddings added, "so the training labels can be recovered from them", along with per-feature statistics. `THIRD_PARTY:175` says "A public release of these two files waits for the organisers' agreement", which contradicts the public link at `README.md:131`.
- `LICENSE` (MIT, © 2026 Navid Shabani Barzegar), `README.md:49,379-382` and `models/README.md:22`: the competitor declares the fine-tuned weights MIT-released. Per THIRD_PARTY:59, the weights bundle also carries its own LICENSE, a timm Apache-2.0 text and the TabICL licence.
- `template_v6.npy`: 442,496 B, sha256 `4d3629d5a06aadd8d638e2333f37169de5a460524be3314944553b23fb2b0672` (`SHA256SUMS.*:60`). It is the label-free mean of all 1,362 registered training crops (`REPRODUCIBILITY.md:327-329`). It is absent from the source tree (`inference/assets/` holds only `.gitkeep`) and present in the harness zip and both weights bundles. It is derived from competition data, so it falls under the weights ruling. `pipeline/train_all.sh:48-49` aborts without it, and `pipeline/02_features.sh:33-52` copies it from `inference/assets/` and checks its sha256. A rebuild command exists (`REPRODUCIBILITY.md:331-345`, `src/register3.py` on the stage-04 `crops_v3_b.npz`) but is not wired into the pipeline, and stage 02 runs before stage 04.
- Secrets grep (`token|secret|password|rlkey|aws_|AKIA|HF_TOKEN|api_key|bearer|s3://|gs://|private keys`): the only hits are the two Dropbox share tokens above and HF "set a HF_TOKEN" warnings in 8 repro logs, which carry no token value. No credentials and no private-bucket references were found.
- Path and username grep (`/home/ /Users/ /mnt/ /Volumes/ /tmp /var/folders`): the only hits are system mount strings `/System/Volumes/Data` from `df` output (`docs/repro/.../run/metadata_start.txt:11`, `run/driver.log:8`). Logs were sanitised with `<repo>`, `<original-project>`, `<run>`, `<host>`, `<release>`, `<copy>` and `<tag>`. `metadata_start.txt` also records the battery id and disk usage, which is trivial. The personal name appears only in the LICENSE copyright line. The scored (non-public) TorchScript files carry `*.debug_pkl` records that, per the competitor, hold the exporting machine's absolute paths (`README.md:147-149`, `THIRD_PARTY:57`). This was not verified because the members were not extracted. The public SHA256 list covers the scrubbed copies.
- UID grep: 0 of 1,362 train UIDs and 0 of 524 test UIDs appear in any text file of `2nd Place/` or of the extracted harness code and json (including `assets/inference_config.json`, `linear_lin*.json` and `tabular_recipe.json`). This agrees with the competitor's `run/analyze.log` ("58 log/text files scanned, 0 contain scan ids").
- `configs/feature_columns/features_v13_reg7f.header.csv` (2,632 B) and `features_v4_reg7f.header.csv` (1,035 B) hold one header line each, with column names only.
- `inference/CANDIDATE.json` holds aggregate OOF and validation metrics, byte sizes and hashes only. Its "HELD LOCALLY" status string is stale, which `README.md:364-365` acknowledges.
- `docs/repro/.../results/*.json` and `run/retrain_report.json` hold aggregate metrics, args, hashes and file lists. No list is longer than 82 entries and there are no per-UID arrays. The predictions are deliberately excluded (`docs/repro/.../README.md:55-56`).
- `docs/repro/.../stage_logs/eda_real.log` and `hdr_fingerprint.log` hold training-set aggregates: shape and spacing counts, extent quantiles, class-wise vmax/p99/vmean, header-field cardinalities and label counts (747/615). There are no per-scan rows.
- `docs/figures/` (not viewed; organizer to inspect if needed):
  - `drift.png`: 70,718 B, sha256 `3be93db31962d32977b41fc974a790876dcafdb66dab3c3c56d73605c0e74848`
  - `family_loss.png`: 202,878 B, sha256 `a1aed8c40b77c568387cb2a0e8229e6c649698eae6d031eb6b14809e57ff6533`
  - `lin_coefficients.png`: 269,463 B, sha256 `93adfedc16c5b20ac845d78b1e9b6993fb2b63adff459e1625ec33c43a721a8b`
  - `pipeline_diagram.png`: 370,529 B, sha256 `bef279099d3d57222d125340f29ab20b73fed1b591672aef5cbb98b8ff4d7984`
  - `pipeline_diagram.svg`: 32,523 B, sha256 `83b0a9b13736775ad1d4961186a8d36a0e3bde3b1815e9752b71c4890924159d`
  - `score_progression.png`: 169,372 B, sha256 `ee7b3fefdadc9be824d2960e05db83b6348bc458a001f8e77efb36e5541b8c08`
  - `make_figures.py` draws only hard-coded aggregates: public and private platform scores per submission (`:88-113`), per-inferred-acquisition-centre training OOF loss, AUROC, n and prevalence for 6 site-proxy groups (`:117-146`), linear-model coefficients read from `linear_lin*.json` (`:452-470`) and shipped constants (`:538-545`). It has no imshow, nibabel or npz reads, so the figures should contain no scan imagery.
- Requirements: all versions are pinned. `requirements-inference.txt` lists 14 top-level packages, pins scipy per architecture (1.15.3 on x86_64), and pins torch 2.12.1 with a cu129 install note. The lock file is a 32-package macOS arm64 freeze. `requirements-train.txt` is a 67-package freeze (torch 2.13.0, numpy 2.5.1, sklearn 1.9.0) that includes experiment leftovers (monai, torchio, simpleitk, surfa, scikit-image, lightgbm, matplotlib). There is no Linux or CUDA lock. Licences are inventoried in `THIRD_PARTY_NOTICES.md` §4; all are permissive (the GPL-with-GCC-exception and LGPL runtime libraries exist only inside the numpy and scipy wheels). CatBoost is imported by `tabular/_original/bench_v16.py:13` but is in neither environment.
- `inference/tabicl_lib.zip` (286,222 B, sha256 `6a57bb28…9874`) holds 91 `tabicl/*` members, version 2.2.0 (`__about__.py`), with no licence file or dist-info inside. The upstream attribution headers are retained: `_sklearn/sklearn_utils.py` ("Adapted from" scikit-learn), `forecast/_ts_dataframe.py` ("Modified from" AutoGluon), `train/_muon.py` (fabian-sp/sda and Megatron-LM), and `_sklearn/preprocessing.py` (an attribution line further down the file). `inference/tabicl_licenses/LICENSE` (sha256 `c959a1db…6874`) has BSD-3 © Soda team @ Inria plus the Apache-2.0 notice for `forecast/` (Prior Labs). The "unmodified vs the PyPI wheel RECORD" claim was not verified because no wheel was fetched (G2).
- Size: the harness zip is 1,090,109,497 B (1,138,933,628 B uncompressed). By asset type: TabICL contexts 456.7 MB, 2D TorchScript 448.3 MB, 3D CNN 122.3 MB, TabICL checkpoint 110.4 MB, other 0.73 MB. The source tree is 2.8 MB, of which about 1.08 MB is PNG and SVG figures and 286 KB is `tabicl_lib.zip`.
- Winning Model Documentation: the 2nd-place tree has no `driven_data.md` or equivalent (a repo-wide search found only `3rd Place/driven_data.md`). `README.md:40-42` says the write-up "was submitted separately to DrivenData".
- Fresh-verifier docs: README Setup, Run training and Run inference are complete and consistent. `REPRODUCIBILITY.md` covers 6 levels with an explicit list of untested cases (§6: no CUDA or CPU training, no Linux local run, no Windows, no test-scale CPU timing). The `docs/repro` run is well documented. Gaps: (a) the 150-scan local verification set and its reference predictions are not distributed (`REPRODUCIBILITY.md:32-33`), so the parity checks cannot be repeated as written; (b) the run driver, export and analysis harness behind `driver.log` and `analyze.log` is not in the repo (`docs/repro/.../README.md:49-50`); (c) the platform log of 322397 was not kept; (d) the "archived mode" of stage 07 and the provenance comparisons need the original development tree, which is not distributed (`docs/stages/07_tabular.md:274`, `docs/stages/README.md:136-141`); (e) `README.md:51` lists a `.gitignore`, but none is in the provided tree.

Findings:
- Severity: high
  Summary: The README publishes public Dropbox links to two bundles of weights trained on competition data, contrary to the organizer ruling that such weights may be held only in a private, organizer-only location until the dataset is published.
  Evidence: `README.md:131,142` (`navattack_v17_weights.zip`, `rlkey=fdf3jzwxqe13…`), `README.md:153` (`navattack_v17_cnn_raw_checkpoints.zip`, `rlkey=sbggsruaxbpy…`); `models/README.md:13-20`; `THIRD_PARTY:57`; rules file line 71; governing rule E.
  Recommendation: Ask the competitor to revoke both Dropbox shared links now (rlkey links stay live until revoked, even after the README is edited), and to hand the bundles to DrivenData privately. Before any code release, remove the URLs from `README.md:131,142,153`. Reword the weights instructions to "weights available from the organizers on request / after dataset publication", or to the scored `submission.zip` held by DrivenData.

- Severity: high
  Summary: The public weights bundle contains the frozen TabICL contexts, from which the competitor says training labels can be recovered, and per-feature training statistics. The competitor's own notice says their public release awaits organizer agreement, yet the link is already public.
  Evidence: `README.md:369-375`; `models/README.md:25-31`; `THIRD_PARTY_NOTICES.md:175`; harness `assets/tabular_raw.pkl.xz` (229,985,568 B, sha256 `167455ac…5fab`) and `assets/tabular_log.pkl.xz` (226,742,860 B, sha256 `af158889…570c`) per `inference/CANDIDATE.json`.
  Recommendation: Treat these as competition-data derivatives under rule E. They should stay organizer-only (same action as above). Organizers should decide whether they can ever be released publicly, even after dataset publication. Ask the competitor to confirm deletion of any other public copies.

- Severity: medium
  Summary: `template_v6.npy`, the mean registered image of all 1,362 training scans, is not in the code tree but ships in both the harness zip and the public weights bundle, and the documented training pipeline cannot run without it.
  Evidence: `SHA256SUMS.public.txt:60`, `SHA256SUMS.scored.txt:60` (sha256 `4d3629d5…0672`, 442,496 B); `REPRODUCIBILITY.md:327-345`; `README.md:252-259`; `pipeline/train_all.sh:48-49`; `pipeline/02_features.sh:33-52`.
  Recommendation: Keep it with the private weights. For the code-only release, ask the competitor to add a template-rebuild step to the pipeline (stage 04, then `register3.py`, then stage 02) or a documented path that does not need the bundle. They should also make `train_all.sh` and `07_tabular.sh` able to fetch the public TabICL checkpoint from HF (`jingang/TabICL` at the pinned revision) instead of requiring it in `inference/assets/`. Otherwise a public user cannot run training.

- Severity: medium
  Summary: The LICENSE, README and models/README declare the comp-trained "fine-tuned weights" released under MIT, which conflicts with the ruling that they may not be publicly released yet.
  Evidence: `README.md:49,379-382`; `models/README.md:22-23`; `THIRD_PARTY:59` (the bundle carries its own LICENSE).
  Recommendation: Reword to "the code is released under MIT; the trained weights are licensed MIT but distributed by DrivenData only once release is permitted", or similar wording as organizers prefer. The MIT licence of the code itself is compliant.

- Severity: medium
  Summary: No Winning Model Documentation Template (`driven_data.md` or equivalent) is present in the submission tree.
  Evidence: no such file in `2nd Place/`; `README.md:40-42` says the write-up was submitted separately; rules file line 68.
  Recommendation: Organizers should confirm that the completed template was received, and add it to the verification materials so Data Source can cross-check disclosures (ImageNet backbones, TabICL, no external data, AI-tool use) against the code.

- Severity: low
  Summary: The release docs contain training-set aggregates of the unpublished dataset: shape and spacing distributions, class-wise intensity stats, label counts, and per-inferred-site n, prevalence and OOF performance in `family_loss.png`. There are no per-scan values or UIDs.
  Evidence: `docs/repro/.../stage_logs/eda_real.log`, `hdr_fingerprint.log`; `docs/figures/make_figures.py:117-146`; `family_loss.png` sha256 `a1aed8c4…6533` (not viewed).
  Recommendation: Organizers should decide whether aggregate cohort statistics and site-proxy breakdowns may be published before the dataset release. If not, drop those two logs and `family_loss.png` (and its README section) from the public code release. An organizer should inspect the PNGs before release.

- Severity: low
  Summary: The weights bundle redistributes the public TabICL checkpoint (110 MB) and holds 456 MB of label-bearing contexts. The bundle totals 1.14 GB, of which only about 680 MB is the team's own trained CNN and linear weights.
  Evidence: `notes/zip_listing.txt` totals; `THIRD_PARTY:27-35`.
  Recommendation: For eventual release, fetch the TabICL checkpoint from HF at the pinned revision (BSD-3, redistributable but unnecessary to rehost), and consider shipping a recipe for regenerating the contexts instead of the contexts themselves. This is not blocking.

- Severity: low
  Summary: Some reproducibility evidence for a fresh verifier is not distributed: the 150-scan verification set and reference predictions, the run driver and analysis harness, the original development tree for archived-mode checks, and the platform log of 322397.
  Evidence: `REPRODUCIBILITY.md:32-33`; `docs/repro/.../README.md:49-56`; `docs/stages/README.md:136-141`; `REPRODUCIBILITY.md:51-55`.
  Recommendation: Not blocking. If organizers want independent reproduction, ask for the list of 150 training UIDs (privately), or have the verifier compare a rerun against the scored CSV `2nd_place_submission.csv` on the organizers' test set. Optionally ask the competitor to add the driver and analysis scripts.

- Severity: low
  Summary: Cleanup items for the release. The `tabular/_original/` scripts are non-runnable: they reference `../20260914/vendor`, `submissions/v13c_clean_2seed/submission.zip` and `outputs/site_diag_rows.csv`, and need CatBoost, which is not in either env. `nets3.py` carries dead MedicalNet code (Tencent Med3D weights under `external/`) plus the `monai` dependency. `requirements-train.txt` has unused packages. There are 17 research `DAT_*` env gates in inference. `CANDIDATE.json` has a stale status. A T = 1.0 comment does not match the shipped T = 1.4. README lists a `.gitignore` the tree lacks. The smoke script is labelled correctly (one epoch, `smoke_` tags).
  Evidence: `tabular/_original/bench_v16.py:6-13`, `common_20260914.py:17-21`; `src/nets3.py:226-307` (identical in `inference/nets3.py`); `requirements-train.txt:37,57,59,65`; `README.md:51,353-354,364-367`; `pipeline/smoke_cnn.sh:3-10`.
  Recommendation: Optionally, drop `tabular/_original/` or mark it as archival, remove the MedicalNet configs and `monai` from `src/`, and trim `requirements-train.txt`. Keep `inference/` byte-identical to the scored archive and document the leftovers there instead of editing them. Add the missing `.gitignore` or drop it from the layout.

- Severity: info
  Summary: No secrets, credentials, private buckets, test or train UIDs, or private absolute paths were found in the code-only tree. The vendored TabICL keeps its licence (as a sibling file) and the upstream attribution headers.
  Evidence: the grep results above; `inference/tabicl_licenses/LICENSE`; `THIRD_PARTY:5-25`.
  Recommendation: When redistributing, always keep `inference/tabicl_licenses/LICENSE` next to `tabicl_lib.zip`, as `THIRD_PARTY:18` already notes. Never release the scored, unscrubbed TorchScript files: per the competitor, their `debug_pkl` records hold local paths.

Open questions:
- Have the Dropbox shared links been revoked? Can the competitor confirm that no other public copies exist (GitHub release, HF, etc.)?
- Is the Winning Model Documentation Template on file with DrivenData? Can it be added to the verification inputs?
- May the cohort-level aggregates and the per-site-proxy breakdown (`eda_real.log`, `hdr_fingerprint.log`, `family_loss.png`) be published before the dataset release?
- After dataset publication, may the label-recoverable TabICL contexts and `template_v6.npy` be released, or should they stay organizer-only permanently?
- Does BOAS indexing need anything beyond the MIT-licensed code (for example a model card)? The competitor's README and docs are otherwise adequate.

---

### Appendix: working log (chronological)
- Read coordinator_context.md, verification_comp_specific_rules.md (Release expectations), governing_rules.md (rule E), solution_overview.md §7/§8, reviewers.md §5, report.md.
- Inventory: `2nd Place/` = 177 files, 2.8 MB on disk. No `driven_data.md` / Winning Model Documentation file in tree (to confirm by grep).
- README.md:131,142 — public Dropbox link `navattack_v17_weights.zip` (rlkey=fdf3jzwx…, 60 assets incl. tabular_raw/log.pkl.xz and template_v6.npy). README.md:152-153 — public Dropbox link `navattack_v17_cnn_raw_checkpoints.zip` (rlkey=sbggsrua…, 18 raw 2D state dicts). Not fetched.
- README.md:369-375, models/README.md:25-31, THIRD_PARTY:175 — competitor states the TabICL contexts let training labels be recovered. THIRD_PARTY:175 says "A public release of these two files waits for the organisers' agreement", yet README:131 already points to a public bundle containing them (models/README.md:17 lists contents). Internal contradiction.
- models/SHA256SUMS.{public,scored}.txt:60 — template_v6.npy (sha256 4d3629d5…0672) is part of both weights bundles; README:252-256 calls it a "released file" (mean of registered training crops) -> comp-data-derived, falls under the weights ruling.
- LICENSE: MIT, Copyright 2026 Navid Shabani Barzegar. Points to THIRD_PARTY for tabicl (BSD-3).
- README:51 lists `.gitignore` in layout, but no .gitignore is in the provided tree (find list). Minor doc mismatch.
- Secrets grep (token|secret|password|rlkey|AWS|AKIA|HF_TOKEN|api_key|bearer|s3://|gs://|ssh keys): only hits are the 2 Dropbox URLs with `rlkey=` share tokens (README:131,142,153) and HF "unauthenticated requests ... set a HF_TOKEN" warnings in repro logs (no token value). No credentials found.
- Private-path grep (/home/, /Users/, /mnt/, /Volumes/, /tmp, /var/folders): only `/System/Volumes/Data` df output in run/metadata_start.txt:11 and run/driver.log:8 (system mount, harmless). Logs sanitised with placeholders `<repo>`(11) `<original-project>`(3) `<run>`(3) `<host>` `<release>` `<copy>` `<tag>`. metadata_start.txt also records battery id / disk usage (trivial). Personal name only in LICENSE:3 copyright (expected).
- UID grep: 0 of 1,362 train uids and 0 of 524 test uids appear in any text file of `2nd Place/` or the extracted harness code/json (incl. assets/*.json). Competitor's own analyze.log: "58 log/text files scanned, 0 contain scan ids".
- configs/feature_columns/*.header.csv: one header line each (column names only, 2,632 B and 1,035 B). No data rows.
- inference/CANDIDATE.json: aggregate OOF/validation metrics, hashes, byte sizes; no per-scan values. Status string "HELD LOCALLY" is stale (README:364-365 acknowledges).
- docs/repro results/*.json and run/retrain_report.json: aggregate metrics, args, hashes, file lists only (no list >82 entries; no per-uid arrays).
- docs/repro stage_logs/eda_real.log, hdr_fingerprint.log: aggregate distributions of the training set (shape/spacing counts, extent quantiles, class-wise vmax/p99/vmean, header-field uniqueness counts, label prevalence 747/615). No per-scan rows. Low sensitivity; dataset not yet public.
- docs/figures (NOT VIEWED; organizer to inspect if needed):
  - drift.png 70,718 B sha256 3be93db3…4848
  - family_loss.png 202,878 B sha256 a1aed8c4…6533
  - lin_coefficients.png 269,463 B sha256 93adfedc…8a8b
  - pipeline_diagram.png 370,529 B sha256 bef27909…7984
  - pipeline_diagram.svg 32,523 B sha256 83b0a9b1…159d
  - score_progression.png 169,372 B sha256 ee7b3fef…8c08
  - make_figures.py (read): hard-coded aggregates only — platform public/private scores per submission (DATA 1), per-inferred-acquisition-centre OOF loss/AUROC/n/prevalence for 6 training groups (DATA 2, make_figures.py:117-146), linear coefficients from linear_lin*.json, shipped constants. No imshow/nibabel/npz reads, so no scan imagery. family_loss.png therefore discloses per-site-proxy aggregates of the unpublished training set (n, prevalence, intensity scale, geometry) — low.
- Requirements: requirements-inference.txt pins 14 top-level pkgs (torch 2.12.1, numpy 2.2.6, scipy 1.15.3 x86_64 / 1.17.1 other, sklearn 1.8.0, pandas 3.0.3, ...). Lock file = 32-pkg macOS arm64 freeze (scipy 1.17.1). requirements-train.txt = 67-pkg freeze (torch 2.13.0, numpy 2.5.1, sklearn 1.9.0) incl. monai, torchio, simpleitk, surfa, lightgbm, matplotlib (experiment leftovers; THIRD_PARTY marks most "not imported by repository code"). No CUDA lock; Linux untested locally (REPRODUCIBILITY §6). All pinned; licences inventoried in THIRD_PARTY §4 (all permissive; GPL-with-GCC-exception/LGPL runtime libs inside numpy/scipy wheels only).
- Template dependency: pipeline/train_all.sh:48-49 aborts unless inference/assets/template_v6.npy and the TabICL ckpt exist (from the weights bundle); 02_features.sh:33-52 copies the template from inference/assets and sha256-checks it. template_v6.npy = label-free mean of all 1,362 registered training crops (REPRODUCIBILITY.md §4), 442,496 B, sha256 4d3629d5…0672, in both weights bundles (SHA256SUMS.*:60). Rebuild command documented (REPRODUCIBILITY §4: `cd src && python register3.py --template ... --out ...` on stage-04 crops_v3_b.npz) but not wired into train_all.sh. => if the weights bundle is withdrawn from public hosting, a code-only user cannot run training as documented without (a) a template-rebuild step and stage reorder (04 before 02), and (b) fetching the TabICL ckpt from HF directly.
- Repro gaps for a fresh verifier: the 150-scan "local verification set" ids and reference predictions are not distributed (REPRODUCIBILITY §1:32-33); the run driver/export/analysis harness that produced docs/repro (driver.log, analyze.log) is "kept with the run and not part of this repository" (docs/repro/.../README.md). Training only ever run on MPS; CUDA training, Linux, CPU training untested (REPRODUCIBILITY §6). Platform log of 322397 not kept.
