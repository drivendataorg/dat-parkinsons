# Reviewer 4 — Data Source and Model Provenance (2nd place, NavAttack)

Working notes (appended as review progresses; final handoff at the bottom).

## Scope / method
- Static, read-only. Grep + reading text files in `2nd Place/` and `verification_workspace_2nd_place/extracted/harness_zip/`. `tabicl_lib.zip` read via `unzip -l` / `unzip -p` text members only.
- No competitor code executed; no artifacts loaded; no images viewed; no competitor-hosted URLs fetched.

## Notes

### N1. Forbidden/allowed name grep (2nd Place/, excl. images)
- Pattern: `ppmi|loni|datscan|ioflupane|ida\.|medicalnet|med3d|imagenet|kaggle|zenodo|physionet|openneuro|figshare|tcia|adni|biobank`.
- 0 hits for ppmi, loni, ida.loni, datscan, ioflupane, adni, tcia, openneuro, kaggle, zenodo, physionet, figshare, biobank.
- Hits only for ImageNet (timm backbones; README:28,249,382; THIRD_PARTY_NOTICES.md:46-61; tools/fetch_backbones.py; pipeline/06_cnn2d.sh; docs) and MedicalNet/Med3D (`src/nets3.py:225-308`, identical in `inference/nets3.py`; THIRD_PARTY_NOTICES.md:109).

### N2. Disclosure documents read
- THIRD_PARTY_NOTICES.md (full): §1 tabicl 2.2.0 (BSD-3 + Apache-2.0 forecast subdir), §2 TabICLv2 ckpt (HF jingang/TabICL rev 4dcd344e…, sha256 bdc7dbd5…, BSD-3), §3 timm B0/B3 (Apache-2.0, ImageNet-1k pretraining; :61 states ImageNet terms are non-commercial), §4 package table, §5 competition data (frozen TabICL contexts carry recoverable labels; "public release ... waits for the organisers' agreement").
- README:131,153 — public Dropbox links (weights bundle; raw CNN checkpoints). README:249-251 "Nothing else needs a network". README:369-375 label-recoverability statement.
- models/README.md:17-31 — bundle contents incl. both frozen contexts; label-recoverability.
- MedicalNet: NOT disclosed as a used source (correctly, it is unused), only listed via monai row "MedicalNet experiment only".

### N3. 150-scan "local verification set" (`<original-project>/data-verify`, driver.log:1)
- Used for parity checks only (not training). Labelled (log loss 0.09811, AUROC 0.99873, analyze.log:13). Not distributed; origin not stated explicitly anywhere (REPRODUCIBILITY.md:30-31, stages/README.md:138-140). In-sample-looking LL (0.098 vs OOF ~0.126) is consistent with a subset of the 1,362 training scans scored by full-data models. Competitor had no test labels, so it cannot be test data with labels. Open question (low): confirm it is a subset of training scans.

### N4. Network / third-party upload paths (grep `urllib|requests|socket|urlopen|urlretrieve|hf_hub_download|snapshot_download|from_pretrained|pretrained=True|boto3|s3://|gs://|azure|wandb|mlflow|comet|neptune|openai|anthropic|gemini|huggingface_hub|upload|curl|wget|http(s)://` over *.py/*.sh/*.txt/*.json, excl. docs/repro)
- Download-only paths, training/setup only:
  - `src/cnn2d.py:203-210` `timm.create_model(backbone, pretrained=pretrained, ...)` (HF Hub pull of public timm weights); driven by `tools/fetch_backbones.py:54-60` (pins sha256).
  - `scripts/install_weights.py:33-34,106-116` `urllib.request.urlretrieve(args.url, tmp)` — user-supplied URL, download of competitor's own bundle; optional (`--from-zip` alternative).
  - `requirements-inference.txt:9` pip index URL (comment).
- Inference: `inference/tabular_adapter.py:5-6` sets `HF_HUB_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`; TabICL built with `allow_auto_download=False` (`:42`, frozen path uses local `model_path` `:29`). 2D nets are TorchScript (no timm). No urllib/requests in `inference/`.
- Subprocess: `tabular/derive_groups.py:54` (local python scripts), `scripts/predict_folder.py:72` (local predict.sh), `tabular/_original/prepare_candidate.py:16` (`/bin/cp`). No remote.
- No upload code, no cloud storage client, no experiment tracker, no hosted-LLM client anywhere. (Note: `wandb` appears only as an optional import inside the vendored tabicl package, per THIRD_PARTY:173; not installed.)

### N5. Header reads / privacy
- Inference: only `img.header.get_zooms()` (`inference/preprocessing3.py:84`) plus canonical-orientation affine.
- `src/eda_real.py:27-56`: shape, zooms, axcodes, dtype, intensity stats. Training scans only (`DAT_DATA_DIR/niftis`).
- `tools/hdr_fingerprint.py:11-28` reads free-text `descrip`, `db_name`, `aux_file`, `intent_name`, plus geometry/codes, and counts (not content) of header extensions. Shipped log `docs/repro/.../stage_logs/hdr_fingerprint.log`: all four free-text fields are `''` for 1362/1362; extensions 0 for all. Only `pixdim1`, `pixdim3`, `aff_offdiag_max` are consumed (`derive_groups.py:78`), and only `pixdim3` enters the `fam2` rule (`:87-92`). Per-uid CSV stays in `outputs/` (gitignored, not shipped). No patient fields exist; nothing links scans to individuals; groups are site/scanner proxies (organizer said no extra restriction on inferring site).
- Note: a recursive grep matched two RDF metadata lines (`cc:Agent`) inside `docs/figures/pipeline_diagram.svg` as text; the image was not opened or rendered. All later greps exclude images.

### N6. Public third-party docs consulted (WebFetch, no competition content sent)
- https://huggingface.co/jingang/TabICL — `license: bsd-3-clause`.
- https://arxiv.org/abs/2602.11139 (TabICLv2 abstract) — pretraining on "a novel synthetic data generation engine designed for high pretraining diversity"; no real datasets mentioned.
- https://github.com/soda-inria/tabicl README — "tabicl.train generates synthetic prior datasets on the fly ... this is how the TabICLv2 checkpoints were trained". Vendored `tabicl/_sklearn/classifier.py:126-131` names `tabicl-classifier-v2-20260212.ckpt` as "the version used in our TabICLv2 paper".
- https://huggingface.co/timm/efficientnet_b0.ra_in1k — `License: apache-2.0`, "Dataset: ImageNet-1k" (b3.ra2_in1k is the same timm family; competitor's notice says the same; not separately fetched).
- https://www.image-net.org/download.php — "Researcher shall use the Database only for non-commercial research and educational purposes."

### N7. Acquisition groups (rule F, Q3)
- `tabular/derive_groups.py:76-95`: deterministic rules on `pixdim3`, `nz`, `native_spacing` (rounded), and `log10(p99+1)<2.25`. Inputs from `src/eda_real.py`, `tools/hdr_fingerprint.py` and the stage-02 feature table, all over the 1,362 training scans. Docstring :3-5 says the original came from an unsaved interactive step on 2026-09-03 and the rules were "recovered from the original development session logs"; stages/07_tabular.md:158 reports 1362/1362 agreement with the archived `site_diag_rows.csv`, and `acquisition_groups.csv` sha256 `1b704a02…` is the same either way.
- Used only in training (`prepare_tabular.py:67` sample weights, leave-one-fam2-out priors for context rows, balanced `log` subset). Not computed or read at inference (no `fam2`/groups in `inference/`). Per-uid `acquisition_groups.csv` is not shipped, but it can be regenerated automatically from the shipped code.
- The thresholds are hand-chosen constants: a human picked them from training-data exploration. No per-scan manual labelling took place.

### N8. Shipped JSON assets
- `tabular_recipe.json`, `inference_config.json`, `linear_lin.json` hold 0 of the 1,362 training uids (fixed-string grep). The recipe holds only aggregates: 68 feature names, 72 medians, log_prep, 4 priors' coef, and training_policy. I parsed it with the system python `json` module, which is not competitor code.
- The TabICL ckpt load (`classifier.py:349-380`) downloads only when `model_path` is None. The frozen states set `model_path` to the local asset and `allow_auto_download=False` (`freeze_tabular.py:77`), so no download path is reachable at inference.

### N9. Shipped arch
- The harness `inference_config.json` has a single `"arch": "mil"`. `pipeline/05_cnn3d.sh:69` uses `--arch mil` for every run. MedicalNet (`med*` archs, `resnet_*_23dataset.pth` under `<project>/external/`) is never used, and no MedicalNet weights are in the tree or the zip.

---

## Subagent: Reviewer 4 — Data Source and Model Provenance

Status: needs organizer review

No competitor code executed: yes (confirmed). No competitor code was run or imported, no artifacts (.pt/.ts.pt/.ckpt/.pkl.xz/.npy) were loaded or unpickled, no images were viewed (a text grep matched two RDF metadata lines inside `docs/figures/pipeline_diagram.svg`; it was not opened or rendered), no competitor-hosted URL (Dropbox) was fetched, and nothing was written under `2nd Place/` or `references/`.

Read-only inspection performed:
- `2nd Place/` README.md, THIRD_PARTY_NOTICES.md, LICENSE, models/README.md, requirements-*.txt, inference/{tabular_adapter.py, TABICL_MODEL_CARD.md, CANDIDATE.json, tabicl_licenses/LICENSE}, src/{cnn2d.py, nets3.py, eda_real.py}, tools/{hdr_fingerprint.py, fetch_backbones.py}, tabular/{derive_groups.py, freeze_tabular.py (grep), prepare_tabular.py (grep)}, scripts/install_weights.py (grep), pipeline/*.sh (grep), docs/stages/07_tabular.md, docs/REPRODUCIBILITY.md, docs/repro/mode_full_2026-09-28/{README.md, parity/scored_run.log, run/driver.log, run/analyze.log, stage_logs/hdr_fingerprint.log}.
- Harness copy `verification_workspace_2nd_place/extracted/harness_zip/assets/{inference_config.json, tabular_recipe.json, linear_lin.json}` (text/JSON only).
- `inference/tabicl_lib.zip` via `unzip -l` / `unzip -p tabicl/_sklearn/classifier.py`.
- Repo-wide greps for dataset names and aliases, network/upload APIs, AI-tool mentions, and header fields.
- Public third-party docs: HF jingang/TabICL, arXiv 2602.11139, github soda-inria/tabicl, HF timm/efficientnet_b0.ra_in1k, image-net.org/download.php.

Competition-specific checks applied:
- Forbidden/allowed dataset lists (RF "Forbidden/Allowed datasets"; PPMI/LONI/DaTscan/ioflupane aliases).
- Pretrained models: no NC; pretraining data subject to same rules (RF "Pretrained models and licensing").
- Disclosure of every external source (RF "Other – Data Source").
- GR A (AI tools retaining data), GR B (permissive licence of external data), GR E (data used solely for the competition), GR F (annotations).
- RF "Third-party APIs" (no uploads; network only where allowed).
- RF "Privacy" (no identifying header fields, no linking to individuals).
- RF "Release expectations" + organizer ruling (weights trained on comp data only in a private, organizer-only location; code-only publication).

Evidence:
- **External data inventory:** none. 0 grep hits for ppmi, loni, ida.loni, datscan, ioflupane, adni, tcia, openneuro, kaggle, zenodo, physionet, figshare, biobank. All training reads `DAT_DATA_DIR` (`data/raw/niftis`, `labels.csv`), staged from the competition download (`pipeline/00_stage_data.sh`). One undistributed labelled 150-scan "local verification set" (`<original-project>/data-verify`, `run/driver.log:1`) is used only for parity checks (LL 0.09811, `run/analyze.log:13`). Its origin is not stated. Its in-sample-level loss fits a subset of the training scans, and it cannot be labelled test data.
- **Pretrained models inventory:**
  - (1) timm `efficientnet_b0.ra_in1k` and `efficientnet_b3.ra2_in1k`: Apache-2.0 weights, pretrained on ImageNet-1k (HF model card). Pulled via `src/cnn2d.py:208` `timm.create_model(pretrained=True)` and `tools/fetch_backbones.py` (sha256 pinned). They feed 18 shipped TorchScript members.
  - (2) TabICLv2 `tabicl-classifier-v2-20260212.ckpt` (HF jingang/TabICL rev 4dcd344e…, sha256 bdc7dbd5…): BSD-3-Clause (model card). Pretrained only on synthetic prior data (arXiv 2602.11139 abstract; tabicl README). It is shipped in the harness zip, and no real-data pretraining is documented.
  - (3) tabicl 2.2.0 library, vendored: BSD-3-Clause, with an Apache-2.0 forecast subdir (`inference/tabicl_licenses/LICENSE`).
  - (4) MedicalNet / Med3D `resnet_{10,18}_23dataset.pth`: referenced only in unused archs (`src/nets3.py:225-308`). It is loaded from an absent `<project>/external/`, the shipped arch is `mil` (harness `inference_config.json`; `pipeline/05_cnn3d.sh:69`), and no MedicalNet weights are present.
- **ImageNet framing (Q1):** the rules ban NC-licensed external data and models, and they say the pretraining data of a pretrained model "is subject to the same rules". The timm weights themselves are Apache-2.0, so the model licence passes. ImageNet-1k is the pretraining data, and its terms limit use of the database to "non-commercial research and educational purposes" (image-net.org). Under a literal reading of the rules file, then, the pretraining data is NC-restricted and the B0/B3 members fail. Two points weigh the other way. The competitor never accessed ImageNet images (THIRD_PARTY:61), and ImageNet's terms bind researchers who download the database, not users of models trained on it. The terms are also not a CC-NC licence. This is a policy call for the organizer. The competitor disclosed the issue themselves (THIRD_PARTY_NOTICES.md:61; docs/stages/04-06_cnn.md:135-137).
- **Disclosure:** THIRD_PARTY_NOTICES.md §§1-3 and README:243-256,377-382 disclose every external model and source the code loads (TabICL lib, TabICLv2 ckpt with revision and hash, timm B0/B3 with hashes and ImageNet-1k pretraining). MedicalNet is mentioned only as an experiment (THIRD_PARTY:109), which is correct since it is unused. No external dataset exists to disclose. The CNN and TabICL artifacts carry §5 competition-data notes.
- **Network and uploads:** network use is download-only, at training/setup: the HF Hub for public timm weights (`cnn2d.py:208`, `fetch_backbones.py:54-60`), and optional `urllib.request.urlretrieve` of the competitor's own bundle (`scripts/install_weights.py:106-116`). Inference is offline: `tabular_adapter.py:5-6` sets `HF_HUB_OFFLINE`/telemetry off, `allow_auto_download=False`, the ckpt path is local (`classifier.py:349-380`), and the 2D nets are TorchScript. No upload code, cloud SDK, experiment tracker or hosted-LLM client appears anywhere. Subprocess calls run local scripts only.
- **AI tools (GR A):** there is no AI-tool disclosure and no agent config files (`.claude/`, `.cursor/`, `AGENTS.md`, `CLAUDE.md`, `.github/copilot*`; the only dotfiles are `.gitkeep`). There are no mentions of Claude, Codex, ChatGPT, GPT, Copilot, Gemini, OpenAI, Anthropic or LLM. Indirect traces:
  - (a) `src/cnn2d.py:72` (= `inference/cnn2d.py:72`): "co-located axial channels (2026-09-07, external review member E)". This credits a design idea to member "E" of some external review. The idea belongs to the `ax3` view, which is unused (shipped `views: mip3`, `pipeline/06_cnn2d.sh:75`; config :733,786).
  - (b) References to a non-distributed "HANDOFF §0.00" document (`src/roigeom.py:7`, `src/make_regfeat32.py:14`).
  - (c) "original development session logs" (`tabular/derive_groups.py:5`; `docs/stages/07_tabular.md:25`).
  - Lettered "review members" and "HANDOFF" documents are common in LLM multi-agent workflows, but they fit human reviewers too. The evidence is inconclusive. If "E" is a human outside the single-person team, the question is whether code built on competition data, or the data itself, was shared privately (RF "Third-party APIs" last bullet). If "E" is a hosted AI tool, the question is whether competition data (as opposed to code or ideas) was uploaded to a service that retains it (GR A). No static evidence shows data was shared.
- **Rule F / Q3:** `fam2`×`scale` come from deterministic rules (`derive_groups.py:84-95`) over header `pixdim3`, `nz`, `native_spacing` and an intensity p99. They are computed automatically for the 1,362 training scans only, never at inference, and never on test scans. The rules are shipped as code (`tabular/derive_groups.py`, `tools/hdr_fingerprint.py`, `src/eda_real.py`, `pipeline/07_tabular.sh:82-83`). The per-uid CSV is not shipped but regenerates byte-identically (sha `1b704a02…`; 1362/1362 agree, `docs/stages/07_tabular.md:158-175`). In my assessment these are automatically derived training metadata (pseudo-site labels), not manual annotations. Even if the organizer treats them as annotations, the rule-F conditions hold: training only, reproducible from shipped code, inference runs automatically on test data as provided. The thresholds were chosen by hand from training-data exploration, and the original step was interactive and unsaved, then reconstructed. No link to the test set or leaderboard was found for the groups themselves.
- **Privacy:** inference reads only zooms and the affine (`inference/preprocessing3.py:76-87`). `tools/hdr_fingerprint.py:11-28` reads the free-text `descrip`, `db_name`, `aux_file` and `intent_name`, but the shipped log shows all of them empty for 1362/1362 and 0 header extensions. Only geometry fields are consumed downstream. No patient-identifying fields exist or are used, no cross-UID linking to individuals was found, and per-uid outputs are not shipped. Fingerprinting site/scanner is permitted (organizer: no extra restriction).
- **Dependencies:** all are OSS licences that permit commercial use (THIRD_PARTY §4: MIT, BSD, Apache-2.0, MPL-2.0, PSF-2.0, ISC, MIT-CMU; numpy/scipy wheels bundle GPL-3.0-with-GCC-runtime-exception and LGPL-2.1 libquadmath, both standard and permitting commercial use). CatBoost is imported only by archived `tabular/_original/bench_v16.py` (Apache-2.0 upstream, unused). NVIDIA CUDA libraries are proprietary but optional, and the runtime supplies them (CPU fallback exists). Solution licence: MIT (`LICENSE`).
- **Release:** README:131,142,153 links two Dropbox `scl/fi…rlkey` share links (anyone-with-link style): `navattack_v17_weights.zip` (all 60 assets, including the 52 competition-trained CNN files, the linear models, `template_v6.npy` built from training crops, and the two frozen TabICL contexts) and `navattack_v17_cnn_raw_checkpoints.zip`. models/README.md:25-31 and THIRD_PARTY:177 state that the contexts let the training labels of all 1,362 (raw) / 1,148 (log) scans be recovered, along with per-feature statistics. LICENSE/README:377-379 declare the fine-tuned weights MIT-"released". THIRD_PARTY:177 says a public release of the contexts "waits for the organisers' agreement", yet the README already distributes them in the linked bundle. I did not fetch the links, so their current access level is unverified.

Findings:
- Severity: high
  Summary: The competitor distributes weights trained on competition data, including the frozen TabICL contexts from which training labels can be recovered, through public Dropbox share links in the README. This conflicts with the organizer ruling (private, organizer-only location) and with GR E (data used solely for the competition).
  Evidence: `2nd Place/README.md:131,142,153`; `models/README.md:17,25-31`; `THIRD_PARTY_NOTICES.md:57,59,177`; `LICENSE`; README:377-379.
  Recommendation: Ask the competitor to revoke or restrict both Dropbox links (or move them to organizer-only storage), remove the links and the "released ... MIT" wording for weights from README/models/README/THIRD_PARTY before any code publication, and confirm in writing that no other copies are public. Organizer to decide whether the contexts may ever be released before the dataset itself is published.
- Severity: medium
  Summary: 18 of the shipped 2D members are fine-tuned from timm EfficientNet weights pretrained on ImageNet-1k, whose terms are non-commercial research and educational only. The weights' own licence is Apache-2.0.
  Evidence: `THIRD_PARTY_NOTICES.md:48-61`; `src/cnn2d.py:203-210`; `tools/fetch_backbones.py:18`; HF timm/efficientnet_b0.ra_in1k model card; image-net.org/download.php.
  Recommendation: Organizer ruling on Q1. Under a strict reading of "pretraining data is subject to the same rules", these members are non-compliant. Under the common reading (the model licence governs and the dataset terms bind database downloaders), they pass. The competitor disclosed this. If the ruling is strict, the 2D members (stack weight 0.3+0.3) would have to be removed or retrained, for example with `--scratch` (`cnn2d.py:303`).
- Severity: low
  Summary: There is no AI-tool disclosure, and comments credit an "external review member E" (`cnn2d.py:72`) and cite an undistributed "HANDOFF" document and "development session logs". These hint at either an outside human reviewer or an AI-assistant workflow, and their provenance is unclear.
  Evidence: `src/cnn2d.py:72` (= `inference/cnn2d.py:72`); `src/roigeom.py:7`; `src/make_regfeat32.py:14`; `tabular/derive_groups.py:4-5`; `docs/stages/07_tabular.md:25`.
  Recommendation: Ask the competitor who or what "external review member E" is, and whether any AI coding tools were used. If so, which ones, and was any competition data (scans, labels, per-scan values, logs with uids) ever uploaded to a tool or service that stores or retains data (GR A)? If "E" is a person outside the team, ask whether competition-derived code or data was shared with them. The idea credited to "E" sits in an unused code path (`ax3`).
- Severity: low
  Summary: TabICLv2 checkpoint compliant. It is BSD-3-Clause and pretrained only on synthetic prior data, which answers overview Q2.
  Evidence: HF jingang/TabICL (`license: bsd-3-clause`); arXiv 2602.11139 abstract; soda-inria/tabicl README; `inference/TABICL_MODEL_CARD.md:2`; `THIRD_PARTY_NOTICES.md:27-35`.
  Recommendation: No action. Treat as informational.
- Severity: low
  Summary: The acquisition groups (`fam2`/`scale`) are automatic, rule-based, training-only derived metadata, not manual annotations of test data. They are reproducible from shipped code and never used at inference, which answers overview Q3.
  Evidence: `tabular/derive_groups.py:76-95`; `tools/hdr_fingerprint.py`; `src/eda_real.py`; `pipeline/07_tabular.sh:82-83`; `docs/stages/07_tabular.md:158-175`; no group references in `inference/`.
  Recommendation: Organizer may confirm the reading that these are not "annotations". If they are deemed annotations, rule F is still satisfied (training only, delivered as code, automatic test-time path).
- Severity: low
  Summary: Privacy check passes. The free-text header fields are read in a training-only survey but are empty in every scan. Only geometry is used, and nothing links scans to individuals.
  Evidence: `tools/hdr_fingerprint.py:11-28`; `docs/repro/mode_full_2026-09-28/stage_logs/hdr_fingerprint.log`; `inference/preprocessing3.py:84`.
  Recommendation: No action.
- Severity: low
  Summary: No external datasets, no forbidden sources, and no third-party uploads. Network use is limited to downloads of public weights and of the competitor's own bundle at training/setup, and inference is offline.
  Evidence: grep results (N1, N4); `inference/tabular_adapter.py:5-6,42`; `tabicl/_sklearn/classifier.py:349-380`; `scripts/install_weights.py:106-116`.
  Recommendation: No action. Optionally ask the competitor to confirm that the undistributed 150-scan "data-verify" folder is a subset of the 1,362 training scans.

Open questions:
- Q1 (organizer): Does ImageNet-1k pretraining of Apache-2.0 timm weights violate "pretraining data is subject to the same rules / no NC"?
- Organizer: Should the competitor be required to take down the public Dropbox links now, and may the label-recoverable TabICL contexts ever be released before the dataset is published?
- Competitor: Who or what is "external review member E", and what are the "HANDOFF" document and "development session logs"? Were AI tools used, and did any competition data reach a service that retains data?
- Competitor: Is the 150-scan local verification set (`<original-project>/data-verify`) drawn solely from the 1,362 training scans?
- Not verified statically: the actual contents of the frozen `.pkl.xz` contexts and the current access level of the Dropbox links. Neither was opened or fetched, per the global rules.
