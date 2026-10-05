## Subagent: Reviewer 2 — Eligibility and Disqualifier (2nd place, team NavAttack, comp 311)

Status: needs organizer review

The status is driven by one policy question (Q1, ImageNet-1k pretraining of the Apache-2.0 timm weights). No disqualifier was found. If the organizer accepts ImageNet-pretrained, permissively licensed weights, this review becomes **pass with notes**.

No competitor code executed: yes. No competitor code, script, notebook or tool was run or imported. No artifact was loaded, unpickled or decompressed (`.pt`, `.ts.pt`, `.ckpt`, `.pkl.xz`, `.npy`). `tabicl_lib.zip` was only listed (`unzip -l` / `unzip -Z1`) and its text members were streamed with `unzip -p` into grep or sed. Nothing was extracted to disk. No image was opened or viewed. No web fetch was made of any competitor URL (Dropbox, Hugging Face) or of any other URL. Nothing under `2nd Place/` or `references/` was written.

Read-only inspection performed:
- `verification_workspace_2nd_place/notes/coordinator_context.md`, `notes/governing_rules.md`, `solution_overview.md` (§1, §2, §7)
- `references/verification_comp_specific_rules.md`
- `.claude/skills/submission-verification/references/reviewers.md` §2 and `report.md`
- `2nd Place/LICENSE`, `THIRD_PARTY_NOTICES.md` (all 177 lines), `requirements-inference.txt`, `requirements-inference.lock.txt`, `requirements-train.txt`
- `2nd Place/inference/TABICL_MODEL_CARD.md`, `inference/tabicl_licenses/LICENSE` (236 lines), and the `inference/tabicl_lib.zip` listing (91 members) plus a grep of every member for licence, attribution and network strings
- `inference/main.py`, `inference/tabular_adapter.py:1-50`, `inference/inference_main.py` (device lines), `inference/cnn2d.py:66-78`
- `src/cnn2d.py`, `src/cnn3.py:283-395`, `src/nets3.py:225-325` (pretrained call sites), `tools/fetch_backbones.py:1-40`, `pipeline/05_cnn3d.sh`, `pipeline/06_cnn2d.sh`
- A repo-wide import inventory over all `*.py`
- A repo-wide grep for cheating indicators (test_labels, private, leaderboard/board, probe, pseudo, manual annotation, submission_format) and for AI-tool, external and licence mentions in README, models/README and docs
- `models/SHA256SUMS.scored.txt` and `hashes/harness_zip_members.sha256`, for the TabICL checkpoint hash

Competition-specific checks applied:
- RF "Pretrained models and licensing": the winning solution must be under MIT. Applied to `LICENSE`.
- RF / Rules: "Any software required to generate the solution must also be available under an open source license that does not prohibit free commercial use", and the solution must be reproducible "using only free open source software". Applied to every package, bundled code and the torch/CUDA stack.
- RF Homepage: no NC, CC NC or CC BY-NC licence on any data or model. "The pretraining data of any pretrained model is subject to the same rules". Applied to TabICLv2 and to the timm EfficientNet-B0/B3 weights (CS7, F10, Q1, Q2).
- RF "Forbidden datasets", and GR B (external data must be under a permissive licence such as CC0 or CC-BY). Applied to external data (none found; F13 concurs).
- RF "Eligibility", CS14 and CS15: identity, sponsor, panel, age and territory, submission caps and multiple accounts. These are recorded as confirmed by DrivenData.
- reviewers.md §2: obvious disqualifiers and clear cheating indicators.

### 1. Licence and eligibility summary

| Item | Result | Evidence |
|---|---|---|
| Solution licence | **MIT**, standard text, "Copyright (c) 2026 Navid Shabani Barzegar (DrivenData team "NavAttack")". Third-party components keep their own licences | `2nd Place/LICENSE:1-25` |
| Winner identity: not a sponsor employee or officer, not on the 11-person annotation panel, not at a contributing center | **Confirmed by DrivenData** (organizer, in the rules file). Static review cannot add to this | `references/verification_comp_specific_rules.md` "Other checks / Eligibility" |
| Age 18+, permitted territory, Russian-residency prize bar | Covered by the DrivenData identity confirmation. Not checkable statically | same |
| Submission cap / multiple accounts | **Checked by DrivenData, no issues** (organizer) | rules file "Submission limits" |
| Team composition | LICENSE names a single author. No co-author is named anywhere in the docs | `LICENSE:3` |
| No NC component on the inference or training path | None found (see §2). Open policy question on ImageNet pretraining data (Q1) | below |

### 2. Inventory of external models, packages and code

**2a. Pretrained models**

| Model | Used where | Licence (source) | Pretraining data | Assessment |
|---|---|---|---|---|
| TabICLv2 checkpoint `tabicl-classifier-v2-20260212.ckpt` (HF `jingang/TabICL`, rev `4dcd344e…`) | Inference (TabICL block) and freeze at training | BSD-3-Clause. The copied model card's front matter says `license: bsd-3-clause` (`inference/TABICL_MODEL_CARD.md:2`; `THIRD_PARTY_NOTICES.md:27-35`) | Synthetic priors, per the TabICL papers (competitor's citation). Consistent with the bundled synthetic-prior generator code (`tabicl/prior/_mlp_scm.py`, `_tree_scm.py`, `graph_lib/*` in the zip) | OSI, commercial use OK. The shipped asset hash `bdc7dbd5…d4ed0` equals the competitor's stated HF hash (`models/SHA256SUMS.scored.txt:56`, `hashes/harness_zip_members.sha256:58`, `THIRD_PARTY_NOTICES.md:33`). The HF page was **not fetched**, so the card copy and the hash match are taken as delivered. It is loaded locally with `model_path` (`tabular_adapter.py:29`), and the HF download branch is unreachable (`HF_HUB_OFFLINE=1`, `tabular_adapter.py:5`) |
| timm `efficientnet_b0.ra_in1k` (sha256 `d5698997…`) | Initialisation of the 12 B0 2D members (training). Inference loads the fine-tuned TorchScript files, not timm | Apache-2.0 (HF model card, as stated by the competitor, `THIRD_PARTY_NOTICES.md:50`) | **ImageNet-1k**. Its terms of access are non-commercial research and education (the competitor says so at `THIRD_PARTY_NOTICES.md:61`) | Weights licence OK. **Q1**: pretraining-data policy |
| timm `efficientnet_b3.ra2_in1k` (sha256 `279d2a53…`) | Initialisation of the 6 B3 2D members | Apache-2.0 (`THIRD_PARTY_NOTICES.md:51`) | ImageNet-1k | As above |
| MedicalNet (Tencent Med3D ResNet-10/18, "23 datasets") | **Not used.** Only in archs `med*` (`src/nets3.py:225-325`). It resolves against `<project>/external/`, which is absent. All shipped 3D runs use `--arch mil` (`pipeline/05_cnn3d.sh:21,69`) | Code comment says MIT (`nets3.py:226`) | 23 medical datasets of mixed provenance | Not on the shipped path. No eligibility impact. Recommend removing it or noting it before release |
| 3D HemiMIL CNN | Trained from scratch. The `--init` (SSL on fold-training rows) is not used by `05_cnn3d.sh` | own (MIT) | competition training data only | OK |

Weights fine-tuned on competition data (18 TorchScript, 34 3D `.pt`) and the frozen TabICL contexts are the competitor's own work under MIT, with the Apache-2.0 notice kept for the timm-derived weights (`THIRD_PARTY_NOTICES.md:55-59`). Their release and hosting belong to Release Readiness (F11).

**2b. Bundled code: `inference/tabicl_lib.zip`** (tabicl 2.2.0, `tabicl/__about__.py` = "2.2.0", 91 members)

| Component | Licence | Evidence | On v17 inference path? |
|---|---|---|---|
| tabicl (Soda team @ Inria) | BSD-3-Clause | `tabicl_licenses/LICENSE:1-28` | Yes |
| `tabicl/forecast/*`, derived from TabPFN-TS (Prior Labs) | Apache-2.0 (the full text is in the same LICENSE file; "Copyright 2025 Prior Labs GmbH") | `tabicl_licenses/LICENSE:30-236` | No |
| `forecast/_ts_dataframe.py`, modified from AutoGluon | Apache-2.0 (AutoGluon's own licence; the file header names only the source) | zip member line 1 | No |
| `_sklearn/sklearn_utils.py`, adapted from scikit-learn | BSD-3 | zip member line 6 | Yes |
| `_sklearn/preprocessing.py:547`, adapted from yandex-research/tabular-dl-tabr | MIT (upstream; not stated in the header) | zip member | Yes |
| `_model/rope.py`, "Copy from lucidrains/rotary-embedding-torch" | MIT (upstream; not stated in the header) | zip member line 4 | Yes |
| `train/_muon.py`, from fabian-sp/sda ← Megatron-LM fork | Upstream licences not stated in the header (Megatron-LM is BSD-3-style) | zip member lines 1-5 | No (training module only) |
| No `.dist-info`/RECORD in the zip. The competitor claims every member matches the PyPI 2.2.0 wheel RECORD (`THIRD_PARTY_NOTICES.md:14`) | n/a | `unzip -Z1` shows no dist-info | Not verified (no wheel fetched) |

THIRD_PARTY_NOTICES §1 omits the `rope.py` (lucidrains, MIT) credit. That is a minor attribution gap with no eligibility impact. Optional imports inside tabicl (wandb, shap, shapiq, xgboost, transformers, statsmodels, numba) are not installed and not on the v17 path (`THIRD_PARTY_NOTICES.md:173`). All of them are OSS in any case.

**2c. Python packages** (licences as reported by the competitor from wheel METADATA, `THIRD_PARTY_NOTICES.md:76-147`, cross-checked against the three requirements files and against a repo-wide import inventory)

- Inference env (32 packages, `requirements-inference.lock.txt:10-41`): torch 2.12.1 (BSD-3), numpy 2.2.6 (BSD-3), scipy 1.15.3/1.17.1 (BSD-3), scikit-learn 1.8.0 (BSD-3), pandas 3.0.3 (BSD-3), nibabel (MIT), loguru (MIT), einops (MIT), tqdm (MPL-2.0 AND MIT), psutil (BSD-3), huggingface-hub (Apache-2.0), joblib (BSD-3), threadpoolctl (BSD-3), certifi (MPL-2.0), requests (Apache-2.0), urllib3 (MIT), idna (BSD-3), charset-normalizer (MIT), filelock (MIT), fsspec (BSD-3), hf-xet (Apache-2.0), jinja2 and markupsafe (BSD-3), mpmath and sympy (BSD), networkx (BSD-3), packaging (Apache-2.0 OR BSD-2), python-dateutil (BSD/Apache dual), pyyaml (MIT), setuptools (MIT), six (MIT), typing-extensions (PSF-2.0). **All OSI-approved, commercial use OK.**
- Training env (67 packages, `requirements-train.txt:7-73`) adds timm (Apache-2.0), torch 2.13.0 (compound Apache/BSD/MIT/BSL-1.0), torchvision (BSD), lightgbm (MIT), monai (Apache-2.0, MedicalNet experiment only), matplotlib (PSF-based matplotlib licence, used only by `docs/figures/make_figures.py`), scikit-image (BSD-3), SimpleITK (Apache), torchio (Apache-2.0), surfa (MIT), pillow (MIT-CMU), safetensors (Apache), rich, typer, httpx, httpcore, etc. (MIT/BSD/ISC/Apache). **All OSI-approved, commercial use OK.**
- Not installed in either env: CatBoost (Apache-2.0), imported only by the archived `tabular/_original/bench_v16.py` (not shipped). Also lightgbm in `inference/portable3.py:129`, a lazy import reached only if `gbms` is non-empty. The shipped config has `gbms: {}`.
- Repo-wide third-party imports found: numpy, torch, pandas, sklearn, scipy, matplotlib, timm, tabicl, nibabel, lightgbm, joblib, monai, loguru, huggingface_hub, catboost. **Every one is in the notices table.** No undisclosed package was found.
- Binary libraries in the numpy/scipy wheels: OpenBLAS/LAPACK (BSD-3 variants), libgfortran/libgcc (GPL-3.0+ with GCC Runtime Library Exception), libquadmath (LGPL-2.1+) (`THIRD_PARTY_NOTICES.md:149-160`). All are open source, and the GCC exception permits commercial use of linked programs. No issue.

**2d. torch / CUDA and "free open source software" reproducibility**
- The scored run used DrivenData's runtime image: `torch==2.12.1+cu129`, A100 (`requirements-inference.txt:3,9`; `THIRD_PARTY_NOTICES.md:166-168`). The Linux CUDA torch wheels pull NVIDIA runtime libraries (cuBLAS, cuDNN, NCCL and others, as `nvidia-*` pip wheels). These are **proprietary freeware under NVIDIA's EULA**: free to use and redistribute in that form, but not open source. The competitor states plainly that they did not read those terms (`THIRD_PARTY_NOTICES.md:168`).
- The proprietary part is optional. Inference falls back to CPU (`inference_main.py:67,292`), and `requirements-inference.txt:8,14` installs the CPU/plain torch. The competitor reports CPU inference of the unmodified archive on 150 scans in 530 s to 17 min (`THIRD_PARTY_NOTICES.md:168`). By extrapolation, the 524 test scans would run in roughly 0.5 to 1 h, within the 3 h limit. Training code selects MPS → CUDA → CPU (`src/cnn2d.py:48-51`, `src/cnn3.py:53-55`, `src/make_regfeat32.py:92-93`). Training was actually run only on Apple MPS (proprietary OS framework) and never on CPU or CUDA (`README.md:90-91`; `docs/REPRODUCIBILITY.md:462-470`).
- Assessment: an all-FOSS (CPU) reproduction path exists in code, so the rule is met in principle. Reliance on CUDA/cuDNN or Apple Metal is accelerator-only, and is the same for every GPU-based entry, including the organizer's own runtime. **Low / info.** It is not a disqualifier. Organizer may note it.

### 3. Disqualifier and cheating-indicator scan

- No test UIDs appear in the tree or the harness (R1 grep against the 524 uids, 0 hits; `solution_overview.md:241`). No reference to `test_labels` anywhere. `submission_format.csv` is read only for uids and order in inference, and its values are overwritten (F4).
- No external datasets, and no PPMI/LONI/DaTscan cohort references (F13). The only external inputs are the three pretrained models above.
- No hard-coded predictions and no per-uid lookups. The 0.005/0.995 values are explained by the configured clip (coordinator finding 3).
- No network at inference: `HF_HUB_OFFLINE=1` (`tabular_adapter.py:5-6`), the checkpoint is loaded from a local path, and there is no urllib/requests in `inference/`. Training-time network use is limited to downloading public timm weights (F8). No upload of competition data was found.
- Leaderboard feedback guided T, the clip and the `head_ml` exclusion (`src/fixstack.py:4-5`, `src/train_final3.py:57-60`, `src/cnn3.py:12`). This is F9, owned by Code Integrity and the organizer. It uses public-LB scores only, which is permitted platform feedback, and it is not test-label access. **Not a clear cheating indicator.** It is recorded here only so the organizer's F9 ruling also covers eligibility.
- The figures use private-LB scores (`docs/figures/make_figures.py:91-111`), which could be known only after the competition closed, so this is post-hoc documentation. The figures were not viewed.
- A comment in `inference/cnn2d.py:72` reads "(2026-09-07, external review member E)". It sits on the `ax3` view, which is not shipped (the shipped view is `mip3`). If "member E" was a person outside the single-member team who saw code or data, that would touch the Rules ban on privately sharing code or data developed with the competition data. If it was an AI tool, it would touch GR A, unless the tool runs locally. The evidence is ambiguous, and the provenance is owned by Data Source (F12). **Low; ask the competitor.**

Evidence:
- `2nd Place/LICENSE:1-25`: MIT licence, single named copyright holder, pointer to the third-party notices.
- `2nd Place/inference/tabicl_licenses/LICENSE:1-236`: BSD-3 (Inria) plus an Apache-2.0 section for `forecast/` (Prior Labs). No NC terms. A grep for "non-commercial" in the licence and in all 91 zip members found no hits.
- `2nd Place/inference/TABICL_MODEL_CARD.md:2`: `license: bsd-3-clause`.
- `2nd Place/THIRD_PARTY_NOTICES.md:48-61`: timm B0/B3 are Apache-2.0, ImageNet-1k pretrained. The competitor discloses that ImageNet's terms are non-commercial.
- `2nd Place/THIRD_PARTY_NOTICES.md:166-168`: CUDA libraries were not licence-checked. CPU fallback with timings.
- `2nd Place/models/SHA256SUMS.scored.txt:56` = `hashes/harness_zip_members.sha256:58` = `THIRD_PARTY_NOTICES.md:33`: TabICL ckpt sha256 `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`.
- `2nd Place/pipeline/05_cnn3d.sh:69`: all 3D runs use `--arch mil` and no `--init`, so MedicalNet and SSL init are unused.
- `2nd Place/tools/fetch_backbones.py:34-40`: pinned timm repo ids and hashes.

Findings:
- Severity: medium (policy, needs an organizer ruling)
  Summary: The 18 shipped 2D members are fine-tuned from timm EfficientNet-B0/B3 weights. The weights are Apache-2.0, but they were pretrained on ImageNet-1k, whose terms of access are non-commercial. The rules file says the pretraining data of a pretrained model "is subject to the same rules", so a literal reading could exclude these weights.
  Evidence: `THIRD_PARTY_NOTICES.md:48-61`; `tools/fetch_backbones.py:18-21`; `src/cnn2d.py:203-215,246`; `pipeline/06_cnn2d.sh:13,95`; rules file "Pretrained models and licensing", bullet 2.
  Recommendation: Organizer to rule on Q1 (framed below). If ImageNet-pretrained, permissively licensed weights are accepted, this finding closes as pass-with-note. If they are not accepted, the 2D members (stack weight 0.3 + 0.3 of 2.7 total ratio) would have to be removed or replaced, which changes the scored submission.
- Severity: low
  Summary: The scored GPU run depends on NVIDIA CUDA/cuDNN libraries (proprietary EULA, not OSS), which the competitor did not licence-check, and training ran only on Apple MPS. Code paths exist for CPU-only FOSS reproduction of both inference and training.
  Evidence: `requirements-inference.txt:3,9`; `THIRD_PARTY_NOTICES.md:166-168`; `inference_main.py:67,292`; `src/cnn2d.py:48-51`; `docs/REPRODUCIBILITY.md:462-470`.
  Recommendation: Treat as accelerator-only and acceptable, consistent with the organizer's own GPU runtime. Optionally note in the release that the CPU path is the FOSS-only reproduction route.
- Severity: low
  Summary: An ambiguous code comment credits an "external review member E" for a (non-shipped) design idea. That could mean private sharing outside the team, or a non-local AI tool.
  Evidence: `2nd Place/inference/cnn2d.py:72` (identical copy in `src/cnn2d.py`).
  Recommendation: Ask the competitor who or what "external review member E" is (joint with Data Source F12). No action is needed if it is a local model or a public-forum discussion.
- Severity: low (info)
  Summary: All packages, bundled code and the TabICLv2 checkpoint carry OSI-approved licences that allow commercial use (MIT/BSD/Apache/MPL/PSF/ISC/LGPL/GPL-with-runtime-exception). No NC or proprietary package was found, and every third-party import in the repo is disclosed.
  Evidence: §2 above; `THIRD_PARTY_NOTICES.md:76-160`; the requirements files; the zip listing.
  Recommendation: None for eligibility. Before release, add the `rope.py` (lucidrains, MIT) credit to THIRD_PARTY_NOTICES §1, and drop or annotate the unused MedicalNet code paths.
- Severity: info
  Summary: The solution licence is MIT, as the rules require. Eligibility (identity, sponsor, panel, age and territory) is confirmed by DrivenData, and submission-cap and multi-account checks were done by DrivenData with no issues. No clear cheating indicators were found. LB-informed tuning (F9) is deferred to the Code Integrity and organizer ruling.
  Evidence: `LICENSE`; rules file "Eligibility" and "Submission limits"; §3 above.
  Recommendation: None.

Open questions:
- **Q1 (organizer policy, framed):** The rules say pretrained models may not carry NC licences, and that "the pretraining data of any pretrained model is subject to the same rules". The timm EfficientNet-B0/B3 weights carry Apache-2.0, a permissive licence that allows commercial use. They were trained on ImageNet-1k, whose access terms restrict use of the *images* to non-commercial research and education. The competitor never downloaded ImageNet images (`THIRD_PARTY_NOTICES.md:61`). The organizer must decide one of the following:
  - (a) Only the licence of the released model weights matters. ImageNet-pretrained Apache-2.0 weights are then allowed, which is the common practice across ML competitions and for most ImageNet-pretrained timm and torchvision models.
  - (b) The pretraining dataset's own terms count. ImageNet-pretrained weights would then be prohibited, and this solution, along with any other entry using ImageNet backbones, would need its 2D members removed.

  A consistent ruling should be applied across all three winners. The organizer should also note whether any forum or FAQ guidance on ImageNet backbones was given during the competition (the rules file records "None").
- Q2 (Data Source): Confirm from the TabICLv2 paper or model card that pretraining used synthetic priors only. The static evidence (prior generator code in the zip, the competitor's citation) is consistent with that, but the paper was not fetched.
- Q3: Who or what is "external review member E" (`cnn2d.py:72`)?
- Not verified statically: that the `tabicl_lib.zip` members match the PyPI 2.2.0 wheel RECORD, and that the HF TabICL and timm model cards currently state BSD-3 and Apache-2.0. This would need a web fetch, which is outside this review's rules. The organizer may spot-check the public HF pages.

Confirmation: no competitor code was executed, no competitor module was imported, no artifact was loaded or unpickled, no image was viewed, and no web fetch was performed.
