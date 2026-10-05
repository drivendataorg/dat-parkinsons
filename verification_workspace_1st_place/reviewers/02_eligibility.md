## Subagent: Eligibility and Disqualifier (Reviewer 2), 1st place (TheAvengers: dvtle, SamuelAnsel)

Status: needs organizer review

There are three open organizer items: (Q1) the cross-winner ImageNet/CIFAR pretraining-data ruling, (Q2) the licence and provenance of the synthetic PET data, and (Q3) the MIT relicensing of the solution. No disqualifier and no clear cheating indicator was found. If Q1 and Q2 are resolved favourably and the team relicenses under MIT, this review becomes **pass with notes**.

No competitor code executed: yes. No competitor script, notebook, harness `.py`, justfile, Makefile or Docker build was run or imported. No artifact was loaded or unpickled. `weights.zip` was not extracted, and no `.so` was extracted or executed. Vendored-library text was read only from the existing read-only copies in `extracted/src_zip/`. **No image was opened or viewed**, including `1st Place/img/*`. Web fetches went only to public third-party pages: PyPI JSON for ptwt, openTSNE and pykwalify; the Zenodo record and its API; the GitHub page and API for Krying/PD_SSL_ZOO; the GitHub pages for Tencent/MedicalNet, MIC-DKFZ/MedNeXt and kenshohara/3D-ResNets-PyTorch; the HF model cards for the two `edadaltocg` models; and a web search plus PubMed for the Won et al. paper (ScienceDirect and Cell returned 403). No `dinhviettoanle/*` URL and no OneDrive URL was fetched. Nothing was downloaded, and no competition or submission content was sent anywhere. Nothing under `1st Place/` or `references/` was written.

Read-only inspection performed:
- `notes/coordinator_context.md`, `notes/governing_rules.md`, `solution_overview.md` (§1, §6, §7), `references/verification_comp_specific_rules.md`, skill `reviewers.md` §2 and `report.md`
- `1st Place/README.md` (L10-17, L20-40, L68-100, L230-250), `1st Place/pyproject.toml`, `competition-sfmn-parkinsons-runtime/{LICENSE,runtime/pyproject.toml,runtime/uv.lock,runtime/Dockerfile,runtime/apt.txt}`
- Vendored code: `1_initial_training/models/{medicalnet,mednext}/*`, `src/{medicalnet,mednext,models}/*`
- Pretrained loading: `1_initial_training/train_2d_resnet.py` L100-140, `train_2d_vit.py` L92-115, `src/inference_2d_avg.py` L29,77-79, `src/inference_2d_rnn.py` L36,165-175, `scripts/download_weights.py` L9,53-54 (read only)
- Vendored libs in src.zip: `notes/src_zip_listing.txt`; `extracted/src_zip/src/libs/{xgboost-3.3.0.dist-info/METADATA,WHEEL,sboms/auditwheel.cdx.json}`, plus version files and headers of openTSNE, ptwt, pywt, radiomics, pykwalify and ruamel.yaml
- Greps for cheating indicators across the tree, the harness root and src.zip `.py` (test_labels, leaderboard, network calls, upload)

Competition-specific checks applied: CS13, CS14 (record only), CS15, plus inputs to CS16, CS8 and CS9 (Data Source and Release Readiness own the final calls).

### 1. Eligibility summary

| Item | Result | Source |
|---|---|---|
| Not a sponsor employee, annotation panel member or contributing centre; age and territory (incl. the Russia prize exclusion) | **Confirmed by DrivenData**. Not re-checked | rules file "Other checks"; CS13 |
| Submission cap, multiple accounts, private sharing | **Checked by DrivenData, no issues**. Not re-checked | rules file; CS14 |
| Obvious disqualifiers | None found | §4 |
| Clear cheating indicators | None found. 0 test uids in tree or harness (coordinator finding 4). No `test_labels` reference, no network or upload calls on the inference path (grep). One leaderboard-derived global temperature factor (see F6, owned by Code Integrity) | grep; `validate_pipeline.py` L125, L698 |
| Solution licence = MIT | **No.** There is no LICENSE file at the tree root. README L247-250 says the materials are confidential and may not be shared. The only LICENSE is DrivenData's MIT runtime licence (`competition-sfmn-parkinsons-runtime/LICENSE`, "Copyright (c) 2023 DrivenData") | F1 |
| All software OSS that allows commercial use | Yes for every Python dependency. ptwt is EUPL-1.2 (OSI, copyleft). The NVIDIA CUDA stack is proprietary, but a CPU path exists | §2, F4, F5 |

### 2. Dependency licence inventory

**2a. Team `1st Place/pyproject.toml` (direct deps):** loguru (MIT), matplotlib (Matplotlib/PSF-style, permissive), monai (Apache-2.0), nibabel (MIT), numpy (BSD-3), opencv-python (Apache-2.0), pandas (BSD-3), pyradiomics from git `AIM-Harvard/pyradiomics` (BSD-3), scikit-image (BSD-3), scikit-learn (BSD-3), seaborn (BSD-3), simpleitk (Apache-2.0), timm (Apache-2.0), torch 2.12.1+cu129 and torchvision 0.27.1+cu129 (BSD-3; CUDA builds pull NVIDIA wheels), torchio 1.2.1 (Apache-2.0), tqdm (MPL-2.0/MIT), umap-learn (BSD-3), xgboost (Apache-2.0), omegaconf (BSD-3), torchinfo (MIT), **ptwt (EUPL-1.2)**. Every one is OSI-approved and allows commercial use. Licences come from well-known upstream metadata; ptwt was confirmed on PyPI JSON.

**2b. Runtime `runtime/pyproject.toml` and `uv.lock` (136 packages).** This is DrivenData's runtime, extended by the team: the added entries `opentsne, ptwt, pykwalify, PyWavelets, pyradiomics(git), ruamel.yaml, xgboost` come with a French comment at L60. Notable licences: accelerate, transformers, tokenizers, safetensors, huggingface-hub, hf-xet, peft (Apache-2.0); antspyx (Apache-2.0); lightgbm (MIT); polars (MIT); numba/llvmlite (BSD-2); triton (MIT); pytorch-lightning/torchmetrics (Apache-2.0); statsmodels (BSD-3); scipy (BSD-3); xarray (Apache-2.0); pydantic (MIT); requests (Apache-2.0); opentsne 1.0.4 (BSD-3); pywavelets 1.10.0 (MIT); pykwalify 1.8.0 (MIT); ruamel-yaml 0.19.1 (MIT); docopt (MIT); xgboost 3.4.1 (Apache-2.0). The remaining entries are standard permissive libraries (MIT, BSD, Apache-2.0, PSF, MPL-2.0).
**Proprietary:** `cuda-toolkit 12.9.1`, `cuda-bindings` (NVIDIA Software License), and `nvidia-cublas/cudnn/cufft/cufile/curand/cusolver/cusparse/cusparselt/nvjitlink/nvshmem/nvtx/cuda-runtime/nvrtc/cupti-cu12` (NVIDIA proprietary EULA). `nvidia-nccl-cu12/13` is BSD-3. The Docker base image is `nvidia/cuda:12.9.0-runtime-ubuntu22.04` (`Dockerfile` L2, NVIDIA Deep Learning Container licence). These belong to the organizer-supplied runtime image.

**2c. Libraries vendored in the harness `src.zip` under `src/libs/`** (compiled cp312 x86_64 binaries plus Python sources):

| Lib | Version (file) | Upstream licence | Evidence in zip | Note |
|---|---|---|---|---|
| openTSNE (+ vendored `dependencies/annoy`) | 1.0.4 (`openTSNE/version.py`) | BSD-3-Clause (PyPI); annoy Apache-2.0 | No LICENSE file in zip | 5 `.so` |
| ptwt | 1.0.1 (`ptwt/version.py`) | **EUPL-1.2** (PyPI classifier "OSI Approved :: EUPL 1.2") | No LICENSE file | Copyleft (redistribution must keep EUPL for ptwt). Does not conflict with an MIT licence on the team's own code, since ptwt is an unmodified dependency |
| pykwalify | 1.8.0 (`__init__.py` L11) | MIT (PyPI) | No LICENSE file | pyradiomics dependency |
| pywt (PyWavelets) | — | MIT | `pywt/__init__.py` L6 "See LICENSE for more details", but no LICENSE file is shipped | 4 `.so` |
| radiomics (pyradiomics) | git dev build (`_version.py`) | BSD-3-Clause | No LICENSE file | 2 `.so` |
| ruamel.yaml | 0.19.1 (`__init__.py` L10) | MIT (`clibz/setup.py` L25 `license='MIT'`) | — | |
| xgboost incl. `lib/libxgboost.so` (190,670,977 B) | **3.3.0** (dist-info), but the lock pins 3.4.1 | Apache-2.0 (`dist-info/METADATA` "License: Apache-2.0", OSI classifier) | dist-info present, no LICENSE file | PyPI manylinux_2_28 wheel (`WHEEL`). Large size is consistent with the CUDA-enabled PyPI build. It requires `nvidia-nccl-cu12` (METADATA) |
| `xgboost.libs/libgomp-e985bcbb.so.1.0.0` (253,289 B) | libgomp 8.5.0-28.el8_10.alma.1 (`sboms/auditwheel.cdx.json`) | GPL-3.0+ with GCC Runtime Library Exception 3.1 | — | Grafted by auditwheel. The exception allows redistribution in proprietary or permissive works, so it is OSS and allows commercial use |

Every vendored library is OSI-licensed and allows commercial use, so there is no eligibility issue. Release hygiene: binaries and sources are redistributed **without their LICENSE or NOTICE texts** (F3).

**2d. Vendored third-party model code in the tree**

| Code | Location | Upstream | Licence | Attribution in files |
|---|---|---|---|---|
| MedicalNet (ResNet/ResNeXt 3D, `generate_model`, `setting.py`) | `1_initial_training/models/medicalnet/*`, `src/medicalnet/*` (incl. team-added `attn_resnext.py`), `src/models/medicalnet/*` | Tencent/MedicalNet | MIT (GitHub) | None in headers. `setting.py` L1-4 says "Written by Whalechen" (MedicalNet author). Chinese log strings are kept (`model.py` L87/115). README L240 credits the URL |
| 3D-ResNets ResNeXt backbone | `.../medicalnet/resnext.py` | kenshohara/3D-ResNets-PyTorch | MIT (GitHub) | None in headers. README L241 credits the URL |
| MedNeXt | `1_initial_training/models/mednext/*`, `src/mednext/*`, `src/models/mednext/*` | MIC-DKFZ/MedNeXt | Apache-2.0 (GitHub) | No headers, no NOTICE, no statement of changes. README L242 credits the URL |

All are permissive and compatible with MIT. Copyright and licence texts must be added before release (MIT requires the notice; Apache-2.0 §4 requires the licence copy plus a modification notice) (F3). These modules belong to Block 1 (teacher models). Per `solution_overview.md`, the scored Siamese path does not use them.

### 3. Pretrained models and external data (facts for the pending cross-winner ruling)

| Item | Where used | Weight / data licence | Pretraining-data lineage | Notes |
|---|---|---|---|---|
| HF `edadaltocg/resnet50_cifar100` | Block-1 teacher only (`train_2d_resnet.py` L128-138, `strict=True` load). Reaches the final model only via soft labels on synthetic PET | **MIT** (HF card) | CIFAR-100 (card: "Training data is cifar100"). The init source is not stated; the card shows 300 epochs and an SGD recipe, which suggests training from scratch, but that is unconfirmed | CIFAR-100 has no formal licence. It is a subset of MIT's 80 Million Tiny Images, which its creators withdrew in 2020 |
| HF `edadaltocg/vit_base_patch16_224_in21k_ft_cifar100` | Block-1 teacher only (`train_2d_vit.py` L104-106) | **MIT** (HF card) | Card: "fine-tuned from `timm/vit_base_patch16_224.orig_in21k_ft_in1k`". The chain is **ImageNet-21k → ImageNet-1k → CIFAR-100**. ImageNet's terms of access are non-commercial research and education | Falls squarely under the pending ImageNet ruling. The final model distils from it indirectly (teacher → soft labels → student) |
| MedicalNet pretrained weights (Tencent, Google Drive/Weiyun) | README lists them (L239-241). `--pretrain_path` defaults to None (`setting.py` L47) and no config or script found passes it | Repo MIT. The weights carry no separate licence on the GitHub page | 3DSeg-8 / "23 datasets", mixed public medical sets | Use is unverifiable from code (Data Source DS-3). Ask the team |
| Inference-time backbones | `timm.create_model(..., pretrained=False)` / `resnet50(weights=None)` (`src/inference_2d_avg.py` L29, L77-79). `inference_2d_rnn.py` L36 defaults `pretrained=True`, but `_instantiate_model` passes False (L172) | — | — | No hub download at inference |
| Synthetic DaT/FP-CIT PET (Won et al., *Cell Rep Med* 2025) | Final Siamese training (soft-labelled) | Zenodo 15377683 = **CC-BY-4.0**. However, that record is resource_type "Software": a single 27.6 MB `Krying/PD_SSL_ZOO-v1.0.0.zip` GitHub snapshot. The GitHub repo licence is **null** (GitHub API). The samples the team used come from a **OneDrive/SharePoint link (ulsan.ac.kr)** that the README names, and **no licence or terms are stated for them** (README L72) | The HWDAE generator was trained on 1,934 18F-FP-CIT PET images (PubMed/Cell abstract), most likely institutional Korean hospital data. Provenance was not confirmed (the paper full text returned 403) | Q2. Data Source owns the final call |

### 4. Disqualifier and cheating scan
- None found. The coordinator and Reviewer 1 established that 0 of 524 test uids appear anywhere and that the scored CSV checks pass. Grep found no reference to `test_labels`, no `requests`/`urllib`/socket calls on the inference path, and no upload code. `scripts/download_weights.py` L53-54 pulls `dinhviettoanle/dat-parkinson-models` from HF; that is a dev-time convenience and not on the inference path (Release Readiness owns host visibility; not fetched).
- The only leaderboard-feedback artifact is the global temperature factor "1.245 = offset correction measured on the leaderboard" (`submission_src/validate_pipeline.py` L125, L698; identical in the harness). It is a single scalar, not per-case test information. Code Integrity (CI-2) owns it, and it is not a disqualifier on its own.
- AI tools (GR-A): no mention of ChatGPT, Codex, Copilot or similar in any `.py`, `.md` or `.yaml` (grep). Nothing either way.

### Findings

- Severity: medium (release blocker, curable; needs the team to act)
  Summary: The solution is not MIT-licensed. There is no LICENSE file, and the README "License" section declares the materials confidential and forbids sharing, which contradicts the MIT requirement.
  Evidence: `1st Place/README.md` L247-250; no LICENSE at `1st Place/` root (find); the only LICENSE is DrivenData's runtime `competition-sfmn-parkinsons-runtime/LICENSE` L1-3.
  Recommendation: Require the team to add an MIT LICENSE naming both members and to replace README L247-250 before release (CS16 / Release Readiness).

- Severity: medium (policy; needs an organizer ruling, cross-winner Q1)
  Summary: The two Block-1 teacher backbones carry MIT weights, but the ViT lineage is ImageNet-21k → ImageNet-1k (non-commercial terms of access) → CIFAR-100. The ResNet-50 lineage is CIFAR-100 (no formal licence; derived from the withdrawn 80M Tiny Images), with the init undisclosed. Both affect the final model only through soft labels on the synthetic data.
  Evidence: HF cards (MIT; "fine-tuned from timm/vit_base_patch16_224.orig_in21k_ft_in1k"; "Training data is cifar100"); `train_2d_vit.py` L104-106; `train_2d_resnet.py` L128-138; README L243-244.
  Recommendation: Apply the pending cross-winner ImageNet ruling. Ask the organizer to also cover CIFAR-100 and indirect (teacher/distillation) use. Ask the team whether the ResNet-50 was trained from scratch.

- Severity: medium (needs organizer review; Data Source owns)
  Summary: The synthetic PET data the team actually downloaded (OneDrive/SharePoint) has no stated licence. The CC-BY-4.0 Zenodo record is a 27.6 MB software snapshot, and the GitHub repo has no licence. The generator was trained on 1,934 real FP-CIT PET images of likely institutional origin.
  Evidence: Zenodo API record 15377683 (license `cc-by-4.0`, resource_type Software, 1 file of 27,582,930 B); GitHub API `license: null`; README L16-17, L72; PubMed abstract 40580955.
  Recommendation: Data Source and the organizer should confirm whether the samples sit inside the Zenodo zip (and so fall under CC-BY-4.0) or only on OneDrive. If only on OneDrive, ask the team or the authors for written terms. Rule on whether synthetic data derived from non-public clinical images is "freely and publicly available".

- Severity: low (release hygiene)
  Summary: Vendored third-party code and binaries ship without licence or attribution texts. This covers MedicalNet and ResNeXt (MIT) and MedNeXt (Apache-2.0) in the tree, and openTSNE, annoy, ptwt (EUPL-1.2), pywt, pyradiomics, pykwalify, ruamel.yaml, xgboost and libgomp in `src.zip/src/libs`.
  Evidence: §2c and §2d; no LICENSE/COPYING/NOTICE files (find, `notes/src_zip_listing.txt`); `pywt/__init__.py` L6 refers to a LICENSE that is absent.
  Recommendation: Before release, add a THIRD_PARTY_NOTICES and the upstream licence texts. Prefer declaring the libs as dependencies over vendoring wheels in the released code.

- Severity: low
  Summary: The scored GPU run depends on proprietary NVIDIA CUDA, cuDNN and cuBLAS libraries (runtime base image and lock). An OSS-only path exists: inference falls back to CPU, and training picks CPU when no GPU is present.
  Evidence: `runtime/Dockerfile` L2; `uv.lock` nvidia-*/cuda-toolkit entries; `src/inference_2d_avg.py` L72, L138; `4_siamese_training/train_siamese.py` L424-431.
  Recommendation: Treat this as accelerator-only and acceptable, consistent with the 2nd-place treatment and the organizer's GPU runtime.

- Severity: low (info)
  Summary: The vendored xgboost is 3.3.0 while `uv.lock` pins 3.4.1, and the team modified the DrivenData runtime pyproject to add 7 packages. XGBoost and t-SNE are inactive on the scored path (`solution_overview.md` §6), so this is only a reproducibility and documentation note.
  Evidence: `src/libs/xgboost-3.3.0.dist-info/METADATA`; `runtime/pyproject.toml` L57-64.
  Recommendation: Release Readiness to note it. No eligibility impact.

Open questions:
- Q1 (organizer, cross-winner): Are ImageNet-21k/1k- and CIFAR-100-pretrained, MIT-licensed weights acceptable, including when they are used only as teachers whose soft labels train the final model?
- Q2 (organizer / Data Source): Licence of the OneDrive synthetic FP-CIT PET samples, and whether synthetic data from a generator trained on non-public clinical scans satisfies "freely and publicly available".
- Q3 (team): Confirm they will relicense under MIT (both authors). Did Block-1 MedicalNet runs load Tencent pretrained weights? Was `edadaltocg/resnet50_cifar100` trained from scratch?
