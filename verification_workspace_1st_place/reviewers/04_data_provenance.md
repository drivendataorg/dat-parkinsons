## Subagent: Reviewer 4 — Data Source and Model Provenance (1st place, TheAvengers)

Status: needs organizer review

No competitor code executed: yes. No images viewed. No artifacts loaded. No competitor-hosted URLs fetched.

Path conventions: TREE/ = `1st Place/`; HZ/ = `extracted/harness_zip/`; SZ/ = `extracted/src_zip/src/`.

## A. Static inventory (code-derived)

| # | Item | Type | Loaded where (file:line) | Reaches scored model? | Disclosed in README? |
|---|---|---|---|---|---|
| E1 | Synthetic FP-CIT DaT PET (Won et al.) — `<root>/external/PET_GENERATED/*.nii.gz` | External dataset (manual download) | `0_configs/default.yaml:14`; globbed `3_soft_labeling/soft_labeling.py:24,217`; patches `4_siamese_training/prepare_2d_dataset_siamese.py:80-83`, `prepare_2d_dataset_striatum.py:68-71`; concatenated into train set `4_siamese_training/train_siamese.py:246-262` (gated by `USE_SYNTHETIC = not args.no_synth`, L36) | **Yes, directly** (training inputs of all 100 Siamese models, labels = teacher soft labels) | Yes: README L16-17, L71-83, L183, L242 (paper, GitHub/OneDrive, Zenodo) |
| P1 | HF `edadaltocg/resnet50_cifar100` `model.safetensors` | Pretrained weights (hf_hub_download at train time) | `1_initial_training/train_2d_resnet.py:15,107-137` (torchvision resnet50 `weights=None`, then HF state_dict strict=True) | Indirect (ResNet2D teacher → stacking → soft labels) | Yes, README L243 |
| P2 | HF `edadaltocg/vit_base_patch16_224_in21k_ft_cifar100` `pytorch_model.bin` | Pretrained weights (hf_hub_download) | `1_initial_training/train_2d_vit.py:7,96-117` (timm `pretrained=False`, then HF state_dict strict=False) | Indirect (Vit2D teacher) | Yes, README L244 |
| P3 | Tencent MedicalNet pretrained weights | Pretrained weights (optional) | `src/models/medicalnet/setting.py:47` `--pretrain_path default=None`; loaded only if set (`src/models/medicalnet/model.py:86-114`). `5_pipeline/run_pipeline.py:43-49` MODEL_TO_CMD never passes `--pretrain_path`; `train_medicalnet.py` builds via `src.models.builder.build_model_medicalnet` → `parse_opts()` defaults. | **Delivered code: none loaded (random init).** Whether original run used them: unknown | README L239 lists MedicalNet repo under "Resources" (architecture or weights not stated) |
| C1 | MedicalNet / 3D-ResNets-PyTorch / MedNeXt / MONAI DenseNet121/201 | Architecture code only | `1_initial_training/models/*`, `train_3d_*.py:16-17,183-186,265` (MONAI, no pretrained arg) | Indirect | Yes (L239-241); MONAI not listed (library) |
| A1 | HF repo `dinhviettoanle/dat-parkinson-models` | Competitor-hosted artifact host | `scripts/download_weights.py:9,53-56` (snapshot_download, no token arg) | Optional convenience only; not used in inference | Yes, README L85-96 (host name not in README) |
| L1 | Inference-time models | timm/torchvision constructs | all `pretrained=False` / `weights=None` at instantiation (e.g. `SZ/inference_2d_rnn.py:171-176`, `SZ/inference_2d_avg.py:29,77-79`); default `pretrained=True` in `inference_2d_rnn.py:36` is overridden | No download at inference | n/a |

Pretrained weights for Block-2 Siamese: none (`train_siamese.py:174-181` `create_backbone()` is a from-scratch 4-layer CNN; ImageNet mean/std normalization constants L161 only, no weights). README L245 "Block 2: no additional used resources" is consistent.

Dead options noted: `train_siamese.py:98-99,241-244` `USE_RELABEL=False`, `RELABEL_CSV_PATH=None` (column `is_pathologic_fixed_soft` — suggests an earlier experiment with relabelled/softened *training* labels; inactive; no such CSV delivered). `SYNTH_LABELS=None` (L65) override inactive.

## B. Network calls / third-party uploads (static grep)

Grep over `1st Place/` and `extracted/` (excl. vendored `libs/`, `uv.lock`, dist-info) for requests/urllib/http.client/huggingface_hub/hf_hub/snapshot_download/boto3/google.cloud/gcsfs/s3fs/wandb/mlflow/comet/neptune/tensorboard/openai/anthropic/gemini/mistral/cohere/ollama/api_key/HF_TOKEN/socket/wget/curl/kaggle/gdown/onedrive/zenodo/torch.hub/load_state_dict_from_url:
- Only hits: `train_2d_resnet.py:15,128` and `train_2d_vit.py:7,104` (hf_hub_download of public pretrained weights = **download only, no upload**), `scripts/download_weights.py:9,53` (download only). Runtime `pyproject.toml:18,56` lists `huggingface-hub`, `urllib3` (packages only; runtime test file imports them).
- **No network call in inference code** (HZ/*.py, SZ/*.py): no hits. Inference models instantiated with `pretrained=False`.
- No experiment trackers, no cloud storage, no hosted LLM API clients, no API keys/tokens anywhere.
- Runtime `ci.yml` (Azure login/docker push) is DrivenData's runtime-repo CI, not competitor tooling; `6_runtime/pack_production.py:90-91` uses `uvx rpzip` locally; no upload.
- Root `1st Place/pyproject.toml` does not list `huggingface_hub`/`safetensors` though training scripts import them (transitive via timm) — reproducibility nit only.

## C. Privacy (NIfTI headers)
Only header access: `img.header.get_zooms()[:3]` (voxel spacing) at `src/preprocessing.py:193` (= SZ same line) and `1_initial_training/train_medicalnet.py:128`. No reads of `descrip`, `db_name`, `aux_file`, `intent_name`, extensions, or any DICOM/pydicom use. No cross-UID linkage code. **Pass.**

## D. AI tools (Rule A)
Grep for chatgpt/codex/copilot/cursor/claude/gemini/llm/openai/anthropic/"generated by"; search for `.cursor*`, `CLAUDE.md`, `AGENTS.md`, `.aider*`, `.copilot*` in tree, harness root and src.zip listing: **no hits**. README silent on AI tools. Code style (French comments, emoji log banners, banner-style section headers) is not evidence either way. → Open question for the Winning Model Documentation / team.

## E. Annotations (Rule F)
No manual annotation files of competition data in tree or harness. Striatum localisation is algorithmic (`src/striatum_*.py`). Synthetic data labels are model-generated soft labels (not manual, and not on competition data). Training-label "relabel" option inactive (see A). No annotation of test data found. **Pass (with note on the inactive relabel option).**

## F. Synthetic DaT PET (E1) — provenance and license (public pages only)

Facts from public pages (URLs in §J):
- Paper: Won J, Lee GY, Jo S, … Chung SJ, Kim N. "Enhancing 3D dopamine transporter imaging as a biomarker for Parkinson's disease via self-supervised learning with diffusion models", *Cell Reports Medicine* 6(7):102207, 2025 (PMID 40580955, PMCID PMC12281361). Article itself is "open access … under the CC BY-NC license" (this licenses the article text, not necessarily the released images).
- **Generator training data:** HWDAE/WDDAE trained on 1,934 18F-FP-CIT PET images selected from 2,672 images / 2,600 patients at **Asan Medical Center (Seoul, Korea), Jan 2005–Mar 2022**, retrospective, IRB 2022-0614, consent waived; external validation at Soonchunhyang University Seoul Hospital (Dec 2016–Mar 2024). **Not PPMI.** Paper: "The original image data utilized in this study are not publicly accessible owing to patient privacy concerns. However, data reported in this paper will be shared by the lead contact upon request with institutional review board approval."
- Paper data-availability: "Synthesized images, model weights, and code are available at GitHub and Zenodo: https://github.com/Krying/PD_SSL_ZOO and https://doi.org/10.5281/zenodo.15377683".
- **Zenodo 15377683** = auto-archive of GitHub release v1.0.0, resource type *Software*, **license CC-BY-4.0, access open**, one file `Krying/PD_SSL_ZOO-v1.0.0.zip` (27,582,930 bytes). GitHub tree (metadata via `gh api`, nothing downloaded) shows that release contains only **3** synthetic volumes (`3_RECONSTRUCTION/DATA/Generated_sample_{1,2,3}.nii.gz`, 6.9–10.3 MB).
- **GitHub Krying/PD_SSL_ZOO:** public, **no LICENSE file** (`license: null`). README links two OneDrive folders on a personal institutional account ("pretrained upstream model weights" and "Synthetic FP-CIT-PET Samples"); **no license, terms, or access-request statement** for the OneDrive set. Not fetched (competitor-adjacent / data host; disallowed).
- The competitor's README directs users to the **OneDrive "Synthetic FP-CIT-PET Samples"** folder (README L73), i.e. the bulk set used is the OneDrive set, not the 3-sample Zenodo archive. Number of synthetic volumes used is not stated in code/README.

Assessment (facts for organizer; not a ruling):
- Freely/publicly available: appears **yes** — public link, no application/DUA/IRB stated for the synthetic samples (IRB gate applies only to the *original* AMC images).
- License: **ambiguous.** The only explicit license covering any synthetic images is CC-BY-4.0 on the Zenodo software archive (3 samples). The OneDrive set actually used carries **no stated license** (default: all rights reserved), and the paper — which is the vehicle pointing to the data — is CC BY-NC. Rule B requires "a permissive open license that does not prohibit commercial use (e.g., CC0 or CC-BY)". One reading: the authors designate GitHub+Zenodo (CC-BY-4.0) as the release of the "Synthesized images", so CC-BY-4.0 applies; the other: the OneDrive bulk set is unlicensed. Organizer must decide, or ask the team / Won et al. for an explicit license.
- Generator trained on **private, non-public Korean hospital data** (AMC). Rules say pretraining data of models is subject to the same rules; here the generator is not a model used by the competitor, but its outputs are the competitor's training data. Whether synthetic outputs of a model trained on private (non-freely-available) data are permitted is an **organizer question** (cf. overview Q2). No NC/DUA term is attached to the synthetic samples themselves.
- Overlap with competition data: generator data is Korean 18F-FP-CIT **PET** (2005–2022, AMC/SCH); competition data is French **SPECT** (ioflupane) from ten French centres, not public (organizer). **No plausible overlap / no leakage route.**
- Enters the scored model **directly** (training inputs of all 100 Siamese models). README Alt-2 `--no-synth` exists but "does not reproduce our best submission" (README L143-151).

## G. Pretrained models (P1-P3) — license and pretraining data

| Model | License (HF/GitHub) | Pretraining data | Notes |
|---|---|---|---|
| `edadaltocg/resnet50_cifar100` | **MIT** (HF tag) | CIFAR-100 only, per card: 300 epochs SGD lr 0.1 cosine (consistent with from-scratch training); CIFAR-style stem matches `train_2d_resnet.py:107-121` | CIFAR-100 has no formal license (Krizhevsky; drawn from the withdrawn 80M Tiny Images). No ImageNet in lineage per card. |
| `edadaltocg/vit_base_patch16_224_in21k_ft_cifar100` | **MIT** (HF tag) | base_model `timm/vit_base_patch16_224.orig_in21k_ft_in1k` (**Apache-2.0**; Google ViT, **ImageNet-21k pretrain + ImageNet-1k fine-tune**), then CIFAR-100 fine-tune | Repo name says "in21k" but the base is in21k→**in1k**; ImageNet-1k terms are non-commercial research-only ⇒ falls squarely under the **pending cross-winner ImageNet ruling**. |
| Tencent MedicalNet weights | Repo LICENSE: **MIT** (GitHub API reports NOASSERTION, file text is MIT) | 3DSeg-8 (public segmentation datasets, mixed licenses) | **Not loaded by delivered code** (`--pretrain_path` default None, never passed). Original-run usage unknown → ask team. |
| Architecture code: 3D-ResNets-PyTorch (MIT), MedNeXt (Apache-2.0), MONAI (Apache-2.0), timm (Apache-2.0), torchvision (BSD-3) | OSI | n/a | No weights from these. |

Influence path: P1/P2 are 2 of 12 Block-1 teachers (`default.yaml:106-118`) → stacking meta-model → `synthetic_soft_labels.csv` → targets for synthetic samples in Block-2. They are **not** in the scored inference graph; their knowledge reaches the final model only via soft labels (Q4). README Alt-1 (ResNet2D + MedicalNet50 + SVM) still includes P1; a P1/P2-free reproduction (e.g. MedicalNet50+SVM only, or `--no-synth`) is not documented with scores.

## H. Disclosure cross-check (CS7)
- README "External data" + "Resources" disclose E1 (paper, GitHub/OneDrive, Zenodo), P1, P2, MedicalNet, 3D-ResNets, MedNeXt. Everything the code loads is disclosed. **Pass**, with notes: (a) README cites Zenodo as "the synthetic dataset" (L183, L242) though Zenodo holds only 3 samples; the data actually comes from OneDrive; (b) MedicalNet listed but delivered code uses no MedicalNet weights — clarify; (c) ViT lineage includes ImageNet-1k, not stated in README; (d) download host `dinhviettoanle/dat-parkinson-models` not named in README (only in script).
- **Limitation:** Winning Model Documentation (template) is not in the tree; disclosure could only be checked against README.

## I. `scripts/download_weights.py` (A1) — from code only (not fetched)
`snapshot_download(repo_id="dinhviettoanle/dat-parkinson-models")` (L53-56), **no token argument** (suggests the repo was meant to be downloadable without auth, i.e. possibly public; could also rely on a cached `huggingface-cli login`). Expected layout:
- `[models/]01-soft_labeling/`: `SVM.pkl`, `tabular_oof.csv` (L74-79), `synthetic_soft_labels.csv` (L81-84), and one subfolder per teacher (copied wholesale, L86-105; e.g. `mednext-B/` with `oof_mednext-B.csv` + `mednext-B_fold_*.pth`) ⇒ teacher weights + per-teacher **OOF CSVs on training uids** (label-bearing, competition-derived).
- `[models/]02-siamese/`: the Siamese `*.pth` (L110-115).
All are trained on/derived from competition data. Visibility **unknown** (deliberately not checked). Rule E / release ruling: weights may only be held privately with organizer-only access → hand to Release Readiness (RR-2). Synthetic soft labels contain no competition data directly but are a function of models trained on it.

## J. URLs fetched (public third-party docs only; nothing downloaded; no competition content sent)
1. https://zenodo.org/records/15377683 (record page)
2. https://zenodo.org/api/records/15377683 (record metadata JSON; file list only, file not downloaded)
3. https://github.com/Krying/PD_SSL_ZOO and https://raw.githubusercontent.com/Krying/PD_SSL_ZOO/main/README.md
4. `gh api repos/Krying/PD_SSL_ZOO`, `.../contents`, `.../git/trees/v1.0.0?recursive=1` (metadata only)
5. WebSearch for the paper title; https://pubmed.ncbi.nlm.nih.gov/40580955/ (cookie wall); https://www.cell.com/cell-reports-medicine/fulltext/S2666-3791(25)00280-0 (403); https://www.sciencedirect.com/science/article/pii/S2666379125002800 (403)
6. https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:40580955%20AND%20SRC:MED&resultType=core&format=json ; https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12281361/fullTextXML
7. https://huggingface.co/edadaltocg/resnet50_cifar100 ; https://huggingface.co/edadaltocg/resnet50_cifar100/raw/main/README.md
8. https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_cifar100 ; https://huggingface.co/timm/vit_base_patch16_224.orig_in21k_ft_in1k
9. `gh api repos/{Tencent/MedicalNet,kenshohara/3D-ResNets-PyTorch,MIC-DKFZ/MedNeXt}` (license field); https://raw.githubusercontent.com/Tencent/MedicalNet/master/LICENSE
Not fetched: OneDrive links, any dinhviettoanle URL/HF repo, any weights or data.

## K. Handoff

No competitor code executed: yes. No images viewed (`img/*.png` not opened). No artifact loaded; `weights.zip` not extracted.

Read-only inspection performed: `1st Place/README.md`, `0_configs/default.yaml`, `scripts/download_weights.py`, `1_initial_training/train_2d_resnet.py`, `train_2d_vit.py`, `train_medicalnet.py`, `train_3d_*.py`, `src/models/{builder.py,medicalnet/setting.py,medicalnet/model.py}`, `3_soft_labeling/soft_labeling.py`, `4_siamese_training/*.py`, `5_pipeline/run_pipeline.py`, `6_runtime/pack_production.py`, `pyproject.toml`, runtime `Dockerfile`/`ci.yml`/`pyproject.toml`; greps over tree + `extracted/` (harness root + src.zip text).

Competition-specific checks applied: CS7, CS8, CS9, CS10, CS11, CS12; CS5/CS6 cross-checked.

Findings:
- Severity: high (needs organizer review)
  Summary: Synthetic PET set actually used (OneDrive "Synthetic FP-CIT-PET Samples") has no stated license; only the 3-sample Zenodo software archive is CC-BY-4.0, GitHub repo has no LICENSE, and the paper is CC BY-NC.
  Evidence: README L73, L183, L242; Zenodo API record (CC-BY-4.0, 27.6 MB zip, Software); gh API `license: null`; Europe PMC PMC12281361 license CC BY-NC.
  Recommendation: Organizer to rule whether the paper's "available at GitHub and Zenodo" + Zenodo CC-BY-4.0 covers the OneDrive set; or ask team / Won et al. for an explicit permissive license. If not resolvable, consider requiring the `--no-synth` variant (score impact unknown).
- Severity: medium (needs organizer review)
  Summary: The synthetic-data generator was trained on private Asan Medical Center patient PET (not public; IRB-gated); synthetic outputs are public. Overview Q2.
  Evidence: PMC12281361 Methods + Data availability.
  Recommendation: Organizer ruling on whether outputs of a generator trained on non-public data qualify as "freely and publicly available" external data.
- Severity: medium (pending cross-winner ruling)
  Summary: ViT teacher weights (MIT) derive from `timm/vit_base_patch16_224.orig_in21k_ft_in1k` (Apache-2.0) = ImageNet-21k + ImageNet-1k; ResNet50 teacher is CIFAR-100 only. Both affect final model only via soft labels.
  Evidence: `train_2d_vit.py:104-117`, `train_2d_resnet.py:128-137`; HF cards.
  Recommendation: Apply the same ImageNet ruling as for other winners; CIFAR-100 (no formal license) to be noted under the same ruling.
- Severity: low
  Summary: MedicalNet listed in README but delivered pipeline never loads Tencent weights (random init); unclear whether the original soft-labelling run used them. Tencent MedicalNet code is MIT; 3DSeg-8 data mixed licenses.
  Evidence: `src/models/medicalnet/setting.py:47`, `model.py:86-114`, `run_pipeline.py:43-49`.
  Recommendation: Ask team to confirm.
- Severity: medium (Release Readiness owner)
  Summary: `download_weights.py` pulls HF `dinhviettoanle/dat-parkinson-models` (no token) containing competition-trained weights, per-teacher OOF CSVs, `tabular_oof.csv`, `synthetic_soft_labels.csv`; visibility unknown.
  Evidence: `scripts/download_weights.py:53-115`.
  Recommendation: Organizer/team to confirm repo is private / organizer-only.
- Severity: low
  Summary: Disclosure complete vs code, but README conflates Zenodo with the dataset, omits ImageNet-1k lineage and the HF weights host; Winning Model Documentation unavailable.
  Recommendation: Request corrected disclosure in Winning Model Documentation.
- Severity: info (pass): no network calls in inference; only training-time downloads of public HF weights; no trackers/cloud/LLM APIs/keys; NIfTI header use limited to `get_zooms()`; no manual annotations; inactive `USE_RELABEL` (training labels) option; no evidence of AI-tool use.

Open questions:
- License of OneDrive synthetic set; count of synthetic volumes used.
- Were MedicalNet pretrained weights used in the original run?
- Did the team use any AI coding assistant that retains data (Rule A)? Not evidenced either way.
- Visibility of `dinhviettoanle/dat-parkinson-models`.
- What was `is_pathologic_fixed_soft` / RELABEL CSV (earlier relabelled training labels)? Inactive in delivered code.
