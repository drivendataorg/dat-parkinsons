# Third-party notices

This repository holds my solution to the DaT Parkinson's Challenge (team NavAttack). My own code is under the MIT licence in [LICENSE](LICENSE). The repository, the released model files and the Python environments that run them also contain third-party components. Each component keeps its own licence, and this file lists them.

## 1. TabICL 2.2.0, bundled as inference/tabicl_lib.zip

| Item | Value |
|---|---|
| Component | tabicl Python package, version 2.2.0 |
| Source | https://github.com/soda-inria/tabicl and https://pypi.org/project/tabicl/2.2.0/ |
| Location | `inference/tabicl_lib.zip` (91 files). `inference/main.py` adds the zip to `sys.path`. |
| Licence | BSD-3-Clause, "Copyright (c) 2025, Soda team @ Inria" |
| Licence text | [`inference/tabicl_licenses/LICENSE`](inference/tabicl_licenses/LICENSE), identical to the licence file in the tabicl 2.2.0 wheel |
| Changes | None. Every file in the zip matches the hash recorded for it in the RECORD file of the PyPI 2.2.0 wheel. The zip leaves out one file of the wheel, `prior/graph_lib/README.md`. |

The licence file states that the code in the package's forecast directory derives from TabPFN-TS (https://github.com/PriorLabs/tabpfn-time-series) and falls under the Apache License 2.0. The licence file names that directory `src/tabicl/forecast`, which is `tabicl/forecast/` inside the zip. The Apache appendix in the same file carries the line "Copyright 2025 Prior Labs GmbH".

The zip holds no licence file of its own. Any copy of `inference/tabicl_lib.zip` therefore needs `inference/tabicl_licenses/LICENSE` next to it.

Source-file headers inside the package credit further upstream code, and I left these headers unchanged. According to those headers, the TabICL authors:

- adapted `tabicl/_sklearn/sklearn_utils.py` from scikit-learn (`sklearn/utils/validation.py`),
- adapted `tabicl/_sklearn/preprocessing.py` from yandex-research/tabular-dl-tabr,
- modified `tabicl/forecast/_ts_dataframe.py` from AutoGluon (`autogluon/timeseries/ts_dataframe.py`),
- took `tabicl/train/_muon.py` from fabian-sp/sda, which copied it from a Megatron-LM fork.

## 2. TabICLv2 classifier checkpoint

| Item | Value |
|---|---|
| File | `inference/assets/tabicl-classifier-v2-20260212.ckpt`. `scripts/install_weights.py` installs it. Git does not track it. |
| Source | https://huggingface.co/jingang/TabICL, revision `4dcd344ece2c00be9e831fdd35bed57b5ad83e19` |
| Size and hash | 110,368,038 bytes, sha256 `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0`, equal to the hash of the file on Hugging Face |
| Licence | BSD-3-Clause, declared as `license: bsd-3-clause` in the model card. [`inference/TABICL_MODEL_CARD.md`](inference/TABICL_MODEL_CARD.md) is a copy of that card. |
| Changes | None. The inference code loads the checkpoint frozen, with `allow_auto_download=False`. |

The model card names no copyright holder of its own. For research use, it asks users to cite both papers:

- Qu, J., Holzmüller, D., Varoquaux, G. and Le Morvan, M. TabICL: A Tabular Foundation Model for In-Context Learning on Large Data. International Conference on Machine Learning, 2025. https://arxiv.org/abs/2502.05564
- Qu, J., Holzmüller, D., Varoquaux, G. and Le Morvan, M. TabICLv2: A better, faster, scalable, and open tabular foundation model. arXiv preprint arXiv:2602.11139, 2026. https://arxiv.org/abs/2602.11139

The BibTeX entries are in `inference/TABICL_MODEL_CARD.md`.

Two further asset files come out of this checkpoint: `inference/assets/tabular_raw.pkl.xz` and `inference/assets/tabular_log.pkl.xz`. They hold the frozen TabICL contexts that the checkpoint computed from features of the competition training scans. Section 5 covers the data in them.

## 3. timm ImageNet weights and the fine-tuned 2D models

| Backbone | Source | Original file | Licence | Pretraining data | Fine-tuned files in this solution |
|---|---|---|---|---|---|
| `efficientnet_b0.ra_in1k` | https://huggingface.co/timm/efficientnet_b0.ra_in1k | `model.safetensors`, sha256 `d569899762ea9b1384ee07f4af64805cf8caa1c55f9253ebb1080dc40e87a2cd` | Apache-2.0 (model card) | ImageNet-1k | 12 TorchScript files: `e2dmipf2_fold0` to `fold4`, `e2dmipf2_full0`, `e2dmipf22_fold0` to `fold4`, `e2dmipf22_full0` (`.ts.pt`) |
| `efficientnet_b3.ra2_in1k` | https://huggingface.co/timm/efficientnet_b3.ra2_in1k | `model.safetensors`, sha256 `279d2a53898aa89dab43fd6bd7df9f706aea4cb9cf916223988bbcb8e5850469` | Apache-2.0 (model card) | ImageNet-1k | 6 TorchScript files: `e2db3f2_fold0` to `fold4`, `e2db3f2_full0` (`.ts.pt`) |

`src/cnn2d.py` builds each backbone with `timm.create_model(..., pretrained=True)`. It then fine-tunes the network on the competition training scans. `tools/fetch_backbones.py` downloads the two weight files through the same call. It checks them against the sha256 values above. The repository holds no copy of the original ImageNet weights.

The 18 TorchScript files are derivative works of these Apache-2.0 weights. I modified the weights by fine-tuning the networks on the challenge training scans for 30 epochs and exporting the result to TorchScript with `tools/ts_export.py`. At inference, `inference/inference_main.py` loads the TorchScript files with `torch.jit.load` and does not import timm.

The fine-tuned files exist in two forms. The scored competition archive (platform submission 322397) holds them as exported. The released weights bundle holds copies from which I removed the `*.debug_pkl` records, which contained absolute source paths from the machine that exported them. The scrubbed copies give bit-identical outputs. The released bundle is `navattack_v17_weights.zip` (see `models/README.md`).

Section 4(a) of the Apache License 2.0 requires anyone who redistributes the work or a derivative to give recipients a copy of the licence. The full text is at https://www.apache.org/licenses/LICENSE-2.0. The released weights bundle includes that text as `licenses/timm-Apache-2.0-LICENSE.txt` (the LICENSE file of the timm 1.0.28 distribution), and the TabICL licence as `licenses/TabICL-LICENSE.txt`. It also carries the MIT licence of this solution as `LICENSE`, which covers the fine-tuned weights.

The timm authors pretrained both backbones on ImageNet-1k. ImageNet's terms of access (https://image-net.org/download.php) limit use of the image database to non-commercial research and educational purposes. This project never downloaded ImageNet images and uses only the Apache-2.0 weights.

The timm library itself (version 1.0.28, Apache-2.0, see section 4) runs during training and TorchScript export only.

## 4. Python packages

### Where the licence data comes from

I read the licence fields straight from the `METADATA` file of each installed package: the `License-Expression` field, the `License` field and the `License ::` trove classifiers. The table reports what those fields say and nothing else. Both environments are Python 3.12.11 virtual environments on the development machine (Apple M3 Pro, macOS, arm64), so every wheel is a macOS arm64 build. Linux x86_64 wheels of the same versions can bundle different binary libraries.

- **Inference environment**: the pins in [`requirements-inference.txt`](requirements-inference.txt). That file pins the versions the DrivenData runtime printed during the competition and takes the rest from the runtime repository's lock file. It resolves to 32 packages. On arm64 it installs scipy 1.17.1, because the file pins scipy 1.15.3 for x86_64 only.
- **Development environment**: the frozen package list of the virtual environment I used for training, plus psutil, which that list lacked. It resolves to 67 packages. I ran the unmodified v17 archive in this environment on a 150-scan local check. It reproduced the recorded predictions to a maximum absolute difference of 4.0e-6. The list is `requirements-train.txt`.

The "Imported by" column names the code in this repository that imports the package directly (`inference/`, `src/`, `tools/`, `tabular/`). "tabicl" marks packages that the bundled TabICL package needs on the v17 inference path. "not imported by repository code" marks dependencies of other packages and packages left over from experiments. An empty licence field appears as "(empty)", and a missing classifier as "(none)". Where the two environments hold different versions with different metadata, the cell gives both, prefixed by version.

| Package | Inference env | Development env | Imported by | License-Expression or License field | Licence classifier |
|---|---|---|---|---|---|
| annotated-doc | not installed | 0.0.4 | not imported by repository code | License-Expression: `MIT` | (none) |
| anyio | not installed | 4.14.2 | not imported by repository code | License-Expression: `MIT` | (none) |
| certifi | 2026.7.22 | 2026.7.22 | not imported by repository code | License: `MPL-2.0` | Mozilla Public License 2.0 (MPL 2.0) |
| charset-normalizer | 3.5.1 | not installed | not imported by repository code | License: `MIT` | (none) |
| click | not installed | 8.4.2 | not imported by repository code | License-Expression: `BSD-3-Clause` | (none) |
| contourpy | not installed | 1.3.3 | not imported by repository code | License field holds a licence text headed "BSD 3-Clause License" | BSD License |
| cycler | not installed | 0.12.1 | not imported by repository code | License field holds the cycler licence text (BSD 3-clause wording) | BSD License |
| Deprecated | not installed | 1.3.1 | not imported by repository code | License: `MIT` | MIT License |
| einops | 0.8.2 | 0.8.2 | inference (tabicl) | License: `MIT` | MIT License |
| filelock | 4.0.3 | 3.32.0 | not imported by repository code | License-Expression: `MIT` | MIT License |
| fonttools | not installed | 4.63.0 | not imported by repository code | License: `MIT` | (none) |
| fsspec | 2026.9.0 | 2026.6.0 | not imported by repository code | License-Expression: `BSD-3-Clause` | (none) |
| h11 | not installed | 0.16.0 | not imported by repository code | License: `MIT` | MIT License |
| hf-xet | 1.6.0 | 1.5.2 | not imported by repository code | License-Expression: `Apache-2.0` | Apache Software License |
| httpcore | not installed | 1.0.9 | not imported by repository code | License-Expression: `BSD-3-Clause` | BSD License |
| httpx | not installed | 0.28.1 | not imported by repository code | License: `BSD-3-Clause` | BSD License |
| huggingface_hub | 0.36.2 | 1.24.0 | inference (tabicl, offline), tools/fetch_backbones.py | 0.36.2: License: `Apache`<br>1.24.0: License: `Apache-2.0` | Apache Software License |
| humanize | not installed | 4.16.0 | not imported by repository code | License-Expression: `MIT` | (none) |
| idna | 3.20 | 3.18 | not imported by repository code | License-Expression: `BSD-3-Clause` | (none) |
| ImageIO | not installed | 2.37.4 | not imported by repository code | License-Expression: `BSD-2-Clause` | (none) |
| jaxtyping | not installed | 0.3.11 | not imported by repository code | License field holds a licence text headed "MIT License" | MIT License |
| Jinja2 | 3.1.6 | 3.1.6 | not imported by repository code | (empty) | BSD License |
| joblib | 1.5.3 | 1.5.3 | training | License-Expression: `BSD-3-Clause` | (none) |
| kiwisolver | not installed | 1.5.0 | not imported by repository code | License field holds "The Kiwi licensing terms" (Modified BSD License) | BSD License |
| lazy-loader | not installed | 0.5 | not imported by repository code | License-Expression: `BSD-3-Clause` | (none) |
| lightgbm | not installed | 4.7.0 | training (src/train_final3.py) | License-Expression: `MIT` | (none) |
| loguru | 0.7.3 | 0.7.3 | inference, training | (empty) | MIT License |
| markdown-it-py | not installed | 4.2.0 | not imported by repository code | (empty) | MIT License |
| MarkupSafe | 3.0.3 | 3.0.3 | not imported by repository code | License-Expression: `BSD-3-Clause` | (none) |
| matplotlib | not installed | 3.11.1 | not imported by repository code | License field holds "License agreement for matplotlib versions 1.3.0 and later" | Python Software Foundation License |
| mdurl | not installed | 0.1.2 | not imported by repository code | (empty) | MIT License |
| monai | not installed | 1.6.0 | src/nets3.py, MedicalNet experiment only | License: `Apache License 2.0` | (none) |
| mpmath | 1.3.0 | 1.3.0 | not imported by repository code | License: `BSD` | BSD License |
| narwhals | not installed | 2.24.0 | not imported by repository code | License-Expression: `MIT` | (none) |
| networkx | 3.7 | 3.6.1 | not imported by repository code | License-Expression: `BSD-3-Clause` | (none) |
| nibabel | 5.4.2 | 5.4.2 | inference, training | License-Expression: `MIT` | (none) |
| numpy | 2.2.6 | 2.5.1 | inference, training | 2.2.6: License field holds the NumPy licence text (BSD 3-clause wording)<br>2.5.1: License-Expression: `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0` | 2.2.6: BSD License<br>2.5.1: (none) |
| packaging | 26.3 | 26.2 | not imported by repository code | License-Expression: `Apache-2.0 OR BSD-2-Clause` | (none) |
| pandas | 3.0.3 | 3.0.5 | inference, training | License field holds a licence text headed "BSD 3-Clause License" | BSD License |
| pillow | not installed | 12.3.0 | not imported by repository code | License-Expression: `MIT-CMU` | (none) |
| psutil | 7.2.2 | 7.2.2 | inference (main.py, tabicl) | License: `BSD-3-Clause` | (none) |
| Pygments | not installed | 2.20.0 | not imported by repository code | License-Expression: `BSD-2-Clause` | (none) |
| pyparsing | not installed | 3.3.2 | not imported by repository code | License-Expression: `MIT` | (none) |
| python-dateutil | 2.9.0.post0 | 2.9.0.post0 | not imported by repository code | License: `Dual License` | BSD License, Apache Software License |
| PyYAML | 6.0.3 | 6.0.3 | not imported by repository code | License: `MIT` | MIT License |
| requests | 2.34.2 | not installed | not imported by repository code | License: `Apache-2.0` | Apache Software License |
| rich | not installed | 15.0.0 | not imported by repository code | License: `MIT` | MIT License |
| safetensors | not installed | 0.8.0 | not imported by repository code | (empty) | Apache Software License |
| scikit-image | not installed | 0.26.0 | not imported by repository code | License field lists "License: BSD-3-Clause" for all files | BSD License |
| scikit-learn | 1.8.0 | 1.9.0 | inference, training | License-Expression: `BSD-3-Clause` | (none) |
| scipy | 1.17.1 | 1.18.0 | inference, training | License field holds the SciPy licence text (BSD 3-clause wording) | BSD License |
| setuptools | 81.0.0 | 83.0.0 | not imported by repository code | License-Expression: `MIT` | (none) |
| shellingham | not installed | 1.5.4 | not imported by repository code | License: `ISC License` | ISC License (ISCL) |
| simpleitk | not installed | 2.5.5 | not imported by repository code | License: `Apache` | Apache Software License |
| six | 1.17.0 | 1.17.0 | not imported by repository code | License: `MIT` | MIT License |
| surfa | not installed | 0.6.3 | not imported by repository code | (empty) | MIT License |
| sympy | 1.14.0 | 1.14.0 | not imported by repository code | License: `BSD` | BSD License |
| threadpoolctl | 3.6.0 | 3.6.0 | dependency of scikit-learn and tabicl | License: `BSD-3-Clause` | BSD License |
| tifffile | not installed | 2026.7.14 | not imported by repository code | License: `BSD-3-Clause` | (none) |
| timm | not installed | 1.0.28 | training, tools/ts_export.py, tools/fetch_backbones.py | License: `Apache-2.0` | Apache Software License |
| torch | 2.12.1 | 2.13.0 | inference, training | 2.12.1: License: `BSD-3-Clause`<br>2.13.0: License-Expression: `Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT` | (none) |
| torchio | not installed | 1.2.1 | not imported by repository code | License-Expression: `Apache-2.0` | (none) |
| torchvision | not installed | 0.28.0 | not imported by repository code | License: `BSD` | (none) |
| tqdm | 4.67.3 | 4.69.0 | inference (tabicl) | License: `MPL-2.0 AND MIT` | (none) |
| typer | not installed | 0.27.0 | not imported by repository code | License-Expression: `MIT` | (none) |
| typing_extensions | 4.16.0 | 4.16.0 | not imported by repository code | License-Expression: `PSF-2.0` | (none) |
| urllib3 | 2.8.0 | not installed | not imported by repository code | License-Expression: `MIT` | (none) |
| wadler_lindig | not installed | 0.1.7 | not imported by repository code | License field holds the Apache License 2.0 text | Apache Software License |
| wrapt | not installed | 2.2.2 | not imported by repository code | License-Expression: `BSD-2-Clause` | (none) |
| xxhash | not installed | 4.0.1 | not imported by repository code | License: `BSD-2-Clause` | (none) |

### Binary libraries inside the numpy and scipy wheels

The `LICENSE.txt` files of the numpy and scipy wheels list binary libraries that the wheels bundle. For the macOS arm64 wheels installed here they list:

| Bundled library | Wheels | Licence as listed |
|---|---|---|
| OpenBLAS | numpy 2.2.6 and 2.5.1, scipy 1.17.1 and 1.18.0 | BSD-3-Clause |
| LAPACK (inside the OpenBLAS library) | numpy 2.5.1, scipy 1.17.1 and 1.18.0 | BSD-3-Clause-Open-MPI |
| LAPACK (inside the OpenBLAS library) | numpy 2.2.6 | BSD-3-Clause-Attribution |
| GCC runtime library (libgfortran, libgcc) | numpy 2.5.1, scipy 1.17.1 and 1.18.0 | GPL-3.0-or-later WITH GCC-exception-3.1 |
| GCC runtime library (libgfortran, libgcc) | numpy 2.2.6 | GPL-3.0-with-GCC-exception |
| libquadmath | numpy 2.2.6 and 2.5.1, scipy 1.17.1 and 1.18.0 | LGPL-2.1-or-later |

### PyTorch

The torch 2.12.1 wheel declares `License: BSD-3-Clause`, and its dist-info folder carries a `NOTICE` file that lists the third-party software in PyTorch. The torch 2.13.0 wheel declares the compound `License-Expression` shown in the table and ships a licence file for each third-party component.

### NVIDIA CUDA libraries

The DrivenData runtime ran the CUDA 12.9 build of torch 2.12.1 (`torch==2.12.1+cu129`, recorded in the header of `requirements-inference.txt`). The NVIDIA CUDA libraries that come with that build are absent from both environments in the table, and I did not read their licence terms. They are optional. The inference code falls back to the CPU. On the Apple M3 Pro CPU the unmodified v17 archive took between 530 s and 17 minutes for 150 scans in three runs.

### Packages missing from both environments

- CatBoost: `tabular/_original/bench_v16.py`, an archived experiment script, imports it. No code under `inference/` or `src/` imports it. Neither environment installs it, so I did not read its licence metadata.
- Some modules of the bundled tabicl package import further packages, for example shapiq, statsmodels, xgboost, transformers, wandb, shap and numba. The development environment installs none of them, and the v17 predictions reproduced there without them.

## 5. Competition data

The challenge organisers (the French Society of Nuclear Medicine, SFMN, with the Health Data Hub and GaelO) provided the training data through DrivenData. Every model I trained or fine-tuned for this solution learned from that data. This repository does not contain the data: `.gitignore` excludes `data/`, `outputs/` and all scan, table and weight files. Two asset files also hold per-scan values derived from the training scans: `inference/assets/tabular_raw.pkl.xz` and `inference/assets/tabular_log.pkl.xz`. These frozen TabICL contexts cover all 1,362 training scans (raw context) and an acquisition-balanced subset of 1,148 of them (log context). For every context scan they store eight 512-dimensional TabICL row representations with the scan's label embedding added, so the training labels can be recovered from them. They also store per-feature statistics of the 72 inputs: medians, means, scales, outlier bounds, minima and maxima. They contain no identifiers, images or per-scan feature rows. They are distributed to DrivenData with the weights bundle. A public release of these two files waits for the organisers' agreement.
