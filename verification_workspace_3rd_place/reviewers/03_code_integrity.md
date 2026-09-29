# Reviewer 3: Code Integrity and Anti-Cheating (3rd place, paulonium)

**No competitor code executed: yes.** This was a static, read-only review. I did not import any competitor module and did not `torch.load` or unpickle any artifact.

For the TorchScript models, I streamed zip entries to the scratchpad with `unzip -p` and read only the `code/*.py` text entries of `fu_a1_dnet_f0.ts.pt` and `ellipse.ts.pt`. For the UID grep I read raw bytes of all entries. I did not open `test_labels.csv`. I modified nothing under `3rd Place/` or `references/`.

Paths are relative to the repo root. `harness/` means `verification_workspace_3rd_place/extracted/harness_zip/`, and `tree/` means `3rd Place/`.

## 1. Code path map

### Inference (harness zip: the code that was scored)

`harness/main.py:main()` runs these steps:
1. **Read inputs.** It reads `submission_format.csv` and uses only the `uid` column, to build nifti paths and fix the output order (L83-84).
2. **Load models.** It loads the 50 `fu_*.ts.pt` members (sorted glob, `.eval()`, L85-86), then `calibration.json` (L90-91), then `ellipse.ts.pt` (L105).
3. **Preprocess each scan.** A process pool maps `preprocess_safe` one path at a time (L147-152).
   - `preprocess` (L42-62): `nib.load`, then `get_zooms` (the only header field read), iso zoom, a per-scan percentile threshold, the largest connected component, `extract_box`, and `D.normalize`.
   - `D.normalize` is a per-scan masked mean (`datprep_iso.py:76-81`).
4. **Predict in batches of 8** (`predict`, L112-142):
   - The ellipse localiser runs, and each scan's ellipsoid mask is built from its own (c, r, R).
   - `CZ.thetas` loops per scan (`canonize.py:49-66`). It uses `head_span` (per-scan quantile) and the constants in `canon_const.json`.
   - Then `grid_sample` and `with_slabs`. The slabs are constant indicator tensors.
   - Each member runs with flip-TTA of the same scan, and its logit is clamped to ±6.
   - `torch.stack(zs,0).mean(0)` averages over **members**. `zs` has shape (M, B), so this is not a mean over the batch.
   - Finally `sigmoid(a·z+b)`.
5. **Write output.** L158 writes the CSV in `fmt` order, with 0.5 for any scan whose preprocessing failed.
6. **Inactive code.** `glandrm.py` is never used, because there is no `gland_rm` key in `calibration.json`.

### Training (`tree/`)

- **Caches.** `build/build_box_mmap.py` and `build/build_canonv2_cache.py` build caches from `meta/uids.csv`, which holds the 1,362 train UIDs.
- **Members.** `recipes/train_fusion10.sh` runs `python -m datscan --member fu_a{s}_dnet … splits=meta/cv_round{s+1}.csv seed=s` for s = 1..10 and folds 0..4. That gives 50 fold-models. Each checkpoint is the best epoch by validation-fold log loss.
- **Packaging.** `build/export_members.py` exports the TorchScript traces. `build/build_roster.py` then:
  - collects `{member}_f[0-9].ts.pt` (L40);
  - fits the intercept `b` on **training OOF** with `meta/labels.npy` (L45-58);
  - writes `calibration.json` as `{a: slope, b, member_logit_cap}` (L72);
  - copies `main.py`, `canonize.py`, `datprep_*.py`, `glandrm.py`, `ellipse.ts.pt` and `canon_const.json` from `inference/` (L66-69).
- **Localiser.** `build/train_ellipse.py` and `export_ellipse.py` produce `ellipse.ts.pt` from the competitor's hand annotations on training scans (see Reviewer 1, flag F1; that is the Data Source reviewer's area).

## 2. Test leakage

| Check | Method | Result |
|---|---|---|
| Test UIDs in the tree (all 188 files, including the 21 binary files: npy, pyc, png, pt) | `LC_ALL=C grep -a -r -F -f test_uids.txt "3rd Place"`, case-sensitive and `-i` | **0 hits** |
| Test UIDs in the harness `.py` and `.json` files | same grep on `harness/` | **0 hits** |
| Test UIDs in raw bytes of every harness zip entry (58 entries, including all 51 `.ts.pt`) | `unzip -p <entry>` piped to `grep -a -F -c` | **0 hits** |
| Test UIDs in the deflated inner entries of each `.ts.pt` (`data.pkl`, `code/*.py`, `debug_pkl`, constants) | each inner zip extracted to the scratchpad, excluding tensor `data/N` blobs (already covered as stored bytes above), then grep | **0 hits across 51 traces** |
| Grep method works (positive controls) | train UIDs against the tree; test UIDs against the scored CSV | 21 tree files hit; 524/524 CSV rows hit |
| Train/test UID overlap | train_labels vs submission_format | 0 |

Other checks:

- **Labels and splits.** Training labels are `meta/labels.npy`, which Reviewer 1 found equal to `train_labels.csv`. All `cv_round*.csv` files and `splits.csv` contain only train UIDs.
- **Test references in training code.** A grep for `submission_format|test_labels|pseudo|niftis` in tree code finds only three things: the inference entry points, `datscan/submission_main.py` (stale), and `build/verify_pkg.sh`, which builds a local `submission_format.csv` from **train** niftis for smoke tests.
- **No pseudo-labelling, test-time training or test-set caching** was found.

### Calibration

- **Slope a = 0.78.** This is a manual choice. `build/calibrate.py:9-29` says it was guided by two public-leaderboard submissions (a = 0.85 scored 0.2313 public; a = 0.78 scored 0.2317). The ~0.74 "private optimum" is described as "only visible after the close", i.e. inferred after the fact from the two private scores.
  - Nothing fits `a` to test labels. This is ordinary LB-guided hyperparameter selection, and the rules file does not prohibit it.
  - **Classification: not a violation.** It is informational only.
- **Intercept b = −0.16455324638116142 (shipped).** `build_roster.py:58` fits `b` with `minimize_scalar` on training OOF only. The full-precision float is characteristic of that optimiser.
  - The documented value, −0.147824566 (`recipes/fusion10_s078.json:73`, `make_figures.py:262`, README:139), comes from a different OOF run directory. Most plausibly it is the 2026-09-24 retrain: `docs/repro/md5_weights_training_new_dat.txt:16-19` reports calibrate.py run on the retrained runs, and the original-package OOF ll differs.
  - There is no path by which `b` could be fitted on test data. The competitor never had test labels, and the harness does not read labels.
  - **Classification: a provenance and documentation mismatch, not leakage.**
- **Scored CSV matches the shipped b, not the documented one.** The shipped b gives cap values of 0.98917967 and 0.00780966, which match the scored CSV's 49 and 2 repeated values to within 1 float32 ulp. The documented b would give a high cap of 0.98935723, which is **not** present.
- **"TEST-LIKE clusters" and `hard_weight`.** `datscan/config.py:567-569` (HARD_CLUSTERS, used in `data.py:85-93`) defines clusters inferred from leaderboard behaviour over train acquisition clusters. It uses no test data.
  - It is not set in the winning recipe (default 1.0), and it was refuted (`notes/REFUTED.md:183`).
  - It is LB-guided exploration only.

## 3. Per-observation independence (Rule D)

| Item | Evidence | Assessment |
|---|---|---|
| Preprocessing | `harness/main.py:42-62`, `datprep_iso.py:76-81`, `datprep_box.py:13-34`; pool maps one path per call | Per scan only |
| Canonical warp | `canonize.py:28-37,49-66`: loop over j; `head_span` per scan; `TGT`/`REF`/`SH` from the static `canon_const.json` | Per scan; the constants are frozen assets |
| Slabs | `canonize.py:80-86`: a constant tensor cached across calls | Holds no data, so harmless |
| BatchNorm (members) | `fu_a1_dnet_f0.ts.pt` code: 371 `torch.batch_norm(..., running_mean, running_var, False, 0.1, 1e-5, True)` | `training=False` is baked into the trace, so running statistics are used and nothing mixes across the batch |
| BatchNorm (ellipse) | `ellipse.ts.pt`: 8 BatchNorm3d calls, all `training=False` | Same |
| Dropout | 174 `feature_dropout(x, 0.2, False)` | Inactive |
| Reductions in the traced graphs | The projection (`PhysShape3N`) uses `mean(flatten(x,1),[1])` (per-scan τ), `amax(...,[2,3])`, `sum(...,[-1])`, and logsumexp over the last dimension. The ellipse `cumsum(...,0)` acts on a 1-D coordinate ramp. No reduction over dimension 0 (the batch) was found. | Per sample |
| Ensemble | `main.py:140`: mean over the member axis | Per scan |
| Batch-1 pad | `main.py:116-117`: duplicates the same scan, then drops the copy | Harmless |
| Order effects | Order is fixed by `submission_format`. The only cross-batch state is `_SLABS` (a constant). | None |

The batching is a throughput device only. No test-set statistic, BN adaptation or TTA pooled across scans was found.

**Residual caveat:** I read the graph code of 1 of 50 member traces. The other 49 share the same architecture and recipe, and they were exported by the same `export_members.py`.

## 4. Printing and logging (Rule C)

| Line | Emits | Assessment |
|---|---|---|
| `main.py:38` | `WARN {nifti basename}: {exception repr}`, only when a scan fails preprocessing | Reveals one test UID and an error string. It fires only on failure. The scored CSV has **0** values equal to 0.5, so no scan failed and this line did not print in the scored run (strong inference). Low. |
| `main.py:108` | Number of modules, device, dtypes, anchor type, number of scans, number of workers | Scan count only (524). Negligible. |
| `main.py:159` | Count of hedged scans | An aggregate count; 0 in the scored run. Negligible. |
| `main.py:160` | Output path and row count | Negligible. |
| `main.py:158` | Writes `submission.csv` | Required output. |

No logging module, file dumps of features, or debug writes were found.

## 5. Network calls

- **Harness.** It imports only os, glob, json, numpy, pandas, nibabel, torch, concurrent.futures, scipy, and the local modules. There are no requests, urllib, socket, HTTP or subprocess calls.
- **Tree.** The only network code is `build/provenance_ellipse/segment_striata_server.py:18,289`, a `ThreadingHTTPServer` bound to `127.0.0.1` (a local annotation UI). There is no HF, boto or wandb client code in the tree.
- **Outside the tree.** `notes/REFUTED.md:826-836` mentions a `dd_status.py` (not in the tree) that read the competitor's own DrivenData submission pages. That is outside the scored code path.

## 6. Code vs leaderboard match

**Harness vs tree.** Harness `main.py` differs from tree `inference/main.py` only by the 2026-09-11 `ex_*` mixed-roster path, which is inert with fu-only assets (`notes/main_py_harness_vs_tree.diff`). All other harness code and constants are byte-identical to the tree (Coordinator). README:67's claim that the tree copy "is the exact program submitted" is inaccurate, but the two are functionally equivalent.

**The harness plausibly produced the scored CSV.** Scored CSV sha256 is `104184ba…9022a`; zip sha256 is `43789dc1…78395`.
- **Format.** It has the header `uid,is_pathologic`, 524 rows, and the uid order equals `submission_format` (the harness writes `fmt["uid"]` in order).
- **Values.** All 524 values are exactly float32-representable, which is consistent with `float()` of a torch float32 tensor. The range is [0.0078, 0.9892], and there are no 0.5 hedges.
- **Caps.** The 51 repeated values equal `sigmoid(±0.78·6 − 0.16455)`, i.e. every member's logit hit the ±6 cap. This matches the harness mechanism and the shipped `calibration.json`, and it does **not** match the documented b.
- **No hand edits.** No sign of hand editing was found (distinct values: 475; no round numbers).

**Can the tree produce the shipped assets?** Structurally yes, but only in part:
- **Members.** `train_fusion10.sh` names `fu_a1..a10_dnet`, and `build_roster.py:40` globs `{m}_f[0-9].ts.pt`, which matches the 50 harness filenames `fu_a{1..10}_dnet_f{0..4}.ts.pt`.
- **Calibration JSON.** Its schema matches `build_roster.py:72` (`{a, b, member_logit_cap}`).
- **Not reproducible.**
  - The shipped weights and `calibration.json` are absent from the tree.
  - The documented intercept does not reproduce the shipped one.
  - The retrain is not bit-identical: it has f32 checkpoints, and a spot check reports only 0.95-0.97 weight correlation (`docs/repro/md5_weights_training_new_dat.txt:9-19`), with OOF ll reproduced within 0.0002.
  - `canon_const.json` has no builder in the tree.

## 7. Hard-coded predictions, labels, UIDs and post-processing

- There are no UID lists, dictionaries or per-UID overrides in the harness.
- The constants are:
  - box geometry;
  - `canon_const.json` (spans and reference points);
  - slab ranges `38:64` and `64:90`;
  - the ±6 cap;
  - the calibration `a` and `b`;
  - the 0.5 fallback for failed scans.
- The only leaderboard-influenced constant is `a = 0.78`, from LB-guided selection (see §2).
- There is no post-processing conditioned on uid, row position or test-set statistics.

## Evidence table

| # | Path / artifact | Finding | Class |
|---|---|---|---|
| E1 | `tree/**` (188 files, including binaries); `harness/*.py,*.json`; all 58 zip entries plus the inner entries of 51 `.ts.pt` | 0 test UIDs in every scope (positive controls pass) | Confirmed clean |
| E2 | `harness/main.py:83-84` | `submission_format` used only for the uid list and order | Confirmed clean |
| E3 | `fu_a1_dnet_f0.ts.pt` `code/__torch__/torch/nn/modules/batchnorm/*.py` | 371 batch_norm calls with `training=False`; dropout off | Confirmed clean |
| E4 | `ellipse.ts.pt` `code/__torch__/te.py`, `container.py` | 8 BatchNorm3d with `training=False`; per-sample softmax and centroid | Confirmed clean |
| E5 | `harness/canonize.py:49-66`, `datprep_iso.py:76-81` | Per-scan statistics only | Confirmed clean |
| E6 | `harness/main.py:140` | Mean over members, not over the batch | Confirmed clean |
| E7 | `harness/main.py:38` | Prints the test nifti basename on preprocessing failure; did not fire in the scored run (0 hedges) | Low (Rule C, latent) |
| E8 | `build/calibrate.py:9-29` | a = 0.78 chosen with public-LB feedback (2 submissions) | Info (LB-guided, not a violation) |
| E9 | `harness/assets/calibration.json` vs `recipes/fusion10_s078.json:73` | Shipped b −0.16455 vs documented −0.14782; b is fitted on train OOF only (`build_roster.py:45-58`); the scored CSV matches the shipped b | Medium (provenance/documentation) |
| E10 | `datscan/config.py:567-569`, `data.py:85-93` | "TEST-LIKE" cluster weighting inferred from the LB over train clusters; unused in the winning recipe | Info |
| E11 | `notes/main_py_harness_vs_tree.diff`; README:67 | Harness and tree `main.py` differ (inert ex_* path); README claim inaccurate | Low |
| E12 | `3rd_place_submission.csv` | 524 rows, float32 values, cap values match the harness calibration | Consistent |
| E13 | `harness/*.py` imports; `segment_striata_server.py:289` | No network in the harness; the annotation server is loopback only | Confirmed clean |
| E14 | tree vs harness assets | The 50 member weights and `calibration.json` are not in the tree; the retrain is not bit-identical | Low (reproducibility) |

## Open questions

1. Which OOF run directory produced the shipped b = −0.16455? (Ask the competitor for the `build_roster.py` invocation and log.)
2. Should the latent WARN print of a test filename (`main.py:38`) be treated as a Rule C issue, given that it only fires on failure and did not fire in the scored run?
3. The intake public score is 0.231321, and the Coordinator recomputed 0.231707. Reviewer 1 attributes the difference to the a = 0.85 sibling submission. DrivenData should confirm which submission ID is the scored one.
