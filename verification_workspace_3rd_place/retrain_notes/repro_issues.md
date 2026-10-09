# 3rd place: issues found while re-running the submitted code

Running log of things in the submitted `3rd Place/` tree that error, fail, or are documented
incorrectly when the training/inference pipeline is actually run. Purpose: fix list for the
open-source release.

Environment: 1x NVIDIA L40S 46 GB (Ada; the author used V100S), 8 vCPU, 61 GB RAM, Python 3.12.15 via uv,
`requirements.txt` pins installed exactly (torch 2.10.0+cu128, monai 1.6.0, numpy 1.26.4, ...).
Workspace `~/sfmn3` (venv `.venv`, `DAT_WORK=~/sfmn3/work`); helper scripts and logs are in this folder (`scripts/`, `logs/`).

## Errors (code does not run as documented)

| # | Date | Where | What happens | Workaround used | Suggested fix |
|---|---|---|---|---|---|
| 1 | 2026-10-07 | `build/build_canonv2_cache.py` (README "Reproducing" step 1); same pattern in `build/export_members.py`, `build/verify_traces.py`, `build/build_mask3d.py` | `python build/build_canonv2_cache.py` run from the repo root fails immediately: `ModuleNotFoundError: No module named 'datscan'`. Python puts `build/` on `sys.path`, not the repo root. | `PYTHONPATH="$PWD" python build/...` | Add `sys.path.insert(0, ROOT)` before `from datscan import ...` in each build script (export_members already computes `ROOT` but inserts it *after* the `datscan` import), or document `PYTHONPATH=.` / `python -m build.<script>`. |
| 2 | 2026-10-07 | `build/verify_traces.py` (README "Reproducing" step 3) | Even with `PYTHONPATH` set to the repo root, fails with `ModuleNotFoundError: No module named 'canonize'`: it inserts the repo root on `sys.path`, but `canonize.py` lives in `inference/`. Reproduced by running the script with `PYTHONPATH="$PWD"`. | `PYTHONPATH="$PWD:$PWD/inference"` | Insert `os.path.join(ROOT, "inference")` on `sys.path` as `export_members.py` does. |
| 3 | 2026-10-07 | `build/export_members.py` (README step 3) | On the L40S the export aborts on its own GPU gate: `AssertionError: fu_a1_dnet f0 gpu 0.0021522045135498047` (tolerance 1e-4). Cause, confirmed: PyTorch's default TF32 for cuDNN convolutions on Ampere/Ada GPUs. The author's V100S has no TF32, so the gate passes there. With TF32 off the same trace's GPU gap is 9.5e-7 and export succeeds. Hardware-dependent; would also hit on A100/H100. | `scripts/run_notf32.py` wrapper sets `torch.backends.cudnn.allow_tf32 = False` and `torch.backends.cuda.matmul.allow_tf32 = False`, then runs the script unchanged | Disable TF32 at the top of `export_members.py` (the gate is meant to catch baked-in trace constants, not precision modes), or loosen the GPU tolerance to ~5e-3 on logits. |
| 4 | 2026-10-07 | `build/verify_traces.py` | When the assets dir holds no `*_f[0-9].ts.pt` files it prints `TRACECHECK 0 fold-models … PASS`, a vacuous pass. Seen after the failed export in #3 left the dir empty. | n/a | Fail (non-zero exit) when 0 traces are found, and exit non-zero on FAIL in general (it currently only prints). |

## Documentation inaccuracies (no error, but misleading)

| # | Where | Issue |
|---|---|---|
| D1 | README "Hardware used": "Training: ~2 h 15 per fold" (same claim in the `recipes/train_fusion10.sh` header comment) | The submitted logs (`docs/repro/logs_original/`) show ~51–53 min/fold and the author's retrain ~58–62 min/fold, at 9 trainers sharing 3 V100S. |
| D2 | `tests/test_t3_forward_equivalence.py` docstring/comments: "shipped traces are f16" | `main.py` in the submitted zip reports `loaded 50 modules on cuda (['torch.float32'])`; the traces are f32. |
| D3 | `recipes/fusion10_s078.json` / README calibration intercept −0.1478 | The submitted `assets/calibration.json` has b = −0.16455. Already recorded as F5 in `verification_report.md`. 2026-10-08: `calibrate.py`/`build_roster.py` run on OOF rebuilt from the submitted traces give −0.147955, i.e. the documented value, not the shipped one (see `retrain_log.md`). |
| D4 | `build/calibrate.py` vs `build/build_roster.py`, both print `opt_temp` | Two different definitions under one name: calibrate.py refits b at every slope (1.035 here), build_roster.py holds b fixed at the slope-0.78 fit (1.031 here). Both are compared with the same "~1.04" reference in the docs. Cosmetic; worth one consistent definition. |

## Notes for anyone re-running a partial check

- `build/export_members.py` always loops over folds 0–4 for each member; with only some folds trained it exports the existing ones and then raises `FileNotFoundError` on the first missing checkpoint. Expected behaviour for a full run, but blocks a partial export without a small wrapper.
