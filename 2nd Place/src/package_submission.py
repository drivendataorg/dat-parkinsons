"""package_submission.py -- assemble the v17 submission directory and submission.zip.

Reproduces the last packaging step of the scored v17 entry,
experiments/v17/package_candidate.py, on a stage put together the way the
original lineage did it (v14c build3 zip -> experiments/20260914/build_stage.py
-> experiments/v17/prepare_candidate.py / freeze_candidate.py):

* every file of ``inference/`` copied verbatim: the inference code, main.py,
  CANDIDATE.json, tabicl_lib.zip, tabicl_licenses/LICENSE and
  TABICL_MODEL_CARD.md. Anything under ``inference/assets/`` is ignored; all
  assets come from the build outputs below;
* the image / linear / template assets that build3.py wrote to
  ``submission_src/assets/`` (pipeline/08_stack.sh). They are selected by name
  from its inference_config.json, with the same rule build3.py uses for its
  zip (55 files for v17);
* ``assets/inference_config.json``: build3's config with ``'tabicl': true``
  added and ``feature_order`` extended by the tabular feature names that are
  not in it yet, written with ``json.dumps(indent=2)``. This is what
  build_stage.py:42-46 and prepare_candidate.py:42-45 did;
* ``assets/tabular_recipe.json`` and the frozen TabICL states it names
  (tabular_raw.pkl.xz, tabular_log.pkl.xz) from the tabular stage, plus the
  public TabICLv2 checkpoint under the name the recipe expects.

Zip rules as in package_candidate.py: ZIP_STORED for .xz/.zip, ZIP_DEFLATED
level 6 otherwise; __pycache__, .pyc, .tmp, submission.csv, .DS_Store and
.gitkeep are skipped; no .npz may be shipped. The static gates of
src/check_package.py then run on the finished zip, and MANIFEST.json records
the zip size and sha256 plus the size, CRC32 and sha256 of every entry.

``--check-against REF.zip`` compares every entry's name, size, CRC32 and
compression type with a reference archive (the scored
submissions/v17/submission.zip) and reports mismatches. Entry timestamps, and
so the zip bytes, differ by design.

    python src/package_submission.py                         # -> outputs/v17/
    python src/package_submission.py --check-against path/to/scored/submission.zip
    python src/package_submission.py --check-only --check-against REF.zip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_package import run_checks  # noqa: E402

SKIP_NAMES = {"submission.csv", ".DS_Store", ".gitkeep"}
SKIP_SUFFIXES = {".pyc", ".tmp"}
# sha256 of the public TabICLv2 checkpoint shipped in the scored archive
TABICL_CKPT_SHA256 = "bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0"


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def _skip(p: Path) -> bool:
    return "__pycache__" in p.parts or p.suffix in SKIP_SUFFIXES or p.name in SKIP_NAMES


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def referenced_assets(cfg: dict) -> list[str]:
    """Asset files the inference config needs, the same set build3.py zips."""
    names = {"inference_config.json"}
    if cfg.get("registration"):
        names.add(cfg["registration"]["template"])
    names |= {f"linear_{m}.json" for m in cfg["linears"]}
    for m, seeds in cfg.get("gbms", {}).items():
        names |= {f"lgbm_{m}_s{s}.txt" for s in seeds}
        names |= {f"lgbm_{m}_s{s}.txt.meta.json" for s in seeds}
    for m, seeds in cfg.get("mlps", {}).items():
        names |= {f"mlp_{m}_s{s}.json" for s in seeds}
    for s in cfg["cnns"]:
        ext = ".ts.pt" if s.get("ts") else ".pt"
        names |= {f"{src}_{k}{ext}" for src, k in s["sources"]}
    return sorted(names - {"inference_config.json"})


def load_feature_names(path: Path) -> list[str]:
    obj = json.loads(path.read_text())
    if isinstance(obj, dict):
        obj = obj["feature_order"]
    if not (isinstance(obj, list) and all(isinstance(x, str) for x in obj)):
        raise SystemExit(f"{path.name}: expected a JSON list of feature names "
                         "(or {'feature_order': [...]})")
    return obj


def assemble(args, stage: Path) -> None:
    inference, build_assets, tabular = _abs(args.inference_dir), _abs(args.build_assets), _abs(args.tabular_dir)
    assets = stage / "assets"
    assets.mkdir(parents=True)

    # 1. code, licences, CANDIDATE.json: verbatim copy of inference/
    n_code, ignored = 0, 0
    for p in sorted(inference.rglob("*")):
        if not p.is_file() or _skip(p):
            continue
        rel = p.relative_to(inference)
        if rel.parts[0] == "assets":
            ignored += 1
            continue
        (stage / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, stage / rel)
        n_code += 1
    print(f"[package] inference/: {n_code} files copied verbatim"
          + (f" ({ignored} files under inference/assets ignored)" if ignored else ""))

    # 2. image / linear / template assets from the build3 output
    cfg = json.loads((build_assets / "inference_config.json").read_text())
    print(f"[package] build3 inference_config.json sha256 {_sha256(build_assets / 'inference_config.json')}")
    names = referenced_assets(cfg)
    missing = [n for n in names if not (build_assets / n).is_file()]
    if missing:
        raise SystemExit(f"missing build assets (run pipeline/08_stack.sh): {missing}")
    for n in names:
        shutil.copy2(build_assets / n, assets / n)
    print(f"[package] {args.build_assets}: {len(names)} image/linear/template assets")

    # 3. inference_config.json: + tabicl flag, + tabular feature names
    tab_names = load_feature_names(_abs(args.feature_order))
    n_before = len(cfg["feature_order"])
    cfg["tabicl"] = True
    for name in tab_names:
        if name not in cfg["feature_order"]:
            cfg["feature_order"].append(name)
    (assets / "inference_config.json").write_text(json.dumps(cfg, indent=2))
    final = cfg["feature_order"]
    print(f"[package] inference_config.json: feature_order {n_before} -> {len(final)} "
          f"(tabular list {len(tab_names)} names; equals final order: {final == tab_names}); tabicl=true")
    if len(tab_names) == len(final) and final != tab_names:
        print("[package] note: the tabular list holds every name in a different order; build3's "
              "order is kept and new names are appended, as in the original")

    # 4. tabular block: recipe, the frozen states it names, TabICL checkpoint
    recipe = json.loads((tabular / "tabular_recipe.json").read_text())
    shutil.copy2(tabular / "tabular_recipe.json", assets / "tabular_recipe.json")
    for kind, fn in recipe["frozen_states"].items():
        if not (tabular / fn).is_file():
            raise SystemExit(f"missing frozen TabICL state {fn} (run the tabular stage)")
        shutil.copy2(tabular / fn, assets / fn)
    ckpt = _abs(args.tabicl_ckpt)
    if not ckpt.is_file():
        raise SystemExit(f"TabICL checkpoint not found: {args.tabicl_ckpt}")
    shutil.copy2(ckpt, assets / recipe["checkpoint"])
    ck_sha = _sha256(ckpt)
    print(f"[package] tabular: recipe + {sorted(recipe['frozen_states'].values())} + {recipe['checkpoint']} "
          f"(sha256 {'matches' if ck_sha == TABICL_CKPT_SHA256 else 'DIFFERS FROM'} the shipped checkpoint)")


def write_zip(stage: Path, path: Path) -> None:
    # loop body as in experiments/v17/package_candidate.py
    with zipfile.ZipFile(path, "w") as z:
        for p in sorted(stage.rglob("*")):
            if not p.is_file() or _skip(p):
                continue
            assert p.suffix != ".npz", "Raw rebuild or validation arrays must stay outside the submission"
            mode = zipfile.ZIP_STORED if p.suffix in (".xz", ".zip") else zipfile.ZIP_DEFLATED
            z.write(p, p.relative_to(stage), compress_type=mode,
                    compresslevel=None if mode == zipfile.ZIP_STORED else 6)


def check_against(path: Path, ref: Path) -> dict:
    """Compare entry names, sizes, CRC32 and compression type with a reference zip."""
    a = {i.filename: i for i in zipfile.ZipFile(path).infolist()}
    b = {i.filename: i for i in zipfile.ZipFile(ref).infolist()}
    common = sorted(set(a) & set(b))
    crc = [n for n in common if a[n].CRC != b[n].CRC or a[n].file_size != b[n].file_size]
    ctype = [n for n in common if a[n].compress_type != b[n].compress_type]
    res = dict(reference=ref.name, entries=len(a), reference_entries=len(b),
               crc_and_size_equal=len(common) - len(crc), crc_or_size_mismatch=crc,
               compress_type_mismatch=ctype, only_in_package=sorted(set(a) - set(b)),
               only_in_reference=sorted(set(b) - set(a)),
               same_entry_order=[i for i in a] == [i for i in b])
    res["identical_contents"] = not (crc or ctype or res["only_in_package"] or res["only_in_reference"])
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inference-dir", default="inference")
    ap.add_argument("--build-assets", default="submission_src/assets",
                    help="build3.py output (pipeline/08_stack.sh)")
    ap.add_argument("--tabular-dir", default="outputs/tabular",
                    help="tabular_recipe.json + the frozen states it names")
    ap.add_argument("--feature-order", default="outputs/tabular/feature_order_v17.json",
                    help="JSON list of the deployed feature names (v17: 479)")
    ap.add_argument("--tabicl-ckpt", default="inference/assets/tabicl-classifier-v2-20260212.ckpt")
    ap.add_argument("--out-dir", default="outputs/v17",
                    help="receives submission/ (the stage), submission.zip and MANIFEST.json")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing stage/zip (the original refused to overwrite)")
    ap.add_argument("--check-against", default=None, help="reference zip for a per-entry CRC32 comparison")
    ap.add_argument("--check-only", action="store_true",
                    help="do not build; run the static gates (and --check-against) on the existing zip")
    args = ap.parse_args()

    out = _abs(args.out_dir)
    stage, zpath, mpath = out / "submission", out / "submission.zip", out / "MANIFEST.json"
    t0 = time.time()
    if not args.check_only:
        if (stage.exists() or zpath.exists()) and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite {out.name}/submission[.zip]; pass --overwrite")
        if stage.exists():
            shutil.rmtree(stage)
        zpath.unlink(missing_ok=True)
        out.mkdir(parents=True, exist_ok=True)
        assemble(args, stage)
        write_zip(stage, zpath)
        print(f"[package] wrote {zpath.name} ({time.time() - t0:.0f} s)")
    elif not zpath.exists():
        raise SystemExit(f"--check-only: {zpath} does not exist")

    checks = run_checks(zpath)
    print("[package] static gates:", "PASSED" if checks["passed"] else f"FAILED {checks['failures']}")
    with zipfile.ZipFile(zpath) as z:
        infos = z.infolist()
    ref_res = None
    if args.check_against:
        ref_res = check_against(zpath, _abs(args.check_against))
        print("[package] check-against", json.dumps(ref_res, indent=2))

    if not args.check_only:
        files = []
        for i in infos:
            files.append(dict(name=i.filename, bytes=i.file_size, crc32=f"{i.CRC:08x}",
                              sha256=_sha256(stage / i.filename)))
        manifest = dict(zip_bytes=zpath.stat().st_size, sha256=_sha256(zpath), entries=len(infos),
                        shallow_container_import_passed=checks["passed"], static_checks=checks,
                        check_against=ref_res, files=files)
        mpath.write_text(json.dumps(manifest, indent=2))
        print(json.dumps({k: manifest[k] for k in ("zip_bytes", "sha256", "entries",
                                                   "shallow_container_import_passed")}))
    print(f"[package] done in {time.time() - t0:.0f} s")
    ok = checks["passed"] and (ref_res is None or ref_res["identical_contents"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
