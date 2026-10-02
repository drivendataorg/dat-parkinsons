"""check_package.py -- data-free static gates for a v17 submission (zip or directory).

Reproduces the static checks of the original build:

* GATE 6 of tools/build_f2two.sh: ``grep -nE "logger\\.[a-z]+\\(f"`` must match
  nothing. From v15 on the loguru calls live in inference_main.py (main.py is a
  small wrapper), so the gate runs on inference_main.py and on main.py.
* experiments/v17/package_candidate.py: the archive passes ``testzip()``,
  ``main.py`` sits at the archive root, no ``.npz`` is shipped,
  scale_features.py and mixture_features.py can be executed with
  ``__file__ = /code_execution/<name>`` (the platform's shallow working
  directory; the v15a2 submission failed there), and every ``.py`` compiles.

No competition data is read. The shallow-import check needs numpy and scipy.

    python src/check_package.py outputs/v17/submission.zip
    python src/check_package.py outputs/v17/submission          # a staged directory
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG_FSTRING = re.compile(r"logger\.[a-z]+\(f")
LOG_GATED = ("inference_main.py", "main.py")
SHALLOW = ("scale_features.py", "mixture_features.py")
PLATFORM_DIR = "/code_execution/"


class _Package:
    """Uniform read access to a submission zip or a staged directory."""

    def __init__(self, path: Path):
        self.path = path
        if path.is_dir():
            self.zip = None
            self.names = sorted(p.relative_to(path).as_posix() for p in path.rglob("*")
                                if p.is_file() and "__pycache__" not in p.parts
                                and p.suffix != ".pyc")
        else:
            self.zip = zipfile.ZipFile(path)
            self.names = self.zip.namelist()

    def read(self, name: str) -> bytes:
        return self.zip.read(name) if self.zip is not None else (self.path / name).read_bytes()


def run_checks(path) -> dict:
    """Run every gate; return a JSON-able dict with 'passed' and per-check results."""
    pkg = _Package(Path(path))
    res: dict = {"target": Path(path).name, "entries": len(pkg.names)}
    failures: list[str] = []

    if pkg.zip is not None:
        bad = pkg.zip.testzip()
        res["testzip_ok"] = bad is None
        if bad is not None:
            failures.append(f"testzip: corrupt entry {bad}")
    res["main_py_at_root"] = "main.py" in pkg.names
    if not res["main_py_at_root"]:
        failures.append("main.py missing at archive root")
    npz = [n for n in pkg.names if n.endswith(".npz")]
    res["npz_entries"] = len(npz)
    if npz:
        failures.append(f"{len(npz)} .npz entries shipped")

    # GATE 6: static log lines only
    res["logger_fstring_lines"] = {}
    for name in LOG_GATED:
        if name not in pkg.names:
            continue
        text = pkg.read(name).decode("utf-8")
        hits = [i for i, line in enumerate(text.splitlines(), 1) if LOG_FSTRING.search(line)]
        res["logger_fstring_lines"][name] = hits
        if hits:
            failures.append(f"logger f-string calls in {name} at lines {hits}")
    if "inference_main.py" not in pkg.names:
        failures.append("inference_main.py missing (logging gate not applied)")

    # shallow-container import, as package_candidate.py did it
    res["shallow_import"] = {}
    for name in SHALLOW:
        if name not in pkg.names:
            res["shallow_import"][name] = "missing"
            failures.append(f"{name} missing")
            continue
        namespace = {"__name__": Path(name).stem, "__file__": PLATFORM_DIR + name}
        try:
            exec(compile(pkg.read(name), PLATFORM_DIR + name, "exec"), namespace)
            res["shallow_import"][name] = "ok"
        except Exception as exc:  # report the type only
            res["shallow_import"][name] = type(exc).__name__
            failures.append(f"shallow import of {name} raised {type(exc).__name__}")

    # every module compiles
    py = [n for n in pkg.names if n.endswith(".py")]
    bad_compile = []
    for name in py:
        try:
            compile(pkg.read(name), PLATFORM_DIR + name, "exec")
        except SyntaxError:
            bad_compile.append(name)
    res["py_files_compiled"] = len(py) - len(bad_compile)
    res["py_files_failed"] = bad_compile
    if bad_compile:
        failures.append(f"compile failed: {bad_compile}")

    res["failures"] = failures
    res["passed"] = not failures
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("target", help="submission zip or staged directory (relative paths: repo root)")
    args = ap.parse_args()
    target = Path(args.target)
    if not target.is_absolute():
        target = ROOT / target
    res = run_checks(target)
    print(json.dumps(res, indent=2))
    print("CHECK_PACKAGE", "PASSED" if res["passed"] else "FAILED")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
