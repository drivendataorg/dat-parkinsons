"""Install the trained v17 model weights into inference/assets/ and verify them.

The 60 asset files (34 3D-CNN checkpoints, 18 TorchScript 2D-CNN checkpoints, two
linear models, the registration template, the TabICL checkpoint and the two frozen
TabICL contexts with their recipe, plus inference_config.json) can come from either:

  * the scored competition archive `submission.zip` (platform submission 322397,
    sha256 88b5611e25170d88c2fbf21b62443dbc67b722c519b5a0391a60aab62bbec7fd), or
  * the released weights bundle, whose 18 TorchScript files have had their
    `*.debug_pkl` records removed (these held the exporting machine's absolute
    source paths). Tensors and outputs are identical; see docs/PROVENANCE.md.

Every file is checked against models/SHA256SUMS.scored.txt or
models/SHA256SUMS.public.txt; a file must match one of the two.

Usage (from the repository root):
    python scripts/install_weights.py --from-zip path/to/submission.zip
    python scripts/install_weights.py --from-zip path/to/navattack_v17_weights.zip
    python scripts/install_weights.py --url <direct-download link to navattack_v17_weights.zip>
    python scripts/install_weights.py --verify-only

--url needs a link that returns the zip file itself, for example a Dropbox link
ending in ?dl=1. A Google Drive share page returns an HTML page instead of the file,
so with Google Drive download the zip in a browser and use --from-zip.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "inference" / "assets"
SUMS = [ROOT / "models" / "SHA256SUMS.scored.txt", ROOT / "models" / "SHA256SUMS.public.txt"]


def load_sums() -> dict[str, set[str]]:
    allowed: dict[str, set[str]] = {}
    for f in SUMS:
        for line in f.read_text().splitlines():
            if line.strip():
                digest, name = line.split(maxsplit=1)
                allowed.setdefault(Path(name.strip()).name, set()).add(digest)
    return allowed


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(dest: Path) -> bool:
    allowed = load_sums()
    ok = True
    missing = [n for n in sorted(allowed) if not (dest / n).exists()]
    for n in missing:
        print(f"MISSING  {n}")
        ok = False
    for n in sorted(allowed):
        p = dest / n
        if p.exists() and sha256(p) not in allowed[n]:
            print(f"BAD HASH {n}")
            ok = False
    extra = sorted(p.name for p in dest.glob("*")
                   if p.is_file() and not p.name.startswith(".") and p.name not in allowed)
    for n in extra:
        print(f"UNEXPECTED {n}")
    print(f"{len(allowed) - len(missing)}/{len(allowed)} assets present; "
          f"{'all hashes verified' if ok else 'VERIFICATION FAILED'}")
    return ok


def extract(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    wanted = set(load_sums())
    n = 0
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            # accept both "assets/<file>" (submission.zip) and "<file>" / "*/assets/<file>" layouts
            if info.is_dir() or name not in wanted:
                continue
            with z.open(info) as src, open(dest / name, "wb") as out:
                shutil.copyfileobj(src, out, length=1 << 20)
            n += 1
    print(f"extracted {n} asset files into {dest.relative_to(ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--from-zip", type=Path, help="scored submission.zip or the weights bundle zip")
    g.add_argument("--url", help="download the weights bundle zip from this direct-download link first "
                                 "(e.g. a Dropbox link ending in ?dl=1; Google Drive share pages do not work)")
    g.add_argument("--verify-only", action="store_true", help="only verify inference/assets/")
    args = ap.parse_args()
    if args.url:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "weights.zip"
            print("downloading ...", flush=True)
            try:
                urllib.request.urlretrieve(args.url, tmp)
            except (urllib.error.URLError, OSError) as exc:
                print(f"download failed: {exc}", file=sys.stderr)
                return 2
            if not zipfile.is_zipfile(tmp):
                print("the downloaded file is not a zip archive: --url needs a direct-download link "
                      "(e.g. Dropbox ?dl=1), or download the zip yourself and use --from-zip", file=sys.stderr)
                return 2
            extract(tmp, DEST)
    elif args.from_zip:
        if not (args.from_zip.is_file() and zipfile.is_zipfile(args.from_zip)):
            print(f"not a readable zip file: {args.from_zip}", file=sys.stderr)
            return 2
        extract(args.from_zip, DEST)
    return 0 if verify(DEST) else 1


if __name__ == "__main__":
    sys.exit(main())
