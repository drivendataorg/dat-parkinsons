"""scrub_torchscript_debug.py -- remove *.debug_pkl records from TorchScript archives.

torch.jit.save stores, next to the compiled code, one `*.debug_pkl` record per module
with the source ranges of the Python code that was traced. Those records contain the
absolute file paths of the machine that exported the model. They are not needed to
load or run the model. This tool rewrites each archive without them and copies every
other record byte for byte, with its original compression type and order. It then
checks that the original and the scrubbed module give bit-identical outputs.

This is how the 18 TorchScript files of the released weights bundle were produced
from the scored archive (see docs/PROVENANCE.md, section 3; their hashes are in
models/SHA256SUMS.public.txt).

Usage (repo root):
    python tools/scrub_torchscript_debug.py SRC_DIR DST_DIR
Scrubs every *.ts.pt in SRC_DIR into DST_DIR and prints, per file, the number of
records dropped, whether the outputs are bit-identical on fixed synthetic inputs,
and the new sha256.
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import torch


def scrub(src: Path, dst: Path) -> int:
    dropped = 0
    with zipfile.ZipFile(src) as zi, zipfile.ZipFile(dst, "w") as zo:
        for info in zi.infolist():
            if info.filename.endswith(".debug_pkl"):
                dropped += 1
                continue
            out = zipfile.ZipInfo(info.filename, date_time=info.date_time)
            out.compress_type = info.compress_type
            out.external_attr = info.external_attr
            zo.writestr(out, zi.read(info.filename))
    return dropped


def same_outputs(a: Path, b: Path) -> bool:
    torch.manual_seed(0)
    res = 224 if "b3" in a.name else 192          # input size of the B3 / B0 members
    x = torch.rand(6, 3, res, res) * 4 - 2
    ma = torch.jit.load(str(a), map_location="cpu").eval()
    mb = torch.jit.load(str(b), map_location="cpu").eval()
    with torch.no_grad():
        return torch.equal(ma(x), mb(x))


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src_dir, dst_dir = Path(sys.argv[1]), Path(sys.argv[2])
    dst_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for p in sorted(src_dir.glob("*.ts.pt")):
        q = dst_dir / p.name
        n = scrub(p, q)
        eq = same_outputs(p, q)
        ok &= eq
        print(f"{p.name}: dropped {n} record(s), outputs identical: {eq}, "
              f"sha256 {hashlib.sha256(q.read_bytes()).hexdigest()}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
