"""fetch_backbones.py -- download (once) and pin the two ImageNet backbones that
the 2D members (src/cnn2d.py) are fine-tuned from.

In the original run the weights were pre-downloaded into the Hugging Face cache
and every training chain then ran with HF_HUB_OFFLINE=1 (tools/gpu_queue2.sh,
f2seed2_chain.sh, b3_chain.sh, post_b3_chain.sh, v14c_chain.sh). Their revision
and hash were never recorded, so this script makes the download an explicit,
checkable step instead of a side effect of the first training run:

  * builds each backbone through the SAME call the trainer uses
    (cnn2d.build -> timm.create_model(name, pretrained=True, num_classes=1,
    drop_rate=0.3, in_chans=3)), which downloads it into the HF cache if absent;
  * locates the cached weight file and prints the HF repo id, the snapshot
    (commit) it resolved to, and the file's sha256;
  * compares the sha256 with the pinned value and exits non-zero on mismatch
    (set ALLOW_BACKBONE_MISMATCH=1 to continue anyway).

Pinned values (model.safetensors, Apache-2.0, pretrained on ImageNet-1k):
  timm/efficientnet_b0.ra_in1k  -> used by e2dmipf2, e2dmipf22
  timm/efficientnet_b3.ra2_in1k -> used by e2db3f2

Usage: python tools/fetch_backbones.py        (network needed the first time)
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

BACKBONES = {
    "efficientnet_b0.ra_in1k": (
        "timm/efficientnet_b0.ra_in1k",
        "d569899762ea9b1384ee07f4af64805cf8caa1c55f9253ebb1080dc40e87a2cd"),
    "efficientnet_b3.ra2_in1k": (
        "timm/efficientnet_b3.ra2_in1k",
        "279d2a53898aa89dab43fd6bd7df9f706aea4cb9cf916223988bbcb8e5850469"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    import timm
    from huggingface_hub import try_to_load_from_cache
    import cnn2d

    print(f"[backbones] timm {timm.__version__}")
    bad = 0
    for name, (repo, want) in BACKBONES.items():
        cnn2d.build(name, pretrained=True, drop=0.3)       # downloads if not cached
        path = None
        for fn in ("model.safetensors", "pytorch_model.bin"):
            p = try_to_load_from_cache(repo, fn)
            if isinstance(p, str) and Path(p).exists():
                path = Path(p)
                break
        if path is None:
            print(f"[backbones] {name}: weight file not found in the HF cache")
            bad += 1
            continue
        got = sha256(path)
        rev = path.parent.name          # .../snapshots/<commit>/<file>
        ok = got == want
        bad += 0 if ok else 1
        print(f"[backbones] {name}: repo {repo} revision {rev} file {path.name} "
              f"size {path.stat().st_size} sha256 {got} {'OK' if ok else 'MISMATCH (pinned ' + want + ')'}")
    if bad and os.environ.get("ALLOW_BACKBONE_MISMATCH") != "1":
        print("[backbones] hash mismatch or missing file -- refusing to continue "
              "(ALLOW_BACKBONE_MISMATCH=1 overrides)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
