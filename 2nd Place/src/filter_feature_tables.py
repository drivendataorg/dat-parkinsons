"""
filter_feature_tables.py -- turn the raw registered tables written by
make_regfeat32.py (``*_m2.csv``) into the tables the linear members were fitted on
(``*_m2fnh.csv``).

This file did not exist in the original project. It reproduces, line for line,
two ad-hoc pandas heredocs recovered from the original development session logs:

  (a) 2026-09-02 02:18 UTC, m2 -> m2f: restrict each table to the champion-v7
      column list, in champion-v7 order (uid first, label last). This drops
      ``hole_ml`` from the rich table (make_regfeat32 emits it, the v7 list does
      not have it) and reorders the columns. The original read the header of
      ``champion_v7_CV0.2139/features_v{13,4}_reg7f.csv``; those header lines are
      shipped verbatim as ``configs/feature_columns/*.header.csv``.

          keep = ['uid'] + [c for c in v7.columns if c not in ('uid','is_pathologic')]
                 + ['is_pathologic']
          r[keep].to_csv('outputs/features_v13_m2f.csv', index=False)

  (b) 2026-09-03 04:29 UTC, m2f -> m2fnh: re-read each m2f table and drop
      ``head_ml``.

          d = pd.read_csv('outputs/'+a); d = d.drop(columns=['head_ml'])
          d.to_csv('outputs/'+b, index=False)

The CSV round trip in (b) is kept on purpose: the archived m2fnh files were
written from the re-read m2f text, and keeping the same round trip is what makes
the output byte-identical.

Only shapes and column counts are printed.

Usage (from the repo root or anywhere):
    python src/filter_feature_tables.py            # tag m2 (the v17 lineage)
    python src/filter_feature_tables.py --tag m2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
CFG = ROOT / "configs" / "feature_columns"


def _v7_keep(header_csv: Path) -> list[str]:
    v7 = pd.read_csv(header_csv, nrows=2)
    return ["uid"] + [c for c in v7.columns if c not in ("uid", "is_pathologic")] + ["is_pathologic"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", default="m2", help="table tag written by make_regfeat32 (default m2)")
    ap.add_argument("--rich-header", default=str(CFG / "features_v13_reg7f.header.csv"))
    ap.add_argument("--narrow-header", default=str(CFG / "features_v4_reg7f.header.csv"))
    args = ap.parse_args()
    tag = args.tag

    # (a) m2 -> m2f : champion-v7 column list and order (drops hole_ml)
    r = pd.read_csv(OUT / f"features_v13_{tag}.csv")
    n = pd.read_csv(OUT / f"features_v4_{tag}.csv")
    # Guard (not in the original): make_regfeat32 writes the narrow table as a column
    # subset of the registered rich table. `cache3.py _<tag>` writes a canonical-frame
    # table under the same name, features_v4_<tag>.csv; if it ran later, it overwrote
    # the registered one. Refuse to continue in that case.
    if any(c not in r.columns for c in n.columns) or not all(r[c].equals(n[c]) for c in n.columns):
        raise SystemExit(f"features_v4_{tag}.csv is not a column subset of features_v13_{tag}.csv; "
                         f"it was probably overwritten by `cache3.py _{tag}`. Re-run pipeline/02_features.sh.")
    keep = _v7_keep(Path(args.rich_header))
    keepn = _v7_keep(Path(args.narrow_header))
    dropped_r = sorted(set(r.columns) - set(keep))
    dropped_n = sorted(set(n.columns) - set(keepn))
    r[keep].to_csv(OUT / f"features_v13_{tag}f.csv", index=False)
    n[keepn].to_csv(OUT / f"features_v4_{tag}f.csv", index=False)
    print(f"[filter] {tag} -> {tag}f: rich {r.shape[1]} -> {len(keep)} cols (dropped {dropped_r}), "
          f"narrow {n.shape[1]} -> {len(keepn)} cols (dropped {dropped_n})", flush=True)

    # (b) m2f -> m2fnh : drop head_ml (re-read from disk exactly as the original did)
    for a, b in [(f"features_v13_{tag}f.csv", f"features_v13_{tag}fnh.csv"),
                 (f"features_v4_{tag}f.csv", f"features_v4_{tag}fnh.csv")]:
        d = pd.read_csv(OUT / a)
        d = d.drop(columns=["head_ml"])
        d.to_csv(OUT / b, index=False)
        assert "head_ml" not in d.columns and "hole_ml" not in d.columns
        print(f"[filter] {b} shape {d.shape}", flush=True)


if __name__ == "__main__":
    main()
