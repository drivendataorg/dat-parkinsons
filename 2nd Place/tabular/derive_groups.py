"""Acquisition groups (fam2 x scale) used by the v17 tabular block.

Reproduces the fam2 and scale columns of the original outputs/site_diag_rows.csv
(site_diag.py merge plus an interactive step on 2026-09-03 that was never saved
as a script; the rules below were recovered from the original development session logs):

    site_sp = native_spacing.round(2); sp = site_sp as float
    fam2    = 'thin39' if round(pixdim3,1)==3.9 and nz<60
              'full39' if round(pixdim3,1)==3.9
              's246' / 's23' / 's24' if sp is close to 2.46 / 2.3 / 2.4
              'small' otherwise
    scale   = 'low' if log10(p99+1) < 2.25 else 'high'

Inputs (all under outputs/, as the original code wrote them):
  eda_real_geometry.csv  <- python src/eda_real.py            (nz, p99)
  hdr_fingerprint.csv    <- python tools/hdr_fingerprint.py <niftis dir>  (pixdim3)
  features_v13_m2f.csv   <- clinical-feature stage             (native_spacing)
The two image scans are run here when their outputs are missing (or with
--force-scan). Output: outputs/tabular/acquisition_groups.csv (uid, fam2, scale).

Only aggregate counts are printed; no per-scan value is logged.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def rel(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return '<external>/' + path.name


def run_logged(cmd, log, cwd, env=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'w') as f:
        subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, check=True, env=env)


def ensure_scans(force, python):
    """Run the two original scans (their stdout goes to outputs/logs only)."""
    data = Path(os.environ.get('DAT_DATA_DIR', str(ROOT / 'data' / 'raw')))
    env = dict(os.environ)
    # eda_real.py uses joblib n_jobs=-1; cap the worker count on a shared machine.
    env.setdefault('LOKY_MAX_CPU_COUNT', '3')
    geo = OUT / 'eda_real_geometry.csv'
    if force or not geo.exists():
        print('[groups] running src/eda_real.py (log: outputs/logs/eda_real.log)', flush=True)
        run_logged([python, 'src/eda_real.py'], OUT / 'logs/eda_real.log', ROOT, env)
    hdr = OUT / 'hdr_fingerprint.csv'
    if force or not hdr.exists():
        print('[groups] running tools/hdr_fingerprint.py (log: outputs/logs/hdr_fingerprint.log)', flush=True)
        # Original invocation: cwd=project root, `python tools/hdr_fingerprint.py data-demo/niftis`;
        # the script writes outputs/hdr_fingerprint.csv relative to the cwd.
        run_logged([python, 'tools/hdr_fingerprint.py', str(data / 'niftis')], OUT / 'logs/hdr_fingerprint.log', ROOT, env)
    return geo, hdr


def derive(geo, hdr, meta):
    g = pd.read_csv(geo, dtype={'uid': str})[['uid', 'nx', 'ny', 'nz', 'sx', 'sz', 'vmax', 'p99']]
    h = pd.read_csv(hdr, dtype={'uid': str})[['uid', 'pixdim1', 'pixdim3', 'aff_offdiag_max']]
    q = pd.read_csv(meta, dtype={'uid': str})[['uid', 'native_spacing']]
    for name, frame in [('eda_real_geometry', g), ('hdr_fingerprint', h), ('native_spacing', q)]:
        assert frame.uid.is_unique, f'duplicate uids in {name}'
    d = g.merge(h, on='uid', how='inner').merge(q, on='uid', how='inner')
    assert len(d) == len(g) == len(h) == len(q), (len(d), len(g), len(h), len(q))
    # site_diag.py: site_sp = native_spacing.round(2).astype(str); fam2 used sp = site_sp.astype(float)
    site_sp = d.native_spacing.round(2).astype(str)
    sp = site_sp.astype(float)
    p3 = d.pixdim3.round(1)
    fam2 = np.where((p3 == 3.9) & (d.nz < 60), 'thin39',
           np.where(p3 == 3.9, 'full39',
           np.where(np.isclose(sp, 2.46), 's246',
           np.where(np.isclose(sp, 2.3), 's23',
           np.where(np.isclose(sp, 2.4), 's24', 'small')))))
    lp99 = np.log10(d.p99 + 1)
    scale = np.where(lp99 < 2.25, 'low', 'high')
    return pd.DataFrame(dict(uid=d.uid, fam2=fam2, scale=scale))


def verify(groups, reference):
    ref = pd.read_csv(reference, dtype={'uid': str}).set_index('uid')
    assert ref.index.is_unique and set(ref.index) == set(groups.uid), 'uid sets differ from reference'
    ref = ref.loc[groups.uid]
    out = {}
    for col in ['fam2', 'scale']:
        agree = int((ref[col].to_numpy() == groups[col].to_numpy()).sum())
        out[col] = dict(agree=agree, total=len(groups))
        print(f'[verify] {col}: {agree}/{len(groups)} agree with reference', flush=True)
        tab = pd.crosstab(pd.Series(groups[col].to_numpy(), name='derived'), pd.Series(ref[col].to_numpy(), name='reference'))
        print(tab.to_string(), flush=True)
        out[col]['crosstab'] = {str(k): {str(c): int(v) for c, v in row.items() if v} for k, row in tab.iterrows()}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--native-spacing-csv', default=None,
                    help='table with uid,native_spacing (default outputs/features_v13_m2f.csv)')
    ap.add_argument('--out', default=str(OUT / 'tabular/acquisition_groups.csv'))
    ap.add_argument('--force-scan', action='store_true', help='re-run eda_real.py and hdr_fingerprint.py')
    ap.add_argument('--no-scan', action='store_true', help='never run the scans; require their outputs')
    ap.add_argument('--verify-against', default=None,
                    help='archived site_diag_rows.csv with fam2/scale columns (aggregate agreement only)')
    ap.add_argument('--python', default=sys.executable)
    args = ap.parse_args()

    if args.no_scan:
        geo, hdr = OUT / 'eda_real_geometry.csv', OUT / 'hdr_fingerprint.csv'
    else:
        geo, hdr = ensure_scans(args.force_scan, args.python)
    meta = Path(args.native_spacing_csv) if args.native_spacing_csv else OUT / 'features_v13_m2f.csv'
    if not meta.exists():
        raise SystemExit(f'native_spacing table not found: {meta} (run the clinical-feature stage or pass --native-spacing-csv)')
    groups = derive(geo, hdr, meta)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    groups.to_csv(out, index=False)
    fam_counts = groups.fam2.value_counts().to_dict()
    grp_counts = (groups.fam2 + '/' + groups.scale).value_counts().to_dict()
    info = dict(rows=len(groups), fam2_counts=fam_counts, group_counts=grp_counts,
                inputs={'eda_real_geometry': dict(path=rel(geo), sha256=sha256(geo)),
                        'hdr_fingerprint': dict(path=rel(hdr), sha256=sha256(hdr)),
                        'native_spacing': dict(path=rel(meta), sha256=sha256(meta))},
                output=dict(path=rel(out), sha256=sha256(out)))
    print(json.dumps(dict(rows=len(groups), fam2_counts=fam_counts, group_counts=grp_counts)), flush=True)
    if args.verify_against:
        info['verification'] = verify(groups, args.verify_against)
    out.with_name(out.stem + '_info.json').write_text(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
