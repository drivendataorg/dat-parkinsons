"""Header-only pass over the training NIfTIs: every field that could fingerprint a
centre/reconstruction (no voxel data is read). Writes outputs/hdr_fingerprint.csv."""
import sys, glob, os
import numpy as np, pandas as pd, nibabel as nib
root = sys.argv[1] if len(sys.argv) > 1 else 'data-demo/niftis'
rows = []
for f in sorted(glob.glob(os.path.join(root, '*.nii.gz'))):
    uid = os.path.basename(f).split('.')[0]
    h = nib.load(f).header
    r = dict(uid=uid)
    for k in ('descrip', 'db_name', 'aux_file', 'intent_name'):
        r[k] = h[k].tobytes().decode('latin1').strip('\x00').strip()
    for k in ('qform_code', 'sform_code', 'intent_code', 'slice_code', 'xyzt_units', 'dim_info',
              'cal_max', 'cal_min', 'scl_slope', 'scl_inter', 'toffset', 'slice_duration',
              'slice_start', 'slice_end', 'vox_offset', 'sizeof_hdr', 'datatype', 'bitpix', 'glmax', 'glmin'):
        r[k] = float(h[k]) if np.ndim(h[k]) == 0 else str(h[k])
    r['qoffset_x'], r['qoffset_y'], r['qoffset_z'] = float(h['qoffset_x']), float(h['qoffset_y']), float(h['qoffset_z'])
    pd_ = np.asarray(h['pixdim'], float)
    for i in range(8): r[f'pixdim{i}'] = pd_[i]
    for i, v in enumerate(np.asarray(h['quatern_b']).ravel().tolist() + np.asarray(h['quatern_c']).ravel().tolist() + np.asarray(h['quatern_d']).ravel().tolist()):
        r[f'quat{i}'] = v
    A = h.get_best_affine()
    for i in range(3): r[f'aff_t{i}'] = A[i, 3]
    r['aff_diag'] = ','.join(f'{A[i,i]:.4f}' for i in range(3))
    r['aff_offdiag_max'] = float(np.max(np.abs(A[:3,:3] - np.diag(np.diag(A[:3,:3])))))
    r['dim'] = 'x'.join(str(int(x)) for x in h['dim'][1:4])
    r['magic'] = h['magic'].tobytes().decode('latin1').strip('\x00')
    r['extensions'] = len(nib.load(f).header.extensions)
    rows.append(r)
d = pd.DataFrame(rows)
d.to_csv('outputs/hdr_fingerprint.csv', index=False)
print('wrote', len(d), 'rows,', d.shape[1], 'cols')
for c in d.columns:
    if c == 'uid': continue
    nun = d[c].nunique()
    if nun <= 40:
        print(f'{c:>16}: {nun:3d} unique -> {d[c].value_counts().head(8).to_dict()}')
    else:
        print(f'{c:>16}: {nun:3d} unique')
