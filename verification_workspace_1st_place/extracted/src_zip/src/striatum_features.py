"""
striatum_features.py
====================
Port VERBATIM de la section features du notebook. Ce fichier et le notebook
doivent rester identiques : c'est `validate_production.py` qui le garantit.

Expose :
    pair_features(mip_L, mip_R)          -> dict de features BRUTES
    spine_profile(img, refs, n_points)   -> (T, 3) pour la branche spine
    head_tail_masks(img)                 -> (head, tail) float32
"""
import heapq
import numpy as np
from scipy.ndimage import label, distance_transform_edt, gaussian_filter1d
from scipy.stats import skew, kurtosis
from scipy.interpolate import interp1d
from skimage.morphology import dilation, disk, convex_hull_image
from skimage.measure import regionprops, perimeter

EPS = 1e-6
HEAD_SIDE = 'right'
HEAD_DROP_FRAC = 0.5
SEED_WEIGHTS = {'position': 0.2, 'convexity': 0.3, 'brightness': 0.2, 'width': 0.3}
DIRECTIONS = {'right': (0, 1), 'left': (0, -1), 'up': (-1, 0), 'down': (1, 0)}
NB8 = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
       (-1,-1,np.sqrt(2)),(-1,1,np.sqrt(2)),(1,-1,np.sqrt(2)),(1,1,np.sqrt(2))]

SHAPE_STAT_KEYS = ['area','mean_int','max_int','sum_int','std_int','cv_int',
                   'p10_int','p90_int','skew_int','kurt_int',
                   'major_axis','minor_axis','eccentricity','solidity',
                   'circularity','complexity','centroid_x','centroid_y']

# --- spine ---
SPINE_GAMMA, SPINE_SMOOTH = 2.0, 1.0
NORMALIZE_BY_BG = True


# ============================================================ masque / segmentation
def extract_striatum_mask(img_np):
    H, W = img_np.shape
    cy, cx = H // 2, W // 2
    img_max = img_np.max()
    if img_max == 0:
        return None
    for t in np.arange(0.50, 0.95, 0.05):
        labeled, num_features = label(img_np > (img_max * t))
        if num_features == 0:
            continue
        min_dist, central_label = float('inf'), 0
        for i in range(1, num_features + 1):
            coords = np.argwhere(labeled == i)
            if len(coords) == 0:
                continue
            my, mx = coords.mean(axis=0)
            d = (my - cy) ** 2 + (mx - cx) ** 2
            if d < min_dist:
                min_dist, central_label = d, i
        if central_label == 0:
            continue
        mask = (labeled == central_label)
        if not (mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()):
            return mask
    labeled, _ = label(img_np > (img_max * 0.9))
    if labeled.max() > 0:
        sizes = np.bincount(labeled.flat)[1:]
        return labeled == (sizes.argmax() + 1)
    return img_np > (img_max * 0.9)


def compute_background_stats(img_np, mask, exclude_dilation=2):
    if mask is None or mask.sum() == 0:
        return np.nan, np.nan
    bg = img_np[~dilation(mask, disk(exclude_dilation))]
    if bg.size == 0:
        return np.nan, np.nan
    return float(bg.mean()), float(bg.std())


def compute_seed_score(img_np, mask, head_side=HEAD_SIDE, weights=None):
    weights = weights or SEED_WEIGHTS
    H, W = img_np.shape
    dy, dx = DIRECTIONS[head_side]
    yy, xx = np.mgrid[0:H, 0:W]
    pos = (yy - (H - 1) / 2) * dy + (xx - (W - 1) / 2) * dx
    pos = (pos - pos.min()) / (pos.max() - pos.min() + 1e-9)
    try:
        deficiency = convex_hull_image(mask) & ~mask
        cd = (distance_transform_edt(~deficiency) if deficiency.any()
              else distance_transform_edt(np.ones_like(mask)))
        conv = cd / (cd[mask].max() + 1e-9)
    except Exception:
        conv = np.zeros((H, W))
    vals = img_np[mask]
    bright = (img_np - vals.min()) / (vals.max() - vals.min() + 1e-9)
    width = distance_transform_edt(mask)
    wid = width / (width.max() + 1e-9)
    s = (weights['position'] * pos + weights['convexity'] * conv +
         weights['brightness'] * bright + weights['width'] * wid)
    return np.where(mask, s, -np.inf)


def find_seed(img_np, mask, head_side=HEAD_SIDE, weights=None):
    score = compute_seed_score(img_np, mask, head_side, weights)
    sy, sx = np.unravel_index(np.argmax(score), score.shape)
    return (sx, sy), score[sy, sx]


def geodesic_distance(mask, seed_yx):
    H, W = mask.shape
    dist = np.full((H, W), np.inf)
    sy, sx = seed_yx
    dist[sy, sx] = 0.0
    vis = np.zeros((H, W), bool)
    pq = [(0.0, sy, sx)]
    while pq:
        d, y, x = heapq.heappop(pq)
        if vis[y, x]:
            continue
        vis[y, x] = True
        for ddy, ddx, w in NB8:
            ny, nx = y + ddy, x + ddx
            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not vis[ny, nx]:
                nd = d + w
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(pq, (nd, ny, nx))
    return dist


def segment_head_tail_pixels(img_np, mask, head_side=HEAD_SIDE, drop_frac=HEAD_DROP_FRAC,
                             weights=None, bin_width=1.0, return_profile=False):
    if mask is None or mask.sum() == 0:
        return None
    seed, seed_score = find_seed(img_np, mask, head_side, weights)
    sx, sy = seed
    geo = geodesic_distance(mask, (sy, sx))
    fv = geo[mask]
    fv = fv[np.isfinite(fv)]
    if fv.size == 0 or fv.max() < bin_width:
        r = {'seed': seed, 'seed_score': seed_score, 'geo': geo, 'd_split': 0.0,
             'head_mask': mask.copy(), 'tail_mask': np.zeros_like(mask)}
        if return_profile:
            r['profile'] = (np.array([]), np.array([]), 0, 0.0)
        return r
    bins = np.arange(bin_width, fv.max() + bin_width, bin_width)
    W_front = np.array([int((mask & (geo >= b - bin_width) & (geo < b)).sum()) for b in bins])
    peak_idx = int(np.argmax(W_front))
    threshold = drop_frac * W_front[peak_idx]
    below = np.where((np.arange(len(W_front)) > peak_idx) & (W_front < threshold))[0]
    d_split = float(bins[below[0]]) if below.size > 0 else float(bins[-1])
    r = {'seed': seed, 'seed_score': seed_score, 'geo': geo, 'd_split': d_split,
         'head_mask': mask & (geo <= d_split),
         'tail_mask': mask & (geo > d_split) & np.isfinite(geo)}
    if return_profile:
        r['profile'] = (bins, W_front, peak_idx, threshold)
    return r


# ============================================================ features
def region_stats(img_np, region_mask):
    area = int(region_mask.sum())
    if area == 0:
        return {k: np.nan for k in SHAPE_STAT_KEYS}
    vals = img_np[region_mask]
    labeled, n = label(region_mask)
    biggest = (labeled == (np.bincount(labeled.flat)[1:].argmax() + 1)) if n > 1 else region_mask
    props = regionprops(biggest.astype(int))
    if props:
        p = props[0]
        major = (p.axis_major_length if hasattr(p, 'axis_major_length')
                 else p.major_axis_length)
        minor = (p.axis_minor_length if hasattr(p, 'axis_minor_length')
                 else p.minor_axis_length)
        cy, cx = p.centroid
        eccentricity, solidity = p.eccentricity, p.solidity
    else:
        major = minor = eccentricity = solidity = cy = cx = np.nan
    perim = perimeter(biggest)
    circularity = (4 * np.pi * area) / (perim ** 2) if perim > 0 else np.nan
    complexity = (perim ** 2) / area if area > 0 else np.nan
    std_int = float(vals.std())
    return {'area': float(area), 'mean_int': float(vals.mean()), 'max_int': float(vals.max()),
            'sum_int': float(vals.sum()), 'std_int': std_int,
            'cv_int': std_int / (float(vals.mean()) + EPS),
            'p10_int': float(np.percentile(vals, 10)), 'p90_int': float(np.percentile(vals, 90)),
            'skew_int': float(skew(vals)) if len(vals) > 2 else np.nan,
            'kurt_int': float(kurtosis(vals)) if len(vals) > 2 else np.nan,
            'major_axis': major, 'minor_axis': minor, 'eccentricity': eccentricity,
            'solidity': solidity, 'circularity': circularity, 'complexity': complexity,
            'centroid_x': cx, 'centroid_y': cy}


def extract_shape_features(img_np, prefix='', head_side=HEAD_SIDE):
    mask = extract_striatum_mask(img_np)
    feats = {}
    if mask is None or mask.sum() == 0:
        feats[f'{prefix}detected'] = 0
        return feats

    seg = segment_head_tail_pixels(img_np, mask, head_side=head_side, return_profile=True)
    feats[f'{prefix}detected'] = 1
    feats[f'{prefix}seed_score'] = float(seg['seed_score'])
    feats[f'{prefix}d_split'] = float(seg['d_split'])

    mask_area = float(mask.sum())
    feats[f'{prefix}striatum_area'] = mask_area
    feats[f'{prefix}striatum_sum_int'] = float(img_np[mask].sum())
    feats[f'{prefix}striatum_max_int'] = float(img_np[mask].max())

    bins, W_front, peak_idx, threshold = seg.get('profile', ([], [], 0, 0))
    if len(bins) > 0:
        feats[f'{prefix}peak_width'] = float(W_front[peak_idx])
        feats[f'{prefix}neck_width'] = float(threshold)
        feats[f'{prefix}peak_neck_ratio'] = feats[f'{prefix}peak_width'] / (feats[f'{prefix}neck_width'] + EPS)
        if len(bins) > 1:
            f_ = interp1d(bins, W_front, kind='linear', fill_value='extrapolate')
            w_resampled = np.clip(f_(np.linspace(bins[0], bins[-1], 20)), 0, None)
        else:
            w_resampled = np.zeros(20)
    else:
        feats[f'{prefix}peak_width'] = np.nan
        feats[f'{prefix}neck_width'] = np.nan
        feats[f'{prefix}peak_neck_ratio'] = np.nan
        w_resampled = np.zeros(20)
    for i in range(20):
        feats[f'{prefix}profile_w_{i:02d}'] = float(w_resampled[i])

    head_stats = region_stats(img_np, seg['head_mask'])
    tail_stats = region_stats(img_np, seg['tail_mask'])
    for k in SHAPE_STAT_KEYS:
        feats[f'{prefix}head_{k}'] = head_stats[k]
        feats[f'{prefix}tail_{k}'] = tail_stats[k]

    if head_stats['area'] > 0 and tail_stats['area'] > 0:
        dx = head_stats['centroid_x'] - tail_stats['centroid_x']
        dy = head_stats['centroid_y'] - tail_stats['centroid_y']
        feats[f'{prefix}head_tail_dist'] = float(np.hypot(dx, dy))
        feats[f'{prefix}head_tail_area_ratio'] = head_stats['area'] / (tail_stats['area'] + EPS)
        feats[f'{prefix}head_tail_int_ratio'] = head_stats['mean_int'] / (tail_stats['mean_int'] + EPS)
        tail_geo = seg['geo'][seg['tail_mask']]
        tail_coords = np.argwhere(seg['tail_mask'])
        idx_max = np.argmax(tail_geo)
        y_end, x_end = tail_coords[idx_max]
        neck = np.argwhere(mask & (seg['geo'] >= seg['d_split'] - 1.0) & (seg['geo'] <= seg['d_split'] + 1.0))
        y_n, x_n = neck.mean(axis=0) if len(neck) > 0 else (seg['seed'][1], seg['seed'][0])
        euclid_d = np.hypot(y_end - y_n, x_end - x_n)
        feats[f'{prefix}tail_curvature_idx'] = (tail_geo[idx_max] - seg['d_split']) / (euclid_d + EPS)
    else:
        for k in ('head_tail_dist', 'head_tail_area_ratio', 'head_tail_int_ratio', 'tail_curvature_idx'):
            feats[f'{prefix}{k}'] = np.nan

    mask_mean_int = float(img_np[mask].mean())
    bg_mean, bg_std = compute_background_stats(img_np, mask)
    ok = not np.isnan(bg_mean)
    feats[f'{prefix}bg_mean'] = bg_mean
    feats[f'{prefix}bg_std'] = bg_std
    feats[f'{prefix}striatum_contrast_ratio'] = mask_mean_int / (bg_mean + EPS) if ok else np.nan
    feats[f'{prefix}striatum_contrast_snr'] = (mask_mean_int - bg_mean) / (bg_std + EPS) if ok else np.nan
    for r_, st in (('head', head_stats), ('tail', tail_stats)):
        good = st['area'] > 0 and ok
        feats[f'{prefix}{r_}_contrast_ratio'] = st['mean_int'] / (bg_mean + EPS) if good else np.nan
        feats[f'{prefix}{r_}_contrast_snr'] = (st['mean_int'] - bg_mean) / (bg_std + EPS) if good else np.nan
    feats[f'{prefix}head_area_frac'] = head_stats['area'] / (mask_area + EPS)
    feats[f'{prefix}tail_area_frac'] = tail_stats['area'] / (mask_area + EPS)
    return feats


def pair_features(mip_L, mip_R, head_side=HEAD_SIDE):
    """Features BRUTES d'un patient. Mêmes noms et mêmes valeurs que le notebook."""
    feats = {}
    feats.update(extract_shape_features(mip_L, prefix='L_', head_side=head_side))
    feats.update(extract_shape_features(mip_R, prefix='R_', head_side=head_side))

    for lk in [k for k in list(feats) if k.startswith('L_') and k != 'L_detected']:
        suffix = lk[2:]
        rk = f'R_{suffix}'
        if rk not in feats or 'profile_w_' in suffix:
            continue
        l, r = feats[lk], feats[rk]
        lv = l is not None and not (isinstance(l, float) and np.isnan(l))
        rv = r is not None and not (isinstance(r, float) and np.isnan(r))
        if lv and rv:
            feats[f'asym_{suffix}_diff'] = abs(l - r)
            feats[f'asym_{suffix}_ratio'] = l / (r + EPS)
            feats[f'asym_{suffix}_AI'] = 200.0 * abs(l - r) / (abs(l) + abs(r) + EPS)
        else:
            feats[f'asym_{suffix}_diff'] = np.nan
            feats[f'asym_{suffix}_ratio'] = np.nan
            feats[f'asym_{suffix}_AI'] = np.nan
    return feats


# ============================================================ spine
def _min_cost_path(img, mask, start, end, gamma=SPINE_GAMMA):
    H, W = img.shape
    vmax = img[mask].max() if mask.any() else 1.0
    cost = np.where(mask, (vmax / (img + 1e-6)) ** gamma, np.inf)
    dist = np.full((H, W), np.inf)
    prev = np.full((H, W, 2), -1, dtype=int)
    sy, sx = start; ey, ex = end
    dist[sy, sx] = 0.0
    vis = np.zeros((H, W), bool)
    pq = [(0.0, sy, sx)]
    while pq:
        d, y, x = heapq.heappop(pq)
        if vis[y, x]:
            continue
        vis[y, x] = True
        if (y, x) == (ey, ex):
            break
        for dy, dx, ln in NB8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not vis[ny, nx]:
                nd = d + ln * 0.5 * (cost[y, x] + cost[ny, nx])
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd; prev[ny, nx] = (y, x)
                    heapq.heappush(pq, (nd, ny, nx))
    if not np.isfinite(dist[ey, ex]):
        return None
    path = [(ey, ex)]
    while path[-1] != (sy, sx):
        p = prev[path[-1][0], path[-1][1]]
        if p[0] < 0:
            return None
        path.append((int(p[0]), int(p[1])))
    return np.array(path[::-1], dtype=float)


def _bilinear(img, y, x):
    H, W = img.shape
    y = np.clip(y, 0, H - 1); x = np.clip(x, 0, W - 1)
    y0 = np.floor(y).astype(int); x0 = np.floor(x).astype(int)
    y1 = np.minimum(y0 + 1, H - 1); x1 = np.minimum(x0 + 1, W - 1)
    wy, wx = y - y0, x - x0
    return (img[y0,x0]*(1-wy)*(1-wx) + img[y1,x0]*wy*(1-wx)
            + img[y0,x1]*(1-wy)*wx + img[y1,x1]*wy*wx)


def extract_spine(img, mask, seg, head_side=HEAD_SIDE,
                  gamma=SPINE_GAMMA, smooth_sigma=SPINE_SMOOTH):
    if mask is None or seg is None or mask.sum() < 3:
        return None
    tail = seg['tail_mask']
    src = tail if tail.sum() > 0 else mask
    tip = np.unravel_index(np.argmax(np.where(src, seg['geo'], -np.inf)), img.shape)
    geo_tip = geodesic_distance(mask, tip)
    head = seg['head_mask'] if seg['head_mask'].sum() > 0 else mask
    head_end = np.unravel_index(np.argmax(np.where(head, geo_tip, -np.inf)), img.shape)
    path = _min_cost_path(img, mask, head_end, tip, gamma)
    if path is None or len(path) < 3:
        return None
    dy, dx = DIRECTIONS[head_side]
    if (path[0,0]*dy + path[0,1]*dx) < (path[-1,0]*dy + path[-1,1]*dx):
        path = path[::-1].copy()
    if smooth_sigma > 0 and len(path) > 4:
        sm = path.copy()
        sm[:,0] = gaussian_filter1d(path[:,0], smooth_sigma, mode='nearest')
        sm[:,1] = gaussian_filter1d(path[:,1], smooth_sigma, mode='nearest')
        sm[0], sm[-1] = path[0], path[-1]
        path = sm
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*(np.diff(path, axis=0).T)))])
    ip = np.round(path).astype(int)
    hidx = [i for i, (y, x) in enumerate(ip)
            if 0 <= y < head.shape[0] and 0 <= x < head.shape[1] and head[y, x]]
    vp = _bilinear(img, path[:,0], path[:,1])
    i_seed = max(hidx, key=lambda i: vp[i]) if hidx else int(np.argmax(vp))
    return {'path': path, 's_centered': s - s[i_seed]}


def u_to_s(u_grid, refs):
    return np.where(u_grid < 0, u_grid * refs['head'], u_grid * refs['tail'])


def spine_profile(img, refs, n_points, head_side=HEAD_SIDE, normalize_by_bg=NORMALIZE_BY_BG):
    """(T, 3) = [intensité, validité, u], comme les colonnes spine_w_XX du CSV."""
    u = np.linspace(-1, 1, n_points).astype(np.float32)
    mask = extract_striatum_mask(img)
    seg = segment_head_tail_pixels(img, mask, head_side) if mask is not None else None
    sp = extract_spine(img, mask, seg, head_side) if seg is not None else None
    if sp is None:
        z = np.zeros(n_points, np.float32)
        return np.stack([z, z, u], -1)
    sc, path = sp['s_centered'], sp['path']
    st = u_to_s(u, refs)
    inside = (st >= sc[0]) & (st <= sc[-1])
    vals = np.where(inside, _bilinear(img, np.interp(st, sc, path[:,0]),
                                      np.interp(st, sc, path[:,1])), np.nan)
    if normalize_by_bg:
        bg, _ = compute_background_stats(img, mask)
        if bg and not np.isnan(bg):
            vals = vals / (bg + 1e-6)
    return np.stack([np.nan_to_num(vals, 0.).astype(np.float32),
                     np.isfinite(vals).astype(np.float32), u], -1)


def head_tail_masks(img, head_side=HEAD_SIDE):
    mask = extract_striatum_mask(img)
    seg = segment_head_tail_pixels(img, mask, head_side) if mask is not None else None
    if seg is None:
        z = np.zeros(img.shape, np.float32)
        return z, z
    return seg['head_mask'].astype(np.float32), seg['tail_mask'].astype(np.float32)