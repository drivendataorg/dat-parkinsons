"""
striatum_geometry.py
====================
Geometrie du striatum : masque, segmentation tete/queue, spine, features.

Module PARTAGE entre le notebook d'entrainement et le pipeline d'inference.
Toute divergence entre les deux produirait un decalage silencieux : c'est
exactement pour l'eviter que ce code est isole ici.
"""
import heapq
import numpy as np
from scipy.ndimage import label, distance_transform_edt, gaussian_filter1d
from skimage.morphology import convex_hull_image, dilation, disk
from skimage.measure import regionprops

EPS = 1e-6
DIRECTIONS = {'right': (0, 1), 'left': (0, -1), 'up': (-1, 0), 'down': (1, 0)}
SEED_WEIGHTS = {'position': 0.2, 'convexity': 0.3, 'brightness': 0.2, 'width': 0.3}
NB8 = [(-1,0,1.),(1,0,1.),(0,-1,1.),(0,1,1.),
       (-1,-1,np.sqrt(2)),(-1,1,np.sqrt(2)),(1,-1,np.sqrt(2)),(1,1,np.sqrt(2))]

# ---------------------------------------------------------------- masque
def extract_striatum_mask(img):
    H, W = img.shape
    cy, cx = H // 2, W // 2
    vmax = img.max()
    if vmax == 0:
        return None
    for t in np.arange(0.50, 0.95, 0.05):
        labeled, n = label(img > vmax * t)
        if n == 0:
            continue
        best, cl = float('inf'), 0
        for i in range(1, n + 1):
            c = np.argwhere(labeled == i)
            if len(c) == 0:
                continue
            my, mx = c.mean(axis=0)
            d = (my - cy) ** 2 + (mx - cx) ** 2
            if d < best:
                best, cl = d, i
        if cl == 0:
            continue
        m = labeled == cl
        if not (m[0, :].any() or m[-1, :].any() or m[:, 0].any() or m[:, -1].any()):
            return m
    labeled, _ = label(img > vmax * 0.9)
    if labeled.max() > 0:
        sizes = np.bincount(labeled.flat)[1:]
        return labeled == (sizes.argmax() + 1)
    return img > vmax * 0.9

def background_stats(img, mask, exclude_dilation=2):
    if mask is None or mask.sum() == 0:
        return np.nan, np.nan
    bg = img[~dilation(mask, disk(exclude_dilation))]
    if bg.size == 0:
        return np.nan, np.nan
    return float(bg.mean()), float(bg.std())

# ---------------------------------------------------------------- tete / queue
def seed_score(img, mask, head_side='right', weights=None):
    w = weights or SEED_WEIGHTS
    H, W = img.shape
    dy, dx = DIRECTIONS[head_side]
    yy, xx = np.mgrid[0:H, 0:W]
    pos = (yy - (H - 1) / 2) * dy + (xx - (W - 1) / 2) * dx
    pos = (pos - pos.min()) / (pos.max() - pos.min() + EPS)
    try:
        defic = convex_hull_image(mask) & ~mask
        cd = distance_transform_edt(~defic) if defic.any() else distance_transform_edt(np.ones_like(mask))
        conv = cd / (cd[mask].max() + EPS)
    except Exception:
        conv = np.zeros((H, W))
    v = img[mask]
    bright = (img - v.min()) / (v.max() - v.min() + EPS)
    width = distance_transform_edt(mask)
    wid = width / (width.max() + EPS)
    s = (w['position'] * pos + w['convexity'] * conv + w['brightness'] * bright + w['width'] * wid)
    return np.where(mask, s, -np.inf)

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
        for dy, dx, w in NB8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not vis[ny, nx]:
                nd = d + w
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(pq, (nd, ny, nx))
    return dist

def segment_head_tail(img, mask, head_side='right', drop_frac=0.5, bin_width=1.0):
    if mask is None or mask.sum() == 0:
        return None
    sy, sx = np.unravel_index(np.argmax(seed_score(img, mask, head_side)), img.shape)
    geo = geodesic_distance(mask, (sy, sx))
    fv = geo[mask]
    fv = fv[np.isfinite(fv)]
    if fv.size == 0 or fv.max() < bin_width:
        return {'seed': (sx, sy), 'geo': geo, 'd_split': 0.0,
                'head_mask': mask.copy(), 'tail_mask': np.zeros_like(mask)}
    bins = np.arange(bin_width, fv.max() + bin_width, bin_width)
    Wf = np.array([int((mask & (geo >= b - bin_width) & (geo < b)).sum()) for b in bins])
    pk = int(np.argmax(Wf))
    below = np.where((np.arange(len(Wf)) > pk) & (Wf < drop_frac * Wf[pk]))[0]
    d_split = float(bins[below[0]]) if below.size else float(bins[-1])
    return {'seed': (sx, sy), 'geo': geo, 'd_split': d_split,
            'head_mask': mask & (geo <= d_split),
            'tail_mask': mask & (geo > d_split) & np.isfinite(geo)}

# ---------------------------------------------------------------- spine
def _min_cost_path(img, mask, start, end, gamma=2.0):
    H, W = img.shape
    vmax = img[mask].max() if mask.any() else 1.0
    cost = np.where(mask, (vmax / (img + EPS)) ** gamma, np.inf)
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
    wy = y - y0; wx = x - x0
    return (img[y0,x0]*(1-wy)*(1-wx) + img[y1,x0]*wy*(1-wx)
            + img[y0,x1]*(1-wy)*wx + img[y1,x1]*wy*wx)

def extract_spine(img, mask, seg, head_side='right', gamma=2.0, smooth_sigma=1.0):
    """Spine tete -> queue, orientation CANONIQUE imposee (sans quoi une partie
    des profils serait inversee)."""
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
    hidx = [i for i,(y,x) in enumerate(ip)
            if 0 <= y < head.shape[0] and 0 <= x < head.shape[1] and head[y,x]]
    vp = _bilinear(img, path[:,0], path[:,1])
    i_seed = max(hidx, key=lambda i: vp[i]) if hidx else int(np.argmax(vp))
    sc = s - s[i_seed]
    return {'path': path, 's_centered': sc, 'i_seed': i_seed,
            'len_head': float(-sc[0]), 'len_tail': float(sc[-1]), 'len_total': float(s[-1])}

def sample_spine(img, spine, s_targets):
    sc, path = spine['s_centered'], spine['path']
    inside = (s_targets >= sc[0]) & (s_targets <= sc[-1])
    y = np.interp(s_targets, sc, path[:,0])
    x = np.interp(s_targets, sc, path[:,1])
    return np.where(inside, _bilinear(img, y, x), np.nan), inside

def u_to_s(u_grid, refs):
    return np.where(u_grid < 0, u_grid * refs['head'], u_grid * refs['tail'])

def spine_profile(img, refs, n_points, head_side='right', normalize_by_bg=True):
    """Retourne le tableau (T, 3) [intensite, validite, u] attendu par SpineEncoder."""
    u = np.linspace(-1, 1, n_points).astype(np.float32)
    mask = extract_striatum_mask(img)
    seg = segment_head_tail(img, mask, head_side) if mask is not None else None
    sp = extract_spine(img, mask, seg, head_side) if seg is not None else None
    if sp is None:
        return np.stack([np.zeros(n_points, np.float32),
                         np.zeros(n_points, np.float32), u], -1)
    vals, inside = sample_spine(img, sp, u_to_s(u, refs))
    if normalize_by_bg:
        bg, _ = background_stats(img, mask)
        if bg and not np.isnan(bg):
            vals = vals / (bg + EPS)
    return np.stack([np.nan_to_num(vals, 0.).astype(np.float32),
                     np.isfinite(vals).astype(np.float32), u], -1)

def head_tail_masks(img, head_side='right'):
    """(head, tail) booleens, ou zeros si la segmentation echoue."""
    mask = extract_striatum_mask(img)
    if mask is None or mask.sum() == 0:
        z = np.zeros(img.shape, np.float32)
        return z, z
    seg = segment_head_tail(img, mask, head_side)
    if seg is None:
        z = np.zeros(img.shape, np.float32)
        return z, z
    return seg['head_mask'].astype(np.float32), seg['tail_mask'].astype(np.float32)