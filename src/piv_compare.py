# Compares bulk-motion methods and trackers on the synthetic sequences, plus a large-motion stress test.
# Output: piv_outputs/<n>/ (motion.csv, tracking.csv, results.xlsx, stress/, figures).
# Uses: scale.py, hybrid_segment.py. Used by: cls_compare.py, cls_compare2.py.
import argparse, contextlib, glob, io, os, re, time, warnings
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import scale as SC

warnings.filterwarnings('ignore', category=RuntimeWarning)

# settings in px at the reference scale (converted by _apply_scale)
MIN_AREA = 50                     # px: smallest particle detected and counted (Song et al. 2025)
QF_TAU = 0.3                      # hybrid_segment cut-off: low = more detections (recall) over correct ones (precision)
FLOW_SCALE = 0.5                  # dense optical flow runs on frames shrunk by this factor (speed)

# stage 1: bulk motion
PIV_WINDOW, PIV_OVERLAP = 320, 160     # Song et al. (2025): 320 x 320 px windows, 50 % overlap
UOD_EPS, UOD_THRESH = 0.1, 2.0         # Westerweel & Scarano (2005) recommended values
OPENPIV_EXT = dict(window=128, search=256, overlap=128)    # finds shifts up to +-64 px
OPENPIV_MULTI = dict(windows=(256, 128, 64), overlaps=(128, 64, 32))
RANSAC_GATE = 80                  # px: centroid pairs further apart than this are not tried
RANSAC_THRESH = 4                 # px: a pair agreeing with the fit to within this is an inlier
FARNEBACK = dict(pyr_scale=0.5, levels=5, winsize=31, iterations=5, poly_n=7, poly_sigma=1.5, flags=0)
ILK_RADIUS = 15                   # px (on the shrunk frame)

# stage 2: Song's tracker (values from main_tracking.m)
SONG_MARGIN = (16, 32)            # px added round the template: small (<= 48 px) / large particles
SONG_SIZE_SWITCH = 48             # px
SONG_CORR_END = 0.5               # below this a lost particle ends with -2 (weak correlation)
SONG_CORR_STRONG = 0.7            # paper: a match needs r > 0.7; also used by -4 and the backward pass
SONG_AREA_SMALL, SONG_AREA_LARGE = 100, 2000   # px: end codes -3 / -5
AGG_TOL = 0.25                    # merged area within +-25 % of the sum of the parts = aggregation
GAP_FRAMES = 36                   # frames searched for re-appearance: 12 h at 20 min (Song et al. 2025)
GAP_NEAR, GAP_NEAR_AREA = 8, 1.5  # px, area ratio: accept without correlation
GAP_FAR, GAP_FAR_AREA, GAP_CORR = 16, 2.0, 0.5   # px, area ratio, correlation
WIDE_MAX_MARGIN = 160             # px: largest margin tracking tries

# stage 2: other trackers
LINK_GATE = 50                    # px from the predicted position (Song's 99th percentile was 54.5 px)
AREA_WEIGHT = 20                  # px of cost per e-fold change in area (hungarian)
NO_LINK_COST = 25                 # px: cost of leaving a particle without a partner (hungarian)
LK_WIN, LK_LEVELS, LK_SNAP = 31, 4, 8   # px window, pyramid levels, px snap distance
REID_MAX = 200                    # px: reid looks this far from the predicted position for a lost particle
REID_AREA = 1.5                   # largest area ratio allowed between the two views
REID_CORR = 0.7                   # smallest correlation of the two cut-outs (Song's 0.7)

# scoring
OWN_BINS = (5, 20)                # px of motion of a particle relative to its neighbours
DISP_BINS = (10, 25, 50)          # px of total motion
LOCAL_R = 150                     # px: neighbours used to define the local bulk motion
NEAR_PX, SEARCH_PX = 3, 16        # px: "accurate" prediction / prediction inside Song's search margin

# smaller pixel settings (also scaled)
SONG_CLOSE_R = 2                  # px: radius of the closing in Song's double threshold
SONG_TOP_PX = 50                  # px: end code -4 compares the mean of the particle's pixels after its 50 brightest
RANSAC_REFINE = 3                 # px: inlier distance of the second RANSAC refinement
APPEAR_PAD = 4                    # px: padding round a cut-out in the appearance correlation
OPENPIV_MAXDISP = 300             # px: largest displacement OpenPIV's multi-pass keeps
STRESS_BULK = (0, 40)             # px: bulk motion at the cup rim in the stress test
STRESS_DISP = (0, 10, 25, 50, 100, 150)   # px: extra motion of the moved particles
STRESS_AREA = (100, 5000)         # px: particles that may be moved
STRESS_EDGE, STRESS_GAP, STRESS_NEAR, STRESS_RING = 8, 4, 3, 10   # px: distance from the edge, free space, ring
FIG_PAD, FIG_TILE = 60, 64        # px: margin round a trajectory, smallest cut-out in the trajectory figure
EXAMPLE_PAD = 50                  # px: margin in the stress example
PIV_MODE = 'native'               # 'native': PIV on the full frames; 'reference': on shrunk frames (faster)
PIV_SCALE = 1.0                   # set from PIV_MODE by _apply_scale (1 = full frames)

_REF = {k: v for k, v in dict(globals()).items() if k.isupper() and not k.startswith('_')}   # reference values


def _apply_scale():
    """Convert the pixel settings above to the current resolution."""
    g, r, L, iL, iA = globals(), _REF, SC.length, SC.ilength, SC.iarea
    g['MIN_AREA'] = SC.MIN_AREA
    g['FLOW_SCALE'] = r['FLOW_SCALE'] / SC.K              # shrunk frames keep the reference pixel size
    g['PIV_WINDOW'] = SC.mult8(r['PIV_WINDOW'])
    g['PIV_OVERLAP'] = g['PIV_WINDOW'] // 2
    w, sa = SC.mult8(r['OPENPIV_EXT']['window']), SC.mult8(r['OPENPIV_EXT']['search'])
    g['OPENPIV_EXT'] = dict(window=w, search=sa, overlap=w)
    ws = [SC.mult8(x) for x in r['OPENPIV_MULTI']['windows']]
    g['OPENPIV_MULTI'] = dict(windows=tuple(ws), overlaps=tuple(x // 2 for x in ws))
    g['OPENPIV_MAXDISP'] = iL(r['OPENPIV_MAXDISP'])
    g['RANSAC_GATE'], g['RANSAC_THRESH'], g['RANSAC_REFINE'] = L(r['RANSAC_GATE']), L(r['RANSAC_THRESH']), L(r['RANSAC_REFINE'])
    g['SONG_MARGIN'] = tuple(iL(m) for m in r['SONG_MARGIN'])
    g['SONG_SIZE_SWITCH'] = iL(r['SONG_SIZE_SWITCH'])
    g['SONG_AREA_SMALL'], g['SONG_AREA_LARGE'] = iA(r['SONG_AREA_SMALL']), iA(r['SONG_AREA_LARGE'])
    g['SONG_CLOSE_R'], g['SONG_TOP_PX'] = iL(r['SONG_CLOSE_R']), iA(r['SONG_TOP_PX'])
    g['GAP_NEAR'], g['GAP_FAR'] = L(r['GAP_NEAR']), L(r['GAP_FAR'])
    g['WIDE_MAX_MARGIN'] = iL(r['WIDE_MAX_MARGIN'])
    g['LINK_GATE'], g['AREA_WEIGHT'], g['NO_LINK_COST'] = L(r['LINK_GATE']), L(r['AREA_WEIGHT']), L(r['NO_LINK_COST'])
    g['LK_WIN'], g['LK_SNAP'] = SC.odd(r['LK_WIN']), L(r['LK_SNAP'])
    g['REID_MAX'] = L(r['REID_MAX'])
    g['APPEAR_PAD'] = iL(r['APPEAR_PAD'])
    g['OWN_BINS'] = tuple(iL(v) for v in r['OWN_BINS'])
    g['DISP_BINS'] = tuple(iL(v) for v in r['DISP_BINS'])
    g['LOCAL_R'] = L(r['LOCAL_R'])
    g['NEAR_PX'], g['SEARCH_PX'] = iL(r['NEAR_PX']), iL(r['SEARCH_PX'])
    g['STRESS_BULK'] = tuple(iL(v, 0) for v in r['STRESS_BULK'])
    g['STRESS_DISP'] = tuple(iL(v, 0) for v in r['STRESS_DISP'])
    g['STRESS_AREA'] = tuple(iA(v) for v in r['STRESS_AREA'])
    for k in ('STRESS_EDGE', 'STRESS_GAP', 'STRESS_NEAR', 'STRESS_RING', 'FIG_PAD', 'FIG_TILE', 'EXAMPLE_PAD'):
        g[k] = iL(r[k])
    g['PIV_SCALE'] = 1.0 if g['PIV_MODE'] == 'native' else 1.0 / SC.K


def set_piv_mode(mode):
    """'native': PIV on full frames; 'reference': on frames shrunk to the reference scale (about 3x faster)."""
    if mode not in ('native', 'reference'):
        raise ValueError("PIV mode must be 'native' or 'reference'")
    globals()['PIV_MODE'] = mode
    _apply_scale()


def settings():
    """The pixel settings in use (for settings.json)."""
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in globals().items() if k in _REF}


SC.register(_apply_scale)


# frames kept small in memory
class Frames:
    """A list of frames stored compactly (8-bit images, 16-bit label maps) and handed out as `dtype` when used."""

    def __init__(self, arrays, dtype):
        self.arrays, self.dtype = list(arrays), dtype

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, i):
        if isinstance(i, slice):
            return Frames(self.arrays[i], self.dtype)
        return self.arrays[i].astype(self.dtype)

    def __iter__(self):
        return (a.astype(self.dtype) for a in self.arrays)


def compact_labels(lab):
    """A label map in the smallest unsigned type that holds it."""
    return lab.astype(np.uint16) if lab.max() < 65536 else lab.astype(np.uint32)


# loading and detection
def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def big_only(lab, min_area=None):
    min_area = MIN_AREA if min_area is None else min_area
    area = np.bincount(lab.ravel())
    small = area < min_area
    small[0] = True
    lab = lab.copy()
    lab[small[lab]] = 0
    return lab


def song_segment(img, mask, high):
    """Song et al. (2025) double threshold (as in seg_compare.py)."""
    seeds = big_only(label(img > high, connectivity=2), MIN_AREA) > 0
    low = cv2.morphologyEx((img > 0.15 * 255).astype(np.uint8), cv2.MORPH_CLOSE, disk(SONG_CLOSE_R)) > 0
    lab = label(low, connectivity=2)
    ids = np.unique(lab[seeds])
    keep = np.isin(lab, ids[ids > 0]) & mask
    return big_only(label(keep, connectivity=2))


def frame_files(seq_dir):
    files = sorted(glob.glob(os.path.join(seq_dir, 'bgrem', 'bgrem_*.png')))
    return files, [int(re.findall(r'\d+', os.path.basename(f))[-1]) for f in files]


def load_mask(path, shape):
    return cv2.imread(path, 0) > 0 if path and os.path.exists(path) else np.ones(shape, bool)


QF = {}                           # hybrid_segment quality filter: set with use_hybrid_segment()
DET_CACHE = None                  # folder of saved hybrid_segment label maps (set by use_hybrid_segment); None = no cache


def use_hybrid_segment(filter_file, tau=QF_TAU, data_dir=None, frame_step=2):
    """Load the hybrid_segment quality filter (quality_filter.joblib) and set its cut-off tau."""
    import joblib
    import hybrid_segment as hq
    if data_dir is not None:
        model, saved_tau = hq.ensure_filter(data_dir, filter_file, frame_step)
    else:
        model, saved_tau = joblib.load(filter_file)
    QF.update(model=model, tau=float(tau), saved_tau=float(saved_tau))
    # detections are cached next to the filter
    import hashlib
    st = os.stat(filter_file)
    key = hashlib.md5(f'{st.st_size}_{int(st.st_mtime)}_{float(tau)}_{SC.UM_PER_PX}_{SC.MIN_AREA}'.encode()).hexdigest()[:8]
    globals()['DET_CACHE'] = os.path.join(os.path.dirname(os.path.abspath(filter_file)), f'detections_tau{tau}_{key}')
    print(f'hybrid_segment filter loaded; tau {tau} (the filter was saved with tau {saved_tau:.2f}); detections saved in {DET_CACHE}')


def segment(detections, img, mask, high, truth=None):
    """Instance labels of one frame."""
    if detections == 'truth':
        return big_only(truth)
    if detections == 'threshold':
        return big_only(label(ndi.binary_fill_holes((img > 0.5 * high) & mask), connectivity=2).astype(np.int64))
    if detections == 'hybrid_segment':
        import hybrid_segment as hq
        if not QF:
            raise RuntimeError('call use_hybrid_segment(filter_file, tau) first')
        lab, _ = hq.detect(img, mask, high, QF['model'], QF['tau'])
        return big_only(lab.astype(np.int64))
    return song_segment(img, mask, high)


def load_sequence(files, mask_path=None, truth_files=None, detections='song', high_files=None, high=None):
    """Read frames, detect particles and collect everything the trackers need."""
    raw = [cv2.imread(f, 0) for f in files]                            # 8-bit frames as read
    mask = load_mask(mask_path, raw[0].shape)
    imgs = Frames([np.where(mask, im, 0).astype(np.uint8) for im in raw], np.float32)
    truth = None
    if truth_files:
        truth = Frames([compact_labels(np.where(mask, cv2.imread(f, cv2.IMREAD_UNCHANGED), 0)) for f in truth_files],
                       np.int64)
    # Song's level: mean of the per-frame Otsu thresholds
    if high is None:
        if high_files and list(high_files) != list(files):
            high = float(np.mean([threshold_otsu(cv2.imread(f, 0)[mask]) for f in high_files]))
        else:
            high = float(np.mean([threshold_otsu(im[mask]) for im in raw]))
    high = float(high)
    rows, crops, bmasks = [], [], []
    for f in range(len(imgs)):
        im = imgs[f]
        src = raw[f].astype(np.float32) if detections == 'hybrid_segment' else im   # hybrid_segment: unmasked frame, as trained
        cache = None
        if detections == 'hybrid_segment' and DET_CACHE:
            seq = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(files[f]))))
            cache = os.path.join(DET_CACHE, seq, os.path.basename(files[f])[:-4] + '.npz')
        if cache and os.path.exists(cache):
            lab = np.load(cache)['lab'].astype(np.int64)
        else:
            lab = segment(detections, src, mask, high, truth[f] if truth is not None else None)
            if cache:
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                np.savez_compressed(cache + '.tmp.npz', lab=compact_labels(lab))
                os.replace(cache + '.tmp.npz', cache)         # complete files only (safe if the session dies)
        if truth is not None:
            t_area = np.bincount(truth[f].ravel())
        for r in regionprops(lab):
            y0, x0, y1, x1 = r.bbox
            uid = 0
            if truth is not None:
                if detections == 'truth':
                    uid = r.label
                else:
                    vals = truth[f][r.coords[:, 0], r.coords[:, 1]]
                    vals = vals[vals > 0]
                    if len(vals):
                        u, n = np.unique(vals, return_counts=True)
                        k = np.argmax(n)
                        if n[k] >= 0.5 * r.area or n[k] >= 0.5 * t_area[u[k]]:
                            uid = int(u[k])
            rows.append(dict(pid=len(rows), frame=f, x=r.centroid[1], y=r.centroid[0], area=int(r.area),
                             x0=x0, y0=y0, x1=x1, y1=y1, uid=uid))
            crops.append(im[y0:y1, x0:x1].copy())
            bmasks.append(r.image.copy())
        raw[f] = None                                                  # the unmasked frame is no longer needed
    P = pd.DataFrame(rows, columns=['pid', 'frame', 'x', 'y', 'area', 'x0', 'y0', 'x1', 'y1', 'uid'])
    ctx = dict(high=high, crops=crops, bmasks=bmasks, dets=[P[P.frame == f] for f in range(len(imgs))])
    return dict(imgs=imgs, mask=mask, truth=truth, P=P, ctx=ctx)


def load_seq(seq_dir, detections='song', max_frames=None, frames=None):
    """Frames, mask, detections and truth of one synthetic sequence."""
    files_all, nums_all = frame_files(seq_dir)
    files_all, nums_all = files_all[:max_frames], nums_all[:max_frames]
    pick = list(range(len(files_all))) if frames is None else [i for i in frames if i < len(files_all)]
    files, nums = [files_all[i] for i in pick], [nums_all[i] for i in pick]
    tf = [os.path.join(seq_dir, 'truth', 'labels', f'frame_{n:04d}.png') for n in nums]
    tf = tf if all(os.path.exists(f) for f in tf) else None
    S = load_sequence(files, os.path.join(seq_dir, 'truth', 'static_mask.png'), tf, detections, files_all)
    S.update(name=os.path.basename(seq_dir), nums=nums, files=files, arrival=None)
    pf = os.path.join(seq_dir, 'truth', 'particles.csv')         # arrival frame of every particle
    if os.path.exists(pf):
        p = pd.read_csv(pf)
        if {'uid', 'arrival_frame'} <= set(p.columns):
            S['arrival'] = p.set_index('uid').arrival_frame > nums[0]   # True = arrived during this sequence
    return S


# ground-truth motion
def truth_table(truth):
    """Visible centroid and area of every truth particle (>= MIN_AREA px) in every frame."""
    h, w = truth[0].shape
    yy, xx = np.mgrid[:h, :w]
    xx, yy = xx.ravel().astype(float), yy.ravel().astype(float)
    out = []
    for f, lab in enumerate(truth):
        L = lab.ravel()
        a = np.bincount(L)
        sx, sy = np.bincount(L, xx, len(a)), np.bincount(L, yy, len(a))
        ids = np.nonzero(a >= MIN_AREA)[0]
        ids = ids[ids > 0]
        out.append(pd.DataFrame(dict(frame=f, uid=ids, x=sx[ids] / a[ids], y=sy[ids] / a[ids], area=a[ids])))
    return pd.concat(out, ignore_index=True)


def truth_pairs(T):
    """True displacement of every particle seen in two consecutive frames."""
    a = T.rename(columns=dict(x='xa', y='ya', area='aa'))
    b = T.rename(columns=dict(x='xb', y='yb', area='ab'))
    b = b.assign(frame=b.frame - 1)
    d = a.merge(b, on=['frame', 'uid'])
    d['dx'], d['dy'] = d.xb - d.xa, d.yb - d.ya
    d['disp'] = np.hypot(d.dx, d.dy)
    r = d.ab / d.aa
    d['usable'] = (r > 0.7) & (r < 1 / 0.7)
    d['own'] = np.nan
    for f, g in d[d.usable].groupby('frame'):
        xy = g[['xa', 'ya']].to_numpy()
        dd = g[['dx', 'dy']].to_numpy()
        tree = cKDTree(xy)
        own = []
        for i, nb in enumerate(tree.query_ball_point(xy, LOCAL_R)):
            nb = [j for j in nb if j != i]
            ref = np.median(dd[nb], 0) if len(nb) >= 3 else np.median(dd, 0)
            own.append(np.hypot(*(dd[i] - ref)))
        d.loc[g.index, 'own'] = own
    return d


def own_bin(v):
    lo, hi = OWN_BINS
    return np.select([v < lo, v < hi, v >= hi], [f'with gel <{lo} px', f'own {lo}-{hi} px', f'own >={hi} px'], 'unclear')


def own_column():
    """Name of the tracking column for particles that moved >= OWN_BINS[1] px on their own (the key column)."""
    return f'Links found, own >={OWN_BINS[1]} px %'


def disp_bin(v):
    a, b, c = DISP_BINS
    return np.select([v < a, v < b, v < c], [f'<{a} px', f'{a}-{b} px', f'{b}-{c} px'], f'>={c} px')


# stage 1: motion fields
class Field:
    """Displacement (u, v) in px per frame from frame a to frame b, at any point (x, y) of frame a."""

    def __init__(self, kind, **kw):
        self.kind = kind
        self.__dict__.update(kw)
        if kind == 'grid':
            U, V = (np.where(np.isfinite(z), z, np.nanmedian(z) if np.isfinite(z).any() else 0) for z in (self.U, self.V))
            if len(self.xs) < 2 or len(self.ys) < 2:
                self.kind, self.M = 'affine', np.array([[1, 0, np.median(U)], [0, 1, np.median(V)]], float)
            else:
                k_y, k_x = min(3, len(self.ys) - 1), min(3, len(self.xs) - 1)   # cubic where possible
                self.su = RectBivariateSpline(self.ys, self.xs, U, kx=k_y, ky=k_x)
                self.sv = RectBivariateSpline(self.ys, self.xs, V, kx=k_y, ky=k_x)

    def __call__(self, x, y):
        x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
        if self.kind == 'affine':
            M = self.M
            return M[0, 0] * x + M[0, 1] * y + M[0, 2] - x, M[1, 0] * x + M[1, 1] * y + M[1, 2] - y
        if self.kind == 'grid':
            cx, cy = np.clip(x, self.xs[0], self.xs[-1]), np.clip(y, self.ys[0], self.ys[-1])
            return self.su.ev(cy, cx), self.sv.ev(cy, cx)
        s = self.scale
        h, w = self.U.shape
        c = [np.clip((y + 0.5) * s - 0.5, 0, h - 1), np.clip((x + 0.5) * s - 0.5, 0, w - 1)]
        return ndi.map_coordinates(self.U, c, order=1) / s, ndi.map_coordinates(self.V, c, order=1) / s


def shrink(im, s=None):
    s = FLOW_SCALE if s is None else s
    return im if s == 1 else cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def u8(im):
    return np.clip(im, 0, 255).astype(np.uint8)


def uod(u, v, eps=UOD_EPS, thr=UOD_THRESH):
    """Universal outlier detection (normalised median test on the 3 x 3 neighbours, Westerweel & Scarano 2005)."""
    h, w = u.shape
    gy, gx = np.mgrid[:h, :w]

    def neigh(z):
        p = np.pad(z, 1, constant_values=np.nan)
        return np.stack([p[i:i + h, j:j + w] for i in range(3) for j in range(3) if (i, j) != (1, 1)])

    def plane(z):
        ok = np.isfinite(z)
        if ok.sum() < 3:
            return np.full_like(z, np.nanmedian(z) if ok.any() else 0.0)
        A = np.c_[np.ones(ok.sum()), gx[ok], gy[ok]]
        c = np.linalg.lstsq(A, z[ok], rcond=None)[0]
        return c[0] + c[1] * gx + c[2] * gy

    out = []
    bad = ~np.isfinite(u) | ~np.isfinite(v)
    pl = [plane(np.where(bad, np.nan, z)) for z in (u, v)]
    res = [z.astype(float) - p for z, p in zip((u, v), pl)]
    if min(h, w) >= 3:
        for z in res:
            nb = neigh(z)
            med = np.nanmedian(nb, 0)
            r = np.nanmedian(np.abs(nb - med), 0)
            bad |= (np.abs(z - med) / (r + eps) > thr) & (np.isfinite(nb).sum(0) >= 3)
    for z, p in zip(res, pl):
        z = z.copy()
        for _ in range(3):
            z[bad] = np.nan
            fill = np.nanmedian(neigh(z), 0)
            z = np.where(bad & np.isfinite(fill), fill, z)
            bad_z = ~np.isfinite(z)
            if not bad_z.any():
                break
        z[~np.isfinite(z)] = 0.0                 # nothing nearby: use the plane itself
        out.append(z + p)
    return out[0], out[1]


def piv_fft(a, b, win, overlap):
    """One-pass FFT cross-correlation PIV (Willert & Gharib 1991), Gaussian 3-point sub-pixel peak."""
    h, w = a.shape
    win = int(min(win, h, w))
    step = max(1, win - overlap)
    ys0 = np.arange(0, h - win + 1, step)
    xs0 = np.arange(0, w - win + 1, step)
    sw = np.lib.stride_tricks.sliding_window_view
    # pick the windows in one step
    iy, ix = ys0[:, None], xs0[None, :]
    wa = sw(a, (win, win))[iy, ix].reshape(-1, win, win).astype(np.float32)
    wb = sw(b, (win, win))[iy, ix].reshape(-1, win, win).astype(np.float32)
    empty = (wa.std((1, 2)) < 1e-3) | (wb.std((1, 2)) < 1e-3)
    wa = wa - wa.mean((1, 2), keepdims=True)
    wb = wb - wb.mean((1, 2), keepdims=True)
    R = np.fft.irfft2(np.conj(np.fft.rfft2(wa)) * np.fft.rfft2(wb), s=(win, win))
    R = np.fft.fftshift(R, axes=(1, 2))
    n = R.shape[0]
    k = R.reshape(n, -1).argmax(1)
    py, px = np.unravel_index(k, (win, win))
    py, px = np.clip(py, 1, win - 2), np.clip(px, 1, win - 2)
    idx = np.arange(n)
    c = R[idx, py, px]
    lg = lambda z: np.log(np.maximum(z / np.maximum(c, 1e-12), 1e-6))   # normalise by the peak first
    l, r_, t, bo = R[idx, py, px - 1], R[idx, py, px + 1], R[idx, py - 1, px], R[idx, py + 1, px]
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = (lg(l) - lg(r_)) / (2 * lg(l) - 4 * lg(c) + 2 * lg(r_))
        dy = (lg(t) - lg(bo)) / (2 * lg(t) - 4 * lg(c) + 2 * lg(bo))
    dx = np.where(np.isfinite(dx) & (np.abs(dx) < 1), dx, 0)
    dy = np.where(np.isfinite(dy) & (np.abs(dy) < 1), dy, 0)
    U = (px + dx - win // 2).astype(float)
    V = (py + dy - win // 2).astype(float)
    U[empty], V[empty] = np.nan, np.nan
    return xs0 + (win - 1) / 2, ys0 + (win - 1) / 2, U.reshape(len(ys0), len(xs0)), V.reshape(len(ys0), len(xs0))


def motion_none(a, b, mask, ctx):
    return Field('affine', M=np.array([[1, 0, 0], [0, 1, 0]], float))


def motion_phasecorr(a, b, mask, ctx):
    from skimage.registration import phase_cross_correlation
    shift = phase_cross_correlation(a, b, upsample_factor=10)[0]       # shift that moves b back onto a
    return Field('affine', M=np.array([[1, 0, -shift[1]], [0, 1, -shift[0]]], float))


def motion_ransac(a, b, mask, ctx):
    """Shift + rotation (+ uniform scale) fitted with RANSAC to every plausible centroid pair."""
    A, B = ctx['dets_a'], ctx['dets_b']
    ident = motion_none(a, b, mask, ctx)
    if len(A) < 3 or len(B) < 3:
        return ident
    pa, pb = A[['x', 'y']].to_numpy(), B[['x', 'y']].to_numpy()
    aa, ab = A.area.to_numpy(float), B.area.to_numpy(float)
    i, j = [], []
    for k, nb in enumerate(cKDTree(pb).query_ball_point(pa, RANSAC_GATE)):
        for m in nb:
            if 0.5 < ab[m] / aa[k] < 2:
                i.append(k)
                j.append(m)
    if len(i) < 3:
        return ident
    M, _ = cv2.estimateAffinePartial2D(pa[i].astype(np.float32), pb[j].astype(np.float32), method=cv2.RANSAC,
                                       ransacReprojThreshold=RANSAC_THRESH, maxIters=5000, confidence=0.999)
    if M is None:
        return ident
    tree = cKDTree(pb)
    for _ in range(2):
        pred = pa @ M[:, :2].T + M[:, 2]
        dist, m = tree.query(pred)
        ok = (dist < 2 * RANSAC_THRESH) & (ab[m] / aa > 0.5) & (ab[m] / aa < 2)
        if ok.sum() < 3:
            break
        M2, _ = cv2.estimateAffinePartial2D(pa[ok].astype(np.float32), pb[m[ok]].astype(np.float32),
                                            method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REFINE)
        if M2 is None:
            break
        M = M2
    return Field('affine', M=M.astype(float))


def motion_song_piv(a, b, mask, ctx):
    xs, ys, U, V = piv_fft(a, b, PIV_WINDOW, PIV_OVERLAP)
    U, V = uod(U, V)
    return Field('grid', xs=xs, ys=ys, U=U, V=V)


def motion_openpiv_ext(a, b, mask, ctx):
    from openpiv import pyprocess, validation
    o = OPENPIV_EXT
    with contextlib.redirect_stdout(io.StringIO()):
        u, v, s2n = pyprocess.extended_search_area_piv(a, b, window_size=o['window'], overlap=o['overlap'],
                                                       search_area_size=o['search'], subpixel_method='gaussian',
                                                       sig2noise_method='peak2peak')
        x, y = pyprocess.get_coordinates(a.shape, o['search'], o['overlap'])
        bad = validation.sig2noise_val(s2n, threshold=1.05) | ~np.isfinite(u) | ~np.isfinite(v)
    u, v = np.where(bad, np.nan, u), np.where(bad, np.nan, v)
    U, V = uod(u, v)
    return Field('grid', xs=x[0].astype(float), ys=y[:, 0].astype(float), U=U, V=V)


def motion_openpiv_multi(a, b, mask, ctx):
    from openpiv import windef
    s = windef.PIVSettings()
    ws, ov = list(OPENPIV_MULTI['windows']), list(OPENPIV_MULTI['overlaps'])
    while (min(a.shape) - ws[0]) // (ws[0] - ov[0]) + 1 < 4 and ws[-1] > 16:   # OpenPIV needs >= 4 rows
        ws, ov = [w // 2 for w in ws], [o // 2 for o in ov]                    # of windows: halve all
    s.windowsizes, s.overlap = tuple(ws), tuple(ov)
    s.num_iterations = len(s.windowsizes)
    s.min_max_u_disp = s.min_max_v_disp = (-OPENPIV_MAXDISP, OPENPIV_MAXDISP)   # default +-30 px deletes large motion
    s.show_plot = s.save_plot = False
    with contextlib.redirect_stdout(io.StringIO()):
        X, Y, U, V, _ = windef.simple_multipass(a, b, s)
    # OpenPIV returns y measured upwards from the bottom and v positive upwards: convert back
    return Field('grid', xs=X[0].astype(float), ys=(a.shape[0] - Y[:, 0]).astype(float),
                 U=np.asarray(U, float), V=-np.asarray(V, float))


def motion_farneback(a, b, mask, ctx):
    f = cv2.calcOpticalFlowFarneback(u8(shrink(a)), u8(shrink(b)), None, **FARNEBACK)
    return Field('dense', U=f[..., 0], V=f[..., 1], scale=FLOW_SCALE)


def motion_dis(a, b, mask, ctx):
    f = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(u8(shrink(a)), u8(shrink(b)), None)
    return Field('dense', U=f[..., 0], V=f[..., 1], scale=FLOW_SCALE)


def motion_tvl1(a, b, mask, ctx):
    from skimage.registration import optical_flow_tvl1
    v, u = optical_flow_tvl1(shrink(a) / 255, shrink(b) / 255)
    return Field('dense', U=u, V=v, scale=FLOW_SCALE)


def motion_ilk(a, b, mask, ctx):
    from skimage.registration import optical_flow_ilk
    v, u = optical_flow_ilk(shrink(a) / 255, shrink(b) / 255, radius=ILK_RADIUS)
    return Field('dense', U=u, V=v, scale=FLOW_SCALE)


_RAFT = {}


def motion_raft(a, b, mask, ctx):
    """torchvision RAFT-large, weights trained on Sintel/KITTI etc."""
    import torch
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    if not _RAFT:
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        _RAFT.update(model=raft_large(weights=Raft_Large_Weights.DEFAULT).eval().to(dev), dev=dev)
    sa, sb = shrink(a), shrink(b)
    h, w = sa.shape
    ph, pw = (-h) % 8, (-w) % 8                                  # RAFT needs sizes divisible by 8
    t = [torch.from_numpy(np.repeat(np.pad(s, ((0, ph), (0, pw)))[None, None], 3, 1) / 127.5 - 1).float()
         .to(_RAFT['dev']) for s in (sa, sb)]
    with torch.no_grad():
        f = _RAFT['model'](t[0], t[1])[-1][0].cpu().numpy()
    return Field('dense', U=f[0, :h, :w], V=f[1, :h, :w], scale=FLOW_SCALE)


def piv_at_scale(fn):
    """Run a PIV method on frames shrunk by PIV_SCALE and return the field in full-frame pixels."""
    def run(a, b, mask, ctx):
        s = PIV_SCALE
        if s == 1:
            return fn(a, b, mask, ctx)
        g = globals()
        keep = {k: g[k] for k in ('PIV_WINDOW', 'PIV_OVERLAP', 'OPENPIV_EXT', 'OPENPIV_MULTI', 'OPENPIV_MAXDISP')}
        g.update({k: _REF[k] for k in keep})                     # the shrunk frames have the reference pixel size
        try:
            f = fn(shrink(a, s), shrink(b, s), None, ctx)
        finally:
            g.update(keep)
        if f.kind == 'affine':
            M = f.M.copy()
            M[:, 2] /= s
            return Field('affine', M=M)
        return Field('grid', xs=(f.xs + 0.5) / s - 0.5, ys=(f.ys + 0.5) / s - 0.5, U=f.U / s, V=f.V / s)
    return run


MOTIONS = {'none': motion_none, 'phasecorr': motion_phasecorr, 'ransac': motion_ransac,
           'song_piv': piv_at_scale(motion_song_piv), 'openpiv_ext': piv_at_scale(motion_openpiv_ext),
           'openpiv_multi': piv_at_scale(motion_openpiv_multi),
           'farneback': motion_farneback, 'dis': motion_dis, 'tvl1': motion_tvl1, 'ilk': motion_ilk,
           'raft': motion_raft}


def default_motions():
    """Every motion method (RAFT only on a GPU)."""
    names = [m for m in MOTIONS if m != 'raft']
    try:
        import torch, torchvision  # noqa: F401
        if torch.cuda.is_available():
            names.append('raft')
    except ImportError:
        pass
    return names


def estimate(name, a, b, mask, dets_a, dets_b):
    return MOTIONS[name](a, b, mask, dict(dets_a=dets_a, dets_b=dets_b))


# stage 2: linking one frame pair
def crop_padded(img, x0, y0, w, h):
    out = np.zeros((h, w), np.float32)
    H, W = img.shape
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + w, W), min(y0 + h, H)
    if xb > xa and yb > ya:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = img[ya:yb, xa:xb]
    return out


def xcorr(template, img, cx, cy, margin):
    """Normalised cross-correlation of a template in a window around (cx, cy); returns (r, x, y)."""
    h, w = template.shape
    x0, y0 = int(round(cx - (w - 1) / 2)) - margin, int(round(cy - (h - 1) / 2)) - margin
    win = crop_padded(img, x0, y0, w + 2 * margin, h + 2 * margin)
    if template.std() < 1e-6 or win.std() < 1e-6:
        return np.nan, cx, cy
    R = np.nan_to_num(cv2.matchTemplate(win, template.astype(np.float32), cv2.TM_CCOEFF_NORMED), nan=-1)
    py, px = np.unravel_index(np.argmax(R), R.shape)
    return float(R[py, px]), x0 + px + (w - 1) / 2, y0 + py + (h - 1) / 2


def in_bbox(xp, yp, B, exclude=()):
    """Detections of B whose box contains (xp, yp)."""
    if not len(B):
        return None
    x0, y0, x1, y1 = (B[c].to_numpy() for c in ('x0', 'y0', 'x1', 'y1'))
    hit = (xp >= x0 - 0.5) & (xp <= x1 - 0.5) & (yp >= y0 - 0.5) & (yp <= y1 - 0.5)
    hit &= ~B.pid.isin(list(exclude)).to_numpy()
    if not hit.any():
        return None
    r = np.abs(xp - (x0 + x1 - 1) / 2) / (x1 - x0) + np.abs(yp - (y0 + y1 - 1) / 2) / (y1 - y0)
    r[~hit] = np.inf
    return int(B.pid.to_numpy()[np.argmin(r)])


def in_mask(mask, x, y):
    xi, yi = int(round(x)), int(round(y))
    return 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and bool(mask[yi, xi])


def song_end_code(a, corr, du, dv, mask, ctx):
    """Why a particle was not found (main_tracking.m lines 225-279, the conditions in code order)."""
    if not in_mask(mask, a.x + du, a.y + dv):
        return -1
    for cx in (a.x0, a.x1 - 1):
        for cy in (a.y0, a.y1 - 1):
            if not in_mask(mask, cx + du, cy + dv):
                return -1
    if not np.isfinite(corr) or corr < SONG_CORR_END:
        return -2
    if a.area <= SONG_AREA_SMALL:
        return -3
    if a.area > SONG_AREA_LARGE:
        return -5
    px = np.sort(ctx['crops'][a.pid].ravel())[::-1]
    if np.mean(px[SONG_TOP_PX:int(a.area)]) <= ctx['high'] and corr >= SONG_CORR_STRONG:
        return -4
    return -6


def song_one(a, du, dv, img_b, B, mask, ctx, wide=False, exclude=()):
    """One particle: predicted window, cross-correlation, box test (main_tracking.m)."""
    t = ctx['crops'][a.pid]
    cx, cy = (a.x0 + a.x1 - 1) / 2 + du, (a.y0 + a.y1 - 1) / 2 + dv
    m0 = SONG_MARGIN[0] if max(t.shape) <= SONG_SIZE_SWITCH else SONG_MARGIN[1]
    margins = [m0]
    while wide and margins[-1] * 2 <= WIDE_MAX_MARGIN:
        margins.append(margins[-1] * 2)
    first_corr = np.nan
    for k, m in enumerate(margins):
        corr, xp, yp = xcorr(t, img_b, cx, cy, m)
        if k == 0:
            first_corr = corr
        if np.isfinite(corr) and (k == 0 or corr >= SONG_CORR_STRONG):   # wider windows: r >= 0.7 (paper)
            b = in_bbox(xp, yp, B, exclude)
            if b is not None:
                return b, corr, 0
    return None, first_corr, song_end_code(a, first_corr, du, dv, mask, ctx)


def link_song(A, B, img_a, img_b, field, mask, ctx, wide=False):
    """Song's frame-to-frame step, including the overlap check of main_tracking.m lines 298-806."""
    du, dv = field(A.x.to_numpy(), A.y.to_numpy())
    res = {}
    rows = {a.pid: (a, du[i], dv[i]) for i, a in enumerate(A.itertuples(index=False))}
    for p, (a, u, v) in rows.items():
        res[p] = song_one(a, u, v, img_b, B, mask, ctx, wide)
    areaB = dict(zip(B.pid, B.area))
    claims = {}
    for p, (b, corr, code) in res.items():
        if b is not None:
            claims.setdefault(b, []).append(p)
    for b, ps in claims.items():
        if len(ps) < 2:
            continue
        tot = sum(rows[p][0].area for p in ps)
        if abs(areaB[b] - tot) <= AGG_TOL * tot:
            ctx['n_agg'] = ctx.get('n_agg', 0) + 1
            continue
        ps = sorted(ps, key=lambda p: -np.nan_to_num(res[p][1], nan=-9))
        brow = B[B.pid == b].iloc[0]
        img2 = img_b.copy()
        sl = (slice(int(brow.y0), int(brow.y1)), slice(int(brow.x0), int(brow.x1)))
        img2[sl][ctx['bmasks'][b]] = 0                       # blank the particle that was claimed
        for p in ps[1:]:
            a, u, v = rows[p]
            nb, corr, code = song_one(a, u, v, img2, B, mask, ctx, wide, exclude={b})
            taken = nb is not None and any(r[0] == nb for q, r in res.items() if q != p)
            res[p] = (None, corr, code if nb is None else -6) if (nb is None or taken) else (nb, corr, 0)
    links = {p: r[0] for p, r in res.items() if r[0] is not None}
    codes = {p: r[2] for p, r in res.items() if r[0] is None}
    return links, codes


def lost_code(A, du, dv, mask):
    return {p: (-1 if not in_mask(mask, x + u, y + v) else -6)
            for p, x, y, u, v in zip(A.pid, A.x, A.y, du, dv)}


def assign(C, ok, no_link):
    """Optimal one-to-one links with the option of NOT linking (Jaqaman et al. 2008)."""
    na, nb = C.shape
    big = 1e9
    top = np.hstack([np.where(ok, C, big), np.where(np.eye(na, dtype=bool), no_link, big)])
    bot = np.hstack([np.where(np.eye(nb, dtype=bool), no_link, big), np.where(ok.T, 0.0, big)])
    r, c = linear_sum_assignment(np.vstack([top, bot]))
    keep = (r < na) & (c < nb)
    return r[keep], c[keep]


def link_hungarian(A, B, img_a, img_b, field, mask, ctx):
    """Hungarian assignment; cost = distance to the predicted position + AREA_WEIGHT x |log area ratio|."""
    du, dv = field(A.x.to_numpy(), A.y.to_numpy())
    if not len(A) or not len(B):
        return {}, lost_code(A, du, dv, mask)
    pa = np.c_[A.x + du, A.y + dv]
    D = np.hypot(pa[:, None, 0] - B.x.to_numpy()[None], pa[:, None, 1] - B.y.to_numpy()[None])
    lr = np.abs(np.log(B.area.to_numpy()[None] / A.area.to_numpy()[:, None]))
    ok = (D <= LINK_GATE) & (lr <= np.log(3))
    r, c = assign(D + AREA_WEIGHT * lr, ok, NO_LINK_COST)
    pidA, pidB = A.pid.to_numpy(), B.pid.to_numpy()
    links = {int(pidA[i]): int(pidB[j]) for i, j in zip(r, c)}
    lost = lost_code(A, du, dv, mask)
    return links, {p: c for p, c in lost.items() if p not in links}


def link_nearest(A, B, img_a, img_b, field, mask, ctx):
    """Greedy nearest neighbour to the predicted position (Crocker & Grier 1996), within LINK_GATE."""
    du, dv = field(A.x.to_numpy(), A.y.to_numpy())
    links = {}
    if len(A) and len(B):
        pa = np.c_[A.x + du, A.y + dv]
        D = np.hypot(pa[:, None, 0] - B.x.to_numpy()[None], pa[:, None, 1] - B.y.to_numpy()[None])
        used_a, used_b = set(), set()
        for k in np.argsort(D, axis=None):
            i, j = np.unravel_index(k, D.shape)
            if D[i, j] > LINK_GATE:
                break
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            links[int(A.pid.iloc[i])] = int(B.pid.iloc[j])
    lost = lost_code(A, du, dv, mask)
    return links, {p: c for p, c in lost.items() if p not in links}


def link_lk(A, B, img_a, img_b, field, mask, ctx):
    """Pyramidal Lucas-Kanade (Bouguet 2001) on every particle centre, started from the predicted position."""
    du, dv = field(A.x.to_numpy(), A.y.to_numpy())
    if not len(A) or not len(B):
        return {}, lost_code(A, du, dv, mask)
    p0 = np.c_[A.x, A.y].astype(np.float32).reshape(-1, 1, 2)
    p1 = np.c_[A.x + du, A.y + dv].astype(np.float32).reshape(-1, 1, 2)
    p1, st, err = cv2.calcOpticalFlowPyrLK(u8(img_a), u8(img_b), p0, p1.copy(), winSize=(LK_WIN, LK_WIN),
                                           maxLevel=LK_LEVELS, flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    tree = cKDTree(B[['x', 'y']].to_numpy())
    best = {}
    for i, p in enumerate(A.pid):
        if not st[i, 0]:
            continue
        x, y = p1[i, 0]
        b = in_bbox(x, y, B)
        if b is None:
            d, j = tree.query([x, y])
            b = int(B.pid.iloc[j]) if d <= LK_SNAP else None
        if b is not None and (b not in best or err[i, 0] < best[b][1]):
            best[b] = (int(p), float(err[i, 0]))
    links = {p: b for b, (p, e) in best.items()}
    lost = lost_code(A, du, dv, mask)
    return links, {p: c for p, c in lost.items() if p not in links}


def appearance_corr(ta, tb):
    """Best normalised cross-correlation of cut-out ta anywhere over cut-out tb (padded to fit)."""
    if ta.std() < 1e-6 or tb.std() < 1e-6:
        return np.nan
    ph, pw = max(ta.shape[0] - tb.shape[0], 0) // 2 + APPEAR_PAD, max(ta.shape[1] - tb.shape[1], 0) // 2 + APPEAR_PAD
    tb = np.pad(tb, ((ph, ph), (pw, pw)))
    return float(cv2.matchTemplate(tb.astype(np.float32), ta.astype(np.float32), cv2.TM_CCOEFF_NORMED).max())


def link_reid(A, B, img_a, img_b, field, mask, ctx):
    """hungarian, then RE-IDENTIFICATION of what is left."""
    links, codes = link_hungarian(A, B, img_a, img_b, field, mask, ctx)
    la, lb = A[~A.pid.isin(list(links))], B[~B.pid.isin(list(links.values()))]
    if not len(la) or not len(lb):
        return links, codes
    du, dv = field(la.x.to_numpy(), la.y.to_numpy())
    D = np.hypot((la.x + du).to_numpy()[:, None] - lb.x.to_numpy()[None], (la.y + dv).to_numpy()[:, None] - lb.y.to_numpy()[None])
    ratio = np.maximum(la.area.to_numpy()[:, None], lb.area.to_numpy()[None]) / \
        np.minimum(la.area.to_numpy()[:, None], lb.area.to_numpy()[None])
    corr = np.zeros(D.shape)
    for i, j in zip(*np.nonzero((D <= REID_MAX) & (ratio <= REID_AREA))):
        r = appearance_corr(ctx['crops'][int(la.pid.iloc[i])], ctx['crops'][int(lb.pid.iloc[j])])
        corr[i, j] = r if np.isfinite(r) else 0
    # cost: unlike appearance first, then size change and distance as tie-breakers between look-alikes
    C = (1 - corr) + 0.3 * np.log(ratio) + 0.2 * D / REID_MAX
    for i, j in zip(*assign(C, corr >= REID_CORR, 1.0)):
        p = int(la.pid.iloc[i])
        links[p] = int(lb.pid.iloc[j])
        codes.pop(p, None)
    return links, codes


LINKERS = {'song': link_song, 'tracking': lambda *a: link_song(*a, wide=True),
           'hungarian': link_hungarian, 'reid': link_reid, 'nearest': link_nearest, 'lk': link_lk}


# stage 2: whole sequences
def join_chains(tracks, nxt):
    """nxt[i] = j: track j continues track i."""
    heads = [i for i in range(len(tracks)) if i not in set(nxt.values())]
    out = []
    for h in heads:
        t = dict(tracks[h], pids=list(tracks[h]['pids']))
        i, seen = h, {h}
        while i in nxt and nxt[i] not in seen:
            i = nxt[i]
            seen.add(i)
            t['pids'] += tracks[i]['pids']
            t['end'] = tracks[i]['end']
        out.append(t)
    return out


def gap_close(tracks, S, fields, song_rules):
    """Gap closing: join a track that ended to one that starts up to GAP_FRAMES later near its predicted position."""
    P, ctx, imgs = S['P'], S['ctx'], S['imgs']
    T = len(imgs)
    frame = P.frame.to_numpy()
    first = np.array([frame[t['pids'][0]] for t in tracks])
    last = np.array([frame[t['pids'][-1]] for t in tracks])
    ok_end = {-1, -2, -3, -4, -6} if song_rules else {-1, -2, -3, -4, -5, -6}
    starts = np.nonzero(first > 0)[0]
    choice = {}
    for j, t in enumerate(tracks):
        if t['end'] not in ok_end or last[j] >= T - 1:
            continue
        e = P.loc[t['pids'][-1]]
        cand = starts[(first[starts] > last[j]) & (first[starts] - last[j] <= GAP_FRAMES)]
        if not len(cand):
            continue
        pos = [np.array([e.x, e.y])]
        for k in range(int(first[cand].max() - last[j])):
            u, v = fields[int(last[j]) + k](*pos[-1])
            pos.append(pos[-1] + np.r_[u[0], v[0]])
        best = None
        for i in cand:
            s = P.loc[tracks[i]['pids'][0]]
            d = np.hypot(*(np.r_[s.x, s.y] - pos[int(first[i] - last[j])]))
            ratio = max(s.area, e.area) / max(min(s.area, e.area), 1)
            good = d <= GAP_NEAR and ratio < GAP_NEAR_AREA if song_rules else d <= GAP_FAR and ratio < GAP_FAR_AREA
            if song_rules and not good and d <= GAP_FAR and ratio < GAP_FAR_AREA:
                r = appearance_corr(ctx['crops'][int(e.pid)], ctx['crops'][int(s.pid)])   # MATLAB pads bottom/right only
                good = np.isfinite(r) and r >= GAP_CORR
            key = (first[i], d)
            if good and (best is None or key < best[0]):
                best = (key, i)
        if best is not None:
            choice[j] = best[1]
    owner = {}                      # a start claimed by several ends goes to the latest end (MATLAB)
    for j, i in choice.items():
        if i not in owner or last[j] > last[owner[i]]:
            owner[i] = j
    S['ctx']['n_gap'] = S['ctx'].get('n_gap', 0) + len(owner)
    return join_chains(tracks, {j: i for i, j in owner.items()})


def song_backward(tracks, S, fields):
    """Backward pass: rejoin tracks broken by a missed link (main_tracking.m)."""
    P, ctx, imgs = S['P'], S['ctx'], S['imgs']
    frame = P.frame.to_numpy()
    end_of = {t['pids'][-1]: k for k, t in enumerate(tracks) if t['end'] < 0}
    where = {p: k for k, t in enumerate(tracks) for p in t['pids']}
    nxt = {}
    for i in range(len(tracks) - 1, -1, -1):
        b = P.loc[tracks[i]['pids'][0]]
        f = int(frame[int(b.pid)])
        if f == 0:
            continue
        u, v = fields[f - 1](b.x, b.y)
        t = ctx['crops'][int(b.pid)]
        m = SONG_MARGIN[0] if max(t.shape) <= SONG_SIZE_SWITCH else SONG_MARGIN[1]
        corr, xp, yp = xcorr(t, imgs[f - 1], (b.x0 + b.x1 - 1) / 2 - u[0], (b.y0 + b.y1 - 1) / 2 - v[0], m)
        if not np.isfinite(corr) or corr <= SONG_CORR_STRONG:
            continue
        a = in_bbox(xp, yp, ctx['dets'][f - 1])
        if a is None:
            continue
        j = end_of.get(a)
        if j is not None and j != i and j not in nxt and i not in nxt.values():
            nxt[j] = i
        elif a in where and where[a] != i:
            ctx['n_frag'] = ctx.get('n_frag', 0) + 1
    ctx['n_back'] = ctx.get('n_back', 0) + len(nxt)
    return join_chains(tracks, nxt)


def entry_flag(tracks, S, fields):
    """Flag -7 for tracks that may have come in from the masked area (main_tracking.m)."""
    P, mask = S['P'], S['mask']
    for t in tracks:
        b = P.loc[t['pids'][0]]
        f = int(b.frame)
        t['flag'] = 0
        if f == 0:
            continue
        u, v = fields[f - 1](b.x, b.y)
        dx, dy = -u[0], -v[0]
        if not all(in_mask(mask, cx + dx, cy + dy) for cx in (b.x0, b.x1 - 1) for cy in (b.y0, b.y1 - 1)):
            t['flag'] = -7
    return tracks


def track_sequence(tracker, S, fields):
    """Link a whole sequence."""
    ctx, imgs, mask = S['ctx'], S['imgs'], S['mask']
    for k in ('n_agg', 'n_frag', 'n_back', 'n_gap'):
        ctx[k] = 0
    tracks = [dict(pids=[int(p)], end=0, flag=0) for p in ctx['dets'][0].pid]
    alive = {t['pids'][0]: [k] for k, t in enumerate(tracks)}
    for f in range(1, len(imgs)):
        A, B = ctx['dets'][f - 1], ctx['dets'][f]
        links, codes = LINKERS[tracker](A, B, imgs[f - 1], imgs[f], fields[f - 1], mask, ctx)
        new = {}
        for a, ks in alive.items():
            b = links.get(a)
            for k in ks:
                if b is None:
                    tracks[k]['end'] = codes.get(a, -6)
                else:
                    tracks[k]['pids'].append(b)
                    new.setdefault(b, []).append(k)
        for b in B.pid:
            if int(b) not in new:
                tracks.append(dict(pids=[int(b)], end=0, flag=0))
                new[int(b)] = [len(tracks) - 1]
        alive = new
    if tracker.startswith('song'):
        tracks = song_backward(tracks, S, fields)
        tracks = gap_close(tracks, S, fields, song_rules=True)
        tracks = entry_flag(tracks, S, fields)
    elif tracker in ('hungarian', 'reid'):
        tracks = gap_close(tracks, S, fields, song_rules=False)
    return tracks


def tracks_table(tracks, P):
    rows = [dict(track_id=k, pid=p, end_code=t['end'], flag=t.get('flag', 0)) for k, t in enumerate(tracks)
            for p in t['pids']]
    return pd.DataFrame(rows).merge(P, on='pid')


# scoring
def score_motion(S, fields, pairs):
    """Error of each method's predicted displacement at every truth particle (usable pairs)."""
    d = pairs[pairs.usable].copy()
    out = []
    for f, g in d.groupby('frame'):
        u, v = fields[int(f)](g.xa.to_numpy(), g.ya.to_numpy())
        out.append(g.assign(pu=u, pv=v, err=np.hypot(u - g.dx, v - g.dy)))
    return pd.concat(out) if out else d.assign(pu=[], pv=[], err=[])


def score_tracks(tracks, S, pairs):
    """Counts for one tracker on one sequence (turned into percentages by summarise_tracking)."""
    P = S['P']
    uid, frame = P.uid.to_numpy(), P.frame.to_numpy()
    pred = set()
    for t in tracks:
        pred.update(zip(t['pids'][:-1], t['pids'][1:]))
    c = dict(n_links=len(pred), n_links_correct=sum(uid[a] > 0 and uid[a] == uid[b] for a, b in pred))
    # true links: same truth particle detected in consecutive frames (largest detection per frame)
    D = P[P.uid > 0].sort_values('area').drop_duplicates(['uid', 'frame'], keep='last')
    nxt = D.assign(frame=D.frame - 1)[['uid', 'frame', 'pid']].rename(columns=dict(pid='pid_b'))
    TL = D[['uid', 'frame', 'pid']].merge(nxt, on=['uid', 'frame'])
    own = pairs.set_index(['uid', 'frame']).own
    TL['bin'] = own_bin(own.reindex(pd.MultiIndex.from_frame(TL[['uid', 'frame']])).to_numpy())  # NaN -> unclear
    TL['found'] = [(a, b) in pred for a, b in zip(TL.pid, TL.pid_b)]
    c['n_true_links'], c['n_true_links_found'] = len(TL), int(TL.found.sum())
    for k in own_bin(np.array([0.0, float(OWN_BINS[0]), float(OWN_BINS[1])])):
        g = TL[TL.bin == k]
        c[f'true links {k}'], c[f'found {k}'] = len(g), int(g.found.sum())
    # arrivals: a track starting after frame 0 is counted as a new particle (Song's flux count)
    first_det = P[P.uid > 0].groupby('uid').frame.min()
    if S.get('arrival') is not None:
        new = S['arrival'].reindex(first_det.index).fillna(False).astype(bool).to_numpy()
    else:
        new = (S['ttab'].groupby('uid').frame.min().reindex(first_det.index) > 0).to_numpy()
    arrivals = set(first_det.index[new])
    starts = [(t['pids'][0], t.get('flag', 0)) for t in tracks if frame[t['pids'][0]] > 0]
    good, seen = 0, set()
    for p, fl in starts:
        u = uid[p]
        if u in arrivals and u not in seen and frame[p] == first_det[u]:
            good += 1
            seen.add(u)
    c.update(n_arrivals=len(arrivals), n_arrivals_found=len(seen), n_starts=len(starts), n_starts_correct=good,
             n_starts_recount=sum(uid[p] > 0 and frame[p] > first_det.get(uid[p], 1e9) for p, _ in starts),
             n_starts_flag7=sum(fl == -7 for _, fl in starts))
    # unbroken: every detection of a particle sits in one track
    where = {}
    for k, t in enumerate(tracks):
        for p in t['pids']:
            where.setdefault(p, set()).add(k)
    n2 = unb = 0
    for u, g in D.groupby('uid'):
        if len(g) < 2:
            continue
        n2 += 1
        common = set.intersection(*[where.get(p, set()) for p in g.pid])
        unb += bool(common)
    tt = S['ttab']
    seen_t = set(zip(P.uid[P.uid > 0], P.frame[P.uid > 0]))
    c.update(n_truth_obj=len(tt), n_truth_det=sum((u, f) in seen_t for u, f in zip(tt.uid, tt.frame)),
             n_det=len(P), n_det_matched=int((P.uid > 0).sum()))
    c.update(n_multi=n2, n_unbroken=unb, n_agg=S['ctx'].get('n_agg', 0), n_frag=S['ctx'].get('n_frag', 0),
             n_back=S['ctx'].get('n_back', 0), n_gap=S['ctx'].get('n_gap', 0))
    return c


def pct(a, b):
    b = np.asarray(b, float)
    return np.round(100 * np.asarray(a, float) / np.where(b == 0, np.nan, b), 1)


def summarise_motion(E):
    """Percentage table for the motion methods from the per-particle errors."""
    rows = []
    for m, g in E.groupby('motion', sort=False):
        ok = g.err <= SEARCH_PX
        r = {'Median error px': round(float(g.err.median()), 2),
             'Median error um': round(float(g.err.median()) * SC.UM_PER_PX, 1),
             f'Error <={NEAR_PX} px %': pct((g.err <= NEAR_PX).sum(), len(g)),
             f'Error <={SEARCH_PX} px (inside Song search) %': pct(ok.sum(), len(g))}
        db = disp_bin(g.disp.to_numpy())
        for k in disp_bin(np.array([0.0] + [float(v) for v in DISP_BINS])):
            r[f'Error <={SEARCH_PX} px, moved {k} %'] = pct(ok[db == k].sum(), (db == k).sum())
        ob = own_bin(g.own.to_numpy())
        for k in own_bin(np.array([0.0] + [float(v) for v in OWN_BINS])):
            r[f'Error <={SEARCH_PX} px, {k} %'] = pct(ok[ob == k].sum(), (ob == k).sum())
        per = g.groupby('seq_id').agg(s=('seconds', 'first'), n=('frame', 'nunique'))
        r['ms per frame pair'] = round(1000 * per.s.sum() / max(per.n.sum(), 1))
        r['particle moves scored (n)'] = len(g)
        rows.append(pd.Series(r, name=m))
    return pd.DataFrame(rows)


def summarise_tracking(C, by=('tracker', 'motion')):
    g = C.groupby(list(by), sort=False).sum(numeric_only=True)
    r = pd.DataFrame(index=g.index)
    r['Particles detected %'] = pct(g.n_truth_det, g.n_truth_obj)              # segmentation recall
    r['Detections on a particle %'] = pct(g.n_det_matched, g.n_det)          # segmentation precision
    r['Links correct (precision) %'] = pct(g.n_links_correct, g.n_links)
    r['True links found (recall) %'] = pct(g.n_true_links_found, g.n_true_links)
    for k in own_bin(np.array([0.0, float(OWN_BINS[0]), float(OWN_BINS[1])])):
        r[f'Links found, {k} %'] = pct(g[f'found {k}'], g[f'true links {k}'])
    r['New arrivals found %'] = pct(g.n_arrivals_found, g.n_arrivals)
    r['Counted arrivals correct %'] = pct(g.n_starts_correct, g.n_starts)
    r['Counted arrivals, % of true'] = pct(g.n_starts, g.n_arrivals)
    r['Old particles re-counted, % of counted'] = pct(g.n_starts_recount, g.n_starts)
    r['Particles never broken %'] = pct(g.n_unbroken, g.n_multi)
    r['Starts flagged -7 %'] = pct(g.n_starts_flag7, g.n_starts)
    for k, n in (('aggregations', 'n_agg'), ('fragments', 'n_frag'), ('backward joins', 'n_back'),
                 ('gap joins', 'n_gap')):
        r[f'{k} (n)'] = g[n]
    r['s per sequence'] = (g.seconds / g.n_seq).round(1)
    return r


# running on the synthetic sequences
def run_sequence(seq_dir, out_dir, motions, trackers, detections='song', max_frames=None):
    S = load_seq(seq_dir, detections, max_frames)
    if S['truth'] is None:
        raise FileNotFoundError(f'{seq_dir}: truth/labels missing')
    S['ttab'] = truth_table(S['truth'])
    pairs = truth_pairs(S['ttab'])
    sid, T = S['name'], len(S['imgs'])
    tdir = os.path.join(out_dir, 'tracks', sid)
    os.makedirs(tdir, exist_ok=True)
    S['P'].to_csv(os.path.join(tdir, 'detections.csv'), index=False)
    E, C = [], []
    for m in motions:
        t0 = time.time()
        fields = [estimate(m, S['imgs'][f], S['imgs'][f + 1], S['mask'], S['ctx']['dets'][f], S['ctx']['dets'][f + 1])
                  for f in range(T - 1)]
        sec = time.time() - t0
        E.append(score_motion(S, fields, pairs).assign(motion=m, seq_id=sid, seconds=sec))
        t1 = time.time()
        for tr in trackers:
            t0 = time.time()
            tracks = track_sequence(tr, S, fields)
            sec_t = time.time() - t0
            tracks_table(tracks, S['P']).to_csv(os.path.join(tdir, f'{m}__{tr}.csv'), index=False)
            C.append(dict(seq_id=sid, motion=m, tracker=tr, seconds=sec_t, n_seq=1, **score_tracks(tracks, S, pairs)))
        print(f'  {sid} {m}: motion {sec:.0f} s, all trackers {time.time() - t1:.0f} s', flush=True)
    keep = ['seq_id', 'motion', 'frame', 'uid', 'xa', 'ya', 'dx', 'dy', 'disp', 'own', 'pu', 'pv', 'err', 'seconds']
    return pd.concat(E)[keep], pd.DataFrame(C)


def run(data, out_dir, splits=('val', 'test'), motions=None, trackers=None, detections='song', max_frames=None):
    """Every motion method x every tracker on every sequence of the chosen splits."""
    motions, trackers = motions or default_motions(), trackers or list(LINKERS)
    os.makedirs(os.path.join(out_dir, 'per_seq'), exist_ok=True)
    for d in sorted(glob.glob(os.path.join(data, 'seq*'))):
        sid = os.path.basename(d)
        f_e = os.path.join(out_dir, 'per_seq', f'{sid}_motion.csv')
        if sid.split('_')[-1] not in splits or os.path.exists(f_e):
            continue
        t0 = time.time()
        E, C = run_sequence(d, out_dir, motions, trackers, detections, max_frames)
        C.to_csv(os.path.join(out_dir, 'per_seq', f'{sid}_tracking.csv'), index=False)
        E.to_csv(f_e, index=False)
        print(f'{sid}: done in {time.time() - t0:.0f} s', flush=True)
    return results(out_dir)


def results(out_dir):
    E = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(out_dir, 'per_seq', '*_motion.csv')))])
    C = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(out_dir, 'per_seq', '*_tracking.csv')))])
    tm, tt = summarise_motion(E), summarise_tracking(C)
    tm.to_csv(os.path.join(out_dir, 'motion_accuracy.csv'))
    tt.to_csv(os.path.join(out_dir, 'tracking.csv'))
    with pd.ExcelWriter(os.path.join(out_dir, 'piv_comparison.xlsx')) as xw:
        tm.to_excel(xw, sheet_name='motion accuracy')
        tt.to_excel(xw, sheet_name='tracking')
        summarise_tracking(C, by=('seq_id', 'tracker', 'motion')).to_excel(xw, sheet_name='tracking per sequence')
        C.to_excel(xw, sheet_name='raw tracking counts', index=False)
    return tm, tt


# large-motion stress test
def objects_from_labels(lab, img):
    rows, crops, bmasks = [], [], []
    for r in regionprops(big_only(lab)):
        y0, x0, y1, x1 = r.bbox
        rows.append(dict(pid=len(rows), frame=0, x=r.centroid[1], y=r.centroid[0], area=int(r.area),
                         x0=x0, y0=y0, x1=x1, y1=y1, uid=r.label))
        crops.append(img[y0:y1, x0:x1].copy())
        bmasks.append(r.image.copy())
    return pd.DataFrame(rows, columns=['pid', 'frame', 'x', 'y', 'area', 'x0', 'y0', 'x1', 'y1', 'uid']), crops, bmasks


def make_stress_pair(img, lab, mask, bulk_px, disp_px, n_move, seed):
    """Frame b = frame a moved by a known bulk motion, with a few particles moved further on their own."""
    rng = np.random.default_rng(seed)
    H, W = img.shape
    yy, xx = np.nonzero(mask)
    cx, cy = xx.mean(), yy.mean()
    R = np.sqrt(mask.sum() / np.pi)
    ang = rng.uniform(0, 2 * np.pi)
    rot = np.degrees(0.5 * bulk_px / R) * rng.choice([-1, 1])
    M = cv2.getRotationMatrix2D((cx, cy), rot, 1.0)
    M[:, 2] += 0.5 * bulk_px * np.array([np.cos(ang), np.sin(ang)])
    b = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0) * mask
    lb = cv2.warpAffine(lab.astype(np.float32), M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
    lb = np.where(mask, lb, 0).astype(np.int64)
    inner = cv2.erode(mask.astype(np.uint8), disk(STRESS_EDGE)) > 0
    area_a = np.bincount(lab.ravel())
    area_b = np.bincount(lb.ravel(), minlength=len(area_a))
    cand = [u for u in np.nonzero((area_a >= STRESS_AREA[0]) & (area_a <= STRESS_AREA[1]))[0]
            if u > 0 and area_b[u] >= 0.9 * area_a[u]]
    rng.shuffle(cand)
    moved, busy = [], cv2.dilate((lb > 0).astype(np.uint8), disk(STRESS_GAP)) > 0
    for u in cand:
        if len(moved) >= n_move:
            break
        m = lb == u
        md = cv2.dilate(m.astype(np.uint8), disk(STRESS_NEAR)) > 0
        others = busy & ~(cv2.dilate(m.astype(np.uint8), disk(STRESS_EDGE)) > 0)
        if (md & others).any() or not inner[m].all():
            continue                                       # not isolated or too close to the edge
        ys, xs = np.nonzero(md)
        own = cv2.dilate(m.astype(np.uint8), disk(STRESS_GAP)) > 0   # the part of `busy` that is this particle
        if disp_px == 0:
            moved.append(int(u))
            continue
        for _ in range(40):
            t = rng.uniform(0, 2 * np.pi)
            dx, dy = int(round(disp_px * np.cos(t))), int(round(disp_px * np.sin(t)))
            ny, nx = ys + dy, xs + dx
            if ny.min() < 0 or nx.min() < 0 or ny.max() >= H or nx.max() >= W:
                continue
            if not inner[ny, nx].all() or (busy[ny, nx] & ~own[ny, nx]).any():
                continue
            patch = b[ys, xs].copy()
            ring = (cv2.dilate(m.astype(np.uint8), disk(STRESS_RING)) > 0) & ~md & (lb == 0) & mask
            b[ys, xs] = np.median(b[ring]) if ring.any() else 0
            lb[m] = 0
            b[ny, nx] = np.maximum(b[ny, nx], patch)
            lb[ny[m[ys, xs]], nx[m[ys, xs]]] = u
            busy[ny, nx] = True
            moved.append(int(u))
            break
    return b.astype(np.float32), lb, moved


def stress_case(img, lab, mask, bulk, disp, n_move, seed, motions, trackers):
    """Run every motion x tracker on one synthetic pair."""
    b, lb, moved = make_stress_pair(img, lab, mask, bulk, disp, n_move, seed)
    A, ca, ma = objects_from_labels(lab, img)
    B, cb, mb = objects_from_labels(lb, b)
    B = B.assign(pid=B.pid + len(A), frame=1)
    ctx = dict(crops=ca + cb, bmasks=ma + mb, high=float(threshold_otsu(img[mask])))
    ta = A.set_index('uid')
    tb = B.set_index('uid')
    common = [u for u in ta.index if u in tb.index and 0.7 < tb.area[u] / ta.area[u] < 1 / 0.7]
    rows = []
    for m in motions:
        field = estimate(m, img, b, mask, A, B)
        for tr in trackers:
            links, _ = LINKERS[tr](A, B, img, b, field, mask, ctx)
            for u in common:
                pa, pb = int(ta.pid[u]), int(tb.pid[u])
                du, dv = field(ta.x[u], ta.y[u])
                rows.append(dict(motion=m, tracker=tr, uid=u, moved=u in moved, linked=links.get(pa) == pb,
                                 err=float(np.hypot(du[0] - (tb.x[u] - ta.x[u]), dv[0] - (tb.y[u] - ta.y[u])))))
    return pd.DataFrame(rows), len(moved)


def stress_frame(seq_dir, fn):
    """Background-removed frame, truth labels and mask of frame number fn."""
    raw = cv2.imread(os.path.join(seq_dir, 'bgrem', f'bgrem_{fn:04d}.png'), 0)
    mask = load_mask(os.path.join(seq_dir, 'truth', 'static_mask.png'), raw.shape)
    lab = cv2.imread(os.path.join(seq_dir, 'truth', 'labels', f'frame_{fn:04d}.png'), cv2.IMREAD_UNCHANGED)
    return np.where(mask, raw, 0).astype(np.float32), np.where(mask, lab, 0).astype(np.int64), mask


def stress_frames(data, splits, n_pairs, seed=0):
    """(sequence dir, frame number) pairs used by the stress test."""
    seqs = [d for d in sorted(glob.glob(os.path.join(data, 'seq*'))) if d.split('_')[-1] in splits]
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_pairs):
        d = seqs[k % len(seqs)]
        _, nums = frame_files(d)
        out.append((d, int(rng.choice(nums[len(nums) // 2:]))))
    return out


def run_stress(data, out_dir, splits=('val', 'test'), motions=None, trackers=None, n_pairs=4,
               bulks=None, disps=None, n_move=6, seed=0):
    """Large-motion stress test (resumable, one CSV per frame in out_dir/stress)."""
    motions, trackers = motions or default_motions(), trackers or list(LINKERS)
    bulks = STRESS_BULK if bulks is None else bulks
    disps = STRESS_DISP if disps is None else disps
    sd = os.path.join(out_dir, 'stress')
    os.makedirs(sd, exist_ok=True)
    for k, (d, fn) in enumerate(stress_frames(data, splits, n_pairs, seed)):
        f_out = os.path.join(sd, f'{k:02d}_{os.path.basename(d)}_{fn:04d}.csv')
        if os.path.exists(f_out):
            continue
        t0 = time.time()
        img, lab, mask = stress_frame(d, fn)
        out = []
        for bulk in bulks:
            for disp in disps:
                r, n = stress_case(img, lab.astype(np.int64), mask, bulk, disp, n_move, seed + 1000 * k, motions, trackers)
                out.append(r.assign(seq_id=os.path.basename(d), frame=fn, bulk_px=bulk, disp_px=disp, n_moved=n))
        pd.concat(out).to_csv(f_out, index=False)
        print(f'stress {k + 1}/{n_pairs} ({os.path.basename(d)} frame {fn}): {time.time() - t0:.0f} s', flush=True)
    return stress_results(out_dir)


def stress_results(out_dir):
    R = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(out_dir, 'stress', '*.csv')))])
    mv = R[R.moved].groupby(['bulk_px', 'tracker', 'motion', 'disp_px']).linked.mean().mul(100).round(1)
    tab = mv.unstack('disp_px')
    tab.columns = [f'moved {c} px ({c * SC.UM_PER_PX / 1000:.2f} mm): linked %' for c in tab.columns]
    oth = R[~R.moved].groupby(['bulk_px', 'tracker', 'motion']).linked.mean().mul(100).round(1)
    tab['other particles: linked %'] = oth
    tab['moved particles per distance (n)'] = (R[R.moved].groupby(['bulk_px', 'tracker', 'motion']).size()
                                               / R.disp_px.nunique()).round(0).astype(int)
    tab.to_csv(os.path.join(out_dir, 'stress_test.csv'))
    with pd.ExcelWriter(os.path.join(out_dir, 'stress_test.xlsx')) as xw:
        tab.to_excel(xw, sheet_name='linked %')
        R[R.moved].groupby(['bulk_px', 'motion', 'disp_px']).err.median().round(1).unstack('disp_px') \
            .to_excel(xw, sheet_name='field error at moved px')
    return tab, R


# figures
def _bright(im, gain=2.0):
    return np.clip(im * gain, 0, 255).astype(np.uint8)


def trajectory_figure(imgs, T, out_png, n=4, track_ids=None, truth_tab=None, bmasks=None, title=''):
    """Song-style view of single particles."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from skimage.segmentation import find_boundaries
    T = T.sort_values(['track_id', 'frame'])
    if track_ids is None:
        step = {k: np.hypot(np.diff(g.x), np.diff(g.y)).max() for k, g in T.groupby('track_id') if len(g) >= 3}
        track_ids = sorted(step, key=step.get, reverse=True)[:n]
    if not len(track_ids):
        print('no tracks of 3 or more frames to draw')
        return None
    n, ntile, (H, W) = len(track_ids), 6, imgs[0].shape
    fig = plt.figure(figsize=(17, 3.5 * n))
    gs = fig.add_gridspec(n, 3 + ntile, width_ratios=[1, 1, 1] + [0.7] * ntile)
    for r, tid in enumerate(track_ids):
        g = T[T.track_id == tid]
        x, y, fr = g.x.to_numpy(), g.y.to_numpy(), g.frame.to_numpy().astype(int)
        pad = FIG_PAD
        tt = None
        if truth_tab is not None and int(g.uid.iloc[0]) > 0:        # whole true path of the first particle
            tt = truth_tab[truth_tab.uid == g.uid.iloc[0]].sort_values('frame')
        px = np.r_[g.x0.min(), g.x1.max()] if tt is None else np.r_[g.x0.min(), g.x1.max(), tt.x.min(), tt.x.max()]
        py = np.r_[g.y0.min(), g.y1.max()] if tt is None else np.r_[g.y0.min(), g.y1.max(), tt.y.min(), tt.y.max()]
        x0, x1 = int(max(px.min() - pad, 0)), int(min(px.max() + pad, W))
        y0, y1 = int(max(py.min() - pad, 0)), int(min(py.max() + pad, H))
        ext = (x0 - .5, x1 - .5, y1 - .5, y0 - .5)
        ax = fig.add_subplot(gs[r, :3])
        ax.imshow(_bright(imgs[fr[-1]][y0:y1, x0:x1]), cmap='gray', extent=ext)
        if tt is not None:
            ax.plot(tt.x, tt.y, '--', color='lime', lw=1.3, zorder=5, label=f'true path (frames {tt.frame.min()}-{tt.frame.max()})')
            ax.legend(loc='upper right', fontsize=7)
        if len(g) > 1:
            lc = LineCollection(np.stack([np.c_[x[:-1], y[:-1]], np.c_[x[1:], y[1:]]], 1), cmap='plasma', lw=2.2)
            lc.set_array(fr[1:].astype(float))
            lc.set_clim(fr.min(), fr.max() + 1e-9)
            ax.add_collection(lc)
        sc = ax.scatter(x, y, c=fr, cmap='plasma', s=18, zorder=3, vmin=fr.min(), vmax=max(fr.max(), fr.min() + 1))
        ax.scatter(x[:1], y[:1], s=90, facecolors='none', edgecolors='white', lw=1.5, zorder=4)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        nu = g.uid[g.uid > 0].nunique()
        who = '' if not g.uid.any() else (', one true particle' if nu == 1 else f', jumps across {nu} true particles')
        if tt is not None and nu == 1 and (tt.frame.max() > fr.max() or tt.frame.min() < fr.min()):
            who += ' (track broken: the particle is seen longer)'
        ax.set_title(f'track {tid}: frames {fr.min()}-{fr.max()}, largest step '
                     f'{np.hypot(np.diff(x), np.diff(y)).max(initial=0):.0f} px{who}', fontsize=9,
                     color='red' if nu > 1 or 'broken' in who else 'black')
        ax.axis('off')
        plt.colorbar(sc, ax=ax, fraction=0.03, label='frame')
        side = int(max(FIG_TILE, 2.2 * max((g.x1 - g.x0).max(), (g.y1 - g.y0).max())))
        for c, k in enumerate(np.unique(np.linspace(0, len(g) - 1, min(ntile, len(g))).round().astype(int))):
            row = g.iloc[k]
            ox, oy = int(round(row.x)) - side // 2, int(round(row.y)) - side // 2
            rgb = np.dstack([_bright(crop_padded(imgs[int(row.frame)], ox, oy, side, side))] * 3)
            sel = np.zeros((side, side), bool)
            bm = bmasks[int(row.pid)] if bmasks is not None else np.ones((int(row.y1 - row.y0), int(row.x1 - row.x0)), bool)
            ys, xs = np.nonzero(bm)
            ys, xs = ys + int(row.y0) - oy, xs + int(row.x0) - ox
            ok = (ys >= 0) & (ys < side) & (xs >= 0) & (xs < side)
            sel[ys[ok], xs[ok]] = True
            rgb[find_boundaries(sel, mode='inner')] = (255, 220, 0)
            ax = fig.add_subplot(gs[r, 3 + c])
            ax.imshow(rgb)
            ax.set_title(f'frame {int(row.frame)}', fontsize=8)
            ax.axis('off')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    return fig


def seq_figures(seq_dir, out_dir, motion, tracker, n=4, track_ids=None, detections='song', max_frames=None):
    """Trajectory and overview figures for one synthetic sequence, from the tracks saved by run()."""
    S = load_seq(seq_dir, detections, max_frames)
    T = pd.read_csv(os.path.join(out_dir, 'tracks', S['name'], f'{motion}__{tracker}.csv'))
    tt = truth_table(S['truth']) if S['truth'] is not None else None
    if track_ids is None and tt is not None:
        # the particles that truly moved most on their own (slides, swimmers), and the track holding each
        pr = truth_pairs(tt)
        pr = pr[pr.usable].sort_values('own', ascending=False)
        D = T.sort_values('area').drop_duplicates(['uid', 'frame'], keep='last').set_index(['uid', 'frame'])
        track_ids = []
        for u, f in zip(pr.uid, pr.frame):
            if (u, f) in D.index and D.track_id[(u, f)] not in track_ids:
                track_ids.append(int(D.track_id[(u, f)]))
            if len(track_ids) == n:
                break
    stem = os.path.join(out_dir, f'{S["name"]}_{motion}__{tracker}')
    title = f'{S["name"]}: {motion} + {tracker}'
    f1 = trajectory_figure(S['imgs'], T, stem + '_trajectories.png', n, track_ids, tt, S['ctx']['bmasks'], title)
    f2 = overview_figure(S['imgs'], T, stem + '_overview.png', title=title + ' (tracks of 3+ frames, red = new arrivals)')
    return f1, f2


def overview_figure(imgs, T, out_png, frame=None, min_len=3, title=''):
    """Every track of at least min_len frames drawn over one frame (default: the last)."""
    import matplotlib.pyplot as plt
    frame = len(imgs) - 1 if frame is None else frame
    fig, ax = plt.subplots(figsize=(11, 11 * imgs[0].shape[0] / imgs[0].shape[1]))
    ax.imshow(_bright(imgs[frame]), cmap='gray')
    rng = np.random.default_rng(1)
    for tid, g in T.sort_values('frame').groupby('track_id'):
        if len(g) < min_len:
            continue
        ax.plot(g.x, g.y, '-', lw=1, color=rng.uniform(0.3, 1, 3))
        ax.plot(g.x.iloc[-1], g.y.iloc[-1], '.', ms=3, color='w')
    starts = T.sort_values('frame').groupby('track_id').head(1)
    starts = starts[starts.frame > 0]
    ax.scatter(starts.x, starts.y, s=14, facecolors='none', edgecolors='red', lw=0.8, label='new arrivals')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    return fig


def field_figure(seq_dir, frame_index, motions, out_png, detections='truth'):
    """Predicted displacement (red) against the true displacement (green) of every particle for one frame pair."""
    import matplotlib.pyplot as plt
    S = load_seq(seq_dir, detections, frame_index + 2)
    S['ttab'] = truth_table(S['truth'])
    pr = truth_pairs(S['ttab'])
    pr = pr[(pr.frame == frame_index) & pr.usable]
    a, b = S['imgs'][frame_index], S['imgs'][frame_index + 1]
    cols = 4
    nr = -(-len(motions) // cols)
    fig, axs = plt.subplots(nr, cols, figsize=(4.2 * cols, 4.2 * nr * a.shape[0] / a.shape[1] + 0.4))
    axs = np.atleast_1d(axs).ravel()
    for ax in axs:
        ax.axis('off')
    for ax, m in zip(axs, motions):
        t0 = time.time()
        fld = estimate(m, a, b, S['mask'], S['ctx']['dets'][frame_index], S['ctx']['dets'][frame_index + 1])
        sec = time.time() - t0
        u, v = fld(pr.xa.to_numpy(), pr.ya.to_numpy())
        err = np.hypot(u - pr.dx, v - pr.dy)
        ax.imshow(_bright(a, 1.5), cmap='gray')
        kw = dict(angles='xy', scale_units='xy', scale=1, width=0.004)
        ax.quiver(pr.xa, pr.ya, pr.dx, pr.dy, color='lime', **kw)
        ax.quiver(pr.xa, pr.ya, u, v, color='red', **kw)
        ax.set_title(f'{m}: median error {np.median(err):.1f} px ({sec:.1f} s)', fontsize=9)
    fig.suptitle(f'{os.path.basename(seq_dir)}, frames {frame_index}->{frame_index + 1}: true green, method red', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    return fig


def stress_curves(R, out_png):
    """% of the moved particles linked correctly against how far they moved on their own."""
    import matplotlib.pyplot as plt
    mv = R[R.moved].groupby(['bulk_px', 'tracker', 'motion', 'disp_px']).linked.mean().mul(100).reset_index()
    bulks = sorted(mv.bulk_px.unique())
    trackers = [t for t in LINKERS if t in set(mv.tracker)]
    motions = list(dict.fromkeys(mv.motion))
    cmap = plt.get_cmap('tab20' if len(motions) > 10 else 'tab10')
    fig, axs = plt.subplots(len(bulks), len(trackers), figsize=(3.6 * len(trackers), 3.2 * len(bulks)),
                            squeeze=False, sharex=True, sharey=True)
    for i, bk in enumerate(bulks):
        for j, tr in enumerate(trackers):
            ax = axs[i, j]
            for k, m in enumerate(motions):
                g = mv[(mv.bulk_px == bk) & (mv.tracker == tr) & (mv.motion == m)]
                ax.plot(g.disp_px, g.linked, '-o', ms=3, color=cmap(k), label=m)
            ax.set_title(f'{tr}, bulk {bk} px ({bk * SC.UM_PER_PX / 1000:.2f} mm)', fontsize=9)
            ax.set_ylim(-3, 103)
            ax.grid(alpha=0.3)
            if i == len(bulks) - 1:
                ax.set_xlabel(f'extra motion of the particle (px at {SC.UM_PER_PX:g} um/px)')
            if j == 0:
                ax.set_ylabel('moved particles linked %')
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc='center right', fontsize=8, title='motion method')
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(out_png, dpi=110)
    return fig


def stress_example(seq_dir, frame, bulk, disp, combos, out_png, n_move=6, seed=0, which=0):
    """One particle moved by `disp` px on its own."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    img, lab, mask = stress_frame(seq_dir, frame)
    b, lb, moved = make_stress_pair(img, lab, mask, bulk, disp, n_move, seed)
    if not moved:
        print('no particle could be moved this far here; try a smaller disp or another frame')
        return None
    u = moved[which % len(moved)]
    A, ca, ma = objects_from_labels(lab, img)
    B, cb, mb = objects_from_labels(lb, b)
    B = B.assign(pid=B.pid + len(A), frame=1)
    ctx = dict(crops=ca + cb, bmasks=ma + mb, high=float(threshold_otsu(img[mask])))
    ra, rb = A[A.uid == u].iloc[0], B[B.uid == u].iloc[0]
    pad = EXAMPLE_PAD
    x0 = int(max(min(ra.x0, rb.x0) - pad, 0))
    x1 = int(min(max(ra.x1, rb.x1) + pad, img.shape[1]))
    y0 = int(max(min(ra.y0, rb.y0) - pad, 0))
    y1 = int(min(max(ra.y1, rb.y1) + pad, img.shape[0]))
    ext = (x0 - .5, x1 - .5, y1 - .5, y0 - .5)
    cols = min(4, len(combos) + 1)
    nr = -(-(len(combos) + 1) // cols)
    fig, axs = plt.subplots(nr, cols, figsize=(3.8 * cols, 3.8 * nr), squeeze=False)
    for ax in axs.ravel():
        ax.axis('off')
    box = lambda ax, r, c, ls='-': ax.add_patch(Rectangle((r.x0 - .5, r.y0 - .5), r.x1 - r.x0, r.y1 - r.y0,
                                                          fill=False, ec=c, lw=1.6, ls=ls))
    ax = axs[0, 0]
    ax.imshow(_bright(img[y0:y1, x0:x1]), cmap='gray', extent=ext)
    box(ax, ra, 'lime')
    ax.set_title('frame a (particle in green)', fontsize=9)
    fields = {}
    for k, (m, tr) in enumerate(combos, start=1):
        ax = axs.ravel()[k]
        if m not in fields:
            fields[m] = estimate(m, img, b, mask, A, B)
        links, _ = LINKERS[tr](A, B, img, b, fields[m], mask, ctx)
        du, dv = fields[m](ra.x, ra.y)
        ax.imshow(_bright(b[y0:y1, x0:x1]), cmap='gray', extent=ext)
        box(ax, ra, 'lime', ':')
        box(ax, rb, 'lime')
        ax.annotate('', xy=(ra.x + du[0], ra.y + dv[0]), xytext=(ra.x, ra.y),
                    arrowprops=dict(arrowstyle='->', color='cyan', lw=1.5))
        got = links.get(int(ra.pid))
        if got is not None:
            box(ax, B[B.pid == got].iloc[0], 'red')
        ok = got == int(rb.pid)
        ax.set_title(f'{m} + {tr}: ' + ('correct' if ok else 'not linked' if got is None else 'WRONG particle'),
                     fontsize=9, color='green' if ok else 'red')
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
    fig.suptitle(f'{os.path.basename(seq_dir)} frame {frame}: bulk {bulk} px, particle moved a further {disp} px '
                 '(dotted green = old place, cyan = predicted move)', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    return fig


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--splits', default='val,test', help='comma-separated: train,val,test')
    ap.add_argument('--detections', default='song', choices=['song', 'hybrid_segment', 'truth'],
                    help='song = Song double threshold; hybrid_segment = needs --qf-filter; truth = perfect detections')
    ap.add_argument('--qf-filter', help='quality_filter.joblib for hybrid_segment (hybrid_segment.py must be importable)')
    ap.add_argument('--qf-tau', type=float, default=QF_TAU, help='hybrid_segment cut-off (default 0.3: favours recall)')
    ap.add_argument('--motions', default=None, help='comma-separated, default all')
    ap.add_argument('--trackers', default=None, help='comma-separated, default all')
    ap.add_argument('--stress', action='store_true', help='run the large-motion stress test instead')
    ap.add_argument('--um-per-px', type=float, default=None, help='default: read from the data (truth/config.json)')
    a = ap.parse_args()
    if a.um_per_px:
        SC.configure(a.um_per_px)
    else:
        SC.configure_from_data(a.data_dir)
    if a.detections == 'hybrid_segment':
        use_hybrid_segment(a.qf_filter, a.qf_tau, data_dir=a.data_dir)
    mo = a.motions.split(',') if a.motions else None
    tr = a.trackers.split(',') if a.trackers else None
    if a.stress:
        tab, _ = run_stress(a.data_dir, a.out_dir, tuple(a.splits.split(',')), mo, tr)
        print(tab.to_string())
    else:
        tm, tt = run(a.data_dir, a.out_dir, tuple(a.splits.split(',')), mo, tr, a.detections)
        print(tm.iloc[:, :3].to_string())
        print(tt.iloc[:, :4].to_string())
