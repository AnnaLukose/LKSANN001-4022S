# Experiment 1: feature models and CNNs on the particles the pipeline finds, plus the resolution test.
# Output: cls_tables/ (particle tables) and cls_outputs/<n>/ (scores, decisions, figures).
# Uses: scale.py, piv_compare.py.
import os, glob, time, warnings, argparse
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi
from skimage.measure import regionprops
from skimage.morphology import skeletonize
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import scale as SC

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

UM_PER_PX, MIN_AREA = 13.0, 50   # pixel size and smallest detection (px); set from the data by scale.py
CANVAS, FIT = 80, 67             # crop size and largest particle kept at native scale (px)
CHANNELS = ('raw_r', 'raw_g', 'raw_b', 'bgrem', 'mask')
SEED = 0
BOX_PAD = 4               # px round a particle's box for its measurements  (reference scale)
BG_RING = (1, 3)          # px: the gel ring used for the background colour  (reference scale)
EDGE_RING = 2             # px: the ring just outside the outline (ring_mean)  (reference scale)
AUG_SHIFT = 3             # px: largest random shift of a crop while training a CNN  (reference scale)


def _apply_scale():
    """Pixel settings at the current resolution (scale)."""
    global UM_PER_PX, MIN_AREA, CANVAS, FIT, BOX_PAD, BG_RING, EDGE_RING, AUG_SHIFT
    UM_PER_PX, MIN_AREA, CANVAS, FIT = SC.UM_PER_PX, SC.MIN_AREA, SC.CANVAS, SC.FIT
    BOX_PAD, BG_RING, EDGE_RING, AUG_SHIFT = SC.ilength(4), (SC.ilength(1), SC.ilength(3)), SC.ilength(2), SC.ilength(3)


SC.register(_apply_scale)

# class schemes
FLUX9 = ['aggregate', 'dense_detritus', 'large_loose_pellet', 'long_fecal_pellet', 'short_pellet', 'mini_pellet',
         'salp_pellet', 'rhizaria', 'phytoplankton']
SCHEMES = {
    # Amaral & Durkin (2025)
    'amaral7u': dict(aggregate='aggregate', dense_detritus='aggregate', unidentifiable='aggregate',
                     large_loose_pellet='long_pellet', long_fecal_pellet='long_pellet', short_pellet='short_pellet',
                     mini_pellet='mini_pellet', salp_pellet='salp_pellet', rhizaria='rhizaria',
                     phytoplankton='phytoplankton'),
    'amaral7': dict(aggregate='aggregate', dense_detritus='aggregate', unidentifiable='unidentifiable',
                    large_loose_pellet='long_pellet', long_fecal_pellet='long_pellet', short_pellet='short_pellet',
                    mini_pellet='mini_pellet', salp_pellet='salp_pellet', rhizaria='rhizaria',
                    phytoplankton='phytoplankton'),
    # Song et al. (2025): small ellipsoids (short + mini pellets) put in dense detritus at GelCam resolution
    'song6u': dict(aggregate='aggregate', dense_detritus='dense_detritus', short_pellet='dense_detritus',
                   mini_pellet='dense_detritus', unidentifiable='aggregate', large_loose_pellet='large_loose_pellet',
                   long_fecal_pellet='long_pellet', salp_pellet='salp_pellet', rhizaria='rhizaria',
                   phytoplankton='phytoplankton'),
    # Durkin et al. (2021): all nine flux classes
    'durkin9u': dict(aggregate='aggregate', dense_detritus='dense_detritus', unidentifiable='aggregate',
                     large_loose_pellet='large_loose_pellet', long_fecal_pellet='long_pellet',
                     short_pellet='short_pellet', mini_pellet='mini_pellet', salp_pellet='salp_pellet',
                     rhizaria='rhizaria', phytoplankton='phytoplankton'),
}
NOISE = 'false_detection'
NONFLUX = 'non_flux'          # swimmers (copepods, amphipods, pteropods, ...) and contaminants (fibres, pollen)
NO_CARBON = {NOISE, NONFLUX, 'unidentifiable'}


def scheme_label(source_class, scheme):
    m = SCHEMES[scheme]
    s = pd.Series(source_class)
    return np.where(s == NOISE, NOISE, s.map(m).fillna(NONFLUX)).astype(object)


# carbon model (Durkin et al. 2021, Table 2)
CARBON_TABLE = {
    'aggregate':          ('sphere',    0.1e-9,   0.80,  None),
    'dense_detritus':     ('sphere',    0.1e-9,   0.83,  None),
    'large_loose_pellet': ('cylinder',  0.1e-9,   0.83,  ('hyp', 553.0, 996.0)),
    'long_fecal_pellet':  ('cylinder',  0.1e-9,   1.00,  ('hyp', 187.0, 424.0)),
    'short_pellet':       ('ellipsoid', 0.1e-9,   1.00,  ('lin', 0.54)),
    'mini_pellet':        ('sphere',    0.1e-9,   1.00,  None),
    'salp_pellet':        ('cuboid',    0.04e-9,  1.00,  ('lin', 0.63)),
    'rhizaria':           ('sphere',    0.004e-9, 0.939, None),
    'phytoplankton':      ('sphere',    0.288e-9, 0.811, None),
}
# merged classes take the carbon model of their dominant member (re-scaled later from the training data)
CARBON_OF = dict(aggregate='aggregate', dense_detritus='dense_detritus', large_loose_pellet='large_loose_pellet',
                 long_pellet='long_fecal_pellet', short_pellet='short_pellet', mini_pellet='mini_pellet',
                 salp_pellet='salp_pellet', rhizaria='rhizaria', phytoplankton='phytoplankton')


def volume_um3(shape, esd, wrule=None):
    esd = np.asarray(esd, float)
    if shape == 'sphere':
        return np.pi / 6 * esd ** 3
    w = wrule[1] * esd / (esd + wrule[2]) if wrule[0] == 'hyp' else wrule[1] * esd
    w = np.minimum(w, esd)
    area = np.pi / 4 * esd ** 2
    if shape == 'cylinder':
        return np.pi * (w / 2) ** 2 * (area / w)
    if shape == 'ellipsoid':
        return 4 / 3 * np.pi * (esd ** 2 / w / 2) * (w / 2) ** 2
    if shape == 'cuboid':
        return (area / w) * w * w / 4
    raise ValueError(shape)


def carbon_mg(cls, esd):
    """Carbon (mg) of particles of class `cls` (scheme name or Durkin name) with ESD `esd` (um)."""
    key = CARBON_OF.get(cls, cls)
    esd = np.asarray(esd, float)
    if key not in CARBON_TABLE:
        return np.zeros_like(esd)
    shape, A, B, w = CARBON_TABLE[key]
    return A * volume_um3(shape, esd, w) ** B


# features
FEATURE_GROUPS = {
    'size': ['area_px', 'log_area', 'esd_um', 'major_px', 'minor_px'],
    'shape': ['aspect', 'eccentricity', 'solidity', 'extent', 'circularity', 'perim_ratio', 'feret_ratio',
              'thick_max', 'thick_mean', 'thick_ratio', 'skel_ratio'] + [f'hu{i}' for i in range(1, 8)] +
             [f'fd{i}' for i in range(2, 10)],
    'intensity': ['int_mean', 'int_max', 'int_std', 'int_p10', 'int_p25', 'int_p50', 'int_p75', 'int_p90', 'int_cv',
                  'mean_peak', 'core_frac', 'int_skew', 'int_kurt', 'radial0', 'radial1', 'radial2', 'radial3',
                  'rim_sobel_mean', 'rim_sobel_max', 'ring_mean', 'lap_var', 'rel_otsu'],
    'texture': [f'lbp{i}' for i in range(10)] + ['glcm_contrast', 'glcm_homogeneity', 'glcm_energy',
                                                 'glcm_correlation'],
    'colour': ['col_r', 'col_g', 'col_b', 'col_dr', 'col_dg', 'col_db', 'chrom_r', 'chrom_g', 'col_g_over_r',
               'col_b_over_r', 'col_r_minus_b', 'hue_mean', 'sat_mean', 'val_std'],
}
FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
# the quantities Durkin et al. (2021) use to define the classes by eye
DURKIN_FEATURES = ['esd_um', 'aspect', 'solidity', 'circularity', 'perim_ratio', 'mean_peak', 'int_mean',
                   'core_frac', 'col_r_minus_b', 'chrom_r', 'skel_ratio']


def _fourier(mask, n=8):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return np.zeros(n)
    c = max(cs, key=len)[:, 0, :].astype(float)
    if len(c) < 2 * n + 2:
        return np.zeros(n)
    F = np.abs(np.fft.fft((c[:, 0] + 1j * c[:, 1]) - (c[:, 0] + 1j * c[:, 1]).mean()))
    return np.array([F[k] + F[-k] for k in range(2, n + 2)]) / max(F[1], 1e-9)


def object_features(m, g, rgb, bg_rgb, otsu):
    """Measured features of one particle."""
    f = {}
    r = regionprops(m.astype(np.uint8), intensity_image=g)[0]
    a = float(r.area)
    maj, mnr = max(r.axis_major_length, 1.0), max(r.axis_minor_length, 1.0)
    per = max(r.perimeter, 1.0)
    f.update(area_px=a, log_area=np.log(a), esd_um=2 * np.sqrt(a / np.pi) * UM_PER_PX, major_px=maj, minor_px=mnr,
             aspect=maj / mnr, eccentricity=r.eccentricity, solidity=r.solidity, extent=r.extent,
             circularity=4 * np.pi * a / per ** 2, perim_ratio=per / np.sqrt(a),
             feret_ratio=r.feret_diameter_max / np.sqrt(a))
    hu = r.moments_hu
    for i in range(7):
        f[f'hu{i + 1}'] = -np.sign(hu[i]) * np.log10(abs(hu[i]) + 1e-12)
    for i, v in enumerate(_fourier(m)):
        f[f'fd{i + 2}'] = v
    dist = ndi.distance_transform_edt(m)
    f.update(thick_max=dist.max(), thick_mean=dist[m].mean(), thick_ratio=dist.max() / maj,
             skel_ratio=skeletonize(m).sum() / maj)
    v = g[m]
    pk = v.max() + 1e-6
    sd = v.std() + 1e-6
    f.update(int_mean=v.mean(), int_max=v.max(), int_std=v.std(), int_cv=v.std() / (v.mean() + 1e-6),
             mean_peak=v.mean() / pk, core_frac=(v >= 0.8 * pk).mean(),
             int_skew=((v - v.mean()) ** 3).mean() / sd ** 3, int_kurt=((v - v.mean()) ** 4).mean() / sd ** 4,
             rel_otsu=v.mean() / max(otsu, 1e-6))
    for q in (10, 25, 50, 75, 90):
        f[f'int_p{q}'] = np.percentile(v, q) / pk
    dn = dist / (dist.max() + 1e-6)
    for k in range(4):
        sel = m & (dn > k / 4) & (dn <= (k + 1) / 4 + 1e-9)
        f[f'radial{k}'] = g[sel].mean() / pk if sel.any() else 0.0
    sob = np.hypot(ndi.sobel(g, 0), ndi.sobel(g, 1))          # Durkin's in-focus test: Sobel on the outer edge
    rim = m & ~ndi.binary_erosion(m)
    ring = ndi.binary_dilation(m, iterations=EDGE_RING) & ~m
    f.update(rim_sobel_mean=sob[rim].mean() / pk if rim.any() else 0.0,
             rim_sobel_max=sob[rim].max() / pk if rim.any() else 0.0,
             ring_mean=g[ring].mean() / pk if ring.any() else 0.0,
             lap_var=ndi.laplace(g)[m].var() / pk ** 2)
    q = np.clip(g / pk * 15, 0, 15).astype(np.uint8)
    lbp = local_binary_pattern(q, 8, 1, 'uniform')
    h = np.bincount(lbp[m].astype(int), minlength=10)[:10] / m.sum()
    for i, x in enumerate(h):
        f[f'lbp{i}'] = x
    gl = graycomatrix(np.where(m, q, 0), [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=16, symmetric=True,
                      normed=True)
    for p in ('contrast', 'homogeneity', 'energy', 'correlation'):
        f[f'glcm_{p}'] = float(np.nan_to_num(graycoprops(gl, p)).mean())
    c = rgb[m].mean(0)
    s = c.sum() + 1e-6
    hsv = cv2.cvtColor(np.clip(rgb[m], 0, 255).astype(np.uint8)[None], cv2.COLOR_RGB2HSV)[0].astype(float)
    f.update(col_r=c[0], col_g=c[1], col_b=c[2], col_dr=c[0] - bg_rgb[0], col_dg=c[1] - bg_rgb[1],
             col_db=c[2] - bg_rgb[2], chrom_r=c[0] / s, chrom_g=c[1] / s, col_g_over_r=(c[1] + 1) / (c[0] + 1),
             col_b_over_r=(c[2] + 1) / (c[0] + 1),
             col_r_minus_b=(c[0] - bg_rgb[0]) - (c[2] - bg_rgb[2]),      # Durkin's oblique red-minus-blue cue
             hue_mean=np.angle(np.exp(1j * hsv[:, 0] * np.pi / 90).mean()) * 90 / np.pi, sat_mean=hsv[:, 1].mean(),
             val_std=hsv[:, 2].std())
    return f


def crop(raw, g, lab, label, cy, cx, side):
    """5 x CANVAS x CANVAS uint8 crop centred on the particle at native scale."""
    H, W = g.shape
    s = 1.0 if side <= FIT else FIT / side
    half = CANVAS / (2 * s)
    ys, xs = int(max(cy - half - 2, 0)), int(max(cx - half - 2, 0))
    ye, xe = int(min(cy + half + 3, H)), int(min(cx + half + 3, W))
    M = np.array([[s, 0, CANVAS / 2 - s * (cx - xs)], [0, s, CANVAS / 2 - s * (cy - ys)]], np.float32)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    sub_l = lab[ys:ye, xs:xe]
    own = sub_l == label
    sub_g = np.where((sub_l > 0) & ~own, 0, g[ys:ye, xs:xe]).astype(np.float32)
    ch = [cv2.warpAffine(np.ascontiguousarray(raw[ys:ye, xs:xe, i]), M, (CANVAS, CANVAS), flags=interp)
          for i in range(3)]
    ch.append(cv2.warpAffine(sub_g, M, (CANVAS, CANVAS), flags=interp))
    ch.append(cv2.warpAffine(own.astype(np.float32) * 255, M, (CANVAS, CANVAS), flags=cv2.INTER_LINEAR))
    return np.clip(np.stack(ch), 0, 255).astype(np.uint8), s


def frame_objects(raw, g, lab, rows, otsu):
    """Features + crops for every object of one frame."""
    H, W = g.shape
    feats, crops = [], []
    objs = {r.label: r for r in regionprops(lab)}
    for row in rows:
        r = objs.get(row['label'])
        if r is None:
            continue
        y0, x0, y1, x1 = r.bbox
        Y0, X0, Y1, X1 = max(y0 - BOX_PAD, 0), max(x0 - BOX_PAD, 0), min(y1 + BOX_PAD, H), min(x1 + BOX_PAD, W)
        lm = np.zeros((Y1 - Y0, X1 - X0), bool)
        lm[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = r.image
        gc, rc = g[Y0:Y1, X0:X1], raw[Y0:Y1, X0:X1]
        ring = ndi.binary_dilation(lm, iterations=BG_RING[1]) & ~ndi.binary_dilation(lm, iterations=BG_RING[0]) & \
            ~((lab[Y0:Y1, X0:X1] > 0) & ~lm)
        bg = rc[ring].mean(0) if ring.any() else (rc[~lm].mean(0) if (~lm).any() else np.zeros(3))
        f = object_features(lm, gc, rc, bg, otsu)
        c, s = crop(raw, g, lab, r.label, r.centroid[0], r.centroid[1], max(y1 - y0, x1 - x0))
        f.update(row, x=r.centroid[1], y=r.centroid[0], crop_scale=s)
        feats.append(f)
        crops.append(c)
    return feats, crops


# Part A: detections
def frame_nums(seq_dir):
    return sorted(int(os.path.basename(f)[6:10]) for f in glob.glob(os.path.join(seq_dir, 'bgrem', 'bgrem_*.png')))


def load_mask(seq_dir, shape):
    m = cv2.imread(os.path.join(seq_dir, 'truth', 'static_mask.png'), 0)
    return m > 0 if m is not None else np.ones(shape, bool)


def read_raw(seq_dir, n, shape):
    for f in (f'frame_{n:04d}.jpg', f'frame_{n:04d}.png'):
        p = os.path.join(seq_dir, 'raw', f)
        if os.path.exists(p):
            return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32)
    g = cv2.imread(os.path.join(seq_dir, 'bgrem', f'bgrem_{n:04d}.png'), 0).astype(np.float32)   # no raw: grey
    return np.repeat(g[..., None], 3, 2)


def truth_first_visible(seq_dir, nums, mask):
    """First frame index in which each truth particle covers >= MIN_AREA px (piv_compare's arrival rule)."""
    first = {}
    for i, n in enumerate(nums):
        t = np.where(mask, cv2.imread(os.path.join(seq_dir, 'truth', 'labels', f'frame_{n:04d}.png'), -1), 0)
        a = np.bincount(t.ravel().astype(np.int64))
        for u in np.nonzero(a >= MIN_AREA)[0]:
            if u and u not in first:
                first[int(u)] = i
    return first


def match_uid(coords, truth, t_area):
    """piv_compare's rule."""
    vals = truth[coords[:, 0], coords[:, 1]]
    vals = vals[vals > 0]
    if not len(vals):
        return 0, 0.0
    u, n = np.unique(vals, return_counts=True)
    k = int(np.argmax(n))
    iou = n[k] / (len(coords) + t_area[u[k]] - n[k])
    ok = n[k] >= 0.5 * len(coords) or n[k] >= 0.5 * t_area[u[k]]
    return (int(u[k]) if ok else 0), float(iou)


def one_track_per_detection(TT):
    """tracking can let two tracks run through the same detection (two particles merging into one outline)."""
    n = TT.groupby('track_id').pid.transform('size')
    return TT.assign(_n=n).sort_values(['_n', 'track_id'], ascending=[False, True]) \
        .drop_duplicates('pid').drop(columns='_n').sort_values(['track_id', 'pid'])


def detections_pipeline(seq_dir, motion='ransac', tracker='tracking', track=True, tracks_csv=None, every=1):
    """hybrid_segment detections and tracks (call piv_compare.use_hybrid_segment first)."""
    import piv_compare as pc
    frames = None
    if every > 1 and not track and not (tracks_csv and os.path.exists(tracks_csv)):
        frames = list(range(0, len(pc.frame_files(seq_dir)[0]), every))
    S = pc.load_seq(seq_dir, 'hybrid_segment', frames=frames)
    P = S['P']
    tid = pd.Series(-1, index=P.pid.values)
    if tracks_csv and os.path.exists(tracks_csv):
        TT = one_track_per_detection(pd.read_csv(tracks_csv))
        assert len(TT) == len(P), f'{tracks_csv}: {len(TT)} detections but hybrid_segment gives {len(P)}; set PIV_TRACKS = None'
        tid = TT.set_index('pid').track_id.reindex(P.pid.values)
    elif track:
        T = len(S['imgs'])
        fields = [pc.estimate(motion, S['imgs'][f], S['imgs'][f + 1], S['mask'], S['ctx']['dets'][f],
                              S['ctx']['dets'][f + 1]) for f in range(T - 1)]
        TT = one_track_per_detection(pc.tracks_table(pc.track_sequence(tracker, S, fields), P))
        tid = TT.set_index('pid').track_id.reindex(P.pid.values)
    rows = [[dict(label=p.pid + 1, det_id=int(p.pid), track_id=int(tid.get(p.pid, -1)), uid=int(p.uid), iou=np.nan)
             for p in P[P.frame == f].itertuples()] for f in range(len(S['imgs']))]
    labs = LazyLabels(P, S['ctx']['bmasks'], S['mask'].shape, len(S['imgs']))
    fidx = frames if frames is not None else list(range(len(S['imgs'])))
    return labs, rows, list(S['nums']), float(S['ctx']['high']), S['mask'], fidx


class LazyLabels:
    """The label image of each loaded frame (value = pid + 1), made when it is asked for."""

    def __init__(self, P, bmasks, shape, n):
        self.P, self.bmasks, self.shape, self.n = P, bmasks, shape, n

    def __len__(self):
        return self.n

    def __getitem__(self, f):
        lab = np.zeros(self.shape, np.int32)
        for p in self.P[self.P.frame == f].itertuples():
            sub = lab[p.y0:p.y1, p.x0:p.x1]
            sub[self.bmasks[p.pid]] = p.pid + 1
        return lab


def detections_truth(seq_dir, nums, mask):
    """Perfect detections."""
    labs, rows = [], []
    for n in nums:
        t = np.where(mask, cv2.imread(os.path.join(seq_dir, 'truth', 'labels', f'frame_{n:04d}.png'), -1), 0)
        t = t.astype(np.int64)
        a = np.bincount(t.ravel())
        keep = a >= MIN_AREA
        keep[0] = False
        t = np.where(keep[t], t, 0)
        labs.append(t)
        rows.append([dict(label=int(u), det_id=int(u), track_id=int(u), uid=int(u), iou=1.0)
                     for u in np.nonzero(keep)[0]])
    return labs, rows


def detections_saved(seq_dir, nums, mask):
    """The pipeline's own saved segmentation and tracks (seg/labels, seg/detections.csv)."""
    D = pd.read_csv(os.path.join(seq_dir, 'seg', 'detections.csv'))
    labs, rows = [], []
    for n in nums:
        lab = np.where(mask, cv2.imread(os.path.join(seq_dir, 'seg', 'labels', f'frame_{n:04d}.png'), -1), 0)
        labs.append(lab.astype(np.int64))
        d = D[D.frame == n]
        rows.append([dict(label=int(r.det_id), det_id=int(r.det_id), track_id=int(r.track_id), uid=-1)
                     for r in d.itertuples()])
    return labs, rows


def build_sequence(seq_dir, out_dir, source='pipeline', every=1, track=True, tracks_csv=None,
                   motion='ransac', tracker='tracking'):
    """Measure and crop every detection of one sequence."""
    sid = os.path.basename(seq_dir.rstrip('/'))
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    nums_all = frame_nums(seq_dir)
    g0 = cv2.imread(os.path.join(seq_dir, 'bgrem', f'bgrem_{nums_all[0]:04d}.png'), 0)
    mask = load_mask(seq_dir, g0.shape)
    otsu = None
    step = every                                         # measure every step-th of the loaded frames
    if source == 'pipeline':
        labs, rows, nums, otsu, mask, fidx = detections_pipeline(seq_dir, motion, tracker, track, tracks_csv, every)
        if len(nums) < len(nums_all):                    # only the measured frames were segmented
            step = 1
    elif source == 'truth':
        nums = nums_all
        labs, rows = detections_truth(seq_dir, nums, mask)
    elif source == 'saved':
        nums = nums_all
        labs, rows = detections_saved(seq_dir, nums, mask)
    else:
        raise ValueError(source)
    if source != 'pipeline':
        fidx = list(range(len(nums)))
    P = pd.read_csv(os.path.join(seq_dir, 'truth', 'particles.csv')).set_index('uid')
    first_vis = truth_first_visible(seq_dir, nums_all, mask)
    if otsu is None:
        from skimage.filters import threshold_otsu
        otsu = float(np.mean([threshold_otsu(cv2.imread(os.path.join(seq_dir, 'bgrem', f'bgrem_{n:04d}.png'), 0)[mask])
                              for n in nums]))
    feats, crops = [], []
    for i, n in enumerate(nums):
        if step > 1 and i % step:
            continue
        g = cv2.imread(os.path.join(seq_dir, 'bgrem', f'bgrem_{n:04d}.png'), 0).astype(np.float32)
        raw = read_raw(seq_dir, n, g.shape)
        lab = labs[i]
        rr = rows[i]
        if any(r['uid'] < 0 for r in rr):                # saved source: match to truth here
            t = np.where(mask, cv2.imread(os.path.join(seq_dir, 'truth', 'labels', f'frame_{n:04d}.png'), -1), 0)
            t = t.astype(np.int64)
            ta = np.bincount(t.ravel())
            objs = {r.label: r for r in regionprops(lab)}
            for r in rr:
                o = objs.get(r['label'])
                r['uid'], r['iou'] = match_uid(o.coords, t, ta) if o is not None else (0, 0.0)
        for r in rr:
            r.update(frame=n, fidx=fidx[i])
        f, c = frame_objects(raw, g, lab, rr, otsu)
        feats += f
        crops += c
    T = pd.DataFrame(feats)
    if not len(T):
        T = pd.DataFrame(columns=['uid'])
    T['seq_id'] = sid
    T['split'] = sid.split('_')[-1]
    u = T.uid.astype(int)
    T['source_class'] = np.where(u > 0, P.source_class.reindex(u).values, NOISE)
    T['source_uid'] = np.where(u > 0, P.source_uid.reindex(u).astype(str).values, 'none')
    T['carbon_true'] = np.where(u > 0, P.carbon_mgC_true.reindex(u).fillna(0).values, 0.0)
    T['esd_true'] = np.where(u > 0, P.esd_um_true.reindex(u).values, np.nan)
    T['detectable'] = np.where(u > 0, P.detectable_song.reindex(u).fillna(False).astype(bool).values, False)
    T['first_vis'] = u.map(first_vis).fillna(-1).astype(int)
    T['otsu'] = otsu
    T.to_pickle(os.path.join(out_dir, f'{sid}.pkl'))
    np.save(os.path.join(out_dir, f'{sid}_crops.npy'), np.stack(crops) if crops else
            np.zeros((0, len(CHANNELS), CANVAS, CANVAS), np.uint8))
    # truth arrivals of this sequence (for recall and carbon): every particle first visible after frame 0
    arr = [dict(seq_id=sid, uid=u_, first_vis=fv, source_class=P.source_class.get(u_),
                carbon_true=float(np.nan_to_num(P.carbon_mgC_true.get(u_, 0.0))),
                detectable=bool(P.detectable_song.get(u_, False)), source_uid=str(P.source_uid.get(u_)))
           for u_, fv in first_vis.items() if fv > 0]
    pd.DataFrame(arr).to_csv(os.path.join(out_dir, f'{sid}_truth_arrivals.csv'), index=False)
    print(f'{sid}: {len(T)} detections, {T.track_id.nunique() if len(T) else 0} tracks ({time.time() - t0:.0f} s)',
          flush=True)
    return T


def build_all(seq_dirs, out_dir, source='pipeline', every_train=3, tracks_dir=None, motion='ransac',
              tracker='tracking'):
    """Part A over many sequences."""
    for d in seq_dirs:
        sid = os.path.basename(d.rstrip('/'))
        if os.path.exists(os.path.join(out_dir, f'{sid}.pkl')):
            continue
        is_train = sid.endswith('_train')
        tcsv = os.path.join(tracks_dir, sid, f'{motion}__{tracker}.csv') if tracks_dir else None
        build_sequence(d, out_dir, source, every_train if is_train else 1, track=not is_train, tracks_csv=tcsv,
                       motion=motion, tracker=tracker)


def load_tables(cls_dir, splits=('train', 'val', 'test'), crops=True):
    """All built sequences of the given splits -> (table, crops array)."""
    Ts, Cs = [], []
    for f in sorted(glob.glob(os.path.join(cls_dir, '*.pkl'))):
        sid = os.path.basename(f)[:-4]
        if sid.split('_')[-1] not in splits:
            continue
        T = pd.read_pickle(f)
        Ts.append(T)
        if crops:
            Cs.append(np.load(os.path.join(cls_dir, f'{sid}_crops.npy'), mmap_mode='r'))
    if not Ts:
        raise FileNotFoundError(f'no built sequences of {splits} in {cls_dir}: run Part A')
    T = pd.concat(Ts, ignore_index=True)
    C = np.concatenate([np.asarray(c) for c in Cs]) if crops else None
    return T, C


def truth_arrivals(cls_dir, splits):
    f = [x for x in sorted(glob.glob(os.path.join(cls_dir, '*_truth_arrivals.csv')))
         if os.path.basename(x).replace('_truth_arrivals.csv', '').split('_')[-1] in splits]
    return pd.concat([pd.read_csv(x) for x in f], ignore_index=True) if f else pd.DataFrame()


# labels and weights
def add_labels(T, scheme):
    T = T.copy()
    T['label'] = scheme_label(T.source_class.values, scheme)
    # one "particle" per truth uid (train) or per track (val/test); false detections are single detections
    key = np.where(T.uid > 0, T.seq_id + ':u' + T.uid.astype(str), T.seq_id + ':d' + T.index.astype(str))
    T['pkey'] = key
    T['w_particle'] = 1.0 / T.groupby('pkey').pkey.transform('size').values
    return T


def class_weights(T, power=0.5):
    """Weight per row = 1/(rows of its particle) x (largest class / its class)^power."""
    mass = T.groupby('label').w_particle.sum()
    cw = (mass.max() / mass) ** power
    return T.w_particle.values * cw.reindex(T.label.values).values


# feature models
def make_feature_model(name, n_classes, seed=SEED):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == 'durkin_tree':
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=seed)
    if name == 'lda':
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    if name == 'logreg':
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5))
    if name == 'knn':
        from sklearn.neighbors import KNeighborsClassifier
        return make_pipeline(StandardScaler(), KNeighborsClassifier(25, weights='distance'))
    if name == 'svm':
        from sklearn.svm import SVC
        return make_pipeline(StandardScaler(), SVC(C=3, gamma='scale', probability=True, random_state=seed))
    if name == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(300, min_samples_leaf=3, max_features='sqrt', n_jobs=-1, random_state=seed)
    if name == 'extratrees':
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(300, min_samples_leaf=3, max_features='sqrt', n_jobs=-1, random_state=seed)
    if name == 'hgb':
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                              l2_regularization=1.0, random_state=seed)
    if name == 'lgbm':
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=500, learning_rate=0.04, num_leaves=31, min_child_samples=30,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=2.0,
                                  verbose=-1, random_state=seed, n_jobs=-1)
    if name == 'xgb':
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8,
                                 colsample_bytree=0.6, reg_lambda=2.0, min_child_weight=3, tree_method='hist',
                                 random_state=seed, n_jobs=-1)
    if name == 'mlp':
        from sklearn.neural_network import MLPClassifier
        return make_pipeline(StandardScaler(), MLPClassifier((256, 128), alpha=1e-3, learning_rate_init=1e-3,
                                                             early_stopping=True, max_iter=300, random_state=seed))
    raise ValueError(name)


FEATURE_MODELS = ['durkin_tree', 'lda', 'logreg', 'knn', 'svm', 'rf', 'extratrees', 'hgb', 'lgbm', 'xgb', 'mlp']
NO_SAMPLE_WEIGHT = {'lda', 'knn', 'mlp'}        # these ignore weights: trained on a class-balanced resample instead
# models trained on the class index instead of the class name
LABELS_AS_INDEX = {'xgb', 'mlp'}


def _balanced_resample(n_rows, w, seed, max_rows=60000):
    rng = np.random.default_rng(seed)
    p = w / w.sum()
    return rng.choice(n_rows, size=min(max_rows, n_rows), replace=True, p=p)


def fit_feature_model(name, tr, feats=None, power=0.5, seed=SEED, max_rows_svm=12000):
    """Train one feature model on the table tr (needs 'label')."""
    feats = feats or (DURKIN_FEATURES if name == 'durkin_tree' else FEATURES)
    X = tr[feats].astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0).values
    y = tr.label.values
    w = class_weights(tr, power)
    classes = np.array(sorted(np.unique(y)))
    m = make_feature_model(name, len(classes), seed)
    yf = np.searchsorted(classes, y) if name in LABELS_AS_INDEX else y
    if name in NO_SAMPLE_WEIGHT or name == 'svm':
        idx = _balanced_resample(len(X), w, seed, max_rows_svm if name in ('svm', 'knn') else 60000)
        m.fit(X[idx], yf[idx])
    elif name == 'logreg':
        m.fit(X, yf, logisticregression__sample_weight=w / w.mean())
    else:
        m.fit(X, yf, sample_weight=w / w.mean())
    m.feats_, m.labels_ = feats, classes
    return m


def predict_feature_model(m, T):
    X = T[m.feats_].astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0).values
    P = m.predict_proba(X)
    # the classes behind the columns of P: their names, or (models trained on the index) the names they stand for
    cl = np.asarray(m.classes_) if hasattr(m, 'classes_') else np.arange(P.shape[1])
    got = np.asarray(m.labels_)[cl.astype(int)] if np.issubdtype(cl.dtype, np.number) else cl
    out = np.zeros((len(T), len(m.labels_)))
    for j, c in enumerate(got):
        out[:, np.searchsorted(m.labels_, c)] = P[:, j]
    return np.clip(out, 1e-6, 1)


def constant_probs(n_rows, classes, cls='aggregate'):
    """The 'always aggregate' baseline."""
    P = np.full((n_rows, len(classes)), 1e-6)
    P[:, list(classes).index(cls)] = 1.0
    return P


def natural_prior(tr):
    """Class frequencies (one count per particle) in the NATURAL training sequences only."""
    nat = tr[~tr.seq_id.str.startswith('rare')]
    nat = nat if len(nat) else tr
    return nat.drop_duplicates('pkey').label.value_counts(normalize=True).to_dict()


# CNNs
CNN_MODELS = ['resnet18_amaral', 'mobilenet_w', 'native_cnn', 'hybrid']


def _torch():
    import torch
    return torch


def build_net(arch, n_cls, n_feat=0):
    import torch.nn as nn
    torch = _torch()
    if arch in ('resnet18_amaral', 'mobilenet_w'):
        import timm
        name = 'resnet18' if arch == 'resnet18_amaral' else 'mobilenetv2_100'
        try:
            net = timm.create_model(name, pretrained=True, num_classes=n_cls, in_chans=3)
        except Exception as e:                                       # offline: no ImageNet weights
            print('ImageNet weights unavailable, training from scratch:', e)
            net = timm.create_model(name, pretrained=False, num_classes=n_cls, in_chans=3)
        net.input_size, net.channels = 224, [0, 1, 2]                 # RGB crop, resized to 224 (Amaral & Durkin)
        net.n_feat = 0
        return net

    class Native(nn.Module):
        """Four conv blocks at full resolution for CANVAS px crops."""
        def __init__(self, in_ch, n_feat, n_cls, w=32, drop=0.3):
            super().__init__()
            def blk(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                                     nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))
            self.blocks = nn.ModuleList([blk(in_ch, w), blk(w, 2 * w), blk(2 * w, 4 * w), blk(4 * w, 8 * w)])
            self.pool = nn.MaxPool2d(2)
            self.n_feat = n_feat
            if n_feat:
                self.fmlp = nn.Sequential(nn.Linear(n_feat, 128), nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(drop),
                                          nn.Linear(128, 128), nn.ReLU(True))
            self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(8 * w + (128 if n_feat else 0), 256), nn.ReLU(True),
                                      nn.Dropout(drop), nn.Linear(256, n_cls))

        def embed(self, x, f=None):
            for i, b in enumerate(self.blocks):
                x = b(x)
                if i < 3:
                    x = self.pool(x)
            e = x.mean((2, 3))
            if self.n_feat:
                e = torch.cat([e, self.fmlp(f)], 1)
            return e

        def forward(self, x, f=None):
            return self.head(self.embed(x, f))

    net = Native(len(CHANNELS), n_feat if arch == 'hybrid' else 0, n_cls)
    net.input_size, net.channels = CANVAS, list(range(len(CHANNELS)))
    return net


def _batch(C, idx, net, device, train, feats=None):
    torch = _torch()
    import torch.nn.functional as Fn
    x = torch.from_numpy(np.ascontiguousarray(C[idx][:, net.channels])).to(device).float() / 255.0
    if train:
        k = np.random.randint(4)
        x = torch.rot90(x, k, (2, 3))
        if np.random.rand() < 0.5:
            x = torch.flip(x, (3,))
        n_img = min(4, x.shape[1]) if len(net.channels) == len(CHANNELS) else x.shape[1]
        g = torch.empty(len(idx), 1, 1, 1, device=device).uniform_(0.7, 1.4)        # gamma
        b = torch.empty(len(idx), 1, 1, 1, device=device).uniform_(0.8, 1.25)       # gain
        x = torch.cat([(x[:, :n_img].clamp(1e-4, 1) ** g) * b, x[:, n_img:]], 1)
        sh = tuple(int(v) for v in np.random.randint(-AUG_SHIFT, AUG_SHIFT + 1, 2))
        x = torch.roll(x, sh, (2, 3))
    if net.input_size != x.shape[-1]:
        x = Fn.interpolate(x, size=(net.input_size, net.input_size), mode='bilinear', align_corners=False)
    f = None
    if feats is not None and getattr(net, 'n_feat', 0):
        f = torch.from_numpy(feats[idx]).to(device).float()
        if train:
            f = f + 0.05 * torch.randn_like(f)
    return x, f


def _forward(net, x, f):
    return net(x, f) if getattr(net, 'n_feat', 0) else net(x)


def cnn_predict(net, C, feats=None, device='cpu', bs=512, tta=True, embed=False):
    """Class probabilities (or embeddings), averaged over the 8 rotations / flips (TTA) in log space."""
    torch = _torch()
    import torch.nn.functional as Fn
    net.eval()
    out = []
    if net.input_size >= 128:                     # 224 px ImageNet inputs: smaller batches keep GPU memory in bounds
        bs = min(bs, 128)
    with torch.no_grad():
        for i in range(0, len(C), bs):
            idx = np.arange(i, min(i + bs, len(C)))
            x, f = _batch(C, idx, net, device, False, feats)
            views = [(k, fl) for k in range(4) for fl in (False, True)] if tta else [(0, False)]
            acc = 0
            for k, fl in views:
                xt = torch.rot90(x, k, (2, 3))
                if fl:
                    xt = torch.flip(xt, (3,))
                with torch.autocast('cuda', enabled=device == 'cuda'):
                    o = net.embed(xt, f) if embed else Fn.log_softmax(_forward(net, xt, f).float(), 1)
                acc = acc + o.float()
            acc = acc / len(views)
            out.append(acc.cpu().numpy() if embed else acc.exp().cpu().numpy())
    P = np.concatenate(out) if out else np.zeros((0, 1))
    return P if embed else P / P.sum(1, keepdims=True)


def feature_matrix(T, stats=None):
    X = T[FEATURES].astype(np.float32).replace([np.inf, -np.inf], np.nan).fillna(0).values
    if stats is None:
        stats = (X.mean(0), X.std(0) + 1e-6)
    return np.clip((X - stats[0]) / stats[1], -6, 6).astype(np.float32), stats


def train_cnn(arch, tr, Ctr, va=None, Cva=None, epochs=30, patience=6, power=0.5, lr=None, wd=None, bs=128,
              seed=SEED, device=None, max_per_epoch=None, label_smoothing=None, log=print, score_fn=None):
    """Train one CNN."""
    torch = _torch()
    import torch.nn.functional as Fn
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)
    classes = np.array(sorted(tr.label.unique()))
    y = np.searchsorted(classes, tr.label.values)
    Ftr, stats = feature_matrix(tr)
    Fva = feature_matrix(va, stats)[0] if va is not None else None
    net = build_net(arch, len(classes), Ftr.shape[1]).to(device)
    imagenet = arch in ('resnet18_amaral', 'mobilenet_w')
    lr = lr or (1e-3 if imagenet else 2e-3)
    wd = wd if wd is not None else (0.01 if arch == 'resnet18_amaral' else 1e-4)
    label_smoothing = label_smoothing if label_smoothing is not None else (0.0 if imagenet else 0.05)
    max_per_epoch = max_per_epoch or (30000 if imagenet else 60000)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    loss_w = None
    if imagenet:
        w = tr.w_particle.values                                    # natural class frequencies
        if arch == 'mobilenet_w':
            cw = pd.Series(class_weights(tr, power) / tr.w_particle.values).groupby(tr.label.values).first()
            loss_w = torch.tensor((cw / cw.mean()).reindex(classes).values, dtype=torch.float32, device=device)
    else:
        w = class_weights(tr, power)
    use_amp = device == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    n_ep = min(len(tr), max_per_epoch)
    steps = epochs * (n_ep // bs + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    yt = torch.from_numpy(y).to(device)
    best, best_state, bad, hist = -1, None, 0, []
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        net.train()
        t0 = time.time()
        idx_all = rng.choice(len(tr), size=n_ep, replace=True, p=w / w.sum())
        tot = 0.0
        for i in range(0, n_ep, bs):
            idx = np.sort(idx_all[i:i + bs])
            if len(idx) < 2:
                continue
            x, f = _batch(Ctr, idx, net, device, True, Ftr)
            opt.zero_grad()
            with torch.autocast('cuda', enabled=use_amp):
                loss = Fn.cross_entropy(_forward(net, x, f), yt[idx], weight=loss_w, label_smoothing=label_smoothing)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        rec = dict(epoch=ep, train_loss=tot / n_ep, seconds=round(time.time() - t0, 1))
        if va is not None:
            Pva = cnn_predict(net, Cva, Fva, device, tta=False)
            s = score_fn(va, Pva, classes) if score_fn else particle_balanced_accuracy(va, Pva, classes)
            rec['val_score'] = s
            if s > best:
                best, bad = s, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
        hist.append(rec)
        log(rec)
        if va is not None and bad >= patience:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.classes_, net.fstats_, net.arch_ = classes, stats, arch
    net.history_ = pd.DataFrame(hist)
    return net


def particle_balanced_accuracy(T, P, classes):
    """Quick per-particle balanced accuracy (mean probability per particle; used for early stopping)."""
    df = pd.DataFrame(P, columns=classes)
    df['pkey'] = T.pkey.values
    g = df.groupby('pkey').mean()
    true = T.groupby('pkey').label.first().reindex(g.index)
    pred = np.asarray(classes)[g.values.argmax(1)]
    ok = pd.Series(pred == true.values, index=g.index)
    return float(ok.groupby(true.values).mean().mean())


def save_cnn(net, path):
    torch = _torch()
    torch.save(dict(state=net.state_dict(), classes=list(net.classes_), fstats=net.fstats_, arch=net.arch_), path)


def load_cnn(path, device=None):
    torch = _torch()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    d = torch.load(path, map_location=device, weights_only=False)
    net = build_net(d['arch'], len(d['classes']), len(FEATURES)).to(device)
    net.load_state_dict(d['state'])
    net.classes_, net.fstats_, net.arch_ = np.array(d['classes']), d['fstats'], d['arch']
    return net


def predict_cnn(net, T, C, device=None, tta=True, embed=False):
    torch = _torch()
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    F = feature_matrix(T, net.fstats_)[0]
    return cnn_predict(net, C, F, device, tta=tta, embed=embed)


# particles and decisions
def particle_table(T, P, classes, K=None, how='geo'):
    """One row per particle (track) with combined class probabilities over its first K detections."""
    df = pd.DataFrame(np.log(np.clip(P, 1e-6, 1)) if how == 'geo' else P, columns=classes)
    df['tkey'] = (T.seq_id + ':' + T.track_id.astype(str)).values
    df['fidx'] = T.fidx.values
    df = df.sort_values(['tkey', 'fidx'])
    if K:
        df = df.groupby('tkey').head(K)
    pt = df.groupby('tkey')[list(classes)].mean()
    if how == 'geo':
        pt = np.exp(pt.sub(pt.max(1), axis=0))
    pt = pt.div(pt.sum(1), axis=0)
    return pt


def particles_info(T, K=None):
    """Per track: truth (majority uid and its class), start frame."""
    T = T.assign(tkey=(T.seq_id + ':' + T.track_id.astype(str)).values).sort_values(['tkey', 'fidx'])
    head = T.groupby('tkey').head(K) if K else T
    g = T.groupby('tkey')
    info = pd.DataFrame({'seq_id': g.seq_id.first(), 'start': g.fidx.min(), 'n_det': g.size(),
                         'uid': g.uid.agg(lambda x: x.mode().iloc[0])})
    h = head.groupby('tkey')
    for c in ('esd_um', 'solidity', 'aspect', 'mean_peak', 'area_px'):
        info[c] = h[c].median()
    first = T[T.uid > 0].groupby(['seq_id', 'uid']).fidx.min()
    fd = first.reindex(pd.MultiIndex.from_arrays([info.seq_id, info.uid])).values
    by_uid = T[T.uid > 0].drop_duplicates(['seq_id', 'uid']).set_index(['seq_id', 'uid'])
    ix = pd.MultiIndex.from_arrays([info.seq_id, info.uid])
    for c in ('source_class', 'carbon_true', 'esd_true', 'first_vis', 'source_uid', 'detectable'):
        info[c] = by_uid[c].reindex(ix).values
    info.loc[info.uid <= 0, 'source_class'] = NOISE
    info['carbon_true'] = info.carbon_true.fillna(0.0)
    info['counted'] = info.start > 0
    ok = (info.uid > 0) & info.counted & (info.first_vis > 0) & (info.start == fd)
    # a particle split into two outlines in its first frame gives two tracks that both start correctly
    order = info[ok].assign(_k=-info.n_det[ok]).sort_values(['seq_id', 'uid', '_k'])
    keep = ~order.duplicated(['seq_id', 'uid'])
    info['true_arrival'] = False
    info.loc[order.index[keep.values], 'true_arrival'] = True
    info['category'] = np.select(
        [~info.counted, info.true_arrival, ok, info.uid <= 0],
        ['not counted (in view at frame 0)', 'arrival', 'duplicate track', 'false detection'],
        'track break / late start')
    return info


def decide(pt, classes, reject=0.5, tau=0.0, prior=None):
    """Arg-max after logit adjustment (Menon 2021)."""
    classes = np.asarray(classes)
    L = np.log(np.clip(pt.values, 1e-9, 1))
    if tau and prior is not None:
        L = L - tau * np.log(np.clip(np.asarray([prior.get(c, 1e-3) for c in classes]), 1e-6, 1))[None]
    real = classes != NOISE
    pred = classes[real][L[:, real].argmax(1)] if real.any() else classes[L.argmax(1)]
    if (~real).any():
        pn = pt.values[:, ~real].max(1)
        pred = np.where(pn >= reject, NOISE, pred)
    return pd.Series(pred, index=pt.index)


# carbon
def fit_size_model(T):
    """Robust regression of log(true ESD) on log(measured ESD) and shape, per particle (median over its frames)."""
    from sklearn.linear_model import HuberRegressor
    g = T[(T.uid > 0) & T.esd_true.notna()].groupby('pkey').agg(m=('esd_um', 'median'), t=('esd_true', 'first'),
                                                                 sol=('solidity', 'median'), asp=('aspect', 'median'),
                                                                 mp=('mean_peak', 'median'))
    X = np.c_[np.log(g.m), np.log(g.m) ** 2, g.sol, np.log(g.asp), g.mp]
    r = HuberRegressor(max_iter=1000).fit(X, np.log(g.t))
    return dict(coef=r.coef_.tolist(), intercept=float(r.intercept_))


def apply_size_model(sm, info):
    if sm is None:
        return info.esd_um.values
    X = np.c_[np.log(info.esd_um), np.log(info.esd_um) ** 2, info.solidity, np.log(info.aspect), info.mean_peak]
    return np.exp(X @ np.asarray(sm['coef']) + sm['intercept'])


def particle_carbon(pred, esd, scales=None):
    scales = scales or {}
    return np.array([float(carbon_mg(c, e)) * scales.get(c, 1.0) for c, e in zip(pred, esd)])


def fit_class_scales(pred, esd, carbon_true, min_n=20, clip=(0.05, 20)):
    """Per predicted class."""
    df = pd.DataFrame(dict(pred=pred, c=particle_carbon(pred, esd), t=carbon_true))
    out = {}
    for c, g in df.groupby('pred'):
        if c in NO_CARBON or len(g) < min_n or g.c.sum() <= 0:
            continue
        out[c] = float(np.clip(g.t.sum() / g.c.sum(), *clip))
    return out


# scoring
def score(info, pred, carbon_pred=None, truth_arr=None, scheme='amaral7u'):
    """Metrics over particles."""
    pred = pd.Series(pred, index=info.index)
    true = pd.Series(scheme_label(info.source_class.values, scheme), index=info.index)
    ta = info.true_arrival.values
    ok = (pred.values == true.values)[ta]
    tl = true.values[ta]
    per = pd.Series(ok).groupby(tl).mean()
    out = dict(n_particles=int(ta.sum()), accuracy=float(ok.mean()) if ta.any() else np.nan,
               balanced_accuracy=float(per.mean()) if len(per) else np.nan)
    from sklearn.metrics import f1_score
    out['macro_f1'] = float(f1_score(tl, pred.values[ta], average='macro', labels=sorted(set(tl)))) if ta.any() else np.nan
    a, b = out['accuracy'], out['balanced_accuracy']
    # harmonic mean of plain and balanced accuracy
    out['acc_bal_hmean'] = float(2 * a * b / (a + b)) if a + b > 0 else np.nan
    flux = [c for c in per.index if c not in NO_CARBON]
    out['flux_balanced_accuracy'] = float(per[flux].mean()) if flux else np.nan
    ct = info.carbon_true.values[ta]
    out['carbon_weighted_accuracy'] = float(ct[ok].sum() / ct.sum()) if ct.sum() > 0 else np.nan
    big = info.esd_um.values[ta] >= 500
    out['accuracy_esd_ge_500um'] = float(ok[big].mean()) if big.any() else np.nan
    if carbon_pred is not None:
        cp = pd.Series(carbon_pred, index=info.index)
        # classification only: same particles, true vs predicted carbon (size measured on the image)
        out['carbon_error_found'] = float(cp.values[ta].sum() / ct.sum() - 1) if ct.sum() > 0 else np.nan
        if truth_arr is not None and len(truth_arr):
            tt = truth_arr[truth_arr.seq_id.isin(info.seq_id.unique())]
            tot = tt[tt.detectable.astype(bool)].carbon_true.sum()
            out['carbon_error_all'] = float(cp[info.counted.values].sum() / tot - 1) if tot > 0 else np.nan
    return out, per


def carbon_budget(info, carbon_pred, truth_arr):
    """Where the counted carbon comes from."""
    d = pd.DataFrame({'category': info.category.values, 'pred': np.asarray(carbon_pred, float),
                      'true': info.carbon_true.values, 'counted': info.counted.values})
    d = d[d.counted]
    tab = d.groupby('category').agg(tracks=('pred', 'size'), predicted_ug=('pred', 'sum'), true_of_its_particle_ug=('true', 'sum'))
    tab[['predicted_ug', 'true_of_its_particle_ug']] *= 1000
    if truth_arr is not None and len(truth_arr):
        tt = truth_arr[truth_arr.seq_id.isin(info.seq_id.unique()) & truth_arr.detectable.astype(bool)]
        found = set(zip(info.seq_id[info.true_arrival], info.uid[info.true_arrival]))
        miss = tt[[(s, u) not in found for s, u in zip(tt.seq_id, tt.uid)]]
        tab.loc['missed arrivals (no track)'] = [len(miss), 0.0, 1000 * miss.carbon_true.sum()]
        tab.loc['TRUE TOTAL (detectable arrivals)'] = [len(tt), np.nan, 1000 * tt.carbon_true.sum()]
        tab.loc['PIPELINE TOTAL (all counted tracks)'] = [len(d), 1000 * d.pred.sum(), np.nan]
    return tab


def per_sequence(info, pred, carbon_pred, truth_arr, scheme):
    rows = []
    for s in sorted(info.seq_id.unique()):
        m = (info.seq_id == s).values
        r, _ = score(info[m], np.asarray(pred)[m], np.asarray(carbon_pred)[m], truth_arr, scheme)
        rows.append(dict(seq_id=s, **r))
    return pd.DataFrame(rows)


def bootstrap_ci(info, pred, carbon_pred, truth_arr, scheme, n=500, seed=SEED,
                 keys=('accuracy', 'balanced_accuracy', 'acc_bal_hmean', 'carbon_weighted_accuracy', 'carbon_error_found')):
    """95 % intervals by resampling source sprites (one sprite = several particles in a sequence)."""
    rng = np.random.default_rng(seed)
    sp = info.source_uid.fillna('none').astype(str).values
    groups = {k: np.nonzero(sp == k)[0] for k in np.unique(sp)}
    names = np.array(list(groups))
    vals = {k: [] for k in keys}
    pred, carbon_pred = np.asarray(pred), np.asarray(carbon_pred)
    for _ in range(n):
        pick = np.concatenate([groups[k] for k in rng.choice(names, len(names), replace=True)])
        r, _ = score(info.iloc[pick].reset_index(drop=True), pred[pick], carbon_pred[pick], None, scheme)
        for k in keys:
            vals[k].append(r.get(k, np.nan))
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in vals.items()}


def confusion(info, pred, scheme):
    ta = info.true_arrival.values
    true = scheme_label(info.source_class.values, scheme)[ta]
    return pd.crosstab(pd.Series(true, name='true'), pd.Series(np.asarray(pred)[ta], name='predicted'))


# helpers for the notebook
def evaluate_probs(Tv, Pv, classes, K, how, reject, tau, prior, sm, scales, truth_arr, scheme, carbon_tau=0.0):
    """Probabilities per detection -> particle decision -> carbon -> metrics."""
    pt = particle_table(Tv, Pv, classes, K, how)
    info = particles_info(Tv, K).reindex(pt.index)
    pred = decide(pt, classes, reject, tau, prior)
    cpred = pred if carbon_tau is None or carbon_tau == tau else decide(pt, classes, reject, carbon_tau, prior)
    info['carbon_class'] = cpred.values
    esd = apply_size_model(sm, info)
    cp = particle_carbon(cpred.values, esd, scales)
    r, per = score(info, pred.values, cp, truth_arr, scheme)
    return r, info, pred, cp, per


def oof_feature_probs(name, tr, n_folds=4, seed=SEED, power=0.5):
    """Out-of-fold probabilities on the training table, folds = training sequences (GroupKFold)."""
    from sklearn.model_selection import GroupKFold
    classes = np.array(sorted(tr.label.unique()))
    P = np.zeros((len(tr), len(classes)))
    seqs = tr.seq_id.values
    for k, (a, b) in enumerate(GroupKFold(min(n_folds, len(set(seqs)))).split(tr, groups=seqs)):
        m = fit_feature_model(name, tr.iloc[a], power=power, seed=seed + k)
        Pk = predict_feature_model(m, tr.iloc[b])
        for j, c in enumerate(m.labels_):
            P[b, np.searchsorted(classes, c)] = Pk[:, j]
    return P, classes


def train_particles(tr, P, classes, how='geo'):
    """Training-set particles (grouped by truth uid) with combined OOF probabilities, for carbon scales."""
    df = pd.DataFrame(np.log(np.clip(P, 1e-6, 1)), columns=classes)
    df['pkey'] = tr.pkey.values
    pt = df.groupby('pkey').mean()
    pt = np.exp(pt.sub(pt.max(1), axis=0))
    pt = pt.div(pt.sum(1), axis=0)
    g = tr.groupby('pkey')
    info = pd.DataFrame({'esd_um': g.esd_um.median(), 'solidity': g.solidity.median(), 'aspect': g.aspect.median(),
                         'mean_peak': g.mean_peak.median(), 'carbon_true': g.carbon_true.first(),
                         'uid': g.uid.first()}).reindex(pt.index)
    return pt, info



if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd')
    b = sub.add_parser('build')
    b.add_argument('data_dirs', nargs='+')
    b.add_argument('--out', required=True)
    b.add_argument('--source', default='pipeline', choices=['pipeline', 'truth', 'saved'])
    b.add_argument('--qf', default=None)
    b.add_argument('--tau', type=float, default=0.3)
    b.add_argument('--every-train', type=int, default=3)
    a = ap.parse_args()
    if a.cmd == 'build':
        SC.configure_from_data(a.data_dirs[0])
        if a.source == 'pipeline':
            import piv_compare as pc
            pc.use_hybrid_segment(a.qf, a.tau, data_dir=a.data_dirs[0])
        seqs = sorted(d for r in a.data_dirs for d in glob.glob(os.path.join(r, '*_*')) if os.path.isdir(d))
        build_all(seqs, a.out, a.source, a.every_train)
