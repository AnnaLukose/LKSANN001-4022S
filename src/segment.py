# Step 6: hybrid_segment - candidate outlines per blob, a trained quality filter and a size correction.
# Output: seg/<seq>/labels/*.png, detections.csv and scores.csv (scored against the truth).
# Uses: config.py. Used by: track.py, tables.py, flux.py.
import glob, os
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu, sobel
from skimage.measure import label, regionprops
from skimage.morphology import h_maxima
from skimage.segmentation import watershed, relabel_sequential
from . import config as C

# settings
IOU_MIN = 0.5                        # a detection matches a true particle when IoU >= this
LOW, SEED = 0.5, 0.8                 # blob floor and minimum peak, x T_seq
FRACS = (0.35, 0.5, 0.65)            # outline options relative to the blob's peak
LOCAL_H = 0.3                        # valley depth (x T_seq) that separates two bright cores; None = off
LOCAL_FRACS = (0.35, 0.5)            # outline options relative to each core's own peak
SELECT = 'particle'                  # 'particle' (best non-overlapping candidates) or 'blob' (one option per blob)
LONG_RATIO = 3.0                     # length / width above which a candidate is 'long' (may get its own tau)
TAU_RULE = 'total'                   # tau whose out-of-fold total carbon is closest to 100 % ('recall': volume rule)
MIN_CARBON_PRECISION = 90.0          # %: only for TAU_RULE = 'recall'
TAUS = np.round(np.arange(0.05, 0.96, 0.05), 2)
CALIBRATE_SIZE = True                # size correction per size class
LAM = 1.0                            # 'blob' selection: expected correct - LAM x expected wrong
FEATURES = ['area', 'peak', 'mean', 'mean_peak', 'rim_grey', 'rim_sobel_mean', 'rim_sobel_max', 'ring_mean',
            'ring_bright', 'solidity', 'ecc', 'extent', 'perim_ratio', 'nn_dist', 'n_within_20', 'frac']
YELLOW = (0, 255, 255)               # BGR, for the box figures


def px():
    """Pixel settings at the current resolution (reference values in brackets)."""
    k = C.K
    return dict(min_area=C.MIN_AREA,                                   # [50 px]
                cand_min_area=max(4, int(round(0.8 * C.MIN_AREA))),     # [40 px]: thin particles are drawn smaller
                calib_bins=tuple(int(round(C.MIN_AREA * f)) for f in (1, 2, 8)),   # [50, 100, 400 px]
                local_sigma=1.0 * k,                                    # [1 px] smoothing before finding cores
                crowd_r=20.0 * k,                                       # [20 px] radius of the crowding count
                box_pad=max(2, int(round(2 * k))))


def size_bins():
    b = px()['calib_bins']
    return {f'small {b[0]}-{b[1] - 1} px': (b[0], b[1]), f'medium {b[1]}-{b[2] - 1} px': (b[1], b[2]),
            f'large >={b[2]} px': (b[2], 10 ** 12)}


# helpers
def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def peaks(img, lab):
    """Brightest pixel of every label (index = label)."""
    out = np.zeros(lab.max() + 1, np.float32)
    on = lab > 0
    np.maximum.at(out, lab[on], img[on])
    return out


def drop_small(lab, min_area):
    area = np.bincount(lab.ravel())
    small = area < min_area
    small[0] = True
    lab = lab.copy()
    lab[small[lab]] = 0
    return lab


def frames_of(seq_dir):
    return sorted(glob.glob(os.path.join(seq_dir, 'bgrem', 'bgrem_*.png')))


def frame_num(path):
    return int(os.path.basename(path).split('_')[-1][:4])


def load_mask(seq_dir, shape):
    f = os.path.join(seq_dir, 'truth', 'static_mask.png')
    return cv2.imread(f, 0) > 0 if os.path.exists(f) else np.ones(shape, bool)


def sequence_otsu(seq_dir, mask):
    """T_seq: mean of the per-frame Otsu thresholds inside the mask (Song et al. 2025's 'high' threshold)."""
    return float(np.mean([threshold_otsu(cv2.imread(f, 0)[mask]) for f in frames_of(seq_dir)]))


def load_truth(seq_dir, fi, mask):
    f = os.path.join(seq_dir, 'truth', 'labels', f'frame_{fi:04d}.png')
    return np.where(mask, cv2.imread(f, cv2.IMREAD_UNCHANGED), 0).astype(np.int64) if os.path.exists(f) else None


def volume(area_px):
    """Equivalent spherical volume (mm^3) of areas in px."""
    esd = 2 * np.sqrt(np.asarray(area_px, float) / np.pi) * C.UM_PER_PX / 1000
    return np.pi / 6 * esd ** 3


def corrected(area_px, scale):
    """Areas x the size correction of their size class (scale: {lower edge px: factor} or None)."""
    area_px = np.asarray(area_px, float)
    if not scale:
        return area_px
    e = np.array(sorted(scale))
    f = np.array([scale[x] for x in e])
    return area_px * f[np.clip(np.searchsorted(e, area_px, side='right') - 1, 0, len(f) - 1)]


def elongation(ecc):
    return 1 / np.sqrt(np.clip(1 - np.asarray(ecc, float) ** 2, 1e-6, 1))


# true carbon
class Carbon:
    """True carbon of a sequence's particles, and the carbon a detection gets after a correct classification."""

    def __init__(self, seq_dir):
        f = os.path.join(seq_dir, 'truth', 'particles.csv')
        self.ok = os.path.exists(f)
        if not self.ok:
            return
        self.p = pd.read_csv(f).set_index('uid')
        self.laws = {}
        good = self.p[self.p.carbon_mgC_true.notna() & (self.p.area_um2_true > 0)]
        for c, g in good.groupby('source_class'):
            if len(g) >= 2 and g.area_um2_true.nunique() > 1:
                b, a = np.polyfit(np.log(g.area_um2_true), np.log(g.carbon_mgC_true), 1)
                self.laws[c] = (a, b)

    def truth(self, ids):
        """(carbon, is_flux, physical area in px) of true particle ids."""
        if not self.ok:
            return np.zeros(len(ids)), np.ones(len(ids), bool), np.full(len(ids), np.nan)
        m = self.p.reindex(ids)
        flux = m.is_flux.fillna(False).values.astype(bool)
        return np.where(flux, m.carbon_mgC_true.fillna(0).values, 0.0), flux, m.area_um2_true.values / C.UM_PER_PX ** 2

    def detections(self, oid, area_px):
        """Carbon of detections overlapping true particle oid most (0 = none), at the given areas (px)."""
        area_px = np.asarray(area_px, float)
        out = np.zeros(len(oid))
        a_um = area_px * C.UM_PER_PX ** 2
        has = oid > 0
        m = self.p.reindex(oid[has])
        flux = m.is_flux.fillna(False).values.astype(bool)
        b = np.array([self.laws.get(c, (0.0, 0.0))[1] for c in m.source_class.values])
        v = np.where(flux, m.carbon_mgC_true.values * (a_um[has] / m.area_um2_true.values) ** b, 0.0)
        out[has] = np.nan_to_num(v)
        if 'aggregate' in self.laws:                                    # a detection of nothing: aggregate law
            a0, b0 = self.laws['aggregate']
            out[~has] = np.exp(a0) * a_um[~has] ** b0
        return out


# detection
def detection_features(img, lab, om):
    """One row of measurements per candidate."""
    n = lab.max() + 1
    area = np.bincount(lab.ravel(), minlength=n).astype(float)
    pk = peaks(img, lab)
    mean = np.bincount(lab.ravel(), weights=img.ravel(), minlength=n) / np.maximum(area, 1)
    fg = (lab > 0).astype(np.uint8)
    rim = np.where((fg > 0) & (cv2.erode(fg, disk(1)) == 0), lab, 0)            # outer pixel layer
    ring = (cv2.dilate(fg, disk(3)) > 0) & (cv2.dilate(fg, disk(1)) == 0)        # 2-3 px outside
    ring = np.where(ring, cv2.dilate(lab.astype(np.float32), disk(3)).astype(np.int64), 0)
    sb = sobel(img)
    per = lambda w, l: np.bincount(l.ravel(), weights=w.ravel(), minlength=n) / np.maximum(np.bincount(l.ravel(), minlength=n), 1)
    rows = [dict(label=r.label, y0=r.bbox[0], x0=r.bbox[1], y1=r.bbox[2], x1=r.bbox[3], cy=r.centroid[0],
                 cx=r.centroid[1], area=np.log(area[r.label]), peak=pk[r.label] / om, mean=mean[r.label] / om,
                 mean_peak=mean[r.label] / max(pk[r.label], 1), solidity=r.solidity, ecc=r.eccentricity,
                 extent=r.extent, perim_ratio=r.perimeter ** 2 / (4 * np.pi * r.area)) for r in regionprops(lab)]
    d = pd.DataFrame(rows, columns=['label', 'y0', 'x0', 'y1', 'x1', 'cy', 'cx', 'area', 'peak', 'mean', 'mean_peak',
                                    'solidity', 'ecc', 'extent', 'perim_ratio'])
    if not len(d):
        return d
    k = d.label.values
    d['area_px'] = area[k].astype(int)
    d['rim_grey'] = per(img, rim)[k] / np.maximum(pk[k], 1)
    d['rim_sobel_mean'] = per(sb, rim)[k] / om
    d['rim_sobel_max'] = peaks(np.where(rim > 0, sb, 0).astype(np.float32), rim)[k] / om
    d['ring_mean'] = per(img, ring)[k] / np.maximum(pk[k], 1)
    d['ring_bright'] = per((img > 0.5 * om).astype(np.float32), ring)[k]
    c = d[['cy', 'cx']].values
    tree = cKDTree(c)
    d['nn_dist'] = tree.query(c, k=2)[0][:, 1] / C.K if len(c) > 1 else 99.0      # in reference px
    d['n_within_20'] = [len(x) - 1 for x in tree.query_ball_point(c, px()['crowd_r'])]
    return d


def _finish(lab, img, mask, om):
    """Fill holes per particle, keep candidates bright and big enough."""
    lab = watershed(np.zeros_like(img), lab, mask=ndi.binary_fill_holes(lab > 0))
    lab = np.where((peaks(img, lab) >= SEED * om)[lab], lab, 0)
    return drop_small(np.where(mask, lab, 0), px()['cand_min_area']).astype(np.int64)


def _pairs(a, b):
    """A unique label for every (a, b) pair with both > 0."""
    key = np.where((a > 0) & (b > 0), a.astype(np.int64) * (b.max() + 1) + b, 0)
    u, inv = np.unique(key, return_inverse=True)
    return np.where(key > 0, inv.reshape(key.shape), 0)


def outline_options(img, mask, om):
    """Blobs, and per outline option its label image and candidate table."""
    blobs = label((img > LOW * om) & mask, connectivity=2)
    pk = peaks(img, blobs)
    labs = {}
    for f in FRACS:
        labs[(f'{f}', f)] = _finish(label((blobs > 0) & (img >= f * pk[blobs]), connectivity=2), img, mask, om)
    if LOCAL_H:
        sm = ndi.gaussian_filter(img, px()['local_sigma'])
        cores = label(h_maxima(np.where(blobs > 0, sm / om, 0), LOCAL_H) & (blobs > 0), connectivity=2)
        regions = watershed(-sm, cores, mask=blobs > 0)
        rpk = peaks(img, regions)
        for f in LOCAL_FRACS:
            comp = label((regions > 0) & (img >= f * rpk[regions]), connectivity=2)
            labs[(f'{f}+local', f + 0.02)] = _finish(_pairs(regions, comp), img, mask, om)
    out = {}
    for (name, code), lab in labs.items():
        d = detection_features(img, lab, om)
        if len(d):
            owner = np.zeros(lab.max() + 1, np.int64)
            on = (lab > 0) & (blobs > 0)
            owner[lab[on]] = blobs[on]
            d['blob'], d['frac'], d['opt'] = owner[d.label.values], code, name
        out[name] = (lab, d)
    return blobs, out


def overlaps(out):
    """Pairs of candidates (option, label) from different options that share pixels."""
    names = [n for n in out if out[n][0].max() > 0]
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            la, lb = out[a][0], out[b][0]
            both = (la > 0) & (lb > 0)
            if both.any():
                k = np.unique(la[both].astype(np.int64) * (lb.max() + 1) + lb[both])
                rows.append(pd.DataFrame(dict(optA=a, labA=k // (lb.max() + 1), optB=b, labB=k % (lb.max() + 1))))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=['optA', 'labA', 'optB', 'labB'])


def select(d, ov, tau, tau_long=None):
    """Kept candidates (boolean array over d)."""
    long_ = elongation(d.ecc.values) > LONG_RATIO
    ok = d.p.values >= np.where(long_, tau if tau_long is None else tau_long, tau)
    if SELECT == 'blob':
        v = np.where(ok, d.p.values - LAM * (1 - d.p.values), 0.0)
        key = d.frame.astype(str) + ':' + d.blob.astype(str) if 'frame' in d else d.blob.astype(str)
        s = pd.DataFrame(dict(key=key.values, opt=d.opt.values, v=v, k=ok)).groupby(['key', 'opt']).agg(v=('v', 'sum'), k=('k', 'any'))
        s = s[s.k].reset_index()
        best = set(zip(*s.loc[s.groupby('key').v.idxmax(), ['key', 'opt']].values.T)) if len(s) else set()
        return ok & np.array([(a, b) in best for a, b in zip(key.values, d.opt.values)])
    fr = d.frame.values if 'frame' in d else np.zeros(len(d), int)
    idx = {k: i for i, k in enumerate(zip(fr, d.opt.values, d.label.values))}
    nb = [[] for _ in range(len(d))]
    ofr = ov.frame.values if 'frame' in ov else np.zeros(len(ov), int)
    for f_, a, la, b, lb in zip(ofr, ov.optA.values, ov.labA.values, ov.optB.values, ov.labB.values):
        i, j = idx.get((f_, a, la)), idx.get((f_, b, lb))
        if i is not None and j is not None:
            nb[i].append(j)
            nb[j].append(i)
    score = d.p.values * volume(d.area_px.values)          # best first: likely correct AND big
    kept, blocked = np.zeros(len(d), bool), np.zeros(len(d), bool)
    for i in np.argsort(-score):
        if ok[i] and not blocked[i]:
            kept[i] = True
            for j in nb[i]:
                blocked[j] = True
    return kept


def detect_taus(img, mask, om, filt, taus):
    """hybrid_segment at several cut-offs tau on one frame."""
    _, out = outline_options(img, mask, om)
    ds = [d for _, d in out.values() if len(d)]
    res = {}
    if not ds:
        return {t: (np.zeros(img.shape, np.int64), pd.DataFrame()) for t in taus}
    d = pd.concat(ds, ignore_index=True)
    d['p'] = filt['model'].predict_proba(d[FEATURES])[:, 1]
    ov = overlaps(out)
    for t in taus:
        keep = select(d, ov, t, None)
        final, nxt, kept = np.zeros(img.shape, np.int64), 0, []
        for name, (lab, _) in out.items():
            sel = d[(d.opt == name) & keep]
            m = np.zeros(lab.max() + 1, bool)
            m[sel.label.values] = True
            final[m[lab]] = lab[m[lab]] + nxt
            kept.append(sel.assign(label=sel.label + nxt))
            nxt += lab.max() + 1
        res[t] = (final, pd.concat(kept, ignore_index=True))
    return res


def detect(img, mask, om, filt):
    """hybrid_segment on one frame: label image and one row per kept detection."""
    _, out = outline_options(img, mask, om)
    ds = [d for _, d in out.values() if len(d)]
    final = np.zeros(img.shape, np.int64)
    if not ds:
        return final, pd.DataFrame()
    d = pd.concat(ds, ignore_index=True)
    d['p'] = filt['model'].predict_proba(d[FEATURES])[:, 1]
    keep = select(d, overlaps(out), filt['tau'], filt.get('tau_long'))
    nxt, kept = 0, []
    for name, (lab, _) in out.items():
        sel = d[(d.opt == name) & keep]
        m = np.zeros(lab.max() + 1, bool)
        m[sel.label.values] = True
        final[m[lab]] = lab[m[lab]] + nxt
        kept.append(sel.assign(label=sel.label + nxt))
        nxt += lab.max() + 1
    det = pd.concat(kept, ignore_index=True)
    final, fw, _ = relabel_sequential(final)                       # labels 1..n
    det['label'] = fw[det.label.to_numpy().astype(np.int64).copy()]
    det['area_corrected_px'] = corrected(det.area_px, filt.get('scale'))
    return final, det.sort_values('label').reset_index(drop=True)


# matching to truth
def match(lab, t):
    """Match detections to true particles (IoU >= IOU_MIN): correct, ignored, matched and found."""
    p, tt = lab.ravel(), t.ravel()
    n_p, n_t = p.max() + 1, tt.max() + 1
    pa, ta = np.bincount(p, minlength=n_p), np.bincount(tt, minlength=n_t)
    both = (p > 0) & (tt > 0)
    key, ov = np.unique(p[both] * n_t + tt[both], return_counts=True)
    pi, ti = key // n_t, key % n_t
    iou = ov / (pa[pi] + ta[ti] - ov)
    big = ta >= C.MIN_AREA
    big[0] = False
    hit = iou >= IOU_MIN
    correct = np.zeros(n_p, bool)
    correct[pi[hit & big[ti]]] = True
    ignored = np.zeros(n_p, bool)
    ignored[pi[hit & ~big[ti]]] = True
    ignored &= ~correct
    tid = np.zeros(n_p, np.int64)
    m = hit & big[ti]
    tid[pi[m]] = ti[m]
    oid = np.zeros(n_p, np.int64)
    o = np.argsort(ov)
    oid[pi[o]] = ti[o]
    n_held = np.bincount(pi[(ov >= 0.5 * ta[ti]) & big[ti]], minlength=n_p)
    frag = (ov >= 0.5 * pa[pi]) & big[ti] & ~hit
    n_frag = np.bincount(ti[frag], minlength=n_t)
    found = np.zeros(n_t, bool)
    found[ti[m]] = True
    covered = np.bincount(ti, weights=ov, minlength=n_t) / np.maximum(ta, 1)
    return dict(pa=pa, ta=ta, big=big, correct=correct, ignored=ignored, tid=tid, oid=oid, n_held=n_held,
                n_frag=n_frag, found=found, covered=covered)


# training the filter
def _candidate_table(seq_dirs, frame_step):
    parts, truth, ovs = [], [], []
    for d in seq_dirs:
        files = frames_of(d)
        mask = load_mask(d, cv2.imread(files[0], 0).shape)
        om = sequence_otsu(d, mask)
        cb = Carbon(d)
        for f in files[::frame_step]:
            fi = frame_num(f)
            fid = f'{d}#{fi}'
            img = cv2.imread(f, 0).astype(np.float32)
            t = load_truth(d, fi, mask)
            ta = np.bincount(t.ravel())
            big = ta >= C.MIN_AREA
            big[0] = False
            ids = np.nonzero(big)[0]
            tc, flux, phys = cb.truth(ids)
            truth.append(pd.DataFrame(dict(frame=fid, seq=d, tid=ids, tarea=ta[ids], carbon=tc, flux=flux, phys=phys)))
            _, out = outline_options(img, mask, om)
            for name, (lab, x) in out.items():
                if not len(x):
                    continue
                mt = match(lab, t)
                k = x.label.values
                parts.append(x.assign(frame=fid, seq=d, correct=mt['correct'][k], ignored=mt['ignored'][k],
                                      tid=mt['tid'][k], oid=mt['oid'][k], group=fid if len(seq_dirs) == 1 else d))
            ovs.append(overlaps(out).assign(frame=fid))
        print('training data from', os.path.basename(d), flush=True)
    return pd.concat(parts, ignore_index=True), pd.concat(truth, ignore_index=True), pd.concat(ovs, ignore_index=True)


def _carbon_of(c, seqs, area):
    out = np.zeros(len(c))
    for s in np.unique(c.seq.values):
        sel = c.seq.values == s
        out[sel] = seqs[s].detections(c.oid.values[sel], area[sel]) if seqs[s].ok else volume(area[sel])
    return out


def evaluate_table(c, truth, kept, seqs, scale=None):
    """Pooled measures of one selection on the candidate table (one row of the tau curve)."""
    cor = kept & c.correct.values
    ok = kept & ~c.ignored.values
    a = corrected(c.area_px.values, scale)
    if all(seqs[s].ok for s in seqs):
        tc, flux, cd = truth.carbon.values, truth.flux.values, _carbon_of(c, seqs, a)
    else:
        tc, flux, cd = volume(truth.tarea.values), np.ones(len(truth), bool), volume(a)
    found = set(zip(c.frame.values[cor], c.tid.values[cor]))
    isf = np.array([x in found for x in zip(truth.frame.values, truth.tid.values)])
    v = volume(c.area_px.values)
    return dict(tau_detections=int(ok.sum()), precision=100 * cor.sum() / max(ok.sum(), 1),
                recall=100 * isf.mean(), flux_recall=100 * isf[flux].mean() if flux.any() else np.nan,
                carbon_found=100 * tc[isf].sum() / max(tc.sum(), 1e-30),
                volume_precision=100 * v[cor].sum() / max(v[ok].sum(), 1e-30),
                total_carbon=100 * cd[kept].sum() / max(tc.sum(), 1e-30))


def _scales(c, truth, kept, use_phys):
    """Size correction per size class."""
    cor = kept & c.correct.values
    ref = truth.set_index(['frame', 'tid']).reindex(list(zip(c.frame.values[cor], c.tid.values[cor])))
    target = ref.phys.values if use_phys else ref.tarea.values
    good = np.isfinite(target) & (ref.flux.values.astype(bool) if use_phys else True)
    a, target = c.area_px.values[cor][good], target[good]
    bins = px()['calib_bins']
    cls = np.searchsorted(bins, a, side='right') - 1
    return {lo: (round(float(target[cls == i].sum() / a[cls == i].sum()), 3) if (cls == i).sum() >= 20 else 1.0)
            for i, lo in enumerate(bins)}


def choose_tau(curve, rule=None):
    """tau from an out-of-fold curve."""
    rule = rule or TAU_RULE
    if rule == 'total' and curve.total_carbon.notna().any() and curve.total_carbon.max() > 0:
        return float(curve.tau[(curve.total_carbon - 100).abs().idxmin()])
    hit = curve[curve.volume_precision >= MIN_CARBON_PRECISION]
    return float(hit.tau.min()) if len(hit) else float(curve.loc[curve.volume_precision.idxmax(), 'tau'])


def train_filter(seq_dirs, frame_step=2, log=print):
    """Train the quality filter on training sequences and choose tau and the size correction out of fold."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    c, truth, ov = _candidate_table(seq_dirs, frame_step)
    seqs = {d: Carbon(d) for d in seq_dirs}
    make = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                  l2_regularization=1.0, random_state=0)
    use = ~c.ignored
    c['p'] = 0.0
    for a, b in GroupKFold(min(5, c.group.nunique())).split(c, groups=c.group):
        tr = c.iloc[a][use.iloc[a].values]
        c.loc[c.index[b], 'p'] = make().fit(tr[FEATURES], tr.correct).predict_proba(c.iloc[b][FEATURES])[:, 1]
    curve = pd.DataFrame([dict(tau=t, **evaluate_table(c, truth, select(c, ov, t), seqs)) for t in TAUS])
    curve = curve.round({k: 1 for k in curve.columns if k != 'tau'})
    tau = choose_tau(curve)
    kept = select(c, ov, tau)
    scale = _scales(c, truth, kept, all(s.ok for s in seqs.values())) if CALIBRATE_SIZE else None
    log(f'out-of-fold choice: tau {tau}, size correction {scale}')
    model = make().fit(c.loc[use, FEATURES], c.correct[use])
    return dict(model=model, tau=tau, tau_long=None, scale=scale, curve=curve, resolution=C.summary())


def retune(filt, tau=None, tau_long=None, rule=None):
    """Set tau (a number) or re-choose it from the stored out-of-fold curve (tau=None), without retraining."""
    filt = dict(filt)
    if tau is None and 'curve' in filt:
        tau = choose_tau(filt['curve'], rule)
    if tau is not None:
        filt['tau'] = float(tau)
    tl = filt.get('tau_long') if tau_long is None else tau_long
    filt['tau_long'] = None if tl is None else float(min(tl, filt['tau']))
    return filt


# scoring
def score_frame(lab, t, img, om, cb, scale=None):
    """Counts and carbon sums of one frame against its truth label map."""
    mt = match(lab, t)
    pa, ta, big = mt['pa'], mt['ta'], mt['big']
    isdet = pa > 0
    isdet[0] = False
    ids = np.nonzero(big)[0]
    tc, flux, phys = cb.truth(ids)
    if not cb.ok:
        tc = volume(ta[ids])
    found = mt['found'][ids]
    dl = np.nonzero(isdet)[0]
    a_raw = pa[dl].astype(float)
    a = corrected(a_raw, scale)
    cd = cb.detections(mt['oid'][dl], a) if cb.ok else volume(a)
    cor = mt['correct'][dl]
    tmatch = mt['tid'][dl][cor]
    pos = {u: i for i, u in enumerate(ids)}
    ti = np.array([pos[u] for u in tmatch], int)
    fl_m = flux[ti] if len(ti) else np.zeros(0, bool)
    grey = np.bincount(t.ravel(), weights=img.ravel(), minlength=len(ta)) / np.maximum(ta, 1)
    faint = grey[ids] < om
    long_ = np.zeros(len(ids), bool)
    if cb.ok:
        long_ = np.nan_to_num(cb.p.reindex(ids).aspect_true.values) > LONG_RATIO
    c = dict(n_true=int(big.sum()), n_found=int(found.sum()), n_pred=int((isdet & ~mt['ignored']).sum()),
             n_ignored=int(mt['ignored'][1:].sum()), n_correct=int(mt['correct'][1:].sum()),
             n_flux=int(flux.sum()), n_flux_found=int((flux & found).sum()),
             n_long=int((flux & long_).sum()), n_long_found=int((flux & long_ & found).sum()),
             carbon_true=float(tc.sum()), carbon_found=float(tc[found].sum()), carbon_est=float(cd.sum()),
             carbon_est_matched=float(cd[cor][fl_m].sum()) if len(ti) else 0.0,
             carbon_true_matched=float(tc[ti][fl_m].sum()) if len(ti) else 0.0,
             carbon_false=float(cd[mt['oid'][dl] == 0].sum()),
             area_matched=float(a_raw[cor].sum()), area_true_matched=float(ta[tmatch].sum()),
             area_corr_matched_flux=float(a[cor][fl_m].sum()) if len(ti) else 0.0,
             phys_matched_flux=float(np.nansum(phys[ti][fl_m])) if (len(ti) and cb.ok) else 0.0,
             n_size_ok=int((np.abs(a_raw[cor] / np.maximum(ta[tmatch], 1) - 1) <= 0.2).sum()),
             n_merged=int(mt['n_held'][mt['n_held'] >= 2].sum()), n_split=int((mt['n_frag'][ids] >= 2).sum()),
             n_unseen=int((~found & (mt['covered'][ids] < 0.2)).sum()),
             n_faint=int(faint.sum()), n_faint_found=int((faint & found).sum()))
    for k, (lo, hi) in size_bins().items():
        b = (ta[ids] >= lo) & (ta[ids] < hi)
        c[f'n_true {k}'], c[f'n_found {k}'] = int(b.sum()), int((b & found).sum())
    return c


def percentages(c):
    """Summed counts (dict or row) -> the percentage columns reported for segmentation."""
    pct = lambda a, b: round(100 * a / b, 1) if b else np.nan
    r, p = pct(c['n_found'], c['n_true']), pct(c['n_correct'], c['n_pred'])
    out = {'detections': int(c['n_pred']), 'true particles': int(c['n_true']), 'flux particles': int(c['n_flux']),
           'Total carbon, % of true': pct(c['carbon_est'], c['carbon_true']),
           'Carbon found (carbon-weighted recall) %': pct(c['carbon_found'], c['carbon_true']),
           'Matched particle carbon, % of true': pct(c['carbon_est_matched'], c['carbon_true_matched']),
           'False carbon (no particle under it), % of true': pct(c['carbon_false'], c['carbon_true']),
           'Flux particles found %': pct(c['n_flux_found'], c['n_flux']),
           'Particles found (recall) %': r,
           'Long flux particles found %': pct(c['n_long_found'], c['n_long']),
           'Matched size vs physical area, %': pct(c['area_corr_matched_flux'], c['phys_matched_flux']),
           'Matched size vs labelled area (uncorrected), %': pct(c['area_matched'], c['area_true_matched']),
           'Matched particles within 20% of labelled size %': pct(c['n_size_ok'], c['n_found']),
           'Detections correct (precision) %': p,
           'Particle F1 %': round(2 * r * p / (r + p), 1) if r and p else np.nan,
           'Merged with a neighbour %': pct(c['n_merged'], c['n_true']),
           'Split into pieces %': pct(c['n_split'], c['n_true']),
           'Ignored detections (match a particle < MIN_AREA px)': int(c['n_ignored'])}
    for k in [x for x in c.keys() if str(x).startswith('n_true ')]:
        b = k[len('n_true '):]
        out[f'Found: {b} %'] = pct(c[f'n_found {b}'], c[k])
    out['Faint particles found %'] = pct(c['n_faint_found'], c['n_faint'])
    out['Missed: not detected at all %'] = pct(c['n_unseen'], c['n_true'])
    return out


# running on sequences
def seg_dir(out_dir, sid):
    return os.path.join(out_dir, 'seg', sid)


def segment_sequence(seq_dir, out_dir, filt, frames='all'):
    """Detect particles in the chosen frames of one sequence ('all', 'every3', or a list of frame numbers)."""
    sid = os.path.basename(seq_dir.rstrip('/'))
    od = seg_dir(out_dir, sid)
    if os.path.exists(os.path.join(od, 'detections.csv')):
        return pd.read_csv(os.path.join(od, 'detections.csv'))
    os.makedirs(os.path.join(od, 'labels'), exist_ok=True)
    files = frames_of(seq_dir)
    if frames == 'all':
        pick = files
    elif isinstance(frames, str) and frames.startswith('every'):
        pick = files[::int(frames[5:])]
    else:
        pick = [f for f in files if frame_num(f) in set(frames)]
    mask = load_mask(seq_dir, cv2.imread(files[0], 0).shape)
    om = sequence_otsu(seq_dir, mask)
    cb = Carbon(seq_dir)
    dets, rows = [], []
    for f in pick:
        fi = frame_num(f)
        img = cv2.imread(f, 0).astype(np.float32)
        lab, det = detect(img, mask, om, filt)
        cv2.imwrite(os.path.join(od, 'labels', f'frame_{fi:04d}.png'), lab.astype(np.uint16))
        t = load_truth(seq_dir, fi, mask)
        if t is not None:
            rows.append(dict(seq_id=sid, frame=fi, **score_frame(lab, t, img, om, cb, filt.get('scale'))))
            if len(det):
                mt = match(lab, t)
                det['correct'], det['ignored'] = mt['correct'][det.label.values], mt['ignored'][det.label.values]
        if len(det):
            cols = ['label', 'y0', 'x0', 'y1', 'x1', 'cy', 'cx', 'area_px', 'area_corrected_px', 'opt', 'p'] + \
                   (['correct', 'ignored'] if 'correct' in det else [])
            dets.append(det[cols].assign(seq_id=sid, frame=fi))
    pd.DataFrame(rows).to_csv(os.path.join(od, 'scores.csv'), index=False)
    D = pd.concat(dets, ignore_index=True) if dets else pd.DataFrame(columns=['seq_id', 'frame', 'label'])
    D['otsu'] = om
    D.to_csv(os.path.join(od, 'detections.csv'), index=False)
    print(f'{sid}: {len(pick)} frames, {len(D)} detections', flush=True)
    return D


def load_labels(out_dir, sid, fi):
    f = os.path.join(seg_dir(out_dir, sid), 'labels', f'frame_{fi:04d}.png')
    return cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(np.int64) if os.path.exists(f) else None


def summarise(out_dir, seq_ids=None):
    """Per-frame and per-sequence / per-split / overall percentage tables from every scores.csv."""
    files = sorted(glob.glob(os.path.join(out_dir, 'seg', '*', 'scores.csv')))
    parts = [pd.read_csv(f) for f in files if os.path.getsize(f) > 2]
    counts = pd.concat([p for p in parts if len(p)], ignore_index=True)
    if seq_ids is not None:
        counts = counts[counts.seq_id.isin(seq_ids)]
    per_frame = pd.DataFrame([dict(seq_id=r.seq_id, frame=r.frame, **percentages(r)) for _, r in counts.iterrows()])
    per_seq = pd.DataFrame([dict(seq_id=s, **percentages(g.sum(numeric_only=True))) for s, g in counts.groupby('seq_id')])
    split = counts.seq_id.str.split('_').str[-1]
    per_split = pd.DataFrame([dict(seq_id=f'all {s}', **percentages(g.sum(numeric_only=True))) for s, g in counts.groupby(split)])
    return per_frame, pd.concat([per_seq, per_split], ignore_index=True)


def draw_boxes(img, dets, out_png=None, gain=1.0):
    """The bgrem frame with a yellow box around every detection."""
    vis = cv2.cvtColor(np.clip(img * gain, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    H, W = img.shape
    pad, lw = px()['box_pad'], max(2, int(round(2 * C.K)))
    for r in dets.itertuples():
        cv2.rectangle(vis, (max(int(r.x0) - pad, 0), max(int(r.y0) - pad, 0)),
                      (min(int(r.x1) - 1 + pad, W - 1), min(int(r.y1) - 1 + pad, H - 1)), YELLOW, lw)
    if out_png:
        cv2.imwrite(out_png, vis)
    return vis
