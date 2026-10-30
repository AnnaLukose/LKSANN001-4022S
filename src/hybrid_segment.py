# hybrid_segment for the comparison notebooks: candidate outlines per blob kept or dropped by a quality filter.
# Output: hybrid_segment_outputs/quality_filter.joblib and cached detections.
# Uses: scale.py. Used by: piv_compare.py.
import argparse, glob, json, os, time
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu, sobel
from skimage.measure import label, regionprops
from skimage.segmentation import watershed
import scale as SC

MIN_AREA = 50                       # px: smallest particle kept and counted in the truth (set by scale)
IOU_MIN = 0.5                       # a detection matches a true particle when IoU >= this
LOW, SEED = 0.5, 0.8                # blob floor and minimum peak, x the sequence's Otsu level
FRACS = (0.35, 0.5, 0.65)           # outline options: fractions of each blob's peak
LAM = 1.0                           # outline choice: expected correct - LAM x expected wrong
FEATURES = ['area', 'peak', 'mean', 'mean_peak', 'rim_grey', 'rim_sobel_mean', 'rim_sobel_max', 'ring_mean',
            'ring_bright', 'solidity', 'ecc', 'extent', 'perim_ratio', 'nn_dist', 'n_within_20', 'frac']
YELLOW = (0, 255, 255)              # BGR
BOX_PAD, BOX_WIDTH = 2, 2           # px: space between a particle and its box, line width
RING_IN, RING_OUT = 1, 3            # px: the surrounding ring runs from just outside RING_IN to RING_OUT
CROWD_R = 20                        # px: radius for the crowding count n_within_20


def _apply_scale():
    """Pixel settings at the current resolution (scale)."""
    global MIN_AREA, BOX_PAD, BOX_WIDTH, RING_IN, RING_OUT, CROWD_R
    MIN_AREA = SC.MIN_AREA
    BOX_PAD, BOX_WIDTH = SC.ilength(2), SC.ilength(2)
    RING_IN, RING_OUT = SC.ilength(1), SC.ilength(3)
    CROWD_R = SC.length(20)


SC.register(_apply_scale)


# helpers
def disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def peaks(img, lab):
    """Brightest pixel of every label (index = label)."""
    out = np.zeros(lab.max() + 1, np.float32)
    on = lab > 0
    np.maximum.at(out, lab[on], img[on])
    return out


def big_only(lab, min_area):
    area = np.bincount(lab.ravel())
    small = area < min_area
    small[0] = True
    lab = lab.copy()
    lab[small[lab]] = 0
    return lab


def frames_of(seq_dir):
    return sorted(glob.glob(os.path.join(seq_dir, 'bgrem', 'bgrem_*.png')))


def load_mask(seq_dir, shape):
    f = os.path.join(seq_dir, 'truth', 'static_mask.png')
    return cv2.imread(f, 0) > 0 if os.path.exists(f) else np.ones(shape, bool)


def sequence_otsu(seq_dir, mask):
    return float(np.mean([threshold_otsu(cv2.imread(f, 0)[mask]) for f in frames_of(seq_dir)]))


def load_truth(seq_dir, fi, mask):
    f = os.path.join(seq_dir, 'truth', 'labels', f'frame_{fi:04d}.png')
    return np.where(mask, cv2.imread(f, cv2.IMREAD_UNCHANGED), 0).astype(np.int64) if os.path.exists(f) else None


def pick_frames(n):
    """An early frame (not the first), the middle frame and the last frame (indices into the sorted frames)."""
    return sorted({max(1, round(0.1 * (n - 1))), (n - 1) // 2, n - 1})


# detection
def detection_features(img, lab, om):
    """One row of measurements per detection."""
    n = lab.max() + 1
    area = np.bincount(lab.ravel(), minlength=n).astype(float)
    pk = peaks(img, lab)
    mean = np.bincount(lab.ravel(), weights=img.ravel(), minlength=n) / np.maximum(area, 1)
    fg = (lab > 0).astype(np.uint8)
    rim = np.where((fg > 0) & (cv2.erode(fg, disk(1)) == 0), lab, 0)            # outer pixel layer
    ring = (cv2.dilate(fg, disk(RING_OUT)) > 0) & (cv2.dilate(fg, disk(RING_IN)) == 0)   # just outside the outline
    ring = np.where(ring, cv2.dilate(lab.astype(np.float32), disk(RING_OUT)).astype(np.int64), 0)
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
    d['rim_grey'] = per(img, rim)[k] / np.maximum(pk[k], 1)
    d['rim_sobel_mean'] = per(sb, rim)[k] / om
    d['rim_sobel_max'] = peaks(np.where(rim > 0, sb, 0).astype(np.float32), rim)[k] / om
    d['ring_mean'] = per(img, ring)[k] / np.maximum(pk[k], 1)
    d['ring_bright'] = per((img > 0.5 * om).astype(np.float32), ring)[k]
    c = d[['cy', 'cx']].values
    tree = cKDTree(c)
    d['nn_dist'] = tree.query(c, k=2)[0][:, 1] if len(c) > 1 else 99.0
    d['n_within_20'] = [len(x) - 1 for x in tree.query_ball_point(c, CROWD_R)]
    return d


def outline_options(img, mask, om):
    """Low-floor blobs and, per outline fraction, the detections (label image) and their measurements."""
    blobs = label((img > LOW * om) & mask, connectivity=2)
    pk = peaks(img, blobs)
    out = {}
    for f in FRACS:
        lab = label((blobs > 0) & (img >= f * pk[blobs]), connectivity=2)
        lab = watershed(np.zeros_like(img), lab, mask=ndi.binary_fill_holes(lab > 0))   # fill holes per particle
        lab = np.where((peaks(img, lab) >= SEED * om)[lab], lab, 0)
        lab = big_only(np.where(mask, lab, 0), MIN_AREA).astype(np.int64)
        d = detection_features(img, lab, om)
        if len(d):
            owner = np.zeros(lab.max() + 1, np.int64)
            on = (lab > 0) & (blobs > 0)
            owner[lab[on]] = blobs[on]
            d['blob'], d['frac'] = owner[d.label.values], f
        out[f] = (lab, d)
    return blobs, out


def choose_outlines(d, tau, lam=LAM):
    """Per blob keep the outline option with the best sum(p - lam * (1 - p)) over its detections scoring >= tau."""
    kept = d.p >= tau
    val = (d.p - lam * (1 - d.p)).where(kept, 0).groupby([d.blob, d.frac]).sum()
    val = val.where(kept.groupby([d.blob, d.frac]).any(), 0)
    best = val.groupby(level=0).idxmax().map(lambda x: x[1])
    return kept & (d.frac.values == best.reindex(d.blob).values)


def detect(img, mask, om, model, tau):
    """hybrid_segment on one frame -> (label image, one row per kept detection with its box and score p)."""
    _, out = outline_options(img, mask, om)
    ds = [d for _, d in out.values() if len(d)]
    final = np.zeros(img.shape, np.int64)
    if not ds:
        return final, pd.DataFrame()
    d = pd.concat(ds, ignore_index=True)
    d['p'] = model.predict_proba(d[FEATURES])[:, 1]
    d['keep'] = choose_outlines(d, tau).values
    nxt, kept = 0, []
    for f, (lab, _) in out.items():
        sel = d[(d.frac == f) & d.keep]
        keep = np.zeros(lab.max() + 1, bool)
        keep[sel.label.values] = True
        final[keep[lab]] = lab[keep[lab]] + nxt
        kept.append(sel.assign(label=sel.label + nxt))
        nxt += lab.max() + 1
    return final, pd.concat(kept, ignore_index=True)


# training
def training_rows(d, frame_step, single):
    """Every candidate outline of every frame_step-th frame of one sequence, with whether it is correct."""
    files = frames_of(d)
    mask = load_mask(d, cv2.imread(files[0], 0).shape)
    om = sequence_otsu(d, mask)
    parts = []
    for f in files[::frame_step]:
        fi = int(os.path.basename(f)[6:10])
        img = cv2.imread(f, 0).astype(np.float32)
        t = load_truth(d, fi, mask)
        ta = np.bincount(t.ravel())
        _, out = outline_options(img, mask, om)
        for lab, feats in out.values():
            if len(feats):
                parts.append(feats.assign(correct=best_iou(lab, t, ta)[feats.label.values] >= IOU_MIN,
                                          frame=f'{d}#{fi}', group=f'{d}#{fi}' if single else d))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def train_filter(seq_dirs, frame_step=2, target=90.0, cache_dir=None):
    """Train the quality filter on seq_dirs (sequences with truth)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    parts = []
    for d in seq_dirs:
        t0 = time.time()
        f = os.path.join(cache_dir, os.path.basename(d) + '.pkl') if cache_dir else None
        if f and os.path.exists(f):
            parts.append(pd.read_pickle(f))
            continue
        rows = training_rows(d, frame_step, len(seq_dirs) == 1)
        if f:
            os.makedirs(cache_dir, exist_ok=True)
            rows.to_pickle(f)
        parts.append(rows)
        print(f'training data from {os.path.basename(d)}: {len(rows)} candidate outlines ({time.time() - t0:.0f} s)',
              flush=True)
    data = pd.concat(parts, ignore_index=True)
    data['blob'] = data.frame + ':' + data.blob.astype(str)
    make = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                  l2_regularization=1.0, random_state=0)
    data['p'] = 0.0
    for a, b in GroupKFold(min(5, data.group.nunique())).split(data, groups=data.group):
        m = make().fit(data.iloc[a][FEATURES], data.iloc[a].correct)
        data.loc[data.index[b], 'p'] = m.predict_proba(data.iloc[b][FEATURES])[:, 1]
    curve = pd.DataFrame([dict(tau=t, detections=int(k.sum()), precision=round(100 * data.correct[k].mean(), 1))
                          for t in np.arange(0.3, 0.96, 0.05).round(2) for k in [choose_outlines(data, t)]])
    hit = curve[curve.precision >= target]
    tau = float(hit.tau.min()) if len(hit) else float(curve.loc[curve.precision.idxmax(), 'tau'])
    return make().fit(data[FEATURES], data.correct), tau, curve


def ensure_filter(data_dir, filter_file, frame_step=2, target=90.0):
    """The quality filter for this pixel size."""
    import joblib
    if os.path.exists(filter_file):
        model, tau = joblib.load(filter_file)
        info = os.path.join(os.path.dirname(filter_file), 'filter_settings.json')
        if os.path.exists(info):
            saved = json.load(open(info)).get('um_per_px')
            if saved is not None and abs(float(saved) - SC.UM_PER_PX) > 1e-6:
                raise ValueError(f'{filter_file} was trained at {saved} um/px, the data are {SC.UM_PER_PX} um/px')
        print(f'quality filter found: {filter_file} (saved tau {tau:.2f})')
        return model, tau
    out = os.path.dirname(filter_file)
    os.makedirs(out, exist_ok=True)
    seqs = sorted(glob.glob(os.path.join(data_dir, 'seq*_train')))
    if not seqs:
        raise FileNotFoundError(f'no training sequences (seq*_train) in {data_dir}')
    print(f'training the hybrid_segment quality filter at {SC.UM_PER_PX} um/px on {len(seqs)} training sequences, '
          f'one frame in {frame_step}', flush=True)
    t0 = time.time()
    model, tau, curve = train_filter(seqs, frame_step, target, cache_dir=os.path.join(out, 'train_data'))
    joblib.dump((model, tau), filter_file)
    curve.to_csv(os.path.join(out, 'precision_curve.csv'), index=False)
    json.dump(dict(SC.summary(), data=os.path.basename(data_dir.rstrip('/')), n_train_sequences=len(seqs),
                   frame_step=frame_step, target_precision=target, saved_tau=tau,
                   minutes=round((time.time() - t0) / 60, 1)), open(os.path.join(out, 'filter_settings.json'), 'w'),
              indent=1)
    print(f'saved {filter_file} (tau for {target:.0f} % precision: {tau:.2f}; {(time.time() - t0) / 60:.0f} min)')
    return model, tau


# scoring and drawing
def best_iou(lab, t, ta=None):
    """Best IoU of every detection label with any true particle."""
    pa = np.bincount(lab.ravel())
    ta = np.bincount(t.ravel()) if ta is None else ta
    both = (lab > 0) & (t > 0)
    key, ov = np.unique(lab[both] * (t.max() + 1) + t[both], return_counts=True)
    pi, ti = key // (t.max() + 1), key % (t.max() + 1)
    best = np.zeros(len(pa))
    np.maximum.at(best, pi, ov / (pa[pi] + ta[ti] - ov))
    return best


def score_frame(lab, t, img, om):
    """Counts for one frame (same definitions as seg_compare.py)."""
    p, tt = lab.ravel(), t.ravel()
    n_p, n_t = p.max() + 1, tt.max() + 1
    pa, ta = np.bincount(p, minlength=n_p), np.bincount(tt, minlength=n_t)
    both = (p > 0) & (tt > 0)
    key, ov = np.unique(p[both] * n_t + tt[both], return_counts=True)
    pi, ti = key // n_t, key % n_t
    iou = ov / (pa[pi] + ta[ti] - ov)
    big = ta >= MIN_AREA
    big[0] = False
    m = (iou >= IOU_MIN) & big[ti]
    found = np.zeros(n_t, bool)
    found[ti[m]] = True
    correct = np.zeros(n_p, bool)
    correct[pi[iou >= IOU_MIN]] = True
    covered = np.bincount(ti, weights=ov, minlength=n_t) / np.maximum(ta, 1)
    grey = np.bincount(tt, weights=img.ravel(), minlength=n_t) / np.maximum(ta, 1)
    faint = big & (grey < om)
    return dict(n_true=int(big.sum()), n_found=int(found[big].sum()), n_pred=int((pa[1:] > 0).sum()),
                n_correct=int(correct[1:].sum()), area_pred_matched=int(pa[pi[m]].sum()),
                area_true_matched=int(ta[ti[m]].sum()), n_unseen=int((big & ~found & (covered < 0.2)).sum()),
                n_faint=int(faint.sum()), n_faint_found=int((faint & found).sum()))


def percentages(c):
    """Turn summed counts (dict or DataFrame row) into the percentage columns."""
    pct = lambda a, b: round(100 * a / b, 1) if b else np.nan
    r, p = pct(c['n_found'], c['n_true']), pct(c['n_correct'], c['n_pred'])
    return {'detections': int(c['n_pred']), 'true particles': int(c['n_true']),
            'Particles found (recall) %': r, 'Detections correct (precision) %': p,
            'Particle F1 %': round(2 * r * p / (r + p), 1) if r and p else np.nan,
            'Matched particle size, % of true': pct(c['area_pred_matched'], c['area_true_matched']),
            'Faint particles found %': pct(c['n_faint_found'], c['n_faint']),
            'Missed: not detected at all %': pct(c['n_unseen'], c['n_true'])}


def draw_boxes(img, dets, out_png):
    """The bgrem frame with a yellow box around every detection, and nothing else."""
    vis = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    H, W = img.shape
    for r in dets.itertuples():                       # bbox: y0, x0 inclusive; y1, x1 exclusive
        cv2.rectangle(vis, (max(int(r.x0) - BOX_PAD, 0), max(int(r.y0) - BOX_PAD, 0)),
                      (min(int(r.x1) - 1 + BOX_PAD, W - 1), min(int(r.y1) - 1 + BOX_PAD, H - 1)), YELLOW, BOX_WIDTH)
    cv2.imwrite(out_png, vis)


def run(seq_dirs, out_dir, model, tau, frames=None):
    """hybrid_segment on 3 frames of every sequence -> boxes/<seq>_frame_XXXX.png."""
    os.makedirs(os.path.join(out_dir, 'boxes'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'per_seq'), exist_ok=True)
    for d in seq_dirs:
        sid = os.path.basename(d)
        done = os.path.join(out_dir, 'per_seq', f'{sid}.csv')
        if os.path.exists(done):
            continue
        files = frames_of(d)
        mask = load_mask(d, cv2.imread(files[0], 0).shape)
        om = sequence_otsu(d, mask)
        rows, dets = [], []
        for i in (frames or pick_frames(len(files))):
            fi = int(os.path.basename(files[i])[6:10])
            img = cv2.imread(files[i], 0).astype(np.float32)
            lab, det = detect(img, mask, om, model, tau)
            draw_boxes(img, det, os.path.join(out_dir, 'boxes', f'{sid}_frame_{fi:04d}.png'))
            t = load_truth(d, fi, mask)
            if t is not None:
                rows.append(dict(seq_id=sid, frame=fi, **score_frame(lab, t, img, om)))
                if len(det):
                    det['correct'] = best_iou(lab, t)[det.label.values] >= IOU_MIN
            if len(det):
                cols = ['label', 'y0', 'x0', 'y1', 'x1', 'p'] + (['correct'] if 'correct' in det else [])
                dets.append(det[cols].assign(seq_id=sid, frame=fi))
        if dets:
            pd.concat(dets).to_csv(os.path.join(out_dir, 'per_seq', f'{sid}_detections.csv'), index=False)
        pd.DataFrame(rows).to_csv(done, index=False)
        print(sid, 'done', flush=True)
    return summarise(out_dir)


def summarise(out_dir):
    """Per-frame table and a per-sequence / per-split / overall table, as percentages."""
    files = [f for f in sorted(glob.glob(os.path.join(out_dir, 'per_seq', '*.csv'))) if not f.endswith('_detections.csv')]
    counts = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    per_frame = pd.DataFrame([dict(seq_id=r.seq_id, frame=r.frame, **percentages(r)) for _, r in counts.iterrows()])
    per_seq = pd.DataFrame([dict(seq_id=s, **percentages(g.sum(numeric_only=True))) for s, g in counts.groupby('seq_id')])
    split = counts.seq_id.str.split('_').str[-1]                     # seq017_val -> val
    per_split = pd.DataFrame([dict(seq_id=f'all {s}', **percentages(g.sum(numeric_only=True)))
                              for s, g in counts.groupby(split)])
    overall = pd.DataFrame([dict(seq_id='all', **percentages(counts.sum(numeric_only=True)))])
    summary = pd.concat([per_seq, per_split, overall], ignore_index=True)
    per_frame.to_csv(os.path.join(out_dir, 'per_frame.csv'), index=False)
    summary.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    return per_frame, summary


if __name__ == '__main__':
    import joblib
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('data_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--filter', help='quality_filter.joblib saved by seg_compare.ipynb D1 or by --train')
    ap.add_argument('--train', action='store_true', help='train the filter on the train sequences first')
    ap.add_argument('--splits', default='train,val,test')
    ap.add_argument('--um-per-px', type=float, default=None, help='default: read from the data (truth/config.json)')
    a = ap.parse_args()
    if a.um_per_px:
        SC.configure(a.um_per_px)
    else:
        SC.configure_from_data(a.data_dir)
    os.makedirs(a.out_dir, exist_ok=True)
    if a.train:
        model, tau, curve = train_filter(sorted(glob.glob(os.path.join(a.data_dir, 'seq*_train'))))
        joblib.dump((model, tau), os.path.join(a.out_dir, 'quality_filter.joblib'))
        print(curve.to_string(index=False))
    else:
        model, tau = joblib.load(a.filter)
    seqs = [d for d in sorted(glob.glob(os.path.join(a.data_dir, 'seq*'))) if d.split('_')[-1] in a.splits.split(',')]
    per_frame, summary = run(seqs, a.out_dir, model, tau)
    print(summary.to_string(index=False))
