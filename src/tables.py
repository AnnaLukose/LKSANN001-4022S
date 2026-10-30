# Step 8: measurements, an RGB crop and the truth of every detection.
# Output: tables/<seq>.pkl, <seq>_crops.npy and <seq>_truth_arrivals.csv.
# Uses: config.py, segment.py. Used by: classify.py, hitl.py.
import glob, os, time
import numpy as np
import pandas as pd
import cv2
from skimage.measure import regionprops
from . import config as C
from . import segment as sg

FEATURES = ['area_px', 'esd_um', 'major_px', 'minor_px', 'aspect', 'eccentricity', 'solidity', 'extent',
            'circularity', 'int_mean', 'int_max', 'mean_peak']


def read_raw(seq_dir, n, shape):
    """Raw RGB frame (float)."""
    for f in (f'frame_{n:04d}.jpg', f'frame_{n:04d}.png'):
        p = os.path.join(seq_dir, 'raw', f)
        if os.path.exists(p):
            return cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB).astype(np.float32)
    g = cv2.imread(os.path.join(seq_dir, 'bgrem', f'bgrem_{n:04d}.png'), 0).astype(np.float32)
    return np.repeat(g[..., None], 3, 2)


def crop_rgb(raw, cy, cx, side):
    """3 x CANVAS x CANVAS crop centred on the particle at native scale (shrunk only if bigger than FIT px)."""
    H, W = raw.shape[:2]
    s = 1.0 if side <= C.FIT else C.FIT / side
    half = C.CANVAS / (2 * s)
    ys, xs = int(max(cy - half - 2, 0)), int(max(cx - half - 2, 0))
    ye, xe = int(min(cy + half + 3, H)), int(min(cx + half + 3, W))
    M = np.array([[s, 0, C.CANVAS / 2 - s * (cx - xs)], [0, s, C.CANVAS / 2 - s * (cy - ys)]], np.float32)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    ch = [cv2.warpAffine(np.ascontiguousarray(raw[ys:ye, xs:xe, i]), M, (C.CANVAS, C.CANVAS), flags=interp)
          for i in range(3)]
    return np.clip(np.stack(ch), 0, 255).astype(np.uint8), s


def measure(r, g):
    """Measurements of one detection (regionprops object r) on the bgrem frame g."""
    a = float(r.area)
    maj, mnr = max(r.axis_major_length, 1.0), max(r.axis_minor_length, 1.0)
    per = max(r.perimeter, 1.0)
    v = g[r.coords[:, 0], r.coords[:, 1]]
    pk = v.max() + 1e-6
    return dict(area_px=a, esd_um=C.esd_um(a), major_px=maj, minor_px=mnr, aspect=maj / mnr,
                eccentricity=r.eccentricity, solidity=r.solidity, extent=r.extent,
                circularity=4 * np.pi * a / per ** 2, int_mean=float(v.mean()), int_max=float(v.max()),
                mean_peak=float(v.mean() / pk))


def match_uid(coords, truth, t_area):
    """The true particle covering most of the detection."""
    vals = truth[coords[:, 0], coords[:, 1]]
    vals = vals[vals > 0]
    if not len(vals):
        return 0, 0.0
    u, n = np.unique(vals, return_counts=True)
    k = int(np.argmax(n))
    iou = n[k] / (len(coords) + t_area[u[k]] - n[k])
    ok = n[k] >= 0.5 * len(coords) or n[k] >= 0.5 * t_area[u[k]]
    return (int(u[k]) if ok else 0), float(iou)


def truth_first_visible(seq_dir, nums, mask):
    """Frame index in which each true particle first covers >= MIN_AREA px inside the mask (the arrival rule)."""
    first = {}
    for i, n in enumerate(nums):
        t = sg.load_truth(seq_dir, n, mask)
        a = np.bincount(t.ravel())
        for u in np.nonzero(a >= C.MIN_AREA)[0]:
            if u and u not in first:
                first[int(u)] = i
    return first


def one_track_per_detection(TT):
    """tracking can let two tracks run through one detection (two particles merging into one outline)."""
    n = TT.groupby('track_id').pid.transform('size')
    return TT.assign(_n=n).sort_values(['_n', 'track_id'], ascending=[False, True]) \
        .drop_duplicates('pid').drop(columns='_n').sort_values(['track_id', 'pid'])


def build_sequence(seq_dir, out_dir):
    """Measure, crop and label every detection of one sequence."""
    sid = os.path.basename(seq_dir.rstrip('/'))
    td = os.path.join(out_dir, 'tables')
    os.makedirs(td, exist_ok=True)
    f_pkl = os.path.join(td, f'{sid}.pkl')
    if os.path.exists(f_pkl):
        return pd.read_pickle(f_pkl)
    t0 = time.time()
    D = pd.read_csv(os.path.join(sg.seg_dir(out_dir, sid), 'detections.csv'))
    files = sg.frames_of(seq_dir)
    nums_all = [sg.frame_num(f) for f in files]
    fidx = {n: i for i, n in enumerate(nums_all)}
    mask = sg.load_mask(seq_dir, cv2.imread(files[0], 0).shape)
    nums = sorted(D.frame.unique()) if len(D) else []
    # tracks (val / test): (frame number, label) -> track id and flag
    tmap = {}
    f_tr = os.path.join(out_dir, 'tracks', f'{sid}.csv')
    if os.path.exists(f_tr):
        TT = one_track_per_detection(pd.read_csv(f_tr))
        tmap = {(int(a), int(b)): (int(c), int(d)) for a, b, c, d in
                zip(TT.frame_num, TT.label, TT.track_id, TT.flag)}
    corr = {(int(a), int(b)): float(c) for a, b, c in zip(D.frame, D.label, D.area_corrected_px)} if len(D) else {}
    P = pd.read_csv(os.path.join(seq_dir, 'truth', 'particles.csv')).set_index('uid')
    first_vis = truth_first_visible(seq_dir, nums_all, mask)
    rows, crops = [], []
    for n in nums:
        g = cv2.imread(os.path.join(seq_dir, 'bgrem', f'bgrem_{n:04d}.png'), 0).astype(np.float32)
        raw = read_raw(seq_dir, n, g.shape)
        lab = sg.load_labels(out_dir, sid, n)
        t = sg.load_truth(seq_dir, n, mask)
        ta = np.bincount(t.ravel())
        for r in regionprops(lab):
            uid, iou = match_uid(r.coords, t, ta)
            tid, flag = tmap.get((int(n), int(r.label)), (-1, 0))
            f = measure(r, g)
            ac = corr.get((int(n), int(r.label)), f['area_px'])
            y0, x0, y1, x1 = r.bbox
            c, s = crop_rgb(raw, r.centroid[0], r.centroid[1], max(y1 - y0, x1 - x0))
            f.update(seq_id=sid, frame=int(n), fidx=fidx[n], label=int(r.label), det_id=len(rows), track_id=tid,
                     flag=flag, uid=uid, iou=iou, x=r.centroid[1], y=r.centroid[0], x0=x0, y0=y0, x1=x1, y1=y1,
                     area_corrected_px=ac, esd_corrected_um=C.esd_um(ac), crop_scale=s)
            rows.append(f)
            crops.append(c)
    T = pd.DataFrame(rows)
    if not len(T):
        T = pd.DataFrame(columns=['uid', 'seq_id', 'track_id'])
    T['split'] = sid.split('_')[-1]
    u = T.uid.astype(int)
    T['source_class'] = np.where(u > 0, P.source_class.reindex(u).values, C.NOISE)
    T['source_uid'] = np.where(u > 0, P.source_uid.reindex(u).astype(str).values, 'none')
    T['carbon_true'] = np.where(u > 0, P.carbon_mgC_true.reindex(u).fillna(0).values, 0.0)
    T['esd_true'] = np.where(u > 0, P.esd_um_true.reindex(u).values, np.nan)
    T['detectable'] = np.where(u > 0, P.detectable_song.reindex(u).fillna(False).astype(bool).values, False)
    T['first_vis'] = u.map(first_vis).fillna(-1).astype(int)
    T['otsu'] = float(D.otsu.iloc[0]) if len(D) else np.nan
    T.to_pickle(f_pkl)
    np.save(os.path.join(td, f'{sid}_crops.npy'), np.stack(crops) if crops else
            np.zeros((0, 3, C.CANVAS, C.CANVAS), np.uint8))
    # every true particle that became visible after the first frame (the truth for recall, carbon and flux)
    arr = [dict(seq_id=sid, uid=u_, first_vis=fv, source_class=P.source_class.get(u_),
                carbon_true=float(np.nan_to_num(P.carbon_mgC_true.get(u_, 0.0))),
                area_um2_true=float(P.area_um2_true.get(u_, np.nan)), esd_true=float(P.esd_um_true.get(u_, np.nan)),
                detectable=bool(P.detectable_song.get(u_, False)), source_uid=str(P.source_uid.get(u_)))
           for u_, fv in first_vis.items() if fv > 0]
    pd.DataFrame(arr).to_csv(os.path.join(td, f'{sid}_truth_arrivals.csv'), index=False)
    n_tr = T[T.track_id >= 0].track_id.nunique() if len(T) else 0
    print(f'{sid}: {len(T)} detections, ' + (f'{n_tr} tracks' if n_tr else 'not tracked (training: one particle = one uid)')
          + f' ({time.time() - t0:.0f} s)', flush=True)
    return T


def load_tables(out_dir, splits=('train', 'val', 'test'), crops=True):
    """Every built sequence of the given splits -> (table, crops)."""
    Ts, Cs = [], []
    for f in sorted(glob.glob(os.path.join(out_dir, 'tables', '*.pkl'))):
        sid = os.path.basename(f)[:-4]
        if sid.split('_')[-1] not in splits:
            continue
        Ts.append(pd.read_pickle(f))
        if crops:
            Cs.append(np.load(os.path.join(out_dir, 'tables', f'{sid}_crops.npy'), mmap_mode='r'))
    if not Ts:
        raise FileNotFoundError(f'no particle tables of {splits} in {out_dir}/tables: run Part D')
    T = pd.concat(Ts, ignore_index=True)
    return T, (np.concatenate([np.asarray(c) for c in Cs]) if crops else None)


def truth_arrivals(out_dir, splits):
    f = [x for x in sorted(glob.glob(os.path.join(out_dir, 'tables', '*_truth_arrivals.csv')))
         if os.path.basename(x).replace('_truth_arrivals.csv', '').split('_')[-1] in splits]
    return pd.concat([pd.read_csv(x) for x in f], ignore_index=True) if f else pd.DataFrame()
