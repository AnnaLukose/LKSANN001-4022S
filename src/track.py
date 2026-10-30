# Step 7: RANSAC bulk motion of the gel + tracking (cross-correlation with a widening search window).
# Output: tracks/<seq>.csv (every track) and tracks/<seq>_counts.csv (scores against the truth).
# Uses: config.py, segment.py.
import os, time, warnings
import numpy as np
import pandas as pd
import cv2
from scipy.spatial import cKDTree
from skimage.measure import regionprops
from . import config as C
from . import segment as sg

warnings.filterwarnings('ignore', category=RuntimeWarning)

# settings in px at the reference scale (L() and A() convert them)
RANSAC_GATE = 80                  # px: centroid pairs further apart than this are not tried
RANSAC_THRESH = 4                 # px: a pair agreeing with the fit to within this is an inlier
SONG_MARGIN = (16, 32)            # px added round the template: small (<= 48 px) / large particles (main_tracking.m)
SONG_SIZE_SWITCH = 48             # px
SONG_CORR_END = 0.5               # below this a lost particle ends with -2 (weak correlation)
SONG_CORR_STRONG = 0.7            # a match in a widened window needs r >= 0.7 (paper); also the backward pass
SONG_AREA_SMALL, SONG_AREA_LARGE = 100, 2000   # px: end codes -3 / -5
SONG_BRIGHT_SKIP = 50             # px: brightest pixels skipped in the -4 (faded) test
AGG_TOL = 0.25                    # merged area within +-25 % of the sum of the parts = aggregation
GAP_FRAMES = 36                   # frames searched for re-appearance: 12 h at 20 min (Song et al. 2025)
GAP_NEAR, GAP_NEAR_AREA = 8, 1.5  # px, area ratio: accept without correlation
GAP_FAR, GAP_FAR_AREA, GAP_CORR = 16, 2.0, 0.5   # px, area ratio, correlation
WIDE_MAX_MARGIN = 160             # px: largest margin tracking tries (this project)
OWN_BINS = (5, 20)                # px of motion relative to the neighbours (for scoring)
LOCAL_R = 150                     # px: neighbours that define the local bulk motion (for scoring)


def L(v):
    """Reference length -> px now."""
    return C.length(v)


def A(v):
    """Reference area -> px now."""
    return C.area(v)


# loading
def load_sequence(seq_dir, out_dir):
    """Everything the tracker needs for one sequence, from its bgrem frames and its segmentation."""
    sid = os.path.basename(seq_dir.rstrip('/'))
    D = pd.read_csv(os.path.join(sg.seg_dir(out_dir, sid), 'detections.csv'))
    files = sg.frames_of(seq_dir)
    nums = [sg.frame_num(f) for f in files]
    mask = sg.load_mask(seq_dir, cv2.imread(files[0], 0).shape)
    high = float(D.otsu.iloc[0]) if len(D) else sg.sequence_otsu(seq_dir, mask)
    imgs, truth, rows, crops, bmasks = [], [], [], [], []
    has_truth = os.path.isdir(os.path.join(seq_dir, 'truth', 'labels'))
    for f, (path, n) in enumerate(zip(files, nums)):
        im = np.where(mask, cv2.imread(path, 0), 0).astype(np.uint8)          # uint8: long deployments fit in memory
        imgs.append(im)
        lab = sg.load_labels(out_dir, sid, n)
        if lab is None:
            raise FileNotFoundError(f'{sid} frame {n}: no segmentation (run segment_sequence on every frame)')
        t = sg.load_truth(seq_dir, n, mask) if has_truth else None
        truth.append(t)
        t_area = np.bincount(t.ravel()) if t is not None else None
        for r in regionprops(lab):
            y0, x0, y1, x1 = r.bbox
            uid = 0
            if t is not None:
                vals = t[r.coords[:, 0], r.coords[:, 1]]
                vals = vals[vals > 0]
                if len(vals):
                    u, cnt = np.unique(vals, return_counts=True)
                    k = np.argmax(cnt)
                    if cnt[k] >= 0.5 * r.area or cnt[k] >= 0.5 * t_area[u[k]]:
                        uid = int(u[k])
            rows.append(dict(pid=len(rows), frame=f, frame_num=n, label=r.label, x=r.centroid[1], y=r.centroid[0],
                             area=int(r.area), x0=x0, y0=y0, x1=x1, y1=y1, uid=uid))
            crops.append(im[y0:y1, x0:x1].astype(np.float32))
            bmasks.append(r.image.copy())
    P = pd.DataFrame(rows, columns=['pid', 'frame', 'frame_num', 'label', 'x', 'y', 'area', 'x0', 'y0', 'x1', 'y1', 'uid'])
    ctx = dict(high=high, crops=crops, bmasks=bmasks, dets=[P[P.frame == f] for f in range(len(imgs))])
    arrival = None
    pf = os.path.join(seq_dir, 'truth', 'particles.csv')
    if os.path.exists(pf):
        p = pd.read_csv(pf)
        arrival = p.set_index('uid').arrival_frame > nums[0]          # True = arrived during the sequence
    return dict(name=sid, imgs=imgs, mask=mask, truth=truth if has_truth else None, P=P, ctx=ctx, nums=nums,
                files=files, arrival=arrival)


# bulk motion
class Field:
    """Displacement (u, v) in px per frame from frame a to frame b at points (x, y) of frame a."""

    def __init__(self, M):
        self.M = np.asarray(M, float)

    def __call__(self, x, y):
        x, y = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))
        M = self.M
        return M[0, 0] * x + M[0, 1] * y + M[0, 2] - x, M[1, 0] * x + M[1, 1] * y + M[1, 2] - y


IDENTITY = np.array([[1, 0, 0], [0, 1, 0]], float)


def motion_ransac(A_, B_):
    """Shift + rotation (+ uniform scale) fitted with RANSAC to every plausible centroid pair."""
    if len(A_) < 3 or len(B_) < 3:
        return Field(IDENTITY)
    pa, pb = A_[['x', 'y']].to_numpy(), B_[['x', 'y']].to_numpy()
    aa, ab = A_.area.to_numpy(float), B_.area.to_numpy(float)
    i, j = [], []
    for k, nb in enumerate(cKDTree(pb).query_ball_point(pa, L(RANSAC_GATE))):
        for m in nb:
            if 0.5 < ab[m] / aa[k] < 2:
                i.append(k)
                j.append(m)
    if len(i) < 3:
        return Field(IDENTITY)
    thr = L(RANSAC_THRESH)
    M, _ = cv2.estimateAffinePartial2D(pa[i].astype(np.float32), pb[j].astype(np.float32), method=cv2.RANSAC,
                                       ransacReprojThreshold=thr, maxIters=5000, confidence=0.999)
    if M is None:
        return Field(IDENTITY)
    tree = cKDTree(pb)
    for _ in range(2):
        pred = pa @ M[:, :2].T + M[:, 2]
        dist, m = tree.query(pred)
        ok = (dist < 2 * thr) & (ab[m] / aa > 0.5) & (ab[m] / aa < 2)
        if ok.sum() < 3:
            break
        M2, _ = cv2.estimateAffinePartial2D(pa[ok].astype(np.float32), pb[m[ok]].astype(np.float32),
                                            method=cv2.RANSAC, ransacReprojThreshold=max(thr - L(1), 1.0))
        if M2 is None:
            break
        M = M2
    return Field(M)


# linking one frame pair
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
    margin = int(round(margin))
    x0, y0 = int(round(cx - (w - 1) / 2)) - margin, int(round(cy - (h - 1) / 2)) - margin
    win = crop_padded(img, x0, y0, w + 2 * margin, h + 2 * margin)
    if template.std() < 1e-6 or win.std() < 1e-6:
        return np.nan, cx, cy
    R = np.nan_to_num(cv2.matchTemplate(win, template.astype(np.float32), cv2.TM_CCOEFF_NORMED), nan=-1)
    py, px_ = np.unravel_index(np.argmax(R), R.shape)
    return float(R[py, px_]), x0 + px_ + (w - 1) / 2, y0 + py + (h - 1) / 2


def in_bbox(xp, yp, B, exclude=()):
    """The detection of B whose box contains (xp, yp), nearest its box centre (Song's rx + ry)."""
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
    """Why a particle was not found (main_tracking.m)."""
    if not in_mask(mask, a.x + du, a.y + dv):
        return -1
    for cx in (a.x0, a.x1 - 1):
        for cy in (a.y0, a.y1 - 1):
            if not in_mask(mask, cx + du, cy + dv):
                return -1
    if not np.isfinite(corr) or corr < SONG_CORR_END:
        return -2
    if a.area <= A(SONG_AREA_SMALL):
        return -3
    if a.area > A(SONG_AREA_LARGE):
        return -5
    px_ = np.sort(ctx['crops'][a.pid].ravel())[::-1]
    if np.mean(px_[int(A(SONG_BRIGHT_SKIP)):int(a.area)]) <= ctx['high'] and corr >= SONG_CORR_STRONG:
        return -4
    return -6


def song_one(a, du, dv, img_b, B, mask, ctx, wide=True, exclude=()):
    """One particle: predicted window, cross-correlation, widened windows if nothing is found."""
    t = ctx['crops'][a.pid]
    cx, cy = (a.x0 + a.x1 - 1) / 2 + du, (a.y0 + a.y1 - 1) / 2 + dv
    m0 = L(SONG_MARGIN[0]) if max(t.shape) <= L(SONG_SIZE_SWITCH) else L(SONG_MARGIN[1])
    margins = [m0]
    while wide and margins[-1] * 2 <= L(WIDE_MAX_MARGIN) + 1e-9:
        margins.append(margins[-1] * 2)
    first_corr = np.nan
    for k, m in enumerate(margins):
        corr, xp, yp = xcorr(t, img_b, cx, cy, m)
        if k == 0:
            first_corr = corr
        if np.isfinite(corr) and (k == 0 or corr >= SONG_CORR_STRONG):
            b = in_bbox(xp, yp, B, exclude)
            if b is not None:
                return b, corr, 0
    return None, first_corr, song_end_code(a, first_corr, du, dv, mask, ctx)


def link_song(A_, B_, img_b, field, mask, ctx, wide=True):
    """Song's frame-to-frame step with the overlap check done at once."""
    du, dv = field(A_.x.to_numpy(), A_.y.to_numpy())
    rows = {a.pid: (a, du[i], dv[i]) for i, a in enumerate(A_.itertuples(index=False))}
    res = {p: song_one(a, u, v, img_b, B_, mask, ctx, wide) for p, (a, u, v) in rows.items()}
    areaB = dict(zip(B_.pid, B_.area))
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
        brow = B_[B_.pid == b].iloc[0]
        img2 = img_b.copy()
        sl = (slice(int(brow.y0), int(brow.y1)), slice(int(brow.x0), int(brow.x1)))
        img2[sl][ctx['bmasks'][b]] = 0
        for p in ps[1:]:
            a, u, v = rows[p]
            nb, corr, code = song_one(a, u, v, img2, B_, mask, ctx, wide, exclude={b})
            taken = nb is not None and any(r[0] == nb for q, r in res.items() if q != p)
            res[p] = (None, corr, code if nb is None else -6) if (nb is None or taken) else (nb, corr, 0)
    links = {p: r[0] for p, r in res.items() if r[0] is not None}
    codes = {p: r[2] for p, r in res.items() if r[0] is None}
    return links, codes


def appearance_corr(ta, tb):
    """Best normalised cross-correlation of cut-out ta anywhere over cut-out tb (padded to fit)."""
    if ta.std() < 1e-6 or tb.std() < 1e-6:
        return np.nan
    ph, pw = max(ta.shape[0] - tb.shape[0], 0) // 2 + 4, max(ta.shape[1] - tb.shape[1], 0) // 2 + 4
    tb = np.pad(tb, ((ph, ph), (pw, pw)))
    return float(cv2.matchTemplate(tb.astype(np.float32), ta.astype(np.float32), cv2.TM_CCOEFF_NORMED).max())


# whole sequences
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


def song_backward(tracks, S, fields):
    """Every track that starts after the first frame is searched for in the previous frame."""
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
        m = L(SONG_MARGIN[0]) if max(t.shape) <= L(SONG_SIZE_SWITCH) else L(SONG_MARGIN[1])
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


def gap_close(tracks, S, fields):
    """Gap closing: join a track that ended to one that starts up to GAP_FRAMES later near its predicted position."""
    P, ctx, imgs = S['P'], S['ctx'], S['imgs']
    T = len(imgs)
    frame = P.frame.to_numpy()
    first = np.array([frame[t['pids'][0]] for t in tracks])
    last = np.array([frame[t['pids'][-1]] for t in tracks])
    starts = np.nonzero(first > 0)[0]
    choice = {}
    for j, t in enumerate(tracks):
        if t['end'] not in {-1, -2, -3, -4, -6} or last[j] >= T - 1:
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
            good = d <= L(GAP_NEAR) and ratio < GAP_NEAR_AREA
            if not good and d <= L(GAP_FAR) and ratio < GAP_FAR_AREA:
                r = appearance_corr(ctx['crops'][int(e.pid)], ctx['crops'][int(s.pid)])
                good = np.isfinite(r) and r >= GAP_CORR
            key = (first[i], d)
            if good and (best is None or key < best[0]):
                best = (key, i)
        if best is not None:
            choice[j] = best[1]
    owner = {}                     # a start claimed by several ends goes to the latest end (as in the MATLAB)
    for j, i in choice.items():
        if i not in owner or last[j] > last[owner[i]]:
            owner[i] = j
    ctx['n_gap'] = ctx.get('n_gap', 0) + len(owner)
    return join_chains(tracks, {j: i for i, j in owner.items()})


def entry_flag(tracks, S, fields):
    """A track starting after frame 0 whose box, carried one frame back."""
    P, mask = S['P'], S['mask']
    for t in tracks:
        b = P.loc[t['pids'][0]]
        f = int(b.frame)
        t['flag'] = 0
        if f == 0:
            continue
        u, v = fields[f - 1](b.x, b.y)
        if not all(in_mask(mask, cx - u[0], cy - v[0]) for cx in (b.x0, b.x1 - 1) for cy in (b.y0, b.y1 - 1)):
            t['flag'] = -7
    return tracks


def bulk_fields(S):
    """RANSAC bulk motion for every frame pair."""
    d = S['ctx']['dets']
    return [motion_ransac(d[f], d[f + 1]) for f in range(len(S['imgs']) - 1)]


def track(S, fields, wide=True):
    """Link a whole sequence (tracking by default; wide=False = Song's own tracker)."""
    ctx, imgs, mask = S['ctx'], S['imgs'], S['mask']
    for k in ('n_agg', 'n_frag', 'n_back', 'n_gap'):
        ctx[k] = 0
    tracks = [dict(pids=[int(p)], end=0, flag=0) for p in ctx['dets'][0].pid]
    alive = {t['pids'][0]: [k] for k, t in enumerate(tracks)}
    for f in range(1, len(imgs)):
        links, codes = link_song(ctx['dets'][f - 1], ctx['dets'][f], imgs[f], fields[f - 1], mask, ctx, wide)
        new = {}
        for a, ks in alive.items():
            b = links.get(a)
            for k in ks:
                if b is None:
                    tracks[k]['end'] = codes.get(a, -6)
                else:
                    tracks[k]['pids'].append(b)
                    new.setdefault(b, []).append(k)
        for b in ctx['dets'][f].pid:
            if int(b) not in new:
                tracks.append(dict(pids=[int(b)], end=0, flag=0))
                new[int(b)] = [len(tracks) - 1]
        alive = new
    tracks = song_backward(tracks, S, fields)
    tracks = gap_close(tracks, S, fields)
    return entry_flag(tracks, S, fields)


def tracks_table(tracks, P):
    rows = [dict(track_id=k, pid=p, end_code=t['end'], flag=t.get('flag', 0)) for k, t in enumerate(tracks)
            for p in t['pids']]
    return pd.DataFrame(rows).merge(P, on='pid')


# scoring against truth
def truth_table(truth):
    """Visible centroid and area of every true particle (>= MIN_AREA px) in every frame."""
    h, w = truth[0].shape
    yy, xx = np.mgrid[:h, :w]
    xx, yy = xx.ravel().astype(float), yy.ravel().astype(float)
    out = []
    for f, lab in enumerate(truth):
        Lb = lab.ravel()
        a = np.bincount(Lb)
        sx, sy = np.bincount(Lb, xx, len(a)), np.bincount(Lb, yy, len(a))
        ids = np.nonzero(a >= C.MIN_AREA)[0]
        ids = ids[ids > 0]
        out.append(pd.DataFrame(dict(frame=f, uid=ids, x=sx[ids] / a[ids], y=sy[ids] / a[ids], area=a[ids])))
    return pd.concat(out, ignore_index=True)


def truth_pairs(T):
    """True displacement of every particle seen in two consecutive frames."""
    a = T.rename(columns=dict(x='xa', y='ya', area='aa'))
    b = T.rename(columns=dict(x='xb', y='yb', area='ab')).assign(frame=lambda d: d.frame - 1)
    d = a.merge(b, on=['frame', 'uid'])
    d['dx'], d['dy'] = d.xb - d.xa, d.yb - d.ya
    d['disp'] = np.hypot(d.dx, d.dy)
    r = d.ab / d.aa
    d['usable'] = (r > 0.7) & (r < 1 / 0.7)
    d['own'] = np.nan
    for f, g in d[d.usable].groupby('frame'):
        xy, dd = g[['xa', 'ya']].to_numpy(), g[['dx', 'dy']].to_numpy()
        own = []
        for i, nb in enumerate(cKDTree(xy).query_ball_point(xy, L(LOCAL_R))):
            nb = [j for j in nb if j != i]
            ref = np.median(dd[nb], 0) if len(nb) >= 3 else np.median(dd, 0)
            own.append(np.hypot(*(dd[i] - ref)))
        d.loc[g.index, 'own'] = own
    return d


def own_bin(v):
    """Own motion in reference px (so the bins mean the same physical distance at every resolution)."""
    lo, hi = OWN_BINS
    v = np.asarray(v, float) / C.K
    return np.select([v < lo, v < hi, v >= hi], [f'with gel <{lo} px', f'own {lo}-{hi} px', f'own >={hi} px'], 'unclear')


def score_tracks(tracks, S, pairs):
    """Link and arrival counts of one sequence (turned into percentages by summarise)."""
    P = S['P']
    uid, frame = P.uid.to_numpy(), P.frame.to_numpy()
    pred = set()
    for t in tracks:
        pred.update(zip(t['pids'][:-1], t['pids'][1:]))
    c = dict(n_links=len(pred), n_links_correct=sum(uid[a] > 0 and uid[a] == uid[b] for a, b in pred))
    D = P[P.uid > 0].sort_values('area').drop_duplicates(['uid', 'frame'], keep='last')
    nxt = D.assign(frame=D.frame - 1)[['uid', 'frame', 'pid']].rename(columns=dict(pid='pid_b'))
    TL = D[['uid', 'frame', 'pid']].merge(nxt, on=['uid', 'frame'])
    own = pairs.set_index(['uid', 'frame']).own
    TL['bin'] = own_bin(own.reindex(pd.MultiIndex.from_frame(TL[['uid', 'frame']])).to_numpy())
    TL['found'] = [(a, b) in pred for a, b in zip(TL.pid, TL.pid_b)]
    c['n_true_links'], c['n_true_links_found'] = len(TL), int(TL.found.sum())
    for k in own_bin(np.array([0.0, OWN_BINS[0], OWN_BINS[1]]) * C.K):
        g = TL[TL.bin == k]
        c[f'true links {k}'], c[f'found {k}'] = len(g), int(g.found.sum())
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
    where = {}
    for k, t in enumerate(tracks):
        for p in t['pids']:
            where.setdefault(p, set()).add(k)
    n2 = unb = 0
    for u, g in D.groupby('uid'):
        if len(g) < 2:
            continue
        n2 += 1
        unb += bool(set.intersection(*[where.get(p, set()) for p in g.pid]))
    tt = S['ttab']
    seen_t = set(zip(P.uid[P.uid > 0], P.frame[P.uid > 0]))
    c.update(n_truth_obj=len(tt), n_truth_det=sum((u, f) in seen_t for u, f in zip(tt.uid, tt.frame)),
             n_det=len(P), n_det_matched=int((P.uid > 0).sum()), n_multi=n2, n_unbroken=unb,
             n_agg=S['ctx'].get('n_agg', 0), n_frag=S['ctx'].get('n_frag', 0), n_back=S['ctx'].get('n_back', 0),
             n_gap=S['ctx'].get('n_gap', 0))
    return c


def pct(a, b):
    b = np.asarray(b, float)
    return np.round(100 * np.asarray(a, float) / np.where(b == 0, np.nan, b), 1)


def summarise(Ccounts, by='seq_id'):
    """Percentages per sequence (by='seq_id'), per split ('split') or pooled (None)."""
    Cc = Ccounts.copy()
    Cc['split'] = Cc.seq_id.str.split('_').str[-1]
    g = Cc.groupby(by).sum(numeric_only=True) if by else Cc.sum(numeric_only=True).to_frame('all').T
    r = pd.DataFrame(index=g.index)
    r['Particles detected %'] = pct(g.n_truth_det, g.n_truth_obj)
    r['Detections on a particle %'] = pct(g.n_det_matched, g.n_det)
    r['Links correct (precision) %'] = pct(g.n_links_correct, g.n_links)
    r['True links found (recall) %'] = pct(g.n_true_links_found, g.n_true_links)
    for k in own_bin(np.array([0.0, OWN_BINS[0], OWN_BINS[1]]) * C.K):
        r[f'Links found, {k} %'] = pct(g[f'found {k}'], g[f'true links {k}'])
    r['New arrivals found %'] = pct(g.n_arrivals_found, g.n_arrivals)
    r['Counted arrivals correct %'] = pct(g.n_starts_correct, g.n_starts)
    r['Counted arrivals, % of true'] = pct(g.n_starts, g.n_arrivals)
    r['Old particles re-counted, % of counted'] = pct(g.n_starts_recount, g.n_starts)
    r['Particles never broken %'] = pct(g.n_unbroken, g.n_multi)
    r['Starts flagged -7 %'] = pct(g.n_starts_flag7, g.n_starts)
    for k, n in (('aggregations', 'n_agg'), ('backward joins', 'n_back'), ('gap joins', 'n_gap')):
        r[f'{k} (n)'] = g[n]
    if 'seconds' in g:
        r['s per sequence'] = (g.seconds / g.n_seq).round(1)
    return r


# running
def track_sequence(seq_dir, out_dir, wide=True):
    """Track one sequence and score it against the truth."""
    sid = os.path.basename(seq_dir.rstrip('/'))
    od = os.path.join(out_dir, 'tracks')
    os.makedirs(od, exist_ok=True)
    f_t, f_c = os.path.join(od, f'{sid}.csv'), os.path.join(od, f'{sid}_counts.csv')
    if os.path.exists(f_t) and os.path.exists(f_c):
        return pd.read_csv(f_t), (pd.read_csv(f_c) if os.path.getsize(f_c) > 2 else pd.DataFrame())   # no truth: empty
    t0 = time.time()
    S = load_sequence(seq_dir, out_dir)
    fields = bulk_fields(S)
    tracks = track(S, fields, wide)
    sec = time.time() - t0
    TT = tracks_table(tracks, S['P']).assign(seq_id=sid)
    Cc = pd.DataFrame()
    if S['truth'] is not None:
        S['ttab'] = truth_table(S['truth'])
        pairs = truth_pairs(S['ttab'])
        Cc = pd.DataFrame([dict(seq_id=sid, seconds=sec, n_seq=1, **score_tracks(tracks, S, pairs))])
    TT.to_csv(f_t, index=False)
    Cc.to_csv(f_c, index=False)
    print(f'{sid}: {TT.track_id.nunique()} tracks, {int((TT.groupby("track_id").frame.min() > 0).sum())} '
          f'counted arrivals ({sec:.0f} s)', flush=True)
    return TT, Cc
