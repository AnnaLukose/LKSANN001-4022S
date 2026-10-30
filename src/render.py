# Step 4: renders the synthetic sequences (30 frames each) with their ground truth.
# Output: <seq>/raw/frame_XXXX.jpg and <seq>/truth/ (particles, observations, frames, labels, mask, truth.xlsx).
# Uses: config.py, carbon.py, excel.py.
import argparse, json, os, math, datetime
import numpy as np
import pandas as pd
import cv2
from . import config as C
from .carbon import carbon_mg

# generator settings; px values are at the reference scale and multiplied by K where they are physical
DEFAULTS = dict(
    interval_min=C.INTERVAL_MIN, interval_jitter_min=2.5, n_frames=30, mode='gelcam',
    cup_radius_frac=0.52, cup_center_jitter=0.03,
    detectable_arrivals_per_frame=[6, 18], ref_detectable_area_px=C.REF_MIN_AREA_PX,
    initial_load_detectable=[0, 120], diel_arrival_amp=[0.0, 0.5], size_slope=[2.5, 4.0],
    swimmers_per_frame=[0.0, 0.15], swimmer_life=[1, 4], swimmer_settle_prob=0.3, swimmer_speed_px=60.0,
    rot_sigma_deg=[0.15, 0.9], rot_ar=0.9, drift_px=[-2.0, 2.0], drift_y_frac=0.3, drift_noise_px=0.5,
    jitter_px=0.3, slide_prob=0.003, slide_px=[5, 40], remove_prob=0.0005,
    depth_defocus_px=[0.0, 1.2], psf_sigma_px=[0.6, 1.0], gain=[1.2, 2.2], gain_obl=[3.3, 6.0], gamma=[0.6, 1.3],
    edge_glow=[0.0, 0.3], occlusion_k=1.4, tint_mode='random_warm', dark_gain=[0.15, 0.5],
    night_level=[18, 32], day_green=[25, 60], day_blue=[15, 40], day_red_leak=[1, 5],
    shot_k=0.5, read_noise=1.5, bayer=True, jpeg_quality=[85, 95],
    transit_prob=0.0, transit_defocus_px=[2.0, 5.0], transit_streak_px=[5, 30],
    label_alpha=0.35, min_render_area_px=8, haze_end_dn=[6, 16], dust_per_frame=[40, 160], dust_amp=[15, 90],
)


def _u(rng, v):
    """[lo, hi] -> a uniform draw; a single value is used as it is."""
    return rng.uniform(*v) if isinstance(v, (list, tuple)) else v


def _coarse(rng, lo, hi, step):
    """Coarse random field on the reference grid, so the random stream does not depend on the frame size."""
    return rng.uniform(lo, hi, (C.REF_HEIGHT // step + 1, C.REF_WIDTH // step + 1)).astype(np.float32)


def make_background(cfg, rng):
    """Empty instrument: cup, LED ring, tube rings, vignetting, ambient light and the static mask."""
    H, W = C.HEIGHT, C.WIDTH
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx = W / 2 + rng.uniform(-1, 1) * cfg['cup_center_jitter'] * W
    cy = H / 2 + rng.uniform(-1, 1) * cfg['cup_center_jitter'] * H
    R = cfg['cup_radius_frac'] * H
    r = np.hypot(xx - cx, yy - cy)
    night = _u(rng, cfg['night_level'])
    base = night * (1 + 0.9 * np.exp(-np.clip(R - r, 0, None) / (0.07 * H)))      # brighter near the LEDs
    for rr, amp in ((0.30 * H * rng.uniform(.9, 1.1), -0.25), (0.40 * H * rng.uniform(.9, 1.1), -0.2)):
        base *= 1 + amp * np.exp(-((r - rr) / (0.012 * H)) ** 2)                   # darker tube rings
    base *= 1 - 0.3 * (r / (0.75 * H)) ** 2                                         # vignetting
    rgb = np.dstack([base, base * rng.uniform(.88, 1.0), base * rng.uniform(.95, 1.1)])
    ring = np.exp(-((r - R) / (0.012 * H)) ** 2)                                    # LED ring
    rgb += 255 * ring[..., None] * np.array([1.0, 1.0, 0.97])
    rgb[r > R + 0.02 * H] *= 0.35                                                   # outside the cup
    mask = (r < R - 0.015 * H).astype(np.uint8)
    amb = cv2.GaussianBlur(_coarse(rng, 0.4, 1.0, 64), (0, 0), 1.5)
    amb = cv2.resize(amb, (W, H), interpolation=cv2.INTER_CUBIC) * (0.6 + 0.4 * np.clip(r / R, 0, 1))
    return rgb.astype(np.float32), amb.astype(np.float32), mask, dict(cx=cx, cy=cy, R=R)


def diel_light(hours, amp):
    """Daylight level (0 at night) from the hour of the day."""
    return amp * np.clip(np.sin(2 * np.pi * (hours - 6.0) / 24.0), 0, 1) ** 1.3


class SpriteCache:
    """Loads each sprite once as [bright RGB, dark RGB, opacity], anti-aliased for the camera scale."""

    def __init__(self, lib_dir, um_target):
        self.dir = os.path.join(lib_dir, 'sprites')
        self.um_t = um_target
        self.cache = {}

    def get(self, row):
        k = row['uid']
        if k not in self.cache:
            im = cv2.imread(os.path.join(self.dir, k + '.png'), cv2.IMREAD_UNCHANGED)
            rgb = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32)
            a = im[..., 3].astype(np.float32) / 255.0
            m = a > 0
            if row.get('illum', 'bf') == 'obl':
                bg = np.array([row.get('bg_r', 55), row.get('bg_g', 55), row.get('bg_b', 55)], np.float32)
                bright = np.clip((rgb - bg) / np.maximum(255 - bg, 1), 0, 1) * m[..., None]
                hue = rgb / np.maximum(rgb.max(2, keepdims=True), 1)
                dark = np.clip((bg.mean() - rgb.mean(2)) / max(bg.mean(), 1), 0, 1)[..., None] * hue * m[..., None]
                src = np.dstack([bright, dark, a])           # 0-2 bright RGB, 3-5 dark RGB, 6 opacity
            else:
                z = np.zeros_like(a)
                src = np.dstack([a, a, a, z, z, z, a])
            f = row['um_per_px'] / self.um_t                 # source -> camera scale
            if f < 1:                                        # anti-alias before shrinking
                src = cv2.GaussianBlur(src, (0, 0), max(0.3, 0.45 / f))
            src = cv2.copyMakeBorder(src, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=0)
            col = rgb[m].mean(0) if m.any() else np.array([200, 200, 200], np.float32)
            self.cache[k] = (src.astype(np.float32), f, col / max(col.max(), 1))
        return self.cache[k]


def warp_patch(a, f, angle_deg, cx, cy):
    """Rotate + scale a sprite (source px) into frame coordinates centred on (cx, cy)."""
    h, w = a.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, f)
    M[:, 2] += np.array([cx - w / 2.0, cy - h / 2.0])
    corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], np.float32) @ M.T
    x0, y0 = np.floor(corners.min(0)).astype(int) - 2
    x1, y1 = np.ceil(corners.max(0)).astype(int) + 2
    M2 = M.copy()
    M2[:, 2] -= [x0, y0]
    pw, ph = max(1, x1 - x0), max(1, y1 - y0)
    parts = [cv2.warpAffine(np.ascontiguousarray(a[..., i:i + 4]), M2, (pw, ph), flags=cv2.INTER_LINEAR,
                            borderValue=0) for i in range(0, a.shape[2], 4)]
    return np.dstack([p if p.ndim == 3 else p[..., None] for p in parts]), x0, y0


def streak(patch, length, ang):
    """Motion blur along a line (particles caught while sinking)."""
    k = np.zeros((int(length) | 1, int(length) | 1), np.float32)
    c = k.shape[0] // 2
    dx, dy = math.cos(ang), math.sin(ang)
    for t in np.linspace(-c, c, 4 * k.shape[0]):
        k[int(round(c + t * dy)), int(round(c + t * dx))] = 1
    k /= k.sum()
    p = np.pad(patch, ((c, c), (c, c)) + ((0, 0),) * (patch.ndim - 2))
    return cv2.filter2D(p, -1, k), c


def sampling_weights(pool, slope):
    """Within-class weights so the sampled sizes follow N(ESD) ~ ESD^-slope."""
    w = np.zeros(len(pool))
    for c, g in pool.groupby('class'):
        e = g['esd_um'].values
        b = np.floor(np.log10(np.clip(e, 1, None)) * 6)
        cnt = pd.Series(b).map(pd.Series(b).value_counts()).values
        ww = e ** (1.0 - slope) / cnt
        w[pool.index.get_indexer(g.index)] = ww / ww.sum()
    return w


class Particle:
    __slots__ = ('uid', 'row', 'x', 'y', 'theta', 'arrival', 'state', 'gain', 'gamma', 'glow', 'tint', 'dark_gain',
                 'defocus', 'swim_left', 'settles', 'vel', 'fate', 'depart', 'transit_left')


def render_sequence(lib, lib_dir, out_dir, seq_id, split, pattern, cfg, seed):
    """Render one sequence and its ground truth into out_dir."""
    rng = np.random.RandomState(seed)
    rng_px = np.random.RandomState(seed + 7919)          # sensor noise at the actual frame size
    os.makedirs(os.path.join(out_dir, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'truth', 'labels'), exist_ok=True)
    H, W, um_t, k = C.HEIGHT, C.WIDTH, C.UM_PER_PX, C.K
    ref_shape = (C.REF_HEIGHT, C.REF_WIDTH, 3)
    pool = lib[lib['split'] == split].reset_index(drop=True)
    swimmers = pool['class'].isin(C.SWIMMERS)
    flux_pool = pool[~swimmers]
    # particles below min_render_area are left to the dust layer
    a_t_all = flux_pool['mask_area_px'] * (flux_pool['um_per_px'] / um_t) ** 2
    flux_pool = flux_pool[a_t_all >= C.area(cfg['min_render_area_px'])].reset_index(drop=True)
    swim_pool = pool[swimmers].reset_index(drop=True)
    comp = {c: v for c, v in pattern['composition'].items() if c in set(flux_pool['class']) and v > 0}
    if not comp:
        comp = flux_pool['class'].value_counts(normalize=True).to_dict()
    tot = sum(comp.values())
    comp = {c: v / tot for c, v in comp.items()}
    slope = _u(rng, cfg['size_slope'])
    wts = sampling_weights(flux_pool, slope)
    # arrival rate is set for particles >= 170 um, so the scene is the same at every resolution
    area_t = flux_pool['mask_area_px'] * (flux_pool['um_per_px'] / um_t) ** 2
    det = (area_t >= C.area(cfg['ref_detectable_area_px'])).values
    p_det = sum(comp[c] * wts[(flux_pool['class'] == c).values][det[(flux_pool['class'] == c).values]].sum()
                for c in comp)
    p_det = max(p_det, 1e-3)
    rate_det = _u(rng, cfg['detectable_arrivals_per_frame'])
    lam = min(rate_det / p_det, 600.0)
    cls_list = list(comp)
    cls_p = np.array([comp[c] for c in cls_list])
    idx_by_cls = {c: np.where(flux_pool['class'].values == c)[0] for c in cls_list}

    def draw_flux():
        c = cls_list[rng.choice(len(cls_list), p=cls_p)]
        ii = idx_by_cls[c]
        p = wts[ii] / wts[ii].sum()
        return flux_pool.iloc[ii[rng.choice(len(ii), p=p)]]

    bg, amb, smask, geo = make_background(cfg, rng)
    cv2.imwrite(os.path.join(out_dir, 'truth', 'static_mask.png'), smask * 255)
    cache = SpriteCache(lib_dir, um_t)
    cx, cy, R = geo['cx'], geo['cy'], geo['R']
    dayamp = rng.uniform(0.6, 1.0)
    g_day, b_day, r_leak = _u(rng, cfg['day_green']), _u(rng, cfg['day_blue']), _u(rng, cfg['day_red_leak'])
    psf = _u(rng, cfg['psf_sigma_px'])
    jq = int(_u(rng, cfg['jpeg_quality']))
    rot_sig = _u(rng, cfg['rot_sigma_deg'])
    haze_end = _u(rng, cfg['haze_end_dn'])
    hz = cv2.resize(cv2.GaussianBlur(_coarse(rng, 0.3, 1.0, 128), (0, 0), 1.0), (W, H),
                    interpolation=cv2.INTER_CUBIC) * (smask > 0)
    drift = np.array([_u(rng, cfg['drift_px']), _u(rng, cfg['drift_px']) * cfg['drift_y_frac']]) * k
    diel_amp_arr = _u(rng, cfg['diel_arrival_amp'])
    t0 = datetime.datetime(2018, 8, 31, 0, 0) + datetime.timedelta(minutes=int(rng.uniform(0, 1440)))
    swim_rate = _u(rng, cfg['swimmers_per_frame']) if len(swim_pool) else 0.0
    particles, next_uid = [], 1
    dust_xy = np.zeros((0, 2), np.float32)
    dust_amp = np.zeros(0, np.float32)
    dust_rate = _u(rng, cfg['dust_per_frame'])
    obs_rows, frame_rows = [], []
    omega, theta_tot, shift_tot = 0.0, 0.0, np.zeros(2)

    def new_particle(row, frame, swimmer=False):
        nonlocal next_uid
        p = Particle()
        p.uid, p.row, p.arrival = next_uid, row, frame
        next_uid += 1
        rr = R * 0.96 * math.sqrt(rng.uniform())
        ph = rng.uniform(0, 2 * math.pi)
        p.x, p.y, p.theta = cx + rr * math.cos(ph), cy + rr * math.sin(ph), rng.uniform(0, 360)
        # gain ranges calibrated against real background-removed GelCam frames
        gk = 'gain_obl' if row.get('illum', 'bf') == 'obl' else 'gain'
        p.gain, p.gamma, p.glow = _u(rng, cfg[gk]), _u(rng, cfg['gamma']), _u(rng, cfg['edge_glow'])
        p.dark_gain = _u(rng, cfg['dark_gain'])
        if cfg['tint_mode'] == 'sprite':
            p.tint = cache.get(row)[2]
        else:
            gg = rng.uniform(.85, 1.0)
            p.tint = np.array([1.0, gg, gg * rng.uniform(.6, .95)], np.float32)     # white .. golden
        p.defocus = _u(rng, cfg['depth_defocus_px']) * k                             # blur of a physical depth
        p.state = 'swimming' if swimmer else 'settled'
        p.swim_left = int(rng.randint(cfg['swimmer_life'][0], cfg['swimmer_life'][1] + 1)) if swimmer else 0
        p.settles = swimmer and rng.uniform() < cfg['swimmer_settle_prob']
        p.vel = rng.normal(0, cfg['swimmer_speed_px'] * k, 2) if swimmer else np.zeros(2)
        p.fate, p.depart = 'present_at_end', -1
        p.transit_left = 1 if (not swimmer and frame >= 0 and rng.uniform() < cfg['transit_prob']) else 0
        if p.transit_left:
            p.state = 'transit'
        particles.append(p)
        return p

    for _ in range(int(_u(rng, cfg['initial_load_detectable']) / p_det)):     # particles already there at frame 0
        new_particle(draw_flux(), -1)

    t_min = 0.0
    for fi in range(cfg['n_frames']):
        if fi > 0:
            t_min += cfg['interval_min'] + rng.uniform(0, cfg['interval_jitter_min'])
        ts = t0 + datetime.timedelta(minutes=t_min)
        hours = ts.hour + ts.minute / 60.0
        # gel motion, common to every settled particle
        if fi > 0:
            omega = cfg['rot_ar'] * omega + rng.normal(0, rot_sig) * math.sqrt(1 - cfg['rot_ar'] ** 2)
            dshift = drift + rng.normal(0, cfg['drift_noise_px'] * k, 2)
            theta_tot += omega
            shift_tot += dshift
            co, si = math.cos(math.radians(omega)), math.sin(math.radians(omega))
            for p in particles:
                if p.fate != 'present_at_end':
                    continue
                if p.state == 'swimming':
                    p.vel = 0.6 * p.vel + rng.normal(0, cfg['swimmer_speed_px'] * k, 2)
                    p.x += p.vel[0]
                    p.y += p.vel[1]
                    p.theta += rng.uniform(-90, 90)
                    p.swim_left -= 1
                    if p.swim_left <= 0:
                        if p.settles:
                            p.state = 'settled'
                        else:
                            p.fate, p.depart = 'swam_out', fi
                    continue
                if p.state == 'transit':
                    p.state = 'settled'
                dx, dy = p.x - cx, p.y - cy
                p.x = cx + co * dx + si * dy + dshift[0] + rng.normal(0, cfg['jitter_px'] * k)
                p.y = cy - si * dx + co * dy + dshift[1] + rng.normal(0, cfg['jitter_px'] * k)
                p.theta += omega
                if rng.uniform() < cfg['slide_prob']:
                    p.x += rng.choice([-1, 1]) * _u(rng, cfg['slide_px']) * k
                if rng.uniform() < cfg['remove_prob']:
                    p.fate, p.depart = 'resuspended', fi
                if math.hypot(p.x - cx, p.y - cy) > R * 1.02:
                    p.fate, p.depart = 'left_fov', fi
            dxy = dust_xy - [cx, cy]
            dust_xy = np.stack([cx + co * dxy[:, 0] + si * dxy[:, 1] + dshift[0],
                                cy - si * dxy[:, 0] + co * dxy[:, 1] + dshift[1]], 1).astype(np.float32)
            # new arrivals (rate follows the time of day when diel_arrival_amp > 0)
            lam_t = lam * (1 + diel_amp_arr * math.cos(2 * math.pi * hours / 24.0))
            for _ in range(rng.poisson(max(lam_t, 0))):
                new_particle(draw_flux(), fi)
            for _ in range(rng.poisson(swim_rate)):
                new_particle(swim_pool.iloc[rng.randint(len(swim_pool))], fi, swimmer=True)

        nd = rng.poisson(dust_rate if fi > 0 else dust_rate * 6)                   # some dust already there at start
        rr = R * 0.97 * np.sqrt(rng.uniform(size=nd))
        ph = rng.uniform(0, 2 * np.pi, nd)
        dust_xy = np.vstack([dust_xy, np.stack([cx + rr * np.cos(ph), cy + rr * np.sin(ph)], 1)]).astype(np.float32)
        dust_amp = np.concatenate([dust_amp, rng.uniform(*cfg['dust_amp'], nd).astype(np.float32)])
        # compose the frame: background, daylight, haze, particles
        d = diel_light(hours, dayamp)
        canvas = bg.copy()
        canvas[..., 1] += g_day * d * amb
        canvas[..., 2] += b_day * d * amb
        canvas[..., 0] += r_leak * d * amb
        # diffuse veil from sub-resolution material, growing through the deployment
        canvas += (haze_end * fi / max(cfg['n_frames'] - 1, 1)) * hz[..., None] * np.array([1.0, .95, .85], np.float32)
        labels = np.zeros((H, W), np.uint16)
        n_present, n_new, pstats = 0, 0, []
        for p in particles:
            if p.fate != 'present_at_end':
                continue
            a, f, _ = cache.get(p.row)
            patch, x0, y0 = warp_patch(a, f, p.theta, p.x, p.y)
            sig, stk = p.defocus, 0
            if p.state == 'transit':
                sig = math.hypot(sig, _u(rng, cfg['transit_defocus_px']) * k)
                patch, c = streak(patch, _u(rng, cfg['transit_streak_px']) * k, rng.uniform(0, math.pi))
                x0 -= c
                y0 -= c
                stk = 1
            if sig > 0.3:
                patch = cv2.GaussianBlur(patch, (0, 0), sig)
            ph_, pw_ = patch.shape[:2]
            xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x0 + pw_, W), min(y0 + ph_, H)
            if xb <= xa or yb <= ya:
                continue
            pp = patch[ya - y0:yb - y0, xa - x0:xb - x0]
            pa = pp[..., 6]                                                  # opacity
            if p.row.get('illum', 'bf') == 'obl':                            # each pixel keeps its own colour
                rad = p.gain * np.power(np.clip(pp[..., :3] + p.dark_gain * pp[..., 3:6], 0, None), p.gamma)
            else:                                                            # brightfield: opacity x warm tint
                rad = p.gain * np.power(pa, p.gamma)[..., None] * p.tint
            if p.glow > 0:
                rad = rad + p.glow * np.clip(-cv2.Laplacian(pa, cv2.CV_32F), 0, 1)[..., None]
            ao = np.clip(cfg['occlusion_k'] * pa, 0, 1)[..., None]
            reg = canvas[ya:yb, xa:xb]
            canvas[ya:yb, xa:xb] = reg * (1 - ao) + 255.0 * rad
            lm = pa > cfg['label_alpha']
            labels[ya:yb, xa:xb][lm] = p.uid
            full = int(lm.sum())
            if full:
                ys, xs = np.nonzero(lm)
                pstats.append((p, full, xs.mean() + xa, ys.mean() + ya, xa + xs.min(), ya + ys.min(),
                               xa + xs.max(), ya + ys.max(), float(pa[lm].mean()), stk))
            else:
                pstats.append((p, 0, p.x, p.y, int(p.x), int(p.y), int(p.x), int(p.y), 0.0, stk))
            n_present += 1
            n_new += int(p.arrival == fi)
        dimg = np.zeros((H, W), np.float32)
        ix, iy = dust_xy[:, 0].astype(int), dust_xy[:, 1].astype(int)
        ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        np.add.at(dimg, (iy[ok], ix[ok]), dust_amp[ok])
        dimg = cv2.GaussianBlur(dimg, (0, 0), 0.7)
        canvas += dimg[..., None] * np.array([1.0, 0.92, 0.75], np.float32)
        vis = np.bincount(labels.ravel(), minlength=next_uid + 1)
        vis_in = np.bincount(labels[smask > 0].ravel(), minlength=next_uid + 1)
        for (p, full, mx, my, bx0, by0, bx1, by1, mrad, stk) in pstats:
            obs_rows.append(dict(seq_id=seq_id, frame=fi, uid=p.uid, state=p.state,
                                 x=round(mx, 2), y=round(my, 2), bbox_x0=bx0, bbox_y0=by0,
                                 bbox_x1=bx1, bbox_y1=by1, area_full_px=full,
                                 area_visible_px=int(vis[p.uid]), area_visible_unmasked_px=int(vis_in[p.uid]),
                                 occluded_frac=round(1 - vis[p.uid] / full, 3) if full else 0.0,
                                 angle_deg=round(p.theta % 360, 1), mean_opacity=round(mrad, 3),
                                 transit_streak=stk))
        # optics and sensor: PSF, shot + read noise, Bayer mosaic, JPEG
        canvas = cv2.GaussianBlur(canvas, (0, 0), psf)
        canvas = np.clip(canvas, 0, None)
        noise = rng.normal(0, 1, ref_shape).astype(np.float32)          # keeps the random stream resolution-free
        if noise.shape != canvas.shape:
            noise = rng_px.normal(0, 1, canvas.shape).astype(np.float32)
        canvas += noise * np.sqrt(cfg['shot_k'] * canvas + cfg['read_noise'] ** 2)
        img = np.clip(canvas, 0, 255).astype(np.uint8)
        if cfg['bayer']:
            mos = np.zeros((H, W), np.uint8)
            mos[0::2, 0::2] = img[0::2, 0::2, 0]   # R
            mos[0::2, 1::2] = img[0::2, 1::2, 1]   # G
            mos[1::2, 0::2] = img[1::2, 0::2, 1]   # G
            mos[1::2, 1::2] = img[1::2, 1::2, 2]   # B
            bgr = cv2.cvtColor(mos, cv2.COLOR_BayerBG2BGR)     # RGGB mosaic (BayerRG2BGR would swap red and blue)
        else:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        fname = f'frame_{fi:04d}.jpg'
        cv2.imwrite(os.path.join(out_dir, 'raw', fname), bgr, [cv2.IMWRITE_JPEG_QUALITY, jq])
        cv2.imwrite(os.path.join(out_dir, 'truth', 'labels', f'frame_{fi:04d}.png'), labels)
        frame_rows.append(dict(seq_id=seq_id, frame=fi, filename=fname, timestamp=ts.isoformat(timespec='seconds'),
                               hours_since_start=round(t_min / 60, 3), diel_light=round(float(d), 3),
                               gel_rotation_deg=round(-theta_tot, 3), shift_x=round(shift_tot[0], 2),
                               shift_y=round(shift_tot[1], 2), n_present=n_present, n_new=n_new,
                               n_dust_specks=int(ok.sum())))

    # one row per particle
    first_area = {}
    for o in obs_rows:
        if o['uid'] not in first_area and o['state'] != 'transit':
            first_area[o['uid']] = o['area_full_px']
    prow = []
    for p in particles:
        r = p.row
        a_t = first_area.get(p.uid, 0)
        d = dict(seq_id=seq_id, uid=p.uid, source_uid=r['uid'], source_file=r['file'], split=split,
                 source_class=r['class'])
        for s in C.SCHEMES:
            d[f'class_{s}'] = C.scheme_label([r['class']], s)[0]
        d.update(is_flux=r['class'] in C.FLUX, is_swimmer=r['class'] in C.SWIMMERS,
                 arrival_frame=p.arrival, departure_frame=p.depart, fate=p.fate,
                 esd_um_true=round(r['esd_um'], 1), area_um2_true=round(r['area_um2'], 1),
                 major_um_true=round(r['major_um'], 1), minor_um_true=round(r['minor_um'], 1),
                 aspect_true=round(r['aspect'], 3),
                 carbon_mgC_true=carbon_mg(r['class'], r['esd_um']),
                 render_area_px=a_t, render_area_um2=round(a_t * um_t ** 2, 1),
                 render_esd_um=round(2 * math.sqrt(a_t / math.pi) * um_t, 1),
                 detectable_song=a_t >= C.MIN_AREA,                      # pipeline's limit (config.MIN_AREA)
                 detectable_ref=a_t * um_t ** 2 >= C.REF_MIN_AREA_PX * C.REF_UM_PER_PX ** 2,   # >= ~170 um
                 brightness_gain=round(p.gain, 3), gamma=round(p.gamma, 3), defocus_px=round(p.defocus, 2),
                 source_um_per_px=r['um_per_px'], source_mag=r['mag'], source_illum=r['illum'])
        prow.append(d)
    meta = dict(seq_id=seq_id, split=split, seed=seed, pattern=pattern['name'],
                pattern_composition=comp, size_slope=slope, lambda_all_sizes=lam,
                p_detectable=p_det, detectable_rate=rate_det, psf_sigma=psf, jpeg_quality=jq,
                rot_sigma_deg=rot_sig, drift_px=drift.tolist(), geometry=geo, resolution=C.summary(), cfg=cfg)
    pdf, odf, fdf = pd.DataFrame(prow), pd.DataFrame(obs_rows), pd.DataFrame(frame_rows)
    tdir = os.path.join(out_dir, 'truth')
    pdf.to_csv(os.path.join(tdir, 'particles.csv'), index=False)
    odf.to_csv(os.path.join(tdir, 'observations.csv'), index=False)
    fdf.to_csv(os.path.join(tdir, 'frames.csv'), index=False)
    json.dump(meta, open(os.path.join(tdir, 'config.json'), 'w'), indent=1, default=float)
    from .excel import write_workbook
    cfg_rows = pd.DataFrame([(k_, json.dumps(v, default=float)) for k_, v in meta.items() if k_ != 'cfg'] +
                            [('cfg.' + k_, json.dumps(v)) for k_, v in cfg.items()], columns=['key', 'value'])
    write_workbook(os.path.join(tdir, 'truth.xlsx'),
                   {'particles': pdf, 'observations': odf, 'frames': fdf, 'config': cfg_rows}, readme=TRUTH_README)
    return meta, pdf


TRUTH_README = [
    ('particles', 'One row per physical particle. *_true are the measured microscope sizes (um) of the source sprite; '
                  'render_* is its footprint in the camera on arrival (px and um2). carbon_mgC_true = Durkin et al. '
                  '(2021) Table 2 on the true class and ESD. arrival_frame = -1: already there at frame 0. '
                  'detectable_song: footprint >= the pipeline minimum area; detectable_ref: footprint >= ~170 um '
                  '(Song et al. 2025). class_amaral7u = the 8 classifier classes.'),
    ('observations', 'One row per particle per frame it is in view: centroid (x, y, px), box, full footprint, area '
                     'left visible after occlusion, visible area inside the analysed (static-mask) area, orientation.'),
    ('frames', 'One row per frame: file name, time stamp, daylight, cumulative gel rotation (deg) and drift (px).'),
    ('config', 'Every generator setting, the seed and the camera resolution, so the sequence can be regenerated.'),
    ('labels/', 'truth/labels/frame_XXXX.png: 16-bit instance map, pixel value = particle uid (top-most wins).'),
]


def split_plan(n_seq, plan):
    """'16,2,2' or '8,1,1' -> list of splits for n_seq sequences (a requested split gets at least one)."""
    ratio = np.array([float(x) for x in str(plan).split(',')])
    counts = np.round(ratio / ratio.sum() * n_seq).astype(int)
    counts[1:] = np.where((ratio[1:] > 0) & (counts[1:] == 0), 1, counts[1:])
    counts[0] = max(1, n_seq - counts[1:].sum())
    return ['train'] * counts[0] + ['val'] * counts[1] + ['test'] * counts[2]


def render_dataset(lib_dir, out_dir, n_seq=20, frames=30, plan='16,2,2', seed=0, equal_classes=False, swimmers=None,
                   prefix='seq', max_new=None, cfg_over=None, after_each=None):
    """Render a whole dataset (resumable: a sequence with truth/truth.xlsx is skipped)."""
    cfg = dict(DEFAULTS, **(cfg_over or {}))
    cfg['n_frames'] = frames
    if swimmers:
        cfg['swimmers_per_frame'] = [float(x) for x in swimmers]
    lib = pd.read_csv(os.path.join(lib_dir, 'library.csv'))
    pj = json.load(open(os.path.join(lib_dir, 'patterns.json')))
    pats = pj['patterns']
    if equal_classes:
        pats = [dict(name='equal_classes', composition={c: 1.0 / len(pj['classes']) for c in pj['classes']})]
    os.makedirs(out_dir, exist_ok=True)
    man, done_new = [], 0
    for i, split in enumerate(split_plan(n_seq, plan)):
        pat = pats[i % len(pats)]                              # every split sees every pattern in turn
        sid = f'{prefix}{i:03d}_{split}'
        sdir = os.path.join(out_dir, sid)
        if os.path.exists(os.path.join(sdir, 'truth', 'truth.xlsx')):
            pdf = pd.read_csv(os.path.join(sdir, 'truth', 'particles.csv'))
            man.append(dict(seq_id=sid, split=split, pattern=pat['name'], n_particles=len(pdf),
                            n_detectable=int(pdf['detectable_song'].sum())))
            continue
        if max_new is not None and done_new >= max_new:
            continue
        meta, pdf = render_sequence(lib, lib_dir, sdir, sid, split, pat, cfg, seed * 1000 + i)
        done_new += 1
        man.append(dict(seq_id=sid, split=split, pattern=pat['name'], n_particles=len(pdf),
                        n_detectable=int(pdf['detectable_song'].sum()), size_slope=round(meta['size_slope'], 2)))
        print(man[-1], flush=True)
        if after_each:
            after_each(sdir)
    pd.DataFrame(man).to_csv(os.path.join(out_dir, 'manifest.csv'), index=False)
    return pd.DataFrame(man)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('lib_dir')
    ap.add_argument('out_dir')
    ap.add_argument('--um-per-px', type=float, default=13.0)
    ap.add_argument('--n-seq', type=int, default=20)
    ap.add_argument('--frames', type=int, default=30)
    ap.add_argument('--split-plan', default='16,2,2')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--equal-classes', action='store_true')
    ap.add_argument('--swimmers', help='swimmers per frame as "lo,hi", e.g. 0.8,1.2')
    ap.add_argument('--prefix', default='seq')
    ap.add_argument('--max-new', type=int)
    a = ap.parse_args()
    C.configure(a.um_per_px)
    render_dataset(a.lib_dir, a.out_dir, a.n_seq, a.frames, a.split_plan, a.seed, a.equal_classes,
                   a.swimmers.split(',') if a.swimmers else None, a.prefix, a.max_new)


if __name__ == '__main__':
    main()
