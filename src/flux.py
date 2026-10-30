# Step 10: cumulative carbon and flux per sequence and class, from the pipeline and from the truth.
# Output: tables returned to the notebook (saved to eval/stage5_flux.xlsx).
# Uses: config.py, segment.py.
import os
import numpy as np
import pandas as pd
import cv2
from . import config as C
from . import segment as sg


def seq_geometry(seq_dir):
    """Imaged area (m^2), frame times (hours since the first frame) and deployment length (days)."""
    mask = cv2.imread(os.path.join(seq_dir, 'truth', 'static_mask.png'), 0) > 0
    area_m2 = mask.sum() * (C.UM_PER_PX * 1e-6) ** 2
    fr = pd.read_csv(os.path.join(seq_dir, 'truth', 'frames.csv'))
    hours = fr.hours_since_start.values
    return dict(area_m2=float(area_m2), hours=hours, days=float(max(hours[-1] - hours[0], 1e-9) / 24.0),
                mask=mask, n_frames=len(hours))


def counted_mask(parts, exclude_edge=False):
    """Tracks counted as new arrivals."""
    c = parts.counted.values.astype(bool)
    if exclude_edge:
        c &= parts.flag.values != -7
    return c


def truth_all_arrivals(seq_dir, scheme='amaral7u'):
    """Every true flux particle that arrived during the sequence and was ever visible inside the analysed area."""
    p = pd.read_csv(os.path.join(seq_dir, 'truth', 'particles.csv'))
    o = pd.read_csv(os.path.join(seq_dir, 'truth', 'observations.csv'))
    seen = o[o.area_visible_unmasked_px > 0].groupby('uid').frame.min()
    p = p[(p.arrival_frame >= 1) & p.is_flux & p.uid.isin(seen.index)].copy()
    p['first_seen'] = p.uid.map(seen).astype(int)
    p['label'] = C.scheme_label(p.source_class.values, scheme)
    p['carbon_true'] = p.carbon_mgC_true.fillna(0.0)
    return p


def cumulative(parts, truth_arr, seq_dir, scheme='amaral7u', exclude_edge=False):
    """Cumulative carbon and area per frame: pipeline, detectable truth and all truth."""
    sid = os.path.basename(seq_dir.rstrip('/'))
    g = seq_geometry(seq_dir)
    P = parts[parts.seq_id == sid]
    cnt = counted_mask(P, exclude_edge)
    P = P[cnt]
    ta = truth_arr[(truth_arr.seq_id == sid) & truth_arr.detectable.astype(bool)]
    tall = truth_all_arrivals(seq_dir, scheme)
    nf = g['n_frames']
    px_mm2 = (C.UM_PER_PX / 1000) ** 2
    rows = []
    for f in range(nf):
        rows.append(dict(seq_id=sid, frame=f, hours=g['hours'][f],
                         carbon_pipeline=P.carbon_pred[P.start <= f].sum() / g['area_m2'],
                         carbon_truth_detectable=ta.carbon_true[ta.first_vis <= f].sum() / g['area_m2'],
                         carbon_truth_all=tall.carbon_true[tall.first_seen <= f].sum() / g['area_m2'],
                         area_pipeline_mm2=(P.area_px[P.start <= f] * px_mm2).sum(),
                         area_truth_mm2=(ta.area_um2_true[ta.first_vis <= f] / 1e6).sum(),
                         n_pipeline=int((P.start <= f).sum()), n_truth_detectable=int((ta.first_vis <= f).sum())))
    return pd.DataFrame(rows)


def white_pixels(seq_dir, otsu=None):
    """Song's white-pixel method: area above T_seq in every frame (mm^2), no tracking."""
    files = sg.frames_of(seq_dir)
    mask = sg.load_mask(seq_dir, cv2.imread(files[0], 0).shape)
    otsu = otsu or sg.sequence_otsu(seq_dir, mask)
    px_mm2 = (C.UM_PER_PX / 1000) ** 2
    return np.array([((cv2.imread(f, 0) > otsu) & mask).sum() * px_mm2 for f in files])


def flux_table(parts, truth_arr, seq_dirs, scheme='amaral7u', exclude_edge=False, by_class=False):
    """Flux (mg C m^-2 d^-1) per sequence (and per class)."""
    rows = []
    for d in seq_dirs:
        sid = os.path.basename(d.rstrip('/'))
        g = seq_geometry(d)
        P = parts[parts.seq_id == sid]
        P = P[counted_mask(P, exclude_edge)]
        ta = truth_arr[(truth_arr.seq_id == sid) & truth_arr.detectable.astype(bool)].copy()
        ta['label'] = C.scheme_label(ta.source_class.values, scheme)
        tall = truth_all_arrivals(d, scheme)
        k = 1.0 / (g['area_m2'] * g['days'])
        if by_class:
            cls = sorted(set(P.carbon_class) | set(ta.label) | set(tall.label))
            for c in cls:
                if c in C.NO_CARBON:
                    continue
                rows.append(dict(seq_id=sid, cls=c, flux_pipeline=k * P.carbon_pred[P.carbon_class == c].sum(),
                                 flux_truth_detectable=k * ta.carbon_true[ta.label == c].sum(),
                                 flux_truth_all=k * tall.carbon_true[tall.label == c].sum()))
        else:
            fp, ft, fa = k * P.carbon_pred.sum(), k * ta.carbon_true.sum(), k * tall.carbon_true.sum()
            rows.append(dict(seq_id=sid, days=round(g['days'], 3), area_cm2=round(g['area_m2'] * 1e4, 2),
                             n_counted=len(P), n_true_detectable=len(ta), n_true_all=len(tall),
                             flux_pipeline=fp, flux_truth_detectable=ft, flux_truth_all=fa,
                             error_vs_detectable_pct=100 * (fp / ft - 1) if ft > 0 else np.nan,
                             error_vs_all_pct=100 * (fp / fa - 1) if fa > 0 else np.nan,
                             detectable_share_of_true_pct=100 * ft / fa if fa > 0 else np.nan))
    return pd.DataFrame(rows)


def mae_by_class(flux_cls):
    """Mean absolute error of the per-class flux over sequences (Amaral & Durkin Fig. 6), against both truths."""
    f = flux_cls.assign(ae_det=(flux_cls.flux_pipeline - flux_cls.flux_truth_detectable).abs(),
                        ae_all=(flux_cls.flux_pipeline - flux_cls.flux_truth_all).abs())
    return f.groupby('cls').agg(mae_vs_detectable=('ae_det', 'mean'), mae_vs_all=('ae_all', 'mean'),
                                mean_truth_detectable=('flux_truth_detectable', 'mean'))
