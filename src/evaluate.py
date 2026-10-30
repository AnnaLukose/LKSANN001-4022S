# Checks of the generator and of background removal against the truth, plus one summary table.
# Output: tables returned to the notebook (saved to eval/).
# Uses: config.py, segment.py.
import os
import numpy as np
import pandas as pd
import cv2
from . import config as C
from . import segment as sg

SEG_KEYS = ['Particles found (recall) %', 'Detections correct (precision) %', 'Flux particles found %',
            'Carbon found (carbon-weighted recall) %', 'Total carbon, % of true',
            'False carbon (no particle under it), % of true']
TRK_KEYS = ['Links correct (precision) %', 'True links found (recall) %', 'New arrivals found %',
            'Counted arrivals correct %', 'Old particles re-counted, % of counted', 'Particles never broken %']


# stage 0: generator
def check_generation(seq_dirs):
    """Per sequence: camera, particles, arrivals and rendered vs true size."""
    rows = []
    for d in seq_dirs:
        p = pd.read_csv(os.path.join(d, 'truth', 'particles.csv'))
        fr = pd.read_csv(os.path.join(d, 'truth', 'frames.csv'))
        im = cv2.imread(os.path.join(d, 'raw', 'frame_0000.jpg'))
        import json
        res = json.load(open(os.path.join(d, 'truth', 'config.json'))).get('resolution', {})
        flux = p[p.is_flux & (p.render_area_px >= C.MIN_AREA)]
        ratio = flux.render_esd_um / flux.esd_um_true
        rows.append(dict(seq_id=os.path.basename(d), width=im.shape[1], height=im.shape[0],
                         um_per_px=res.get('um_per_px'), frames=len(fr), particles=len(p),
                         arrived_during=int((p.arrival_frame >= 1).sum()),
                         detectable=int(p.detectable_song.sum()),
                         detectable_arrivals=int(((p.arrival_frame >= 1) & p.detectable_song).sum()),
                         swimmers=int(p.is_swimmer.sum()),
                         aggregate_share_pct=round(100 * (p.class_amaral7u == 'aggregate').mean(), 1),
                         rendered_vs_true_ESD_median=round(float(ratio.median()), 3) if len(ratio) else np.nan,
                         rendered_ESD_within_15pct=round(100 * float((ratio.sub(1).abs() <= 0.15).mean()), 1) if len(ratio) else np.nan,
                         true_carbon_ug=round(1000 * p.carbon_mgC_true.fillna(0).sum(), 2)))
    return pd.DataFrame(rows)


def compare_scenes(dir_a, dir_b):
    """Is sequence dir_b the same scene as dir_a (same seed at another resolution)?"""
    pa = pd.read_csv(os.path.join(dir_a, 'truth', 'particles.csv'))
    pb = pd.read_csv(os.path.join(dir_b, 'truth', 'particles.csv'))
    same = len(pa) == len(pb) and (pa.source_uid.values == pb.source_uid.values).all() and \
        (pa.arrival_frame.values == pb.arrival_frame.values).all()
    out = dict(seq_a=os.path.basename(dir_a), seq_b=os.path.basename(dir_b), particles_a=len(pa), particles_b=len(pb),
               same_particles_and_arrivals=bool(same))
    if same:
        oa = pd.read_csv(os.path.join(dir_a, 'truth', 'observations.csv'))
        ob = pd.read_csv(os.path.join(dir_b, 'truth', 'observations.csv'))
        m = oa.merge(ob, on=['frame', 'uid'])
        wa = cv2.imread(os.path.join(dir_a, 'raw', 'frame_0000.jpg')).shape[1]
        wb = cv2.imread(os.path.join(dir_b, 'raw', 'frame_0000.jpg')).shape[1]
        out['position_ratio_median'] = round(float(np.median(m.x_y / m.x_x)), 4)
        out['frame_width_ratio'] = round(wb / wa, 4)
    return out


# stage 1: preprocessing
def check_preprocessing(seq_dir, frames=None):
    """Per frame after background removal: background left, particle brightness and share above T_seq."""
    files = sg.frames_of(seq_dir)
    mask = sg.load_mask(seq_dir, cv2.imread(files[0], 0).shape)
    om = sg.sequence_otsu(seq_dir, mask)
    fr = pd.read_csv(os.path.join(seq_dir, 'truth', 'frames.csv')).set_index('frame')
    pick = frames if frames is not None else range(len(files))
    rows = []
    for i in pick:
        n = sg.frame_num(files[i])
        g = cv2.imread(files[i], 0).astype(np.float32)
        t = sg.load_truth(seq_dir, n, mask)
        a = np.bincount(t.ravel())
        mean = np.bincount(t.ravel(), weights=g.ravel(), minlength=len(a)) / np.maximum(a, 1)
        big = np.nonzero(a >= C.MIN_AREA)[0]
        big = big[big > 0]
        bg = g[mask & (cv2.dilate((t > 0).astype(np.uint8), np.ones((5, 5), np.uint8)) == 0)]
        rows.append(dict(seq_id=os.path.basename(seq_dir), frame=n, daylight=float(fr.diel_light.get(n, np.nan)),
                         T_seq=round(om, 1), bg_p50=float(np.percentile(bg, 50)), bg_p90=float(np.percentile(bg, 90)),
                         bg_p99=float(np.percentile(bg, 99)),
                         particle_mean_median=float(np.median(mean[big])) if len(big) else np.nan,
                         particles_above_T_seq_pct=round(100 * float((mean[big] > om).mean()), 1) if len(big) else np.nan,
                         contrast_particle_over_bg_p90=round(float(np.median(mean[big]) / max(np.percentile(bg, 90), 1)), 2) if len(big) else np.nan,
                         n_true_particles=len(big)))
    return pd.DataFrame(rows)


# summary
def summary_table(seg_summary=None, trk_summary=None, cls_results=None, flux_seq=None, split='test'):
    """The key number of every stage in one table."""
    rows = []
    if seg_summary is not None and f'all {split}' in set(seg_summary.seq_id):
        s = seg_summary.set_index('seq_id').loc[f'all {split}']
        rows += [dict(stage='2 segmentation', measure=k, value=s.get(k)) for k in SEG_KEYS]
    if trk_summary is not None:
        t = trk_summary.iloc[0]
        rows += [dict(stage='3 tracking', measure=k, value=t.get(k)) for k in TRK_KEYS]
    if cls_results is not None and len(cls_results):
        for _, g in cls_results[cls_results.domain == split].iterrows():
            for col in ('accuracy', 'balanced_accuracy', 'accuracy_esd_ge_500um', 'carbon_error_found', 'carbon_error_all'):
                v = None if pd.isna(g[col]) else round(100 * float(g[col]), 1)
                rows.append(dict(stage='4-5 classification + carbon', measure=f"{g['key']} {col} %",
                                 checked_pct=round(g['checked_pct'], 1), value=v))
    if flux_seq is not None and len(flux_seq):
        f = flux_seq[flux_seq.seq_id.str.endswith(split)]
        rows.append(dict(stage='5 flux', measure='flux error vs detectable truth %',
                         value=round(float(f.error_vs_detectable_pct.mean()), 1)))
        rows.append(dict(stage='5 flux', measure='detectable share of the true flux %',
                         value=round(float(f.detectable_share_of_true_pct.mean()), 1)))
    return pd.DataFrame(rows).reindex(columns=['stage', 'measure', 'checked_pct', 'value'])
