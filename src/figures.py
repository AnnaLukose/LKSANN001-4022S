# All figures: Song et al. (2025), Amaral & Durkin (2025) and the per-stage checks.
# Output: PNGs in figures/.
# Uses: config.py, segment.py, track.py.
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import config as C
from . import segment as sg

# one fixed colour per class
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
CLASS_COLORS = dict(zip(['aggregate', 'long_pellet', 'short_pellet', 'mini_pellet', 'salp_pellet', 'rhizaria',
                         'phytoplankton', 'non_flux'], SERIES))
CLASS_COLORS['false_detection'] = '#8c8b87'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
TRUTH = '#0b0b0b'


def style():
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                         'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
                         'axes.titlesize': 10, 'axes.titleweight': 'bold', 'lines.linewidth': 2,
                         'figure.dpi': 110, 'savefig.dpi': 150, 'savefig.bbox': 'tight'})


style()


def _save(fig, out_png):
    if out_png:
        fig.savefig(out_png)
    return fig


def _bright(im, gain=2.0):
    return np.clip(im * gain, 0, 255).astype(np.uint8)


def cls_color(c):
    return CLASS_COLORS.get(c, '#8c8b87')


# Song et al. (2025)
def _single(img, title, out_png, cmap=None, width=8):
    """One image as its own figure (no axes), saved to out_png."""
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(width, width * h / w + 0.4))
    ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=255 if cmap else None)
    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    return _save(fig, out_png)


def preprocessing(seq_dir, frame, out_dir=None, out_prefix=None, gain=3.0):
    """Song Fig. 2 as separate figures, one per step."""
    raw = cv2.cvtColor(cv2.imread(os.path.join(seq_dir, 'raw', f'frame_{frame:04d}.jpg')), cv2.COLOR_BGR2RGB)
    g = cv2.imread(os.path.join(seq_dir, 'bgrem', f'bgrem_{frame:04d}.png'), 0)
    files = sorted(os.listdir(os.path.join(seq_dir, 'raw')))
    gb = np.array([cv2.imread(os.path.join(seq_dir, 'raw', f))[..., :2].reshape(-1, 2).mean(0) for f in files])
    fr = pd.read_csv(os.path.join(seq_dir, 'truth', 'frames.csv'))
    out = []

    def name(tag):
        f = f'{out_prefix}_{tag}.png' if out_prefix else None
        out.append(f)
        return f

    _single(raw, f'(A) raw frame {frame}', name('A_raw'))
    _single(raw[..., 1], '(B) green channel', name('B_green'), 'gray')
    _single(raw[..., 2], '(B) blue channel', name('B_blue'), 'gray')
    fig, a = plt.subplots(figsize=(7, 3.4))
    a.plot(fr.hours_since_start, gb[:, 1], color=SERIES[5], label='green')
    a.plot(fr.hours_since_start, gb[:, 0], color=SERIES[0], ls=':', label='blue')
    a.set_xlabel('hours since first frame')
    a.set_ylabel('mean pixel value')
    a.set_title('(C) day-night cycle')
    a.legend()
    _save(fig, name('C_day_night'))
    _single(raw[..., 0], '(D) red channel', name('D_red'), 'gray')
    _single(_bright(g, gain), f'(E) background removed (x{gain:g} for display)', name('E_bgrem'), 'gray')
    out += binarized_and_outlines(g, seq_dir, frame, out_dir, out_prefix, 'F', 'G', gain)
    return out


def binarized_and_outlines(g, seq_dir, frame, out_dir, out_prefix, tag_bin='F', tag_out='G', gain=3.0, otsu=None):
    """Binarized frame and the segmentation outlines, as two figures."""
    from skimage.segmentation import find_boundaries
    sid = os.path.basename(seq_dir.rstrip('/'))
    mask = sg.load_mask(seq_dir, g.shape)
    otsu = otsu or sg.sequence_otsu(seq_dir, mask)
    files = []
    f1 = f'{out_prefix}_{tag_bin}_binarized.png' if out_prefix else None
    _single(((g > otsu) & mask).astype(np.uint8) * 255, f'({tag_bin}) binarized at T_seq = {otsu:.1f}', f1, 'gray')
    files.append(f1)
    vis = np.dstack([_bright(g, gain)] * 3)
    lab = sg.load_labels(out_dir, sid, frame) if out_dir else None
    if lab is not None:
        vis[find_boundaries(lab, mode='outer')] = (255, 220, 0)
    f2 = f'{out_prefix}_{tag_out}_outlines.png' if out_prefix else None
    _single(vis, f'({tag_out}) particles outlined (segmentation)', f2)
    files.append(f2)
    return files


def trajectories(seq_dir, tracks, n=6, out_png=None, min_len=5):
    """Song Fig. 3: the n longest tracks drawn over the static mask with their start and end times (left)."""
    fr = pd.read_csv(os.path.join(seq_dir, 'truth', 'frames.csv'))
    mask = cv2.imread(os.path.join(seq_dir, 'truth', 'static_mask.png'), 0)
    T = tracks.sort_values(['track_id', 'frame'])
    lens = T.groupby('track_id').size()
    ids = list(lens[lens >= min_len].sort_values(ascending=False).index[:n])
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(2, 5, width_ratios=[2.2, 1, 1, 1, 1])
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(mask, cmap='gray', vmin=0, vmax=600)
    for k, tid in enumerate(ids):
        g = T[T.track_id == tid]
        col = SERIES[k % len(SERIES)]
        ax.plot(g.x, g.y, '-', color=col, lw=2.6)
        ax.plot(g.x.iloc[0], g.y.iloc[0], 'o', mfc='white', mec=col, ms=6)
        t0, t1 = fr.timestamp.iloc[int(g.frame.iloc[0])], fr.timestamp.iloc[int(g.frame.iloc[-1])]
        ax.annotate(f'{t0[5:16]} - {t1[11:16]}', (g.x.iloc[-1], g.y.iloc[-1]), color=INK, fontsize=7,
                    xytext=(4, 4), textcoords='offset points')
    ax.set_title(f'{len(ids)} tracked trajectories (white circle = start)')
    ax.axis('off')
    if ids:
        g = T[T.track_id == ids[0]]
        side = int(max(64 * C.K, 2.4 * max((g.x1 - g.x0).max(), (g.y1 - g.y0).max())))
        pick = np.unique(np.linspace(0, len(g) - 1, 8).round().astype(int))
        for c, kk in enumerate(pick):
            row = g.iloc[kk]
            im = cv2.imread(os.path.join(seq_dir, 'bgrem', f'bgrem_{int(row.frame_num):04d}.png'), 0).astype(np.float32)
            ox, oy = int(round(row.x)) - side // 2, int(round(row.y)) - side // 2
            from .track import crop_padded
            rgb = np.dstack([_bright(crop_padded(im, ox, oy, side, side), 2.5)] * 3)
            ax2 = fig.add_subplot(gs[c // 4, 1 + c % 4])
            ax2.imshow(rgb)
            ax2.add_patch(plt.Rectangle((row.x0 - ox, row.y0 - oy), row.x1 - row.x0, row.y1 - row.y0, fill=False,
                                        ec='#eda100', lw=1.2))
            ax2.set_title(fr.timestamp.iloc[int(row.frame)][11:16], fontsize=8)
            ax2.axis('off')
    fig.suptitle(f'{os.path.basename(seq_dir)}: trajectories and the morphology of the longest-tracked particle')
    fig.tight_layout()
    return _save(fig, out_png)


def class_examples(T, Cr, classes, n=8, out_png=None, label_col='label', seed=0):
    """Song Fig. 4 / Amaral & Durkin Fig. 2."""
    rng = np.random.default_rng(seed)
    classes = [c for c in classes if (T[label_col] == c).any()]
    fig, ax = plt.subplots(len(classes), n, figsize=(1.3 * n, 1.4 * len(classes)))
    ax = np.atleast_2d(ax)
    for r, c in enumerate(classes):
        idx = np.nonzero((T[label_col] == c).values)[0]
        big = idx[np.argsort(-T.area_px.values[idx])][:max(n * 4, n)]
        pick = rng.choice(big, min(n, len(big)), replace=False)
        for k in range(n):
            a = ax[r, k]
            a.axis('off')
            if k < len(pick):
                a.imshow(np.clip(np.asarray(Cr[pick[k]]).transpose(1, 2, 0).astype(float) * 1.6, 0, 255).astype(np.uint8))
        ax[r, 0].set_title(c, loc='left', fontsize=8, color=INK)
    fig.suptitle(f'example particles per class ({C.CANVAS} px crops = {C.CANVAS * C.UM_PER_PX / 1000:.2f} mm)')
    fig.tight_layout()
    return _save(fig, out_png)


def cumulative_area(cum, white, seq_id, out_png=None):
    """Song Fig. 5: cumulative particle area, white-pixel method vs tracking vs truth."""
    g = cum[cum.seq_id == seq_id]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.plot(g.hours, white, ':', color=INK2, lw=1.5, label='white pixels (binarized frame)')
    ax.plot(g.hours, g.area_pipeline_mm2 + (white[0] if len(white) else 0), color=SERIES[1],
            label='tracking: area at arrival (+ frame 0 white pixels)')
    ax.plot(g.hours, g.area_truth_mm2 + (white[0] if len(white) else 0), '--', color=TRUTH, lw=1.3,
            label='truth: detectable arrivals (+ frame 0)')
    ax.set_xlabel('hours since first frame')
    ax.set_ylabel('cumulative particle area (mm²)')
    ax.set_title(f'{seq_id}: cumulative particle area')
    ax.legend(fontsize=8)
    return _save(fig, out_png)


def flux_by_type(flux_cls, out_png=None, title='carbon flux by particle type'):
    """Song Fig. 6: stacked bars of the pipeline's flux per class for every sequence."""
    seqs = sorted(flux_cls.seq_id.unique())
    cls = [c for c in CLASS_COLORS if c in set(flux_cls.cls)]
    fig, ax = plt.subplots(figsize=(1.6 + 1.4 * len(seqs), 4))
    x = np.arange(len(seqs))
    bottom = np.zeros(len(seqs))
    for c in cls:
        v = np.array([flux_cls[(flux_cls.seq_id == s) & (flux_cls.cls == c)].flux_pipeline.sum() for s in seqs])
        ax.bar(x, v, 0.55, bottom=bottom, color=cls_color(c), edgecolor='white', linewidth=1, label=c)
        bottom += v
    td = [flux_cls[flux_cls.seq_id == s].flux_truth_detectable.sum() for s in seqs]
    tall = [flux_cls[flux_cls.seq_id == s].flux_truth_all.sum() for s in seqs]
    ax.scatter(x, td, s=60, facecolors='white', edgecolors=TRUTH, zorder=5, label='truth (detectable)')
    ax.scatter(x, tall, s=50, marker='x', color=TRUTH, zorder=5, label='truth (all arrivals)')
    ax.set_xticks(x, seqs, rotation=20)
    ax.set_ylabel('POC flux (mg C m⁻² d⁻¹)')
    ax.set_title(title)
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')
    return _save(fig, out_png)


def cumulative_carbon(cum, out_png=None):
    """Accumulating carbon through every sequence."""
    seqs = sorted(cum.seq_id.unique())
    fig, ax = plt.subplots(1, len(seqs), figsize=(4.2 * len(seqs), 3.4), squeeze=False)
    for a, s in zip(ax[0], seqs):
        g = cum[cum.seq_id == s]
        a.plot(g.hours, g.carbon_pipeline, color=SERIES[0], label='pipeline')
        a.plot(g.hours, g.carbon_truth_detectable, '--', color=TRUTH, lw=1.3, label='truth (detectable)')
        a.plot(g.hours, g.carbon_truth_all, ':', color=INK2, lw=1.3, label='truth (all arrivals)')
        a.set_title(s)
        a.set_xlabel('hours since first frame')
    ax[0, 0].set_ylabel('cumulative carbon (mg C m⁻²)')
    ax[0, 0].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, out_png)


# Amaral & Durkin (2025)
def class_distribution(T, out_png=None):
    """Amaral & Durkin Fig. 3."""
    p = T.drop_duplicates('pkey')
    tab = p.groupby(['label', 'split']).size().unstack(fill_value=0)
    tab = tab.reindex([c for c in list(CLASS_COLORS) if c in tab.index])
    splits = [s for s in ('train', 'val', 'test') if s in tab.columns]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.8 / len(splits)
    x = np.arange(len(tab))
    for i, s in enumerate(splits):
        ax.bar(x + (i - (len(splits) - 1) / 2) * w, tab[s], w * 0.92, color=SERIES[i], label=s)
    ax.set_yscale('log')
    ax.set_xticks(x, tab.index, rotation=25)
    ax.set_ylabel('particles (log scale)')
    ax.set_title('labelled particles per class and domain')
    ax.legend()
    return _save(fig, out_png)


def flux_scatter(flux_by_stage, out_png=None):
    """Amaral & Durkin Fig. 5."""
    st = list(flux_by_stage)
    fig, ax = plt.subplots(1, len(st), figsize=(3.6 * len(st), 3.6), squeeze=False)
    hi = max(max(f[['flux_pipeline', 'flux_truth_detectable']].max().max() for f in flux_by_stage.values()), 1e-9)
    for a, s in zip(ax[0], st):
        f = flux_by_stage[s]
        for c, g in f.groupby('cls'):
            a.scatter(g.flux_truth_detectable, g.flux_pipeline, s=26, color=cls_color(c), label=c,
                      edgecolors='white', linewidths=0.6)
        a.plot([0, hi * 1.05], [0, hi * 1.05], '--', color=INK2, lw=1)
        mae = (f.flux_pipeline - f.flux_truth_detectable).abs().mean()
        a.set_title(f'{s}  (MAE {mae:.3g})')
        a.set_xlabel('true flux (mg C m⁻² d⁻¹)')
    ax[0, 0].set_ylabel('pipeline flux (mg C m⁻² d⁻¹)')
    ax[0, -1].legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc='upper left')
    fig.tight_layout()
    return _save(fig, out_png)


def mae_by_stage(mae_tabs, out_png=None):
    """Amaral & Durkin Fig. 6."""
    st = list(mae_tabs)
    cls = [c for c in CLASS_COLORS if any(c in m.index for m in mae_tabs.values())]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.8 / len(st)
    x = np.arange(len(cls))
    for i, s in enumerate(st):
        v = mae_tabs[s].mae_vs_detectable.reindex(cls).values
        ax.bar(x + (i - (len(st) - 1) / 2) * w, v, w * 0.92, color=SERIES[i % len(SERIES)], label=s)
    ax.set_xticks(x, cls, rotation=25)
    ax.set_ylabel('MAE of flux (mg C m⁻² d⁻¹)')
    ax.set_title('flux error per class and stage (vs detectable truth)')
    ax.legend(fontsize=8)
    return _save(fig, out_png)


def precision_recall(pr_tabs, out_png=None):
    """Amaral & Durkin Fig. 7."""
    st = list(pr_tabs)
    cls = [c for c in CLASS_COLORS if any(c in set(p.cls) for p in pr_tabs.values())]
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    w = 0.8 / len(st)
    x = np.arange(len(cls))
    for i, s in enumerate(st):
        p = pr_tabs[s].set_index('cls').reindex(cls)
        for a, col in zip(ax, ('precision', 'recall')):
            a.bar(x + (i - (len(st) - 1) / 2) * w, 100 * p[col].values, w * 0.92, color=SERIES[i % len(SERIES)],
                  label=s)
    ax[0].set_ylabel('precision (%)')
    ax[1].set_ylabel('recall (%)')
    ax[1].set_xticks(x, cls, rotation=25)
    ax[0].legend(fontsize=7, ncol=3)
    ax[0].set_title('precision and recall per class (never-checked particles)')
    return _save(fig, out_png)


def flux_by_class_stage(flux_cls_by_stage, out_png=None):
    """Amaral & Durkin Fig. 8."""
    st = list(flux_cls_by_stage)
    f0 = flux_cls_by_stage[st[0]]
    seqs = sorted(f0.seq_id.unique())
    cls = [c for c in CLASS_COLORS if c in set(f0.cls)]
    fig, ax = plt.subplots(len(seqs), 1, figsize=(10, 2.6 * len(seqs)), squeeze=False)
    groups = ['truth (detectable)'] + st
    w = 0.8 / len(groups)
    x = np.arange(len(cls))
    for a, s in zip(ax[:, 0], seqs):
        for i, gname in enumerate(groups):
            src = f0 if i == 0 else flux_cls_by_stage[gname]
            col = 'flux_truth_detectable' if i == 0 else 'flux_pipeline'
            v = src[src.seq_id == s].set_index('cls')[col].reindex(cls).fillna(0).values
            a.bar(x + (i - (len(groups) - 1) / 2) * w, v, w * 0.92,
                  color=TRUTH if i == 0 else SERIES[(i - 1) % len(SERIES)], label=gname)
        a.set_title(s, loc='left')
        a.set_ylabel('mg C m⁻² d⁻¹')
        a.set_xticks(x, cls)
    ax[0, 0].legend(fontsize=7, ncol=len(groups))
    fig.tight_layout()
    return _save(fig, out_png)


# stage checks
def generation(seq_dir, frame, out_png=None):
    """Generator check."""
    raw = cv2.cvtColor(cv2.imread(os.path.join(seq_dir, 'raw', f'frame_{frame:04d}.jpg')), cv2.COLOR_BGR2RGB)
    lab = cv2.imread(os.path.join(seq_dir, 'truth', 'labels', f'frame_{frame:04d}.png'), -1)
    p = pd.read_csv(os.path.join(seq_dir, 'truth', 'particles.csv'))
    p = p[p.render_area_px > 0]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw=dict(width_ratios=[1.3, 1.3, 1]))
    ax[0].imshow(raw)
    ax[0].set_title(f'raw frame {frame} ({raw.shape[1]} x {raw.shape[0]} px, {C.UM_PER_PX:g} um/px)')
    rng = np.random.default_rng(0)
    cmap = np.vstack([[0, 0, 0], rng.uniform(0.25, 1, (int(lab.max()) + 1, 3))])
    ax[1].imshow(cmap[np.minimum(lab, len(cmap) - 1)])
    ax[1].set_title('truth: instance labels')
    for a in ax[:2]:
        a.axis('off')
    for c, g in p.groupby('class_amaral7u'):
        ax[2].scatter(g.esd_um_true, g.render_esd_um, s=8, color=cls_color(c), label=c, alpha=0.8)
    lim = [p.esd_um_true.min() * 0.8, p.esd_um_true.max() * 1.2]
    ax[2].plot(lim, lim, '--', color=INK2, lw=1)
    ax[2].axhline(C.esd_um(C.MIN_AREA), color=SERIES[7], lw=1, ls=':')
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_xlabel('true ESD (um, microscope)')
    ax[2].set_ylabel('rendered ESD (um)')
    ax[2].set_title('size kept in um (dotted: detection limit)')
    ax[2].legend(fontsize=6, markerscale=1.5)
    fig.tight_layout()
    return _save(fig, out_png)


def seg_boxes(seq_dir, out_dir, frame, out_png=None, zoom=0.35):
    """Segmentation check."""
    sid = os.path.basename(seq_dir.rstrip('/'))
    D = pd.read_csv(os.path.join(sg.seg_dir(out_dir, sid), 'detections.csv'))
    img = cv2.imread(os.path.join(seq_dir, 'bgrem', f'bgrem_{frame:04d}.png'), 0).astype(np.float32)
    vis = cv2.cvtColor(sg.draw_boxes(img, D[D.frame == frame], gain=3.0), cv2.COLOR_BGR2RGB)
    H, W = img.shape
    zh, zw = int(H * zoom), int(W * zoom)
    y0, x0 = (H - zh) // 2, (W - zw) // 2
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))
    ax[0].imshow(vis)
    ax[0].add_patch(plt.Rectangle((x0, y0), zw, zh, fill=False, ec='#2a78d6', lw=1.5))
    ax[0].set_title(f'{sid} frame {frame}: {int((D.frame == frame).sum())} detections')
    ax[1].imshow(vis[y0:y0 + zh, x0:x0 + zw])
    ax[1].set_title('blue box enlarged')
    for a in ax:
        a.axis('off')
    fig.tight_layout()
    return _save(fig, out_png)


def tau_curve(curve, tau, out_png=None):
    """Out-of-fold results of the quality filter at every cut-off tau (training sequences)."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    for i, c in enumerate(('recall', 'flux_recall', 'precision', 'carbon_found')):
        ax[0].plot(curve.tau, curve[c], marker='o', ms=4, color=SERIES[i], label=c)
    for i, c in enumerate(('total_carbon', 'volume_precision')):
        ax[1].plot(curve.tau, curve[c], marker='o', ms=4, color=SERIES[i + 4], label=c)
    ax[1].axhline(100, color=INK2, lw=0.8)
    for a in ax:
        a.axvline(tau, color=INK2, ls='--', lw=0.8)
        a.set_xlabel('cut-off tau')
        a.set_ylabel('%')
        a.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, out_png)


def carbon_per_frame(per_frame, out_png=None, col='Total carbon, % of true'):
    """Total (or found) carbon frame by frame for every sequence, as % of the true carbon in that frame."""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, (s, g) in enumerate(per_frame.groupby('seq_id')):
        ax.plot(g.frame, g[col], marker='o', ms=3, color=SERIES[i % len(SERIES)], label=s)
    ax.axhline(100, color=INK2, lw=0.8)
    ax.set_xlabel('frame')
    ax.set_ylabel(col)
    ax.legend(fontsize=8)
    return _save(fig, out_png)


def effort_curve(R, out_png=None):
    """Accuracy, balanced accuracy and carbon error against the share of the deployment the expert checked."""
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
    for di, dom in enumerate(sorted(R.domain.unique())):
        e = R[R.domain == dom]
        for stage, ls in (('+verify', '-'), ('+rareboost', '--')):
            g = pd.concat([e[e.stage == 'OOD'], e[e.stage == stage]]).sort_values('checked_pct')
            col = SERIES[di * 2 + (stage == '+rareboost')]
            ax[0].plot(g.checked_pct, 100 * g.accuracy, marker='o', ls=ls, color=col, label=f'{dom} {stage}')
            ax[1].plot(g.checked_pct, 100 * g.balanced_accuracy, marker='o', ls=ls, color=col, label=f'{dom} {stage} (8)')
            ax[1].plot(g.checked_pct, 100 * g.gelcam5_balanced_accuracy, marker='s', ls=ls, color=col, alpha=0.5,
                       label=f'{dom} {stage} (GELCAM5)')
            ax[2].plot(g.checked_pct, 100 * g.carbon_error_all, marker='o', ls=ls, color=col, label=f'{dom} {stage}')
    for a, t in zip(ax, ('accuracy, never-checked (%)', 'balanced accuracy (%)', 'carbon error, all counted (%)')):
        a.set_xlabel("% of the deployment's particles checked")
        a.set_ylabel(t)
        a.legend(fontsize=6)
    ax[2].axhline(0, color=INK2, lw=0.8)
    fig.tight_layout()
    return _save(fig, out_png)


def confusion(cm, out_png=None, title='confusion (never-checked arrivals)'):
    """Confusion matrix as row percentages (recall), with the counts in each cell."""
    pct = cm.div(cm.sum(1).replace(0, 1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(1.2 + 0.75 * cm.shape[1], 1.0 + 0.6 * cm.shape[0]))
    ax.imshow(pct.values, cmap='Blues', vmin=0, vmax=100)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{int(cm.values[i, j])}', ha='center', va='center', fontsize=7,
                    color='white' if pct.values[i, j] > 60 else INK)
    ax.set_xticks(range(cm.shape[1]), cm.columns, rotation=40, ha='right')
    ax.set_yticks(range(cm.shape[0]), cm.index)
    ax.set_xlabel('predicted')
    ax.set_ylabel('true')
    ax.set_title(title)
    return _save(fig, out_png)
