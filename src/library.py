# Step 2: cuts particles out of the labelled gel-trap crops and splits them by field of view.
# Output: lib/library.csv and lib/sprites/*.png.
# Used by: patterns.py, render.py.
import argparse, os, glob, json
import numpy as np
import pandas as pd
import cv2
from skimage.measure import label, regionprops
from .config import parse_name, um_per_px_for

OPAQUE_ABSORBANCE = 0.6        # brightfield: 60% of the gel's light absorbed = fully opaque (my choice)
OBLIQUE_FULL_CONTRAST = 45.0   # oblique: 45 grey levels above the ~55 background = fully opaque (my choice)


def _corners_bg(gray, k=3):
    h, w = gray.shape
    k = max(1, min(k, h // 4, w // 4))
    patches = [gray[:k, :k], gray[:k, -k:], gray[-k:, :k], gray[-k:, -k:]]
    return float(np.median(np.concatenate([p.ravel() for p in patches])))


def contrast_map(rgb, illum):
    """Contrast against the gel (read from the crop corners): darker for brightfield, any change for oblique."""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    bg = _corners_bg(g)
    if illum == 'obl':
        rb = rgb[..., 0].astype(np.float32) - rgb[..., 2].astype(np.float32)
        return np.maximum(np.abs(g - bg), np.abs(rb - _corners_bg(rb))), bg
    return np.clip(bg - g, 0, None), bg


def segment_sprite(c, target_area=None):
    """Largest blob above 0.6 x Otsu (or closest to a known area)."""
    c8 = np.clip(c, 0, 255).astype(np.uint8)
    if c8.max() < 8:
        return None
    blur = cv2.GaussianBlur(c8, (3, 3), 0)

    def mask_at(t):
        m = (blur > t).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        lab = label(m)
        if lab.max() == 0:
            return None
        rp = max(regionprops(lab), key=lambda r: r.area)
        return (lab == rp.label)

    if target_area:
        best, bd = None, 1e18
        for t in range(4, int(blur.max())):
            m = mask_at(t)
            if m is None:
                break
            d = abs(int(m.sum()) - target_area)
            if d < bd:
                best, bd = m, d
            if m.sum() < target_area * 0.5:
                break
        return best
    from skimage.filters import threshold_otsu
    t = max(8.0, float(threshold_otsu(blur)) * 0.6)  # 0.6: Otsu alone clips faint aggregate edges
    return mask_at(t)


def shape_stats(mask, um):
    r = regionprops(mask.astype(np.uint8))[0]
    area = r.area
    return dict(mask_area_px=int(area), esd_um=2 * np.sqrt(area / np.pi) * um,
                area_um2=area * um * um, major_um=r.axis_major_length * um,
                minor_um=r.axis_minor_length * um,
                aspect=r.axis_major_length / max(r.axis_minor_length, 1e-6),
                solidity=float(r.solidity), extent=float(r.extent))


def assign_splits(df, group_col, seed=0, fracs=(0.8, 0.1, 0.1)):
    """Grouped, class-stratified 8:1:1 split."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    rng = np.random.RandomState(seed)
    split = np.array(['train'] * len(df), dtype=object)
    y = df['class'].values
    nfold = 10
    groups = df[group_col].values if group_col else None
    ok_groups = groups is not None and len(np.unique(groups)) >= nfold
    min_count = pd.Series(y).value_counts().min()
    if ok_groups:
        cv = StratifiedGroupKFold(n_splits=nfold, shuffle=True, random_state=seed)
        folds = list(cv.split(np.zeros(len(y)), y, groups))
    elif min_count >= 2:
        cv = StratifiedKFold(n_splits=min(nfold, max(2, min_count)), shuffle=True, random_state=seed)
        folds = list(cv.split(np.zeros(len(y)), y))
    else:
        idx = rng.permutation(len(y))
        folds = [(None, idx[i::nfold]) for i in range(nfold)]
    if len(folds) >= 10:
        split[folds[0][1]] = 'test'
        split[folds[1][1]] = 'val'
    else:  # too few groups: random 10 % / 10 %
        idx = rng.permutation(len(y))
        n = len(y)
        split[idx[:int(round(fracs[2] * n))]] = 'test'
        split[idx[int(round(fracs[2] * n)):int(round((fracs[1] + fracs[2]) * n))]] = 'val'
    return split, ('grouped:' + group_col) if ok_groups else 'stratified_random(groups too few)'


def build(crops_dir, out_dir, csv=None, group_by='micrograph', max_per_class=None, seed=0,
          um_override=None, classes=None):
    os.makedirs(os.path.join(out_dir, 'sprites'), exist_ok=True)
    meta = None
    if csv:
        meta = pd.read_csv(csv, low_memory=False)
        meta = meta.drop_duplicates('particle_image_name').set_index('particle_image_name')
    rng = np.random.RandomState(seed)
    rows, skipped = [], {}
    for cls_dir in sorted(glob.glob(os.path.join(crops_dir, '*'))):
        if not os.path.isdir(cls_dir):
            continue
        cls = os.path.basename(cls_dir)
        if classes and cls not in classes:
            continue
        files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.png', '.tif')))
        files = [f for f in files if not f.startswith('.')]
        if max_per_class and len(files) > max_per_class:
            files = list(rng.choice(files, max_per_class, replace=False))
        for f in files:
            info = parse_name(f)
            um = um_per_px_for(info['cruise'], info['mag'], um_override)
            if um is None:
                skipped['no_calibration'] = skipped.get('no_calibration', 0) + 1
                continue
            bgr = cv2.imread(os.path.join(cls_dir, f))
            if bgr is None or min(bgr.shape[:2]) < 3:
                skipped['unreadable'] = skipped.get('unreadable', 0) + 1
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            illum = info['illum']
            if meta is not None and f in meta.index and 'Type' in meta.columns:
                illum = 'obl' if str(meta.loc[f, 'Type']).lower().startswith('obl') else 'bf'
            c, bg = contrast_map(rgb, illum)
            target = None
            if meta is not None and f in meta.index and 'Area' in meta.columns:
                target = int(round(float(meta.loc[f, 'Area']) / (um * um)))
            m = segment_sprite(c, target)
            if m is None or m.sum() < 4:
                skipped['no_particle'] = skipped.get('no_particle', 0) + 1
                continue
            bg_rgb = [_corners_bg(rgb[..., i].astype(np.float32)) for i in range(3)]
            if illum == 'bf':
                # absorbed fraction of the gel's transmitted light
                opac = np.clip(c / max(bg, 1.0) / OPAQUE_ABSORBANCE, 0, 1) * m
            else:
                # fixed scale (not per-sprite) so faint, translucent particles stay faint
                opac = np.clip(c / OBLIQUE_FULL_CONTRAST, 0, 1) * m
            opac = cv2.GaussianBlur(opac.astype(np.float32), (3, 3), 0) * cv2.dilate(
                m.astype(np.uint8), np.ones((3, 3), np.uint8))
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
            focus = float(cv2.Laplacian(gray, cv2.CV_32F)[m].var()) if m.sum() > 9 else 0.0
            st = shape_stats(m, um)
            uid = f'{cls}__{os.path.splitext(f)[0]}'
            rgba = np.dstack([rgb, np.clip(opac * 255, 0, 255).astype(np.uint8)])
            cv2.imwrite(os.path.join(out_dir, 'sprites', uid + '.png'),
                        cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            row = dict(uid=uid, file=f, cls_folder=cls, **{'class': cls},
                       cruise=info['cruise'], platform=info['platform'], mag=info['mag'],
                       illum=illum, micrograph=info['micrograph'],
                       sample=info['sample'], um_per_px=um,
                       bg_r=bg_rgb[0], bg_g=bg_rgb[1], bg_b=bg_rgb[2],
                       crop_w=rgb.shape[1], crop_h=rgb.shape[0], background=bg,
                       mean_opacity=float(opac[m].mean()), focus=focus, **st)
            if meta is not None and f in meta.index and 'ESD' in meta.columns:
                row['esd_um_durkin'] = float(meta.loc[f, 'ESD'])
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('no sprites built - check the folder layout (one sub-folder per class)')
    gcol = None if group_by == 'none' else group_by
    df['split'], how = assign_splits(df, gcol, seed)
    df.to_csv(os.path.join(out_dir, 'library.csv'), index=False)
    summary = dict(n=len(df), skipped=skipped, split_method=how,
                   per_class=df.groupby(['class', 'split']).size().unstack(fill_value=0).to_dict('index'))
    json.dump(summary, open(os.path.join(out_dir, 'library_summary.json'), 'w'), indent=1, default=int)
    return df, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('crops_dir'); ap.add_argument('out_dir')
    ap.add_argument('--csv', help='Durkin classified_particle_images.csv (sets exact areas, illum)')
    ap.add_argument('--group-by', default='micrograph', choices=['micrograph', 'sample', 'none'])
    ap.add_argument('--max-per-class', type=int); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    df, s = build(a.crops_dir, a.out_dir, a.csv, a.group_by, a.max_per_class, a.seed)
    print(json.dumps({k: v for k, v in s.items() if k != 'per_class'}, indent=1, default=int))
    print(df.groupby(['class', 'split']).size().unstack(fill_value=0))


if __name__ == '__main__':
    main()
