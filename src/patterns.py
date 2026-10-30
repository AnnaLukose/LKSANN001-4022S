# Step 3: class mix of each real trap sample, used as the scene pattern of a sequence.
# Output: lib/patterns.json.
# Used by: render.py.
import argparse, json, os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def gap_statistic(X, kmax=8, B=20, seed=0):
    rng = np.random.RandomState(seed)
    lo, hi = X.min(0), X.max(0)
    res = []
    for k in range(1, min(kmax, len(X) - 1) + 1):
        km = KMeans(k, n_init=10, random_state=seed).fit(X)
        wk = np.log(km.inertia_ + 1e-12)
        ref = []
        for _ in range(B):
            R = rng.uniform(lo, hi, size=X.shape)
            ref.append(np.log(KMeans(k, n_init=3, random_state=seed).fit(R).inertia_ + 1e-12))
        ref = np.array(ref)
        res.append(dict(k=k, gap=ref.mean() - wk, s=ref.std() * np.sqrt(1 + 1.0 / B)))
    best = res[-1]['k']
    for i in range(len(res) - 1):
        if res[i]['gap'] >= res[i + 1]['gap'] - res[i + 1]['s']:
            best = res[i]['k']
            break
    return best, res


def counts_from_archive(crops_dir):
    """Class counts per sample from the FULL archive (file names only), not the capped library."""
    from .config import parse_name
    rows = []
    for cls in os.listdir(crops_dir):
        d = os.path.join(crops_dir, cls)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.startswith('.'):
                continue
            i = parse_name(f)
            rows.append(dict(sample=i['sample'], mag=i['mag'], **{'class': cls}))
    return pd.DataFrame(rows)


def build(lib_dir, classes=None, kmax=8, min_samples=8, mag_weights=None, seed=0, n_fallback=6,
          crops_dir=None):
    lib = counts_from_archive(crops_dir) if crops_dir else pd.read_csv(os.path.join(lib_dir, 'library.csv'))
    if classes:
        lib = lib[lib['class'].isin(classes)]
    w = lib['mag'].map(lambda m: (mag_weights or {}).get(int(m), 1.0)).fillna(1.0)
    lib = lib.assign(w=w)
    comp = lib.pivot_table(index='sample', columns='class', values='w', aggfunc='sum', fill_value=0)
    comp = comp[comp.sum(axis=1) >= 20]
    comp_frac = comp.div(comp.sum(axis=1), axis=0)
    cls = list(comp_frac.columns) if len(comp_frac) else sorted(lib['class'].unique())
    glob_comp = lib.groupby('class')['w'].sum()
    glob_comp = (glob_comp / glob_comp.sum()).reindex(cls).fillna(0)
    out = dict(classes=cls, n_samples=int(len(comp_frac)))
    if len(comp_frac) >= min_samples:
        X = comp_frac.values
        pca = PCA(n_components=min(len(cls), len(X))).fit(X)
        ncomp = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.86) + 1)
        Z = pca.transform(X)[:, :ncomp]
        k, gaps = gap_statistic(Z, kmax=kmax, seed=seed)
        pats = []
        if k == 1:
            # no defensible clusters: every real trap sample becomes its own pattern
            for smp in comp_frac.index:
                pats.append(dict(name=smp, composition=comp_frac.loc[smp].round(4).to_dict(),
                                 spread={}, samples=[smp], n_particles=int(comp.loc[smp].sum())))
            out.update(source='per_sample (gap statistic chose K=1)', pca_components=ncomp,
                       pca_explained=list(np.round(pca.explained_variance_ratio_[:ncomp], 4)),
                       gap=gaps, patterns=pats)
            json.dump(out, open(os.path.join(lib_dir, 'patterns.json'), 'w'), indent=1, default=float)
            return out
        km = KMeans(k, n_init=20, random_state=seed).fit(Z)
        for j in range(k):
            members = comp_frac.index[km.labels_ == j]
            mean = comp_frac.loc[members].mean()
            pats.append(dict(name=f'P{j + 1}', composition=mean.round(4).to_dict(),
                             spread=comp_frac.loc[members].std().fillna(0).round(4).to_dict(),
                             samples=list(members)))
        out.update(source='archive_clusters', pca_components=ncomp,
                   pca_explained=list(np.round(pca.explained_variance_ratio_[:ncomp], 4)),
                   gap=gaps, patterns=pats)
    else:
        rng = np.random.RandomState(seed)
        pats = []
        for j in range(n_fallback):
            alpha = 5.0 * glob_comp.values + 0.05
            pats.append(dict(name=f'D{j + 1}', composition=dict(zip(cls, np.round(rng.dirichlet(alpha), 4))),
                             spread={}, samples=[]))
        out.update(source='dirichlet_fallback (too few samples for clustering)',
                   library_composition=glob_comp.round(4).to_dict(), patterns=pats)
    json.dump(out, open(os.path.join(lib_dir, 'patterns.json'), 'w'), indent=1, default=float)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('lib_dir'); ap.add_argument('--classes', nargs='*')
    ap.add_argument('--kmax', type=int, default=8); ap.add_argument('--min-samples', type=int, default=8)
    ap.add_argument('--mag-weights', help='e.g. 7=1,20=6.5,40=30')
    ap.add_argument('--crops-dir', help='count compositions from the full archive (recommended)')
    a = ap.parse_args()
    mw = None
    if a.mag_weights:
        mw = {int(k): float(v) for k, v in (p.split('=') for p in a.mag_weights.split(','))}
    out = build(a.lib_dir, a.classes, a.kmax, a.min_samples, mw, crops_dir=a.crops_dir)
    print(out['source'], '|', len(out['patterns']), 'patterns from', out['n_samples'], 'samples')


if __name__ == '__main__':
    main()
