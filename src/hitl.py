# Step 9: human-in-the-loop runs (OOD, +verify, +rareboost) with a ResNet-18 ensemble.
# Output: cls/models/*.pt, cls/scores/*.npy and the expert selections.
# Uses: classify.py, tables.py, carbon.py.
import json, os, time
import numpy as np
import pandas as pd
from . import config as C
from . import classify as cl
from . import tables as tb
from .carbon import carbon_array

DEFAULTS = dict(scheme='amaral7u', seeds=(0, 1, 2), ood_epochs=30, ft_epochs=8, input_size=128, class_power=0.5,
                target_weight=3.0, max_per_epoch=40000, ft_max_per_epoch=30000, domains=('val', 'test'),
                budgets=(25, 50, 100), minboost=30, rare_shown=30, lam=0.25, carbon_lam=1.0, K=3, reject=0.5,
                pretrained=True)


class Hitl:
    """Holds the data and settings of one classification run."""

    def __init__(self, out_dir, **kw):
        self.out = out_dir
        self.s = dict(DEFAULTS, **kw)
        self.d = os.path.join(out_dir, 'cls')
        for sub in ('models', 'scores'):
            os.makedirs(os.path.join(self.d, sub), exist_ok=True)
        self._cache = {}
        self.tr, self.Ctr = self.tables('train', crops=True)
        self.classes = np.array(sorted(self.tr.label.unique()))
        self.prior = cl.natural_prior(self.tr, self.classes).values
        self.logp = np.log(self.prior)
        p = self.tr[self.tr.uid > 0].groupby('pkey').agg(y=('label', 'first'), e=('esd_um', 'median'))
        self.gate = cl.size_gate(p.y.values, p.e.values, self.classes)        # used for carbon decisions only
        self.size_model = cl.fit_size_model(self.tr)
        self.truth_arr = tb.truth_arrivals(out_dir, tuple(self.s['domains']))

    # data
    def tables(self, split, crops=False):
        key = (split, crops)
        if key not in self._cache:
            T, Cr = tb.load_tables(self.out, (split,), crops=crops)
            self._cache[key] = (cl.add_labels(T, self.s['scheme']), Cr)
        return self._cache[key]

    def info(self, dom):
        if ('info', dom) not in self._cache:
            self._cache[('info', dom)] = cl.particles_info(self.tables(dom)[0], self.s['K'])
        return self._cache[('info', dom)]

    def zpath(self, dom, stage, seed):
        return os.path.join(self.d, 'scores', f'{dom}_{stage}_s{seed}.npy')

    def particles(self, dom, stage):
        """(track keys, [particle scores per network]) of one domain at one stage."""
        T = self.tables(dom)[0]
        return cl.particle_inputs(T, [np.load(self.zpath(dom, stage, s)) for s in self.s['seeds']], self.s['K'])

    def has(self, dom, stage):
        return all(os.path.exists(self.zpath(dom, stage, s)) for s in self.s['seeds'])

    # out-of-domain models
    def ood_path(self, seed):
        return os.path.join(self.d, 'models', f'OOD_s{seed}.pt')

    def train_ood(self, log=print):
        """Train (or reuse) one ResNet-18 per seed on the training detections, then score every domain."""
        for seed in self.s['seeds']:
            f = self.ood_path(seed)
            if not os.path.exists(f):
                t0 = time.time()
                net = cl.train_cnn(self.tr, self.Ctr, self.s['ood_epochs'], self.s['class_power'], seed,
                                   self.s['input_size'], self.s['max_per_epoch'], pretrained=self.s['pretrained'],
                                   log=lambda r: log(f'seed {seed}', r) if r['epoch'] % 5 == 4 or r['epoch'] == 0 else None)
                cl.save_cnn(net, f)
                net.history_.to_csv(f.replace('.pt', '_history.csv'), index=False)
                log(f'seed {seed}: trained in {time.time() - t0:.0f} s')
        for dom in self.s['domains']:
            Cr = self.tables(dom, crops=True)[1]
            for seed in self.s['seeds']:
                if not os.path.exists(self.zpath(dom, 'OOD', seed)):
                    np.save(self.zpath(dom, 'OOD', seed), cl.cnn_scores(cl.load_cnn(self.ood_path(seed)), Cr, self.classes))
        log('out-of-domain scores done')

    # expert checks
    def sel_path(self, dom, b):
        return os.path.join(self.d, f'{dom}_b{b}_selection.json')

    def selection(self, dom, b):
        return json.load(open(self.sel_path(dom, b)))

    def adapt(self, dom, stage, labels, log=print):
        """Fine-tune every network from its OOD weights on training + the checked particles."""
        T, Cr = self.tables(dom, crops=True)
        A, keep = cl.added_rows(T, labels)
        CA = Cr[keep]
        log(f'  {dom} {stage}: {A.pkey.nunique()} checked particles', A.drop_duplicates('pkey').label.value_counts().to_dict())
        for seed in self.s['seeds']:
            if os.path.exists(self.zpath(dom, stage, seed)):
                continue
            t0 = time.time()
            net = cl.finetune_cnn(cl.load_cnn(self.ood_path(seed)), self.tr, self.Ctr, A, CA, self.s['ft_epochs'],
                                  self.s['class_power'], self.s['target_weight'], seed=seed,
                                  max_per_epoch=self.s['ft_max_per_epoch'])
            np.save(self.zpath(dom, stage, seed), cl.cnn_scores(net, Cr, self.classes))
            log(f'    seed {seed}: {time.time() - t0:.0f} s')
            del net
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass

    def run_budgets(self, log=print):
        """+verify and +rareboost for every domain and budget (resumable)."""
        s = self.s
        for dom in s['domains']:
            expert = cl.expert_label(self.info(dom), s['scheme'])
            for b in s['budgets']:
                f = self.sel_path(dom, b)
                if not os.path.exists(f):
                    idx, Zs = self.particles(dom, 'OOD')
                    top = cl.select_top(Zs, self.classes, s['lam'], self.logp, s['reject'], b)
                    json.dump(dict(top={idx[r]: c for r, c in top.items()}), open(f, 'w'))
                sel = self.selection(dom, b)
                verified = {k: expert.get(k) for k in sel['top']}
                wrong = sum(verified[k] != c for k, c in sel['top'].items())
                log(f'== {dom}, budget {b}: expert checked {len(verified)}, corrected {wrong} '
                    f'({100 * wrong / max(len(verified), 1):.0f} %)')
                self.adapt(dom, f'b{b}_+verify', verified, log)
                if 'rare' not in sel:
                    idx, Zs = self.particles(dom, f'b{b}_+verify')
                    counts = pd.Series(list(verified.values())).value_counts().to_dict()
                    shown = set(np.nonzero(pd.Index(idx).isin(list(verified)))[0])
                    rare = cl.select_rare(Zs, self.classes, s['lam'], self.logp, s['reject'], counts, s['minboost'],
                                          s['rare_shown'], shown)
                    sel['rare'] = {idx[r]: c for r, c in rare.items()}
                    json.dump(sel, open(f, 'w'))
                hits = pd.DataFrame([(c, expert.get(k)) for k, c in sel['rare'].items()], columns=['shown_for', 'expert'])
                if len(hits):
                    log('  +rareboost, shown / found per class:',
                        {str(c): f'{(g.expert == c).sum()}/{len(g)}' for c, g in hits.groupby('shown_for')})
                self.adapt(dom, f'b{b}_+rareboost', {**verified, **{k: expert.get(k) for k in sel['rare']}}, log)
        log('all budgets done')

    # results
    def stages(self, dom):
        """[(stage key, budget, stage name, set of checked track keys)] for every stage with saved scores."""
        out = [('OOD', 0, 'OOD', set())]
        for b in self.s['budgets']:
            if not os.path.exists(self.sel_path(dom, b)):
                continue
            sel = self.selection(dom, b)
            out += [(f'b{b}_+verify', b, '+verify', set(sel['top'])),
                    (f'b{b}_+rareboost', b, '+rareboost', set(sel['top']) | set(sel.get('rare', {})))]
        return [r for r in out if self.has(dom, r[0])]

    def common_unseen(self, dom):
        """Track keys the expert saw under ANY budget (excluded from classification scores)."""
        seen = set()
        for b in self.s['budgets']:
            if os.path.exists(self.sel_path(dom, b)):
                sel = self.selection(dom, b)
                seen |= set(sel['top']) | set(sel.get('rare', {}))
        return seen

    def stage_particles(self, dom, stage, checked, size='measured', gate=True, lam=None):
        """One row per track of one domain at one stage."""
        s = self.s
        idx, Zs = self.particles(dom, stage)
        Z = np.mean(Zs, 0)
        info = self.info(dom).reindex(idx).copy()
        pred, P = cl.decide_z(Z, self.classes, s['lam'] if lam is None else lam, self.logp, s['reject'])
        pen = cl.gate_penalty(self.classes, info.esd_um.values, self.gate) if gate else None
        cpred, _ = cl.decide_z(Z, self.classes, s['carbon_lam'], self.logp, s['reject'], pen)
        expert = cl.expert_label(info, s['scheme'])
        chk = info.index.isin(list(checked)) & expert.notna().values
        cpred = np.where(chk, expert.values, cpred)
        esd = cl.particle_esd(info, size, self.size_model)
        info['true_label'] = C.scheme_label(info.source_class.values, s['scheme'])
        info['pred'] = pred
        info['p_max'] = P.max(1)
        info['carbon_class'] = cpred
        info['checked'] = chk
        info['esd_for_carbon'] = esd
        info['carbon_pred'] = carbon_array(cpred, esd)
        return info

    def stage_metrics(self, info, exclude, counted=None):
        """Classification metrics on tracks not in `exclude`."""
        s = self.s
        seen = info.index.isin(list(exclude))
        r_un, per = cl.score(info[~seen], info.pred.values[~seen], None, None, s['scheme'])
        r_all, _ = cl.score(info, info.pred.values, info.carbon_pred.values, self.truth_arr, s['scheme'], counted)
        g5, _, _ = cl.merged_metrics(info, info.pred.values, exclude, s['scheme'], C.GELCAM5)
        out = {k: r_un[k] for k in ('accuracy', 'balanced_accuracy', 'acc_bal_hmean', 'macro_f1',
                                     'flux_balanced_accuracy', 'carbon_weighted_accuracy', 'accuracy_esd_ge_500um',
                                     'n_particles')}
        out.update(gelcam5_accuracy=g5['accuracy'], gelcam5_balanced_accuracy=g5['balanced_accuracy'],
                   carbon_error_found=r_all['carbon_error_found'], carbon_error_all=r_all.get('carbon_error_all'),
                   n_checked=int(info.checked.sum()), n_tracks=len(info),
                   checked_pct=float(100 * info.checked.sum() / max(len(info), 1)))
        return out, per

    def results(self, size='measured', gate=True):
        """Every domain x stage x budget."""
        rows, recalls = [], []
        for dom in self.s['domains']:
            common = self.common_unseen(dom)
            for key, b, stage, checked in self.stages(dom):
                info = self.stage_particles(dom, key, checked, size, gate)
                r, per = self.stage_metrics(info, common)
                rows.append(dict(domain=dom, budget=b, stage=stage, key=key, **r))
                recalls.append(per.rename(f'{dom} {key}'))
        return pd.DataFrame(rows), pd.concat(recalls, axis=1) if recalls else pd.DataFrame()

    def choose_carbon_setting(self, budget, dom='val'):
        """Carbon setting with the smallest carbon error of the found particles, chosen on validation only."""
        st = [x for x in self.stages(dom) if x[1] == budget and x[2] == '+rareboost'] or self.stages(dom)[-1:]
        key, _, _, checked = st[0]
        rows = []
        for size in ('measured', 'calibrated', 'size_model'):
            for gate in (True, False):
                info = self.stage_particles(dom, key, checked, size, gate)
                r, _ = cl.score(info, info.pred.values, info.carbon_pred.values, self.truth_arr, self.s['scheme'])
                rows.append(dict(size=size, gate=gate, stage=key, carbon_error_found=r['carbon_error_found'],
                                 carbon_error_all=r.get('carbon_error_all')))
        R = pd.DataFrame(rows)
        best = R.loc[R.carbon_error_found.abs().idxmin()]
        return dict(size=best['size'], gate=bool(best['gate'])), R
