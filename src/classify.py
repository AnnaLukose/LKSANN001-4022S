# Step 9: ResNet-18 training, prior-free scores, decisions per track and scoring against the truth.
# Output: functions only (models and scores are saved by hitl.py).
# Uses: config.py, carbon.py.
import time
import numpy as np
import pandas as pd
from . import config as C

NOISE = C.NOISE
NO_CARBON = C.NO_CARBON


# labels and weights
def add_labels(T, scheme='amaral7u'):
    """Class label of every detection, and the 'particle' it belongs to."""
    T = T.copy()
    T['label'] = C.scheme_label(T.source_class.values, scheme)
    T['pkey'] = np.where(T.uid > 0, T.seq_id + ':u' + T.uid.astype(str), T.seq_id + ':d' + T.index.astype(str))
    T['w_particle'] = 1.0 / T.groupby('pkey').pkey.transform('size').values
    return T


def class_weights(T, power=0.5):
    """Weight per row = 1/(rows of its particle) x (largest class / its class)^power."""
    mass = T.groupby('label').w_particle.sum()
    cw = (mass.max() / mass) ** power
    return T.w_particle.values * cw.reindex(T.label.values).values


def effective_prior(labels, w, classes):
    """The class mix a network was trained on (after rare-enriched sequences, sampling and class weights)."""
    s = pd.Series(np.asarray(w, float)).groupby(np.asarray(labels)).sum().reindex(classes).fillna(0).values + 1e-9
    return s / s.sum()


def natural_prior(tr, classes=None):
    """Class frequencies (one count per particle) in the NATURAL training sequences only."""
    nat = tr[~tr.seq_id.str.startswith('rare')]
    nat = nat if len(nat) else tr
    p = nat.drop_duplicates('pkey').label.value_counts(normalize=True)
    if classes is not None:
        p = p.reindex(classes).fillna(0) + 1e-4
        p = p / p.sum()
    return p


def prior_free(P, prior):
    """log p(class | x) - log(training prior)."""
    return np.log(np.clip(P, 1e-6, 1)) - np.log(np.clip(prior, 1e-9, 1))[None]


def align(Z, have, classes, fill=-10.0):
    """Columns of Z (classes `have`) placed in the global class order."""
    out = np.full((len(Z), len(classes)), fill, np.float32)
    out[:, np.searchsorted(classes, have)] = Z
    return out


# ResNet-18
def _torch():
    import torch
    return torch


def device_name():
    return 'cuda' if _torch().cuda.is_available() else 'cpu'


def build_resnet(n_cls, input_size=128, pretrained=True):
    """ResNet-18 (He et al. 2016) with ImageNet weights, as in Amaral & Durkin (2025)."""
    import timm
    try:
        net = timm.create_model('resnet18', pretrained=pretrained, num_classes=n_cls, in_chans=3)
    except Exception as e:
        print('ImageNet weights unavailable, training from scratch:', str(e)[:120])
        net = timm.create_model('resnet18', pretrained=False, num_classes=n_cls, in_chans=3)
    net.input_size = input_size
    return net


def _batch(Cr, idx, net, device, train):
    """A batch of crops as a float tensor in [0, 1], resized to the network's input size."""
    torch = _torch()
    import torch.nn.functional as Fn
    x = torch.from_numpy(np.ascontiguousarray(Cr[idx][:, :3])).to(device).float() / 255.0
    if train:
        x = torch.rot90(x, np.random.randint(4), (2, 3))
        if np.random.rand() < 0.5:
            x = torch.flip(x, (3,))
        g = torch.empty(len(idx), 1, 1, 1, device=device).uniform_(0.7, 1.4)
        b = torch.empty(len(idx), 1, 1, 1, device=device).uniform_(0.8, 1.25)
        x = (x.clamp(1e-4, 1) ** g) * b
        sh = max(1, int(round(3 * C.CANVAS / 48)))
        x = torch.roll(x, tuple(int(v) for v in np.random.randint(-sh, sh + 1, 2)), (2, 3))
    if net.input_size != x.shape[-1]:
        x = Fn.interpolate(x, size=(net.input_size, net.input_size), mode='bilinear', align_corners=False)
    return x


def cnn_predict(net, Cr, device=None, bs=256, tta=True):
    """Class probabilities, averaged over the 8 rotations / flips."""
    torch = _torch()
    import torch.nn.functional as Fn
    device = device or device_name()
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(Cr), bs):
            x = _batch(Cr, np.arange(i, min(i + bs, len(Cr))), net, device, False)
            views = [(k, fl) for k in range(4) for fl in (False, True)] if tta else [(0, False)]
            acc = 0
            for k, fl in views:
                xt = torch.rot90(x, k, (2, 3))
                if fl:
                    xt = torch.flip(xt, (3,))
                with torch.autocast('cuda', enabled=device == 'cuda'):
                    acc = acc + Fn.log_softmax(net(xt).float(), 1)
            out.append((acc / len(views)).exp().cpu().numpy())
    P = np.concatenate(out) if out else np.zeros((0, 1))
    return P / P.sum(1, keepdims=True)


def _refresh_bn(net, Cr, device, n_batches=60, bs=256):
    """After weight averaging the BatchNorm statistics no longer match the weights."""
    torch = _torch()
    bns = [m for m in net.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    net.train()
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for _ in range(n_batches):
            idx = np.sort(rng.choice(len(Cr), min(bs, len(Cr)), replace=False))
            with torch.autocast('cuda', enabled=device == 'cuda'):
                net(_batch(Cr, idx, net, device, False))
    for m in bns:
        m.momentum = 0.1


def _fit(net, T, Cr, w, classes, epochs, lr, wd, bs, seed, device, max_per_epoch, sched_kind, swa_frac,
         label_smoothing, log):
    """The training loop shared by the first training and the fine-tuning."""
    torch = _torch()
    import torch.nn.functional as Fn
    torch.manual_seed(seed)
    np.random.seed(seed)
    ok = np.isin(T.label.values, classes)
    w = np.where(ok, w, 0)
    y = np.searchsorted(classes, np.where(ok, T.label.values, classes[0]))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    n_ep = min(len(T), max_per_epoch)
    steps = epochs * ((n_ep + bs - 1) // bs)
    sched = (torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps + 5, pct_start=0.1)
             if sched_kind == 'onecycle' else torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps + 1))
    use_amp = device == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    yt = torch.from_numpy(y).to(device)
    rng = np.random.default_rng(seed)
    swa_start = epochs - max(1, int(round(swa_frac * epochs))) if swa_frac else epochs
    swa, n_swa, hist = None, 0, []
    for ep in range(epochs):
        net.train()
        t0, tot = time.time(), 0.0
        idx_all = rng.choice(len(T), size=n_ep, replace=True, p=w / w.sum())
        for i in range(0, n_ep, bs):
            idx = np.sort(idx_all[i:i + bs])
            if len(idx) < 2:
                continue
            x = _batch(Cr, idx, net, device, True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=use_amp):
                loss = Fn.cross_entropy(net(x), yt[idx], label_smoothing=label_smoothing)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        if ep >= swa_start:                     # running average of the weights over the last epochs
            sd = {k: v.detach().float().clone() for k, v in net.state_dict().items()}
            if swa is None:
                swa = sd
            else:
                for k in swa:
                    if swa[k].is_floating_point():
                        swa[k] += (sd[k] - swa[k]) / (n_swa + 1)
            n_swa += 1
        rec = dict(epoch=ep, train_loss=round(tot / n_ep, 4), seconds=round(time.time() - t0, 1))
        hist.append(rec)
        if log:
            log(rec)
    if swa is not None and n_swa > 1:
        net.load_state_dict({k: v.to(net.state_dict()[k].dtype) for k, v in swa.items()})
        _refresh_bn(net, Cr, device)
    net.prior_ = effective_prior(T.label.values[ok], w[ok], classes)
    net.history_ = pd.DataFrame(hist)
    return net


def train_cnn(tr, Ctr, epochs=30, power=0.5, seed=0, input_size=128, max_per_epoch=40000, lr=1e-3, bs=128,
              swa_frac=0.3, label_smoothing=0.05, log=print, pretrained=True):
    """Train one ResNet-18 on the training detections (the 'out-of-domain' model of Amaral & Durkin)."""
    device = device_name()
    classes = np.array(sorted(tr.label.unique()))
    net = build_resnet(len(classes), input_size, pretrained).to(device)
    net = _fit(net, tr, Ctr, class_weights(tr, power), classes, epochs, lr, 0.01, bs, seed, device, max_per_epoch,
               'onecycle', swa_frac, label_smoothing, log)
    net.classes_ = classes
    return net


def save_cnn(net, path):
    _torch().save(dict(state=net.state_dict(), classes=list(net.classes_), prior=net.prior_,
                       input_size=net.input_size), path)


def load_cnn(path):
    torch = _torch()
    d = torch.load(path, map_location=device_name(), weights_only=False)
    net = build_resnet(len(d['classes']), d.get('input_size', 128), pretrained=False).to(device_name())
    net.load_state_dict(d['state'])
    net.classes_, net.prior_ = np.array(d['classes']), d['prior']
    return net


def cnn_scores(net, Cr, classes, tta=True):
    """Prior-free log scores of one network for every crop, in the global class order."""
    return align(prior_free(cnn_predict(net, Cr, tta=tta), net.prior_), net.classes_, classes)


# human in the loop
def expert_label(info, scheme='amaral7u'):
    """The simulated expert's answer for each particle (track)."""
    lab = pd.Series(C.scheme_label(info.source_class.fillna(NOISE).values, scheme), index=info.index, dtype=object)
    lab[(info.uid <= 0).values] = NOISE
    return lab


def added_rows(T_target, labels):
    """The detections (all frames) of the particles the expert checked, with the expert's labels."""
    tk = track_key(T_target)
    lab = pd.Series(tk).map({k: v for k, v in labels.items() if v is not None})
    keep = lab.notna().values
    A = T_target[keep].copy()
    A['label'] = lab[keep].values
    A['pkey'] = 'target:' + tk[keep]
    A['w_particle'] = 1.0 / A.groupby('pkey').pkey.transform('size').values
    return A, keep


def combined_weights(tr, A, power, target_weight):
    """Training weights of training + checked target particles."""
    if A is None or not len(A):
        return class_weights(tr, power), np.zeros(0)
    w = class_weights(pd.concat([tr, A], ignore_index=True), power)
    return w[:len(tr)], w[len(tr):] * target_weight


def finetune_cnn(net, tr, Ctr, A, CA, epochs=8, power=0.5, target_weight=3.0, lr=3e-4, bs=128, seed=0,
                 max_per_epoch=30000, log=None):
    """Fine-tune an out-of-domain network on training + checked target particles."""
    T = pd.concat([tr, A], ignore_index=True) if A is not None and len(A) else tr
    Cr = np.concatenate([Ctr, CA]) if A is not None and len(A) else Ctr
    w_tr, w_a = combined_weights(tr, A, power, target_weight)
    classes = net.classes_
    net = _fit(net, T, Cr, np.r_[w_tr, w_a], classes, epochs, lr, 0.01, bs, seed, device_name(), max_per_epoch,
               'cosine', 0, 0.05, log)
    net.classes_ = classes
    return net


# particles
def track_key(T):
    return (T.seq_id + ':' + T.track_id.astype(str)).values


def particle_scores(T, Z, key, K=None):
    """Mean of the per-detection scores over each particle's first K detections (in frame order)."""
    df = pd.DataFrame(Z)
    df['_k'] = np.asarray(key)
    df['_f'] = T.fidx.values
    df = df.sort_values(['_k', '_f'])
    if K:
        df = df.groupby('_k').head(K)
    return df.drop(columns='_f').groupby('_k').mean()


def particle_inputs(T, Zs, K=None):
    """Per track: the K-frame mean scores of every network."""
    key = track_key(T)
    Ps = [particle_scores(T, Z, key, K) for Z in Zs]
    idx = Ps[0].index
    return idx, [P.reindex(idx).values for P in Ps]


def particles_info(T, K=None):
    """One row per track."""
    T = T.assign(tkey=track_key(T)).sort_values(['tkey', 'fidx'])
    head = T.groupby('tkey').head(K) if K else T
    g = T.groupby('tkey')
    info = pd.DataFrame({'seq_id': g.seq_id.first(), 'start': g.fidx.min(), 'n_det': g.size(),
                         'flag': g.flag.first(), 'uid': g.uid.agg(lambda x: x.mode().iloc[0])})
    h = head.groupby('tkey')
    for c in ('esd_um', 'esd_corrected_um', 'solidity', 'aspect', 'mean_peak', 'area_px', 'area_corrected_px'):
        info[c] = h[c].median()
    first = T[T.uid > 0].groupby(['seq_id', 'uid']).fidx.min()
    ix = pd.MultiIndex.from_arrays([info.seq_id, info.uid])
    fd = first.reindex(ix).values
    by_uid = T[T.uid > 0].drop_duplicates(['seq_id', 'uid']).set_index(['seq_id', 'uid'])
    for c in ('source_class', 'carbon_true', 'esd_true', 'first_vis', 'source_uid', 'detectable'):
        info[c] = by_uid[c].reindex(ix).values
    info.loc[info.uid <= 0, 'source_class'] = NOISE
    info['carbon_true'] = info.carbon_true.fillna(0.0).astype(float)
    info['counted'] = info.start > 0
    ok = (info.uid > 0) & info.counted & (info.first_vis > 0) & (info.start == fd)
    order = info[ok].assign(_k=-info.n_det[ok]).sort_values(['seq_id', 'uid', '_k'])
    keep = ~order.duplicated(['seq_id', 'uid'])
    info['true_arrival'] = False
    info.loc[order.index[keep.values], 'true_arrival'] = True
    info['category'] = np.select([~info.counted, info.true_arrival, ok, info.uid <= 0],
                                 ['not counted (in view at frame 0)', 'arrival', 'duplicate track', 'false detection'],
                                 'track break / late start')
    return info


# decisions
def softmax(Lg):
    Lg = Lg - Lg.max(1, keepdims=True)
    E = np.exp(Lg)
    return E / E.sum(1, keepdims=True)


def size_gate(labels, esd, classes, q=(0.005, 0.995), widen=1.3, min_n=5):
    """Per class, the range of measured sizes seen in training (0.5-99.5 % quantiles, widened 30 %)."""
    df = pd.DataFrame(dict(y=np.asarray(labels), e=np.asarray(esd, float)))
    out = {}
    for c, g in df.groupby('y'):
        if c == NOISE or len(g) < min_n:
            continue
        lo, hi = np.quantile(g.e, q)
        out[c] = (float(lo / widen), float(hi * widen))
    return out


def gate_penalty(classes, esd, gate, penalty=8.0):
    pen = np.zeros((len(esd), len(classes)), np.float32)
    esd = np.asarray(esd, float)
    for j, c in enumerate(classes):
        if c in gate:
            lo, hi = gate[c]
            pen[:, j] = penalty * ((esd < lo) | (esd > hi))
    return pen


def decide_z(Z, classes, lam, log_prior, reject=0.5, pen=None):
    """Decision from prior-free scores."""
    classes = np.asarray(classes)
    lp = np.asarray(log_prior, float)
    Lg = Z + lam * (lp[None] if lp.ndim == 1 else lp)
    if pen is not None:
        Lg = Lg - pen
    P = softmax(Lg)
    real = classes != NOISE
    pred = classes[real][Lg[:, real].argmax(1)] if real.any() else classes[Lg.argmax(1)]
    if (~real).any():
        pred = np.where(P[:, ~real].max(1) >= reject, NOISE, pred)
    return pred, P


def model_votes(Zs, classes, lam, log_prior, reject):
    """Each network's own decision per particle, and the ensemble's mean posterior."""
    preds, posts = [], []
    for Z in Zs:
        p, P = decide_z(Z, classes, lam, log_prior, reject)
        preds.append(p)
        posts.append(P)
    return np.array(preds), np.mean(posts, 0)


def select_top(Zs, classes, lam, log_prior, reject, n_per_class):
    """Amaral & Durkin's ensemble voting."""
    preds, P = model_votes(Zs, classes, lam, log_prior, reject)
    agree = (preds == preds[0]).all(0)
    out, cl = {}, list(classes)
    for c in classes:
        rows = np.nonzero(agree & (preds[0] == c))[0]
        rows = sorted(rows, key=lambda r: -P[r, cl.index(c)])[:n_per_class]
        out.update({int(r): str(c) for r in rows})
    return out


def select_rare(Zs, classes, lam, log_prior, reject, verified_counts, target, n_show, shown,
                skip=(NOISE, 'aggregate')):
    """Class-targeted checks."""
    _, P = model_votes(Zs, classes, lam, log_prior, reject)
    cl, out, taken = list(classes), {}, set(shown)
    for c in classes:
        if c in skip or verified_counts.get(c, 0) >= target:
            continue
        rows = [int(r) for r in np.argsort(-P[:, cl.index(c)]) if r not in taken][:n_show]
        out.update({r: str(c) for r in rows})
        taken.update(rows)
    return out


# carbon and sizes
def fit_size_model(T):
    """Robust regression of log(true ESD) on log(measured ESD) and shape, per training particle."""
    from sklearn.linear_model import HuberRegressor
    g = T[(T.uid > 0) & T.esd_true.notna()].groupby('pkey').agg(m=('esd_um', 'median'), t=('esd_true', 'first'),
                                                                 sol=('solidity', 'median'), asp=('aspect', 'median'),
                                                                 mp=('mean_peak', 'median'))
    X = np.c_[np.log(g.m), np.log(g.m) ** 2, g.sol, np.log(g.asp), g.mp]
    r = HuberRegressor(max_iter=1000).fit(X, np.log(g.t))
    return dict(coef=r.coef_.tolist(), intercept=float(r.intercept_))


def particle_esd(info, how='measured', sm=None):
    """ESD used for carbon."""
    if how == 'calibrated':
        return info.esd_corrected_um.values
    if how == 'size_model' and sm is not None:
        X = np.c_[np.log(info.esd_um), np.log(info.esd_um) ** 2, info.solidity, np.log(info.aspect), info.mean_peak]
        return np.exp(X @ np.asarray(sm['coef']) + sm['intercept'])
    return info.esd_um.values


# scoring
def score(info, pred, carbon_pred=None, truth_arr=None, scheme='amaral7u', counted=None):
    """Metrics over particles."""
    from sklearn.metrics import f1_score
    pred = pd.Series(pred, index=info.index)
    true = pd.Series(C.scheme_label(info.source_class.values, scheme), index=info.index)
    ta = info.true_arrival.values
    ok = (pred.values == true.values)[ta]
    tl = true.values[ta]
    per = pd.Series(ok).groupby(tl).mean()
    out = dict(n_particles=int(ta.sum()), accuracy=float(ok.mean()) if ta.any() else np.nan,
               balanced_accuracy=float(per.mean()) if len(per) else np.nan)
    out['macro_f1'] = float(f1_score(tl, pred.values[ta], average='macro', labels=sorted(set(tl)))) if ta.any() else np.nan
    a, b = out['accuracy'], out['balanced_accuracy']
    out['acc_bal_hmean'] = float(2 * a * b / (a + b)) if a + b > 0 else np.nan
    flux = [c for c in per.index if c not in NO_CARBON]
    out['flux_balanced_accuracy'] = float(per[flux].mean()) if flux else np.nan
    ct = info.carbon_true.values[ta]
    out['carbon_weighted_accuracy'] = float(ct[ok].sum() / ct.sum()) if ct.sum() > 0 else np.nan
    big = info.esd_um.values[ta] >= 500
    out['accuracy_esd_ge_500um'] = float(ok[big].mean()) if big.any() else np.nan
    if carbon_pred is not None:
        cp = np.asarray(carbon_pred, float)
        out['carbon_error_found'] = float(cp[ta].sum() / ct.sum() - 1) if ct.sum() > 0 else np.nan
        if truth_arr is not None and len(truth_arr):
            tt = truth_arr[truth_arr.seq_id.isin(info.seq_id.unique())]
            tot = tt[tt.detectable.astype(bool)].carbon_true.sum()
            cnt = info.counted.values if counted is None else counted
            out['carbon_error_all'] = float(cp[cnt].sum() / tot - 1) if tot > 0 else np.nan
    return out, per


def merged_metrics(info, pred, exclude=(), scheme='amaral7u', merge=None):
    """Accuracy and balanced accuracy over real arrivals NOT in `exclude`."""
    merge = merge or {}
    keep = info.true_arrival.values & ~info.index.isin(list(exclude))
    true = pd.Series(C.scheme_label(info.source_class.values, scheme)[keep])
    pr = pd.Series(np.asarray(pred)[keep])
    if merge:
        true, pr = true.map(lambda x: merge.get(x, x)), pr.map(lambda x: merge.get(x, x))
    ok = true.values == pr.values
    per = pd.Series(ok).groupby(true.values).mean()
    acc, bal = (float(ok.mean()), float(per.mean())) if len(ok) else (np.nan, np.nan)
    return dict(accuracy=acc, balanced_accuracy=bal, acc_bal_hmean=2 * acc * bal / max(acc + bal, 1e-9)), per, \
        true.value_counts()


def precision_recall(info, pred, exclude=(), scheme='amaral7u'):
    """Precision and recall per class over the real arrivals not in `exclude` (Amaral & Durkin Fig. 7)."""
    keep = info.true_arrival.values & ~info.index.isin(list(exclude))
    true = C.scheme_label(info.source_class.values, scheme)[keep]
    pr = np.asarray(pred)[keep]
    rows = []
    for c in sorted(set(true) | set(pr)):
        tp = int(((pr == c) & (true == c)).sum())
        rows.append(dict(cls=c, n_true=int((true == c).sum()), n_pred=int((pr == c).sum()),
                         precision=tp / max((pr == c).sum(), 1) if (pr == c).any() else np.nan,
                         recall=tp / max((true == c).sum(), 1) if (true == c).any() else np.nan))
    return pd.DataFrame(rows)


def confusion(info, pred, scheme='amaral7u', exclude=()):
    keep = info.true_arrival.values & ~info.index.isin(list(exclude))
    true = C.scheme_label(info.source_class.values, scheme)[keep]
    return pd.crosstab(pd.Series(true, name='true'), pd.Series(np.asarray(pred)[keep], name='predicted'))


def carbon_budget(info, carbon_pred, truth_arr, counted=None):
    """Where the counted carbon comes from."""
    cnt = info.counted.values if counted is None else counted
    d = pd.DataFrame({'category': info.category.values, 'pred': np.asarray(carbon_pred, float),
                      'true': info.carbon_true.values})[cnt]
    tab = d.groupby('category').agg(tracks=('pred', 'size'), predicted_ug=('pred', 'sum'),
                                    true_of_its_particle_ug=('true', 'sum'))
    tab[['predicted_ug', 'true_of_its_particle_ug']] *= 1000
    if truth_arr is not None and len(truth_arr):
        tt = truth_arr[truth_arr.seq_id.isin(info.seq_id.unique()) & truth_arr.detectable.astype(bool)]
        found = set(zip(info.seq_id[info.true_arrival], info.uid[info.true_arrival]))
        miss = tt[[(s, u) not in found for s, u in zip(tt.seq_id, tt.uid)]]
        tab.loc['missed arrivals (no track)'] = [len(miss), 0.0, 1000 * miss.carbon_true.sum()]
        tab.loc['TRUE TOTAL (detectable arrivals)'] = [len(tt), np.nan, 1000 * tt.carbon_true.sum()]
        tab.loc['PIPELINE TOTAL (all counted tracks)'] = [len(d), 1000 * d.pred.sum(), np.nan]
    return tab
