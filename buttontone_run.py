#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
buttontone_run.py — held-out ERP generalization test for CC-Tucker.

Consumes the cache from buttontone_erp.py and runs the SAME class-conditional
Tucker + whitening + KW-selection + 4-classifier + soft-vote pipeline used on
MSU/RepOD, but on the button-tone ERP tensor. NO tuning: ranks, lambda and
select-mode are the MSU-locked values. This is what makes it a legitimate
held-out generalization test rather than a third fitted dataset.

KEY DIFFERENCES FROM THE EEG PATH (all forced by the data, not by choice)
  * Unit of observation = one subject (ERP is already trial-averaged), so
    LOSO here is genuine per-subject hold-out and there is no epoch pooling.
  * Basis = TEMPORAL. ERPs are time-locked, so mode-2 (time) carries real,
    consistent structure across subjects -- the temporal basis is correct here,
    unlike resting EEG where it annihilated the signal.
  * Tucker mode-3 (subjects) is small (~72 train), so ranks are capped by n.
  * With one sample per subject, GroupKFold == StratifiedKFold; we also run
    LOSO. Metrics are pooled across held-out subjects (real confusion matrix).

The decisive comparison, exactly as on the EEG datasets:
    B1 = ERP scalar features only        (raw ERP baseline)
    T1 = CC-Tucker features only         (does the subspace model see anything?)
    C3 = ERP features + CC-Tucker        (does Tucker ADD over the baseline?)

USAGE
    cd /Users/sikhan/Documents/NayaDuarPaper3
    python buttontone_run.py                      # LOSO + 10-fold, all configs
    python buttontone_run.py --ranks 6,15,6 --lam 0.05
"""
import argparse
import importlib.util
import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler

BASE = '/Users/sikhan/Documents/NayaDuarPaper3'
RES = os.path.join(BASE, 'outputs', 'results_v2')
CORE = os.path.join(BASE, 'cc_tucker_v2.py')   # reuse the audited core


def load_core():
    """Import the verified functions from cc_tucker_v2 without running its CLI."""
    spec = importlib.util.spec_from_file_location('core', CORE)
    m = importlib.util.module_from_spec(spec)
    sys.argv = ['core']                         # stop its argparse from firing
    spec.loader.exec_module(m)
    return m


def tucker_erp_features(Xtr_list, ytr, Xte, ranks, lam, core):
    """
    Class-conditional Tucker on ERP slabs (temporal basis).
    Xtr_list : list of (C, T) per-subject condition-1 ERP matrices (train)
    Xte      : (n_te, C, T) test ERPs
    Returns (F_tr, F_te) Tucker feature matrices.

    Mirrors cc_tucker_v2.tucker_features but for the one-slab-per-subject ERP
    case: each subject IS a slab, no epoch pooling, mode-2 is ERP time.
    """
    C = Xtr_list[0].shape[0]
    W = _erp_whitener(Xtr_list, reg=0.05)

    labels = np.asarray(ytr)
    models = {}
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        if len(idx) < 2:
            idx = np.arange(len(labels))
        T = np.stack([W @ Xtr_list[i].astype(np.float64) for i in idx], axis=2)
        r = [min(ranks[0], C), min(ranks[1], T.shape[1]), min(ranks[2], len(idx))]
        f, core_g, err = core.tucker_hosvd_sparse(T, r, lam=lam)
        models[cls] = dict(U1=f[0], U2=f[1])
    models['W'] = W

    F_tr = _erp_tuck_feat(np.stack([x for x in Xtr_list]), models)
    F_te = _erp_tuck_feat(Xte, models)
    return F_tr, F_te


def _erp_whitener(Xlist, reg=0.05):
    C = Xlist[0].shape[0]
    S = np.zeros((C, C))
    for X in Xlist:
        E = X.astype(np.float64)
        E = E - E.mean(axis=1, keepdims=True)
        S += E @ E.T
    S /= max(len(Xlist), 1)
    S = (1 - reg) * S + reg * np.trace(S) / C * np.eye(C)
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, 1e-10)
    return V @ np.diag(w ** -0.5) @ V.T


def _erp_tuck_feat(X, models):
    """Per-subject ERP Tucker features (temporal basis)."""
    W = models['W']
    m0, m1 = models[0], models[1]
    C = W.shape[0]
    I = np.eye(C)
    out = []
    for E in X.astype(np.float64):
        Ew = W @ E                                  # (C, T)
        row = []
        rec = {}
        for cls, mm in ((0, m0), (1, m1)):
            U1, U2 = mm['U1'], mm['U2']
            Sp = U1 @ U1.T
            Ehat = Sp @ Ew
            # temporal-domain reconstruction via both factors
            proj = U1.T @ Ew @ U2                    # (R1, R2)
            recon = U1 @ proj @ U2.T
            num = np.linalg.norm(Ew - recon)
            den = np.linalg.norm(Ew) + 1e-9
            rec[cls] = num / den
            # energy features
            row += [np.linalg.norm(Ehat, axis=1).mean(),   # mean spatial energy
                    np.linalg.norm(proj, axis=1).sum(),     # subspace energy
                    float((Ehat ** 2).mean())]
        # residual after removing SZ subspace
        res = Ew - (m1['U1'] @ m1['U1'].T) @ Ew
        row.append(float((res ** 2).mean()))
        e0, e1 = rec[0], rec[1]
        row += [e0, e1, e1 - e0, e1 / (e0 + 1e-9), np.log(e1 / (e0 + 1e-9) + 1e-9)]
        out.append(row)
    return np.array(out, dtype=np.float64)


CONFIGS = {
    'B1': ('erp',),                 # raw ERP scalar features (baseline)
    'T1': ('tucker',),              # CC-Tucker only
    'C3': ('erp', 'tucker'),        # ERP + CC-Tucker  <-- the decisive one
}


def run(core, args):
    X1 = np.load(os.path.join(RES, 'bt_X.npy'))       # (n, C, T) cond-1
    y = np.load(os.path.join(RES, 'bt_y.npy'))
    erpf = np.load(os.path.join(RES, 'bt_erpfeat.npy'))
    meta = json.load(open(os.path.join(RES, 'bt_meta.json')))
    n = len(y)
    print(f'button-tone ERP cache: {n} subjects '
          f'(HC={int((y==0).sum())} SZ={int((y==1).sum())}), '
          f'tensor {X1.shape}, {erpf.shape[1]} ERP features')
    print(f'ranks={args.ranks}  lam={args.lam}  (MSU-locked, no tuning)\n')

    protocols = {'loso': LeaveOneOut(),
                 'gkf': StratifiedKFold(10, shuffle=True, random_state=42)}

    results = {}
    for cfg, groups in CONFIGS.items():
        results[cfg] = {}
        for pname, splitter in protocols.items():
            store = {m: dict(true=[], pred=[], proba=[]) for m in core.METHODS}
            for tr, te in splitter.split(np.zeros(n), y):
                ytr, yte = y[tr], y[te]
                blocks_tr, blocks_te = [], []

                if 'erp' in groups:
                    blocks_tr.append(erpf[tr])
                    blocks_te.append(erpf[te])
                if 'tucker' in groups:
                    Ftr, Fte = tucker_erp_features(
                        [X1[i] for i in tr], ytr, X1[te],
                        args.ranks, args.lam, core)
                    blocks_tr.append(Ftr)
                    blocks_te.append(Fte)

                Ftr = np.hstack(blocks_tr)
                Fte = np.hstack(blocks_te)

                k = min(args.topk, Ftr.shape[1])
                sel = core.kw_select(Ftr, ytr, k)
                Ftr, Fte = Ftr[:, sel], Fte[:, sel]

                sc = StandardScaler()
                Ftr = sc.fit_transform(Ftr)
                Fte = sc.transform(Fte)

                fitted = {}
                for mname, clf in core.make_clfs().items():
                    try:
                        clf.fit(Ftr, ytr)
                        pred = clf.predict(Fte)
                        try:
                            pr = clf.predict_proba(Fte)[:, 1]
                        except Exception:
                            pr = np.full(len(pred), np.nan)
                        fitted[mname] = clf
                        store[mname]['true'].extend(yte.tolist())
                        store[mname]['pred'].extend(pred.tolist())
                        store[mname]['proba'].extend(pr.tolist())
                    except Exception as exc:
                        print(f'    [warn] {mname}: {exc}')
                pe = core.ensemble_proba(fitted, Fte)
                if pe is not None:
                    store['ENSEMBLE']['true'].extend(yte.tolist())
                    store['ENSEMBLE']['pred'].extend((pe >= 0.5).astype(int).tolist())
                    store['ENSEMBLE']['proba'].extend(pe.tolist())

            summ = {}
            for m, s in store.items():
                if not s['true']:
                    continue
                pr = np.array(s['proba'], float)
                pr = None if np.all(np.isnan(pr)) else pr
                summ[m] = core.full_metrics(s['true'], s['pred'], pr)
            results[cfg][pname] = summ

            e = summ.get('ENSEMBLE', {})
            print(f'  {cfg:3s} {pname.upper():4s}  '
                  f'acc={e.get("acc",0):6.2f}  sen={e.get("sen",0):6.2f}  '
                  f'spe={e.get("spe",0):6.2f}  MCC={e.get("mcc",0):+.4f}  '
                  f'AUC={e.get("auc") if e.get("auc") is None else round(e["auc"],4)}')

    # decisive contrast
    print('\n  ' + '=' * 60)
    for p in ('loso', 'gkf'):
        b = results['B1'][p]['ENSEMBLE']['acc']
        c = results['C3'][p]['ENSEMBLE']['acc']
        t = results['T1'][p]['ENSEMBLE']['acc']
        print(f'  {p.upper():4s}  B1(ERP)={b:.2f}  T1(Tucker)={t:.2f}  '
              f'C3(both)={c:.2f}   C3-B1={c-b:+.2f}')
    print('  ' + '=' * 60)

    out = os.path.join(RES, f'bt_result_r{"_".join(map(str,args.ranks))}'
                       f'_lam{args.lam}.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=1, default=float)
    print(f'\n  saved -> {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ranks', default='8,20,6')
    ap.add_argument('--lam', type=float, default=0.05)
    ap.add_argument('--topk', type=int, default=80)
    args = ap.parse_args()
    args.ranks = tuple(int(v) for v in args.ranks.split(','))
    core = load_core()
    run(core, args)


if __name__ == '__main__':
    main()
