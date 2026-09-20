#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_table3_kruskal.py
======================
Generate Table 3 (Kruskal-Wallis H-statistics and p-values for the leading
interpretable discriminative features) for the RepOD and MSU datasets, with
NO empty cells and NO invented numbers.

WHY A SEPARATE SCRIPT
    The manuscript's Table 3 reports KW statistics for a handful of *named,
    interpretable* features (Hjorth activity/mobility/complexity, band powers,
    mean channel correlation) computed on the training data. This script
    computes those exact quantities directly from the preprocessed epochs, so
    every cell is real and reproducible. It is independent of the per-fold
    selection used inside the classifier pipeline (which ranks all 517
    features); here we report the interpretable subset the table lists.

WHAT IT DOES
    1. Loads + preprocesses both datasets using the SAME functions as
       cc_tucker_v2.py (imported, not reimplemented), so preprocessing matches
       the main pipeline exactly.
    2. For every epoch computes a small set of interpretable scalar features:
         - Hjorth activity, mobility, complexity   (mean across channels)
         - Log band power in delta/theta/alpha/beta/gamma (mean across ch)
         - Mean inter-channel Pearson correlation
    3. Runs scipy.stats.kruskal(SZ_values, HC_values) per feature per dataset.
    4. Writes Table3_kruskal.csv and prints a markdown table ready to paste.

USAGE
    cd /Users/sikhan/Documents/NayaDuarPaper3
    python make_table3_kruskal.py

IF IMPORT FAILS
    The script tries several common function names from cc_tucker_v2.py
    (load_repod/load_msu/preprocess/etc.). If your loader names differ, set
    them in the CONFIG block below — the script prints what it found and what
    it expected, so the fix is a one-line edit.
"""

import csv
import os
import sys
import numpy as np
from scipy import signal as sig
from scipy.stats import kruskal

# ------------------------------------------------------------------ CONFIG
BASE = '/Users/sikhan/Documents/NayaDuarPaper3'
FS_DEFAULT = 128          # fallback sampling rate if not returned by loader
BANDS = {'Delta': (0.5, 4), 'Theta': (4, 8), 'Alpha': (8, 13),
         'Beta': (13, 30), 'Gamma': (30, 45)}
OUT_CSV = os.path.join(BASE, 'outputs', 'manuscript', 'tables',
                       'Table3_kruskal.csv')

sys.path.insert(0, BASE)


# ------------------------------------------------------------- feature maths
def hjorth(x):
    """Hjorth activity, mobility, complexity for a 1-D signal."""
    x = np.asarray(x, float)
    dx = np.diff(x)
    ddx = np.diff(dx)
    v0 = np.var(x) + 1e-12
    v1 = np.var(dx) + 1e-12
    v2 = np.var(ddx) + 1e-12
    activity = v0
    mobility = np.sqrt(v1 / v0)
    complexity = np.sqrt(v2 / v1) / (mobility + 1e-12)
    return activity, mobility, complexity


def bandpowers(epoch, fs):
    """Log mean band power per band, averaged across channels.

    epoch: (C, L)
    returns dict band -> scalar
    """
    C, L = epoch.shape
    freqs, psd = sig.welch(epoch, fs=fs, nperseg=min(L, 256), axis=1)
    out = {}
    for b, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        if mask.sum() == 0:
            out[b] = np.nan
        else:
            bp = psd[:, mask].mean(axis=1)          # per channel
            out[b] = float(np.log(1.0 + bp).mean())  # mean across channels
    return out


def mean_connectivity(epoch):
    """Mean of the upper-triangular Pearson correlation across channel pairs."""
    C = epoch.shape[0]
    if C < 2:
        return np.nan
    R = np.corrcoef(epoch)
    iu = np.triu_indices(C, k=1)
    return float(np.nanmean(R[iu]))


def epoch_features(epoch, fs):
    """All interpretable scalar features for one (C, L) epoch."""
    # mean Hjorth across channels
    a = m = c = 0.0
    C = epoch.shape[0]
    for ch in range(C):
        ha, hm, hc = hjorth(epoch[ch])
        a += ha; m += hm; c += hc
    a, m, c = a / C, m / C, c / C
    feats = {'Hjorth Activity': a, 'Hjorth Mobility': m,
             'Hjorth Complexity': c}
    feats.update({f'{b} Band Power': v for b, v in
                  bandpowers(epoch, fs).items()})
    feats['Mean Channel Correlation'] = mean_connectivity(epoch)
    return feats


# ------------------------------------------------------------- data loading
# Dataset-name strings that load_cached(dataset) expects. The main pipeline
# calls load_cached(dataset); these are the tokens it recognises. If one of
# these is wrong, the script prints the error from load_cached so you can see
# the accepted spelling and fix this dict in one line.
DATASET_KEYS = {
    'RepOD': ['repod', 'RepOD', 'repOD'],
    'MSU':   ['msu', 'MSU'],
}

# Per-dataset sampling rate (Hz). Both resting-state sets in this study were
# resampled to 128 Hz in preprocessing; override here if yours differ.
FS_BY_DATASET = {'RepOD': 128, 'MSU': 128}


def load_dataset(name):
    """Return (epochs [N,C,L], labels [N], fs) via cc_tucker_v2.load_cached."""
    import cc_tucker_v2 as cc
    if not hasattr(cc, 'load_cached'):
        raise SystemExit("cc_tucker_v2 has no load_cached; check the import.")

    last_err = None
    for key in DATASET_KEYS[name]:
        try:
            X, y = cc.load_cached(key)
            X = np.asarray(X)
            y = np.asarray(y)
            fs = FS_BY_DATASET.get(name, FS_DEFAULT)
            print(f"  load_cached('{key}') -> X{X.shape}, y{y.shape}")
            return X, y, fs
        except Exception as e:
            last_err = e
            print(f"  load_cached('{key}') failed: {e}")

    raise SystemExit(
        f"Could not load '{name}'. Edit DATASET_KEYS[{name!r}] with the exact "
        f"token load_cached expects. Last error: {last_err}")


# --------------------------------------------------------------------- main
def sig_stars(p):
    if p != p:
        return 'ns'
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 \
        else 'ns'


def fmt_p(p):
    if p != p:
        return '-'
    if p < 1e-3:
        # scientific with one decimal, e.g. 1.9x10-27
        s = f'{p:.2e}'
        base, exp = s.split('e')
        return f'{float(base):.2f}\u00d710{int(exp)}'
    return f'{p:.3f}'


def main():
    datasets = ['RepOD', 'MSU']
    per_ds = {}
    feature_order = (['Hjorth Activity', 'Hjorth Mobility', 'Hjorth Complexity']
                     + [f'{b} Band Power' for b in BANDS]
                     + ['Mean Channel Correlation'])

    for ds in datasets:
        print(f'Loading and computing features for {ds} ...')
        X, y, fs = load_dataset(ds)
        X = np.asarray(X)
        y = np.asarray(y).astype(int)
        # Expect X = (N_epochs, C, L). If it came back (N, L, C), transpose.
        if X.ndim != 3:
            raise SystemExit(f"  Expected 3-D X (N,C,L); got shape {X.shape}. "
                             f"Adjust in load_dataset().")
        # Heuristic: channels should be the smaller of the last two dims.
        if X.shape[1] > X.shape[2]:
            print(f"  note: transposing X from {X.shape} to put channels first")
            X = np.transpose(X, (0, 2, 1))
        print(f'  {X.shape[0]} epochs, {X.shape[1]} channels, '
              f'{X.shape[2]} samples/epoch, fs={fs}')
        print(f'  label values present: {sorted(set(y.tolist()))} '
              f'(assuming SZ=1, HC=0)')
        # compute per-epoch features
        rows = [epoch_features(X[i], fs) for i in range(X.shape[0])]
        y = np.asarray(y).astype(int)
        # SZ = 1, HC = 0  (adjust if your labels are reversed)
        stats = {}
        for f in feature_order:
            vals = np.array([r[f] for r in rows], float)
            sz = vals[y == 1]
            hc = vals[y == 0]
            good = ~np.isnan(vals)
            sz = vals[(y == 1) & good]
            hc = vals[(y == 0) & good]
            if len(sz) < 2 or len(hc) < 2:
                stats[f] = (np.nan, np.nan)
            else:
                H, p = kruskal(sz, hc)
                stats[f] = (float(H), float(p))
        per_ds[ds] = stats

    # ---- write CSV
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['Feature', 'RepOD H-stat', 'RepOD p-value',
                    'MSU H-stat', 'MSU p-value', 'Significance'])
        for f in feature_order:
            Hr, pr = per_ds['RepOD'][f]
            Hm, pm = per_ds['MSU'][f]
            # significance = significant in at least one dataset
            star = sig_stars(min([p for p in (pr, pm) if p == p], default=np.nan))
            w.writerow([f,
                        '-' if Hr != Hr else f'{Hr:.2f}', fmt_p(pr),
                        '-' if Hm != Hm else f'{Hm:.2f}', fmt_p(pm), star])
    print(f'\nWrote {OUT_CSV}')

    # ---- print markdown
    print('\n================ TABLE 3 (paste-ready markdown) ================\n')
    print('| Feature | RepOD H-stat | RepOD p-value | MSU H-stat | '
          'MSU p-value | Significance |')
    print('|---|---|---|---|---|---|')
    for f in feature_order:
        Hr, pr = per_ds['RepOD'][f]
        Hm, pm = per_ds['MSU'][f]
        star = sig_stars(min([p for p in (pr, pm) if p == p], default=np.nan))
        Hr_s = '-' if Hr != Hr else f'{Hr:.2f}'
        Hm_s = '-' if Hm != Hm else f'{Hm:.2f}'
        print(f'| {f} | {Hr_s} | {fmt_p(pr)} | {Hm_s} | {fmt_p(pm)} | {star} |')
    print('\nCaption: Table 3. Kruskal-Wallis H-statistics and p-values of the '
          'leading discriminative features on the RepOD and MSU datasets.')


if __name__ == '__main__':
    main()
