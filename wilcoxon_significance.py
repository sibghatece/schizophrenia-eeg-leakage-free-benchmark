#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wilcoxon_significance.py — paired Wilcoxon signed-rank tests on per-fold
accuracies from the LOCKED result JSONs. Reads only; writes a CSV table.

WHAT IT TESTS
    For each dataset x protocol, the per-fold ENSEMBLE accuracy of the proposed
    configuration (C3 = raw + CC-Tucker) is compared against each baseline using
    a two-sided Wilcoxon signed-rank test on the PAIRED per-fold values. The
    folds are identical across configs (same CV split, same random_state), so
    fold i is the same held-out subject(s) in both arrays -- the pairing is
    valid. Comparisons run:
        C3 vs B1  (raw baseline)          <- the headline "does Tucker help"
        C3 vs A2  (connectivity only)     <- the RepOD-relevant contrast
        (MSU/RepOD only; button-tone lacks per_fold, see note below)

    Effect size reported alongside p: median paired difference, and the
    matched-pairs rank-biserial correlation r = (W+ - W-)/(W+ + W-).

HONEST CAVEATS (printed and worth stating in the paper)
    * LOSO gives many folds (28/84) -> adequately powered.
    * GKF gives only 10 folds -> low power; reported as corroborating only.
    * Ties (folds where both configs score identically, common at the subject
      level) are dropped by the signed-rank test, further reducing effective n.
    * Button-tone (bt_result_*.json) does NOT store per-fold arrays, so it
      cannot be tested here without re-running buttontone_run.py with per-fold
      saving enabled. It is listed as "n/a (no per-fold data)".

USAGE
    cd /Users/sikhan/Documents/NayaDuarPaper3
    python wilcoxon_significance.py
"""
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

BASE = '/Users/sikhan/Documents/NayaDuarPaper3'
RES = os.path.join(BASE, 'outputs', 'results_v2')
OUT = os.path.join(BASE, 'outputs', 'manuscript', 'tables')

SRC = {
    'MSU': 'v3_msu_spectral_ablation_b5_pooled_ceil_20260725_1032.json',
    'RepOD': 'v3_repod_spectral_ablation_b5_pooled_ceil_20260725_0351.json',
}
BT = 'bt_result_r8_20_6_lam0.05.json'

# comparisons: proposed vs each baseline
COMPARISONS = [('C3', 'B1', 'raw baseline'),
               ('C3', 'A2', 'connectivity-only')]


def rank_biserial(x, y):
    """Matched-pairs rank-biserial effect size for paired Wilcoxon."""
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0, 0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    tot = w_pos + w_neg
    return (w_pos - w_neg) / tot if tot else 0.0, len(d)


def test_pair(a, b):
    """Two-sided Wilcoxon signed-rank; returns dict of stats."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    diff = a - b
    n_nonzero = int((diff != 0).sum())
    out = dict(n_folds=n, n_effective=n_nonzero,
               mean_a=a.mean(), mean_b=b.mean(),
               median_diff=float(np.median(diff)),
               mean_diff=float(diff.mean()),
               wins=int((diff > 0).sum()), losses=int((diff < 0).sum()),
               ties=int((diff == 0).sum()))
    if n_nonzero < 1:
        out.update(stat=float('nan'), p=1.0, rbc=0.0, note='all ties')
        return out
    try:
        stat, p = wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
        rbc, _ = rank_biserial(a, b)
        out.update(stat=float(stat), p=float(p), rbc=float(rbc), note='')
    except Exception as e:
        out.update(stat=float('nan'), p=float('nan'), rbc=0.0, note=str(e))
    return out


def stars(p):
    if p != p:  # nan
        return ''
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    print(f'{"Dataset":7s} {"Proto":5s} {"Comparison":22s} {"nEff":>4s} '
          f'{"W/L/T":>10s} {"medΔ":>6s} {"p":>8s} {"sig":>4s} {"rbc":>6s}')
    print('-' * 82)

    for ds, fn in SRC.items():
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            print(f'  [MISSING] {fn}')
            continue
        d = json.load(open(p))
        for proto in ('loso', 'gkf'):
            for prop, base, desc in COMPARISONS:
                if prop not in d or base not in d:
                    continue
                a = d[prop][proto]['ENSEMBLE'].get('per_fold')
                b = d[base][proto]['ENSEMBLE'].get('per_fold')
                if not a or not b or len(a) != len(b):
                    continue
                r = test_pair(a, b)
                sig = stars(r['p'])
                wlt = f"{r['wins']}/{r['losses']}/{r['ties']}"
                print(f'{ds:7s} {proto.upper():5s} '
                      f'{prop+" vs "+base+" ("+desc+")":22.22s} '
                      f'{r["n_effective"]:4d} {wlt:>10s} '
                      f'{r["median_diff"]:+6.2f} {r["p"]:8.4f} {sig:>4s} '
                      f'{r["rbc"]:+6.3f}')
                rows.append([ds, proto.upper(), f'{prop} vs {base}', desc,
                             r['n_folds'], r['n_effective'],
                             f"{r['mean_a']:.2f}", f"{r['mean_b']:.2f}",
                             f"{r['median_diff']:+.2f}", f"{r['mean_diff']:+.2f}",
                             wlt,
                             '' if r['stat'] != r['stat'] else f"{r['stat']:.1f}",
                             f"{r['p']:.4f}", sig, f"{r['rbc']:+.3f}",
                             r['note']])

    # button-tone note
    bt_path = os.path.join(RES, BT)
    if os.path.exists(bt_path):
        d = json.load(open(bt_path))
        has_pf = 'per_fold' in d.get('C3', {}).get('loso', {}).get('ENSEMBLE', {})
        if not has_pf:
            for proto in ('LOSO', 'GKF'):
                rows.append(['Button-tone', proto, 'C3 vs B1', 'ERP baseline',
                             '', '', '', '', '', '', '', '', '', 'n/a',
                             'no per-fold data saved'])
            print('\n  Button-tone: no per-fold arrays saved -> Wilcoxon n/a '
                  '(re-run buttontone_run.py with per-fold saving to enable)')

    header = ['Dataset', 'Protocol', 'Comparison', 'Baseline', 'n_folds',
              'n_effective', 'mean_proposed', 'mean_baseline', 'median_diff',
              'mean_diff', 'W/L/T', 'W_stat', 'p_value', 'significance',
              'rank_biserial', 'note']
    out_csv = os.path.join(OUT, 'Table11_wilcoxon.csv')
    with open(out_csv, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows)
    print(f'\nWrote {out_csv}  ({len(rows)} comparisons)')
    print('\nInterpretation guide:')
    print('  p<0.05 (*) => the proposed config differs significantly from the')
    print('  baseline across folds. LOSO is well powered; GKF (10 folds) is')
    print('  corroborating only. Ties reduce effective n at subject level.')


if __name__ == '__main__':
    main()
