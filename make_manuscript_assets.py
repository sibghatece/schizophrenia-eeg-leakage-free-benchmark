#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_manuscript_assets.py — generate all manuscript tables (.csv) and figures
(.svg) from the LOCKED v3 result JSONs. Reads only; writes into
outputs/manuscript/tables/ and outputs/manuscript/figures/.

CANONICAL SOURCES (edit SRC below if you rename anything)
  RepOD/MSU ablation (all configs, pooled)  -> *_ablation_b5_pooled_ceil_*
  RepOD/MSU headline main (C3, pooled b5)   -> *_main_b5_pooled_d3pool_*
  spectral-vs-temporal basis A/B (patched)  -> *_main_*_ab2_*
  lambda sweep (MSU)                        -> *_lamsweep_*
  rank sweep (MSU)                          -> *_ranksweep_*
  button-tone ERP result                    -> bt_result_r8_20_6_lam0.05.json

OUTPUT
  tables/  Table01..Table10  (.csv)
  figures/ Fig01..Fig20      (.svg, grayscale/print-safe)

USAGE
  cd /Users/sikhan/Documents/NayaDuarPaper3
  python make_manuscript_assets.py
"""
import json
import os
import sys
import csv
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------------------------------------------------------------
BASE = '/Users/sikhan/Documents/NayaDuarPaper3'
RES = os.path.join(BASE, 'outputs', 'results_v2')
OUT = os.path.join(BASE, 'outputs', 'manuscript')
TAB = os.path.join(OUT, 'tables')
FIG = os.path.join(OUT, 'figures')

SRC = {
    'repod_abl': 'v3_repod_spectral_ablation_b5_pooled_ceil_20260725_0351.json',
    'msu_abl':   'v3_msu_spectral_ablation_b5_pooled_ceil_20260725_1032.json',
    'repod_main': 'v3_repod_spectral_ablation_b5_pooled_ceil_20260725_0351.json',
    'msu_main':   'v3_msu_spectral_ablation_b5_pooled_ceil_20260725_1032.json',
    'repod_spec_ab2': 'v2_repod_spectral_main_ab2_20260724_1308.json',
    'repod_temp_ab2': 'v2_repod_temporal_main_ab2_20260724_1111.json',
    'msu_spec_ab2':   'v2_msu_spectral_main_ab2_20260724_1308.json',
    'msu_temp_ab2':   'v2_msu_temporal_main_ab2_20260724_1111.json',
    'lam':  'v3_msu_spectral_lamsweep_b1_pooled_lam_20260724_2156.json',
    'rank': 'v3_msu_spectral_ranksweep_b1_pooled_rank_20260725_0032.json',
    'bt':   'bt_result_r8_20_6_lam0.05.json',
}

METHODS = ['SVM', 'RF', 'KNN', 'LDA', 'ENSEMBLE']
CFG_ORDER = ['A1', 'A2', 'A3', 'B1', 'C1', 'C2', 'C3', 'D1', 'D2', 'D3', 'E1',
             'E2', 'E3', 'E4', 'E5']
CFG_DESC = {
    'A1': 'Hjorth+PSR', 'A2': 'Connectivity (broadband)', 'A3': 'BandPower+Conn',
    'B1': 'Raw baseline', 'C1': 'TuckerErr+Conn', 'C2': 'CC-Tucker only',
    'C3': 'Raw+CC-Tucker', 'D1': 'Raw+BandConn', 'D2': 'BandConn only',
    'D3': 'Raw+BandConn+Tucker', 'E1': 'Conn+BandConn', 'E2': 'BP+BandConn',
    'E3': 'BP+Conn+BandConn', 'E4': 'Hjorth+BP+Conn+BandConn',
    'E5': 'Conn+BandConn+Tucker',
}

# leakage-free external benchmarks (Racz & Csukly medRxiv; honest button-tone CNN)
SOTA = {
    'RepOD': [('ResNet-18 CNN', 58.06), ('Boosted trees', 64.73),
              ('SVM (subject)', 56.38), ('REVE foundation', 73.07)],
    'MSU': [('ResNet-18 CNN', 81.30), ('Boosted trees', 82.27),
            ('SVM (subject)', 80.77), ('REVE foundation', 83.47)],
    'Button-tone': [('Single-trial CNN', 62.65)],
}

# print-safe grayscale palette
GRAYS = ['#111111', '#555555', '#888888', '#aaaaaa', '#cccccc']
plt.rcParams.update({
    'font.size': 9, 'font.family': 'DejaVu Sans', 'axes.linewidth': 0.8,
    'axes.edgecolor': '#333333', 'savefig.bbox': 'tight', 'figure.dpi': 150,
    'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5,
})


def load(key):
    p = os.path.join(RES, SRC[key])
    if not os.path.exists(p):
        print(f'  [MISSING] {SRC[key]} — table/figures using it will be skipped')
        return None
    return json.load(open(p))


def ens(d, proto, cfg=None):
    """ENSEMBLE metrics dict for a protocol (and config if ablation)."""
    node = d[cfg][proto] if cfg else d[proto]
    return node['ENSEMBLE']


def sv(d, proto, cfg=None):
    node = d[cfg][proto] if cfg else d[proto]
    return node['ENSEMBLE'].get('subject_level', {})


def writerows(fn, header, rows):
    path = os.path.join(TAB, fn)
    with open(path, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows)
    print(f'  wrote {fn}  ({len(rows)} rows)')


def savefig(fig, fn):
    fig.savefig(os.path.join(FIG, fn), format='svg')
    plt.close(fig)
    print(f'  wrote {fn}')


# ============================================================================
# TABLES
# ============================================================================

def table01():
    """Dataset characteristics."""
    rows = [
        ['RepOD (Warsaw/IPN)', 'Resting-state', 28, '14 SZ / 14 HC', 16, 250,
         '4 s, 50% overlap', 14396, 'LOSO, Group-10-Fold'],
        ['MSU (Moscow)', 'Resting-state', 84, '45 SZ / 39 HC', 16, 128,
         '4 s, 50% overlap', 2436, 'LOSO, Group-10-Fold'],
        ['Button-tone (Ford/Roach)', 'Task (auditory ERP)', 81, '49 SZ / 32 HC',
         9, 1024, 'trial-averaged ERP', 81, 'LOSO, Strat-10-Fold'],
    ]
    writerows('Table01_datasets.csv',
              ['Dataset', 'Paradigm', 'Subjects', 'SZ/HC', 'Channels',
               'Fs (Hz)', 'Epoching', 'Observations', 'Cross-validation'], rows)


def _main_rows(d, label):
    rows = []
    for proto in ('loso', 'gkf'):
        if proto not in (d if 'loso' in d else d.get('C3', {})):
            pass
        node = d['C3'] if 'C3' in d else d
        if proto not in node:
            continue
        for m in METHODS:
            r = node[proto][m]
            slv = node[proto][m].get('subject_level', {})
            rows.append([label, proto.upper(), m,
                         f"{r['acc']:.2f}", f"{r['bal_acc']:.2f}",
                         f"{r['sen']:.2f}", f"{r['spe']:.2f}",
                         f"{r['ppv']:.2f}", f"{r['npv']:.2f}",
                         f"{r['f1']:.4f}", f"{r['mcc']:.4f}",
                         f"{r['kappa']:.4f}",
                         '' if r['auc'] is None else f"{r['auc']:.4f}",
                         f"{slv.get('acc', float('nan')):.2f}" if slv else ''])
    return rows


def table_main(key, label, fn):
    d = load(key)
    if d is None:
        return
    header = ['Dataset', 'Protocol', 'Classifier', 'Acc', 'BalAcc', 'Sen',
              'Spe', 'PPV', 'NPV', 'F1', 'MCC', 'Kappa', 'AUC', 'SubjAcc']
    writerows(fn, header, _main_rows(d, label))


def table04_buttontone():
    d = load('bt')
    if d is None:
        return
    rows = []
    for cfg in ('B1', 'T1', 'C3'):
        for proto in ('loso', 'gkf'):
            for m in METHODS:
                r = d[cfg][proto][m]
                rows.append([{'B1': 'ERP-only', 'T1': 'Tucker-only',
                              'C3': 'ERP+Tucker'}[cfg], proto.upper(), m,
                             f"{r['acc']:.2f}", f"{r['bal_acc']:.2f}",
                             f"{r['sen']:.2f}", f"{r['spe']:.2f}",
                             f"{r['ppv']:.2f}", f"{r['npv']:.2f}",
                             f"{r['f1']:.4f}", f"{r['mcc']:.4f}",
                             '' if r['auc'] is None else f"{r['auc']:.4f}"])
    writerows('Table04_buttontone_results.csv',
              ['Config', 'Protocol', 'Classifier', 'Acc', 'BalAcc', 'Sen',
               'Spe', 'PPV', 'NPV', 'F1', 'MCC', 'AUC'], rows)


def table_ablation(key, label, fn):
    d = load(key)
    if d is None:
        return
    rows = []
    for cfg in CFG_ORDER:
        if cfg not in d:
            continue
        row = [cfg, CFG_DESC[cfg]]
        for proto in ('loso', 'gkf'):
            if proto in d[cfg]:
                e = ens(d, proto, cfg)
                s = sv(d, proto, cfg)
                row += [f"{e['acc']:.2f}", f"{e['mcc']:.4f}",
                        f"{s.get('acc', float('nan')):.2f}" if s else '']
            else:
                row += ['', '', '']
        rows.append(row)
    # append C3-B1 / D3-B1 contrast note as trailing rows
    writerows(fn, ['Config', 'Feature blocks',
                   'LOSO Acc', 'LOSO MCC', 'LOSO SubjAcc',
                   'GKF Acc', 'GKF MCC', 'GKF SubjAcc'], rows)


def table07_sota():
    """Leakage-free SOTA comparison."""
    ours = {
        'RepOD': ('Connectivity (this work)', 76.57, 78.57),
        'MSU': ('CC-Tucker (this work, GKF)', 85.59, 88.10),
        'Button-tone': ('ERP+CC-Tucker (this work)', 65.43, None),
    }
    rows = []
    for ds in ('RepOD', 'MSU', 'Button-tone'):
        for name, acc in SOTA[ds]:
            rows.append([ds, name, f"{acc:.2f}", '', 'leakage-free (prior)'])
        nm, ep, sb = ours[ds]
        rows.append([ds, nm, f"{ep:.2f}",
                     '' if sb is None else f"{sb:.2f}", 'THIS WORK'])
    writerows('Table07_sota_leakagefree.csv',
              ['Dataset', 'Method', 'Epoch Acc', 'Subject Acc', 'Source'], rows)


def table08_lambda():
    d = load('lam')
    if d is None:
        return
    rows = []
    for k in sorted(d.keys(), key=lambda x: float(x.replace('lam', ''))):
        e = d[k]['ENSEMBLE']
        rows.append([k.replace('lam', ''), f"{e['acc']:.2f}",
                     f"{e['mcc']:.4f}",
                     '' if e['auc'] is None else f"{e['auc']:.4f}",
                     f"{e.get('subject_level', {}).get('acc', float('nan')):.2f}"])
    writerows('Table08_lambda_sweep.csv',
              ['Lambda', 'Acc', 'MCC', 'AUC', 'SubjAcc'], rows)


def table09_rank():
    d = load('rank')
    if d is None:
        return
    rows = []
    for k in sorted(d.keys()):
        e = d[k]['ENSEMBLE']
        r1, r2 = k.replace('R', '').split('_')
        rows.append([f"({r1},{r2},6)", f"{e['acc']:.2f}", f"{e['mcc']:.4f}",
                     '' if e['auc'] is None else f"{e['auc']:.4f}"])
    writerows('Table09_rank_sweep.csv',
              ['Ranks (R1,R2,R3)', 'Acc', 'MCC', 'AUC'], rows)


def table10_basis():
    """Spectral vs temporal basis on both datasets."""
    rows = []
    pairs = [('RepOD', 'repod_spec_ab2', 'repod_temp_ab2'),
             ('MSU', 'msu_spec_ab2', 'msu_temp_ab2')]
    for label, sk, tk in pairs:
        ds_s, ds_t = load(sk), load(tk)
        if ds_s is None or ds_t is None:
            continue
        for proto in ('loso', 'gkf'):
            es = ens(ds_s, proto)
            et = ens(ds_t, proto)
            rows.append([label, proto.upper(),
                         f"{et['acc']:.2f}", f"{es['acc']:.2f}",
                         f"{es['acc'] - et['acc']:+.2f}",
                         f"{et['mcc']:.4f}", f"{es['mcc']:.4f}"])
    writerows('Table10_basis_comparison.csv',
              ['Dataset', 'Protocol', 'Temporal Acc', 'Spectral Acc',
               'Gain', 'Temporal MCC', 'Spectral MCC'], rows)


# ============================================================================
# FIGURES
# ============================================================================

def roc_from_store(node):
    """ROC curve from stored proba+true. Returns (fpr, tpr, auc) or None."""
    if 'proba' not in node or 'true' not in node:
        return None
    y = np.array(node['true'])
    p = np.array(node['proba'], float)
    if np.all(np.isnan(p)) or len(np.unique(y)) < 2:
        return None
    thr = np.unique(p[~np.isnan(p)])
    thr = np.sort(thr)[::-1]
    tpr, fpr = [0], [0]
    P = (y == 1).sum()
    N = (y == 0).sum()
    for t in thr:
        pred = (p >= t)
        tpr.append(((pred) & (y == 1)).sum() / max(P, 1))
        fpr.append(((pred) & (y == 0)).sum() / max(N, 1))
    tpr.append(1)
    fpr.append(1)
    fpr, tpr = np.array(fpr), np.array(tpr)
    auc = np.trapezoid(tpr, fpr)
    return fpr, tpr, auc


def fig_perfold(key, label, fn):
    d = load(key)
    if d is None:
        return
    node = d['C3'] if 'C3' in d else d
    if 'loso' not in node:
        return
    pf = np.array(node['loso']['ENSEMBLE'].get('per_fold', []))
    if len(pf) == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.hist(pf, bins=20, color=GRAYS[2], edgecolor=GRAYS[0], linewidth=0.6)
    ax.axvline(pf.mean(), color=GRAYS[0], linestyle='--', linewidth=1.2,
               label=f'mean = {pf.mean():.1f}%')
    ax.set_xlabel('Per-fold accuracy (%)')
    ax.set_ylabel('Number of folds')
    ax.set_title(f'{label} — LOSO per-subject accuracy')
    ax.legend(frameon=False)
    savefig(fig, fn)


def fig_roc(key, label, fn, is_bt=False):
    d = load(key)
    if d is None:
        return
    fig, ax = plt.subplots(figsize=(4, 4))
    styles = ['-', '--', '-.', ':', '-']
    plotted = False
    node = d['C3'] if ('C3' in d and not is_bt) else (d['C3'] if is_bt else d)
    for i, proto in enumerate(('loso', 'gkf')):
        if proto not in node:
            continue
        r = roc_from_store(node[proto]['ENSEMBLE'])
        if r is None:
            # fall back to reported AUC as a diagonal-anchored marker
            auc = node[proto]['ENSEMBLE'].get('auc')
            if auc:
                ax.plot([0, 1], [0, 1], color='#dddddd', lw=0.8)
                ax.scatter([], [], label=f'{proto.upper()} AUC={auc:.3f}')
                plotted = True
            continue
        fpr, tpr, auc = r
        ax.plot(fpr, tpr, styles[i], color=GRAYS[i], lw=1.4,
                label=f'{proto.upper()} (AUC={auc:.3f})')
        plotted = True
    ax.plot([0, 1], [0, 1], color='#bbbbbb', lw=0.8, linestyle=':')
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(f'{label} — ROC (ENSEMBLE)')
    if plotted:
        ax.legend(frameon=False, loc='lower right', fontsize=8)
    savefig(fig, fn)


def fig_confusion(key, label, fn, is_bt=False):
    d = load(key)
    if d is None:
        return
    node = d['C3'] if 'C3' in d else d
    e = node['loso']['ENSEMBLE']
    cm = np.array([[e['tn'], e['fp']], [e['fn'], e['tp']]])
    fig, ax = plt.subplots(figsize=(3.2, 3))
    ax.imshow(cm, cmap='Greys', vmin=0, vmax=cm.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=12)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['HC', 'SZ']); ax.set_yticklabels(['HC', 'SZ'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'{label} — confusion (LOSO)')
    ax.grid(False)
    savefig(fig, fn)


def fig_c3_b1_generalization(fn):
    data = []
    for label, key in [('MSU', 'msu_abl'), ('RepOD', 'repod_abl')]:
        d = load(key)
        if d is None:
            continue
        for proto in ('loso', 'gkf'):
            b1 = ens(d, proto, 'B1')['acc']
            c3 = ens(d, proto, 'C3')['acc']
            data.append((f'{label}\n{proto.upper()}', b1, c3))
    bt = load('bt')
    if bt is not None:
        for proto in ('loso', 'gkf'):
            data.append((f'Button-tone\n{proto.upper()}',
                         bt['B1'][proto]['ENSEMBLE']['acc'],
                         bt['C3'][proto]['ENSEMBLE']['acc']))
    if not data:
        return
    labels = [x[0] for x in data]
    b1 = [x[1] for x in data]
    c3 = [x[2] for x in data]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar(x - 0.2, b1, 0.38, label='Baseline (B1)', color=GRAYS[3],
           edgecolor=GRAYS[0], linewidth=0.6)
    ax.bar(x + 0.2, c3, 0.38, label='+ CC-Tucker (C3)', color=GRAYS[1],
           edgecolor=GRAYS[0], linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Accuracy (%)'); ax.set_ylim(50, 90)
    ax.set_title('CC-Tucker contribution across datasets and protocols')
    ax.legend(frameon=False)
    savefig(fig, fn)


def fig_basis(fn):
    pairs = [('RepOD', 'repod_spec_ab2', 'repod_temp_ab2'),
             ('MSU', 'msu_spec_ab2', 'msu_temp_ab2')]
    labels, temp, spec = [], [], []
    for label, sk, tk in pairs:
        ds_s, ds_t = load(sk), load(tk)
        if ds_s is None or ds_t is None:
            continue
        for proto in ('loso', 'gkf'):
            labels.append(f'{label}\n{proto.upper()}')
            temp.append(ens(ds_t, proto)['acc'])
            spec.append(ens(ds_s, proto)['acc'])
    if not labels:
        return
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(x - 0.2, temp, 0.38, label='Temporal basis', color=GRAYS[3],
           edgecolor=GRAYS[0], linewidth=0.6)
    ax.bar(x + 0.2, spec, 0.38, label='Spectral basis', color=GRAYS[1],
           edgecolor=GRAYS[0], linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Spectral vs temporal subject basis')
    ax.legend(frameon=False)
    savefig(fig, fn)


def fig_lambda(fn):
    d = load('lam')
    if d is None:
        return
    ks = sorted(d.keys(), key=lambda x: float(x.replace('lam', '')))
    lam = [float(k.replace('lam', '')) for k in ks]
    acc = [d[k]['ENSEMBLE']['acc'] for k in ks]
    auc = [d[k]['ENSEMBLE']['auc'] for k in ks]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(lam, acc, 'o-', color=GRAYS[0], label='Accuracy')
    ax.set_xscale('symlog', linthresh=1e-3)
    ax.set_xlabel('Sparsity λ'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Robustness to sparsity λ (MSU)')
    ax.set_ylim(min(acc) - 1, max(acc) + 1)
    ax.legend(frameon=False)
    savefig(fig, fn)


def fig_rank(fn):
    d = load('rank')
    if d is None:
        return
    import itertools
    r1s = [4, 8, 12]; r2s = [10, 20, 30]
    grid = np.full((len(r1s), len(r2s)), np.nan)
    for i, r1 in enumerate(r1s):
        for j, r2 in enumerate(r2s):
            k = f'R{r1}_{r2}'
            if k in d:
                grid[i, j] = d[k]['ENSEMBLE']['acc']
    fig, ax = plt.subplots(figsize=(4, 3.4))
    im = ax.imshow(grid, cmap='Greys_r', aspect='auto')
    for i in range(len(r1s)):
        for j in range(len(r2s)):
            ax.text(j, i, f'{grid[i,j]:.1f}', ha='center', va='center',
                    color='black' if grid[i, j] > np.nanmean(grid) else 'white')
    ax.set_xticks(range(len(r2s))); ax.set_xticklabels(r2s)
    ax.set_yticks(range(len(r1s))); ax.set_yticklabels(r1s)
    ax.set_xlabel('R2 (spectral rank)'); ax.set_ylabel('R1 (spatial rank)')
    ax.set_title('Rank sweep accuracy (MSU)')
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, label='Acc (%)')
    savefig(fig, fn)


def fig_ablation_bars(key, label, fn):
    d = load(key)
    if d is None:
        return
    cfgs = [c for c in CFG_ORDER if c in d]
    acc = [ens(d, 'loso', c)['acc'] for c in cfgs]
    colors = [GRAYS[1] if c in ('C3', 'D3') else
              (GRAYS[0] if c == 'B1' else GRAYS[3]) for c in cfgs]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(len(cfgs)), acc, color=colors, edgecolor=GRAYS[0], linewidth=0.6)
    b1 = ens(d, 'loso', 'B1')['acc']
    ax.axhline(b1, color=GRAYS[0], linestyle='--', linewidth=1,
               label=f'baseline B1 = {b1:.1f}%')
    ax.set_xticks(range(len(cfgs))); ax.set_xticklabels(cfgs, fontsize=8)
    ax.set_ylabel('LOSO Accuracy (%)')
    ax.set_ylim(50, max(acc) + 3)
    ax.set_title(f'{label} — ablation (LOSO)')
    ax.legend(frameon=False)
    savefig(fig, fn)


def fig_subj_vs_epoch(fn):
    pts = []
    for label, key in [('MSU', 'msu_abl'), ('RepOD', 'repod_abl')]:
        d = load(key)
        if d is None:
            continue
        for proto in ('loso', 'gkf'):
            e = ens(d, proto, 'C3')
            s = sv(d, proto, 'C3')
            if s:
                pts.append((e['acc'], s['acc'], f'{label} {proto.upper()}'))
    bt = load('bt')
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(4.2, 4))
    for ep, sb, lab in pts:
        ax.scatter(ep, sb, color=GRAYS[0], s=40)
        ax.annotate(lab, (ep, sb), fontsize=7, xytext=(4, 4),
                    textcoords='offset points')
    lims = [50, 95]
    ax.plot(lims, lims, ':', color='#bbbbbb')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('Epoch-level accuracy (%)')
    ax.set_ylabel('Subject-level accuracy (%)')
    ax.set_title('Subject- vs epoch-level (majority vote lifts)')
    savefig(fig, fn)


def fig_1oversqrtN(fn):
    """The mode-3 diagnostic: amplitude surviving temporal mean ~ 1/sqrt(N)."""
    N = np.array([10, 29, 100, 200, 500])
    surv = 1.0 / np.sqrt(N)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(N, surv, 'o-', color=GRAYS[0])
    ax.axvline(29, color=GRAYS[2], linestyle='--', linewidth=1)
    ax.axvline(500, color=GRAYS[2], linestyle='--', linewidth=1)
    ax.annotate('MSU (~29 ep)', (29, 1/np.sqrt(29)), fontsize=8,
                xytext=(10, 10), textcoords='offset points')
    ax.annotate('RepOD (~500 ep)', (500, 1/np.sqrt(500)), fontsize=8,
                xytext=(-20, 20), textcoords='offset points')
    ax.set_xlabel('Epochs averaged per subject (N)')
    ax.set_ylabel('Fraction of amplitude surviving')
    ax.set_title('Why temporal averaging annihilates resting EEG')
    savefig(fig, fn)


def fig_sota_bars(fn):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    ours = {'RepOD': 76.57, 'MSU': 85.59, 'Button-tone': 65.43}
    for ax, ds in zip(axes, ('RepOD', 'MSU', 'Button-tone')):
        names = [n for n, _ in SOTA[ds]] + ['This work']
        vals = [v for _, v in SOTA[ds]] + [ours[ds]]
        cols = [GRAYS[3]] * len(SOTA[ds]) + [GRAYS[0]]
        ax.barh(range(len(names)), vals, color=cols, edgecolor=GRAYS[0],
                linewidth=0.6)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
        ax.set_xlabel('Acc (%)'); ax.set_title(ds, fontsize=9)
        ax.set_xlim(50, 90)
    fig.suptitle('Leakage-free: this work vs prior benchmarks', fontsize=10)
    savefig(fig, fn)


def fig_classifier_compare(key, label, fn):
    d = load(key)
    if d is None:
        return
    node = d['C3'] if 'C3' in d else d
    if 'loso' not in node:
        return
    accs = [node['loso'][m]['acc'] for m in METHODS]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(range(len(METHODS)), accs, color=GRAYS[2], edgecolor=GRAYS[0],
           linewidth=0.6)
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS)
    ax.set_ylabel('LOSO Accuracy (%)')
    ax.set_ylim(min(accs) - 3, max(accs) + 3)
    ax.set_title(f'{label} — classifier comparison')
    savefig(fig, fn)


# ============================================================================
def main():
    os.makedirs(TAB, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    print('TABLES')
    table01()
    table_main('repod_main', 'RepOD', 'Table02_repod_results.csv')
    table_main('msu_main', 'MSU', 'Table03_msu_results.csv')
    table04_buttontone()
    table_ablation('repod_abl', 'RepOD', 'Table05_repod_ablation.csv')
    table_ablation('msu_abl', 'MSU', 'Table06_msu_ablation.csv')
    table07_sota()
    table08_lambda()
    table09_rank()
    table10_basis()

    print('\nFIGURES')
    fig_perfold('repod_main', 'RepOD', 'Fig01_repod_perfold.svg')
    fig_perfold('msu_main', 'MSU', 'Fig02_msu_perfold.svg')
    fig_roc('repod_main', 'RepOD', 'Fig03_repod_roc.svg')
    fig_roc('msu_main', 'MSU', 'Fig04_msu_roc.svg')
    fig_roc('bt', 'Button-tone', 'Fig05_buttontone_roc.svg', is_bt=True)
    fig_confusion('repod_main', 'RepOD', 'Fig06_repod_confusion.svg')
    fig_confusion('msu_main', 'MSU', 'Fig07_msu_confusion.svg')
    fig_confusion('bt', 'Button-tone', 'Fig08_buttontone_confusion.svg', is_bt=True)
    fig_c3_b1_generalization('Fig09_c3_vs_b1_generalization.svg')
    fig_basis('Fig10_spectral_vs_temporal.svg')
    fig_lambda('Fig11_lambda_sweep.svg')
    fig_rank('Fig12_rank_sweep.svg')
    fig_ablation_bars('repod_abl', 'RepOD', 'Fig13_repod_ablation.svg')
    fig_ablation_bars('msu_abl', 'MSU', 'Fig14_msu_ablation.svg')
    fig_subj_vs_epoch('Fig15_subject_vs_epoch.svg')
    fig_1oversqrtN('Fig16_mode3_diagnostic.svg')
    fig_sota_bars('Fig17_sota_comparison.svg')
    fig_classifier_compare('msu_main', 'MSU', 'Fig18_msu_classifiers.svg')
    fig_classifier_compare('repod_main', 'RepOD', 'Fig19_repod_classifiers.svg')
    fig_classifier_compare('bt', 'Button-tone', 'Fig20_buttontone_classifiers.svg')

    print(f'\nDone. Tables -> {TAB}\n      Figures -> {FIG}')


if __name__ == '__main__':
    main()
