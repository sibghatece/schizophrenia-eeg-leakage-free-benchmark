#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
buttontone_erp.py — build the leakage-free feature cache for the Ford/Roach
button-tone dataset, as a HELD-OUT ERP generalization test for CC-Tucker.

WHAT THIS IS
    ERPdata.csv is the TRIAL-AVERAGED ERP: exactly 3072 samples (3 s @ 1024 Hz,
    -1500..+1499 ms) per (subject, condition), 81 subjects, 3 conditions.
    Because trials are already averaged, oscillatory content is gone, so PSD is
    meaningless here. Instead we treat button-tone as an ERP-morphology test:
      - condition 1 = self-generated tone (button press makes the tone)
      - condition 2 = passively-heard tone
      N1-suppression = N1(cond2) - N1(cond1) is the corollary-discharge
      biomarker, blunted in schizophrenia. We use conditions 1 AND 2.

WHAT IT PRODUCES  (in outputs/results_v2/, same convention as the EEG cache)
    bt_X.npy   float32 (n_subj, C=9, T)   the condition-1 ERP tensor (temporal)
    bt_X2.npy  float32 (n_subj, C=9, T)   the condition-2 ERP tensor
    bt_y.npy   int     (n_subj,)          0=HC, 1=SZ  (from demographic.csv)
    bt_erpfeat.npy  float32 (n_subj, D)   interpretable ERP scalar features
    bt_meta.json    channel names, times, feature names, downsample factor

    One "epoch" == one subject here (ERP is already a per-subject average), so
    LOSO over subjects and the whole leakage discipline carry over unchanged.

WHY A SEPARATE SCRIPT
    button-tone has a different montage (9 ch), a different feature domain (ERP,
    not spectral) and a different unit of observation (subject-average, not
    windowed epoch). Bolting that onto cc_tucker_v2's EEG loader would tangle
    three code paths. This script prepares a clean cache; the matching runner
    (buttontone_run.py, next) consumes it with the SAME CC-Tucker core.

USAGE
    cd /Users/sikhan/Documents/NayaDuarPaper3
    python buttontone_erp.py button_tone
"""
import json
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- config
FS_BT = 1024.0                      # button-tone sampling rate
CHANS = ['Fz', 'FCz', 'Cz', 'FC3', 'FC4', 'C3', 'C4', 'CP3', 'CP4']
DOWNSAMPLE = 8                      # 1024 -> 128 Hz effective; ERP < ~30 Hz so lossless
BASE = '/Users/sikhan/Documents/NayaDuarPaper3'
OUT_DIR = os.path.join(BASE, 'outputs', 'results_v2')

# ERP component windows (ms) — canonical auditory ERP
WINDOWS = {
    'N100': (80, 150, 'neg'),
    'P200': (150, 250, 'pos'),
    'P300': (250, 500, 'pos'),
}
MEAN_WINDOWS = {  # mean-amplitude windows (ms)
    'early': (80, 250),
    'late': (250, 500),
}


def peak_in_window(wave, t_ms, lo, hi, sign):
    """Peak amplitude and latency of `wave` within [lo,hi] ms."""
    m = (t_ms >= lo) & (t_ms <= hi)
    if not m.any():
        return 0.0, 0.0
    seg = wave[m]
    tt = t_ms[m]
    idx = seg.argmin() if sign == 'neg' else seg.argmax()
    return float(seg[idx]), float(tt[idx])


def erp_features(w1, w2, t_ms):
    """
    Interpretable ERP features per channel, from the two condition waveforms.
    w1, w2: (C, T) condition-1 and condition-2 averaged ERPs.
    Returns a flat feature vector + the ordered feature names.
    """
    C = w1.shape[0]
    feats, names = [], []
    for ci in range(C):
        ch = CHANS[ci]
        for comp, (lo, hi, sgn) in WINDOWS.items():
            a1, l1 = peak_in_window(w1[ci], t_ms, lo, hi, sgn)
            a2, l2 = peak_in_window(w2[ci], t_ms, lo, hi, sgn)
            feats += [a1, l1, a2, l2, a2 - a1]           # cond1, cond2, suppression
            names += [f'{ch}_{comp}_amp1', f'{ch}_{comp}_lat1',
                      f'{ch}_{comp}_amp2', f'{ch}_{comp}_lat2',
                      f'{ch}_{comp}_supp']
        for wn, (lo, hi) in MEAN_WINDOWS.items():
            m = (t_ms >= lo) & (t_ms <= hi)
            m1 = float(w1[ci, m].mean()) if m.any() else 0.0
            m2 = float(w2[ci, m].mean()) if m.any() else 0.0
            feats += [m1, m2, m2 - m1]
            names += [f'{ch}_{wn}_mean1', f'{ch}_{wn}_mean2', f'{ch}_{wn}_meansupp']
    return np.array(feats, dtype=np.float32), names


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else 'button_tone'
    root = os.path.abspath(os.path.expanduser(root))
    erp_path = os.path.join(root, 'ERPdata.csv')
    demo_path = os.path.join(root, 'demographic.csv')
    for p in (erp_path, demo_path):
        if not os.path.exists(p):
            sys.exit(f'ERROR: {p} not found.')

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- labels -------------------------------------------------------
    demo = pd.read_csv(demo_path)
    demo.columns = [c.strip() for c in demo.columns]
    demo['subject'] = demo['subject'].astype(int)
    demo['group'] = demo['group'].astype(int)
    label = dict(zip(demo['subject'], demo['group']))     # 0=HC, 1=SZ
    print(f'demographic.csv: {len(label)} subjects, '
          f'HC={sum(v==0 for v in label.values())} '
          f'SZ={sum(v==1 for v in label.values())}')

    # --- ERP data -----------------------------------------------------
    print('loading ERPdata.csv ...')
    df = pd.read_csv(erp_path)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in CHANS if c not in df.columns]
    if missing:
        sys.exit(f'ERROR: channels missing from ERPdata.csv: {missing}\n'
                 f'available: {list(df.columns)}')

    subs = sorted(df['subject'].unique())
    _s0 = df[(df['subject'] == subs[0]) & (df['condition'] == 1)]
    t_full = np.sort(_s0['time_ms'].unique())
    t_ds = t_full[::DOWNSAMPLE]
    T = len(t_ds)
    print(f'{len(subs)} subjects, {len(t_full)} samples/epoch '
          f'-> {T} after /{DOWNSAMPLE} downsample '
          f'({t_ds[0]:.0f}..{t_ds[-1]:.0f} ms)')

    X1 = np.zeros((len(subs), len(CHANS), T), np.float32)
    X2 = np.zeros((len(subs), len(CHANS), T), np.float32)
    y = np.zeros(len(subs), int)
    feats, fnames = [], None
    keep = []

    for si, s in enumerate(subs):
        if s not in label:
            print(f'  [skip] subject {s}: no demographic label')
            continue
        d1 = df[(df['subject'] == s) & (df['condition'] == 1)].sort_values('time_ms')
        d2 = df[(df['subject'] == s) & (df['condition'] == 2)].sort_values('time_ms')
        if len(d1) == 0 or len(d2) == 0:
            print(f'  [skip] subject {s}: missing condition 1 or 2')
            continue
        w1 = d1[CHANS].to_numpy(np.float64).T          # (C, Tfull)
        w2 = d2[CHANS].to_numpy(np.float64).T
        # baseline-correct on the pre-stimulus interval (-1500..0 ms)
        pre = t_full < 0
        w1 -= w1[:, pre].mean(axis=1, keepdims=True)
        w2 -= w2[:, pre].mean(axis=1, keepdims=True)
        # ERP scalar features from full-resolution waveforms
        fv, nm = erp_features(w1, w2, t_full)
        if fnames is None:
            fnames = nm
        feats.append(fv)
        # downsampled tensors for CC-Tucker
        X1[len(keep)] = w1[:, ::DOWNSAMPLE].astype(np.float32)
        X2[len(keep)] = w2[:, ::DOWNSAMPLE].astype(np.float32)
        y[len(keep)] = label[s]
        keep.append(s)

    n = len(keep)
    X1, X2, y = X1[:n], X2[:n], y[:n]
    feats = np.vstack(feats)

    np.save(os.path.join(OUT_DIR, 'bt_X.npy'), X1)
    np.save(os.path.join(OUT_DIR, 'bt_X2.npy'), X2)
    np.save(os.path.join(OUT_DIR, 'bt_y.npy'), y)
    np.save(os.path.join(OUT_DIR, 'bt_erpfeat.npy'), feats)
    meta = dict(channels=CHANS, n_subjects=n, T=T, fs_effective=FS_BT / DOWNSAMPLE,
                downsample=DOWNSAMPLE, times_ms=t_ds.tolist(),
                erp_feature_names=fnames,
                label_map={'0': 'HC', '1': 'SZ'},
                conditions_used=[1, 2], note='trial-averaged ERP; temporal basis')
    with open(os.path.join(OUT_DIR, 'bt_meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    print(f'\nWrote cache for {n} subjects (HC={int((y==0).sum())} '
          f'SZ={int((y==1).sum())}):')
    print(f'  bt_X.npy  {X1.shape}   condition-1 ERP tensor')
    print(f'  bt_X2.npy {X2.shape}   condition-2 ERP tensor')
    print(f'  bt_erpfeat.npy {feats.shape}   ({len(fnames)} ERP scalar features)')
    print(f'  bt_y.npy  {y.shape}')
    print(f'  bt_meta.json')
    print('\nNext:  python buttontone_run.py')


if __name__ == '__main__':
    main()
