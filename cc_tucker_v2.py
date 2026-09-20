#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cc_tucker_v2.py — corrected class-conditional Tucker pipeline
================================================================================
Replaces tucker_combined.py. Fixes four defects found in the audit:

  D1  score_dict() only returned acc/bac/kap/f1, so MCC, Sens, Spec, PPV, NPV
      and AUC were never measured (they were later FAKED by
      compute_extended_metrics()).  --> now a full confusion-matrix metric set
      is computed, and per-fold predictions + probabilities are SAVED.

  D2  tucker_hosvd_sparse() soft-thresholded the core G into Gs but returned
      only `factors`. Gs was used solely in the convergence check, so the L1
      sparse-core regularisation (the paper's title claim) had NO EFFECT on
      any feature.  --> the sparse core is now returned AND used in
      reconstruction, so lambda genuinely changes the model.

  D3  fit_cc_tucker() built the class tensor from per-subject TEMPORAL MEAN
      epochs (X.mean(axis=0)). Resting EEG is not phase-locked, so averaging
      N epochs sample-by-sample cancels the signal ~1/sqrt(N). RepOD averages
      ~514 epochs => near-total annihilation; MSU averages 29 => partial.
      That is the whole RepOD-vs-MSU asymmetry.  --> new SPECTRAL basis uses
      per-subject mean log-PSD, which is phase-invariant, so averaging HELPS.

  D4  The two class subspaces were nearly identical (both classes share almost
      the same 16-channel spatial covariance), so eps_SZ ~ eps_HC and the
      "discriminative" error features carried no information.  --> pooled
      spatial whitening (fitted on TRAINING FOLDS ONLY) is applied before the
      class Tucker fits, which forces the subspaces apart. This is the
      multilinear analogue of the CSP whitening step.

LEAKAGE POLICY
    Whitening, Tucker fits, KW selection and the scaler are fitted on training
    folds ONLY. Fold-invariant features (raw Hjorth, raw PSR, band power,
    connectivity, per-epoch log-PSD) depend on a single epoch each and are
    precomputed once outside the CV loop -- this is a pure speed optimisation
    and is NOT leakage, because no cross-epoch statistic is involved.

USAGE
    python cc_tucker_v2.py --dataset both --basis spectral
    python cc_tucker_v2.py --dataset msu  --basis temporal      # old basis, fair A/B
    python cc_tucker_v2.py --dataset msu  --rank-sweep
    python cc_tucker_v2.py --dataset both --lambda-sweep
    python cc_tucker_v2.py --dataset both --ablation
    python cc_tucker_v2.py --dataset msu  --protocol gkf --quick
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime

import numpy as np
from scipy import signal as sig
from scipy import stats
from scipy.stats import chi2

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIG
# ==============================================================================

BASE     = '/Users/sikhan/Documents/NayaDuarPaper3'
RES_DIR  = os.path.join(BASE, 'outputs', 'results')
OUT_DIR  = os.path.join(BASE, 'outputs', 'results_v2')
LOG_DIR  = os.path.join(BASE, 'logs')

FS       = {'repod': 250, 'msu': 128}
N_CH     = 16
RSTATE   = 42
TOP_K    = 80
TAU_PSR  = 1
EPS      = 1e-9

TUCKER_RANKS = (8, 20, 6)     # (spatial, spectral-or-temporal, subject)
LAMBDA       = 0.01
N_ITER_MAX   = 30
TOL          = 1e-4

PSD_NPERSEG  = 256            # Welch window; freq axis cropped to 0.5-45 Hz
PSD_FMIN, PSD_FMAX = 0.5, 45.0

BANDS = [('delta', 0.5, 4), ('theta', 4, 8), ('alpha', 8, 13),
         ('beta', 13, 30), ('gamma', 30, 45)]

N_BLOCKS = [1]        # set by --blocks
SELECT_MODE = ['pooled']  # set by --select-mode

CHUNK = 2048                  # epochs per chunk in memory-heavy steps
SEP   = '=' * 74
SEP2  = '-' * 74


# ==============================================================================
# TENSOR PRIMITIVES
# ==============================================================================

def np_unfold(X, mode):
    return np.moveaxis(X, mode, 0).reshape(X.shape[mode], -1)


def np_fold(M, mode, shape):
    full = [shape[mode]] + [s for i, s in enumerate(shape) if i != mode]
    return np.moveaxis(M.reshape(full), 0, mode)


def np_mode_dot(X, M, mode):
    out_shape = list(X.shape)
    out_shape[mode] = M.shape[0]
    return np_fold(M @ np_unfold(X, mode), mode, out_shape)


def np_multi_mode_dot(X, mats, modes):
    for M, m in zip(mats, modes):
        X = np_mode_dot(X, M, m)
    return X


def soft_threshold(a, lam):
    return np.sign(a) * np.maximum(np.abs(a) - lam, 0.0)


def tucker_hosvd_sparse(T, ranks, lam=LAMBDA, n_iter_max=N_ITER_MAX, tol=TOL):
    """
    HOSVD init + HOOI refinement + L1 soft-thresholded core.

    FIX vs original: returns (factors, sparse_core). The original discarded the
    sparse core, so lambda influenced nothing except the stopping test. Here the
    sparse core is the model, and reconstruction uses it -- so lambda is a real
    hyperparameter that can be swept.
    """
    X = np.asarray(T, dtype=np.float64)
    RL = [min(r, s) for r, s in zip(ranks, X.shape)]

    factors = []
    for mode, R in enumerate(RL):
        unf = np_unfold(X, mode)
        try:
            U, _, _ = np.linalg.svd(unf, full_matrices=False)
        except np.linalg.LinAlgError:
            U, _ = np.linalg.qr(np.random.RandomState(RSTATE).randn(unf.shape[0], R))
        factors.append(np.ascontiguousarray(U[:, :R]))

    prev = np.inf
    normX = np.linalg.norm(X) + EPS
    for _ in range(n_iter_max):
        for mode in range(3):
            om = [m for m in range(3) if m != mode]
            Y = np_multi_mode_dot(X, [factors[m].T for m in om], om)
            unf = np_unfold(Y, mode)
            try:
                U, _, _ = np.linalg.svd(unf, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            factors[mode] = np.ascontiguousarray(U[:, :RL[mode]])

        G = np_multi_mode_dot(X, [f.T for f in factors], [0, 1, 2])
        Gs = soft_threshold(G, lam)
        Xr = np_multi_mode_dot(Gs, factors, [0, 1, 2])
        err = np.linalg.norm(X - Xr) / normX
        if abs(prev - err) < tol:
            prev = err
            break
        prev = err

    G = np_multi_mode_dot(X, [f.T for f in factors], [0, 1, 2])
    Gs = soft_threshold(G, lam)
    return factors, Gs, float(prev)


# ==============================================================================
# PREPROCESSING (only needed if rebuilding cache; cached .npy already filtered)
# ==============================================================================

def preprocess(data, fs):
    data = np.nan_to_num(data, nan=0.)
    hi = min(45., fs / 2. - 1.)
    b, a = sig.butter(4, [0.5 / (fs / 2), hi / (fs / 2)], btype='band')
    data = sig.filtfilt(b, a, data, axis=1)
    if fs >= 100:
        b2, a2 = sig.iirnotch(50., Q=30, fs=fs)
        data = sig.filtfilt(b2, a2, data, axis=1)
    mu = np.mean(data, axis=1, keepdims=True)
    sd = np.std(data, axis=1, keepdims=True) + EPS
    return np.nan_to_num((data - mu) / sd, nan=0.)


# ==============================================================================
# FOLD-INVARIANT FEATURES  (computed once; single-epoch statistics only)
# ==============================================================================

def hjorth_batch(E):
    """E: (N, C, L) -> (N, 3C)  [activity, mobility, complexity]"""
    E = E.astype(np.float64, copy=False)
    d1 = np.diff(E, axis=2)
    d2 = np.diff(d1, axis=2)
    v0 = E.var(axis=2) + EPS
    v1 = d1.var(axis=2) + EPS
    v2 = d2.var(axis=2) + EPS
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / mob
    return np.concatenate([v0, mob, comp], axis=1).astype(np.float32)


def psr2d_batch(E, tau=TAU_PSR):
    """95% confidence ellipse area of the 2-D delay embedding. (N, C)"""
    E = E.astype(np.float64, copy=False)
    x = E[:, :, :-tau]
    y = E[:, :, tau:]
    xm = x.mean(axis=2, keepdims=True)
    ym = y.mean(axis=2, keepdims=True)
    xc, yc = x - xm, y - ym
    n = xc.shape[2]
    sxx = (xc * xc).sum(axis=2) / (n - 1)
    syy = (yc * yc).sum(axis=2) / (n - 1)
    sxy = (xc * yc).sum(axis=2) / (n - 1)
    det = np.maximum(sxx * syy - sxy ** 2, 0.0)
    return (np.pi * chi2.ppf(0.95, 2) * np.sqrt(det)).astype(np.float32)


def psr3d_iqr_batch(E, tau=TAU_PSR):
    """IQR of Euclidean distances from centroid in 3-D delay embedding. (N, C)"""
    E = E.astype(np.float64, copy=False)
    a = E[:, :, :-2 * tau]
    b = E[:, :, tau:-tau]
    c = E[:, :, 2 * tau:]
    W = np.stack([a, b, c], axis=-1)             # (N, C, M, 3)
    W = W - W.mean(axis=2, keepdims=True)
    d = np.sqrt((W ** 2).sum(axis=-1))           # (N, C, M)
    q75, q25 = np.percentile(d, [75, 25], axis=2)
    return (q75 - q25).astype(np.float32)


def band_power_batch(E, fs):
    """Log band power per channel per band. (N, 5C)"""
    out = []
    for _, lo, hi in BANDS:
        hi = min(hi, fs / 2. - 1.)
        b, a = sig.butter(4, [lo / (fs / 2), hi / (fs / 2)], btype='band')
        Xb = sig.filtfilt(b, a, E, axis=2)
        out.append(np.log1p((Xb ** 2).mean(axis=2)))
    return np.concatenate(out, axis=1).astype(np.float32)


def connectivity_batch(E):
    """Upper-triangular Pearson correlations. (N, C(C-1)/2)"""
    E = E.astype(np.float64, copy=False)
    Ec = E - E.mean(axis=2, keepdims=True)
    sd = Ec.std(axis=2) + EPS
    N, C, L = E.shape
    R = np.einsum('ncl,nml->ncm', Ec, Ec) / L
    R /= (sd[:, :, None] * sd[:, None, :])
    iu = np.triu_indices(C, k=1)
    return R[:, iu[0], iu[1]].astype(np.float32)


def band_connectivity_batch(E, fs):
    """
    U3. Per-band Pearson connectivity -> (N, 5*C(C-1)/2) = (N, 600).
    Broadband correlation blurs together band-specific coupling; the
    schizophrenia effect is concentrated in theta/alpha.
    """
    out = []
    for _, lo, hi in BANDS:
        hi = min(hi, fs / 2. - 1.)
        b, a = sig.butter(4, [lo / (fs / 2), hi / (fs / 2)], btype='band')
        Xb = sig.filtfilt(b, a, E, axis=2)
        out.append(connectivity_batch(Xb))
    return np.concatenate(out, axis=1).astype(np.float32)


def logpsd_batch(E, fs):
    """Per-epoch log power spectral density. (N, C, F)"""
    nper = min(PSD_NPERSEG, E.shape[2])
    f, P = sig.welch(E, fs=fs, nperseg=nper, axis=2)
    keep = (f >= PSD_FMIN) & (f <= PSD_FMAX)
    return np.log1p(P[:, :, keep]).astype(np.float32), f[keep]


def second_order_batch(E, tau=TAU_PSR):
    """
    Per-epoch 16x16 second-order matrices.

    SPEED FIX. Hjorth and 2-D PSR of a SPATIALLY filtered signal A@E are
    quadratic forms in these matrices:
        var_c(A E)      = diag(A C0 A')
        var_c(A dE)     = diag(A C1 A')
        var_c(A d2E)    = diag(A C2 A')
        cov_c(AE, AEtau)= diag(A Cxy A')
    Differencing acts on time and the spatial filter acts on channels, so the
    two commute exactly -- these are identities, not approximations. This lets
    every per-fold class filter be evaluated with 16x16 algebra instead of
    rebuilding (N, 16, L) filtered signals, which is ~50x faster.
    """
    E = E.astype(np.float64, copy=False)
    L = E.shape[2]

    def cov(A, B=None, ddof=0):
        # ddof MUST match the consumer: hjorth_batch uses np.var (ddof=0),
        # psr2d_batch uses the sample covariance (ddof=1). Mismatching them
        # introduces an O(1/L) bias and breaks exact agreement.
        Ac = A - A.mean(axis=2, keepdims=True)
        Bc = Ac if B is None else B - B.mean(axis=2, keepdims=True)
        return np.einsum('ncl,nml->ncm', Ac, Bc) / max(Ac.shape[2] - ddof, 1)

    d1 = np.diff(E, axis=2)
    d2 = np.diff(d1, axis=2)
    x, yy = E[:, :, :-tau], E[:, :, tau:]
    return dict(
        S0=np.einsum('ncl,nml->ncm', E, E).astype(np.float64),
        C0=cov(E, ddof=0).astype(np.float64),
        C1=cov(d1, ddof=0).astype(np.float64),
        C2=cov(d2, ddof=0).astype(np.float64),
        Cxx=cov(x, ddof=1).astype(np.float64),
        Cyy=cov(yy, ddof=1).astype(np.float64),
        Cxy=cov(x, yy, ddof=1).astype(np.float64),
        L=L)


def precompute_invariant(X, fs, verbose=True):
    """All features that depend on ONE epoch only -> computed once, not per fold."""
    N = X.shape[0]
    hj, p2, p3, bp, cn, ps = [], [], [], [], [], []
    bc = []
    so = {k: [] for k in ('S0', 'C0', 'C1', 'C2', 'Cxx', 'Cyy', 'Cxy')}
    t0 = time.time()
    for s in range(0, N, CHUNK):
        e = X[s:s + CHUNK]
        hj.append(hjorth_batch(e))
        p2.append(psr2d_batch(e))
        p3.append(psr3d_iqr_batch(e))
        bp.append(band_power_batch(e, fs))
        cn.append(connectivity_batch(e))
        bc.append(band_connectivity_batch(e, fs))
        pp, freqs = logpsd_batch(e, fs)
        ps.append(pp)
        sb = second_order_batch(e)
        for k in so:
            so[k].append(sb[k])
        if verbose:
            done = min(s + CHUNK, N)
            print(f'\r    invariant features {done}/{N} '
                  f'({time.time()-t0:.0f}s)', end='', flush=True)
    if verbose:
        print()
    out = dict(
        hjorth=np.vstack(hj), psr2=np.vstack(p2), psr3=np.vstack(p3),
        bp=np.vstack(bp), conn=np.vstack(cn), bconn=np.vstack(bc),
        psd=np.concatenate(ps, axis=0), freqs=freqs, L=X.shape[2])
    for k in so:
        out[k] = np.concatenate(so[k], axis=0)
    return out


def _diagAMA(A, M):
    """diag(A M A^T) for a stack of M: (N,C,C) -> (N,C). Never forms A M A^T."""
    return np.einsum('ij,njk,ik->ni', A, M.astype(np.float64), A)


def filtered_hjorth(A, so):
    """Hjorth (activity, mobility, complexity) of A@E, from 16x16 matrices."""
    v0 = np.maximum(_diagAMA(A, so['C0']), EPS)
    v1 = np.maximum(_diagAMA(A, so['C1']), EPS)
    v2 = np.maximum(_diagAMA(A, so['C2']), EPS)
    mob = np.sqrt(v1 / v0)
    comp = np.sqrt(v2 / v1) / np.maximum(mob, EPS)
    return np.concatenate([v0, mob, comp], axis=1).astype(np.float32)


def filtered_psr2d(A, so):
    """95% ellipse area of the 2-D delay embedding of A@E."""
    sxx = _diagAMA(A, so['Cxx'])
    syy = _diagAMA(A, so['Cyy'])
    sxy = _diagAMA(A, so['Cxy'])
    det = np.maximum(sxx * syy - sxy ** 2, 0.0)
    return (np.pi * chi2.ppf(0.95, 2) * np.sqrt(det)).astype(np.float32)


# ==============================================================================
# CLASS-CONDITIONAL TUCKER  (whitened; spectral or temporal basis)
# ==============================================================================

def pooled_whitener(X_tr_list, reg=0.05):
    """
    Sigma^{-1/2} from the pooled spatial covariance of TRAINING subjects only.
    Forces the two class subspaces apart (D4). Shrinkage keeps it stable.
    """
    C = X_tr_list[0].shape[1]
    S = np.zeros((C, C))
    n = 0
    for X in X_tr_list:
        E = X.astype(np.float64)
        Ec = E - E.mean(axis=2, keepdims=True)
        S += np.einsum('ncl,nml->cm', Ec, Ec)
        n += E.shape[0] * E.shape[2]
    S /= max(n, 1)
    S = (1 - reg) * S + reg * np.trace(S) / C * np.eye(C)
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, 1e-10)
    return V @ np.diag(w ** -0.5) @ V.T


MIN_EPOCHS_PER_BLOCK = 20


def subject_slabs(X, psd, basis, W, n_blocks=1):
    """
    U1. Representative slabs for ONE subject.

    n_blocks=1 reproduces the old single-slab behaviour. n_blocks>1 splits the
    subject's epochs into contiguous blocks and returns one slab each, which
    multiplies the mode-3 sample count without touching any test data.
    A block is only formed if it keeps >= MIN_EPOCHS_PER_BLOCK epochs, so
    short recordings (MSU, 29 epochs) degrade gracefully to a single slab.
    """
    N = psd.shape[0]
    nb = max(1, min(int(n_blocks), N // MIN_EPOCHS_PER_BLOCK))
    edges = np.linspace(0, N, nb + 1).astype(int)
    slabs = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 2:
            continue
        if basis == 'spectral':
            M = psd[a:b].mean(axis=0).astype(np.float64)
        else:
            M = X[a:b].mean(axis=0).astype(np.float64)
        slabs.append(W @ M)
    if not slabs:
        M = (psd.mean(axis=0) if basis == 'spectral'
             else X.mean(axis=0)).astype(np.float64)
        slabs = [W @ M]
    return slabs


def fit_cc_tucker(X_tr, psd_tr, y_tr, ranks, lam, basis, W, n_blocks=1):
    """Fit one sparse Tucker model per class on whitened subject slabs."""
    slabs, labels = [], []
    for X, P, yy in zip(X_tr, psd_tr, y_tr):
        sl = subject_slabs(X, P, basis, W, n_blocks)
        slabs.extend(sl)
        labels.extend([int(yy[0])] * len(sl))
    models = {}
    for cls in (0, 1):
        idx = [i for i, l in enumerate(labels) if l == cls]
        if len(idx) < 2:
            idx = list(range(len(labels)))
        T = np.stack([slabs[i] for i in idx], axis=2)
        try:
            factors, core, err = tucker_hosvd_sparse(T, ranks, lam=lam)
            models[cls] = dict(U1=factors[0], U2=factors[1],
                               core=core, fit_err=err)
        except Exception as exc:
            print(f'    [WARN] Tucker class {cls} failed: {exc}')
            models[cls] = None
    models['W'] = W
    models['basis'] = basis
    return models


def tucker_features(inv, idx, models, X=None):
    """
    Per-fold Tucker-dependent features.

      spatial class filtering (time domain):
          Ehat_c = U1_c U1_c^T  (W E)     -> Hjorth (3C) + PSR2D (C) per class
          residual = WE - Ehat_1          -> Hjorth (3C)
      subspace reconstruction errors in the model domain:
          eps_0, eps_1, delta, rho, eta   -> 5
      spectral projection energies:
          row/col norms of U1_c^T P U2_c  -> R1 + R2 per class

    Returns (N, D) float32.
    """
    W = models['W']
    m0, m1 = models[0], models[1]
    n_tot = len(idx)
    if m0 is None or m1 is None:
        return np.zeros((n_tot, 1), dtype=np.float32)

    spectral = (models['basis'] == 'spectral')
    C = W.shape[0]
    I = np.eye(C)
    blocks = []

    for s in range(0, n_tot, CHUNK):
        sl = idx[s:s + CHUNK]
        so = {k: inv[k][sl] for k in ('S0', 'C0', 'C1', 'C2',
                                      'Cxx', 'Cyy', 'Cxy')}
        Pw = np.einsum('ij,njf->nif', W, inv['psd'][sl].astype(np.float64))

        feats = []
        recon_err = {}
        for cls, mm in ((0, m0), (1, m1)):
            U1, U2 = mm['U1'], mm['U2']
            Sp = U1 @ U1.T                   # spatial class projector
            A = Sp @ W                       # full class filter on channels

            # Hjorth + PSR of the filtered signal, from 16x16 algebra only
            feats.append(filtered_hjorth(A, so))
            feats.append(filtered_psr2d(A, so))

            # mode-2 reconstruction error, measured where U2 actually lives
            if spectral:
                Mw = Pw
                Mhat = np.einsum('ij,njf->nif', Sp, Mw)
                Mhat = np.einsum('nif,fg,hg->nih', Mhat, U2, U2)
                num = np.linalg.norm(Mw - Mhat, axis=(1, 2))
                den = np.linalg.norm(Mw, axis=(1, 2)) + EPS
            else:
                # ||WE||_F^2 = sum diag(W S0 W'), no need to touch the signal
                normM2 = _diagAMA(W, so['S0']).sum(axis=1)
                E = np.asarray(X[sl]).astype(np.float64)
                Y = np.einsum('ij,njl,lq->niq', W, E, U2)   # (n, C, R2)
                SpY = np.einsum('ij,njq->niq', Sp, Y)
                cross = np.einsum('niq,niq->n', Y, SpY)
                normH2 = np.einsum('niq,niq->n', SpY, SpY)
                num = np.sqrt(np.maximum(normM2 - 2 * cross + normH2, 0.0))
                den = np.sqrt(np.maximum(normM2, 0.0)) + EPS
                Mw = None
            recon_err[cls] = num / den

            # projection energies of the class subspace
            src = Pw if spectral else Y
            Z = (np.einsum('ir,nif,fq->nrq', U1, src, U2) if spectral
                 else np.einsum('ir,niq->nrq', U1, src))
            feats.append(np.linalg.norm(Z, axis=2).astype(np.float32))
            feats.append(np.linalg.norm(Z, axis=1).astype(np.float32))

        # residual after removing the SZ subspace
        A_res = (I - m1['U1'] @ m1['U1'].T) @ W
        feats.append(filtered_hjorth(A_res, so))

        e0, e1 = recon_err[0], recon_err[1]
        delta = e1 - e0
        rho = e1 / (e0 + EPS)
        eta = np.log(rho + EPS)
        feats.append(np.stack([e0, e1, delta, rho, eta], axis=1).astype(np.float32))

        blocks.append(np.concatenate(feats, axis=1))
    return np.vstack(blocks).astype(np.float32)


# ==============================================================================
# FEATURE ASSEMBLY  (ablation-aware)
# ==============================================================================

FEATURE_GROUPS = ('hjorth', 'psr', 'bp', 'conn', 'tucker')

CONFIGS = {
    'A1': ('hjorth', 'psr'),
    'A2': ('conn',),
    'A3': ('bp', 'conn'),
    'B1': ('hjorth', 'psr', 'bp', 'conn'),
    'C1': ('tucker_err', 'conn'),
    'C2': ('tucker',),
    'C3': ('hjorth', 'psr', 'bp', 'conn', 'tucker'),
    'D1': ('hjorth', 'psr', 'bp', 'conn', 'bconn'),
    'D2': ('bconn',),
    'D3': ('hjorth', 'psr', 'bp', 'conn', 'bconn', 'tucker'),
    'E1': ('conn', 'bconn'),
    'E2': ('bp', 'bconn'),
    'E3': ('bp', 'conn', 'bconn'),
    'E4': ('hjorth', 'bp', 'conn', 'bconn'),
    'E5': ('conn', 'bconn', 'tucker'),
}


def assemble(inv, tuck, idx, groups):
    """Returns (F, bounds) where bounds maps block name -> (start, end)."""
    parts, bounds, c = [], {}, 0

    def add(name, arr):
        nonlocal c
        parts.append(arr)
        bounds[name] = (c, c + arr.shape[1])
        c += arr.shape[1]

    if 'hjorth' in groups:
        add('hjorth', inv['hjorth'][idx])
    if 'psr' in groups:
        add('psr', np.hstack([inv['psr2'][idx], inv['psr3'][idx]]))
    if 'bp' in groups:
        add('bp', inv['bp'][idx])
    if 'conn' in groups:
        add('conn', inv['conn'][idx])
    if 'bconn' in groups:
        add('bconn', inv['bconn'][idx])
    if 'tucker' in groups:
        add('tucker', tuck)
    if 'tucker_err' in groups:
        add('tucker_err', tuck[:, -5:])
    return np.concatenate(parts, axis=1).astype(np.float64), bounds


def kw_scores(F, y):
    """Kruskal-Wallis H per feature, computed on TRAINING data only."""
    a, b = F[y == 0], F[y == 1]
    H = np.zeros(F.shape[1])
    for j in range(F.shape[1]):
        try:
            H[j] = stats.kruskal(a[:, j], b[:, j])[0]
        except Exception:
            H[j] = 0.
    return np.nan_to_num(H, nan=0.)


# U2. Share of the top-K budget each block may claim. The Tucker block is
# capped so it cannot evict connectivity, which the ablation showed is the
# most discriminative block on RepOD.
QUOTA = {'hjorth': 0.10, 'psr': 0.10, 'bp': 0.10,
         'conn': 0.20, 'bconn': 0.30, 'tucker': 0.20, 'tucker_err': 0.05}


def kw_select(F, y, k, bounds=None, mode='pooled'):
    """
    mode='pooled' : original single open ranking (kept for A/B).
    mode='quota'  : each block gets its own share of the K budget, then any
                    unfilled remainder is topped up from the global ranking.
    """
    H = kw_scores(F, y)
    k = min(k, F.shape[1])
    if mode != 'quota' or not bounds:
        return np.argsort(H)[::-1][:k]

    chosen = []
    present = {n: q for n, q in QUOTA.items() if n in bounds}
    tot = sum(present.values()) or 1.0
    for name, q in present.items():
        s, e = bounds[name]
        n_take = int(round(k * q / tot))
        if n_take <= 0:
            continue
        local = np.argsort(H[s:e])[::-1][:min(n_take, e - s)]
        chosen.extend((local + s).tolist())
    chosen = list(dict.fromkeys(chosen))
    if len(chosen) < k:
        for j in np.argsort(H)[::-1]:
            if j not in chosen:
                chosen.append(int(j))
            if len(chosen) >= k:
                break
    return np.array(chosen[:k], dtype=int)


# ==============================================================================
# CLASSIFIERS AND *REAL* METRICS
# ==============================================================================

def make_clfs():
    return {
        'SVM': SVC(kernel='rbf', C=10., gamma='scale', probability=True,
                   class_weight='balanced', random_state=RSTATE),
        'RF': RandomForestClassifier(300, class_weight='balanced',
                                     n_jobs=-1, random_state=RSTATE),
        'KNN': KNeighborsClassifier(7, n_jobs=-1),
        'LDA': LinearDiscriminantAnalysis(solver='svd'),
    }


def full_metrics(y_true, y_pred, y_proba=None):
    """
    D1 FIX. Every quantity below comes from the actual confusion matrix.
    Nothing is estimated from balanced accuracy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().astype(float)

    sen = tp / (tp + fn) if (tp + fn) else 0.
    spe = tn / (tn + fp) if (tn + fp) else 0.
    ppv = tp / (tp + fp) if (tp + fp) else 0.
    npv = tn / (tn + fn) if (tn + fn) else 0.
    f1b = 2 * ppv * sen / (ppv + sen) if (ppv + sen) else 0.

    out = dict(
        acc=accuracy_score(y_true, y_pred) * 100,
        bal_acc=balanced_accuracy_score(y_true, y_pred) * 100,
        sen=sen * 100, spe=spe * 100, ppv=ppv * 100, npv=npv * 100,
        f1=f1b,
        f1_weighted=f1_score(y_true, y_pred, average='weighted'),
        mcc=float(matthews_corrcoef(y_true, y_pred)),   # in [-1, +1]
        kappa=float(cohen_kappa_score(y_true, y_pred)),
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
    )
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            out['auc'] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            out['auc'] = None
    else:
        out['auc'] = None
    assert -1.0 <= out['mcc'] <= 1.0, 'MCC out of range — bug'
    return out


def ensemble_proba(fitted, F_te):
    """Soft-voting over SVM + RF + LDA (the three probabilistic members)."""
    ps = []
    for m in ('SVM', 'RF', 'LDA'):
        clf = fitted.get(m)
        if clf is None:
            continue
        try:
            ps.append(clf.predict_proba(F_te)[:, 1])
        except Exception:
            pass
    if not ps:
        return None
    return np.mean(ps, axis=0)


# ==============================================================================
# CROSS-VALIDATION
# ==============================================================================

METHODS = ['SVM', 'RF', 'KNN', 'LDA', 'ENSEMBLE']


def _fold_indices(idx_list, test_ids):
    """
    test_ids MUST stay an ordered sequence. Iterating a set here scrambles the
    test-row order relative to y_te (which is built from the ordered list),
    silently misaligning features and labels on any fold with >1 test subject.
    """
    ts = set(test_ids)
    tr = np.concatenate([idx_list[j] for j in range(len(idx_list))
                         if j not in ts])
    te = np.concatenate([idx_list[j] for j in test_ids])
    assert len(te) == sum(len(idx_list[j]) for j in test_ids)
    return tr, te


def run_cv(Xs, ys, inv, idx_list, splits, ranks, lam, basis,
           groups=('hjorth', 'psr', 'bp', 'conn', 'bconn', 'tucker'),
           label='', verbose=True):
    """
    Xs        : list of per-subject epoch arrays (N_i, C, L)
    ys        : list of per-subject label arrays
    inv       : precomputed fold-invariant feature dict (global epoch order)
    idx_list  : list of global epoch-index arrays, one per subject
    splits    : list of (test_subject_index_list) defining the folds
    """
    need_tucker = ('tucker' in groups) or ('tucker_err' in groups)
    store = {m: dict(acc=[], preds=[], true=[], proba=[], subj=[])
             for m in METHODS}

    for fi, test_ids in enumerate(splits):
        tr_idx, te_idx = _fold_indices(idx_list, test_ids)
        y_tr = np.concatenate([ys[j] for j in range(len(ys))
                               if j not in set(test_ids)])
        y_te = np.concatenate([ys[j] for j in test_ids])
        sid_te = np.concatenate([np.full(len(ys[j]), j) for j in test_ids])

        if need_tucker:
            X_tr_sub = [Xs[j] for j in range(len(Xs)) if j not in set(test_ids)]
            psd_tr_sub = [inv['psd'][idx_list[j]] for j in range(len(Xs))
                          if j not in set(test_ids)]
            y_tr_sub = [ys[j] for j in range(len(ys)) if j not in set(test_ids)]

            W = pooled_whitener(X_tr_sub)
            models = fit_cc_tucker(X_tr_sub, psd_tr_sub, y_tr_sub,
                                   ranks, lam, basis, W,
                                   n_blocks=N_BLOCKS[0])
            Xall_ref = np.vstack(Xs) if basis == 'temporal' else None
            T_tr = tucker_features(inv, tr_idx, models, X=Xall_ref)
            T_te = tucker_features(inv, te_idx, models, X=Xall_ref)
        else:
            T_tr = np.zeros((len(tr_idx), 0), dtype=np.float32)
            T_te = np.zeros((len(te_idx), 0), dtype=np.float32)

        F_tr, bnd = assemble(inv, T_tr, tr_idx, groups)
        F_te, _ = assemble(inv, T_te, te_idx, groups)

        assert F_tr.shape[0] == len(y_tr), 'train rows/labels misaligned'
        assert F_te.shape[0] == len(y_te), 'test rows/labels misaligned'

        sel = kw_select(F_tr, y_tr, TOP_K, bounds=bnd, mode=SELECT_MODE[0])
        F_tr, F_te = F_tr[:, sel], F_te[:, sel]

        sc = StandardScaler()
        F_tr = sc.fit_transform(F_tr)
        F_te = sc.transform(F_te)

        fitted, fold_acc = {}, {}
        for m, clf in make_clfs().items():
            try:
                clf.fit(F_tr, y_tr)
                pred = clf.predict(F_te)
                try:
                    pr = clf.predict_proba(F_te)[:, 1]
                except Exception:
                    pr = np.full(len(pred), np.nan)
                fitted[m] = clf
                store[m]['acc'].append(accuracy_score(y_te, pred) * 100)
                store[m]['preds'].extend(pred.tolist())
                store[m]['true'].extend(y_te.tolist())
                store[m]['proba'].extend(pr.tolist())
                store[m]['subj'].extend(sid_te.tolist())
                fold_acc[m] = store[m]['acc'][-1]
            except Exception as exc:
                store[m]['acc'].append(0.)
                fold_acc[m] = 0.
                print(f'    [WARN] {m}: {exc}')

        pe = ensemble_proba(fitted, F_te)
        if pe is not None:
            pred_e = (pe >= 0.5).astype(int)
            store['ENSEMBLE']['acc'].append(accuracy_score(y_te, pred_e) * 100)
            store['ENSEMBLE']['preds'].extend(pred_e.tolist())
            store['ENSEMBLE']['true'].extend(y_te.tolist())
            store['ENSEMBLE']['proba'].extend(pe.tolist())
            store['ENSEMBLE']['subj'].extend(sid_te.tolist())
            fold_acc['ENSEMBLE'] = store['ENSEMBLE']['acc'][-1]

        if verbose:
            print(f'  fold {fi+1:3d}/{len(splits)}  n_te={len(y_te):5d}  ' +
                  '  '.join(f'{m}={fold_acc.get(m, 0):5.1f}' for m in METHODS))

    summary = {}
    for m, s in store.items():
        if not s['true']:
            continue
        pr = np.array(s['proba'], dtype=float)
        pr = None if np.all(np.isnan(pr)) else pr
        mt = full_metrics(s['true'], s['preds'], pr)
        a = np.array(s['acc'])
        mt['fold_acc_mean'] = float(a.mean())
        mt['fold_acc_std'] = float(a.std())
        mt['per_fold'] = a.tolist()
        if s['subj']:
            sv = np.array(s['subj'])
            tp_, pp_ = np.array(s['true']), np.array(s['preds'])
            su, sp_, st_ = [], [], []
            for u in np.unique(sv):
                mk = sv == u
                sp_.append(int(round(pp_[mk].mean())))
                st_.append(int(round(tp_[mk].mean())))
                su.append(int(u))
            mt['subject_level'] = full_metrics(st_, sp_)
            mt['subject_level']['n_subjects'] = len(su)
        mt['preds'] = s['preds']
        mt['true'] = s['true']
        mt['proba'] = s['proba']
        summary[m] = mt

    if verbose:
        print(f'\n  {SEP2}\n  POOLED METRICS — {label}\n  {SEP2}')
        print(f'  {"Method":9s} {"Acc":>7s} {"Sen":>7s} {"Spe":>7s} '
              f'{"PPV":>7s} {"NPV":>7s} {"F1":>7s} {"MCC":>7s} '
              f'{"kappa":>7s} {"AUC":>7s}')
        for m in METHODS:
            if m not in summary:
                continue
            r = summary[m]
            auc = f"{r['auc']:.4f}" if r['auc'] is not None else '   n/a'
            print(f'  {m:9s} {r["acc"]:7.2f} {r["sen"]:7.2f} {r["spe"]:7.2f} '
                  f'{r["ppv"]:7.2f} {r["npv"]:7.2f} {r["f1"]:7.4f} '
                  f'{r["mcc"]:7.4f} {r["kappa"]:7.4f} {auc:>7s}')
        if 'subject_level' in summary.get('ENSEMBLE', {}):
            sl = summary['ENSEMBLE']['subject_level']
            print(f'  [subject-level majority vote, ENSEMBLE, '
                  f'n={sl["n_subjects"]}]  acc={sl["acc"]:.2f}  '
                  f'sen={sl["sen"]:.2f}  spe={sl["spe"]:.2f}  '
                  f'MCC={sl["mcc"]:+.4f}')
        print()
    return summary


# ==============================================================================
# DATA LOADING (from the cached .npy produced by the original pipeline)
# ==============================================================================

def load_cached(dataset):
    Xp = os.path.join(RES_DIR, f'{dataset}_X.npy')
    yp = os.path.join(RES_DIR, f'{dataset}_y.npy')
    if not (os.path.exists(Xp) and os.path.exists(yp)):
        raise FileNotFoundError(f'Missing cache: {Xp}')
    X = np.load(Xp, mmap_mode='r')
    y = np.load(yp)
    return X, y


def subject_split(X, y, dataset):
    """
    Recover per-subject grouping. The cached arrays are stored subject-by-subject
    in load order, so subject boundaries are where the label changes OR at a
    fixed stride for MSU (29 epochs per subject, verified: 84*29 = 2436).
    """
    n = X.shape[0]
    if dataset == 'msu':
        per = 29
        assert n % per == 0, f'MSU cache is {n} epochs, not divisible by {per}'
        bounds = [(i * per, (i + 1) * per) for i in range(n // per)]
    else:
        sid_path = os.path.join(RES_DIR, 'repod_subject_ids.npy')
        if os.path.exists(sid_path):
            sid = np.load(sid_path)
            bounds = []
            start = 0
            for i in range(1, n + 1):
                if i == n or sid[i] != sid[i - 1]:
                    bounds.append((start, i))
                    start = i
        else:
            raise FileNotFoundError(
                'repod_subject_ids.npy not found.\n'
                'RepOD epochs per subject vary (recordings differ in length), so '
                'subject boundaries cannot be inferred from the cache alone.\n'
                'Run:  python cc_tucker_v2.py --rebuild-ids   (see --help)')
    Xs = [np.asarray(X[a:b]) for a, b in bounds]
    ys = [y[a:b] for a, b in bounds]
    idx = [np.arange(a, b) for a, b in bounds]
    return Xs, ys, idx


def rebuild_repod_ids():
    """Recreate repod_subject_ids.npy by re-reading the EDFs in load order."""
    import mne
    d = os.path.join(BASE, 'repOD_dataset', 'dataverse_files')
    files = sorted(f for f in os.listdir(d) if f.lower().endswith('.edf'))
    fs, ep_sec, ov = FS['repod'], 4.0, 0.5
    L = int(ep_sec * fs)
    step = int(L * (1 - ov))
    ids = []
    for si, fn in enumerate(files):
        raw = mne.io.read_raw_edf(os.path.join(d, fn), preload=True, verbose=False)
        n = raw.get_data().shape[1]
        k = 0 if n < L else (n - L) // step + 1
        ids.extend([si] * k)
        print(f'  {fn}: {n} samples -> {k} epochs')
    ids = np.array(ids, dtype=np.int32)
    out = os.path.join(RES_DIR, 'repod_subject_ids.npy')
    np.save(out, ids)
    print(f'\nSaved {out}  ({len(ids)} epochs, {len(files)} subjects)')
    print('Compare this total against repod_X.npy (should be 14396).')
    return ids


# ==============================================================================
# EXPERIMENT DRIVERS
# ==============================================================================

def make_splits(ys, protocol, k=10):
    n = len(ys)
    if protocol == 'loso':
        return [[i] for i in range(n)]
    labels = np.array([y[0] for y in ys])
    gkf = GroupKFold(n_splits=k)
    groups = np.arange(n)
    return [list(te) for _, te in gkf.split(np.zeros(n), labels, groups)]


def strip_heavy(summary):
    """Drop raw prediction arrays for the compact console/JSON summary."""
    out = {}
    for m, r in summary.items():
        out[m] = {k: v for k, v in r.items()
                  if k not in ('preds', 'true', 'proba', 'per_fold')}
    return out


def run_dataset(dataset, args):
    print(f'\n{SEP}\n  DATASET: {dataset.upper()}   basis={args.basis}   '
          f'ranks={args.ranks}   lambda={args.lam}\n{SEP}')
    fs = FS[dataset]
    X, y = load_cached(dataset)
    print(f'  cache: X{X.shape}  y{y.shape}')
    Xs, ys, idx = subject_split(X, y, dataset)
    print(f'  subjects: {len(Xs)}   epochs/subject: '
          f'min={min(len(a) for a in ys)} max={max(len(a) for a in ys)}')

    if args.quick:
        keep = min(8, len(Xs))
        Xs, ys, idx = Xs[:keep], ys[:keep], idx[:keep]
        print(f'  [QUICK MODE] using first {keep} subjects only')

    print('  precomputing fold-invariant features ...')
    gidx = np.concatenate(idx)
    Xall = np.vstack(Xs)
    inv = precompute_invariant(Xall, fs)
    idx_local, c = [], 0
    for a in idx:
        idx_local.append(np.arange(c, c + len(a)))
        c += len(a)
    idx = idx_local
    print(f'  PSD grid: {inv["psd"].shape[2]} bins '
          f'({inv["freqs"][0]:.1f}-{inv["freqs"][-1]:.1f} Hz)')

    results = {}
    protocols = ['loso', 'gkf'] if args.protocol == 'both' else [args.protocol]

    if args.ablation:
        cfg_items = list(CONFIGS.items())
        if args.only:
            want = [c.strip() for c in args.only.split(',')]
            cfg_items = [(c, g) for c, g in cfg_items if c in want]
            missing = [c for c in want if c not in CONFIGS]
            if missing:
                sys.exit(f'ERROR: unknown config(s) in --only: {missing}')
        for cfg, groups in cfg_items:
            print(f'\n  --- ablation {cfg}: {groups} ---')
            results[cfg] = {}
            for p in protocols:
                sp = make_splits(ys, p)
                s = run_cv(Xs, ys, inv, idx, sp, args.ranks, args.lam,
                           args.basis, groups=groups,
                           label=f'{dataset}/{cfg}/{p}', verbose=args.verbose)
                results[cfg][p] = s
        return results

    if args.rank_sweep:
        for r1 in (4, 8, 12):
            for r2 in (10, 20, 30):
                rk = (r1, r2, 6)
                print(f'\n  --- ranks {rk} ---')
                sp = make_splits(ys, protocols[0])
                s = run_cv(Xs, ys, inv, idx, sp, rk, args.lam, args.basis,
                           label=f'{dataset}/ranks{rk}', verbose=False)
                results[f'R{r1}_{r2}'] = strip_heavy(s)
                print(f'    ENSEMBLE acc={s["ENSEMBLE"]["acc"]:.2f}  '
                      f'MCC={s["ENSEMBLE"]["mcc"]:.4f}')
        return results

    if args.lambda_sweep:
        for lam in (0.0, 0.001, 0.01, 0.05, 0.1, 0.5):
            print(f'\n  --- lambda {lam} ---')
            sp = make_splits(ys, protocols[0])
            s = run_cv(Xs, ys, inv, idx, sp, args.ranks, lam, args.basis,
                       label=f'{dataset}/lam{lam}', verbose=False)
            results[f'lam{lam}'] = strip_heavy(s)
            print(f'    ENSEMBLE acc={s["ENSEMBLE"]["acc"]:.2f}  '
                  f'MCC={s["ENSEMBLE"]["mcc"]:.4f}')
        return results

    for p in protocols:
        sp = make_splits(ys, p)
        print(f'\n  --- protocol {p.upper()} ({len(sp)} folds) ---')
        results[p] = run_cv(Xs, ys, inv, idx, sp, args.ranks, args.lam,
                            args.basis, label=f'{dataset}/{p}',
                            verbose=args.verbose)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset', default='both', choices=['repod', 'msu', 'both'])
    ap.add_argument('--basis', default='spectral', choices=['spectral', 'temporal'],
                    help='spectral = D3 fix (phase-invariant); temporal = original')
    ap.add_argument('--protocol', default='both', choices=['loso', 'gkf', 'both'])
    ap.add_argument('--ranks', default='8,20,6')
    ap.add_argument('--lam', type=float, default=0.05)  # sweep-best
    ap.add_argument('--ablation', action='store_true')
    ap.add_argument('--rank-sweep', action='store_true')
    ap.add_argument('--lambda-sweep', action='store_true')
    ap.add_argument('--rebuild-ids', action='store_true',
                    help='regenerate repod_subject_ids.npy from the EDFs')
    ap.add_argument('--blocks', type=int, default=1,
                    help='U1: slabs per subject (try 5 for RepOD)')
    ap.add_argument('--select-mode', default='pooled',
                    choices=['pooled', 'quota'], help='U2')
    ap.add_argument('--quick', action='store_true', help='first 8 subjects only')
    ap.add_argument('--quiet', dest='verbose', action='store_false')
    ap.add_argument('--only', default='',
                    help='restrict --ablation to these configs, e.g. E1 or A2,E1,E3')
    ap.add_argument('--tag', default='')
    args = ap.parse_args()
    args.ranks = tuple(int(v) for v in args.ranks.split(','))
    N_BLOCKS[0] = args.blocks
    SELECT_MODE[0] = args.select_mode

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    if args.rebuild_ids:
        rebuild_repod_ids()
        return

    t0 = time.time()
    stamp = datetime.now().strftime('%Y%m%d_%H%M')
    datasets = ['repod', 'msu'] if args.dataset == 'both' else [args.dataset]

    for ds in datasets:
        res = run_dataset(ds, args)
        mode = ('ablation' if args.ablation else
                'ranksweep' if args.rank_sweep else
                'lamsweep' if args.lambda_sweep else 'main')
        tag = f'_{args.tag}' if args.tag else ''
        fn = os.path.join(
            OUT_DIR,
            f'v3_{ds}_{args.basis}_{mode}_b{args.blocks}'
            f'_{args.select_mode}{tag}_{stamp}.json')
        with open(fn, 'w') as f:
            json.dump(res, f, indent=1, default=float)
        print(f'  saved -> {fn}')

    print(f'\n{SEP}\n  TOTAL TIME: {(time.time()-t0)/60:.1f} min\n{SEP}')


if __name__ == '__main__':
    main()
