# =============================================================================
# Dataset_Explorer.py
# Deep exploration of RepOD and MSU EEG Schizophrenia datasets
# Output saved to: /Users/sikhan/Documents/NayaDuarPaper3/dataset_exploration.txt
# HOW TO RUN: Kernel → Restart → F5
# =============================================================================

import os
import glob
import numpy as np
import scipy.io as sio
from pathlib import Path
import sys
from collections import Counter

BASE_DIR = '/Users/sikhan/Documents/NayaDuarPaper3'
OUT_FILE = os.path.join(BASE_DIR, 'dataset_exploration.txt')

# ── redirect all print to both console and file ───────────────────────────────
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj); f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

fout = open(OUT_FILE, 'w', encoding='utf-8')
sys.stdout = Tee(sys.__stdout__, fout)

SEP  = '=' * 70
SEP2 = '-' * 70
SEP3 = '.' * 50

def section(title):
    print(f'\n{SEP}')
    print(f'  {title}')
    print(SEP)

def subsection(title):
    print(f'\n{SEP2}')
    print(f'  {title}')
    print(SEP2)

def note(msg):
    print(f'  >> {msg}')

# =============================================================================
# PART 0 — TOP-LEVEL FOLDER STRUCTURE
# =============================================================================
section('PART 0 — TOP-LEVEL FOLDER STRUCTURE')
print(f'\n  Base directory: {BASE_DIR}\n')

for root, dirs, files in os.walk(BASE_DIR):
    level  = root.replace(BASE_DIR, '').count(os.sep)
    indent = '    ' * level
    print(f'{indent}{os.path.basename(root)}/')
    sub_indent = '    ' * (level + 1)
    for f in sorted(files):
        fp   = os.path.join(root, f)
        size = os.path.getsize(fp)
        if size < 1024:
            sz_str = f'{size} B'
        elif size < 1024**2:
            sz_str = f'{size/1024:.1f} KB'
        elif size < 1024**3:
            sz_str = f'{size/1024**2:.2f} MB'
        else:
            sz_str = f'{size/1024**3:.2f} GB'
        print(f'{sub_indent}{f}  [{sz_str}]')

# =============================================================================
# HELPER — collect all files recursively from a directory
# =============================================================================
def collect_files(folder):
    result = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            result.append(os.path.join(root, f))
    return result

# =============================================================================
# PART 1 — REPOD DATASET EXPLORATION
# =============================================================================
section('PART 1 — RepOD DATASET (Olejarczyk & Jernajczyk 2017)')

# Find RepOD folder
all_subdirs = [os.path.join(BASE_DIR, d)
               for d in sorted(os.listdir(BASE_DIR))
               if os.path.isdir(os.path.join(BASE_DIR, d))]

repod_dir = None
for d in all_subdirs:
    dl = os.path.basename(d).lower()
    if any(k in dl for k in ['repod','warsaw','ipn','olejarczyk',
                              'schiz','poland']):
        repod_dir = d
        break

if repod_dir is None and len(all_subdirs) >= 1:
    repod_dir = all_subdirs[0]
    print(f'  [AUTO] Using first subdir as RepOD: {repod_dir}')
else:
    print(f'\n  RepOD folder identified: {repod_dir}')

repod_files = collect_files(repod_dir) if repod_dir else []

# Extension summary
subsection('1A — File Extension Summary')
exts = Counter(Path(f).suffix.lower() for f in repod_files)
for ext, cnt in sorted(exts.items(), key=lambda x: -x[1]):
    print(f'  {ext if ext else "(no extension)":20s}  {cnt:4d} files')

# EDF exploration
edf_files = [f for f in repod_files
             if Path(f).suffix.lower() in ['.edf','.bdf','.gdf']]

subsection(f'1B — EDF/BDF Files ({len(edf_files)} found)')
for f in sorted(edf_files):
    sz = os.path.getsize(f)
    print(f'  {os.path.basename(f):40s}  [{sz/1024**2:.2f} MB]')

if edf_files:
    try:
        import mne
        mne.set_log_level('WARNING')

        subsection('1C — EDF Deep Inspection (up to 6 files)')
        for fp in sorted(edf_files)[:6]:
            print(f'\n  {SEP3}')
            print(f'  File : {os.path.basename(fp)}')
            print(f'  Path : {fp}')
            print(f'  {SEP3}')
            try:
                raw = mne.io.read_raw_edf(fp, preload=True, verbose=False)
                info = raw.info
                data = raw.get_data()
                print(f'  Channels ({info["nchan"]}): {raw.ch_names}')
                print(f'  Sampling rate    : {info["sfreq"]} Hz')
                print(f'  Duration         : {raw.times[-1]:.2f} s '
                      f'({raw.times[-1]/60:.2f} min)')
                print(f'  Total samples    : {data.shape[1]}')
                print(f'  Data shape       : {data.shape}  '
                      f'(n_channels x n_samples)')
                print(f'  Amplitude range  : '
                      f'[{data.min()*1e6:.3f}, {data.max()*1e6:.3f}] µV')
                print(f'  Global mean (µV) : {data.mean()*1e6:.4f}')
                print(f'  Global std  (µV) : {data.std()*1e6:.4f}')
                print(f'  NaN count        : {np.isnan(data).sum()}')
                print(f'  Inf count        : {np.isinf(data).sum()}')
                print(f'\n  Per-channel stats (µV):')
                print(f'  {"Channel":10s} {"Mean":>10s} {"Std":>10s} '
                      f'{"Min":>10s} {"Max":>10s}')
                print(f'  {"-"*10} {"-"*10} {"-"*10} {"-"*10} {"-"*10}')
                for i, ch in enumerate(raw.ch_names):
                    d = data[i] * 1e6
                    print(f'  {ch:10s} {d.mean():10.3f} {d.std():10.3f} '
                          f'{d.min():10.3f} {d.max():10.3f}')
            except Exception as e:
                print(f'  ERROR: {e}')

        subsection('1D — Subject Grouping from Filenames')
        basenames = [os.path.basename(f) for f in sorted(edf_files)]
        print(f'\n  All EDF filenames ({len(basenames)}):')
        for b in basenames:
            print(f'    {b}')
        sz_f = [b for b in basenames if any(k in b.lower()
                for k in ['sz','schiz','patient','h_','s_','s0','s1'])]
        hc_f = [b for b in basenames if any(k in b.lower()
                for k in ['hc','healthy','control','h0','n_','normal'])]
        print(f'\n  Inferred SZ files : {len(sz_f)} -> {sz_f}')
        print(f'  Inferred HC files : {len(hc_f)} -> {hc_f}')
        rem = len(basenames) - len(sz_f) - len(hc_f)
        print(f'  Unclassified      : {rem} files')

    except ImportError:
        note('mne not installed. Run: pip install mne --break-system-packages')

# MAT files
mat_files = [f for f in repod_files if Path(f).suffix.lower() == '.mat']
if mat_files:
    subsection(f'1E — MAT Files ({len(mat_files)} found)')
    for fp in sorted(mat_files)[:5]:
        print(f'\n  File: {os.path.basename(fp)}')
        try:
            mat = sio.loadmat(fp)
            keys = [k for k in mat.keys() if not k.startswith('_')]
            print(f'  Keys: {keys}')
            for k in keys:
                v = mat[k]
                if hasattr(v,'shape'):
                    print(f'    {k}: shape={v.shape} dtype={v.dtype} '
                          f'range=[{np.nanmin(v):.3f},{np.nanmax(v):.3f}]')
                else:
                    print(f'    {k}: {v}')
        except Exception as e:
            print(f'  ERROR: {e}')

# TXT/CSV in RepOD
txt_repod = [f for f in repod_files
             if Path(f).suffix.lower() in ['.txt','.csv']]
if txt_repod:
    subsection(f'1F — TXT/CSV Files in RepOD ({len(txt_repod)} found)')
    for fp in sorted(txt_repod)[:5]:
        print(f'\n  File: {os.path.basename(fp)}')
        try:
            with open(fp,'r',errors='ignore') as fh:
                lines = fh.readlines()
            print(f'  Lines: {len(lines)}')
            for l in lines[:5]:
                print(f'    {l.rstrip()}')
        except Exception as e:
            print(f'  ERROR: {e}')

# Any other extension in RepOD
other_repod = [f for f in repod_files
               if Path(f).suffix.lower() not in
               ['.edf','.bdf','.gdf','.mat','.txt','.csv']]
if other_repod:
    subsection(f'1G — Other Files in RepOD ({len(other_repod)})')
    for f in sorted(other_repod)[:10]:
        print(f'  {os.path.basename(f)}  '
              f'[{os.path.getsize(f)/1024:.1f} KB]')

# =============================================================================
# PART 2 — MSU DATASET EXPLORATION
# =============================================================================
section('PART 2 — MSU DATASET (Gorbachevskaya & Borisov 2002)')

msu_dir = None
for d in all_subdirs:
    dl = os.path.basename(d).lower()
    if any(k in dl for k in ['msu','moscow','gorbach','borisov',
                              'adolescent','brain']):
        msu_dir = d
        break

if msu_dir is None and len(all_subdirs) >= 2:
    msu_dir = all_subdirs[1]
    print(f'  [AUTO] Using second subdir as MSU: {msu_dir}')
elif msu_dir is None and repod_dir and len(all_subdirs) >= 1:
    remaining = [d for d in all_subdirs if d != repod_dir]
    if remaining:
        msu_dir = remaining[0]
        print(f'  [AUTO] Using remaining subdir as MSU: {msu_dir}')
else:
    print(f'\n  MSU folder identified: {msu_dir}')

msu_files = collect_files(msu_dir) if msu_dir else []

# Extension summary
subsection('2A — File Extension Summary')
exts_msu = Counter(Path(f).suffix.lower() for f in msu_files)
for ext, cnt in sorted(exts_msu.items(), key=lambda x: -x[1]):
    print(f'  {ext if ext else "(no extension)":20s}  {cnt:4d} files')

# All TXT / no-extension files
msu_txt = [f for f in msu_files
           if Path(f).suffix.lower() in ['.txt','','.dat']]

subsection(f'2B — TXT/DAT Files ({len(msu_txt)} found) — Full List')
for f in sorted(msu_txt):
    sz = os.path.getsize(f)
    print(f'  {os.path.basename(f):40s}  [{sz/1024:.1f} KB]')

# Deep inspection of MSU TXT files
N_CH_MSU   = 16
N_SAMP_MSU = 7680
CH_MSU     = ['F7','F3','F4','F8','T3','C3','Cz',
               'C4','T4','T5','P3','Pz','P4','T6','O1','O2']

subsection('2C — TXT Deep Inspection (up to 6 files)')
for fp in sorted(msu_txt)[:6]:
    print(f'\n  {SEP3}')
    print(f'  File : {os.path.basename(fp)}')
    print(f'  Path : {fp}')
    print(f'  Size : {os.path.getsize(fp)/1024:.1f} KB')
    print(f'  {SEP3}')
    try:
        # Try reading as single column of numbers
        with open(fp,'r',errors='ignore') as fh:
            raw_lines = fh.readlines()
        print(f'  Total lines in file: {len(raw_lines)}')
        print(f'  First 5 lines raw  :')
        for l in raw_lines[:5]:
            print(f'    |{l.rstrip()}|')

        # Try genfromtxt
        arr = np.genfromtxt(fp, invalid_raise=False)
        arr_flat = arr.flatten()
        arr_clean = arr_flat[np.isfinite(arr_flat)]
        print(f'  Raw array shape    : {arr.shape}')
        print(f'  Clean values count : {len(arr_clean)}')
        print(f'  Expected (16×7680) : {N_CH_MSU * N_SAMP_MSU}')
        print(f'  Matches expected   : {len(arr_clean) == N_CH_MSU * N_SAMP_MSU}')

        # Reshape
        if len(arr_clean) == N_CH_MSU * N_SAMP_MSU:
            data = arr_clean.reshape(N_CH_MSU, N_SAMP_MSU)
            print(f'  Reshaped to        : {data.shape} (ch x samples)')
        elif len(arr_clean) > 0 and len(arr_clean) % N_CH_MSU == 0:
            s = len(arr_clean) // N_CH_MSU
            data = arr_clean.reshape(N_CH_MSU, s)
            print(f'  Reshaped to        : {data.shape} '
                  f'(non-standard sample count)')
        elif len(arr_clean) > 0:
            data = arr_clean.reshape(1, -1)
            print(f'  Could not reshape to 16ch. Flat: {data.shape}')
        else:
            print(f'  No valid numeric data found.')
            data = None

        if data is not None and data.shape[0] == N_CH_MSU:
            print(f'  Amplitude range (µV): [{data.min():.3f}, {data.max():.3f}]')
            print(f'  Global mean (µV)    : {data.mean():.4f}')
            print(f'  Global std  (µV)    : {data.std():.4f}')
            print(f'  NaN count           : {np.isnan(data).sum()}')
            print(f'\n  Per-channel stats (µV):')
            print(f'  {"Ch":6s} {"Name":4s} {"Mean":>10s} {"Std":>10s} '
                  f'{"Min":>10s} {"Max":>10s}')
            print(f'  {"-"*6} {"-"*4} {"-"*10} {"-"*10} {"-"*10} {"-"*10}')
            for i in range(N_CH_MSU):
                d  = data[i]
                ch = CH_MSU[i]
                print(f'  {i:6d} {ch:4s} {d.mean():10.3f} {d.std():10.3f} '
                      f'{d.min():10.3f} {d.max():10.3f}')
        elif data is not None:
            print(f'  Amplitude range: [{data.min():.3f}, {data.max():.3f}]')

    except Exception as e:
        print(f'  ERROR: {e}')

# Subject grouping
subsection('2D — Subject Grouping from Filenames')
basenames_msu = [os.path.basename(f) for f in sorted(msu_txt)]
print(f'\n  All filenames ({len(basenames_msu)}):')
for b in sorted(basenames_msu):
    print(f'    {b}')

# Subdirectory structure
subsection('2E — MSU Full Directory Tree')
if msu_dir:
    for root, dirs, files in os.walk(msu_dir):
        level  = root.replace(msu_dir,'').count(os.sep)
        indent = '  ' * (level + 1)
        print(f'{indent}{os.path.basename(root)}/  ({len(files)} files)')
        for d in sorted(dirs):
            print(f'{indent}  [{d}/]')
        for f in sorted(files)[:30]:
            fp2 = os.path.join(root,f)
            print(f'{indent}  {f}  '
                  f'[{os.path.getsize(fp2)/1024:.1f} KB]')
        if len(files) > 30:
            print(f'{indent}  ... and {len(files)-30} more files')

# =============================================================================
# PART 3 — CROSS-DATASET SUMMARY AND TENSOR STRUCTURE
# =============================================================================
section('PART 3 — CROSS-DATASET SUMMARY AND TUCKER-HOSVD TENSOR STRUCTURE')

print(f'''
  DATASET COMPARISON
  ─────────────────────────────────────────────────────────────────────
  Property              RepOD (Warsaw/IPN)     MSU (Moscow)
  ─────────────────── ─────────────────────── ──────────────────────
  Subjects (SZ)         14                     45
  Subjects (HC)         14                     39
  Total subjects        28                     84
  EEG channels          19                     16
  Sampling rate (Hz)    250                    128
  Recording format      EDF                    TXT (flat column)
  Resting state         Yes (eyes closed)      Yes (resting)
  Population            Adults                 Adolescents
  File count (approx)   28 EDF                 84 TXT
  ─────────────────────────────────────────────────────────────────────

  TUCKER-HOSVD TENSOR STRUCTURE (3-way tensor per epoch)
  ─────────────────────────────────────────────────────────────────────
  Tensor shape = (Channels x Time_samples x Epochs/Subjects)

  At epoch length 4 seconds:
    RepOD: each epoch = 4 x 250 =  1000 samples
           Tensor per subject  : (19, 1000, N_epochs)
           Grand tensor        : (19, 1000, N_total_epochs)

    MSU  : each epoch = 4 x 128 =   512 samples
           Tensor per subject  : (16,  512, N_epochs)
           Grand tensor        : (16,  512, N_total_epochs)

  Common channel set (16 channels, intersection of both datasets):
    F7, F3, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2

  If using all RepOD channels (19):
    Extra channels in RepOD: to be confirmed from EDF channel names
    above (likely Fp1, Fp2, Fz or similar)

  Tucker decomposition modes:
    Mode 1 (Channel mode)  : spatial decomposition
    Mode 2 (Time mode)     : temporal decomposition
    Mode 3 (Epoch mode)    : trial/subject decomposition

  Sparse regularisation will be applied on the core tensor G
  to enforce sparsity and reduce overfitting.
  ─────────────────────────────────────────────────────────────────────
''')

# =============================================================================
# PART 4 — ANY OTHER FILES IN BASE DIR
# =============================================================================
section('PART 4 — OTHER FILES DIRECTLY IN BASE DIRECTORY')

other = [f for f in os.listdir(BASE_DIR)
         if os.path.isfile(os.path.join(BASE_DIR, f))]
print(f'\n  Files directly in base dir: {len(other)}')
for f in sorted(other):
    fp = os.path.join(BASE_DIR, f)
    sz = os.path.getsize(fp)
    print(f'  {f}  [{sz/1024:.1f} KB]')
    if Path(f).suffix.lower() in ['.txt','.md','.csv','.readme']:
        try:
            with open(fp,'r',errors='ignore') as fh:
                lines = fh.readlines()[:15]
            for l in lines:
                print(f'    {l.rstrip()}')
        except Exception:
            pass

print(f'\n{SEP}')
print(f'  EXPLORATION COMPLETE')
print(f'  Results saved to: {OUT_FILE}')
print(SEP)

sys.stdout = sys.__stdout__
fout.close()
print(f'\nDone. Output file: {OUT_FILE}')
