#!/usr/bin/env python3
# =============================================================================
# eea_reader_verify.py
# Verifies the .eea file format for the MSU schizophrenia dataset
# and reads both RepOD EDF and MSU EEA files for the first time
#
# RUN FROM PROJECT ROOT:
#   cd /Users/sikhan/Documents/NayaDuarPaper3
#   source eeg_tucker_env/bin/activate
#   python3 scripts/eea_reader_verify.py
#
# OUTPUT: logs/eea_verification.txt
# =============================================================================

import os, sys
import numpy as np
import struct
from pathlib import Path

BASE    = '/Users/sikhan/Documents/NayaDuarPaper3'
MSU_DIR = os.path.join(BASE, 'MSU_Dataset')
SCH_DIR = os.path.join(MSU_DIR, 'sch')
NRM_DIR = os.path.join(MSU_DIR, 'norm')
LOG_DIR = os.path.join(BASE, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'eea_verification.txt')

# ── Tee stdout to file ────────────────────────────────────────────────────────
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files: f.write(obj); f.flush()
    def flush(self):
        for f in self.files: f.flush()

flog = open(LOG_FILE, 'w', encoding='utf-8')
sys.stdout = Tee(sys.__stdout__, flog)

SEP  = '=' * 65
SEP2 = '-' * 65

N_CH     = 16
N_SAMP   = 7680
FS_MSU   = 128
CH_NAMES = ['F7','F3','F4','F8','T3','C3','Cz',
            'C4','T4','T5','P3','Pz','P4','T6','O1','O2']

print(SEP)
print('  MSU .EEA FORMAT VERIFICATION')
print(SEP)

# Collect one file from each class for testing
sch_files  = sorted(Path(SCH_DIR).glob('*.eea'))
norm_files = sorted(Path(NRM_DIR).glob('*.eea'))

print(f'\n  SZ  files found : {len(sch_files)}  in {SCH_DIR}')
print(f'  HC  files found : {len(norm_files)}  in {NRM_DIR}')

# ─────────────────────────────────────────────────────────────────────────────
# ATTEMPT 1 — Try reading as plain text (most likely)
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{SEP2}')
print('  ATTEMPT 1 — Read as plain text (ASCII numbers)')
print(SEP2)

def try_text_read(fp):
    """Try reading .eea as plain text, return (data, method) or (None, reason)"""
    try:
        with open(fp, 'r', errors='ignore') as f:
            content = f.read()

        # Check first 200 chars for readability
        sample = content[:200]
        printable = sum(1 for c in sample if c.isprintable() or c in '\n\r\t')
        ratio = printable / max(len(sample), 1)

        print(f'\n  First 200 characters of file:')
        print(f'  |{repr(content[:200])}|')
        print(f'  Printable ratio: {ratio:.3f}')

        if ratio > 0.85:
            # Likely text file — try parsing
            arr = np.fromstring(content, dtype=float, sep='\n')
            if len(arr) == 0:
                arr = np.fromstring(content, dtype=float, sep=' ')
            if len(arr) == 0:
                # try genfromtxt
                from io import StringIO
                arr = np.genfromtxt(StringIO(content),
                                    invalid_raise=False).flatten()
                arr = arr[np.isfinite(arr)]
            return arr, 'text'
        else:
            return None, f'Binary (printable ratio={ratio:.3f})'
    except Exception as e:
        return None, str(e)

# Test on first SZ file
test_file = sch_files[0] if sch_files else None
if test_file:
    print(f'\n  Test file: {test_file.name}')
    print(f'  File size: {os.path.getsize(test_file)/1024:.1f} KB')

    arr, method = try_text_read(test_file)

    if arr is not None:
        print(f'\n  Method: {method}')
        print(f'  Values read     : {len(arr)}')
        print(f'  Expected        : {N_CH * N_SAMP} ({N_CH}ch x {N_SAMP} samples)')
        print(f'  Match           : {len(arr) == N_CH * N_SAMP}')

        if len(arr) == N_CH * N_SAMP:
            data = arr.reshape(N_CH, N_SAMP)
            print(f'  Reshape         : SUCCESS → {data.shape}')
            print(f'  Value range     : [{data.min():.3f}, {data.max():.3f}] µV')
            print(f'  Mean            : {data.mean():.4f} µV')
            print(f'  Std             : {data.std():.4f} µV')
            print(f'\n  Per-channel stats:')
            print(f'  {"Ch":4s} {"Mean":>10s} {"Std":>10s} '
                  f'{"Min":>10s} {"Max":>10s}')
            print(f'  {"-"*4} {"-"*10} {"-"*10} {"-"*10} {"-"*10}')
            for i in range(N_CH):
                d = data[i]
                print(f'  {CH_NAMES[i]:4s} {d.mean():10.3f} '
                      f'{d.std():10.3f} {d.min():10.3f} {d.max():10.3f}')
        elif len(arr) > 0:
            print(f'  Partial data: {len(arr)} values. '
                  f'Trying reshape ...')
            best_div = [(r, len(arr)//r)
                        for r in [16,19,14,18,20]
                        if len(arr) % r == 0]
            print(f'  Possible (ch, samples): {best_div}')
    else:
        print(f'  Text read failed: {method}')

# ─────────────────────────────────────────────────────────────────────────────
# ATTEMPT 2 — Try reading as binary (float32 or float64)
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{SEP2}')
print('  ATTEMPT 2 — Read as binary (float32, float64, int16, int32)')
print(SEP2)

def try_binary_read(fp):
    with open(fp, 'rb') as f:
        raw = f.read()
    n_bytes = len(raw)
    print(f'\n  Raw bytes: {n_bytes}')
    print(f'  First 32 bytes (hex): '
          f'{raw[:32].hex()}')
    print(f'  First 32 bytes (repr): {repr(raw[:32])}')

    results = {}
    for dtype, bpp in [('float32',4),('float64',8),
                        ('int16',2),('int32',4)]:
        n_vals = n_bytes // bpp
        try:
            arr = np.frombuffer(raw, dtype=dtype)
            results[dtype] = (len(arr), arr[:5].tolist(),
                              float(arr.min()), float(arr.max()))
            match = (n_vals == N_CH * N_SAMP)
            print(f'  {dtype:8s}: {n_vals:7d} values  '
                  f'match={match}  '
                  f'range=[{arr.min():.3f},{arr.max():.3f}]  '
                  f'first5={arr[:5].tolist()}')
        except Exception as e:
            print(f'  {dtype:8s}: ERROR - {e}')
    return results

if test_file:
    bin_results = try_binary_read(test_file)

# ─────────────────────────────────────────────────────────────────────────────
# ATTEMPT 3 — Try with header skip (some EEA files have short headers)
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{SEP2}')
print('  ATTEMPT 3 — Try skipping header bytes (if any)')
print(SEP2)

if test_file:
    with open(test_file, 'rb') as f:
        raw = f.read()
    expected = N_CH * N_SAMP
    print(f'\n  Looking for float32 block of size {expected} = '
          f'{expected*4} bytes ...')
    for skip in [0, 2, 4, 8, 16, 32, 64, 128, 256, 512]:
        chunk = raw[skip:]
        n = len(chunk) // 4
        if abs(n - expected) < 100:
            arr = np.frombuffer(chunk[:expected*4], dtype=np.float32)
            if np.all(np.isfinite(arr)) and -5000 < arr.min() < arr.max() < 5000:
                print(f'  Skip {skip:4d} bytes → float32 range: '
                      f'[{arr.min():.2f}, {arr.max():.2f}] µV  '
                      f'← PLAUSIBLE EEG range')
            else:
                print(f'  Skip {skip:4d} bytes → float32 range: '
                      f'[{arr.min():.2f}, {arr.max():.2f}] '
                      f'(out of EEG range)')

# ─────────────────────────────────────────────────────────────────────────────
# ATTEMPT 4 — Try reading as EDF/EEA with MNE
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{SEP2}')
print('  ATTEMPT 4 — Try reading with MNE (in case EEA is EDF-like)')
print(SEP2)

if test_file:
    try:
        import mne
        mne.set_log_level('WARNING')
        raw_mne = mne.io.read_raw_edf(str(test_file),
                                       preload=True, verbose=False)
        print(f'  MNE read SUCCESS!')
        print(f'  Channels: {raw_mne.ch_names}')
        print(f'  Sfreq   : {raw_mne.info["sfreq"]} Hz')
        print(f'  Duration: {raw_mne.times[-1]:.2f} s')
        data_mne = raw_mne.get_data()
        print(f'  Shape   : {data_mne.shape}')
        print(f'  Range   : [{data_mne.min()*1e6:.3f}, '
              f'{data_mne.max()*1e6:.3f}] µV')
    except Exception as e:
        print(f'  MNE read failed: {e}')

# ─────────────────────────────────────────────────────────────────────────────
# ATTEMPT 5 — Print exact first/last lines as text
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{SEP2}')
print('  ATTEMPT 5 — Raw content inspection (first and last 20 lines)')
print(SEP2)

if test_file:
    with open(test_file, 'r', errors='replace') as f:
        all_lines = f.readlines()
    print(f'\n  Total lines in file: {len(all_lines)}')
    print(f'\n  First 20 lines:')
    for i, l in enumerate(all_lines[:20]):
        print(f'  [{i+1:4d}] {repr(l.rstrip())}')
    print(f'\n  Last 10 lines:')
    for i, l in enumerate(all_lines[-10:]):
        print(f'  [{len(all_lines)-10+i+1:4d}] {repr(l.rstrip())}')

    # Check if all non-empty lines are numeric
    non_empty = [l.strip() for l in all_lines if l.strip()]
    numeric_count = sum(1 for l in non_empty
                        if l.replace('.','',1).replace('-','',1).strip().isdigit()
                        or l.replace('.','',1).replace('-','',1)
                           .replace('e','',1).replace('E','',1)
                           .replace('+','',1).isdigit())
    print(f'\n  Non-empty lines  : {len(non_empty)}')
    print(f'  Numeric lines    : {numeric_count}')
    print(f'  Numeric ratio    : {numeric_count/max(len(non_empty),1):.3f}')
    print(f'  Expected numeric : {N_CH * N_SAMP}')

# ─────────────────────────────────────────────────────────────────────────────
# ALSO VERIFY REPOD EDF
# ─────────────────────────────────────────────────────────────────────────────
print(f'\n{SEP}')
print('  REPOD EDF VERIFICATION')
print(SEP)

# Find RepOD folder
repod_dir = None
for d in os.listdir(BASE):
    dl = d.lower()
    fp = os.path.join(BASE, d)
    if os.path.isdir(fp) and any(k in dl for k in
                                  ['repod','eeg_schiz','ipn','poland','sz']):
        repod_dir = fp
        break

if repod_dir is None:
    for d in os.listdir(BASE):
        fp = os.path.join(BASE, d)
        if os.path.isdir(fp) and d not in ['MSU_Dataset','eeg_tucker_env',
                                             'data','outputs','scripts','logs']:
            edf_count = len(list(Path(fp).glob('**/*.edf')))
            if edf_count > 0:
                repod_dir = fp
                break

if repod_dir:
    print(f'\n  RepOD folder: {repod_dir}')
    edf_files = sorted(Path(repod_dir).glob('**/*.edf'))
    print(f'  EDF files   : {len(edf_files)}')

    try:
        import mne
        mne.set_log_level('WARNING')

        print(f'\n  Inspecting first SZ and first HC file:')
        for fp in edf_files[:2]:
            print(f'\n  File: {fp.name}')
            raw = mne.io.read_raw_edf(str(fp), preload=True, verbose=False)
            data = raw.get_data()
            print(f'  Channels    : {raw.ch_names}')
            print(f'  N channels  : {raw.info["nchan"]}')
            print(f'  Sfreq       : {raw.info["sfreq"]} Hz')
            print(f'  Duration    : {raw.times[-1]:.2f} s '
                  f'({raw.times[-1]/60:.2f} min)')
            print(f'  Shape       : {data.shape}  (ch x samples)')
            print(f'  Range (µV)  : [{data.min()*1e6:.3f}, '
                  f'{data.max()*1e6:.3f}]')
            print(f'  NaN count   : {np.isnan(data).sum()}')
    except ImportError:
        print('  MNE not available in this environment.')
    except Exception as e:
        print(f'  ERROR: {e}')
else:
    print('  RepOD folder not found automatically.')
    print('  Please check folder name in BASE directory.')

print(f'\n{SEP}')
print('  VERIFICATION COMPLETE')
print(f'  Log saved to: {LOG_FILE}')
print(SEP)

sys.stdout = sys.__stdout__
flog.close()
print(f'\nDone. Log saved to: {LOG_FILE}')
