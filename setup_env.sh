#!/bin/bash
# =============================================================================
# setup_env.sh
# Creates a virtual environment for NayaDuarPaper3 (Tucker-HOSVD EEG Project)
# HOW TO RUN:
#   cd /Users/sikhan/Documents/NayaDuarPaper3
#   chmod +x setup_env.sh
#   ./setup_env.sh
# =============================================================================

set -e   # stop on any error

PROJECT_DIR="/Users/sikhan/Documents/NayaDuarPaper3"
ENV_NAME="eeg_tucker_env"
ENV_PATH="$PROJECT_DIR/$ENV_NAME"

echo "============================================================"
echo "  NayaDuarPaper3 — Virtual Environment Setup"
echo "  Tucker-HOSVD EEG Schizophrenia Detection"
echo "============================================================"
echo ""
echo "  Project dir : $PROJECT_DIR"
echo "  Environment : $ENV_PATH"
echo ""

# ── Step 1: Navigate to project dir ──────────────────────────────────────────
cd "$PROJECT_DIR"
echo "[1/6] Changed to project directory: $(pwd)"

# ── Step 2: Create virtual environment ───────────────────────────────────────
echo ""
echo "[2/6] Creating virtual environment: $ENV_NAME ..."
python3 -m venv "$ENV_NAME"
echo "      Done."

# ── Step 3: Activate environment ─────────────────────────────────────────────
echo ""
echo "[3/6] Activating environment ..."
source "$ENV_PATH/bin/activate"
echo "      Python: $(which python3)"
echo "      Pip   : $(which pip)"
echo "      Python version: $(python3 --version)"

# ── Step 4: Upgrade pip ───────────────────────────────────────────────────────
echo ""
echo "[4/6] Upgrading pip ..."
pip install --upgrade pip --quiet
echo "      pip version: $(pip --version)"

# ── Step 5: Install all required packages ─────────────────────────────────────
echo ""
echo "[5/6] Installing packages ..."
echo "      This may take 3-5 minutes ..."
echo ""

pip install --quiet \
    numpy \
    scipy \
    matplotlib \
    seaborn \
    scikit-learn \
    pandas \
    mne \
    tensorly \
    pyriemann \
    tqdm \
    antropy \
    emd \
    joblib \
    ipykernel \
    jupyter

echo ""
echo "      All packages installed successfully."

# ── Step 6: Verify installations ─────────────────────────────────────────────
echo ""
echo "[6/6] Verifying installations ..."
echo ""

python3 - << 'PYCHECK'
packages = [
    ('numpy',       'np'),
    ('scipy',       'scipy'),
    ('matplotlib',  'matplotlib'),
    ('seaborn',     'sns'),
    ('sklearn',     'sklearn'),
    ('pandas',      'pd'),
    ('mne',         'mne'),
    ('tensorly',    'tl'),
    ('pyriemann',   'pyriemann'),
    ('tqdm',        'tqdm'),
    ('antropy',     'ant'),
    ('emd',         'emd'),
    ('joblib',      'joblib'),
]

print(f'  {"Package":<20s} {"Version":<15s} {"Status"}')
print(f'  {"-"*20} {"-"*15} {"-"*8}')
all_ok = True
for pkg, alias in packages:
    try:
        mod = __import__(pkg)
        ver = getattr(mod, '__version__', 'unknown')
        print(f'  {pkg:<20s} {ver:<15s} OK')
    except ImportError as e:
        print(f'  {pkg:<20s} {"---":<15s} FAILED  ({e})')
        all_ok = False

print()
if all_ok:
    print('  All packages verified successfully.')
else:
    print('  Some packages failed. Run the failed ones manually.')
PYCHECK

# ── Create project folder structure ──────────────────────────────────────────
echo ""
echo "  Creating project folder structure ..."
mkdir -p "$PROJECT_DIR/data/RepOD"
mkdir -p "$PROJECT_DIR/data/MSU"
mkdir -p "$PROJECT_DIR/outputs/results"
mkdir -p "$PROJECT_DIR/outputs/figures"
mkdir -p "$PROJECT_DIR/outputs/models"
mkdir -p "$PROJECT_DIR/scripts"
mkdir -p "$PROJECT_DIR/logs"

echo ""
echo "  Folder structure created:"
echo "  $PROJECT_DIR/"
echo "  ├── $ENV_NAME/          (virtual environment)"
echo "  ├── data/"
echo "  │   ├── RepOD/          (place RepOD EDF files here)"
echo "  │   └── MSU/            (place MSU TXT files here)"
echo "  ├── outputs/"
echo "  │   ├── results/        (saved .npy, .json results)"
echo "  │   ├── figures/        (saved .svg figures)"
echo "  │   └── models/         (saved model weights)"
echo "  ├── scripts/            (all Python scripts go here)"
echo "  └── logs/               (experiment logs)"

# ── Print activation instructions ────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  SETUP COMPLETE"
echo "============================================================"
echo ""
echo "  HOW TO ACTIVATE THIS ENVIRONMENT (every new terminal session):"
echo ""
echo "    cd /Users/sikhan/Documents/NayaDuarPaper3"
echo "    source $ENV_NAME/bin/activate"
echo ""
echo "  HOW TO RUN ANY SCRIPT:"
echo ""
echo "    python3 scripts/Dataset_Explorer.py"
echo "    python3 scripts/pipeline.py"
echo ""
echo "  HOW TO DEACTIVATE:"
echo ""
echo "    deactivate"
echo ""
echo "  HOW TO CHECK ACTIVE ENVIRONMENT:"
echo ""
echo "    which python3"
echo "    (should show: $ENV_PATH/bin/python3)"
echo ""
echo "============================================================"
