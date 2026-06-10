from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── Data paths (edit to match your local setup) ──────────────────────────────
ISIC_DATA_ROOT = ROOT / "data" / "ISIC2016"
PH2_IMAGE_DIR  = ROOT / "data" / "PH2Dataset" / "image"
PH2_LABEL_DIR  = ROOT / "data" / "PH2Dataset" / "label"

# ── Output paths ─────────────────────────────────────────────────────────────
RESULTS_ROOT = ROOT / "results"
FIGURES_DIR  = ROOT / "figures"
VIZ_DIR      = ROOT / "visualizations"

# ── Hyperparameters ───────────────────────────────────────────────────────────
IMAGE_SIZE  = 256
BATCH_SIZE  = 32
NUM_EPOCHS  = 50
LR          = 1e-4
NUM_WORKERS = 4
MC_PASSES   = 20
ECE_BINS    = 10
DROPOUT_P   = 0.3
SEED        = 42

# ── Experiment registry ───────────────────────────────────────────────────────
STRATEGIES = ["baseline", "geometric", "intensity", "mixing", "occlusion"]
LABELS     = ["Baseline", "Geometric", "Intensity", "Mixing", "Occlusion"]
COLORS     = ["#2166AC", "#4DAC26", "#D6604D", "#8073AC", "#E08214"]
