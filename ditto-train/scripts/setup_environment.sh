#!/usr/bin/env bash
# =============================================================================
# setup_environment.sh — RunPod Environment Setup for Bridge-Ditto Training
# =============================================================================
# Run once at the start of a RunPod session.
# Installs all dependencies needed for:
#   1. Mimi encoder (audio tokenization)
#   2. Bridge module (token → feature conversion)
#   3. Ditto training (MotionDiT diffusion model)
#
# Usage:
#   bash scripts/setup_environment.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DITTO_TRAIN_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$DITTO_TRAIN_DIR")"

BOLD="\033[1m"
GREEN="\033[1;32m"
CYAN="\033[1;36m"
YELLOW="\033[1;33m"
RESET="\033[0m"

log() { echo -e "${CYAN}[setup]${RESET} $*"; }
ok()  { echo -e "${GREEN}[setup]${RESET} ✅  $*"; }
warn(){ echo -e "${YELLOW}[setup]${RESET} ⚠️  $*"; }

echo -e "${BOLD}"
echo "═══════════════════════════════════════════════════════════════"
echo "   Bridge-Ditto Training — Environment Setup"
echo "═══════════════════════════════════════════════════════════════"
echo -e "${RESET}"

# ── 1. System dependencies ────────────────────────────────────────────────────
log "Step 1/6 — System packages..."
apt-get update -qq
apt-get install -y -qq ffmpeg libsndfile1 > /dev/null 2>&1
ok "System packages installed."

# ── 2. PyTorch (should already be installed on RunPod) ────────────────────────
log "Step 2/6 — Checking PyTorch..."
python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')" \
    || { warn "PyTorch not found! Installing..."; pip install torch torchvision torchaudio; }
# Moshi / HF helpers expect torch>=2.4; RunPod images often ship 2.2.x.
log "Step 2b/6 — Ensuring PyTorch >= 2.4 (CUDA 12.1 wheels)..."
python -c "import sys,torch; v=torch.__version__.split('+')[0].split('.'); m,M=int(v[0]),int(v[1]); sys.exit(0 if m>2 or (m==2 and M>=4) else 1)" 2>/dev/null \
    || {
        warn "Upgrading PyTorch (recommended for this repo)..."
        pip install --quiet --upgrade \
            "torch>=2.4.0,<2.8.0" "torchvision>=0.19.0,<0.23.0" "torchaudio>=2.4.0,<2.8.0" \
            --index-url https://download.pytorch.org/whl/cu121
    }
ok "PyTorch ready."

# ── 3. Moshi (for Mimi encoder) ──────────────────────────────────────────────
log "Step 3/6 — Installing Moshi (Mimi encoder)..."
pip install --quiet -e "${PROJECT_ROOT}/moshi-inference" 2>/dev/null || true
pip install --quiet \
    "huggingface_hub>=0.24,<1.0.0" \
    "safetensors>=0.4.0" \
    "sentencepiece>=0.2.0,<0.3" \
    "sphn>=0.2.0,<0.3.0"
ok "Moshi/Mimi installed."

# ── 4. Audio processing ──────────────────────────────────────────────────────
log "Step 4/6 — Installing audio processing packages..."
pip install --quiet \
    "torchaudio" \
    "librosa>=0.10.0" \
    "soundfile>=0.12.0" \
    "pyworld>=0.3.4" \
    "pyyaml>=6.0"
ok "Audio packages installed."

# ── 5. Ditto training dependencies ───────────────────────────────────────────
log "Step 5/6 — Installing Ditto training dependencies..."
pip install --quiet \
    "accelerate>=0.33.0" \
    "einops>=0.7.0" \
    "tyro>=0.8.0" \
    "tensorboard>=2.14.0" \
    "tqdm>=4.48" \
    "numpy>=1.26,<2.3" \
    "scipy>=1.10.0" \
    "Pillow>=10.0.0" \
    "opencv-python-headless>=4.8.0" \
    "mediapipe>=0.10.0" \
    "onnxruntime-gpu" \
    "scikit-image>=0.21.0"
ok "Ditto training dependencies installed."

# ── 6. Computer vision (LivePortrait for motion extraction) ──────────────────
log "Step 6/6 — Installing CV packages for data preparation..."
pip install --quiet \
    "face-alignment>=1.4.0" \
    "facenet-pytorch>=2.6.0" \
    "kornia>=0.7.0" \
    "imageio>=2.28.0" \
    "imageio-ffmpeg>=0.4.9" \
    "hsemotion>=0.3.0"
ok "CV packages installed."

# ── Download bridge checkpoint ────────────────────────────────────────────────
BRIDGE_CKPT="${PROJECT_ROOT}/checkpoints/bridge_best.pt"
if [ ! -f "$BRIDGE_CKPT" ]; then
    log "Downloading bridge checkpoint from HuggingFace..."
    python -c "
from huggingface_hub import snapshot_download
import os
os.makedirs('${PROJECT_ROOT}/checkpoints', exist_ok=True)
snapshot_download(
    repo_id='Darknsu/librispeech-full-dataset-model',
    repo_type='dataset',
    local_dir='${PROJECT_ROOT}/checkpoints',
    local_dir_use_symlinks=False,
    allow_patterns=['bridge_best.pt'],
)
print('Downloaded bridge_best.pt')
"
    ok "Bridge checkpoint downloaded."
else
    ok "Bridge checkpoint already exists: $BRIDGE_CKPT"
fi

# ── Verify setup ─────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}"
echo "═══════════════════════════════════════════════════════════════"
echo "   ✅  Environment setup complete!"
echo ""
echo "   Installed:"
echo "     • PyTorch + CUDA"
echo "     • Moshi (Mimi encoder)"
echo "     • Bridge module dependencies"
echo "     • Ditto training dependencies"
echo "     • Data preparation tools"
echo ""
echo "   Next: Run scripts/prepare_and_train.sh"
echo "═══════════════════════════════════════════════════════════════"
echo -e "${RESET}"
