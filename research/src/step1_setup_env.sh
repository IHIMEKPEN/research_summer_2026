#!/bin/bash
# ============================================================
# Step 1 — Environment Setup: OpenVLA + Unitree G1 Simulation
# Research Plan: VLA + ESN for Real-Time Humanoid Robot Control
# Author: Osemudiamen Andrew Ihimekpen | PVAMU CREDIT Center
# Week 1–2 | May 19 – Jun 1, 2026
# ============================================================

set -e  # exit on any error

echo "========================================================"
echo " OpenVLA + Unitree G1 Simulation Environment Setup"
echo "========================================================"

# ── 0. System check ─────────────────────────────────────────
echo ""
echo "[0] Checking system..."
python3 --version
nvcc --version 2>/dev/null || echo "WARNING: CUDA not found — check GPU drivers"
nvidia-smi 2>/dev/null || echo "WARNING: nvidia-smi not available"

# ── 1. Conda environment ────────────────────────────────────
echo ""
echo "[1] Creating conda environment 'openvla_g1'..."
conda create -n openvla_g1 python=3.10 -y
conda activate openvla_g1

# ── 2. Core ML dependencies ─────────────────────────────────
echo ""
echo "[2] Installing PyTorch + HuggingFace stack..."
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.40.0 accelerate==0.29.0 bitsandbytes==0.43.0
pip install sentencepiece timm einops
pip install huggingface_hub datasets

# ── 3. OpenVLA ───────────────────────────────────────────────
echo ""
echo "[3] Installing OpenVLA..."
pip install git+https://github.com/openvla/openvla.git
# Model will be downloaded on first run from: openvla/openvla-7b (HuggingFace)
# INT4 quantisation support:
pip install auto-gptq optimum

# ── 4. Robotics / Simulation stack ──────────────────────────
echo ""
echo "[4] Installing MuJoCo + simulation dependencies..."
pip install mujoco==3.1.6
pip install dm_control
pip install gymnasium==0.29.1
pip install gymnasium-robotics

# Unitree G1 MuJoCo model — clone the official repo
echo ""
echo "[4b] Fetching Unitree G1 MuJoCo MJCF model..."
git clone https://github.com/unitreerobotics/unitree_mujoco.git ~/unitree_mujoco
# Also grab the MuJoCo Menagerie model (higher quality MJCF)
git clone https://github.com/google-deepmind/mujoco_menagerie.git ~/mujoco_menagerie

# ── 5. Scientific + logging stack ───────────────────────────
echo ""
echo "[5] Installing scientific and logging libraries..."
pip install numpy scipy matplotlib seaborn pandas
pip install wandb tensorboard
pip install tqdm rich psutil gputil

# ── 6. Baselines for Week 3–4 comparison ────────────────────
echo ""
echo "[6] Installing baseline dependencies (PID, LSTM)..."
pip install control  # python-control for PID
pip install scikit-learn

# ── 7. Verify install ────────────────────────────────────────
echo ""
echo "[7] Verifying installation..."
python3 - <<'PYCHECK'
import torch
import mujoco
import transformers
print(f"  PyTorch      : {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU          : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM         : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  MuJoCo       : {mujoco.__version__}")
print(f"  Transformers : {transformers.__version__}")
print("  All checks PASSED")
PYCHECK

echo ""
echo "========================================================"
echo " Setup complete. Activate with: conda activate openvla_g1"
echo "========================================================"
