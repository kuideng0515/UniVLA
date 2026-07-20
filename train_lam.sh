#!/bin/bash
set -e

# ============================================================
# LAM Stage-1 Training Launch Script
# 自动检测 GPU 数量，适配单卡/多卡
# ============================================================

# Accelerate(Maybe?)
export TORCH_CUDA_ARCH_LIST="8.0"

# Avoid some logging and update issues
export NO_ALBUMENTATIONS_UPDATE=1
export TF_CPP_MIN_LOG_LEVEL=2

# WANDB
export WANDB_MODE=disabled
export WANDB_API_KEY=""

# 预训练模型（DINOv2 + T5-base）已随仓库打包在 assets/ 下 强制 transformers / hf_hub 走离线，避免任何联网探测。
# 可用 UNIVLA_ASSETS_DIR 覆盖：export UNIVLA_ASSETS_DIR=/path/to/assets
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# 激活 conda 环境（脚本内需要 source conda.sh）
eval "$(conda shell.bash hook)"
conda activate lams


cd ./latent_action_model # 进入 latent_action_model/

GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
GPU_COUNT=${GPU_COUNT:-1}                     # 无 nvidia-smi 时默认 1
echo "[launch] Detected GPUs: ${GPU_COUNT}"

# 可通过环境变量覆盖默认 YAML
CONFIG="${CONFIG:-config/lam-lerobot.yaml}"
LOG_FILE="${LOG_FILE:-output_train.log}"



torchrun \
    --nnodes=1 \
    --nproc_per_node=${GPU_COUNT} \
    main.py fit \
    --config "${CONFIG}" \
    --trainer.devices ${GPU_COUNT} \
    2>&1 | tee "${LOG_FILE}"