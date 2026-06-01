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

# Cache Dir
export TORCH_HOME="/mnt/pfs/dengyiqi/.cache/torch"
export HF_HOME="/mnt/pfs/dengyiqi/.cache/huggingface"

# 激活 conda 环境（脚本内需要 source conda.sh）
eval "$(conda shell.bash hook)"
conda activate lams

# 网络代理（torch.hub 需要访问 GitHub 验证缓存）
export http_proxy=http://10.66.65.186:18000
export https_proxy=http://10.66.65.186:18000


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