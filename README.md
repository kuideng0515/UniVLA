## Install
```bash
conda create -n lams python==3.10 -y
conda activate lams
conda install "ffmpeg" -c conda-forge
pip install -r requirements.txt
# test data loading
cd ./latent_action_model && python -m dataloader.lam_datamodule && cd ..
```

> **提示（内网/受限环境）**：若系统设置了全局 pip 约束文件（环境变量 `PIP_CONSTRAINT`，
> 常见于 NVIDIA 基础镜像的 `/etc/pip/constraint.txt`），它可能锁死 `pillow` 等版本导致
> `ResolutionImpossible`。本次安装临时禁用该约束即可，不改系统文件：
> ```bash
> env -u PIP_CONSTRAINT pip install -r requirements.txt
> ```

## 预训练模型（DINOv2 + T5-base）—— 已随仓库打包，开箱即用

训练需要两个预训练模型，均已打包在仓库的 `assets/` 下，由
`latent_action_model/genie/modules/lam.py` **直接离线加载**（不联网、不依赖 `~/.cache`）：

```
assets/dinov2/facebookresearch_dinov2_main/          # DINOv2 仓库代码
assets/dinov2/dinov2_vitb14_reg4_pretrain.pth        # DINOv2 权重
assets/t5-base/                                       # T5-base（config/tokenizer/safetensors）
```

- **DINOv2**：`load_dino_encoder()` 以 `source='local'` + `pretrained=False` 加载代码，再直接灌
  `assets/` 里的权重，完全绕开 `api.github.com` 与权重 URL 下载。
- **T5-base**：`from_pretrained(assets/t5-base, local_files_only=True)`。

默认指向 `<repo>/assets`；若 assets 放在别处，用环境变量覆盖：
```bash
export UNIVLA_ASSETS_DIR=/path/to/assets
```

> 训练脚本 `train_lam.sh` 已设 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`，确保绝不联网。

（可选）验证离线加载：
```bash
cd latent_action_model
python -c "
import sys; sys.path.insert(0,'.')
from genie.modules.lam import load_dino_encoder, T5_BASE_DIR
from transformers import T5EncoderModel
load_dino_encoder('dinov2_vitb14_reg')
T5EncoderModel.from_pretrained(T5_BASE_DIR, local_files_only=True)
print('offline load from assets OK')
"
```

## Train
```bash
bash train_lam.sh
```
默认配置 `config/lam-lerobot.yaml`（`data_root_dir: /mnt/pfs/dengyiqi/datasets`,
`data_mix: x2w_wm_dataset`）。

## 跨平台复现（硬件 / 驱动 / CUDA / FFmpeg）

本仓库在以下环境验证通过，迁移到其他平台（更弱 CPU、更老 GPU 如 A100）时请对照调整：

| 项 | 本次验证环境 | 迁移注意 |
|---|---|---|
| GPU | **NVIDIA H20**（97 GB，compute capability **sm_90**） | A100 是 **sm_80**，同样被 CUDA 12.x 支持，torch 2.7.0+cu126 wheel 可直接用 |
| 驱动 | **535.183.06** | 需 ≥ 525（CUDA 12.x 的最低驱动）。驱动过老要么升级驱动，要么换更低 CUDA 的 torch build |
| CUDA（torch 运行时） | **12.6**（`torch.version.cuda`；系统 nvcc 为 12.8，不影响 torch） | 平台 CUDA 更老时，从 https://download.pytorch.org/whl 装对应 build（如 `cu121`/`cu118`），torchvision 用相同 `+cuXXX` 标签 |
| torch / torchvision | **2.7.0+cu126 / 0.22.0+cu126** | 两者的 `+cuXXX` 必须一致 |
| cuDNN | 9.5.1（随 torch wheel 自带） | 无需单独装 |
| 系统 FFmpeg | **6.1.1**（`libavcodec.so.60`） | torchcodec 0.5 按系统 FFmpeg 主版本挑内部 core：4.x→58 / 5.x→59 / 6.x→60 / 6.1→61。系统 FFmpeg 若为 7.x（so.62+），torchcodec 0.5 会加载失败——需装更新 torchcodec 或把 FFmpeg 降到 ≤6.1 |
| Python | 3.10 | — |

**GPU 精度**：训练用 `precision: 16-mixed`（fp16 autocast）。sm_80/sm_90 均支持；更老的显卡（sm_70 如 V100）也能跑 fp16，但需确认 torch build 覆盖该架构。

**性能基准（H20，97GB，23 核，本机实测）**：`x2w_wm_dataset` + batch 64 + `num_workers: 12`，训练 **~1.15 it/s（≈74 samples/s），GPU 利用率 70–80%**。这是当前的可接受现状——瓶颈在 CPU 视频解码吞吐，不在 GPU。

**关于视频解码后端 / 为何不用 GPU 解码**（重要，避免重复踩坑）：
- 默认 `video_backend: torchcodec`，**在 CPU 上解码**。这是刻意的：LAM 的数据访问模式是「海量不同 episode、每个只随机取 2 帧」，实测 NVDEC/GPU 解码在此模式下是**负收益**（建解码上下文固定开销远高于 CPU，且每样本只解 2 帧无法摊薄；单 GPU 上 NVDEC 也难像 CPU 多进程那样线性扩展）。
- 训练稳态瓶颈是 **CPU 视频解码吞吐**：解码随 CPU 核数线性扩展，本机 23 核封顶约 120 samples/s（恰在 GPU 需求 ~106 samples/s 附近，故利用率 70–80%）。**CPU 更弱的平台请相应下调 `num_workers`（本机甜点为 12），并预期喂数速率更低、GPU 利用率下降**——此时提升的正道是增加 CPU 核数，而非 GPU 解码。
- 部分数据集为 **AV1** 编码：注意 A100/H20 等数据中心卡的 NVDEC **不支持 AV1 解码**（即便想用 GPU 解码也不行），只能走 CPU。
- **LeRobot 3.0 格式不能提升解码吞吐**（实测反而更慢）：v3.0 把多 episode 合并进单个大视频文件，虽减少文件数，但单个大文件的「建帧索引」打开成本与大文件内随机 seek 距离都更高——实测单核解码 41 clip/s（vs 单-episode 文件 60）。v3.0 优化的是存储/元数据管理，不是随机访问解码吞吐，不要指望它救 GPU 利用率。

### 验证旋转感知动作管线（单元测试）
```bash
cd latent_action_model && python tests/test_rotation_stats.py
```
测试自带合成数据，无外部依赖即可跑（17 项，其中 T2 为与参考实现的交叉校验：
参考实现缺失时自动跳过，可用 `ROTATION_REF_IMPL=/path/to/rotation_torch_utils.py` 指定）。
