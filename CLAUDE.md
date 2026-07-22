# CLAUDE.md — UniVLA 项目上下文

> 这份文件在每次 Claude Code 会话启动时自动加载。用途：会话/对话记录丢失后，快速恢复项目上下文与当前进度。
> **维护约定**：上半部「稳定知识」不常改；下半部「当前进度」每次有实质进展就更新，并更新日期。进度过时了就直接改，别只往下堆。

---

## 项目是什么

UniVLA —— 一个 VLA（视觉-语言-动作）模型仓库。当前活跃开发**全部集中在 `latent_action_model/`（LAM，潜在动作模型）的第一阶段训练**：用冻结的 DINOv2 视觉编码器 + 冻结的 T5-base 文本编码器，做「下一帧预测」来学习潜在动作表示（VQ-VAE 风格）。

目标场景：把 LAM 适配到**自有机器人数据集**上训练，运行在**阿里云内网 / 断网环境**，本体为 x2w / 双臂 EEF。

## 环境与运行

- 训练入口：`bash train_lam.sh`（内部 `conda activate lams` → `python -m ...`）。
- 默认配置：`latent_action_model/config/lam-lerobot.yaml`
  - `data_root_dir: /mnt/pfs/dengyiqi/datasets`，`data_mix: x2w_wm_dataset`
  - `batch_size: 64`，`frame_interval: 30`（1.0s @ 30fps），`video_backend: torchcodec`，`num_workers: 8`
- 数据混合定义在 `latent_action_model/dataloader/gr00t_lerobot/mixtures.py`（`DATASET_NAMED_MIXTURES["x2w_wm_dataset"]`）。
- **内网 pip**：系统可能有全局 `PIP_CONSTRAINT`（NVIDIA 镜像的 `/etc/pip/constraint.txt`）锁死 pillow 等导致 `ResolutionImpossible`。安装用 `env -u PIP_CONSTRAINT pip install -r requirements.txt`。

## 离线资源（已打包，开箱即用）

预训练模型已随仓库打包在 `assets/` 下，**完全离线加载，不联网、不依赖 ~/.cache**：

- `assets/dinov2/facebookresearch_dinov2_main/` + `dinov2_vitb14_reg4_pretrain.pth`
- `assets/t5-base/`（config / tokenizer / safetensors）

加载入口在 `latent_action_model/genie/modules/lam.py`：`load_dino_encoder()` 用 `source='local' + pretrained=False` 再灌权重；T5 用 `from_pretrained(T5_BASE_DIR, local_files_only=True)`。`train_lam.sh` 已设 `HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1`，并已删掉旧的网络代理。assets 路径可用 `UNIVLA_ASSETS_DIR` 覆盖。

> 注：`assets/` 是缓存的资源，通常不用管，不要当成代码改动。

## 架构约定与重要设计

### dataloader 来源
数据管线移植自 StarVLA，基于 GR00T 的 LeRobot 管线，位于 `latent_action_model/dataloader/gr00t_lerobot/`。

### 旋转感知的动作处理（rotation-aware，重要特性）
末端执行器（EEF）的旋转字段不能像平移那样做欧氏减法，必须在 SO(3) 上做相对/差分，再转成 `rotation_6d` 归一化。实现链路：
- `transform/rotation_utils.py`：旋转数学（quaternion / rotation_6d / euler / matrix 互转，`relative_rotation` / `delta_rotation` / `representation_dim`）。
- `gr00t_lerobot/datasets.py`：`_relative_action_statistics` 统一 delta/rel 统计。**非旋转字段行为逐字节保持不变（向后兼容）**；旋转字段在 SO(3) 组合后按目标表示（chunk 级）算统计，存到保留键 `__rotation_stats__`。统计缓存版本 = v3，`target_rotations` 纳入缓存 key（改配置会正确失效旧缓存）。
- `transform/state_action.py`：解除了「相对旋转无法归一化」的旧限制，改用数据驱动统计。
- `data_config.py`：`EEFDataConfig` 是双臂 EEF 本体的参考模板，声明 `target_rotations`（eef 旋转 → rotation_6d）。注册在 `ROBOT_TYPE_CONFIG_MAP["eef"]`。
- 测试：`latent_action_model/tests/test_rotation_stats.py`。

### x2w 计划退役
代码里带 `TODO(user): x2w-specific, safe to remove` 注释的部分（`X2WJointDataConfig`、`LAMX2WConfig`、`X2WLeRobotSingleDataset`）是过渡路径，计划用通用 EEF 路径取代。x2w 专用子类**不走**旋转感知的相对/差分管线。

---

## 当前进度（更新于 2026-07-22）

### 状态概述
两条主线，**均未提交**（工作区 diff / untracked）：

**主线 A — 旋转感知动作特性：代码完整，测试已通过（17/17），待提交。**
- 涉及 `rotation_utils.py`（新）、`datasets.py`、`state_action.py`、`data_config.py`、`lerobot_datasets.py`、`tests/test_rotation_stats.py`（新）。
- ✅ 2026-07-22 用 `lams` 环境跑 `test_rotation_stats.py` → **17/17 全过**（含 SO(3) 往返、与参考实现一致性、非旋转字段逐字节回归）。
- 补充：`lerobot_datasets.py` 里把 DataConfig 的 `target_rotations` 注入 `data_cfg`，让统计管线与运行时转换用同一目标表示（属同一特性）。
- ⏳ 下一步：单独提交这块。

**主线 B — GPU 利用率瓶颈排查：已定位并被端到端实测印证；结论明确（GPU 解码路线排除，格式换 LeRobot 3.0 也无用）。**
- 目标：训练稳态 GPU 利用率 ≥90%。旧 commit `ca2966b` 记的「GPU 4%」是**冷启动假象**（首 batch 要重算统计 + 解码预热 + worker 启动），非稳态。
- **✅ 端到端实测基准（2026-07-22，用户实跑 `train_lam.sh`，H20/97GB/23核，已含下述 3 项优化）：**

  | batch | it/s | ≈samples/s | epoch步数 | 峰值显存 | GPU利用率 |
  |---|---|---|---|---|---|
  | 64 | 1.15 | ~74 | 680 | 40.6 GB | 70–80% |
  | 96 | 0.83 | ~80 (+8%) | 453 | ~61 GB | — |
  | 128 | 0.66 | ~84 (+14%) | 340 | ~81 GB | 更高 |

  - 看 **samples/s** 而非 it/s（it/s 下降只是因每步样本变多）。更大 batch 让单步 GPU 更久 → dataloader 有更多时间备下一批 → GPU 空等更少、利用率更高。
  - **边际收益递减**：64→96 得 +8%，96→128 再得 +5%。因根本瓶颈始终是 CPU 解码 ~85 samples/s 的墙，batch 再大只是填 GPU 空隙。**96 是显存/吞吐的较优平衡点**（拿到大部分收益，显存比 128 宽裕）。
  - **注意**：batch 翻倍若要正经训练需相应上调 lr（当前 `optimizer.init_args.lr: 1e-4`）；测速度可先不动。
  - 这与瓶颈分析完全吻合。**此即当前可接受的现状基准，未继续追 90%。**

- **⚠️ 换 A100 前必看（用户计划迁 A100 测试）：**
  - A100 只有 **40GB / 80GB** 两种规格，均 < H20 的 97GB。显存换算（LAM stage-1，本配置）：batch64≈40.6GB / 96≈61GB / 128≈81GB。
    - **A100-40GB**：batch96/128 都装不下，约只能到 **batch48–56**。
    - **A100-80GB**：batch96 可以，batch128（~81GB）非常贴边有 OOM 风险，建议 **≤112**。
  - A100 算力低于 H20，单步 GPU 会更久；但**真正瓶颈是 CPU 解码，取决于 A100 机器给多少 CPU 核**——核数 ≥23 吞吐可能持平/更好，核数更少会更差。**换卡前先确认该机器的 CPU 核数与 A100 显存规格，比调 batch 更重要。**
  - A100 是 **sm_80**，torch 2.7.0+cu126 wheel 直接支持；见 README「跨平台复现」表。
- **已应用 3 项 CPU 侧优化（保留）：**
  1. 冻结 DINOv2 用 `torch.no_grad()` 包住（`genie/modules/lam.py` 两个 LAM 类的 `vq_encode`）——该 stage 快 ~13% + 省激活显存。
  2. `num_workers: 8 → 12`（`config/lam-lerobot.yaml`，23 核机器上 12 是吞吐甜点，>12 超订变慢）。
  3. 启用 `prefetch_factor=4`（`dataloader/lam_datamodule.py`，原来注释掉了）。
- **实测数据（H20，97GB，23 核）：**
  - GPU 单步 fwd+bwd = 606 ms @ batch64 → 需 ~106 samples/s 才喂满；峰值显存仅 40.6/97 GB。
  - CPU 解码多进程扩展**线性**，但 23 核封顶 ~120 samples/s——**刚好压在 GPU 需求线附近**，所以稳态 GPU 卡 70–80%。
  - 瓶颈 = **CPU 视频解码并行度不足**，本质是「23 核喂不饱 H20」，非代码问题。
- **GPU (NVDEC) 解码调查 → 负收益，已排除：**
  - x2w 数据（`4473_to_4475`）是 **AV1 1280×720**，H20 NVDEC **不支持 AV1**（`av1_cuvid` capabilities 全 0）——硬件限制，无解。
  - dagger 数据（`dagger_data/`）是 **H.264 640×400**，H20 NVDEC 支持。为此克隆环境 `lams_gpu`（已建好、可用；torch 2.7.0+cu126 / torchcodec 0.5+cu126），从 `download.pytorch.org/whl/cu126` 装了带 NVDEC 的 `torchcodec==0.5+cu126`。**GPU 解码成功跑通**，但性能实测：真实访问模式（大量不同 episode、每个只取相隔30的2帧）下 GPU 5 samples/s vs CPU 17 samples/s（单线程），**GPU 慢 3.4×**。原因：NVDEC 建解码上下文固定开销 88ms（CPU 29ms），而每样本只解 2 帧无法摊薄；NVDEC 在单 GPU 上也难像 CPU 那样多进程线性扩展。
  - **关键教训：GPU 视频解码只在「同一文件大批量/连续解码」才划算；LAM 这种「海量小文件随机取几帧」的模式，GPU 解码是负收益。** `lams_gpu` 环境结论已得、可删。
- **LeRobot 3.0 格式实测 → 更慢，不解决瓶颈（曾以为能救，实测否定）：**
  - 用 `/mnt/pfs/dataset/lerobot_data/g2a_task_201_307_0711_0721/`（v3.0，H.264，多 episode 合并进 ~4000 帧/~150MB 的大文件）实测解码拆解：仅打开文件 **14.1ms**（vs 单-episode 文件的 4ms），真实 getitem **24.3ms**（vs 16.7ms），单核 **41 clip/s**（vs 60）。
  - 原因：合并虽减少文件数，但单个大文件「建帧索引」的打开成本更高、大文件内随机 seek 距关键帧更远——**合并省下的打开次数被单次更贵抵消还超支**。v3.0 优化的是存储/元数据管理，**不是随机访问解码吞吐**。
  - **结论：换 LeRobot 3.0 不能提升（反而降低）解码吞吐，不要指望它救 GPU 利用率。**
- **唯一确定能到 90% 的方向：加 CPU 核**（换更多 vCPU 的机器/多机），解码吞吐随核线性涨。**CPU 更弱的平台会更受限**，需下调 `num_workers` 并预期更低利用率。
- profiling / 监控脚本已跑完结论、**已删除**（`bench_dataloader.py` / `profile_step.py` / `monitor_train.py`）——结论都固化进本文件与 README，无需保留。

**文档已更新（为跨平台复现）：**
- `README.md` 新增「跨平台复现」表（GPU/驱动/CUDA/FFmpeg/torch 版本对照，含 A100 sm_80 说明）+「视频解码后端 / 为何不用 GPU 解码」说明。
- `requirements.txt` 补充 torch↔CUDA build 匹配、torchcodec↔系统 FFmpeg 主版本匹配的注释。
- `tests/test_rotation_stats.py` 去除硬编码外部参考路径依赖：参考实现缺失时自动跳过 T2（其余 16 项自足），可用 `ROTATION_REF_IMPL` 覆盖路径 → 测试可在任意平台跑。

### 最近提交（git log）
- `82afcae` offline-load DINOv2 + T5 from bundled assets/, ignore large weights
- `8227eee` update for aliyun（删掉大批下游评测代码 calvin/libero/r2r/simpler，为内网精简）
- `ca2966b` data load successful, but training bottleneck on CPU, GPU 4%（注：该 4% 已查明为冷启动假象）
- `9eb8a66` add dataloader from starVLA
