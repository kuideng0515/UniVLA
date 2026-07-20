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
