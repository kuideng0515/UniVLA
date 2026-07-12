"""
LAMLeRobotDataModule — LightningDataModule adapter that bridges the LeRobot
dataloader to the DINO_LAM training pipeline.

Key responsibilities:
  1. Dynamically build a DataConfig for LAM training (single camera, dual timestep).
  2. Create LeRobot datasets and wrap them in a LightningDataModule.
  3. Provide a ``collate_fn`` that converts LeRobot-format dicts to the
     ``{"videos": (B,2,C,H,W), "task_instruction": List[str]}`` format
     expected by ``DINO_LAM``.
"""

import logging
import re
import string
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from lightning import LightningDataModule
from torch.utils.data import DataLoader

from dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP
from dataloader.gr00t_lerobot.datasets import (
    LeRobotMixtureDataset,
    ModalityConfig,
    X2WLeRobotSingleDataset,
)
from dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from dataloader.gr00t_lerobot.transform import ComposedModalityTransform
from dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subclass: optimized _pack_sample — pure tensor pipeline (no PIL!)
# ---------------------------------------------------------------------------

class _LAMSingleDataset(X2WLeRobotSingleDataset):
    """
    Identical to X2WLeRobotSingleDataset except ``_pack_sample`` retains
    *all* frames returned by ``get_video`` instead of only the first one,
    and uses a pure-tensor pipeline (no PIL conversion) for maximal throughput.

    This allows ``observation_indices = [0, N]`` to produce multi-frame
    samples for next-frame-prediction training.
    """

    def _pack_sample(self, data: dict) -> dict:
        """
        Override: keep all T frames per video key.

        Key optimization: avoid PIL.Image.fromarray().resize() entirely.
        Instead, use torch F.interpolate (bilinear) on the GPU/CPU tensor
        directly, exactly like demo_dataset.py does.
        """
        all_frames = []
        for video_key in self.modality_keys["video"]:
            frames = data[video_key]  # np.ndarray: (T, H, W, C)
            # Convert to torch tensor once: (T, H, W, C) -> (T, C, H, W)
            t = torch.from_numpy(frames).float() / 255.0
            t = t.permute(0, 3, 1, 2)  # (T, C, H, W)
            # Resize all T frames in one shot via bilinear interpolation
            t = F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)
            all_frames.append(t)  # each: (T, C, 224, 224)

        language = data[self.modality_keys["language"][0]][0]
        action = []
        for action_key in self.modality_keys["action"]:
            action.append(data[action_key])
        action = np.concatenate(action, axis=1).astype(np.float16)

        sample = {
            "action": action,
            "image": all_frames,       # List[Tensor], each (T, C, 224, 224)
            "lang": language,
            "robot_tag": self.tag,
        }

        if self.data_cfg is not None and self.data_cfg.get("include_state", False) not in ("False", False):
            state = []
            for state_key in self.modality_keys.get("state", []):
                state.append(data[state_key])
            if state:
                state = np.concatenate(state, axis=1).astype(np.float16)
                sample["state"] = state

        return sample


# ---------------------------------------------------------------------------
# Collate function — bridges LeRobot format → DINO_LAM format
# ---------------------------------------------------------------------------

def lam_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Convert a list of per-sample LeRobot dicts into the batched format
    expected by ``DINO_LAM.forward()``.

    Optimized: works directly with torch tensors (no PIL → Tensor conversion).

    LeRobot input (per sample)::

        {
            'image': [Tensor(T,C,H,W), ...],   # already tensors after _pack_sample optimization
            'lang': str,
            'action': ndarray,
            'state': ndarray (optional),
            'robot_tag': str,
        }

    DINO_LAM output::

        {
            'videos': (B, 2, C, H, W),
            'task_instruction': List[str],
        }
    """
    initial_frames = []
    target_frames = []
    task_instructions = []

    for item in batch:
        images = item["image"]
        # images[0] = initial frame tensor (T,C,H,W), take first timestep
        # images[1] = target frame tensor (T,C,H,W) — if multi-timestep, take last
        # For LAM: images is list of per-camera tensors; we take the first camera
        img_tensor = images[0]  # (T, C, H, W)
        if img_tensor.shape[0] >= 2:
            initial_frames.append(img_tensor[0])   # first timestep
            target_frames.append(img_tensor[-1])    # last timestep
        else:
            initial_frames.append(img_tensor[0])
            target_frames.append(img_tensor[0])

        task_instructions.append(item.get("lang", ""))

    # Normalize language: remove punctuation + lowercase
    task_instructions = [
        re.sub(f"[{string.punctuation}]", "", s).lower() for s in task_instructions
    ]

    # Stack: each is (C, H, W) → (B, C, H, W) → cat along dim=1 → (B, 2, C, H, W)
    videos = torch.stack(
        [torch.stack(initial_frames), torch.stack(target_frames)], dim=1
    )

    return {
        "videos": videos,
        "task_instruction": task_instructions,
    }


# ---------------------------------------------------------------------------
# LightningDataModule
# ---------------------------------------------------------------------------

class LAMLeRobotDataModule(LightningDataModule):
    """
    LightningDataModule that loads LeRobot-format datasets for LAM training.

    Resolves ``data_mix`` via ``DATASET_NAMED_MIXTURES`` (same convention as
    the source starvla config), constructs one ``_LAMSingleDataset`` per
    mixture entry, and wraps them in a ``LeRobotMixtureDataset``.

    YAML usage::

        data:
          class_path: dataloader.lam_datamodule.LAMLeRobotDataModule
          init_args:
            data_root_dir: /mnt/pfs/dengyiqi/datasets
            data_mix: x2w_wm_dataset
            batch_size: 64
            frame_interval: 15
            video_backend: torchcodec
    """

    def __init__(
        self,
        data_root_dir: str,
        data_mix: str = "x2w_wm_dataset",
        batch_size: int = 64,
        frame_interval: int = 15,
        image_aug: bool = False,
        video_backend: str = "torchcodec",
        excluded_segment_statuses: List[str] = None,
        num_workers: int = 0,
        val_batch_size: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.data_root_dir = Path(data_root_dir)
        self.data_mix = data_mix
        self.batch_size = batch_size
        self.frame_interval = frame_interval
        self.image_aug = image_aug 
        self.video_backend = video_backend
        self.excluded_segment_statuses = excluded_segment_statuses or []
        self.num_workers = num_workers
        self.val_batch_size = val_batch_size or batch_size

        self.train_dataset = None
        self.val_dataset = None


    def _resolve_mixture(self):
        """Resolve ``data_mix`` → list of (dataset_name, weight, robot_type)."""
        from dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES

        if self.data_mix not in DATASET_NAMED_MIXTURES:
            raise KeyError(
                f"Unknown data_mix '{self.data_mix}'. "
                f"Available: {list(DATASET_NAMED_MIXTURES.keys())}"
            )
        return DATASET_NAMED_MIXTURES[self.data_mix]

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def setup(self, stage: str) -> None:
        base_config = ROBOT_TYPE_CONFIG_MAP["x2w_lam"]
        # Override frame_interval and image_aug each run
        base_config.set_frame_interval(self.frame_interval)
        base_config.image_aug = self.image_aug

        modality_configs = base_config.modality_config()
        transforms = base_config.transform()
        embodiment_tag = EmbodimentTag.X2W

        mixture_spec = self._resolve_mixture()
        logger.info(
            "LAMLeRobotDataModule: mixture '%s' → %s",
            self.data_mix,
            [(d, w, r) for d, w, r in mixture_spec],
        )

        data_cfg = {
            "excluded_segment_statuses": self.excluded_segment_statuses,
        }

        # 分布式：所有 rank 用同一个 seed。
        # 数据隔离靠 Lightning DDP 自动注入的 DistributedSampler 切分 index 范围实现。
        # index 不同 → safe_hash((epoch, index, seed)) 不同 → 采样不同数据。
        seed = 42

        datasets = []
        seen = set()
        for d_name, d_weight, robot_type in mixture_spec:
            key = (d_name, robot_type)
            if key in seen:
                continue
            seen.add(key)

            dataset_path = self.data_root_dir / d_name
            ds = _LAMSingleDataset(
                dataset_path=dataset_path,
                modality_configs=modality_configs,
                embodiment_tag=embodiment_tag,
                video_backend=self.video_backend,
                transforms=transforms,
                delete_pause_frame=False,
                data_cfg=data_cfg,
            )
            datasets.append((ds, d_weight))

        if stage == "fit":
            self.train_dataset = LeRobotMixtureDataset(
                datasets, mode="train", seed=seed,
                balance_dataset_weights=False,
                balance_trajectory_weights=False,
                data_cfg=data_cfg,
            )
            self.val_dataset = LeRobotMixtureDataset(
                datasets, mode="val", seed=seed,
                balance_dataset_weights=False,
                balance_trajectory_weights=False,
                data_cfg=data_cfg,
            )
        elif stage == "test":
            self.test_dataset = LeRobotMixtureDataset(
                datasets, mode="test", seed=seed,
                balance_dataset_weights=False,
                balance_trajectory_weights=False,
                data_cfg=data_cfg,
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            # prefetch_factor=4 if self.num_workers > 0 else None,
            collate_fn=lam_collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            # prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=lam_collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            collate_fn=lam_collate_fn,
            pin_memory=True,
        )

    def on_train_epoch_start(self) -> None:
        """每 epoch 开始时同步更新 seed，确保 safe_hash((epoch, index, seed)) 产生新的采样映射。"""
        if self.train_dataset is not None:
            self.train_dataset.set_epoch(self.trainer.current_epoch)


# cd ./latent_action_model && python -m dataloader.lam_datamodule && cd ..
if __name__ == "__main__":
    dm = LAMLeRobotDataModule(
        data_root_dir="/mnt/pfs/dengyiqi/datasets",
        data_mix="x2w_wm_dataset",
        batch_size=64,
        frame_interval=30,
        image_aug=True,
        video_backend="torchcodec",
        excluded_segment_statuses=["Start remote operation.", "End remote operation."],
        num_workers=8,
    )

    print("=== setup('fit') ===")
    dm.setup("fit")

    # 检查 train/val 数据集
    print(f"train_dataset len: {len(dm.train_dataset)}")
    print(f"val_dataset   len: {len(dm.val_dataset)}")

    # 检查 raw sample（验证 _LAMSingleDataset._pack_sample 输出）
    raw = dm.train_dataset[0]
    print(f"\nRaw sample keys: {list(raw.keys())}")
    print(f"  image: {len(raw['image'])} frames, type={type(raw['image'][0]).__name__}, "
          f"size={raw['image'][0].size}")
    print(f"  lang:  {raw['lang'][:80]}...")
    print(f"  action shape: {raw['action'].shape}, dtype={raw['action'].dtype}")
    print(f"  robot_tag: {raw['robot_tag']}")

    # 检查 collated batch
    print("\n=== train_dataloader ===")
    loader = dm.train_dataloader()

    import time as _time

    # 预热：跑一个 batch 触发 lazy 初始化
    _ = next(iter(loader))

    # 计时循环
    num_batches = 20
    times = []
    print(f"\n=== Benchmark: {num_batches} batches (bs={dm.batch_size}) ===")
    t_start = _time.time()
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        t_batch = _time.time()
        times.append(t_batch)
        elapsed = (t_batch - t_start) * 1000
        per_sample = elapsed / ((i + 1) * dm.batch_size)
        print(f"  batch {i:3d}: +{elapsed:7.0f}ms total  "
              f"| {per_sample:.1f}ms/sample  "
              f"| videos={batch['videos'].shape}")
    t_end = _time.time()

    total_ms = (t_end - t_start) * 1000
    avg_ms = total_ms / num_batches
    throughput = num_batches * dm.batch_size / (t_end - t_start)
    print(f"\n  Total: {total_ms:.0f}ms ({total_ms/1000:.1f}s)")
    print(f"  Avg per batch: {avg_ms:.0f}ms")
    print(f"  Avg per sample: {avg_ms/dm.batch_size:.1f}ms")
    print(f"  Throughput: {throughput:.0f} samples/s  ({throughput/dm.batch_size:.1f} batches/s)")

    # ---- 抽样检查最后一个 batch 的形状 ----
    print(f"\n  Last batch keys: {list(batch.keys())}")
    print(f"    videos:           {batch['videos'].shape}  "
          f"range=[{batch['videos'].min():.3f}, {batch['videos'].max():.3f}]")
    print(f"    task_instruction: {len(batch['task_instruction'])} strings")

    # ------------------------------------------------------------------
    # 可视化：将 B×2 帧保存为 PNG 网格，供观察后删除
    # ------------------------------------------------------------------
    import os, torchvision.utils as vutils
    from pathlib import Path

    videos = batch["videos"]  # (B, 2, C, H, W)
    B = videos.shape[0]
    all_frames = videos.reshape(-1, *videos.shape[2:])  # (B*2, C, H, W)
    grid = vutils.make_grid(all_frames, nrow=B, padding=2, normalize=False)

    save_path = Path.cwd() / "batch_preview.png"
    vutils.save_image(grid, save_path)
    print(f"\n  Preview saved to: {save_path}")
    print(f"  Top row = initial frames, Bottom row = target frames")
    print(f"  Labels: {batch['task_instruction']}")

    # breakpoint()  # 观察图片后按 'c' 继续

    save_path.unlink()
    print("  Cleaned up.")

    # 验证 shape 契约
    B, T, C, H, W = batch["videos"].shape
    assert T == 2, f"Expected T=2, got {T}"
    assert C == 3, f"Expected C=3, got {C}"
    assert H == 224 and W == 224, f"Expected 224x224, got {H}x{W}"
    assert len(batch["task_instruction"]) == B

    print("\n=== All checks passed! ===")


    # import time, torchcodec, numpy as np, os, glob, random
    # from torchcodec.decoders import VideoDecoder
    # from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

    # all_videos = glob.glob("/mnt/pfs/dataset/dagger_data/003/close-door/8781/videos/chunk-000/observation.images.x2w_camera_head_realsense_compressed/*.mp4", recursive=True)[:200]
    # random.shuffle(all_videos)
    # print(f"Found {len(all_videos)} video files")
    
    # def decode_one_cold(path):
    #     """模拟真实训练：每个文件只打开一次（无 OS cache 加持）"""
    #     d = VideoDecoder(path, device="cpu", seek_mode="approximate",
    #                     num_ffmpeg_threads=1)
    #     return d.get_frames_at(indices=[0, 30])

    # # 清除 OS 缓存（需要 root）
    # # os.system("echo 3 > /proc/sys/vm/drop_caches")

    # N = min(64, len(all_videos))
    # t0 = time.time()
    # for i in range(N):
    #     decode_one_cold(all_videos[i])
    # t1 = time.time()
    # print(f"冷读取: {N} samples in {(t1-t0)*1000:.0f}ms  ({(t1-t0)/N*1000:.1f}ms/sample)")