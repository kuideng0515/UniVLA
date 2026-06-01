import math
from random import choices, randint
from typing import Any, Callable, Dict, List
from pathlib import Path

import os
from torchcodec.decoders import VideoDecoder
import torch
import torch.nn.functional as F
from einops import rearrange
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.utils.data import get_worker_info
from tqdm import tqdm


def exists(var) -> bool:
    return var is not None


def default(var, val) -> Any:
    return var if exists(var) else val


def default_worker_init_fn(worker_id: int) -> None:
    torch.manual_seed(torch.initial_seed() + worker_id)
    worker_info = get_worker_info()

    if exists(worker_info):
        dataset = worker_info.dataset
        glob_start = dataset._start
        glob_end = dataset._end

        per_worker = int((glob_end - glob_start) / worker_info.num_workers)
        worker_id = worker_info.id

        dataset._start = glob_start + worker_id * per_worker
        dataset._end = min(dataset._start + per_worker, glob_end)


def filter_video_files(file_names: List, xdof: bool = False) -> List:
    if xdof:
        return [
            f for f in file_names
            if "left" not in str(f).lower() and "right" not in str(f).lower() and "resize" not in str(f).lower() and "pad" not in str(f).lower()
            and "320_240" in str(f).lower()
        ]
    else:
        return [
            f for f in file_names
            if "left" not in str(f).lower() and "right" not in str(f).lower() and "resize" not in str(f).lower() and "pad" not in str(f).lower()
        ]


class LightningDataset(LightningDataModule):
    """
    Abstract LightningDataModule that represents a dataset we can train a Lightning module on.
    """

    def __init__(
        self,
        *args,
        batch_size: int = 8,
        num_workers: int = 4,
        train_shuffle: bool = True,
        val_shuffle: bool = False,
        val_batch_size: int = None,
        worker_init_fn: Callable = None,
        collate_fn: Callable = None,
        train_sampler: Callable = None,
        test_sampler: Callable = None,
        val_sampler: Callable = None
    ) -> None:
        super(LightningDataset, self).__init__()
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

        val_batch_size = default(val_batch_size, batch_size)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.train_shuffle = train_shuffle
        self.val_shuffle = val_shuffle
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler
        self.val_sampler = val_sampler
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn

    def train_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn
        )

    def val_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.train_dataset,
            sampler=self.val_sampler,
            batch_size=self.val_batch_size,
            shuffle=self.val_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn
        )

    def test_dataloader(self) -> DataLoader:
        if isinstance(self.train_dataset, IterableDataset):
            worker_init_fn = default(self.worker_init_fn, default_worker_init_fn)
        else:
            worker_init_fn = self.worker_init_fn
        return DataLoader(
            self.test_dataset,
            sampler=self.test_sampler,
            batch_size=self.val_batch_size,
            shuffle=self.val_shuffle,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn
        )


class VideoDataset(Dataset):
    def __init__(
        self,
        subset_path: str,
        padding: str = "repeat",
        randomize: bool = False,
        num_frames: int = 16,
        downsample_factor: int = None,
        output_format: str = "t h w c",
        color_aug: bool = True
    ) -> None:
        super(VideoDataset, self).__init__()
        self.padding = padding
        self.randomize = randomize
        self.num_frames = num_frames
        self.downsample_factor = downsample_factor
        self.output_format = output_format
        self.color_aug = color_aug

        # Get all the file path based on the split path
        mp4_list = list(Path(subset_path).rglob("*.mp4"))
        if len(mp4_list) > 0:
            if "xdof" in subset_path:
                self.file_names = filter_video_files(mp4_list, xdof=True)
            else:
                self.file_names = filter_video_files(mp4_list)
        else:
            raise ValueError(f"No video files found in {subset_path}")

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Dict:
        video_path = self.file_names[idx]
        retries = 0
        while True:
            try:
                retries += 1
                video = self.load_video_slice(
                    video_path,
                    self.num_frames,
                    None if self.randomize else 0,
                    self.downsample_factor
                )
                return self.build_data_dict(video)
            except:
                if retries > 10:
                    raise RuntimeError(f"Failed to load video after {retries} attempts: {video_path}")
                idx = randint(0, len(self) - 1)
                video_path = self.file_names[idx]

    def load_video_slice(
        self,
        video_path: str,
        num_frames: int,
        start_frame: int = None,
        frame_skip: int = None
    ) -> Tensor:
        video_path = str(video_path)
        
        # use CPU to avoid blocking GPU with decoding,
        try:
            decoder = VideoDecoder(video_path, device="cpu", seek_mode="approximate")
            total_frames = len(decoder)
        except Exception as e:
            print(f"Error initializing decoder for {video_path}: {e}")
            total_frames = 0
            decoder = None

        frame_skip = randint(1, 50) if frame_skip is None else frame_skip
        
        total_required_span = num_frames * frame_skip
        start_frame = start_frame if exists(start_frame) else randint(0, max(0, total_frames - total_required_span))
        
        indices = [start_frame + i * frame_skip for i in range(num_frames)]
        valid_indices = [idx for idx in indices if idx < total_frames]

        frames = []
        if decoder is not None and len(valid_indices) > 0:
            try:
                frames_batch = decoder.get_frames_at(indices=valid_indices)
                for i in range(frames_batch.data.shape[0]):
                    img = frames_batch.data[i].permute(1, 2, 0)
                    frames.append(img)
            except Exception as e:
                print(f"Error decoding frames for {video_path}: {e}")
            finally:
                del decoder

        if len(frames) == 0:
            print(f"Warning: No frames decoded from {video_path}, returning zero frames.")
            frames.append(torch.zeros((480, 640, 3), dtype=torch.uint8))

        if len(frames) < num_frames:
            print(f"Warning: Only {len(frames)} frames decoded from {video_path}, expected {num_frames}. Applying padding.")
            if self.padding == "none":
                pass
            elif self.padding == "repeat":
                frames.extend([frames[-1]] * (num_frames - len(frames)))
            elif self.padding == "zero":
                frames.extend([torch.zeros_like(frames[-1])] * (num_frames - len(frames)))
            elif self.padding == "random":
                frames.extend([torch.rand_like(frames[-1])] * (num_frames - len(frames)))
            else:
                raise ValueError(f"Invalid padding type: {self.padding}")

        # stack directly, not need frameworks[::frame_skip] anymore since already get indices below
        video = torch.stack(frames) / 255.0

        target_ratio = 640 / 480
        if video.shape[2] / video.shape[1] > target_ratio:
            target_height = video.shape[1]
            target_width = int(video.shape[1] * target_ratio)
        elif video.shape[2] / video.shape[1] < target_ratio:
            target_height = int(video.shape[2] / target_ratio)
            target_width = video.shape[2]
        else:
            target_height = video.shape[1]
            target_width = video.shape[2]
        h_crop = (video.shape[1] - target_height) // 2
        w_crop = (video.shape[2] - target_width) // 2
        video = video[:, h_crop:h_crop + target_height, w_crop:w_crop + target_width]
        video = rearrange(video, "t h w c -> c t h w")
        video = F.interpolate(video, (240, 320), mode="bilinear")
        video = rearrange(video, f"c t h w -> {self.output_format}")
        return video

    def build_data_dict(self, video: Tensor) -> Dict:
        if self.color_aug:
            # Brightness jitter
            video = (video + torch.rand(1) * 0.2 - 0.1).clamp(0, 1)

        data_dict = {
            "videos": video
        }
        return data_dict


class OriginalVideoDataset(Dataset):
    def __init__(
        self,
        dataset_paths: list,
        split: str = "train",
        padding: str = "repeat",
        randomize: bool = False,
        num_frames: int = 16,
        downsample_factor: int = None,
        output_format: str = "t h w c",
        color_aug: bool = True
    ) -> None:
        super(OriginalVideoDataset, self).__init__()
        self.padding = padding
        self.randomize = randomize
        self.num_frames = num_frames
        self.downsample_factor = downsample_factor
        self.output_format = output_format
        self.color_aug = color_aug

        self.file_names = []
        for subset_path in dataset_paths:
            mp4_list = list(Path(subset_path).rglob("*.mp4"))
            if len(mp4_list) > 0:
                if "xdof" in subset_path:
                    self.file_names.extend(filter_video_files(mp4_list, xdof=True))
                else:
                    self.file_names.extend(filter_video_files(mp4_list))
            else:
                raise ValueError(f"No video files found in {subset_path}")

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> Dict:
        video_path = self.file_names[idx]
        retries = 0
        while True:
            try:
                retries += 1
                video = self.load_video_slice(
                    video_path,
                    self.num_frames,
                    None if self.randomize else 0,
                    self.downsample_factor
                )
                return self.build_data_dict(video)
            except:
                if retries > 10:
                    raise RuntimeError(f"Failed to load video after {retries} attempts: {video_path}")
                idx = randint(0, len(self) - 1)
                video_path = self.file_names[idx]

    def load_video_slice(
        self,
        video_path: str,
        num_frames: int,
        start_frame: int = None,
        frame_skip: int = None
    ) -> Tensor:
        video_path = str(video_path)
        
        # use CPU to avoid blocking GPU with decoding,
        try:
            decoder = VideoDecoder(video_path, device="cpu", seek_mode="approximate")
            total_frames = len(decoder)
        except Exception as e:
            print(f"Error initializing decoder for {video_path}: {e}")
            total_frames = 0
            decoder = None

        frame_skip = randint(1, 50) if frame_skip is None else frame_skip
        
        total_required_span = num_frames * frame_skip
        start_frame = start_frame if exists(start_frame) else randint(0, max(0, total_frames - total_required_span))
        
        indices = [start_frame + i * frame_skip for i in range(num_frames)]
        valid_indices = [idx for idx in indices if idx < total_frames]

        frames = []
        if decoder is not None and len(valid_indices) > 0:
            try:
                frames_batch = decoder.get_frames_at(indices=valid_indices)
                for i in range(frames_batch.data.shape[0]):
                    img = frames_batch.data[i].permute(1, 2, 0)
                    frames.append(img)
            except Exception as e:
                print(f"Error decoding frames for {video_path}: {e}")
            finally:
                del decoder

        if len(frames) == 0:
            print(f"Warning: No frames decoded from {video_path}, returning zero frames.")
            frames.append(torch.zeros((480, 640, 3), dtype=torch.uint8))

        if len(frames) < num_frames:
            print(f"Warning: Only {len(frames)} frames decoded from {video_path}, expected {num_frames}. Applying padding.")
            if self.padding == "none":
                pass
            elif self.padding == "repeat":
                frames.extend([frames[-1]] * (num_frames - len(frames)))
            elif self.padding == "zero":
                frames.extend([torch.zeros_like(frames[-1])] * (num_frames - len(frames)))
            elif self.padding == "random":
                frames.extend([torch.rand_like(frames[-1])] * (num_frames - len(frames)))
            else:
                raise ValueError(f"Invalid padding type: {self.padding}")

        # stack directly, not need frameworks[::frame_skip] anymore since already get indices below
        video = torch.stack(frames) / 255.0

        target_ratio = 640 / 480
        if video.shape[2] / video.shape[1] > target_ratio:
            target_height = video.shape[1]
            target_width = int(video.shape[1] * target_ratio)
        elif video.shape[2] / video.shape[1] < target_ratio:
            target_height = int(video.shape[2] / target_ratio)
            target_width = video.shape[2]
        else:
            target_height = video.shape[1]
            target_width = video.shape[2]
        h_crop = (video.shape[1] - target_height) // 2
        w_crop = (video.shape[2] - target_width) // 2
        video = video[:, h_crop:h_crop + target_height, w_crop:w_crop + target_width]
        video = rearrange(video, "t h w c -> c t h w")
        video = F.interpolate(video, (240, 320), mode="bilinear")
        video = rearrange(video, f"c t h w -> {self.output_format}")
        return video

    def build_data_dict(self, video: Tensor) -> Dict:
        if self.color_aug:
            # Brightness jitter
            video = (video + torch.rand(1) * 0.2 - 0.1).clamp(0, 1)

        data_dict = {
            "videos": video
        }
        return data_dict


class MultiSourceSamplerDataset(Dataset):
    def __init__(
        self,
        dataset_paths: list,
        split: str = "train",
        samples_per_epoch: int = 1000000,
        sampling_strategy: str = "sample",
        color_aug: bool = True,
        **kwargs
    ) -> None:
        self.samples_per_epoch = samples_per_epoch

        self.subsets = []
        for subset_path in tqdm(dataset_paths, desc="Loading subsets..."):
            self.subsets.append(VideoDataset(subset_path=subset_path, color_aug=color_aug, **kwargs))
            print("Subset:", subset_path, "Number of samples:", len(self.subsets[-1]))
        print("Number of subsets:", len(self.subsets))

        if sampling_strategy == "sample":
            # Sample uniformly from all samples
            probs = [len(d) for d in self.subsets]
        elif sampling_strategy == "dataset":
            # Sample uniformly from all datasets
            probs = [1 for _ in self.subsets]
        elif sampling_strategy == "log":
            # Generate probabilities according to the scale of each dataset
            probs = [math.log(len(d)) if len(d) else 0 for d in self.subsets]
        elif sampling_strategy == "pi":
            # Generate probabilities according to the scale of each dataset
            probs = [len(d) ** 0.43 for d in self.subsets]
        # elif sampling_strategy == "manual":
        #     probs = [1, 1, 0.2, 0.2, 0.2, 0.2, 0.2, 10, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        else:
            raise ValueError(f"Unavailable sampling strategy: {sampling_strategy}")
        total_prob = sum(probs)
        assert total_prob > 0, "No sample is available"
        self.sample_probs = [x / total_prob for x in probs]
        print("Sample probabilities:", self.sample_probs)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict:
        """
        Args:
        index (int): Index (ignored since we sample randomly).

        Returns:
        TensorDict: Dict containing all the data blocks.
        """

        # Randomly select a subset based on weights
        subset = choices(self.subsets, self.sample_probs)[0]

        # Sample a valid sample with a random index
        sample_idx = randint(0, len(subset) - 1)
        sample_item = subset[sample_idx]
        return sample_item


class LightningVideoDataset(LightningDataset):
    def __init__(
        self,
        dataset_paths: list,
        padding: str = "repeat",
        randomize: bool = False,
        num_frames: int = 16,
        downsample_factor: int = None,
        output_format: str = "t h w c",
        samples_per_epoch: int = 1000000,
        sampling_strategy: str = "sample",
        **kwargs
    ) -> None:
        super(LightningVideoDataset, self).__init__(**kwargs)
        self.dataset_paths = dataset_paths
        self.padding = padding
        self.randomize = randomize
        self.num_frames = num_frames
        self.downsample_factor = downsample_factor
        self.output_format = output_format
        self.samples_per_epoch = samples_per_epoch
        self.sampling_strategy = sampling_strategy

        self.save_hyperparameters()

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = MultiSourceSamplerDataset(
                dataset_paths=self.dataset_paths,
                split="train",
                padding=self.padding,
                randomize=self.randomize,
                num_frames=self.num_frames,
                downsample_factor=self.downsample_factor,
                output_format=self.output_format,
                samples_per_epoch=self.samples_per_epoch,
                sampling_strategy=self.sampling_strategy
            )
            # self.val_dataset = MultiSourceSamplerDataset(
            #     dataset_paths=self.dataset_paths,
            #     split="test",
            #     padding=self.padding,
            #     randomize=self.randomize,
            #     num_frames=self.num_frames,
            #     output_format=self.output_format,
            #     samples_per_epoch=self.samples_per_epoch // 1000,
            #     sampling_strategy=self.sampling_strategy,
            #     color_aug=False
            # )
        elif stage == "test":
            self.test_dataset = OriginalVideoDataset(
                dataset_paths=self.dataset_paths,
                split="test",
                padding=self.padding,
                randomize=self.randomize,
                num_frames=self.num_frames,
                downsample_factor=self.downsample_factor,
                output_format=self.output_format,
                color_aug=False
            )
        else:
            raise ValueError(f"Invalid stage: {stage}")


def main():
    # ========= test VideoDataset =========
    # ds = VideoDataset(
    #     subset_path="/mnt/pfs/dataset/lerobot_data/8271_to_8575/videos/chunk-000",
    #     padding="repeat",
    #     randomize=True,
    #     num_frames=2,
    #     output_format="t h w c",
    #     color_aug=False
    # )

    ds = MultiSourceSamplerDataset(
        dataset_paths=[
            "/mnt/pfs/dataset/lerobot_data/4473_to_4475",
            "/mnt/pfs/dataset/lerobot_data/8271_to_8575/videos/chunk-000"
        ],
        sampling_strategy="dataset",
    )
    
    print("test VideoDataset:")
    for i in range(10):
        sample_idx = randint(0, len(ds) - 1)
        sample_item = ds[sample_idx]

    print(ds[0]['videos'].shape)



    breakpoint()    


if __name__ == "__main__":
    main()
