"""Quick test to verify LeRobot dataloader works with the new import paths."""
import sys
from pathlib import Path

# Ensure latent_action_model/ is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from latent_action_model.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
from torch.utils.data import DataLoader


# --- Test config (mirrors your previous working config) ---
class Cfg:
    data_root_dir = Path("/mnt/pfs/dengyiqi/datasets")
    data_mix = "x2w_wm_dataset"
    delete_pause_frame = False
    video_backend = "torchvision_av"
    include_state = True
    sequential_step_sampling = False

    @staticmethod
    def get(key, default=None):
        return getattr(Cfg, key, default)


print("=== Building dataset ===")
dataset = get_vla_dataset(data_cfg=Cfg, mode="train")
print(f"Dataset created: {type(dataset).__name__}")

print("\n=== Building dataloader ===")
dataloader = DataLoader(
    dataset,
    batch_size=2,
    num_workers=0,
    collate_fn=collate_fn,
)

print("\n=== Iterating ===")
for i, batch in enumerate(dataloader):
    print(f"\nBatch {i}:")
    if isinstance(batch, list):
        for j, item in enumerate(batch):
            print(f"  Item {j} keys: {list(item.keys())}")
            for k, v in item.items():
                if hasattr(v, 'shape'):
                    print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
                elif isinstance(v, str):
                    print(f"    {k}: str = '{v[:80]}...'")
                else:
                    print(f"    {k}: {type(v).__name__}")
    if i >= 2:
        break

print("\n=== Test passed! ===")
