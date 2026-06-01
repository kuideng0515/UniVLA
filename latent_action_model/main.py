from lightning.pytorch.cli import LightningCLI
from dataloader.lam_datamodule import LAMLeRobotDataModule
from genie.model import DINO_LAM

cli = LightningCLI(
    DINO_LAM,
    LAMLeRobotDataModule,
    seed_everything_default=42,
)
