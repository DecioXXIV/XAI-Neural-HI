import os, torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.constants import EXPERIMENTS_ROOT

class CheckpointSaver:
    def __init__(self, experiment_id: str, metric: str):
        self.checkpoint_dir = os.path.join(EXPERIMENTS_ROOT, experiment_id, "fine_tuning", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.metric = metric
    
    def __call__(self, cp_name: str, model: nn.Module, optimizer: Optimizer, scheduler: CosineAnnealingLR | None, early_stopping: object) -> None:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "early_stopping": early_stopping.state_dict() if early_stopping is not None else None
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{cp_name}.pth")
        torch.save(checkpoint, checkpoint_path)