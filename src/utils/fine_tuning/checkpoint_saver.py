import os, torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.utils.constants import EXPERIMENTS_ROOT
from src.models.resnet18 import ResNet18
from src.models.swintiny import SwinTiny

class CheckpointSaver:
    def __init__(self, experiment_id: str):
        self.checkpoint_dir = os.path.join(EXPERIMENTS_ROOT, experiment_id, "fine_tuning", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def __call__(self, epoch_loss: float, min_loss: float, model: nn.Module, optimizer: Optimizer, scheduler: CosineAnnealingLR | None, early_stopping: object, phase: str, check: bool):
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_loss,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "early_stopping": early_stopping.state_dict() if early_stopping is not None else None
        }
        
        if check:
            if epoch_loss < min_loss:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"{phase}_best_model.pth")
                torch.save(checkpoint, checkpoint_path)
                return epoch_loss
            else:
                return min_loss
        
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, "last_checkpoint.pth")
            torch.save(checkpoint, checkpoint_path)
            return None