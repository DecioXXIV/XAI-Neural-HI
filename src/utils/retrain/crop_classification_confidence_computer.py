import torch
import torch.nn as nn
import numpy as np
from typing import List

class CropClassificationConfidenceComputer:
    def __init__(self, model: nn.Module, device: str):
        self.model = model
        self.device = device
    
    def compute_confidence(self, batch: torch.Tensor, labels: torch.Tensor) -> List[float]:
        batch, labels = batch.to(self.device), labels.to(self.device)
        bs = batch.size(0)
        
        with torch.inference_mode():
            logits = self.model(batch)
            true_logits = logits[torch.arange(bs), labels]
            
            masked = logits.clone()
            masked[torch.arange(bs), labels] = -np.inf
            max_other_logits = masked.max(dim=1).values
            confidences = (true_logits - max_other_logits).clamp(min=0.0)
        
        confidences = confidences.detach().cpu().numpy()
        return (1 - np.exp(-0.5 * (confidences**2))).tolist()
    
    def compute_difficulty(self, batch: torch.Tensor, labels: torch.Tensor) -> List[float]:
        batch, labels = batch.to(self.device), labels.to(self.device)
        bs = batch.size(0)
        
        with torch.inference_mode():
            logits = self.model(batch)
            true_logits = logits[torch.arange(bs), labels]
            
            masked = logits.clone()
            masked[torch.arange(bs), labels] = -np.inf
            max_other_logits = masked.max(dim=1).values
            confidences = max_other_logits - true_logits
        
        confidences = confidences.detach().cpu().numpy()
        return np.exp(-0.5 * (confidences**2)).tolist()