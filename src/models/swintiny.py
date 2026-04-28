import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List
from torchvision import transforms as T

from src.utils.models.model_blocks import build_classification_head

class SwinTinyFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        weights = models.Swin_T_Weights.IMAGENET1K_V1
        backbone = models.swin_t(weights=weights)
        
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        if features.ndim > 2: features = torch.flatten(features, start_dim=1)
        return features

class SwinTiny(nn.Module):
    def __init__(self, num_classes: int, ft_mode: str, layers: List[str]):
        super().__init__()
        
        self.feature_encoder = SwinTinyFeatureEncoder()
        if ft_mode == "frozen":
            for param in self.feature_encoder.parameters():
                param.requires_grad = False
        
        self.classification_head = build_classification_head(768, num_classes, layers)
    
    def get_input_size(self) -> int: return 224
    
    def extract_visual_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_encoder(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_encoder(x)
        logits = self.classification_head(features)
        return logits
    
    @staticmethod
    def build_inference_transforms(mean: List[float], std: List[float], input_size: int=224) -> T.Compose:
        return T.Compose([
            T.ToTensor(),
            T.Resize((input_size, input_size)),
            T.Normalize(mean, std)
        ])
    
    def batch_predict_function(self, inputs: List[np.ndarray], mean: List[float], std: List[float], device: torch.device, apply_softmax: bool=True) -> np.ndarray:
        transforms = self.build_inference_transforms(mean, std, self.get_input_size())
        batch = torch.stack([transforms(i) for i in inputs], dim=0).to(device)
        
        self.eval()
        with torch.no_grad():
            logits = self(batch)
            if apply_softmax:
                probs = F.softmax(logits, dim=1)
                output = probs.detach().cpu().numpy()
            else:
                output = logits.detach().cpu().numpy()
        
        return output