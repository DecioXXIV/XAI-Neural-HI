import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List
from torchvision import transforms as T

from src.utils.models.model_blocks import build_classification_head

class SwinSmallFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        weights = models.Swin_S_Weights.IMAGENET1K_V1
        backbone = models.swin_s(weights=weights)
        
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        if features.ndim > 2: features = torch.flatten(features, start_dim=1)
        return features

class SwinSmall(nn.Module):
    def __init__(self, num_classes: int, ft_mode: str, layers: List[str]):
        super().__init__()
        
        self.feature_encoder = SwinSmallFeatureEncoder()
        if ft_mode == "frozen":
            for param in self.feature_encoder.parameters():
                param.requires_grad = False
        
        self.classification_head = build_classification_head(768, num_classes, layers)
    
    def get_input_size(self) -> int: return 224
    
    def get_model_name(self) -> str: return "SwinSmall"
    
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
    
    def batch_predict_function(self, inputs: List[np.ndarray], mean: List[float], std: List[float], device: torch.device, apply_softmax: bool=True, forward_fn=None) -> np.ndarray:
        if not hasattr(self, '_inference_transforms'):
            self._inference_transforms = self.build_inference_transforms(mean, std, self.get_input_size())
        batch = torch.stack([self._inference_transforms(i) for i in inputs], dim=0).to(device)
        
        self.eval()
        with torch.inference_mode():
            _fn = forward_fn if forward_fn is not None else self
            logits = _fn(batch)
            if apply_softmax:
                probs = F.softmax(logits, dim=1)
                output = probs.detach().cpu().numpy()
            else:
                output = logits.detach().cpu().numpy()
        
        return output