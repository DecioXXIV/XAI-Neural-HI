import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torchvision import transforms as T
from abc import ABC, abstractmethod

from src.utils.models.model_blocks import build_classification_head

class BaseClassifier(nn.Module, ABC):

    @abstractmethod
    def get_input_size(self) -> int: ...

    @abstractmethod
    def get_model_name(self) -> str: ...

    @abstractmethod
    def extract_visual_features(self, x: torch.Tensor) -> torch.Tensor: ...

    @staticmethod
    def build_inference_transforms(mean: List[float], std: List[float], input_size: int) -> T.Compose:
        return T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

    def batch_predict_function(self, inputs: List[np.ndarray], mean: List[float], std: List[float], device: torch.device, apply_softmax: bool = True, forward_fn=None) -> np.ndarray:
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

class SwinBase(BaseClassifier):
    FEATURE_DIM: int = 768

    def __init__(self, num_classes: int, ft_mode: str, layers: List[str]):
        super().__init__()

        self.feature_encoder = self._build_encoder()
        if ft_mode == "frozen":
            for param in self.feature_encoder.parameters():
                param.requires_grad = False

        self.classification_head = build_classification_head(self.FEATURE_DIM, num_classes, layers)

    @abstractmethod
    def _build_encoder(self) -> nn.Module: ...

    def get_input_size(self) -> int: return 224

    def extract_visual_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_encoder(x)
        if features.ndim > 2: features = torch.flatten(features, start_dim=1)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_visual_features(x)
        logits = self.classification_head(features)
        return logits