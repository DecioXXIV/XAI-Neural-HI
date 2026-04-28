import torch, os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List
from torchvision import transforms as T

from src.utils.constants import MODELS_ROOT

def _build_fc_block(layer_type: str, in_f: int, out_f: int) -> nn.Sequential:
    fc = nn.Linear(in_f, out_f)
    nn.init.xavier_normal_(fc.weight)
    if fc.bias is not None: nn.init.zeros_(fc.bias)

    if layer_type == "last": return nn.Sequential(fc)
    else: return nn.Sequential(fc, nn.BatchNorm1d(out_f), nn.ReLU())

class ResNet18FeatureEncoder(nn.Module):
    def __init__(self, cp_path: str):
        super().__init__()

        backbone = models.resnet18(weights=None)
        self.enc = nn.Sequential(*list(backbone.children())[:-1])
        self.fc_layers = nn.Sequential()
        expansion_layer = _build_fc_block("last", in_f=512, out_f=1024)
        self.fc_layers.add_module("fc0", expansion_layer)

        checkpoint = torch.load(cp_path)
        state_dict = checkpoint["model_state_dict"].copy()
        state_dict.pop("alpha", None)
        self.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.enc(x)
        if features.ndim > 2: features = torch.flatten(features, start_dim=1)
        expanded_features = self.fc_layers(features)
        return expanded_features
    
    def extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.enc(x)
        if features.ndim > 2: features = torch.flatten(features, start_dim=1)
        return features

class ResNet18(nn.Module):
    def __init__(self, num_classes: int, ft_mode: str):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = 1024
        
        cp_path = os.path.join(MODELS_ROOT, "cp", "Test_3_TL_val_best_model.pth")

        self.feature_encoder = ResNet18FeatureEncoder(cp_path=cp_path)
        if ft_mode == "frozen":
            for param in self.feature_encoder.parameters():
                param.requires_grad = False

        self.classification_head = nn.Sequential(
            _build_fc_block("hidden", in_f=self.feature_dim, out_f=128),
            _build_fc_block("last", in_f=128, out_f=num_classes)
        )

    def get_input_size(self) -> int: return 380

    def extract_visual_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_encoder.extract_backbone_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_encoder(x)
        logits = self.classification_head(features)
        return logits

    @staticmethod
    def build_inference_transforms(mean: List[float], std: List[float], input_size: int=380) -> T.Compose:
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