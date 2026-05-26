import torch, os
import torch.nn as nn
import torchvision.models as models
from typing import List

from src.utils.constants import MODELS_ROOT
from src.utils.models.model_blocks import build_classification_head
from src.models.base_classifiers import BaseClassifier

def _build_fc_block(layer_type: str, in_f: int, out_f: int) -> nn.Sequential:
    fc = nn.Linear(in_f, out_f)
    nn.init.xavier_normal_(fc.weight)
    if fc.bias is not None: nn.init.zeros_(fc.bias)

    if layer_type == "last": return nn.Sequential(fc)
    else: return nn.Sequential(fc, nn.BatchNorm1d(out_f), nn.ReLU())

class ResNet18FeatureEncoder(nn.Module):
    def __init__(self, device: str, cp_path: str):
        super().__init__()

        backbone = models.resnet18(weights=None)
        self.enc = nn.Sequential(*list(backbone.children())[:-1])
        self.fc_layers = nn.Sequential()
        expansion_layer = _build_fc_block("last", in_f=512, out_f=1024)
        self.fc_layers.add_module("fc0", expansion_layer)

        checkpoint = torch.load(cp_path, map_location=device)
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

class ResNet18(BaseClassifier):
    def __init__(self, num_classes: int, ft_mode: str, layers: List[str], device: str):
        super().__init__()
        
        cp_path = os.path.join(MODELS_ROOT, "cp", "Test_3_TL_val_best_model.pth")

        self.feature_encoder = ResNet18FeatureEncoder(device, cp_path)
        if ft_mode == "frozen":
            for param in self.feature_encoder.parameters():
                param.requires_grad = False

        self.classification_head = build_classification_head(1024, num_classes, layers)

    def get_input_size(self) -> int: return 380
    
    def get_model_name(self) -> str: return "ResNet18"

    def extract_visual_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_encoder.extract_backbone_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_encoder(x)
        logits = self.classification_head(features)
        return logits
