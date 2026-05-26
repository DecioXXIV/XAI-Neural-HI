import torch.nn as nn
import torchvision.models as models

from src.models.base_classifiers import SwinBase

class SwinSmall(SwinBase):
    def _build_encoder(self) -> nn.Module:
        weights = models.Swin_S_Weights.IMAGENET1K_V1
        backbone = models.swin_s(weights=weights)
        return nn.Sequential(*list(backbone.children())[:-1])

    def get_model_name(self) -> str: return "SwinSmall"