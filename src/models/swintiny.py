import torch.nn as nn
import torchvision.models as models

from src.models.base_classifiers import SwinBase

class SwinTiny(SwinBase):
    def _build_encoder(self) -> nn.Module:
        weights = models.Swin_T_Weights.IMAGENET1K_V1
        backbone = models.swin_t(weights=weights)
        return nn.Sequential(*list(backbone.children())[:-1])

    def get_model_name(self) -> str: return "SwinTiny"