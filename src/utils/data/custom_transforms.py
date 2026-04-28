import torch
import torchvision.transforms.functional as F

class Invert(object):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return F.invert(x)

class AddGaussianNoise(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.std + self.mean