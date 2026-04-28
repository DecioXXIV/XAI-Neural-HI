CLASSIFIERS = ["ResNet18", "SwinTiny"]
OPTIMIZERS = ["SGD", "Adam", "AdamW"]
LR_SCHEDULERS = ["CosineAnnealingLR"]
FT_MODES = ["frozen", "full"]

__all__ = [CLASSIFIERS, OPTIMIZERS, LR_SCHEDULERS, FT_MODES]