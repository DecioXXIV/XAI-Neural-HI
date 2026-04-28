CLASSIFIERS = ["ResNet18", "SwinTiny"]
METRICS = ["loss", "accuracy", "macrof1", "weightedf1"]
OPTIMIZERS = ["SGD", "Adam", "AdamW"]
LR_SCHEDULERS = ["CosineAnnealingLR"]
FT_MODES = ["frozen", "full"]

__all__ = [CLASSIFIERS, METRICS, OPTIMIZERS, LR_SCHEDULERS, FT_MODES]