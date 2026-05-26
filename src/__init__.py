CLASSIFIERS = ["ResNet18", "SwinTiny", "SwinSmall"]
METRICS = ["loss", "accuracy", "macrof1", "weightedf1"]
OPTIMIZERS = ["SGD", "Adam", "AdamW"]
LR_SCHEDULERS = ["CosineAnnealingLR"]
FT_MODES = ["frozen", "full"]
TRAIN_AUG_TRANSFORMS = ["legacy", "moderate", "backgroundrobust", "aggressive", "xaggressive"]

EXPLAINERS = ["Occlusion", "Lime", "GLimeBinomial"]
SEGMENTATIONS = ["sq_patches"]

__all__ = [CLASSIFIERS, METRICS, OPTIMIZERS, LR_SCHEDULERS, FT_MODES, TRAIN_AUG_TRANSFORMS, EXPLAINERS, SEGMENTATIONS]