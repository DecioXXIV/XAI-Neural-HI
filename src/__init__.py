CLASSIFIERS = ["ResNet18", "SwinTiny", "SwinSmall"]
METRICS = ["loss", "accuracy", "macrof1", "weightedf1"]
OPTIMIZERS = ["SGD", "Adam", "AdamW"]
LR_SCHEDULERS = ["CosineAnnealingLR"]
FT_MODES = ["frozen", "full"]
TRAIN_AUG_TRANSFORMS = ["legacy", "moderate", "backgroundrobust", "aggressive", "xaggressive"]

EXPLAINERS = ["Occlusion", "Lime", "GLimeBinomial"]
SEGMENTATIONS = ["sq_patches", "ink_based"]

INK_SEG_GRANULARITIES = ["raw", "char", "word", "custom"]
INK_SEG_GROUPING_METHODS = ["auto", "hull", "dilation"]

__all__ = [CLASSIFIERS, METRICS, OPTIMIZERS, LR_SCHEDULERS, FT_MODES, TRAIN_AUG_TRANSFORMS, EXPLAINERS, SEGMENTATIONS, INK_SEG_GRANULARITIES, INK_SEG_GROUPING_METHODS]