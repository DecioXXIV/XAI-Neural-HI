import os
from typing import List

from data import LEGACY_TRANSFORMS, MODERATE_TRANSFORMS, BACKGROUND_ROBUST_TRANSFORMS, AGGRESSIVE_TRANSFORMS, XAGGRESSIVE_TRANSFORMS
from src.utils.constants import DATA_ROOT

def get_dataset_instances(dataset: str, classes: List[str], subset: str) -> List[str]:
    instances = []
    
    for cls in classes:
        if subset == "train": class_source_dir = os.path.join(DATA_ROOT, dataset, "train", cls)
        elif subset == "test": class_source_dir = os.path.join(DATA_ROOT, dataset, "test", cls)
        
        instances.extend(os.path.join(class_source_dir, file) for file in os.listdir(class_source_dir))
    
    return sorted(instances)

def get_train_transforms(train_transforms: str):
    if train_transforms == "legacy": return LEGACY_TRANSFORMS
    elif train_transforms == "moderate": return MODERATE_TRANSFORMS
    elif train_transforms == "backgroundrobust": return BACKGROUND_ROBUST_TRANSFORMS
    elif train_transforms == "aggressive": return AGGRESSIVE_TRANSFORMS
    else: return XAGGRESSIVE_TRANSFORMS