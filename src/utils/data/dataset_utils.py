import os
from typing import List

from src.utils.constants import DATA_ROOT

def get_dataset_instances(dataset: str, classes: List[str], subset: str) -> List[str]:
    instances = []
    
    for cls in classes:
        if subset == "train": class_source_dir = os.path.join(DATA_ROOT, dataset, "train", cls)
        elif subset == "test": class_source_dir = os.path.join(DATA_ROOT, dataset, "test", cls)
        
        instances.extend(os.path.join(class_source_dir, file) for file in os.listdir(class_source_dir))
    
    return sorted(instances)