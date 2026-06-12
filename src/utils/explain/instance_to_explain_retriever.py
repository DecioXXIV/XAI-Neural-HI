import os, json
from typing import List, Tuple, Dict, Any

from src.utils.constants import EXPERIMENTS_ROOT, DATA_ROOT

class InstanceToExplainRetriever:
    def __init__(self, experiment_id: str, exp_metadata: Dict[str, Any], subsample: str):
        self.experiment_id = experiment_id
        self.model_name = exp_metadata["MODEL_NAME"]
        self.dataset = exp_metadata["DATASET"]
        self.classes = exp_metadata["CLASSES"]
        self.subsample = subsample
        
        with open(os.path.join(EXPERIMENTS_ROOT, self.experiment_id, "fine_tuning", "class_to_idx.json"), "r") as f:
            self.class_to_idx = json.load(f)
    
    def __call__(self) -> Tuple[List[str], List[int]]:
        instance_paths, labels = [], []
        
        if self.subsample not in ("all", "train", "test"):
            self.subsample = self._parse_subsample()
            instance_paths = [f for f in self.subsample]
        
        else:
            if self.subsample == "all" or self.subsample == "train":
                instance_paths.extend([os.path.join(DATA_ROOT, self.dataset, "train", c, f) for c in self.classes for f in os.listdir(os.path.join(DATA_ROOT, self.dataset, "train", c))])
            if self.subsample == "all" or self.subsample == "test":
                instance_paths.extend([os.path.join(DATA_ROOT, self.dataset, "test", c, f) for c in self.classes for f in os.listdir(os.path.join(DATA_ROOT, self.dataset, "test", c))])
        
        # instance_path = f"{DATA_ROOT}/<dataset>/<split>/<class>/<instance_name>"
        labels = [self.class_to_idx[os.path.basename(os.path.dirname(p))] for p in instance_paths]
            
        return instance_paths, labels

    def _parse_subsample(self) -> List[str]:
        subsample_filepath = os.path.join(EXPERIMENTS_ROOT, self.experiment_id, f"{self.subsample}.txt")
        with open(subsample_filepath, "r") as f:
            return [line.strip() for line in f.readlines()]