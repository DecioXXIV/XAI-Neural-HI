import os
from typing import List, Tuple, Dict

from src.utils.constants import EXPERIMENTS_ROOT

class ExplainedInstancesRetriever:
    def __init__(self, experiment_id: str, xai_algorithm: str, xai_entry: str, xai_instances_metadata: Dict[str, str]):
        self.experiment_id = experiment_id
        self.xai_algorithm = xai_algorithm
        self.xai_entry = xai_entry
        self.xai_instances_metadata = xai_instances_metadata
    
    def __call__(self) -> Tuple[List[str], List[str]]:
        instance_paths = []
        
        for instance_name in self.xai_instances_metadata["INSTANCES"]:
            instance_path = os.path.join(EXPERIMENTS_ROOT, self.experiment_id, "xai", self.xai_algorithm, self.xai_entry, instance_name, f"{instance_name}_forexp.png")
            instance_paths.append(instance_path)
        
        return instance_paths, list(self.xai_instances_metadata["INSTANCES"].keys())