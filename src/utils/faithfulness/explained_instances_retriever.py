import os
from typing import List, Tuple, Dict

from src.utils.constants import EXPERIMENTS_ROOT
from src.utils.data.dataset_utils import get_dataset_instances

class ExplainedTestInstancesRetriever:
    def __init__(self, experiment_id: str, dataset: str, classes: List[str], xai_algorithm: str, xai_entry: str, xai_instances_metadata: Dict[str, str] ):
        self.experiment_id = experiment_id
        self.dataset = dataset
        self.classes = classes
        self.xai_algorithm = xai_algorithm
        self.xai_entry = xai_entry
        self.xai_instances_metadata = xai_instances_metadata
    
    def __call__(self) -> Tuple[List[str], List[str]]:
        test_instance_names = {
            os.path.basename(p).split(".")[0]
            for p in get_dataset_instances(self.dataset, self.classes, "test")
        }

        instance_paths, instance_names = [], []

        for instance_name in self.xai_instances_metadata["INSTANCES"]:
            if instance_name in test_instance_names:
                instance_path = os.path.join(EXPERIMENTS_ROOT, self.experiment_id, "xai", self.xai_algorithm, self.xai_entry, instance_name, f"{instance_name}_forexp.png")
                instance_paths.append(instance_path)
                instance_names.append(instance_name)
        
        return instance_paths, instance_names