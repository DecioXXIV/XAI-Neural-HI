import os, torch
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from tqdm import tqdm

from src.utils.constants import EXPERIMENTS_ROOT
from src.utils.logger import Logger

logger = Logger()

class ImageMasker(ABC):
    def __init__(self, experiment_id: str, xai_algorithm: str, xai_entry: str, mask_rates: List[float], patches_color: str, masking_color: torch.Tensor):
        self.experiment_id = experiment_id
        self.xai_algorithm = xai_algorithm
        self.xai_entry = xai_entry
        self.mask_rates = mask_rates
        self.patches_color = patches_color
        self.masking_color = masking_color
        
        self.masking_root = os.path.join(EXPERIMENTS_ROOT, self.experiment_id, "masked_images", self.xai_algorithm, self.xai_entry)
        os.makedirs(self.masking_root, exist_ok=True)

    @abstractmethod
    def mask_instance(self, instance_path: str, instance_name: str): pass

    def __call__(self, instance_paths: List[str], instance_names: List[str]):
        logger.info(f"*** BEGINNING OF MASKING PROCESS -> Experiment: {self.experiment_id} | XAI Algorithm: {self.xai_algorithm} | XAI Entry: {self.xai_entry} ***")
        logger.info(f"Rule = {self.mask_rule} | Ceil = {self.mask_rates[-1]} | Step = {self.mask_rates[0]} | Patches color = {self.patches_color}")
        
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self.mask_instance, path, name): name
                       for path, name in zip(instance_paths, instance_names)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Masking instances", position=0, leave=True, dynamic_ncols=True):
                future.result()
        
        logger.info(f"*** END OF MASKING PROCESS ***")

