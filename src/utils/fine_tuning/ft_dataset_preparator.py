import os, shutil
from typing import List, Dict
from random import Random
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from src.utils.data.dataset_utils import get_dataset_instances
from src.utils.fine_tuning.crop_retriever import CropRetriever

class FTDatasetPreparator:
    ALL_PHASES = ("train", "val", "test")
    EXTRACTION_PHASES = ("train", "test")
    
    def __init__(self, experiment_dir: str, dataset: str, classes: List[str]):
        self.experiment_dir = experiment_dir
        self.dataset = dataset
        self.classes = classes
        self.rng = Random(24)
    
    def create_subdirectories(self):
        for phase in self.ALL_PHASES:
            for cls in self.classes:
                dir_path = os.path.join(self.experiment_dir, phase, cls)
                os.makedirs(dir_path, exist_ok=True)
    
    def remove_subdirectories(self):
        for phase in self.ALL_PHASES:
            shutil.rmtree(os.path.join(self.experiment_dir, phase), ignore_errors=True)
    
    def extract_base_crops(self, train_replicas: int, crop_size: int) -> Dict[str, Dict[str, int]]:
        crops_per_instance = {"train": {}, "test": {}}
        crop_retriever = CropRetriever(crop_size)
        
        for phase in self.EXTRACTION_PHASES:
            instance_filepaths = get_dataset_instances(self.dataset, self.classes, phase)
            
            # max_workers=None -> min(32, os.cpu_count() + 4) as per Python 3.8+
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(self._extract_crops_from_instance, inst_filepath, phase, train_replicas, crop_retriever):
                    inst_filepath for inst_filepath in instance_filepaths
                }
                
                for future, inst_filepath in tqdm(futures.items(), total=len(futures), desc=f"Extracting crops from '{phase}' instances", position=0, leave=True, dynamic_ncols=True):
                    crops_per_instance[phase][inst_filepath] = future.result()
        
        self._create_validation_set(crops_per_instance)
        
        return crops_per_instance

    def _extract_crops_from_instance(self, instance_filepath: str, phase: str, train_replicas: int, crop_retriever: CropRetriever) -> int:
        cls = os.path.basename(os.path.dirname(instance_filepath))
        filename = os.path.basename(instance_filepath)
        stem, _ = os.path.splitext(filename)
        
        with Image.open(instance_filepath) as image:
            crops = crop_retriever.get_crops(image)
        
        id_pad_width = len(str(len(crops)))
        
        if phase == "train":
            for i in range(train_replicas):
                for n, crop in enumerate(crops):
                    crop_filename = f"{stem}_cp{i+1}_crop{n+1:0{id_pad_width}d}.png"
                    crop.save(os.path.join(self.experiment_dir, "train", cls, crop_filename))
        
        elif phase == "test":
            for n, crop in enumerate(crops):
                crop_filename = f"{stem}_crop{n+1:0{id_pad_width}d}.png"
                crop.save(os.path.join(self.experiment_dir, "test", cls, crop_filename))
        
        return len(crops)

    def _create_validation_set(self, crops_per_instance: Dict[str, Dict[str, int]]):
        for inst_filepath in tqdm(crops_per_instance["test"], desc="Extracting Validation Set", position=0, leave=True, dynamic_ncols=True):
            cls = os.path.basename(os.path.dirname(inst_filepath))
            stem = os.path.splitext(os.path.basename(inst_filepath))[0]

            test_cls_dir = os.path.join(self.experiment_dir, "test", cls)
            val_cls_dir  = os.path.join(self.experiment_dir, "val",  cls)

            all_crops = sorted(
                f for f in os.listdir(test_cls_dir)
                if f.startswith(f"{stem}_crop") and f.endswith(".png")
            )

            n_val = max(1, round(len(all_crops) * 0.25))
            val_crops = self.rng.sample(all_crops, n_val)

            for crop_filename in val_crops:
                shutil.copyfile(
                    os.path.join(test_cls_dir, crop_filename),
                    os.path.join(val_cls_dir,  crop_filename)
                )