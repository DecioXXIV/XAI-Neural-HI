import os, json
from typing import List, Tuple

from src.utils.logger import Logger
from src.utils.fine_tuning.ft_dataset_preparator import FTDatasetPreparator
from src.utils.fine_tuning.train_rgb_mean_std_computer import TrainRGBMeanStdComputer
from src.utils.data.dataloaders import TrainDataLoader, TestDataLoader

logger = Logger()

def create_dataset(experiment_ft_dir: str, dataset: str, classes: List[str], train_replicas: int, crop_size: int):  
    dataset_preparator = FTDatasetPreparator(experiment_ft_dir, dataset, classes)
    dataset_preparator.create_subdirectories()
    
    n_crops_per_instance = dataset_preparator.extract_base_crops(train_replicas, crop_size)
    n_crops_per_instance_path = os.path.join(experiment_ft_dir, "n_crops_per_instance.json")
    with open(n_crops_per_instance_path, "w") as f: json.dump(n_crops_per_instance, f, indent=4)

def get_train_rgb_mean_std(experiment_ft_dir: str, dataset: str, classes: List[str]) -> Tuple[List[float], List[float]]:
    train_mean_std_computer = TrainRGBMeanStdComputer(experiment_ft_dir, dataset, classes)
    return train_mean_std_computer()

def get_dataloader(directory: str, classes: List[str], phase: str, batch_size: int, model_input_size: int, mean_: List[float], std_: List[float], device: str, random_seed: int | None = None, epoch: int = 1):
    if phase == "train": return TrainDataLoader(directory, classes, batch_size, model_input_size, mean_, std_, device, random_seed, epoch)
    else: return TestDataLoader(directory, classes, batch_size, model_input_size, mean_, std_, device)

def remove_subdirectories(experiment_ft_dir: str, dataset: str, classes: List[str]):
    dataset_preparator = FTDatasetPreparator(experiment_ft_dir, dataset, classes)
    dataset_preparator.remove_subdirectories()