import os, json
from typing import List, Tuple, Dict, Any

from src.utils.logger import Logger
from src.utils.fine_tuning.ft_dataset_preparator import FTDatasetPreparator
from src.utils.fine_tuning.train_rgb_mean_std_computer import TrainRGBMeanStdComputer
from src.utils.data.dataloaders import TrainDataLoader, TestDataLoader

logger = Logger()

def create_dataset(experiment_ft_dir: str, train_replicas: int, crop_size: int, exp_metadata: Dict[str, Any]):  
    dataset, classes = exp_metadata["DATASET"], exp_metadata["CLASSES"]
    dataset_preparator = FTDatasetPreparator(experiment_ft_dir, dataset, classes)
    dataset_preparator.create_subdirectories()
    
    n_crops_per_instance = dataset_preparator.extract_base_crops(train_replicas, crop_size)
    n_crops_per_instance_path = os.path.join(experiment_ft_dir, "n_crops_per_instance.json")
    with open(n_crops_per_instance_path, "w") as f: json.dump(n_crops_per_instance, f, indent=4)

def get_train_rgb_mean_std(experiment_ft_dir: str, exp_metadata: Dict[str, Any]) -> Tuple[List[float], List[float]]:
    dataset, classes = exp_metadata["DATASET"], exp_metadata["CLASSES"]
    train_mean_std_computer = TrainRGBMeanStdComputer(experiment_ft_dir, dataset, classes)
    return train_mean_std_computer()

def get_dataloader(directory: str, phase: str, model_input_size: int, mean_: List[float], std_: List[float], exp_metadata: Dict[str, Any], ft_metadata: Dict[str, Any], device: str, epoch: int = 1):
    classes = exp_metadata["CLASSES"]
    batch_size, train_transforms, random_seed = ft_metadata["HYPERPARAMETERS"]["batch_size"], ft_metadata["HYPERPARAMETERS"]["train_transforms"], ft_metadata["HYPERPARAMETERS"]["random_seed"]
    
    if phase == "": data_dir = directory
    else: data_dir = os.path.join(directory, phase)
    
    if phase == "train": return TrainDataLoader(data_dir, classes, batch_size, model_input_size, mean_, std_, device, random_seed, train_transforms, epoch)
    else: return TestDataLoader(data_dir, classes, batch_size, model_input_size, mean_, std_, device)

def remove_subdirectories(experiment_ft_dir: str, exp_metadata: Dict[str, Any]):
    dataset, classes = exp_metadata["DATASET"], exp_metadata["CLASSES"]
    dataset_preparator = FTDatasetPreparator(experiment_ft_dir, dataset, classes)
    dataset_preparator.remove_subdirectories()