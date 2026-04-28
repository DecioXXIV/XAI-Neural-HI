import os, sys, json
import numpy as np
from argparse import ArgumentParser
from typing import List, Tuple

from src import OPTIMIZERS, LR_SCHEDULERS, FT_MODES
from src.utils.logger import Logger
from src.utils.fine_tuning.ft_dataset_preparator import FTDatasetPreparator
from src.utils.fine_tuning.train_rgb_mean_std_computer import TrainRGBMeanStdComputer
from src.utils.data.dataloaders import TrainDataLoader, TestDataLoader

logger = Logger()

def get_ft_args():
    parser = ArgumentParser()
    parser.add_argument("-experiment_id", type=str, required=True)
    parser.add_argument("-crop_size", type=int, required=True)
    parser.add_argument("-batch_size", type=int, required=True)
    parser.add_argument("-opt", type=str, required=True, choices=OPTIMIZERS)
    parser.add_argument("-lr", type=float, required=True)
    parser.add_argument("-lr_scheduler", type=str, required=True, choices=LR_SCHEDULERS)
    parser.add_argument("-lr_final_decay_ratio", type=float, default=0.01)
    parser.add_argument("-early_stopping", type=str, default="true")
    parser.add_argument("-train_replicas", type=int, default=1)
    parser.add_argument("-random_seed", type=int, default=None)
    parser.add_argument("-epochs", type=int, default=50)
    parser.add_argument("-ft_mode", type=str, required=True, choices=FT_MODES)
    parser.add_argument("-keep_crops", type=str, default="false")
    return _validate_args(parser.parse_args())

def _validate_args(args):
    experiment_id = args.experiment_id
    crop_size, batch_size, opt, lr, lr_scheduler, lr_final_decay_ratio, early_stopping = args.crop_size, args.batch_size, args.opt, args.lr, args.lr_scheduler, args.lr_final_decay_ratio, args.early_stopping
    train_replicas, random_seed, epochs, ft_mode, keep_crops = args.train_replicas, args.random_seed, args.epochs, args.ft_mode, args.keep_crops
    
    error_trigger = False
    if crop_size <= 0: 
        logger.error("crop_size must be a positive integer")
        error_trigger = True
    if batch_size <= 0: 
        logger.error("batch_size must be a positive integer")
        error_trigger = True
    if train_replicas <= 0: 
        logger.error("train_replicas must be a positive integer")
        error_trigger = True
    if lr <= 0: 
        logger.error("lr must be a positive float")
        error_trigger = True
    if not (0 < lr_final_decay_ratio < 1):
        logger.error("lr_final_decay_ratio must be a float in the range (0, 1)")
        error_trigger = True
    if epochs <= 0: 
        logger.error("epochs must be a positive integer")
        error_trigger = True
    if random_seed is not None and random_seed < 0:
        logger.error("random_seed must be a non-negative integer")
        error_trigger = True
    
    early_stopping, keep_crops = args.early_stopping.lower(), args.keep_crops.lower()
    if early_stopping not in ["true", "false"]:
        logger.error("early_stopping must be 'true' or 'false'")
        error_trigger = True
    if keep_crops not in ["true", "false"]:
        logger.error("keep_crops must be 'true' or 'false'")
        error_trigger = True
    
    if error_trigger: sys.exit()
    
    if random_seed is None:
        random_seed = int(np.random.randint(0, 2**32 - 1))
        logger.warning(f"No random_seed provided. Using '{random_seed}' (randomly generated).")

    return experiment_id, crop_size, batch_size, opt, lr, lr_scheduler, lr_final_decay_ratio, early_stopping, train_replicas, random_seed, epochs, ft_mode, keep_crops

def create_dataset(experiment_ft_dir: str, dataset: str, classes: List[str], train_replicas: int, crop_size: int):  
    dataset_preparator = FTDatasetPreparator(experiment_ft_dir, dataset, classes)
    dataset_preparator.create_subdirectories()
    
    n_crops_per_instance = dataset_preparator.extract_base_crops(train_replicas, crop_size)
    n_crops_per_instance_path = os.path.join(experiment_ft_dir, "n_crops_per_instance.json")
    with open(n_crops_per_instance_path, "w") as f: json.dump(n_crops_per_instance, f, indent=4)

def get_train_rgb_mean_std(experiment_ft_dir: str, dataset: str, classes: List[str]) -> Tuple[List[float], List[float]]:
    train_mean_std_computer = TrainRGBMeanStdComputer(experiment_ft_dir, dataset, classes)
    return train_mean_std_computer()

def get_dataloader(directory: str, classes: List[str], phase: str, batch_size: int, model_input_size: int, mean_: List[float], std_: List[float], device: str):
    if phase == "train": dataloader = TrainDataLoader(directory, classes, batch_size, model_input_size, mean_, std_, device)
    elif phase == "test": dataloader = TestDataLoader(directory, classes, batch_size, model_input_size, mean_, std_, device)
    return dataloader.load_data()

def remove_subdirectories(experiment_ft_dir: str, dataset: str, classes: List[str]):
    dataset_preparator = FTDatasetPreparator(experiment_ft_dir, dataset, classes)
    dataset_preparator.remove_subdirectories()