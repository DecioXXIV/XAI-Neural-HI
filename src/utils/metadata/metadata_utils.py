import os, sys
from typing import Dict, Any, List

from src.utils.constants import METADATA_ROOT
from src.utils.logger import Logger
from src.utils.metadata.metadata_handler import MetadataHandler

logger = Logger()

def get_experiment_metadata(experiment_id: str) -> Dict[str, Any]:
    exp_metadata_path = os.path.join(METADATA_ROOT, experiment_id, "general-metadata.json")
    return MetadataHandler(exp_metadata_path).load_metadata()

def get_ft_metadata(experiment_id: str) -> Dict[str, Any]:
    ft_metadata_path = os.path.join(METADATA_ROOT, experiment_id, "ft-metadata.json")
    return MetadataHandler(ft_metadata_path).load_metadata()

def create_exp_metadata(experiment_id: str, model_name: str, dataset: str, classes: List[str]):
    os.makedirs(os.path.join(METADATA_ROOT, experiment_id), exist_ok=True)
    exp_metadata_path = os.path.join(METADATA_ROOT, experiment_id, "general-metadata.json")
    
    if os.path.exists(exp_metadata_path):
        logger.warning(f"Experiment ID '{experiment_id}' already exists. Please choose a unique experiment ID.\n")
        sys.exit()
    
    exp_metadata = {"EXPERIMENT_ID": experiment_id, "MODEL_NAME": model_name, "DATASET": dataset, "CLASSES": classes}
    MetadataHandler(exp_metadata_path).save_metadata(exp_metadata)
    logger.info(f"Metadata successfully created for Experiment: '{experiment_id}'\n")

def initialize_ft_metadata(experiment_id: str, ft_metadata: Dict[str, Any], metric: str, ch_layers: str, crop_size: int, batch_size: int, opt: str, lr: float, lr_scheduler: str, lr_final_decay_ratio: float, 
                           weight_decay: float, label_smoothing: float, early_stopping: str, train_replicas: int, random_seed: int, epochs: int, ft_mode: str) -> Dict[str, Any]:
    if "HYPERPARAMETERS" not in ft_metadata:
        ft_metadata["HYPERPARAMETERS"] = {"metric": metric, "ch_layers": ch_layers, "crop_size": crop_size, "batch_size": batch_size, "optimizer": opt, "lr": lr,
                                          "lr_scheduler": lr_scheduler, "lr_final_decay_ratio": lr_final_decay_ratio, "weight_decay": weight_decay, "label_smoothing": label_smoothing, 
                                          "early_stopping": early_stopping, "train_replicas": train_replicas, "random_seed": random_seed, "total_epochs": epochs, "ft_mode": ft_mode}
    if "FINE_TUNING_DETAILS" not in ft_metadata: ft_metadata["FINE_TUNING_DETAILS"] = {}
    if "TIMESTAMPS" not in ft_metadata: ft_metadata["TIMESTAMPS"] = {}
    
    ft_metadata_path = os.path.join(METADATA_ROOT, experiment_id, "ft-metadata.json")
    MetadataHandler(ft_metadata_path).save_metadata(ft_metadata)
    return ft_metadata

def add_timestamp_to_ft_metadata(experiment_id: str, ft_metadata: Dict[str, Any], key: str, timestamp: Any):
    ft_metadata["TIMESTAMPS"][key] = timestamp
    ft_metadata_path = os.path.join(METADATA_ROOT, experiment_id, "ft-metadata.json")
    MetadataHandler(ft_metadata_path).save_metadata(ft_metadata)