import os
from typing import Dict, Any, Tuple

from src.utils.constants import METADATA_ROOT
from src.utils.metadata.metadata_handler import MetadataHandler

def get_experiment_metadata(experiment_id: str) -> Dict[str, Any]:
    exp_metadata_path = os.path.join(METADATA_ROOT, experiment_id, "general-metadata.json")
    mh = MetadataHandler(exp_metadata_path)
    return mh.load_metadata()

def get_ft_metadata(experiment_id: str) -> Tuple[Dict[str, Any], MetadataHandler]:
    ft_metadata_path = os.path.join(METADATA_ROOT, experiment_id, "ft-metadata.json")
    mh = MetadataHandler(ft_metadata_path)
    return mh.load_metadata(), mh

def initialize_ft_metadata(ft_metadata: Dict[str, Any], ft_mh: MetadataHandler, batch_size: int, ch_layers: str, opt: str, lr: float, lr_scheduler: str, lr_final_decay_ratio: float, 
                           early_stopping: str, crop_size: int, train_replicas: int, random_seed: int, epochs: int, ft_mode: str, keep_crops: str) -> Dict[str, Any]:
    if "HYPERPARAMETERS" not in ft_metadata:
        ft_metadata["HYPERPARAMETERS"] = {"batch_size": batch_size, "ch_layers": ch_layers, "optimizer": opt, "lr": lr,  "lr_scheduler": lr_scheduler, "lr_final_decay_ratio": lr_final_decay_ratio,
                                          "early_stopping": early_stopping, "crop_size": crop_size ,"train_replicas": train_replicas, 
                                          "random_seed": random_seed, "total_epochs": epochs, "ft_mode": ft_mode, "keep_crops": keep_crops}
    if "FINE_TUNING_DETAILS" not in ft_metadata: ft_metadata["FINE_TUNING_DETAILS"] = {}
    if "TIMESTAMPS" not in ft_metadata: ft_metadata["TIMESTAMPS"] = {}
    
    ft_mh.save_metadata(ft_metadata)
    return ft_metadata