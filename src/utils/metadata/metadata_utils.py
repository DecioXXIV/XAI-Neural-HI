import os, sys
from typing import Dict, Any, List, Tuple

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
                           weight_decay: float, label_smoothing: float, early_stopping_patience: int, train_replicas: int, random_seed: int, epochs: int, train_transforms: str, ft_mode: str) -> Dict[str, Any]:
    if "HYPERPARAMETERS" not in ft_metadata:
        ft_metadata["HYPERPARAMETERS"] = {"metric": metric, "ch_layers": ch_layers, "crop_size": crop_size, "batch_size": batch_size, "optimizer": opt, "lr": lr,
                                          "lr_scheduler": lr_scheduler, "lr_final_decay_ratio": lr_final_decay_ratio, "weight_decay": weight_decay, "label_smoothing": label_smoothing, 
                                          "early_stopping_patience": early_stopping_patience, "train_replicas": train_replicas, "random_seed": random_seed, "total_epochs": epochs, "train_transforms": train_transforms, "ft_mode": ft_mode}
    if "FINE_TUNING_DETAILS" not in ft_metadata: ft_metadata["FINE_TUNING_DETAILS"] = {}
    if "TIMESTAMPS" not in ft_metadata: ft_metadata["TIMESTAMPS"] = {}
    
    ft_metadata_path = os.path.join(METADATA_ROOT, experiment_id, "ft-metadata.json")
    MetadataHandler(ft_metadata_path).save_metadata(ft_metadata)
    return ft_metadata

def add_timestamp_to_ft_metadata(experiment_id: str, ft_metadata: Dict[str, Any], key: str, timestamp: Any):
    ft_metadata["TIMESTAMPS"][key] = timestamp
    ft_metadata_path = os.path.join(METADATA_ROOT, experiment_id, "ft-metadata.json")
    MetadataHandler(ft_metadata_path).save_metadata(ft_metadata)

def get_xai_metadata(experiment_id: str) -> Dict[str, Any]:
    xai_metadata_path = os.path.join(METADATA_ROOT, experiment_id, "xai-metadata.json")
    return MetadataHandler(xai_metadata_path).load_metadata()

def initialize_xai_metadata(experiment_id: str, xai_metadata: Dict[str, Any], xai_algorithm: str, xai_details: str, subsample: int, save_samples: bool, seg_type: str, patch_dim: int, num_samples: int, kernel_width: float) -> Tuple[Dict[str, Any], str]:
    if xai_algorithm not in xai_metadata: xai_metadata[xai_algorithm] = {}
    
    xai_entry = ""
    if seg_type == "sq_patches":
        xai_entry = f"sq_patches{patch_dim}x{patch_dim}" 
        if xai_algorithm in ("Lime", "GLimeBinomial"):
            xai_entry += f"-kw{kernel_width}-ns{num_samples}"
    xai_entry += f"-{xai_details}"
    
    if xai_entry not in xai_metadata[xai_algorithm]:
        hp_dict = {"seg_type": seg_type}
        if seg_type == "sq_patches": hp_dict["patch_dim"] = patch_dim
        if xai_algorithm in ("Lime", "GLimeBinomial"): 
            hp_dict["kernel_width"] = kernel_width
            hp_dict["num_samples"] = num_samples
        hp_dict["xai_details"] = xai_details
        
        xai_metadata[xai_algorithm][xai_entry] = {"HYPERPARAMETERS": hp_dict}
    
    xai_metadata_path = os.path.join(METADATA_ROOT, experiment_id, "xai-metadata.json")
    MetadataHandler(xai_metadata_path).save_metadata(xai_metadata)
    return xai_metadata, xai_entry

def get_xai_instances_metadata(experiment_xai_dir: str) -> Dict[str, Any]:
    xai_instances_metadata_path = os.path.join(experiment_xai_dir, "xai_instances_metadata.json")
    if os.path.exists(xai_instances_metadata_path):
        return MetadataHandler(xai_instances_metadata_path).load_metadata()
    else:
        return {"INSTANCES": {}}

def add_end_timestamp_to_xai_metadata(experiment_id: str, xai_metadata: Dict[str, Any], xai_algorithm: str, xai_entry: str, timestamp: Any):
    xai_metadata[xai_algorithm][xai_entry]["END_TIMESTAMP"] = timestamp
    xai_metadata_path = os.path.join(METADATA_ROOT, experiment_id, "xai-metadata.json")
    MetadataHandler(xai_metadata_path).save_metadata(xai_metadata)

def initialize_faithfulness_metadata(experiment_id: str, xai_algorithm: str, xai_entry: str) -> Dict[str, Any]:
    faith_metadata_path = os.path.join(METADATA_ROOT, experiment_id, "faithfulness-metadata.json")
    faith_metadata = MetadataHandler(faith_metadata_path).load_metadata()
    
    if xai_algorithm not in faith_metadata: faith_metadata[xai_algorithm] = {}
    if xai_entry not in faith_metadata[xai_algorithm]: faith_metadata[xai_algorithm][xai_entry] = {}
    
    return faith_metadata

def add_end_timestamp_to_faithfulness_metadata(experiment_id: str, faith_metadata: Dict[str, Any], xai_algorithm: str, xai_entry: str, faith_entry: str, timestamp: Any):
    faith_metadata[xai_algorithm][xai_entry][faith_entry] = timestamp
    faith_metadata_path = os.path.join(METADATA_ROOT, experiment_id, "faithfulness-metadata.json")
    MetadataHandler(faith_metadata_path).save_metadata(faith_metadata)

def get_retrain_metadata(experiment_id: str, original_ts_ratio: float, new_ts_ratio: float, selection_rule: str, random_seed: int) -> Dict[str, Any]:
    retrain_metadata_dir = os.path.join(METADATA_ROOT, experiment_id, "retraining")
    os.makedirs(retrain_metadata_dir, exist_ok=True)
    retrain_metadata_path = os.path.join(retrain_metadata_dir, f"{selection_rule}-original{original_ts_ratio}-new{new_ts_ratio}-random_seed{random_seed}-retrain-metadata.json")
    return MetadataHandler(retrain_metadata_path).load_metadata()

def add_timestamp_to_retrain_metadata(retrain_metadata: Dict[str, Any], retrain_metadata_path: str, key: str, timestamp: Any):
    retrain_metadata["TIMESTAMPS"][key] = timestamp
    MetadataHandler(retrain_metadata_path).save_metadata(retrain_metadata)

def initialize_retrain_metadata(experiment_id: str, retrain_metadata: Dict[str, Any], ft_metadata: Dict[str, Any], xai_algorithm: str, xai_entry: str,
                                original_ts_ratio: float, new_ts_ratio: float, selection_rule: str, batch_size: int, lr: float, lr_final_decay_ratio: float, 
                                weight_decay: float, label_smoothing: float, random_seed: int, epochs: int, ft_mode: str, start_point: str) -> Dict[str, Any]:
    
    metric, ch_layers, crop_size = ft_metadata["HYPERPARAMETERS"]["metric"], ft_metadata["HYPERPARAMETERS"]["ch_layers"], ft_metadata["HYPERPARAMETERS"]["crop_size"]
    opt, lr_scheduler, early_stopping = ft_metadata["HYPERPARAMETERS"]["optimizer"], ft_metadata["HYPERPARAMETERS"]["lr_scheduler"], ft_metadata["HYPERPARAMETERS"]["early_stopping"]
    
    if "HYPERPARAMETERS" not in retrain_metadata:
        retrain_metadata["HYPERPARAMETERS"] = {"xai_algorithm": xai_algorithm, "xai_entry": xai_entry, "original_ts_ratio": original_ts_ratio, "new_ts_ratio": new_ts_ratio, 
                                               "selection_rule": selection_rule, "metric": metric, "ch_layers": ch_layers, "crop_size": crop_size, "batch_size": batch_size, "optimizer": opt, "lr": lr, 
                                               "lr_scheduler": lr_scheduler, "lr_final_decay_ratio": lr_final_decay_ratio, "weight_decay": weight_decay, "label_smoothing": label_smoothing, 
                                               "early_stopping": early_stopping, "random_seed": random_seed, "total_epochs": epochs, "ft_mode": ft_mode, "start_point": start_point}
    if "FINE_TUNING_DETAILS" not in retrain_metadata: retrain_metadata["FINE_TUNING_DETAILS"] = {}
    if "TIMESTAMPS" not in retrain_metadata: retrain_metadata["TIMESTAMPS"] = {}
    
    retrain_metadata_path = os.path.join(METADATA_ROOT, experiment_id, "retraining", f"{ft_mode}-{start_point}", selection_rule, f"original{original_ts_ratio}-new{new_ts_ratio}", f"random_seed{random_seed}", "retrain-metadata.json")
    os.makedirs(os.path.dirname(retrain_metadata_path), exist_ok=True)
    MetadataHandler(retrain_metadata_path).save_metadata(retrain_metadata)
    return retrain_metadata, retrain_metadata_path