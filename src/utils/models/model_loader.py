import os, torch
from typing import List, Tuple, Dict, Any

from src.utils.logger import Logger
from src.models.resnet18 import ResNet18
from src.models.swintiny import SwinTiny
from src.models.swinsmall import SwinSmall

logger = Logger()

def _instantiate_model(model_name: str, classes: List[str], ft_mode: str, ch_layers: List[str], device: str) -> ResNet18 | SwinTiny | SwinSmall:
    if model_name == "ResNet18": return ResNet18(num_classes=len(classes), ft_mode=ft_mode, layers=ch_layers, device=device)
    elif model_name == "SwinTiny": return SwinTiny(num_classes=len(classes), ft_mode=ft_mode, layers=ch_layers)
    elif model_name == "SwinSmall": return SwinSmall(num_classes=len(classes), ft_mode=ft_mode, layers=ch_layers)
    raise ValueError(f"Unknown model name: '{model_name}'")

class ModelLoader:
    def __init__(self, base_dir: str, model_name: str, classes: List[str], ft_mode: str, phase: str, ft_metadata: Dict[str, Any]):
        self.base_dir = base_dir
        self.model_name = model_name
        self.classes = classes
        self.ft_mode = ft_mode
        self.phase = phase
        self.ft_metadata = ft_metadata

    def __call__(self, ch_layers: List[str], device: str) -> Tuple[ResNet18 | SwinTiny | SwinSmall, Dict[str, Any] | None]:
        logger.info(f"Loading Model '{self.model_name}' in '{self.phase}' phase...")
        model = _instantiate_model(self.model_name, self.classes, self.ft_mode, ch_layers, device)
        last_cp = None

        if self.phase == "train":
            if "EPOCHS_COMPLETED" in self.ft_metadata["FINE_TUNING_DETAILS"]:
                epochs = self.ft_metadata["HYPERPARAMETERS"]["total_epochs"]
                epochs_completed = self.ft_metadata["FINE_TUNING_DETAILS"]["EPOCHS_COMPLETED"]
                epochs_remaining = epochs - epochs_completed
                    
                logger.warning(f"Resuming Fine-Tuning from a previous checkpoint: {epochs_completed}/{epochs} epochs have already been completed, {epochs_remaining} remaining...\n")
                    
                last_cp_path = os.path.join(self.base_dir, "checkpoints", "last_checkpoint.pth")
                last_cp = torch.load(last_cp_path, map_location=device)
                model.load_state_dict(last_cp["model_state_dict"])
            
        else:
            cp_to_test_path = os.path.join(self.base_dir, "checkpoints", "val_best_model.pth")
            cp_to_test = torch.load(cp_to_test_path, map_location=device)
            model.load_state_dict(cp_to_test["model_state_dict"])
                
        logger.info(f"...Model successfully loaded!")
        return model, last_cp

class FineTunedToRetrainModelLoader:
    def __init__(self, experiment_ft_dir: str, experiment_retrain_dir: str, model_name: str, classes: List[str], ft_mode: str, ft_metadata: Dict[str, Any]):
        self.experiment_ft_dir = experiment_ft_dir
        self.experiment_retrain_dir = experiment_retrain_dir
        self.model_name = model_name
        self.classes = classes
        self.ft_mode = ft_mode
        self.ft_metadata = ft_metadata

    def __call__(self, ch_layers: List[str], device: str) -> Tuple[ResNet18 | SwinTiny | SwinSmall, Dict[str, Any] | None]:
        logger.info(f"Loading Fine-Tuned Model '{self.model_name}'...")
        model = _instantiate_model(self.model_name, self.classes, self.ft_mode, ch_layers, device)
        last_cp = None

        if "EPOCHS_COMPLETED" in self.ft_metadata["FINE_TUNING_DETAILS"]:
            epochs = self.ft_metadata["HYPERPARAMETERS"]["total_epochs"]
            epochs_completed = self.ft_metadata["FINE_TUNING_DETAILS"]["EPOCHS_COMPLETED"]
            epochs_remaining = epochs - epochs_completed
                
            logger.warning(f"Resuming Fine-Tuning from a previous checkpoint: {epochs_completed}/{epochs} epochs have already been completed, {epochs_remaining} remaining...\n")
                
            last_cp_path = os.path.join(self.experiment_retrain_dir, "checkpoints", "last_checkpoint.pth")
            last_cp = torch.load(last_cp_path, map_location=device)
            model.load_state_dict(last_cp["model_state_dict"])
        
        else:
            cp_to_load_path = os.path.join(self.experiment_ft_dir, "checkpoints", "val_best_model.pth")
            cp_to_load = torch.load(cp_to_load_path, map_location=device)
            model.load_state_dict(cp_to_load["model_state_dict"])
            
        logger.info(f"...Fine-Tuned Model successfully loaded!")
        return model, last_cp