import os, torch
from typing import List, Tuple, Dict, Any

from src.utils.logger import Logger
from src.models.resnet18 import ResNet18
from src.models.swintiny import SwinTiny

logger = Logger()

class ModelLoader:
    def __init__(self, base_dir: str, model_name: str, classes: List[str], ft_mode: str, phase: str, ft_metadata: Dict[str, Any]):
        self.base_dir = base_dir
        self.model_name = model_name
        self.classes = classes
        self.ft_mode = ft_mode
        self.phase = phase
        self.ft_metadata = ft_metadata

    def __call__(self, ch_layers: List[str], device: str) -> Tuple[ResNet18 | SwinTiny, Dict[str, Any] | None]:
        logger.info(f"Loading Model '{self.model_name}' in '{self.phase}' phase...")
        model, last_cp = None, None
        
        if self.model_name == "ResNet18": model = ResNet18(num_classes=len(self.classes), ft_mode=self.ft_mode, layers=ch_layers, device=device)
        elif self.model_name == "SwinTiny": model = SwinTiny(num_classes=len(self.classes), ft_mode=self.ft_mode, layers=ch_layers)
            
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

    def __call__(self, ch_layers: List[str], device: str) -> Tuple[ResNet18 | SwinTiny, Dict[str, Any] | None]:
        logger.info(f"Loading Fine-Tuned Model '{self.model_name}'...")
        model, last_cp = None, None
        
        if self.model_name == "ResNet18": model = ResNet18(num_classes=len(self.classes), ft_mode=self.ft_mode, layers=ch_layers, device=device)
        elif self.model_name == "SwinTiny": model = SwinTiny(num_classes=len(self.classes), ft_mode=self.ft_mode, layers=ch_layers)
        
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