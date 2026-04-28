import torch.nn as nn
from typing import List, Tuple, Dict, Any
from torch.utils.data import DataLoader

from src.utils.models.model_loader import ModelLoader
from src.utils.models.model_trainer import ModelTrainer
from src.utils.models.model_tester import ModelTester
from src.utils.fine_tuning.recap_writers import TrainingRecapWriter, TestingRecapWriter

def load_model(experiment_id: str, model_name: str, classes: List[str], ft_mode: str, phase: str, ft_metadata: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    model_loader = ModelLoader(experiment_id, model_name, classes, ft_mode, phase, ft_metadata)
    return model_loader()

def train_model(experiment_id: str, model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, device: str, ft_metadata: Dict[str, Any], last_cp: Dict[str, Any] | None) -> None:
    model_trainer = ModelTrainer(experiment_id, model, train_dl, val_dl, device, ft_metadata, last_cp)
    model_trainer()
    
    training_recap_writer = TrainingRecapWriter(experiment_id)
    training_recap_writer()

def test_model(experiment_id: str, model: nn.Module, test_dl: DataLoader, device: str, ft_metadata: Dict[str, Any], exp_metadata: Dict[str, Any]) -> None:
    model_tester = ModelTester(experiment_id, model, test_dl, device, ft_metadata, exp_metadata)
    crop_labels, crop_preds, page_labels, page_preds = model_tester()

    testing_recap_writer = TestingRecapWriter(experiment_id, test_dl)
    testing_recap_writer(crop_labels, crop_preds, page_labels, page_preds)