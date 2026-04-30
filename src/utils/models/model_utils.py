import torch.nn as nn
from typing import List, Tuple, Dict, Any
from torch.utils.data import DataLoader

from src.utils.models.model_loader import ModelLoader
from src.utils.models.model_trainer import ModelTrainer
from src.utils.models.model_tester import ModelTester
from src.utils.fine_tuning.recap_writers import TrainingRecapWriter, TestingRecapWriter
from src.utils.data.dataloaders import TrainDataLoader, TestDataLoader

def load_model(experiment_id: str, model_name: str, classes: List[str], ft_mode: str, ch_layers: str, phase: str, ft_metadata: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    model_loader = ModelLoader(experiment_id, model_name, classes, ft_mode, phase, ft_metadata)
    return model_loader(ch_layers.split(','))

def train_model(experiment_id: str, model: nn.Module, train_dl: TrainDataLoader, val_dl: TestDataLoader,
                device: str, ft_metadata: Dict[str, Any], last_cp: Dict[str, Any] | None) -> None:
    model_trainer = ModelTrainer(experiment_id, model, train_dl, val_dl, device, ft_metadata, last_cp)
    model_trainer()

    training_recap_writer = TrainingRecapWriter(experiment_id)
    training_recap_writer()

def test_model(experiment_id: str, model: nn.Module, test_dl: TestDataLoader, device: str, ft_metadata: Dict[str, Any], exp_metadata: Dict[str, Any]) -> None:
    _, t_dl = test_dl.load_data()
    model_tester = ModelTester(experiment_id, model, t_dl, device, ft_metadata, exp_metadata)
    crop_labels, crop_preds, crop_logits, crop_probs, crop_preds_per_page, page_labels, page_preds = model_tester()

    testing_recap_writer = TestingRecapWriter(experiment_id, t_dl)
    testing_recap_writer(crop_labels, crop_preds, crop_logits, crop_probs, crop_preds_per_page, page_labels, page_preds)