import gc, torch, random
import numpy as np
import torch.nn as nn
from typing import List, Tuple, Dict, Any

from src.utils.models.model_loader import ModelLoader
from src.utils.models.model_trainer import ModelTrainer
from src.utils.models.model_tester import ModelTester
from src.utils.fine_tuning.recap_writers import TrainingRecapWriter, TestingRecapWriter
from src.utils.data.dataloaders import TrainDataLoader, TestDataLoader
from src.utils.logger import Logger

logger = Logger()

def load_model(base_dir: str, model_name: str, classes: List[str], ft_mode: str, ch_layers: str, phase: str, device: str, ft_metadata: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    model_loader = ModelLoader(base_dir, model_name, classes, ft_mode, phase, ft_metadata)
    return model_loader(ch_layers.split(','), device)

def train_model(base_dir: str, model: nn.Module, train_dl: TrainDataLoader, val_dl: TestDataLoader,
                device: str, ft_metadata: Dict[str, Any], ft_metadata_path: str, last_cp: Dict[str, Any] | None) -> None:
    model_trainer = ModelTrainer(base_dir, model, train_dl, val_dl, device, ft_metadata, last_cp)
    model_trainer(ft_metadata_path)

    training_recap_writer = TrainingRecapWriter(base_dir)
    training_recap_writer()

def test_model(base_dir: str, model: nn.Module, test_dl: TestDataLoader, device: str, ft_metadata: Dict[str, Any], exp_metadata: Dict[str, Any]) -> None:
    _, t_dl = test_dl.load_data()
    model_tester = ModelTester(base_dir, model, t_dl, device, ft_metadata, exp_metadata)
    crop_labels, crop_preds, crop_logits, crop_probs, crop_preds_per_page, page_labels, page_preds = model_tester()

    testing_recap_writer = TestingRecapWriter(base_dir, t_dl)
    testing_recap_writer(crop_labels, crop_preds, crop_logits, crop_probs, crop_preds_per_page, page_labels, page_preds)

def setup_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
        n_devices = torch.cuda.device_count()
        logger.info(f"Device(s): {[torch.cuda.get_device_name(i) for i in range(n_devices)]}")
    return device

def cleanup_memory(device: str) -> None:
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

def set_random_seed(seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
