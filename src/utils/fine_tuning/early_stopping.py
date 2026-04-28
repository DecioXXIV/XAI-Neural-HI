import numpy as np
from typing import Dict, Any

from src.utils.logger import Logger

logger = Logger()

class EarlyStopping(object):
    def __init__(self, metric: str, patience: int=10, delta: float=0.0, max_epochs: int=200):
        self.metric = metric
        self.patience = patience
        self.delta = delta
        self.max_epochs = max_epochs
        self.patience_counter = 0
        self.early_stop = False
        
        if self.metric == "loss": self.best_val_metric = np.inf
        else: self.best_val_metric = -np.inf
    
    def set_best_val_metric(self, value: float): self.best_val_metric = value
    
    def step_before_trigger(self, value: float):
        if self.metric == "loss":
            if value < self.best_val_metric - self.delta:
                self.set_best_val_metric(value)
        else:
            if value > self.best_val_metric + self.delta:
                self.set_best_val_metric(value)
    
    def step(self, value: float) -> bool:
        if self.metric == "loss":
            if value < self.best_val_metric - self.delta:
                logger.info(f"Validation {self.metric} improved from {self.best_val_metric} to {value}. Resetting patience...")
                self.set_best_val_metric(value)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                logger.warning(f"Validation {self.metric} did not improve. Patience counter: {self.patience_counter}/{self.patience}.")
                if self.patience_counter >= self.patience:
                    logger.warning("Early stopping triggered!")
                    self.early_stop = True
        
        else:
            if value > self.best_val_metric + self.delta:
                logger.info(f"Validation {self.metric} improved from {self.best_val_metric} to {value}. Resetting patience...")
                self.set_best_val_metric(value)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                logger.warning(f"Validation {self.metric} did not improve. Patience counter: {self.patience_counter}/{self.patience}.")
                if self.patience_counter >= self.patience:
                    logger.warning("Early stopping triggered!")
                    self.early_stop = True
        
        return self.early_stop
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            "best_val_metric": self.best_val_metric,
            "patience_counter": self.patience_counter,
            "early_stop": self.early_stop
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.best_val_metric = state_dict["best_val_metric"]
        self.patience_counter = state_dict["patience_counter"]
        self.early_stop = state_dict["early_stop"]