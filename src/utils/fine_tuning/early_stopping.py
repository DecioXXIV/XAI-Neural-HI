import numpy as np
from typing import Dict, Any

from src.utils.logger import Logger

logger = Logger()

class EarlyStopping(object):
    def __init__(self, patience: int=10, delta: float=0.0, max_epochs: int=200):
        self.patience = patience
        self.delta = delta
        self.max_epochs = max_epochs
        self.best_val_loss = np.inf
        self.patience_counter = 0
        self.early_stop = False
    
    def set_best_val_loss(self, val_loss: float): self.best_val_loss = val_loss
    
    def step_before_trigger(self, val_loss: float):
        if val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
    
    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_val_loss - self.delta:
            logger.info(f"Validation loss improved from {self.best_val_loss} to {val_loss}. Resetting patience...")
            self.set_best_val_loss(val_loss)
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            logger.warning(f"Validation loss did not improve. Patience counter: {self.patience_counter}/{self.patience}.")
            if self.patience_counter >= self.patience:
                logger.warning("Early stopping triggered!")
                self.early_stop = True
        
        return self.early_stop
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            "best_val_loss": self.best_val_loss,
            "patience_counter": self.patience_counter,
            "early_stop": self.early_stop
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.best_val_loss = state_dict["best_val_loss"]
        self.patience_counter = state_dict["patience_counter"]
        self.early_stop = state_dict["early_stop"]