import os
import pickle as pkl
from typing import List

from src.utils.constants import EXPERIMENTS_ROOT

class HistoryHandler:
    def __init__(self, experiment_id: str):
        self.history_dir = os.path.join(EXPERIMENTS_ROOT, experiment_id, "fine_tuning", "history")
        os.makedirs(self.history_dir, exist_ok=True)
    
    def load_history(self, filename: str) -> List[float]:
        filepath = os.path.join(self.history_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return pkl.load(f)
        else:
            return []
    
    def save_history(self, filename: str, data: List[float]):
        filepath = os.path.join(self.history_dir, filename)
        with open(filepath, "wb") as f:
            pkl.dump(data, f)