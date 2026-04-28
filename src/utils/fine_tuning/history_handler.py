import os
import pickle as pkl
from typing import List, Dict

from src.utils.constants import EXPERIMENTS_ROOT

PHASES = ["train", "val"]

class HistoryHandler:
    def __init__(self, experiment_id: str):
        self.history_dir = os.path.join(EXPERIMENTS_ROOT, experiment_id, "fine_tuning", "history")
        os.makedirs(self.history_dir, exist_ok=True)
    
    def load_history(self) -> Dict[str, Dict[str, List[float]]]:
        history = {}
        for phase in PHASES:
            history[phase] = {
                "loss": self._load_pkl(f"{phase}_losses.pkl"),
                "accuracy": self._load_pkl(f"{phase}_accs.pkl"),
                "macrof1": self._load_pkl(f"{phase}_macrof1s.pkl"),
                "weightedf1": self._load_pkl(f"{phase}_weightedf1s.pkl")
            }
        history["learning_rates"] = self._load_pkl("learning_rates.pkl")
        
        return history
    
    def save_history(self, history: Dict[str, Dict[str, List[float]]]):
        for phase in PHASES:
            self._save_pkl(f"{phase}_losses.pkl", history[phase]["loss"])
            self._save_pkl(f"{phase}_accs.pkl", history[phase]["accuracy"])
            self._save_pkl(f"{phase}_macrof1s.pkl", history[phase]["macrof1"])
            self._save_pkl(f"{phase}_weightedf1s.pkl", history[phase]["weightedf1"])
        self._save_pkl("learning_rates.pkl", history["learning_rates"])
    
    def _load_pkl(self, filename: str) -> List[float]:
        filepath = os.path.join(self.history_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                return pkl.load(f)
        else:
            return []
    
    def _save_pkl(self, filename: str, data: List[float]):
        filepath = os.path.join(self.history_dir, filename)
        with open(filepath, "wb") as f:
            pkl.dump(data, f)