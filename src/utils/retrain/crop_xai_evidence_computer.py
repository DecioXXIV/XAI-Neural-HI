import os, json
import numpy as np
from typing import List, Dict

from src.utils.retrain.crop_utils import get_page_segments_and_scores

class CropXaiEvidenceComputer:
    def __init__(self, base_dir: str, experiment_xai_dir: str, crop_set: str, color: str):
        self.base_dir = base_dir
        self.experiment_xai_dir = experiment_xai_dir
        self.crop_set = crop_set    # "ft1" or "xai_guided"
        self.color = color          # "green" or "red"
        
        if self.crop_set == "ft1": coords_path = os.path.join(self.base_dir, "coords_to_ft1_train_crop.json")
        else: coords_path = os.path.join(self.base_dir, "coords_to_xai_crop.json")
        with open(coords_path, "r") as f: self.coords = json.load(f)
        
        self.page_data_cache = {}
    
    def __call__(self, batch_paths: List[str]) -> List[float]:
        evidences = []
        
        for path in batch_paths:
            left, top, right, bottom = self.coords[path]
            crop_name = os.path.basename(path)
            
            page_name = crop_name.split("_")[0]
            segments, scores = get_page_segments_and_scores(self.page_data_cache, self.experiment_xai_dir, page_name)
            crop_size = bottom - top + 1
            half = crop_size // 2
            cx, cy = left + half, top + half
            segments_padded = np.pad(segments, ((half, half), (half, half)), mode="constant", constant_values=-1)
            crop_segments = segments_padded[cy:cy + crop_size, cx:cx + crop_size]
            crop_patches = np.unique(crop_segments)
            
            greeness, redness = self._compute_greeness(crop_patches, scores), self._compute_redness(crop_patches, scores)
            
            if self.color == "green": evidence = greeness**2 / (greeness + redness + 1e-8)
            else: evidence = redness**2 / (greeness + redness + 1e-8)
            
            evidences.append(evidence)
    
        return evidences
    
    def _compute_greeness(self, crop_patches: np.ndarray, scores: Dict[str, float]) -> float:
        greeness = 0.0
        for patch in crop_patches: greeness += max(0.0, scores.get(str(patch), 0.0))
        return greeness / len(crop_patches) if len(crop_patches) > 0 else 0.0
    
    def _compute_redness(self, crop_patches: np.ndarray, scores: Dict[str, float]) -> float:
        redness = 0.0
        for patch in crop_patches: redness += max(0.0, -scores.get(str(patch), 0.0))
        return redness / len(crop_patches) if len(crop_patches) > 0 else 0.0    