import os, json
import numpy as np
from typing import List, Tuple, Dict

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
            segments, scores = self._get_page_segments_and_scores(page_name)
            crop_segments = segments[top:bottom+1, left:right+1]
            crop_patches = np.unique(crop_segments)
            
            greeness, redness = self._compute_greeness(crop_patches, scores), self._compute_redness(crop_patches, scores)
            
            if self.color == "green": evidence = greeness**2 / (greeness + redness + 1e-8)
            else: evidence = redness**2 / (greeness + redness + 1e-8)
            
            evidences.append(evidence)
    
        return evidences
    
    def _get_page_segments_and_scores(self, page_name: str) -> Tuple[np.ndarray, Dict[str, float]]:
        if page_name in self.page_data_cache:
            segments, scores = self.page_data_cache[page_name]["segments"], self.page_data_cache[page_name]["scores"]
        else:
            segments = np.load(os.path.join(self.experiment_xai_dir, page_name, "segments.npy"))
            scores_path = os.path.join(self.experiment_xai_dir, page_name, "aggregated_scores.json")
            with open(scores_path, "r") as f: scores = json.load(f)
            self.page_data_cache[page_name] = {"segments": segments, "scores": scores}
        
        return segments, scores

    def _compute_greeness(self, crop_patches: np.ndarray, scores: Dict[str, float]) -> float:
        greeness = 0.0
        for patch in crop_patches: greeness += max(0.0, scores[str(patch)])
        return greeness / len(crop_patches) if len(crop_patches) > 0 else 0.0
    
    def _compute_redness(self, crop_patches: np.ndarray, scores: Dict[str, float]) -> float:
        redness = 0.0
        for patch in crop_patches: redness += max(0.0, -scores[str(patch)])
        return redness / len(crop_patches) if len(crop_patches) > 0 else 0.0    