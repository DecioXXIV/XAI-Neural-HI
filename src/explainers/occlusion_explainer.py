import os, json
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

from src.explainers.base_explainers import BaseExplainer

class OcclusionExplainer(BaseExplainer):
    def __init__(self, xai_entry: str, model: nn.Module, mean_: List[float], std_: List[float], ft_metadata: Dict[str, Any], xai_metadata: Dict[str, Any], device: str):
        super().__init__(xai_entry, model, mean_, std_, ft_metadata, xai_metadata, device)
        
        self.seg_type = self.xai_metadata["Occlusion"][self.xai_entry]["HYPERPARAMETERS"]["seg_type"]
    
    def generate_perturbed_binary_vectors(self, crop_segments: np.ndarray, sp_names: np.ndarray) -> np.ndarray:
        n_features = sp_names.shape[0]
        
        bin_vectors = []
        bin_vectors.append(np.ones(n_features))
        for i in range(0, n_features):
            if self.seg_type == "ink_based" and i == 0: continue # The superpixel with id=0 is the background superpixel, so it is never occluded
            occluded_sample = np.ones(n_features)
            occluded_sample[i] = 0
            bin_vectors.append(occluded_sample)
        
        return np.array(bin_vectors)
    
    def compute_attr_scores(self, bin_vectors: np.ndarray, preds: np.ndarray, label: int, crop_segments: np.ndarray, sp_names: np.ndarray) -> Tuple[Dict[int, float], None]:
        l0 = preds[0, label]
        occlusion_outputs = preds[1:, label]
        
        return {int(sp_name): float(l0 - occlusion_outputs[i]) for i, sp_name in enumerate(sp_names)}, None
    
    def aggregate_crop_scores(self, crops_df: pd.DataFrame, crop_r2s: Dict[str, float], page_xai_dir: str) -> Tuple[Dict[int, float], Dict[int, float]]:
        page_scores = {}
        
        for _, row in crops_df.iterrows():
            crop_name = row["crop_id"]
            crop_xai_dir = os.path.join(page_xai_dir, "crops", crop_name)
            
            with open(os.path.join(crop_xai_dir, "scores.json"), "r") as f: crop_raw_attr_scores = json.load(f)
            
            for sp_name, sp_score in crop_raw_attr_scores.items():
                page_scores.setdefault(int(sp_name), []).append(sp_score)
        
        page_scores = {k: v for k, v in sorted(page_scores.items(), key=lambda item: item[0])}
        aggregated_raw_page_scores = {k: float(np.mean(v)) for k, v in page_scores.items()}
        
        return page_scores, aggregated_raw_page_scores