import torch.nn as nn
import numpy as np
from functools import partial
from typing import Dict, Any
from math import comb

from src.explainers.base_explainers import BaseLimeExplainer

class GLimeBinomialExplainer(BaseLimeExplainer):
    def __init__(self, experiment_id: str, xai_entry: str, model: nn.Module, exp_metadata: Dict[str, Any], ft_metadata: Dict[str, Any], xai_metadata: Dict[str, Any], device: str):
        super().__init__(experiment_id, xai_entry, model, exp_metadata, ft_metadata, xai_metadata, device)
        
        self.seg_type = self.xai_metadata["GLimeBinomial"][self.xai_entry]["HYPERPARAMETERS"]["seg_type"]
        self.kernel_width = self.xai_metadata["GLimeBinomial"][self.xai_entry]["HYPERPARAMETERS"]["kernel_width"]
        self.num_samples = self.xai_metadata["GLimeBinomial"][self.xai_entry]["HYPERPARAMETERS"]["num_samples"]
        self.kernel_fn = partial(self._kernel, kernel_width=self.kernel_width)
    
    def generate_perturbed_binary_vectors(self, crop_segments: np.ndarray, sp_names: np.ndarray) -> np.ndarray:
        n_features = sp_names.shape[0]
        
        probs = np.array([comb(n_features, i) * (2**(-n_features)) * np.exp((i - n_features)/(self.kernel_width**2)) for i in range(0, n_features)])
        probs = probs / np.sum(probs)
        sample_length = np.random.choice(range(0, n_features), size=self.num_samples, p=probs)
        
        bin_vectors = []
        for i in range(0, self.num_samples):
            sample = np.zeros(n_features)
            
            active_spxs = np.random.choice(range(0, n_features), size=sample_length[i], replace=False) # Indices of active superpixels
            sample[active_spxs] = 1
            if self.seg_type == "ink_based": sample[0] = 1 # The superpixel with id=0 is the background superpixel
            
            bin_vectors.append(sample)
        
        bin_vectors = np.array(bin_vectors)
        bin_vectors[0, :] = 1 # The first sample is always the original image (all superpixels active)
        
        return bin_vectors
    
    def compute_sample_weights(self, bin_vectors: np.ndarray) -> np.ndarray | None: return None
