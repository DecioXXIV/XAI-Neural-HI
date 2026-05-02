import torch.nn as nn
import numpy as np
from functools import partial
from typing import List, Dict, Any
from math import comb

from src.explainers.base_explainers import BaseLimeExplainer

class GLimeBinomialExplainer(BaseLimeExplainer):
    def __init__(self, xai_entry: str, model: nn.Module, mean_: List[float], std_: List[float], ft_metadata: Dict[str, Any], xai_metadata: Dict[str, Any], device: str):
        super().__init__(xai_entry, model, mean_, std_, ft_metadata, xai_metadata, device)
        
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
            
            bin_vectors.append(sample)
        
        bin_vectors = np.array(bin_vectors)
        bin_vectors[0, :] = 1 # The first sample is always the original image (all superpixels active)
        
        return bin_vectors
    
    def compute_sample_weights(self, bin_vectors: np.ndarray) -> np.ndarray | None: return None