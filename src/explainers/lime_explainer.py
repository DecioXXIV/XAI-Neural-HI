import torch.nn as nn
import numpy as np
from functools import partial
from typing import List, Dict, Any
from sklearn.metrics import pairwise_distances

from src.explainers.base_explainers import BaseLimeExplainer

class LimeExplainer(BaseLimeExplainer):
    def __init__(self, xai_entry: str, model: nn.Module, mean_: List[float], std_: List[float], ft_metadata: Dict[str, Any], xai_metadata: Dict[str, Any], device: str):
        super().__init__(xai_entry, model, mean_, std_, ft_metadata, xai_metadata, device)
        
        self.kernel_width = self.xai_metadata["Lime"][self.xai_entry]["HYPERPARAMETERS"]["kernel_width"]
        self.num_samples = self.xai_metadata["Lime"][self.xai_entry]["HYPERPARAMETERS"]["num_samples"]
        
        self.kernel_fn = partial(self._kernel, kernel_width=self.kernel_width)
    
    def generate_perturbed_binary_vectors(self, crop_segments: np.ndarray, sp_names: np.ndarray) -> np.ndarray:
        n_features = sp_names.shape[0]
        
        bin_vectors = np.random.randint(0, 2, self.num_samples*n_features).reshape(self.num_samples, n_features)
        bin_vectors[0, :] = 1
        
        return bin_vectors
    
    def compute_sample_weights(self, bin_vectors: np.ndarray) -> np.ndarray:
        distances = pairwise_distances(bin_vectors, bin_vectors[0].reshape(1, -1), metric="euclidean").ravel()
        return self.kernel_fn(distances)