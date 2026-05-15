import os, torch
from torch import nn
from typing import List, Dict, Tuple
from tqdm import tqdm

from src.utils.data.dataloaders import TestDataLoader
from src.utils.retraining.crop_inference_executor import CropInferenceExecutor

class FT1ClassesCentroidsSigmasComputer:
    def __init__(self, base_dir: str, model: nn.Module, classes: List[str], mean_: List[float], std_: List[float], device: str, batch_size: int, crop_size: int):
        self.train_subdir = os.path.join(base_dir, "train")
        self.model = model
        self.classes = classes
        self.mean_ = mean_
        self.std_ = std_
        self.device = device
        self.batch_size = batch_size
        self.crop_size = crop_size
    
    def __call__(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        dataset, loader = TestDataLoader(self.train_subdir, self.classes, self.batch_size, self.crop_size, self.mean_, self.std_, self.device).load_data()
        sample_labels = [s[1] for s in dataset.samples]
        idx_to_cls = {v: k for k, v in dataset.class_to_idx.items()}
        
        all_viz_features = CropInferenceExecutor(self.model, self.device).compute_viz_features(loader)
        
        centroids, sigmas = {}, {}
        for label_idx in tqdm(range(len(self.classes)), desc="Computing centroids and sigmas", position=0, leave=True, dynamic_ncols=True):
            indices = [i for i, label in enumerate(sample_labels) if label == label_idx]
            class_features = all_viz_features[indices]                         # (M, feat_dim)
            centroid       = class_features.mean(dim=0)                         # (feat_dim,)
            centroids[idx_to_cls[label_idx]] = centroid

            dists = torch.norm(class_features - centroid.unsqueeze(0), dim=1)   # (M,)
            sigmas[idx_to_cls[label_idx]] = float(dists.mean().item())
        
        return centroids, sigmas