import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

class CropInferenceExecutor:
    def __init__(self, model: nn.Module, device: str):
        self.model = model                                                      
        self.device = device
        
        self.model.eval()
        self.model.to(self.device)
    
    def compute_viz_features(self, loader: DataLoader) -> torch.Tensor:
        viz_features_list = []
        
        with torch.inference_mode():
            for batch, _ in tqdm(loader, desc="Visual Features extraction", position=0, leave=True, dynamic_ncols=True):
                batch = batch.to(self.device)
                viz_features = self.model.feature_encoder(batch)    # (B, feat_dim)
                viz_features_list.append(viz_features.detach().cpu())
        
        return torch.cat(viz_features_list, dim=0)
    
    def compute_probs(self, loader: DataLoader) -> torch.Tensor:
        probs_list = []
        
        with torch.inference_mode():
            for batch, _ in tqdm(loader, desc="Probabilities computation", position=0, leave=True, dynamic_ncols=True):
                batch  = batch.to(self.device)
                logits = self.model(batch) # (B, n_classes)
                probs  = torch.softmax(logits, dim=1)             # (B, n_classes)
                probs_list.append(probs.detach().cpu())
        
        return torch.cat(probs_list, dim=0)