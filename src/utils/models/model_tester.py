import os, json, torch
import numpy as np
import torch.nn as nn
from typing import Dict, Any, Tuple, List
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils.constants import EXPERIMENTS_ROOT
from src.utils.logger import Logger

logger = Logger()

class ModelTester:
    def __init__(self, experiment_id: str, model: nn.Module, test_dl: DataLoader, device: str, ft_metadata: Dict[str, Any], exp_metadata: Dict[str, Any]):
        self.experiment_id = experiment_id
        self.model = model
        self.test_dl = test_dl
        self.device = device
        self.ft_metadata = ft_metadata
        self.exp_metadata = exp_metadata

        self.c_to_idx = test_dl.dataset.class_to_idx
        self.idx_to_c = {v: k for k, v in self.c_to_idx.items()}
        self.target_names = list(self.c_to_idx.keys())

    def _predict(self) -> Tuple[List[int], List[int], List[List[float]], List[List[float]]]:
        labels, preds, logits, probs = [], [], [], []
        pbar = tqdm(self.test_dl, desc="Testing (Crop-Level)", dynamic_ncols=True)

        with torch.no_grad():
            for data, target in pbar:
                if data.dim() == 4:
                    labels.extend(target.numpy().tolist())
                    output = self.model(data.to(self.device))
                    preds.extend(output.argmax(dim=1).cpu().numpy().tolist())
                    logits.extend(output.cpu().numpy().tolist())
                    probs.extend(torch.softmax(output, dim=1).cpu().numpy().tolist())
                elif data.dim() == 5:
                    bs, ncrops, c, h, w = data.size()
                    for i in range(bs):
                        labels.extend([target[i].item()] * ncrops)
                    output = self.model(data.to(self.device).view(-1, c, h, w))
                    preds.extend(output.argmax(dim=1).cpu().numpy().tolist())
                    logits.extend(output.cpu().numpy().tolist())
                    probs.extend(torch.softmax(output, dim=1).cpu().numpy().tolist())

        return labels, preds, logits, probs
    
    def _infer_page_level_predictions(self, crop_labels: List[int], crop_preds: List[int]) -> Tuple[List[int], List[int], Dict[str, List[int]]]:
        crops_per_instance_dict_path = os.path.join(EXPERIMENTS_ROOT, self.experiment_id, "fine_tuning", "n_crops_per_instance.json")
        with open(crops_per_instance_dict_path, "r") as f: crops_per_test_instance_dict = json.load(f)["test"]
        # crop_per_test_instance_dict = {"page_path": n_crops_extracted_from_that_page}
        
        crop_preds_per_page, pl_labels, pl_preds = {}, [], []
        page_list = [os.path.basename(k) for k in crops_per_test_instance_dict.keys()]
        n_crops_per_page_list = list(crops_per_test_instance_dict.values())
        offset = 0
        
        for p, n in zip(page_list, n_crops_per_page_list):
            crop_preds_per_page[p] = crop_preds[offset:offset + n]
            pl_labels.append(crop_labels[offset])
            pl_preds.append(int(np.argmax(np.bincount(crop_preds[offset:offset + n]))))
            offset += n
        
        return pl_labels, pl_preds, crop_preds_per_page

    def __call__(self) -> Tuple[List[int], List[int], List[List[float]], List[List[float]], Dict[str, List[int]], List[int], List[int]]:
        self.model.eval()
        self.model.to(self.device)
        crop_labels, crop_preds, crop_logits, crop_probs = self._predict()
        page_labels, page_preds, crop_preds_per_page = self._infer_page_level_predictions(crop_labels, crop_preds)
        return crop_labels, crop_preds, crop_logits, crop_probs, crop_preds_per_page, page_labels, page_preds