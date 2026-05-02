import os, PIL, json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
from tqdm import tqdm
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

class BaseExplainer(ABC):
    def __init__(self, xai_entry: str, model: nn.Module, mean_: List[float], std_: List[float], ft_metadata: Dict[str, Any], xai_metadata: Dict[str, Any], device: str):
        self.xai_entry = xai_entry
        self.model = model
        self.mean_ = mean_
        self.std_ = std_
        self.ft_metadata = ft_metadata
        self.xai_metadata = xai_metadata
        self.device = device
        
        self.batch_size = ft_metadata["HYPERPARAMETERS"]["batch_size"]
        self.crop_size = ft_metadata["HYPERPARAMETERS"]["crop_size"]
        self._dp_model = nn.DataParallel(model) if torch.cuda.device_count() > 1 else None
    
    def explain_page(self, page: PIL.Image.Image, label: int, segments: np.ndarray, crop_coordinates_df: pd.DataFrame, page_xai_dir: str):
        crops = self._retrieve_crops(page, crop_coordinates_df, page_xai_dir)
        crop_r2s = {}
        with tqdm(total=len(crops), position=0, leave=True, dynamic_ncols=True) as pbar:
            for i, crop in enumerate(crops):
                r2 = self._explain_crop(crop, crop_coordinates_df.loc[i], segments, label, page_xai_dir)
                crop_name = crop_coordinates_df.iloc[i]["crop_id"]
                if crop_name not in crop_r2s and r2 is not None: crop_r2s[crop_name] = r2
                pbar.update(1)
        
        if len(crop_r2s) > 0:
            with open(os.path.join(page_xai_dir, "crop_r2s.json"), "w") as f: 
                json.dump(crop_r2s, f, indent=4)
        
        all_raw_page_scores, aggregated_raw_page_scores = self.aggregate_crop_scores(crop_coordinates_df, crop_r2s, page_xai_dir)
        with open(os.path.join(page_xai_dir, "all_raw_scores.json"), "w") as f: json.dump(all_raw_page_scores, f, indent=4)
        with open(os.path.join(page_xai_dir, "aggregated_raw_scores.json"), "w") as f: json.dump(aggregated_raw_page_scores, f, indent=4)
        
        aggregated_page_scores = self._normalize_scores(aggregated_raw_page_scores)
        with open(os.path.join(page_xai_dir, "aggregated_scores.json"), "w") as f: json.dump(aggregated_page_scores, f, indent=4)

    def _explain_crop(self, crop: PIL.Image.Image, crop_df_row: pd.Series, segments: np.ndarray, label: int, page_xai_dir: str) -> float | None:
        crop_name = crop_df_row["crop_id"]
        crop_xai_dir = os.path.join(page_xai_dir, "crops", crop_name)
        crop_scores_path = os.path.join(crop_xai_dir, "scores.json")
        
        crop_arr = np.array(crop) if isinstance(crop, PIL.Image.Image) else crop
        # crop_arr.shape -> (crop_size, crop_size, 3); values in [0, 255] (np.uint8)
        fudged_crop = np.zeros_like(crop_arr)
        fudged_crop[:] = [255 * m for m in self.mean_]
        
        left, top, right, bottom = crop_df_row[["left_pixel", "top_pixel", "right_pixel", "bottom_pixel"]]
        right_pad, bottom_pad = crop_df_row[["padding_right", "padding_bottom"]]
        crop_segments = segments[top:bottom+bottom_pad+1, left:right+right_pad+1]
        
        if not os.path.exists(crop_scores_path):
            sp_names = np.unique(crop_segments)
            
            # Build binary vectors
            bin_vectors = self.generate_perturbed_binary_vectors(crop_segments, sp_names)
            
            # Generate samples and get predictions
            preds = self.generate_and_predict_samples_images(crop_arr, bin_vectors, crop_segments, sp_names, fudged_crop, crop_xai_dir)
            
            # Compute attribution scores
            crop_raw_attr_scores, r2 = self.compute_attr_scores(bin_vectors, preds, label, crop_segments, sp_names)
            with open(crop_scores_path, "w") as f: json.dump(crop_raw_attr_scores, f, indent=4)
            return r2
        
        else: return None

    def _retrieve_crops(self, page: PIL.Image.Image, crop_coordinates_df: pd.DataFrame, page_xai_dir: str) -> List[PIL.Image.Image]:
        crops = []
        for _, row in crop_coordinates_df.iterrows():
            crop_name = row["crop_id"]
            crop_path = os.path.join(page_xai_dir, "crops", crop_name, f"{crop_name}.png")
            crops.append(Image.open(crop_path).convert("RGB"))
        
        return crops
    
    @abstractmethod
    def generate_perturbed_binary_vectors(self, crop_segments: np.ndarray, sp_names: np.ndarray) -> np.ndarray: pass
    
    def generate_and_predict_samples_images(self, crop: np.ndarray, perturbed_bin_vectors: np.ndarray, crop_segments: np.ndarray, sp_names: np.ndarray, fudged_crop: np.ndarray, crop_xai_dir: str) -> np.ndarray:
        samples, preds, samples_infos = [], [], {}
        
        for i, row in enumerate(perturbed_bin_vectors):
            pert_sample = np.copy(crop)
            sp_idxs_to_zero = np.where(row == 0)[0]
            mask = np.isin(crop_segments, sp_names[sp_idxs_to_zero])
            pert_sample[mask] = fudged_crop[mask]
            samples.append(Image.fromarray(pert_sample))
            
            samples_infos[f"sample_{i}"] = {
                "masked_sp_idxs": sp_idxs_to_zero.tolist(),
                "n_masked_sps": len(sp_idxs_to_zero)
            }
            
            if self.batch_size is not None and len(samples) == self.batch_size:
                batch_preds = self.model.batch_predict_function(samples, self.mean_, self.std_, self.device, apply_softmax=False, forward_fn=self._dp_model)
                preds.extend(batch_preds)
                samples = []
        
        if self.batch_size is not None and len(samples) > 0:
            final_batch_preds = self.model.batch_predict_function(samples, self.mean_, self.std_, self.device, apply_softmax=False, forward_fn=self._dp_model)
            preds.extend(final_batch_preds)
        
        elif self.batch_size is None:
            preds = self.model.batch_predict_function(samples, self.mean_, self.std_, self.device, apply_softmax=False, forward_fn=self._dp_model)
        
        with open(os.path.join(crop_xai_dir, "samples_info.json"), "w") as f: json.dump(samples_infos, f, indent=4)
        return np.array(preds)
    
    @abstractmethod
    def compute_attr_scores(self, perturbed_bin_vectors: np.ndarray, preds: np.ndarray, label: int, crop_segments: np.ndarray, sp_names: np.ndarray) -> Tuple[Dict[int, float], float | None]: pass
    
    @abstractmethod
    def aggregate_crop_scores(self, crops_df: pd.DataFrame, crop_r2s: Dict[str, float], page_xai_dir: str) -> Tuple[Dict[int, float], Dict[int, float]]: pass
    
    def _normalize_scores(self, raw_scores: Dict[int, float], percentile: int = 99) -> Dict[int, float]:
        keys, values = list(raw_scores.keys()), np.array(list(raw_scores.values()))
        sorted_values = np.sort(values)
        
        cum_sums = np.cumsum(np.abs(sorted_values))
        threshold_id = np.where(cum_sums >= cum_sums[-1] * (percentile / 100))[0][0]
        threshold = sorted_values[threshold_id]
        norm_values = values / threshold
        
        return {int(k): float(v) for k, v in zip(keys, norm_values)}

class BaseLimeExplainer(BaseExplainer):
    def __init__(self, xai_entry: str, model: nn.Module, mean_: List[float], std_: List[float], ft_metadata: Dict[str, Any], xai_metadata: Dict[str, Any], device: str):
        super().__init__(xai_entry, model, mean_, std_, ft_metadata, xai_metadata, device)
        self.model_regressor = Ridge(alpha=1.0, fit_intercept=False)
    
    @staticmethod
    def _kernel(d: np.ndarray, kernel_width: float) -> np.ndarray:
        return np.sqrt(np.exp(-(d**2) / (kernel_width ** 2)))
    
    def compute_sample_weights(self, perturbed_bin_vectors: np.ndarray) -> np.ndarray | None:
        """Subclasses override this method"""
        return None
    
    def compute_attr_scores(self, bin_vectors: np.ndarray, preds: np.ndarray, label: int, crop_segments: np.ndarray, sp_names: np.ndarray) -> Tuple[Dict[int, float], float | None]:
        # Centering the predictions around the original image prediction
        y = preds[:, label]
        y_centered = y - y[0]
        
        # Centering the binary vectors around the original vector (all ones)
        X_centered = bin_vectors - 1
        
        weights = self.compute_sample_weights(bin_vectors)
        
        if weights is not None:
            self.model_regressor.fit(X_centered, y_centered, sample_weight=weights)
            regressor_preds = self.model_regressor.predict(X_centered)
            r2 = r2_score(y_centered, regressor_preds, sample_weight=weights)
        else:
            self.model_regressor.fit(X_centered, y_centered)
            regressor_preds = self.model_regressor.predict(X_centered)
            r2 = r2_score(y_centered, regressor_preds)
        
        raw_attr_scores = {int(sp_name): float(coef) for sp_name, coef in zip(sp_names, self.model_regressor.coef_)}
        
        return raw_attr_scores, r2
    
    def aggregate_crop_scores(self, crops_df: pd.DataFrame, crop_r2s: Dict[str, float], page_xai_dir: str) -> Tuple[Dict[int, float], Dict[int, float]]:
        page_scores = {}
        
        clamped_r2s = {k: max(0.0, v) for k, v in crop_r2s.items()}
        valid_crop_names = {k for k, v in clamped_r2s.items() if v > 0.0}
        use_uniform = len(valid_crop_names) == 0
        
        total_weight = 0.0
        for _, row in crops_df.iterrows():
            crop_name = row["crop_id"]
            
            if use_uniform: weight = 1.0
            elif crop_name in valid_crop_names: weight = clamped_r2s[crop_name]
            else: continue
            
            crop_xai_dir = os.path.join(page_xai_dir, "crops", crop_name)
            with open(os.path.join(crop_xai_dir, "scores.json"), "r") as f: crop_raw_attr_scores = json.load(f)
            
            total_weight += weight
            for sp_name, sp_score in crop_raw_attr_scores.items():
                sp_name = int(sp_name)
                page_scores.setdefault(sp_name, []).append(sp_score * weight)
        
        page_scores = {k: v for k, v in sorted(page_scores.items(), key=lambda item: item[0])}
        aggregated_raw_page_scores = {k: float(np.sum(v) / total_weight) for k, v in page_scores.items()}
        
        return page_scores, aggregated_raw_page_scores