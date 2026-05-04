import os, json, torch
import numpy as np

import pandas as pd
import torchvision.transforms as T
from abc import ABC, abstractmethod
from typing import List, Tuple
from PIL import Image

from src.utils.constants import EXPERIMENTS_ROOT
from src.utils.metadata.metadata_handler import MetadataHandler
from src.utils.faithfulness.patch_indexer import PatchIndexer
from src.maskers.image_masker import ImageMasker

class SqPatchesImageMasker(ImageMasker):
    def _build_masking_results(self, instance_name: str) -> pd.DataFrame:
        xai_instance_dir = os.path.join(EXPERIMENTS_ROOT, self.experiment_id, "xai", self.xai_algorithm, self.xai_entry, instance_name)
        
        segments = np.load(os.path.join(xai_instance_dir, "segments.npy"))
        with open(os.path.join(xai_instance_dir, "aggregated_scores.json"), 'r') as f: scores = json.load(f)
        
        patch_indexer = PatchIndexer()
        rows, bboxes = [], patch_indexer(segments)
        
        for patch_id, score in scores.items():
            bb = bboxes[patch_id]
            rows.append([patch_id, score, bb["left"], bb["top"], bb["right"], bb["bottom"], bb["area"]])
        
        df = pd.DataFrame(rows, columns=["patch_id", "score", "left_pixel", "top_pixel", "right_pixel", "bottom_pixel", "area"])
        os.makedirs(os.path.join(self.masking_root, "masking_results"), exist_ok=True)
        df.to_csv(os.path.join(self.masking_root, "masking_results", f"{instance_name}_masking_results.csv"), index=False)
        return df

    def _sort_and_filter_masking_results(self, masking_results: pd.DataFrame) -> pd.DataFrame:
        if self.patches_color == "green": return masking_results[masking_results["score"] >= 0].sort_values(by="score", ascending=False).reset_index(drop=True)
        else: return masking_results[masking_results["score"] < 0].sort_values(by="score", ascending=True).reset_index(drop=True)

    @property
    @abstractmethod
    def mask_rule(self) -> str: pass

    @abstractmethod
    def _get_patch_iteration_data(self, masking_results: pd.DataFrame, filtered_masking_results: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: pass

    @abstractmethod
    def _prepare_iteration(self, img_tensor: torch.Tensor, cumulative_state) -> Tuple[torch.Tensor, List[str]]: pass

    @abstractmethod
    def _update_cumulative_state(self, cumulative_state, masked_patches: List[str]): pass

    def _perform_masking(self, img_tensor: torch.Tensor, mr: float, masking_results: pd.DataFrame, filtered_masking_results: pd.DataFrame, masked_patches: List[str]):
        if len(filtered_masking_results) == 0: return img_tensor, 0, masked_patches
        
        masked_area, page_area = 0, img_tensor.shape[1] * img_tensor.shape[2]
        coords, patch_ids, pick_order = self._get_patch_iteration_data(masking_results, filtered_masking_results)
        
        for idx in pick_order:
            left, top, right, bottom, area = coords[idx]
            img_tensor[:, top:bottom+1, left:right+1] = self.masking_color
            masked_area += area
            patch_id = patch_ids[idx]
            if patch_id not in masked_patches: masked_patches.append(patch_id)
            if self._evaluate_stop_condition(mr, page_area, masked_area, filtered_masking_results, masked_patches): break
        
        return img_tensor, masked_area, masked_patches
    
    def _evaluate_stop_condition(self, mr: float, page_area: int, masked_area: int, filtered_masking_results: pd.DataFrame, masked_patches: List[str]) -> bool:
        return len(masked_patches) >= mr * len(filtered_masking_results)
    
    def mask_instance(self, instance_path: str, instance_name: str):
        original_img = Image.open(instance_path).convert("RGB")
        original_img_tensor = T.ToTensor()(original_img)
        
        # PHASE 1: Patches Mapping
        masking_results_path = os.path.join(self.masking_root, "masking_results", f"{instance_name}_masking_results.csv")
        if os.path.exists(masking_results_path): masking_results = pd.read_csv(masking_results_path)
        else: masking_results = self._build_masking_results(instance_name)
        
        # PHASE 2: Masking Process
        img_area = original_img_tensor.shape[1] * original_img_tensor.shape[2]
        filtered_masking_results = self._sort_and_filter_masking_results(masking_results)
        cumulative_state = None
        
        mask_metadata = {}
        
        for mr in self.mask_rates:
            img_tensor, masked_patches = self._prepare_iteration(original_img_tensor, cumulative_state)
            img_tensor, masked_area, masked_patches = self._perform_masking(img_tensor, mr, masking_results, filtered_masking_results, masked_patches)
            cumulative_state = self._update_cumulative_state(cumulative_state, masked_patches)
            
            out_img = T.ToPILImage()(img_tensor)
            out_dir = os.path.join(self.masking_root, f"{self.mask_rule}-ceil{self.mask_rates[-1]}-step{self.mask_rates[0]}-{self.patches_color}", f"mask_rate{mr}")
            os.makedirs(out_dir, exist_ok=True)
            out_img.save(os.path.join(out_dir, f"{instance_name}.png"))
            
            mask_metadata[mr] = {
                "masked_area_ratio": float(masked_area / img_area),
                "n_masked_patches": len(masked_patches),
                "masked_patches": [str(p) for p in masked_patches],
            }
            
        mask_metadata_dir = os.path.join(self.masking_root, "metadata", f"{self.mask_rule}-ceil{self.mask_rates[-1]}-step{self.mask_rates[0]}-{self.patches_color}")
        os.makedirs(mask_metadata_dir, exist_ok=True)
        mask_metadata_path = os.path.join(mask_metadata_dir, f"{instance_name}.json")

        MetadataHandler(mask_metadata_path).save_metadata(mask_metadata)

class SqPatchesSaliencyImageMasker(SqPatchesImageMasker):
    @property
    def mask_rule(self) -> str: return "saliency"

    def _get_patch_iteration_data(self, masking_results: pd.DataFrame, filtered_masking_results: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        coords = filtered_masking_results[["left_pixel", "top_pixel", "right_pixel", "bottom_pixel", "area"]].to_numpy()
        patch_ids = filtered_masking_results["patch_id"].to_numpy()
        return coords, patch_ids, list(range(len(coords)))

    def _prepare_iteration(self, img_tensor: torch.Tensor, cumulative_state) -> Tuple[torch.Tensor, List[str]]:
        masked_patches = cumulative_state if cumulative_state is not None else []
        return img_tensor, masked_patches  # No '.clone()': in-place masking accumulates across iterations

    def _update_cumulative_state(self, cumulative_state, masked_patches: List[str]) -> List[str]:
        return masked_patches  # Carry the same list forward to the next iteration

class SqPatchesRandomImageMasker(SqPatchesImageMasker):
    @property
    def mask_rule(self) -> str: return "random"

    def _get_patch_iteration_data(self, masking_results: pd.DataFrame, filtered_masking_results: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        coords = masking_results[["left_pixel", "top_pixel", "right_pixel", "bottom_pixel", "area"]].to_numpy()
        patch_ids = masking_results["patch_id"].to_numpy()
        return coords, patch_ids, np.random.permutation(len(coords))

    def _prepare_iteration(self, img_tensor: torch.Tensor, cumulative_state) -> Tuple[torch.Tensor, List[str]]:
        return img_tensor.clone(), []  # '.clone()': each iteration is fully independent

    def _update_cumulative_state(self, cumulative_state, masked_patches: List[str]) -> None:
        return None  # No state to carry forward between independent iterations