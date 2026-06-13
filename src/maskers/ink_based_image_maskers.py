import os, json, hashlib
import torch
import numpy as np
import pandas as pd
from abc import abstractmethod
from typing import List, Tuple
from PIL import Image
from torchvision import transforms as T

from src.utils.constants import EXPERIMENTS_ROOT
from src.utils.explain.ink_based_masking_utils import (
    build_ink_replacement_mask,
    build_ink_replacement_mask_in_bbox,
    estimate_background_color_from_bbox,
)
from src.utils.metadata.metadata_handler import MetadataHandler
from src.utils.faithfulness.patch_indexer import PatchIndexer
from src.maskers.image_masker import ImageMasker

MASKING_RESULTS_COLUMNS = ["segment_id", "score", "left_pixel", "top_pixel", "right_pixel", "bottom_pixel", "segment_area", "bbox_area", "mask_r", "mask_g", "mask_b"]

class InkBasedImageMasker(ImageMasker):
    def _load_segments_and_scores(self, instance_name: str) -> Tuple[np.ndarray, dict]:
        xai_instance_dir = os.path.join(EXPERIMENTS_ROOT, self.experiment_id, "xai", self.xai_algorithm, self.xai_entry, instance_name)
        segments = np.load(os.path.join(xai_instance_dir, "segments.npy"))
        with open(os.path.join(xai_instance_dir, "aggregated_scores.json"), "r") as f: scores = json.load(f)
        return segments, scores

    def _build_masking_results(self, instance_name: str, original_img_arr: np.ndarray) -> pd.DataFrame:
        xai_instance_dir = os.path.join(EXPERIMENTS_ROOT, self.experiment_id, "xai", self.xai_algorithm, self.xai_entry, instance_name)

        segments = np.load(os.path.join(xai_instance_dir, "segments.npy"))
        with open(os.path.join(xai_instance_dir, "aggregated_scores.json"), 'r') as f: scores = json.load(f)

        patch_indexer = PatchIndexer()
        rows, bboxes = [], patch_indexer(segments)
        segment_ids, segment_counts = np.unique(segments, return_counts=True)
        segment_areas = dict(zip(segment_ids, segment_counts))

        for seg_id, score in scores.items():
            if seg_id == "0": continue
            bb = bboxes[seg_id]
            left, top, right, bottom = bb["left"], bb["top"], bb["right"], bb["bottom"]
            segment_area, bb_area = segment_areas[int(seg_id)], bb["area"]
            mask_r, mask_g, mask_b = estimate_background_color_from_bbox(original_img_arr, segments, left, top, right, bottom)
            rows.append([seg_id, score, left, top, right, bottom, segment_area, bb_area, int(mask_r), int(mask_g), int(mask_b)])

        df = pd.DataFrame(rows, columns=MASKING_RESULTS_COLUMNS)
        os.makedirs(os.path.join(self.masking_root, "masking_results"), exist_ok=True)
        df.to_csv(os.path.join(self.masking_root, "masking_results", f"{instance_name}_masking_results.csv"), index=False)
        return df

    @property
    @abstractmethod
    def mask_rule(self) -> str: pass

    def _sort_and_filter_masking_results(self, masking_results: pd.DataFrame) -> pd.DataFrame:
        if self.patches_color == "green":
            return masking_results[masking_results["score"] > 0].sort_values(by="score", ascending=False).reset_index(drop=True)
        else:
            return masking_results[masking_results["score"] < 0].sort_values(by="score", ascending=True).reset_index(drop=True)

    @abstractmethod
    def _get_segment_iteration_data(self, masking_results: pd.DataFrame, filtered_masking_results: pd.DataFrame, rng: np.random.Generator | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]: pass

    @abstractmethod
    def _prepare_iteration(self, img_tensor: torch.Tensor, cumulative_state) -> Tuple[torch.Tensor, List[str]]: pass

    @abstractmethod
    def _update_cumulative_state(self, cumulative_state, masked_segments: List[str]): pass

    def _build_instance_rng(self, instance_name: str) -> np.random.Generator | None:
        return None

    def _get_masking_entry(self) -> str:
        entry = f"{self.mask_rule}-ceil{self.mask_rates[-1]}-step{self.mask_rates[0]}-{self.patches_color}"
        if self.mask_rule == "random" and self.global_seed is not None: entry += f"-seed{self.global_seed}"
        return entry

    def _perform_masking(self, img_tensor: torch.Tensor, segments: np.ndarray, mr: float, masking_results: pd.DataFrame, filtered_masking_results: pd.DataFrame, masked_segments: List[str], rng: np.random.Generator | None = None):
        if len(filtered_masking_results) == 0: return img_tensor, 0, masked_segments
        if self._evaluate_stop_condition(mr, filtered_masking_results, masked_segments):
            masked_area = int(np.count_nonzero(build_ink_replacement_mask(segments, masked_segments)))
            return img_tensor, masked_area, masked_segments

        segment_data, segment_ids, pick_order = self._get_segment_iteration_data(masking_results, filtered_masking_results, rng)

        for idx in pick_order:
            segment_id = int(segment_ids[idx])
            segment_key = str(segment_id)
            if segment_key in masked_segments: continue

            left, top, right, bottom, mask_r, mask_g, mask_b = segment_data[idx]

            bounds, replacement_mask = build_ink_replacement_mask_in_bbox(segments, segment_id, int(left), int(top), int(right), int(bottom))
            if replacement_mask.size == 0 or not np.any(replacement_mask): continue

            left, top, right, bottom = bounds
            replacement_mask = torch.from_numpy(replacement_mask)
            mask_color = torch.tensor([mask_r, mask_g, mask_b], dtype=img_tensor.dtype).view(3, 1) / 255.0
            img_roi = img_tensor[:, top:bottom + 1, left:right + 1]
            img_roi[:, replacement_mask] = mask_color

            masked_segments.append(segment_key)
            if self._evaluate_stop_condition(mr, filtered_masking_results, masked_segments): break

        masked_area = int(np.count_nonzero(build_ink_replacement_mask(segments, masked_segments)))
        return img_tensor, masked_area, masked_segments

    def _evaluate_stop_condition(self, mr: float, filtered_masking_results: pd.DataFrame, masked_segments: List[str]) -> bool:
        return len(masked_segments) >= mr * len(filtered_masking_results)

    def mask_instance(self, instance_path: str, instance_name: str):
        original_img = Image.open(instance_path).convert("RGB")
        original_img_arr = np.array(original_img)
        original_img_tensor = T.ToTensor()(original_img)

        # PHASE 1: Segments Mapping
        masking_results_path = os.path.join(self.masking_root, "masking_results", f"{instance_name}_masking_results.csv")
        if os.path.exists(masking_results_path):
            masking_results = pd.read_csv(masking_results_path)
            if not set(MASKING_RESULTS_COLUMNS).issubset(masking_results.columns):
                masking_results = self._build_masking_results(instance_name, original_img_arr)
        else:
            masking_results = self._build_masking_results(instance_name, original_img_arr)

        xai_instance_dir = os.path.join(EXPERIMENTS_ROOT, self.experiment_id, "xai", self.xai_algorithm, self.xai_entry, instance_name)
        segments = np.load(os.path.join(xai_instance_dir, "segments.npy"))
        if segments.shape != tuple(original_img_tensor.shape[1:]):
            raise ValueError(f"Segments shape {segments.shape} does not match image shape {tuple(original_img_tensor.shape[1:])} for '{instance_name}'.")

        # PHASE 2: Masking Process
        img_area = original_img_tensor.shape[1] * original_img_tensor.shape[2]
        filtered_masking_results = self._sort_and_filter_masking_results(masking_results)
        cumulative_state = None
        rng = self._build_instance_rng(instance_name)

        mask_metadata = {}

        for mr in self.mask_rates:
            img_tensor, masked_segments = self._prepare_iteration(original_img_tensor, cumulative_state)
            img_tensor, masked_area, masked_segments = self._perform_masking(img_tensor, segments, mr, masking_results, filtered_masking_results, masked_segments, rng)
            cumulative_state = self._update_cumulative_state(cumulative_state, masked_segments)

            out_img = T.ToPILImage()(img_tensor)
            out_dir = os.path.join(self.masking_root, self._get_masking_entry(), f"mask_rate{mr}")
            os.makedirs(out_dir, exist_ok=True)
            out_img.save(os.path.join(out_dir, f"{instance_name}.png"))

            mask_metadata[mr] = {
                "masked_area_ratio": float(masked_area / img_area),
                "n_masked_segments": len(masked_segments),
                "masked_segments": [str(s) for s in masked_segments],
            }

        mask_metadata_dir = os.path.join(self.masking_root, "metadata", self._get_masking_entry())
        os.makedirs(mask_metadata_dir, exist_ok=True)
        mask_metadata_path = os.path.join(mask_metadata_dir, f"{instance_name}.json")

        MetadataHandler(mask_metadata_path).save_metadata(mask_metadata)

class InkBasedSaliencyImageMasker(InkBasedImageMasker):
    @property
    def mask_rule(self) -> str: return "saliency"

    def _get_segment_iteration_data(self, masking_results: pd.DataFrame, filtered_masking_results: pd.DataFrame, rng: np.random.Generator | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        segment_data = filtered_masking_results[["left_pixel", "top_pixel", "right_pixel", "bottom_pixel", "mask_r", "mask_g", "mask_b"]].to_numpy()
        segment_ids = filtered_masking_results["segment_id"].to_numpy()
        return segment_data, segment_ids, list(range(len(segment_ids)))

    def _prepare_iteration(self, img_tensor: torch.Tensor, cumulative_state) -> Tuple[torch.Tensor, List[str]]:
        masked_segments = cumulative_state if cumulative_state is not None else []
        return img_tensor, masked_segments  # No '.clone()': in-place masking accumulates across iterations

    def _update_cumulative_state(self, cumulative_state, masked_segments: List[str]) -> List[str]:
        return masked_segments  # Carry the same list forward to the next iteration

class InkBasedRandomImageMasker(InkBasedImageMasker):
    @property
    def mask_rule(self) -> str: return "random"

    def _build_instance_rng(self, instance_name: str) -> np.random.Generator:
        assert self.global_seed is not None, "Inside 'InkBasedRandomImageMasker' 'self.global_seed' must not be 'None'!"

        seed_material = f"{self.global_seed}:{instance_name}".encode("utf-8")
        instance_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], byteorder="big", signed=False)
        return np.random.default_rng(instance_seed)

    def _get_segment_iteration_data(self, masking_results: pd.DataFrame, filtered_masking_results: pd.DataFrame, rng: np.random.Generator | None = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert rng is not None, "Inside 'InkBasedRandomImageMasker' 'rng' cannot be 'None'!"

        segment_data = masking_results[["left_pixel", "top_pixel", "right_pixel", "bottom_pixel", "mask_r", "mask_g", "mask_b"]].to_numpy()
        segment_ids = masking_results["segment_id"].to_numpy()
        return segment_data, segment_ids, rng.permutation(len(segment_ids))

    def _prepare_iteration(self, img_tensor: torch.Tensor, cumulative_state) -> Tuple[torch.Tensor, List[str]]:
        return img_tensor.clone(), []

    def _update_cumulative_state(self, cumulative_state, masked_segments: List[str]) -> None:
        return None  # No state to carry forward between independent iterations
