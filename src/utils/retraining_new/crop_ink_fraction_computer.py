import os, json
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from typing import List

class CropInkFractionComputer:
    def __init__(self, base_dir: str, experiment_xai_dir: str, crop_set: str, color: str):
        self.base_dir = base_dir
        self.experiment_xai_dir = experiment_xai_dir
        self.crop_set = crop_set  # "ft1" or "xai_guided"
        self.color = color        # "green" or "red"
        
        if self.crop_set == "ft1": coords_path = os.path.join(self.base_dir, "coords_to_ft1_train_crop.json")
        else: coords_path = os.path.join(self.base_dir, "coords_to_xai_crop.json")
        with open(coords_path, "r") as f: self.coords = json.load(f)
        
        self.page_data_cache = {}

    def __call__(self, batch_paths: List[str]) -> List[float]:
        fractions = []
        for path in batch_paths:
            left, top, right, bottom = self.coords[path]
            page_name = os.path.basename(path).split("_")[0]

            segments, scores = self._get_page_segments_and_scores(page_name)

            # Load crop as grayscale; use its actual shape to avoid off-by-one with segments
            crop_gray = np.array(Image.open(path).convert("L"))
            crop_segments = segments[top:bottom+1, left:right+1]
            crop_patches = np.unique(crop_segments)

            # Select patches matching the requested color
            if self.color == "green": colored_patches = [p for p in crop_patches if scores.get(str(p), 0.0) >= 0.0]
            else:  # "red"
                colored_patches = [p for p in crop_patches if scores.get(str(p), 0.0) < 0.0]

            if not colored_patches:
                fractions.append(0.0)
                continue

            # Otsu binarization on the individual crop (dark pixels = ink)
            try:
                threshold = threshold_otsu(crop_gray)
                ink_mask = crop_gray < threshold
            except Exception:
                fractions.append(0.0)
                continue

            # Global ink fraction: ink pixels inside colored-patch regions / total pixels in those regions
            colored_mask = np.isin(crop_segments, colored_patches)
            total_in_colored = int(np.sum(colored_mask))
            if total_in_colored == 0: fractions.append(0.0)
            else:
                ink_in_colored = int(np.sum(ink_mask & colored_mask))
                fractions.append(ink_in_colored / total_in_colored)

        return fractions
    
    def _get_page_segments_and_scores(self, page_name: str):
        if page_name in self.page_data_cache:
            segments, scores = self.page_data_cache[page_name]["segments"], self.page_data_cache[page_name]["scores"]
        else:
            segments = np.load(os.path.join(self.experiment_xai_dir, page_name, "segments.npy"))
            scores_path = os.path.join(self.experiment_xai_dir, page_name, "aggregated_scores.json")
            with open(scores_path, "r") as f: scores = json.load(f)
            self.page_data_cache[page_name] = {"segments": segments, "scores": scores}
        
        return segments, scores