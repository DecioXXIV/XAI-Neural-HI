import os, json
import numpy as np
from typing import Dict, List
from tqdm import tqdm

class CropXaiAggScoresComputer:
    def __init__(self, experiment_xai_dir: str, score_sign: int, tqdm_desc: str):
        self.experiment_xai_dir = experiment_xai_dir
        self.score_sign = score_sign
        self.tqdm_desc = tqdm_desc

    def __call__(self, coords_to_crop: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        crops_to_page_name = self._associate_crops_to_page_names(coords_to_crop)

        agg_scores_to_crop: Dict[str, float] = {}
        for page_name in tqdm(crops_to_page_name.keys(), desc=self.tqdm_desc, position=0, leave=True, dynamic_ncols=True):
            page_xai_dir = os.path.join(self.experiment_xai_dir, page_name)
            segments = np.load(os.path.join(page_xai_dir, "segments.npy"))
            with open(os.path.join(page_xai_dir, "aggregated_scores.json"), 'r') as f: scores = json.load(f)

            crop_paths = crops_to_page_name[page_name]
            for crop_path in crop_paths:
                left, top, right, bottom = coords_to_crop[crop_path]
                crop_segments = segments[top:bottom+1, left:right+1]
                crop_patches = np.unique(crop_segments)

                crop_score = 0.0
                for patch in crop_patches: crop_score += max(0.0, self.score_sign * scores[str(patch)])
                crop_score /= len(crop_patches)
                agg_scores_to_crop[crop_path] = crop_score

        return agg_scores_to_crop

    def _associate_crops_to_page_names(self, coords_to_crop: Dict[str, Dict[str, int]]) -> Dict[str, List[str]]:
        crops_to_page_name = {}
        for crop_path in coords_to_crop.keys():
            page_name = os.path.basename(crop_path).split("_")[0]
            if page_name not in crops_to_page_name: crops_to_page_name[page_name] = []
            crops_to_page_name[page_name].append(crop_path)

        return crops_to_page_name
