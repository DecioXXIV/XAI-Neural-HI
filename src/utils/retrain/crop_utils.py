import os, json
import numpy as np
from typing import Dict, Tuple

def get_page_segments_and_scores(page_data_cache: dict, experiment_xai_dir: str, page_name: str) -> Tuple[np.ndarray, Dict[str, float]]:
    if page_name in page_data_cache:
        segments, scores = page_data_cache[page_name]["segments"], page_data_cache[page_name]["scores"]
    else:
        segments = np.load(os.path.join(experiment_xai_dir, page_name, "segments.npy"))
        scores_path = os.path.join(experiment_xai_dir, page_name, "aggregated_scores.json")
        with open(scores_path, "r") as f: scores = json.load(f)
        page_data_cache[page_name] = {"segments": segments, "scores": scores}

    return segments, scores