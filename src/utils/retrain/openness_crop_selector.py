import os, json
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from src.utils.logger import Logger

logger = Logger()


class OpennessCropSelector:
    """Selects XAI-guided crops in two phases.

    Phase 1 (base): for each page, extract ``n_needed // n_pages`` crops
        using round-robin over connected components (CCs) of green patches.
    Phase 2 (remainder): fill ``n_needed % n_pages`` remaining slots by
        scanning all crops globally (openness desc), taking at most one
        extra crop per page.

    Connected components are computed lazily per page (8-connectivity on the
    patch grid) and cached for the lifetime of the selector.
    """

    def __init__(self, openness_scores_df: pd.DataFrame, experiment_xai_dir: str) -> None:
        self.openness_df        = openness_scores_df.sort_values(by="openness", ascending=False)
        self.openness_df["instance_class"] = self.openness_df["instance_class"].astype(str)
        self.crop_to_openness   = openness_scores_df.set_index("crop")["openness"].to_dict()
        self.experiment_xai_dir = experiment_xai_dir
        self.page_cc_cache: Dict[str, Dict[str, int]] = {}
    
    def __call__(self, cls: str, n_crops: int) -> List[str]:
        self.cls_openness_scores_df = self.openness_df[self.openness_df["instance_class"] == cls]
        sorted_crops_to_page = {
            page: crops.sort_values(by="openness", ascending=False)["crop"].tolist()
            for page, crops in self.cls_openness_scores_df.groupby("page")
        }

        n_pages = len(sorted_crops_to_page)
        base, remainder = divmod(n_crops, n_pages)

        selected_crops, extracted_set = self._execute_first_selection_phase(sorted_crops_to_page, base)
        return self._execute_second_selection_phase(selected_crops, extracted_set, remainder)

    def _execute_first_selection_phase(self, page_to_crops: Dict[str, List[str]], base: int) -> Tuple[List[str], Set[str]]:
        """For each page, select `base` crops via round-robin over connected components."""
        if base <= 0:
            return [], set()

        selected_crops: List[str] = []
        for crops in page_to_crops.values():
            page_selected = self._extract_base_crops_from_page(crops, base)
            selected_crops.extend(page_selected)

        return selected_crops, set(selected_crops)

    def _extract_base_crops_from_page(self, crops: List[str], base: int) -> List[str]:
        """
        Selects `base` crops from a single page using round-robin over connected components.

        Steps:
          1. Group crops by connected component (CC). Since `crops` is already sorted
             by openness descending, each CC list is also in openness desc order.
          2. Sort the CCs by the openness score of their best crop (descending),
             so that higher-quality CCs are visited first in each round.
          3. Round-robin: in each round, pick the next best crop from each CC
             (using `round_idx` to track how many crops have already been taken
             from each CC). Stop when `base` crops are selected or all CCs
             are exhausted.
        """
        # Step 1: group crops by CC
        crops_to_cc: Dict[str, List[str]] = defaultdict(list)
        for crop in crops:
            cc_key = self._get_cc_key(crop)
            crops_to_cc[cc_key].append(crop)

        # Step 2: sort CCs by the openness of their best (first) crop
        sorted_cc_keys = sorted(
            crops_to_cc.keys(),
            key=lambda k: self.crop_to_openness.get(crops_to_cc[k][0], 0.0),
            reverse=True
        )

        # Step 3: round-robin across CCs
        selected: List[str] = []
        round_idx = 0  # how many crops we have already taken from each CC
        while len(selected) < base:
            round_made_progress = False
            for cc_key in sorted_cc_keys:
                if len(selected) >= base:
                    break
                cc_crops = crops_to_cc[cc_key]
                if round_idx < len(cc_crops):         # this CC still has crops to offer
                    selected.append(cc_crops[round_idx])
                    round_made_progress = True
            if not round_made_progress:
                break  # all CCs are exhausted, cannot reach `base`
            round_idx += 1

        return selected

    def _execute_second_selection_phase(self, selected_crops: List[str], extracted_set: Set[str], remainder: int) -> List[str]:
        if remainder == 0: return selected_crops

        pages_with_extra = set()
        for _, row in self.cls_openness_scores_df.iterrows():
            if len(pages_with_extra) >= remainder: break
            
            crop, page = row["crop"], row["page"]
            if crop in extracted_set or page in pages_with_extra: continue
            
            selected_crops.append(crop)
            pages_with_extra.add(page)

        return selected_crops

    def _build_patch_adjacency(self, segments: np.ndarray, target_int: np.ndarray) -> Dict[int, Set[int]]:
        """
        Builds {patch_id: set_of_adjacent_target_patch_ids} for all target patches.

        Two patches are adjacent if they are different, both target patches, and share
        at least one pair of 8-connected pixels (horizontal, vertical, diagonal).

        Vectorized approach: at each pixel boundary between two different patches,
        record the (patch_a, patch_b) pair. Then keep only pairs where both are target patches.
        """
        # Horizontal boundaries: pixels (r,c) and (r,c+1) in different patches
        h_mask  = segments[:, :-1] != segments[:, 1:]
        h_left  = segments[:, :-1][h_mask].ravel()
        h_right = segments[:,  1:][h_mask].ravel()

        # Vertical boundaries: pixels (r,c) and (r+1,c) in different patches
        v_mask   = segments[:-1, :] != segments[1:, :]
        v_top    = segments[:-1, :][v_mask].ravel()
        v_bottom = segments[ 1:, :][v_mask].ravel()

        # Diagonal boundaries top-left → bottom-right
        d1_mask   = segments[:-1, :-1] != segments[1:, 1:]
        d1_top    = segments[:-1, :-1][d1_mask].ravel()
        d1_bottom = segments[ 1:,  1:][d1_mask].ravel()

        # Diagonal boundaries top-right → bottom-left
        d2_mask   = segments[:-1, 1:] != segments[1:, :-1]
        d2_top    = segments[:-1,  1:][d2_mask].ravel()
        d2_bottom = segments[ 1:, :-1][d2_mask].ravel()

        # Merge all boundary pairs, then keep only those where both patches are red
        all_a = np.concatenate([h_left,  v_top,    d1_top,    d2_top])
        all_b = np.concatenate([h_right, v_bottom, d1_bottom, d2_bottom])
        both_target = np.isin(all_a, target_int) & np.isin(all_b, target_int)

        adj: Dict[int, Set[int]] = defaultdict(set)
        for a, b in zip(all_a[both_target].tolist(), all_b[both_target].tolist()):
            adj[a].add(b)
            adj[b].add(a)

        return adj

    def _label_connected_components(self, target_patches: Set[str], adjacency: Dict[int, Set[int]]) -> Dict[int, int]:
        """
        Assigns a CC id to each target patch using iterative DFS.
        Returns {patch_id: cc_id}.
        """
        patch_to_cc: Dict[int, int] = {}
        cc_id = 0

        for patch_str in target_patches:
            patch = int(patch_str)
            if patch in patch_to_cc:
                continue  # already assigned to a CC in a previous DFS

            # DFS: visit all patches reachable from this one → same CC
            stack = [patch]
            while stack:
                current = stack.pop()
                if current in patch_to_cc:
                    continue
                patch_to_cc[current] = cc_id
                for neighbor in adjacency[current]:
                    if neighbor not in patch_to_cc:
                        stack.append(neighbor)

            cc_id += 1

        return patch_to_cc

    def _compute_page_cc_map(self, page_xai_dir: str) -> Dict[str, int]:
        """
        Returns {patch_idx_str: cc_id} for all red patches on the page.
        Red patches are those with aggregated_score < 0.
        """
        with open(os.path.join(page_xai_dir, "aggregated_scores.json"), "r") as f:
            scores: Dict[str, float] = json.load(f)

        green_patches = {k for k, v in scores.items() if v > 0.0}
        if not green_patches:
            return {}

        segments    = np.load(os.path.join(page_xai_dir, "segments.npy"))
        green_int   = np.array([int(k) for k in green_patches], dtype=segments.dtype)
        adjacency   = self._build_patch_adjacency(segments, green_int)
        cc_map      = self._label_connected_components(green_patches, adjacency)

        return {str(patch_id): cc_id for patch_id, cc_id in cc_map.items()}

    def _get_cc_key(self, crop_path: str) -> str:
        """
        Returns a unique string identifying the (page, CC) a crop belongs to.
        Crop filenames follow the pattern "{page_name}_patch{patch_idx}.png".
        """
        basename        = os.path.basename(crop_path)
        page_name, rest = basename.rsplit("_patch", 1)
        patch_idx_str   = rest.rsplit(".", 1)[0]

        # Compute and cache the CC map for this page on first access
        if page_name not in self.page_cc_cache:
            page_xai_dir = os.path.join(self.experiment_xai_dir, page_name)
            self.page_cc_cache[page_name] = self._compute_page_cc_map(page_xai_dir)

        cc_id = self.page_cc_cache[page_name].get(patch_idx_str)
        if cc_id is not None: return f"{page_name}::cc{cc_id}"
        # Patch not found among green patches: treat it as its own isolated CC
        return f"{page_name}::solo_{patch_idx_str}"