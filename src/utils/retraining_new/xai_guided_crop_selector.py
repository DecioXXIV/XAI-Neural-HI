import os
import json
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple

from src.utils.logger import Logger

logger = Logger()


class XaiGuidedCropSelector:
    """Selects XAI-guided crops via a round-based greedy algorithm that enforces
    spatial diversity at the level of connected components (CCs) of red patches.

    Algorithm
    ---------
    1. Sort all candidate crops by openness score (descending).
    2. In each round, scan the pool once and pick at most one crop per
       (page, CC) pair — the first (highest-openness) encountered.
       Suppressed crops are deferred to the next round.
    3. Repeat until exactly *n_needed* crops are selected or the pool is empty.

    Connected components are computed lazily per page (4-connectivity on the
    patch grid) and cached for the lifetime of the selector.
    """

    def __init__(self, experiment_xai_dir: str, crop_to_openness: Dict[str, float]) -> None:
        self._experiment_xai_dir = experiment_xai_dir
        self._crop_to_openness   = crop_to_openness
        self._page_cc_cache: Dict[str, Dict[str, int]] = {}

    def _build_red_patch_adjacency(self, segments: np.ndarray, red_int: np.ndarray) -> Dict[int, Set[int]]:
        """Returns an adjacency dict for red patches that are 8-connected neighbors."""
        h_mask  = segments[:, :-1] != segments[:, 1:]
        h_left  = segments[:, :-1][h_mask].ravel()
        h_right = segments[:,  1:][h_mask].ravel()

        v_mask   = segments[:-1, :] != segments[1:, :]
        v_top    = segments[:-1, :][v_mask].ravel()
        v_bottom = segments[ 1:, :][v_mask].ravel()

        d1_mask   = segments[:-1, :-1] != segments[1:, 1:]
        d1_top    = segments[:-1, :-1][d1_mask].ravel()
        d1_bottom = segments[ 1:,  1:][d1_mask].ravel()

        d2_mask   = segments[:-1, 1:] != segments[1:, :-1]
        d2_top    = segments[:-1,  1:][d2_mask].ravel()
        d2_bottom = segments[ 1:, :-1][d2_mask].ravel()

        all_a = np.concatenate([h_left,  v_top,    d1_top,    d2_top])
        all_b = np.concatenate([h_right, v_bottom, d1_bottom, d2_bottom])

        both_red = np.isin(all_a, red_int) & np.isin(all_b, red_int)

        adj: Dict[int, Set[int]] = defaultdict(set)
        for a, b in zip(all_a[both_red].tolist(), all_b[both_red].tolist()):
            adj[a].add(b)
            adj[b].add(a)

        return adj

    def _label_connected_components(self, red_patches: Set[str], adj: Dict[int, Set[int]]) -> Dict[int, int]:
        """Assigns a CC id to each red patch via iterative DFS."""
        visited: Dict[int, int] = {}
        cc_id = 0

        for patch_str in red_patches:
            patch = int(patch_str)
            if patch in visited:
                continue

            stack = [patch]
            while stack:
                curr = stack.pop()
                if curr in visited:
                    continue
                visited[curr] = cc_id
                stack.extend(nb for nb in adj[curr] if nb not in visited)

            cc_id += 1

        return visited

    def _compute_page_red_patch_ccs(self, page_xai_dir: str) -> Dict[str, int]:
        """Returns ``{patch_idx_str: cc_id}`` for every red patch on the page."""
        with open(os.path.join(page_xai_dir, "aggregated_scores.json"), "r") as f:
            scores: Dict[str, float] = json.load(f)

        red_patches = {k for k, v in scores.items() if v < 0.0}
        if not red_patches:
            return {}

        segments = np.load(os.path.join(page_xai_dir, "segments.npy"))
        red_int  = np.array([int(k) for k in red_patches], dtype=segments.dtype)

        adj     = self._build_red_patch_adjacency(segments, red_int)
        visited = self._label_connected_components(red_patches, adj)

        return {str(patch): cc for patch, cc in visited.items()}

    def _parse_crop_path(self, crop_path: str) -> Tuple[str, str]:
        """Parses *crop_path* and returns ``(page_name, patch_idx_str)``."""
        basename = os.path.basename(crop_path)
        page_name, rest = basename.rsplit("_patch", 1)
        patch_idx_str   = rest.rsplit(".", 1)[0]
        return page_name, patch_idx_str

    def _cc_key(self, crop_path: str) -> str:
        """Returns a string key that uniquely identifies the (page, CC) of a crop."""
        page_name, patch_idx_str = self._parse_crop_path(crop_path)

        if page_name not in self._page_cc_cache:
            self._page_cc_cache[page_name] = self._compute_page_red_patch_ccs(
                os.path.join(self._experiment_xai_dir, page_name)
            )

        cc_id = self._page_cc_cache[page_name].get(patch_idx_str)
        if cc_id is not None:
            return f"{page_name}::cc{cc_id}"
        return f"{page_name}::solo_{patch_idx_str}"

    def _run_round(self, pool: List[str], selected: List[str], n_needed: int) -> List[str]:
        """Executes one selection round.

        Iterates *pool* in order, picking the first crop for each unseen
        (page, CC) key.  Returns the list of crops deferred to the next round.
        """
        used_keys: Set[str]  = set()
        next_pool: List[str] = []

        for crop in pool:
            if len(selected) >= n_needed:
                next_pool.append(crop)
                continue

            key = self._cc_key(crop)
            if key not in used_keys:
                selected.append(crop)
                used_keys.add(key)
            else:
                next_pool.append(crop)

        return next_pool

    def __call__(self, crops: List[str], n_needed: int) -> List[str]:
        """Selects *n_needed* crops from *crops* using the round-based CC algorithm."""
        if n_needed <= 0 or not crops:
            return []

        pool     = sorted(crops, key=lambda c: self._crop_to_openness.get(c, 0.0), reverse=True)
        selected: List[str] = []

        while len(selected) < n_needed and pool: pool = self._run_round(pool, selected, n_needed)

        return selected