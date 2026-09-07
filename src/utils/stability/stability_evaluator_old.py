import os, json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable
from itertools import combinations
from tqdm import tqdm
from scipy.stats import pearsonr

from src.utils.constants import EXPERIMENTS_ROOT

class StabilityEvaluator:
    def __init__(self, experiment_id: str, xai_algorithm: str, root_entry: str, xai_entries: List[str]):
        self.experiment_id = experiment_id
        self.xai_algorithm = xai_algorithm
        self.root_entry = root_entry
        self.xai_entries = xai_entries
        
        self._get_xai_entry_pairs()
        
        self.out_dir = os.path.join(EXPERIMENTS_ROOT, self.experiment_id, "stability", self.xai_algorithm, self.root_entry)
        os.makedirs(self.out_dir, exist_ok=True)
    
    def _get_xai_entry_pairs(self):
        self.entry_to_idx = {self.xai_entries[i]: i for i in range(0, len(self.xai_entries))}
        self.xai_entry_pairs = list(combinations(self.xai_entries, 2))
        
        self.xai_entry_idx_pairs = []
        for entry_pair in self.xai_entry_pairs:
            entry1, entry2 = entry_pair
            idx1, idx2 = self.entry_to_idx[entry1], self.entry_to_idx[entry2]
            self.xai_entry_idx_pairs.append((idx1, idx2))
    
    def _align_score_arrays(self, scores1: Dict[str, float], scores2: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        keys = sorted(scores1.keys())
        arr1 = np.fromiter((scores1[k] for k in keys), dtype=float, count=len(keys))
        arr2 = np.fromiter((scores2[k] for k in keys), dtype=float, count=len(keys))
        return arr1, arr2
    
    def _process_instance(self, instance: str, entry1: str, entry2: str) -> float:
        path1 = os.path.join(EXPERIMENTS_ROOT, self.experiment_id, "xai", self.xai_algorithm, entry1, instance, "aggregated_scores.json")
        path2 = os.path.join(EXPERIMENTS_ROOT, self.experiment_id, "xai", self.xai_algorithm, entry2, instance, "aggregated_scores.json")
        
        with open(path1, 'r') as f: scores1 = json.load(f)
        with open(path2, 'r') as f: scores2 = json.load(f)
        
        arr1, arr2 = self._align_score_arrays(scores1, scores2)
        return self._safe_correlation(arr1, arr2, pearsonr)
    
    def _safe_correlation(self, x: np.ndarray, y: np.ndarray, corr_func: Callable) -> float:
        if len(x) < 2 or len(y) < 2: return np.nan
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        
        if np.allclose(x, x[0]) or np.allclose(y, y[0]): return np.nan
        
        try:
            result = corr_func(x, y)
            stat = result.statistic if hasattr(result, 'statistic') else result[0]
            return stat if np.isfinite(stat) else np.nan
        except Exception: return np.nan
    
    def _compute_pearson_comparisons(self, instances: List[str]) -> pd.DataFrame:
        columns = ["instance"] + [f"{idx1}_vs_{idx2}" for idx1, idx2 in self.xai_entry_idx_pairs]
        raw_df = pd.DataFrame(columns=columns)
        raw_df["instance"] = instances
        
        for entry_pair in tqdm(self.xai_entry_pairs, desc="XAI Entry Pairs", position=0, dynamic_ncols=True):
            entry1, entry2 = entry_pair
            column_name = f"{self.entry_to_idx[entry1]}_vs_{self.entry_to_idx[entry2]}"
            raw_df[column_name] = [self._process_instance(instance, entry1, entry2) for instance in tqdm(instances, desc="Instances", position=1, leave=False, dynamic_ncols=True)]
    
        raw_df.to_csv(os.path.join(self.out_dir, "pearson_correlations.csv"), index=False, header=True)
        return raw_df
    
    def _apply_fisher_z(self, raw_df: pd.DataFrame, value_cols: List[str]) -> pd.DataFrame:
        z_df = raw_df.copy()
        z_df[value_cols] = np.arctanh(z_df[value_cols].to_numpy(dtype=float, copy=False)) # raw_df.values in (-1, 1)
        return z_df
        
    def _compute_confidence_intervals(self, raw_df: pd.DataFrame):
        value_cols = [col for col in raw_df.columns if col != "instance"]
        z_df = self._apply_fisher_z(raw_df, value_cols)
        
        mean_z_df = pd.DataFrame(columns=["instance", "mean_z"])
        mean_z_df["instance"] = z_df["instance"]
        mean_z_df["mean_z"] = z_df[value_cols].mean(axis=1)
        
        mean_df = pd.DataFrame(columns=["instance", "mean_pearson"])
        mean_df["instance"] = mean_z_df["instance"]
        mean_df["mean_pearson"] = np.tanh(mean_z_df["mean_z"].to_numpy(dtype=float, copy=False))
        
        quantiles = {"0.95": 1.96, "0.99": 2.576}
        
        mean, std = np.mean(mean_df["mean_pearson"].to_numpy(), dtype=float), np.std(mean_df["mean_pearson"].to_numpy(), dtype=float, ddof=1)
        
        confidence_intervals = {
            "mean": mean,
            "std": std,
            "95%": f"{mean} ± {quantiles['0.95'] * std / np.sqrt(len(mean_df))}",
            "99%": f"{mean} ± {quantiles['0.99'] * std / np.sqrt(len(mean_df))}"
        }
        
        with open(os.path.join(self.out_dir, "confidence_intervals.json"), 'w', encoding="utf-8") as f:
            json.dump(confidence_intervals, f, ensure_ascii=False, indent=4)
        
    def __call__(self):
        root_xai_metadata_path = os.path.join(EXPERIMENTS_ROOT, self.experiment_id, "xai", self.xai_algorithm, self.xai_entries[0], "xai_instances_metadata.json")
        with open(root_xai_metadata_path, 'r') as f: xai_instances_metadata = json.load(f)
        instances = list(xai_instances_metadata["INSTANCES"].keys())
        
        raw_df = self._compute_pearson_comparisons(instances)
        self._compute_confidence_intervals(raw_df)   