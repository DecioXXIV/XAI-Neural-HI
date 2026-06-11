import os, json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from itertools import combinations
from tqdm import tqdm

from src.utils.constants import EXPERIMENTS_ROOT

FISHER_CLIP = 1.0 - 1e-12

class ExpComparisonEvaluator:
    def __init__(self, xai_algorithm: str, out_dir: str, score_sources: List[Dict[str, str]], key_policy: str):
        self.xai_algorithm = xai_algorithm
        self.out_dir = out_dir
        self.score_sources = score_sources
        self.key_policy = key_policy

        os.makedirs(self.out_dir, exist_ok=True)

        self.cache_dir = os.path.join(self.out_dir, "score_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.scores_cache: Dict[Tuple[str, str, str], Tuple[Tuple[str, ...], np.ndarray]] = {}

    def _safe_path_component(self, value: str) -> str:
        return value.replace(os.sep, "__").replace("/", "__")

    def _cached_npz_path(self, source: Dict[str, str], instance: str) -> str:
        safe_instance = self._safe_path_component(instance)
        return os.path.join(self.cache_dir, source["cache_id"], f"{safe_instance}.npz")

    def _cache_is_current(self, cache_path: str, json_path: str) -> bool:
        if not os.path.exists(cache_path): return False
        if not os.path.exists(json_path): return True
        return os.path.getmtime(cache_path) >= os.path.getmtime(json_path)

    def _load_npz_scores(self, path: str):
        if not os.path.exists(path): return None

        with np.load(path, allow_pickle=False) as data:
            if "keys" in data and "values" in data:
                keys = tuple(str(k) for k in data["keys"].tolist())
                values = np.asarray(data["values"], dtype=float).copy()
                return keys, values
            if "scores" in data:
                values = np.asarray(data["scores"], dtype=float).copy()
                keys = tuple(str(i) for i in range(len(values)))
                return keys, values

        return None

    def _save_npz_scores(self, path: str, keys: Tuple[str, ...], values: np.ndarray):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, keys=np.asarray(keys, dtype=str), values=np.asarray(values, dtype=float))

    def _load_json_scores(self, path: str) -> Tuple[Tuple[str, ...], np.ndarray]:
        with open(path, "r", encoding="utf-8") as f: scores = json.load(f)

        keys = tuple(sorted(str(key) for key in scores.keys()))
        values = np.fromiter((scores[k] for k in keys), dtype=float, count=len(keys))
        return keys, values

    def _score_json_path(self, source: Dict[str, str], instance: str) -> str:
        return os.path.join(EXPERIMENTS_ROOT, source["experiment_id"], "xai", self.xai_algorithm, source["xai_entry"], instance, "aggregated_scores.json")

    def _metadata_path(self, source: Dict[str, str]) -> str:
        return os.path.join(EXPERIMENTS_ROOT, source["experiment_id"], "xai", self.xai_algorithm, source["xai_entry"], "xai_instances_metadata.json")

    def _load_scores(self, source: Dict[str, str], instance: str) -> Tuple[Tuple[str, ...], np.ndarray]:
        cache_key = (source["experiment_id"], source["xai_entry"], instance)
        if cache_key in self.scores_cache:
            return self.scores_cache[cache_key]

        json_path = self._score_json_path(source, instance)
        cached_npz_path = self._cached_npz_path(source, instance)
        loaded = None

        if loaded is None and self._cache_is_current(cached_npz_path, json_path):
            loaded = self._load_npz_scores(cached_npz_path)

        if loaded is None:
            loaded = self._load_json_scores(json_path)
            self._save_npz_scores(cached_npz_path, loaded[0], loaded[1])

        self.scores_cache[cache_key] = loaded
        return loaded

    def _get_common_instances(self) -> List[str]:
        instance_sets = []

        for source in self.score_sources:
            metadata_path = self._metadata_path(source)
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            instance_sets.append(set(metadata["INSTANCES"].keys()))

        instances = set.intersection(*instance_sets)
        return sorted(instances)

    def _align_instance_scores(self, instance: str) -> np.ndarray:
        loaded_scores = [self._load_scores(source, instance) for source in self.score_sources]

        if self.key_policy == "strict":
            score_keys = loaded_scores[0][0]
            for source, (keys, _) in zip(self.score_sources, loaded_scores):
                if keys != score_keys:
                    raise ValueError(f"Score keys mismatch for instance '{instance}' in entry '{source['xai_entry']}'")

            matrix = np.vstack([values for _, values in loaded_scores])

        elif self.key_policy == "intersection":
            common_keys = set(loaded_scores[0][0])
            for keys, _ in loaded_scores[1:]:
                common_keys &= set(keys)
            if len(common_keys) == 0: raise ValueError(f"No common score keys found for instance '{instance}'.")

            common_keys = tuple(sorted(common_keys))
            score_dicts = [dict(zip(keys, values)) for keys, values in loaded_scores]
            matrix = np.vstack([
                np.fromiter((scores[k] for k in common_keys), dtype=float, count=len(common_keys))
                for scores in score_dicts
            ])

        else:
            raise ValueError(f"Unknown key alignment policy '{self.key_policy}'.")

        finite_cols = np.all(np.isfinite(matrix), axis=0)
        return matrix[:, finite_cols]


class StabilityEvaluator(ExpComparisonEvaluator):
    def __init__(self, experiment_id: str, xai_algorithm: str, root_entry: str, xai_entries: List[str]):
        self.experiment_id = experiment_id
        self.root_entry = root_entry
        self.xai_entries = xai_entries

        if len(self.xai_entries) < 2: raise ValueError("Stability requires at least two XAI entries.")

        self._get_xai_entry_pairs()

        out_dir = os.path.join(EXPERIMENTS_ROOT, self.experiment_id, "stability", xai_algorithm, self.root_entry)
        score_sources = [
            {"experiment_id": self.experiment_id, "xai_entry": entry, "cache_id": str(self.entry_to_idx[entry])}
            for entry in self.xai_entries
        ]

        super().__init__(xai_algorithm, out_dir, score_sources, key_policy="strict")

    def _get_xai_entry_pairs(self):
        self.entry_to_idx = {self.xai_entries[i]: i for i in range(0, len(self.xai_entries))}
        self.xai_entry_pairs = list(combinations(self.xai_entries, 2))
        self.xai_entry_idx_pairs = [
            (self.entry_to_idx[e1], self.entry_to_idx[e2])
            for e1, e2 in self.xai_entry_pairs
        ]
        self.columns = [f"{idx1}_vs_{idx2}" for idx1, idx2 in self.xai_entry_idx_pairs]

    def _corrcoef(self, matrix: np.ndarray) -> np.ndarray:
        n_entries = len(self.xai_entries)
        correlations = np.full((n_entries, n_entries), np.nan, dtype=float)

        if matrix.shape[1] < 2: return correlations

        valid_rows = np.asarray([not np.allclose(row, row[0]) for row in matrix], dtype=bool)
        valid_indices = np.where(valid_rows)[0]

        if len(valid_indices) < 2: return correlations

        with np.errstate(divide="ignore", invalid="ignore"): valid_correlations = np.corrcoef(matrix[valid_indices])

        correlations[np.ix_(valid_indices, valid_indices)] = valid_correlations
        return correlations

    def _process_instance(self, instance: str) -> Dict[str, object]:
        matrix = self._align_instance_scores(instance)
        correlations = self._corrcoef(matrix)

        row = {"instance": instance}
        for column, (idx1, idx2) in zip(self.columns, self.xai_entry_idx_pairs):
            corr = correlations[idx1, idx2]
            row[column] = float(corr) if np.isfinite(corr) else np.nan
        return row

    def _compute_pearson_comparisons(self, instances: List[str]) -> pd.DataFrame:
        rows = [self._process_instance(instance) for instance in tqdm(instances, desc="Instances", dynamic_ncols=True)]
        raw_df = pd.DataFrame(rows, columns=["instance"] + self.columns)
        raw_df.to_csv(os.path.join(self.out_dir, "pearson_correlations.csv"), index=False, header=True)
        return raw_df

    def _compute_confidence_intervals(self, raw_df: pd.DataFrame) -> Dict[str, object]:
        values = raw_df[self.columns].to_numpy(dtype=float, copy=False)
        values = np.clip(values, -FISHER_CLIP, FISHER_CLIP)
        z_values = np.arctanh(values)

        valid_counts = np.count_nonzero(np.isfinite(z_values), axis=1)
        mean_z = np.full(len(raw_df), np.nan, dtype=float)
        valid_rows = valid_counts > 0
        mean_z[valid_rows] = np.nansum(z_values[valid_rows], axis=1) / valid_counts[valid_rows]

        mean_pearson = np.tanh(mean_z)
        mean_pearson = mean_pearson[np.isfinite(mean_pearson)]

        if len(mean_pearson) == 0:
            mean, std, se = np.nan, np.nan, np.nan
        else:
            mean = float(np.mean(mean_pearson, dtype=float))
            std = (
                float(np.std(mean_pearson, dtype=float, ddof=1))
                if len(mean_pearson) > 1
                else 0.0
            )
            se = std / np.sqrt(len(mean_pearson))

        confidence_intervals = {
            "mean": mean, "std": std,
            "95%": f"{mean} +/- {1.976 * se}", "99%": f"{mean} +/- {2.576 * se}"
        }

        out_path = os.path.join(self.out_dir, "confidence_intervals.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(confidence_intervals, f, ensure_ascii=False, indent=4)

        return confidence_intervals

    def _save_entry_mapping(self):
        mapping_df = pd.DataFrame({
            "entry_idx": list(range(len(self.xai_entries))),
            "xai_entry": self.xai_entries,
        })
        mapping_df.to_csv(os.path.join(self.out_dir, "xai_entry_index.csv"), index=False)

    def __call__(self):
        self._save_entry_mapping()

        instances = self._get_common_instances()
        if len(instances) == 0: raise ValueError("No common explained instances found for the selected XAI entries.")

        raw_df = self._compute_pearson_comparisons(instances)
        confidence_intervals = self._compute_confidence_intervals(raw_df)
        return raw_df, confidence_intervals


class CrossModelAgreementEvaluator(ExpComparisonEvaluator):
    def __init__(self, exp_id1: str, exp_id2: str, xai_algorithm: str, xai_entry1: str, xai_entry2: str, n_bootstrap: int, random_seed: int):
        self.exp_id1 = exp_id1
        self.exp_id2 = exp_id2
        self.xai_entry1 = xai_entry1
        self.xai_entry2 = xai_entry2
        self.n_bootstrap = n_bootstrap
        self.random_seed = random_seed
        self.thresholds = (0.05, 0.10, 0.25)
        self.rules = ("top", "bottom")

        self.columns = [f"{rule}_{int(threshold * 100)}%" for rule in self.rules for threshold in self.thresholds] + ["overall"]

        out_dir = os.path.join(EXPERIMENTS_ROOT, self.exp_id1, "cross_model_agreement", xai_algorithm, self._comparison_dir_name(self.xai_entry1, self.exp_id2, self.xai_entry2))
        self.mirror_out_dir = os.path.join(EXPERIMENTS_ROOT, self.exp_id2, "cross_model_agreement", xai_algorithm, self._comparison_dir_name(self.xai_entry2, self.exp_id1, self.xai_entry1))
        os.makedirs(self.mirror_out_dir, exist_ok=True)

        score_sources = [
            {"experiment_id": self.exp_id1, "xai_entry": self.xai_entry1, "cache_id": "exp1"},
            {"experiment_id": self.exp_id2, "xai_entry": self.xai_entry2, "cache_id": "exp2"},
        ]

        super().__init__(xai_algorithm, out_dir, score_sources, key_policy="intersection")

    def _comparison_dir_name(self, xai_entry: str, exp_id: str, other_xai_entry: str) -> str:
        return f"{xai_entry}_vs_{exp_id}__{other_xai_entry}".replace(os.sep, "__").replace("/", "__")

    def _rank_average(self, values: np.ndarray) -> np.ndarray:
        return pd.Series(values, dtype="float64").rank(method="average").to_numpy(dtype=float)

    def _safe_spearman(self, scores1: np.ndarray, scores2: np.ndarray) -> float:
        if len(scores1) < 2 or len(scores2) < 2: return np.nan

        ranks1, ranks2 = self._rank_average(scores1), self._rank_average(scores2)
        if np.allclose(ranks1, ranks1[0]) or np.allclose(ranks2, ranks2[0]): return np.nan

        with np.errstate(divide="ignore", invalid="ignore"): corr = np.corrcoef(ranks1, ranks2)[0, 1]
        return float(corr) if np.isfinite(corr) else np.nan

    def _get_spearman_comparisons(self, matrix: np.ndarray) -> Dict[str, float]:
        comparisons = {column: np.nan for column in self.columns}
        if matrix.shape[1] < 2: return comparisons

        scores1, scores2 = matrix[0], matrix[1]
        bottom_idx1, bottom_idx2 = np.argsort(scores1), np.argsort(scores2)
        ordered_indices1 = {"bottom": bottom_idx1, "top": bottom_idx1[::-1]}
        ordered_indices2 = {"bottom": bottom_idx2, "top": bottom_idx2[::-1]}
        n_select = {threshold: max(1, int(threshold * matrix.shape[1])) for threshold in self.thresholds}

        for rule in self.rules:
            idx_order1, idx_order2 = ordered_indices1[rule], ordered_indices2[rule]
            for threshold in self.thresholds:
                column = f"{rule}_{int(threshold * 100)}%"
                n = n_select[threshold]
                selected_idx = np.unique(np.concatenate((idx_order1[:n], idx_order2[:n])))
                comparisons[column] = self._safe_spearman(scores1[selected_idx], scores2[selected_idx])

        comparisons["overall"] = self._safe_spearman(scores1, scores2)
        return comparisons

    def _process_instance(self, instance: str) -> Dict[str, object]:
        matrix = self._align_instance_scores(instance)
        comparisons = self._get_spearman_comparisons(matrix)

        row = {"instance": instance}
        for column in self.columns:
            corr = comparisons[column]
            row[column] = float(corr) if np.isfinite(corr) else np.nan
        return row

    def _compute_spearman_comparisons(self, instances: List[str]) -> pd.DataFrame:
        rows = [self._process_instance(instance) for instance in tqdm(instances, desc="Instances", dynamic_ncols=True)]
        raw_df = pd.DataFrame(rows, columns=["instance"] + self.columns)
        raw_df.to_csv(os.path.join(self.out_dir, "spearman_correlations.csv"), index=False, header=True)
        raw_df.to_csv(os.path.join(self.mirror_out_dir, "spearman_correlations.csv"), index=False, header=True)
        return raw_df

    def _compute_confidence_intervals(self, raw_df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
        rng = np.random.default_rng(seed=self.random_seed)
        confidence_intervals = {}

        for column in self.columns:
            values = raw_df[column].to_numpy(dtype=float, copy=False)
            values = values[np.isfinite(values)]

            if len(values) == 0:
                mean, std = np.nan, np.nan
            else:
                mean = float(np.mean(values, dtype=float))
                bootstrap_idx = rng.integers(0, len(values), size=(self.n_bootstrap, len(values)))
                bootstrap_means = np.mean(values[bootstrap_idx], axis=1, dtype=float)
                std = float(np.std(bootstrap_means, dtype=float, ddof=1))

            confidence_intervals[column] = {
                "mean": mean, "std": std,
                "95%": f"{mean} +/- {1.96 * std}", "99%": f"{mean} +/- {2.576 * std}"
            }

        out_path = os.path.join(self.out_dir, "confidence_intervals.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(confidence_intervals, f, ensure_ascii=False, indent=4)

        mirror_out_path = os.path.join(self.mirror_out_dir, "confidence_intervals.json")
        with open(mirror_out_path, "w", encoding="utf-8") as f:
            json.dump(confidence_intervals, f, ensure_ascii=False, indent=4)

        return confidence_intervals

    def __call__(self):
        instances = self._get_common_instances()
        if len(instances) == 0: raise ValueError("No common explained instances found for the selected cross-model comparison.")

        raw_df = self._compute_spearman_comparisons(instances)
        confidence_intervals = self._compute_confidence_intervals(raw_df)
        return raw_df, confidence_intervals
