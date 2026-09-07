import os, json
import numpy as np
from typing import Any, Dict, List

from src.utils.constants import EXPERIMENTS_ROOT
from src.utils.faithfulness.general_utils import compute_mask_rates
from src.utils.logger import Logger

logger = Logger()


class FaithfulnessFiguresRetriever:
    def __init__(self, experiment_id: str, mask_ceil: float, mask_step: float):
        self.experiment_id = experiment_id
        self.mask_ceil = float(mask_ceil)
        self.mask_step = float(mask_step)
        self.mask_rates = np.asarray([0.0] + compute_mask_rates(self.mask_ceil, self.mask_step), dtype=float)

    def get_faithfulness_data(self, xai_algorithm: str, xai_entry: str, patches_color: str) -> Dict[str, Any]:
        saliency_metadata_paths = self._get_faithfulness_metadata_paths(xai_algorithm, xai_entry, "saliency", patches_color)
        random_metadata_paths = self._get_faithfulness_metadata_paths(xai_algorithm, xai_entry, "random", patches_color)

        if len(saliency_metadata_paths) != 1:
            raise ValueError(f"Expected exactly one saliency faithfulness report for '{xai_algorithm}/{xai_entry}', found {len(saliency_metadata_paths)}.")

        saliency_accuracies = self._get_accuracies(saliency_metadata_paths[0])
        random_accuracies = np.vstack([self._get_accuracies(path) for path in random_metadata_paths])

        return {
            "rates": self.mask_rates,
            "saliency_accuracies": saliency_accuracies,
            "random_mean_accuracies": np.mean(random_accuracies, axis=0),
            "random_min_accuracies": np.min(random_accuracies, axis=0),
            "random_max_accuracies": np.max(random_accuracies, axis=0),
            "n_random_runs": random_accuracies.shape[0],
        }

    def get_masked_area_data(self, xai_algorithm: str, xai_entry: str, patches_color: str) -> List[np.ndarray]:
        metadata_dir = self._get_masked_metadata_dir(xai_algorithm, xai_entry, patches_color)
        metadata_paths = [
            os.path.join(metadata_dir, filename)
            for filename in sorted(os.listdir(metadata_dir))
            if filename.endswith(".json")
        ]

        if len(metadata_paths) == 0:
            raise ValueError(f"No masked area metadata files were found in '{metadata_dir}'.")

        area_by_rate = {round(float(rate), 10): [] for rate in self.mask_rates[1:]}

        for metadata_path in metadata_paths:
            metadata = self._load_metadata(metadata_path)
            rate_to_metadata = {round(float(rate), 10): values for rate, values in metadata.items()}
            self._validate_rates(rate_to_metadata, metadata_path, include_zero_rate=False)

            for rate in self.mask_rates[1:]:
                rounded_rate = round(float(rate), 10)
                rate_metadata = rate_to_metadata[rounded_rate]
                if "masked_area_ratio" not in rate_metadata:
                    raise ValueError(f"masked_area_ratio is missing for mask rate {rate} in '{metadata_path}'.")
                area_by_rate[rounded_rate].append(float(rate_metadata["masked_area_ratio"]))

        zero_rate_data = np.zeros(len(metadata_paths), dtype=float)
        return [zero_rate_data] + [np.asarray(area_by_rate[round(float(rate), 10)], dtype=float) for rate in self.mask_rates[1:]]

    def _get_faithfulness_metadata_paths(self, xai_algorithm: str, xai_entry: str, mask_rule: str, patches_color: str) -> List[str]:
        xai_faithfulness_dir = os.path.join(EXPERIMENTS_ROOT, self.experiment_id, "faithfulness", xai_algorithm, xai_entry)
        faithfulness_entry = self._get_faithfulness_entry(mask_rule, patches_color)

        if not os.path.isdir(xai_faithfulness_dir):
            raise FileNotFoundError(f"Faithfulness directory '{xai_faithfulness_dir}' was not found.")

        if mask_rule == "saliency":
            candidate_entries = [faithfulness_entry]
        else:
            candidate_entries = [
                entry for entry in sorted(os.listdir(xai_faithfulness_dir))
                if entry == faithfulness_entry or entry.startswith(f"{faithfulness_entry}-seed")
            ]

        metadata_paths = []
        for entry in candidate_entries:
            metadata_path = os.path.join(xai_faithfulness_dir, entry, "faithfulness_crop_level.json")
            if os.path.isfile(metadata_path): metadata_paths.append(metadata_path)

        if len(metadata_paths) == 0:
            raise FileNotFoundError(f"No crop-level faithfulness report was found for '{xai_algorithm}/{xai_entry}/{faithfulness_entry}'.")

        return metadata_paths

    def _get_masked_metadata_dir(self, xai_algorithm: str, xai_entry: str, patches_color: str) -> str:
        saliency_entry = self._get_faithfulness_entry("saliency", patches_color)
        metadata_dir = os.path.join(EXPERIMENTS_ROOT, self.experiment_id, "masked_images", xai_algorithm, xai_entry, "metadata", saliency_entry)

        if not os.path.isdir(metadata_dir):
            raise FileNotFoundError(f"Masked area metadata directory '{metadata_dir}' was not found.")

        return metadata_dir

    def _get_faithfulness_entry(self, mask_rule: str, patches_color: str) -> str:
        ceil_tag, step_tag = self._get_mask_rate_tag(self.mask_ceil), self._get_mask_rate_tag(self.mask_step)
        return f"{mask_rule}-ceil{ceil_tag}-step{step_tag}-{patches_color}"

    def _get_mask_rate_tag(self, rate: float) -> str:
        return str(float(rate))

    def _get_accuracies(self, metadata_path: str) -> np.ndarray:
        metadata = self._load_metadata(metadata_path)
        rate_to_accuracy = {}

        for rate, values in metadata.items():
            rounded_rate = round(float(rate), 10)
            if "accuracy" not in values:
                raise ValueError(f"accuracy is missing for mask rate {rate} in '{metadata_path}'.")
            rate_to_accuracy[rounded_rate] = float(values["accuracy"])

        self._validate_rates(rate_to_accuracy, metadata_path, include_zero_rate=True)
        return np.asarray([rate_to_accuracy[round(float(rate), 10)] for rate in self.mask_rates], dtype=float)

    def _validate_rates(self, rate_to_data: Dict[float, Any], source_path: str, include_zero_rate: bool):
        expected_rates = self.mask_rates if include_zero_rate else self.mask_rates[1:]
        missing_rates = [rate for rate in expected_rates if round(float(rate), 10) not in rate_to_data]

        if len(missing_rates) > 0:
            raise ValueError(f"Mask rates {missing_rates} are missing in '{source_path}'.")

    def _load_metadata(self, metadata_path: str) -> Dict[str, Any]:
        try:
            with open(metadata_path, "r", encoding="utf-8") as f: return json.load(f)
        except Exception as e:
            logger.exception(f"Failed to load metadata from '{metadata_path}': {e}")
            raise
