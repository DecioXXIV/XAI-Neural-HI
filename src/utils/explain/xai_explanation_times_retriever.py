import os, json
from datetime import datetime
from typing import Any, Dict, List

from src.utils.constants import EXPERIMENTS_ROOT


class XaiExplanationTimesRetriever:
    def __init__(self, experiment_id: str, xai_algorithm: str, xai_entry: str):
        self.experiment_id = experiment_id
        self.xai_algorithm = xai_algorithm
        self.xai_entry = xai_entry
        self.metadata_path = os.path.join(EXPERIMENTS_ROOT, self.experiment_id, "xai", self.xai_algorithm, self.xai_entry, "xai_instances_metadata.json")

    def __call__(self) -> Dict[str, Any]:
        with open(self.metadata_path, "r", encoding="utf-8") as f: metadata = json.load(f)

        instances = metadata.get("INSTANCES")
        if not isinstance(instances, dict):
            raise ValueError(f"Missing or invalid INSTANCES dictionary in '{self.metadata_path}'.")
        if len(instances) < 2:
            raise ValueError(f"At least two completion timestamps are required in '{self.metadata_path}'.")

        timestamps = [datetime.fromisoformat(timestamp) for timestamp in instances.values()]
        durations_seconds = [
            (current - previous).total_seconds()
            for previous, current in zip(timestamps, timestamps[1:])
        ]

        if any(duration <= 0 for duration in durations_seconds):
            raise ValueError(f"Timestamps are not strictly increasing in '{self.metadata_path}'.")

        return {
            "experiment_id": self.experiment_id,
            "xai_algorithm": self.xai_algorithm,
            "xai_entry": self.xai_entry,
            "num_explanations": len(timestamps),
            "durations_seconds": durations_seconds,
            "source_path": self.metadata_path,
        }
