import os, csv
import numpy as np
from typing import Any, Dict

from src.utils.constants import EXPERIMENTS_ROOT
from src.utils.explain.xai_explanation_times_retriever import XaiExplanationTimesRetriever
from src.utils.logger import Logger

logger = Logger()


class XaiExplanationTimesEvaluator:
    def __init__(self, experiment_id: str, xai_algorithm: str, xai_entry: str):
        self.experiment_id = experiment_id
        self.xai_algorithm = xai_algorithm
        self.xai_entry = xai_entry
        self.output_dir = os.path.join(EXPERIMENTS_ROOT, self.experiment_id, "output", "xai_times", self.xai_algorithm, self.xai_entry)

    def __call__(self):
        logger.info(f"*** BEGINNING OF XAI EXPLANATION TIMES COMPUTATION -> Experiment: {self.experiment_id} | XAI: {self.xai_algorithm}/{self.xai_entry} ***")
        result = XaiExplanationTimesRetriever(self.experiment_id, self.xai_algorithm, self.xai_entry)()
        output_path = self._write_result(result)
        logger.info(f"*** XAI EXPLANATION TIMES COMPUTED SUCCESSFULLY -> Output: {output_path} ***")

    def _write_result(self, result: Dict[str, Any]) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, "xai_explanation_times.csv")
        fieldnames = ["experiment_id", "xai_algorithm", "xai_entry", "num_explanations", "num_measured_durations", "mean_seconds", "median_seconds", "std_seconds", "min_seconds", "max_seconds", "source_path"]

        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            durations_seconds = np.asarray(result["durations_seconds"], dtype=float)
            writer.writerow({
                "experiment_id": result["experiment_id"],
                "xai_algorithm": result["xai_algorithm"],
                "xai_entry": result["xai_entry"],
                "num_explanations": result["num_explanations"],
                "num_measured_durations": len(durations_seconds),
                "mean_seconds": f"{np.mean(durations_seconds):.6f}",
                "median_seconds": f"{np.median(durations_seconds):.6f}",
                "std_seconds": f"{np.std(durations_seconds):.6f}",
                "min_seconds": f"{np.min(durations_seconds):.6f}",
                "max_seconds": f"{np.max(durations_seconds):.6f}",
                "source_path": result["source_path"],
            })

        return output_path
