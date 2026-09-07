from typing import Any, Dict, List

from src import EXPLAINERS
from src.utils.logger import Logger

logger = Logger()


def validate_xai_explanation_times_configs(times_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(times_configs) == 0:
        logger.warning("No XAI explanation times configuration has been provided.")
        return []

    valid_configs = []
    for config in times_configs:
        if _validate_xai_explanation_times_config(config): valid_configs.append(config)
        else: logger.warning("Skipping invalid XAI explanation times configuration.")

    return valid_configs


def _validate_xai_explanation_times_config(config: Dict[str, Any]) -> bool:
    if not isinstance(config, dict):
        logger.critical("Each XAI explanation times configuration must be a dictionary.")
        return False

    experiment_ids = config.get("experiment_ids")
    xai_pairs = config.get("xai_pairs")
    error_trigger = False

    if not isinstance(experiment_ids, list) or len(experiment_ids) == 0 or any(not isinstance(experiment_id, str) or len(experiment_id) == 0 for experiment_id in experiment_ids):
        logger.critical("experiment_ids must be a non-empty list of experiment identifiers.")
        error_trigger = True

    if not isinstance(xai_pairs, list) or len(xai_pairs) == 0:
        logger.critical("xai_pairs must be a non-empty list of (xai_algorithm, xai_entry) tuples.")
        error_trigger = True
    else:
        for pair in xai_pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                logger.critical("Each xai pair must be a tuple of (xai_algorithm, xai_entry).")
                error_trigger = True
                continue

            xai_algorithm, xai_entry = pair
            if xai_algorithm not in EXPLAINERS:
                logger.critical(f"Unknown XAI algorithm '{xai_algorithm}'.")
                error_trigger = True
            if not isinstance(xai_entry, str) or len(xai_entry) == 0:
                logger.critical(f"The XAI entry for '{xai_algorithm}' must be non-empty.")
                error_trigger = True

    return not error_trigger
