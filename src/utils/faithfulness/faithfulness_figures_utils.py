from typing import Any, Dict, List

from src import EXPLAINERS
from src.utils.logger import Logger
from src.utils.metadata.metadata_utils import get_xai_metadata

logger = Logger()

XAI_PLOT_STYLES = {
    "GLimeBinomial": {
        "label": "GLIME-Binomial",
        "color": "#0072B2",
        "random_linestyle": (0, (6, 3)),
    },
    "Occlusion": {
        "label": "Occlusion",
        "color": "#D55E00",
        "random_linestyle": (3, (6, 3)),
    },
    "Lime": {
        "label": "LIME",
        "color": "#009E73",
        "random_linestyle": (6, (6, 3)),
    },
}


def get_masking_xlabel(patches_color: str) -> str:
    if patches_color == "green": return "Fraction of positively attributed (green) patches removed"
    elif patches_color == "red": return "Fraction of negatively attributed (red) patches removed"
    else: raise ValueError(f"Unsupported patches_color '{patches_color}'. Expected 'green' or 'red'.")


def get_xai_plot_style(xai_algorithm: str) -> Dict[str, Any]:
    if xai_algorithm not in XAI_PLOT_STYLES:
        raise ValueError(f"No plot style is defined for '{xai_algorithm}'.")
    return XAI_PLOT_STYLES[xai_algorithm]


def validate_faithfulness_figures_configs(figures_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(figures_configs) == 0:
        logger.warning("No faithfulness figures configuration has been provided.")
        return []

    valid_configs = []
    for config in figures_configs:
        if _validate_faithfulness_figures_config(config): valid_configs.append(config)
        else: logger.warning("Skipping invalid faithfulness figures configuration.")

    return valid_configs


def _validate_faithfulness_figures_config(config: Dict[str, Any]) -> bool:
    error_trigger = False

    if not isinstance(config, dict):
        logger.critical("Each faithfulness figures configuration must be a dictionary.")
        return False

    experiment_id = config.get("experiment_id")
    mask_ceil, mask_step = config.get("mask_ceil"), config.get("mask_step")
    patches_colors = config.get("patches_colors")
    xai_comparisons = config.get("xai_comparisons")

    if not isinstance(experiment_id, str) or len(experiment_id) == 0:
        logger.critical("Each faithfulness figures configuration must define a non-empty experiment_id.")
        return False

    valid_mask_ceil = isinstance(mask_ceil, (int, float)) and 0 < mask_ceil <= 1
    valid_mask_step = isinstance(mask_step, (int, float)) and valid_mask_ceil and 0 < mask_step < mask_ceil

    if not valid_mask_ceil:
        logger.critical(f"mask_ceil must be a float in the range (0, 1] for '{experiment_id}'.")
        error_trigger = True

    if not valid_mask_step:
        logger.critical(f"mask_step must be a float in the range (0, mask_ceil) for '{experiment_id}'.")
        error_trigger = True

    if not isinstance(patches_colors, list) or len(patches_colors) == 0 or len(set(patches_colors)) != len(patches_colors) or any(c not in ("green", "red") for c in patches_colors):
        logger.critical(f"patches_colors for '{experiment_id}' must be a non-empty list containing only 'green' and/or 'red'.")
        error_trigger = True

    if not isinstance(xai_comparisons, list) or len(xai_comparisons) == 0:
        logger.critical(f"At least one xai comparison must be configured for '{experiment_id}'.")
        return False

    xai_metadata = get_xai_metadata(experiment_id)
    output_xai_entries = set()

    for comparison in xai_comparisons:
        if not isinstance(comparison, dict):
            logger.critical(f"Each xai comparison for '{experiment_id}' must be a dictionary.")
            error_trigger = True
            continue

        output_xai_entry = comparison.get("output_xai_entry")
        xai_pairs = comparison.get("xai_pairs")

        if not isinstance(output_xai_entry, str) or len(output_xai_entry) == 0:
            logger.critical(f"Each xai comparison for '{experiment_id}' must define a non-empty output_xai_entry.")
            error_trigger = True
        elif output_xai_entry in output_xai_entries:
            logger.critical(f"output_xai_entry '{output_xai_entry}' is repeated for '{experiment_id}'.")
            error_trigger = True
        else: output_xai_entries.add(output_xai_entry)

        if not isinstance(xai_pairs, list) or len(xai_pairs) == 0:
            logger.critical(f"xai_pairs must be a non-empty list for '{experiment_id}/{output_xai_entry}'.")
            error_trigger = True
            continue

        configured_algorithms = set()
        for pair in xai_pairs:
            if not isinstance(pair, tuple) or len(pair) != 2:
                logger.critical(f"Each xai pair for '{experiment_id}/{output_xai_entry}' must be a tuple of (xai_algorithm, xai_entry).")
                error_trigger = True
                continue

            xai_algorithm, xai_entry = pair
            if xai_algorithm not in EXPLAINERS:
                logger.critical(f"Unknown XAI algorithm '{xai_algorithm}' for '{experiment_id}/{output_xai_entry}'.")
                error_trigger = True
                continue

            if xai_algorithm in configured_algorithms:
                logger.critical(f"XAI algorithm '{xai_algorithm}' is repeated for '{experiment_id}/{output_xai_entry}'.")
                error_trigger = True
            else: configured_algorithms.add(xai_algorithm)

            if not isinstance(xai_entry, str) or len(xai_entry) == 0:
                logger.critical(f"The XAI entry for '{xai_algorithm}' in '{experiment_id}/{output_xai_entry}' must be non-empty.")
                error_trigger = True
            elif xai_algorithm not in xai_metadata or xai_entry not in xai_metadata[xai_algorithm]:
                logger.critical(f"XAI entry '{xai_entry}' for '{xai_algorithm}' is not available for '{experiment_id}'.")
                error_trigger = True

    return not error_trigger
