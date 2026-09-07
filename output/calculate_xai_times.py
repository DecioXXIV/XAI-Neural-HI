from src.utils.explain.xai_explanation_times_evaluator import XaiExplanationTimesEvaluator
from src.utils.explain.xai_explanation_times_utils import validate_xai_explanation_times_configs
from src.utils.logger import Logger

logger = Logger()

# Run from the repository root with: python -m output.calculate_xai_times
XAI_EXPLANATION_TIMES_CONFIGS = [
    {
        "experiment_ids": [
            "VatLat653_bs32_cos_5e-4_full_try5",
            "VatLat4221_bs32_cos_5e-4_full_try6",
            "VatLat5951_bs32_cos_5e-4_full_try2",
            "KolnWomen_bs32_cos_5e-4_full_try3",
            "VatLat653_bs32_cos_1e-5_try7",
            "VatLat4221_bs32_cos_1e-5_try4",
            "VatLat5951_bs32_cos_1e-5_try3",
            "KolnWomen_bs32_cos_1e-5_try3",
        ],
        "xai_pairs": [
            ("Occlusion", "sq_patches16x16-base"),
            ("Occlusion", "sq_patches32x32-base"),
            ("GLimeBinomial", "sq_patches16x16-kw0.67-ns2048-base"),
            ("GLimeBinomial", "sq_patches32x32-kw0.67-ns512-base"),
        ],
    },
]


if __name__ == "__main__":
    VALID_XAI_EXPLANATION_TIMES_CONFIGS = validate_xai_explanation_times_configs(XAI_EXPLANATION_TIMES_CONFIGS)
    N_COMPLETED, N_FAILED = 0, 0

    for config in VALID_XAI_EXPLANATION_TIMES_CONFIGS:
        EXPERIMENT_IDS = config["experiment_ids"]
        XAI_PAIRS = config["xai_pairs"]

        for EXPERIMENT_ID in EXPERIMENT_IDS:
            for XAI_ALGORITHM, XAI_ENTRY in XAI_PAIRS:
                try:
                    xai_explanation_times_evaluator = XaiExplanationTimesEvaluator(EXPERIMENT_ID, XAI_ALGORITHM, XAI_ENTRY)
                    xai_explanation_times_evaluator()
                    N_COMPLETED += 1
                except Exception as e:
                    N_FAILED += 1
                    logger.exception(f"Failed to compute XAI explanation times for '{EXPERIMENT_ID}/{XAI_ALGORITHM}/{XAI_ENTRY}': {e}")

    logger.info(f"*** XAI EXPLANATION TIMES COMPUTATION COMPLETED -> Completed: {N_COMPLETED} | Failed: {N_FAILED} ***\n")
