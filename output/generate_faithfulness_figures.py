from src.utils.faithfulness.faithfulness_figures_generator import FaithfulnessFiguresGenerator
from src.utils.faithfulness.faithfulness_figures_utils import validate_faithfulness_figures_configs
from src.utils.logger import Logger

logger = Logger()

# Run from the repository root with: python3 -m output.generate_faithfulness_figures
# Add one configuration for each experiment and combined XAI comparison to generate.
FAITHFULNESS_FIGURES_CONFIGS = [
    {
        "experiment_id": "KolnWomen_bs32_cos_1e-5_try3",
        "mask_ceil": 1.0,
        "mask_step": 0.05,
        "patches_colors": ["green", "red"],
        "xai_comparisons": [
            {
                "output_xai_entry": "sq_patches16x16",
                "xai_pairs": [
                    ("GLimeBinomial", "sq_patches16x16-kw0.67-ns2048-base"),
                ],
            },
            {
                "output_xai_entry": "ink_based-agg1.0-char",
                "xai_pairs": [
                    ("GLimeBinomial", "ink_based-agg1.0-char-kw0.67-ns2048-new_mask"),
                ],
            }
        ],
    },
    {
            "experiment_id": "KolnWomen_bs32_cos_5e-4_full_try3",
            "mask_ceil": 1.0,
            "mask_step": 0.05,
            "patches_colors": ["green", "red"],
            "xai_comparisons": [
                {
                    "output_xai_entry": "sq_patches16x16",
                    "xai_pairs": [
                        ("GLimeBinomial", "sq_patches16x16-kw0.67-ns2048-base"),
                    ],
                },
                {
                    "output_xai_entry": "ink_based-agg1.0-char",
                    "xai_pairs": [
                        ("GLimeBinomial", "ink_based-agg1.0-char-kw0.67-ns2048-new_mask"),
                    ],
                }
            ],
        },
    {
            "experiment_id": "VatLat653_bs32_cos_1e-5_try7",
            "mask_ceil": 1.0,
            "mask_step": 0.05,
            "patches_colors": ["green", "red"],
            "xai_comparisons": [
                {
                    "output_xai_entry": "sq_patches16x16",
                    "xai_pairs": [
                        ("GLimeBinomial", "sq_patches16x16-kw0.67-ns2048-base"),
                    ],
                },
                {
                    "output_xai_entry": "ink_based-agg1.0-char",
                    "xai_pairs": [
                        ("GLimeBinomial", "ink_based-agg1.0-char-kw0.67-ns2048-new_mask"),
                    ],
                }
            ],
        },
    {
            "experiment_id": "VatLat653_bs32_cos_5e-4_full_try5",
            "mask_ceil": 1.0,
            "mask_step": 0.05,
            "patches_colors": ["green", "red"],
            "xai_comparisons": [
                {
                    "output_xai_entry": "sq_patches16x16",
                    "xai_pairs": [
                        ("GLimeBinomial", "sq_patches16x16-kw0.67-ns2048-base"),
                    ],
                },
                {
                    "output_xai_entry": "ink_based-agg1.0-char",
                    "xai_pairs": [
                        ("GLimeBinomial", "ink_based-agg1.0-char-kw0.67-ns2048-new_mask"),
                    ],
                }
            ],
        },
    {
            "experiment_id": "VatLat4221_bs32_cos_1e-5_try4",
            "mask_ceil": 1.0,
            "mask_step": 0.05,
            "patches_colors": ["green", "red"],
            "xai_comparisons": [
                {
                    "output_xai_entry": "sq_patches16x16",
                    "xai_pairs": [
                        ("GLimeBinomial", "sq_patches16x16-kw0.67-ns2048-base"),
                    ],
                },
                {
                    "output_xai_entry": "ink_based-agg1.0-char",
                    "xai_pairs": [
                        ("GLimeBinomial", "ink_based-agg1.0-char-kw0.67-ns2048-new_mask"),
                    ],
                }
            ],
        },
    {
            "experiment_id": "VatLat4221_bs32_cos_5e-4_full_try6",
            "mask_ceil": 1.0,
            "mask_step": 0.05,
            "patches_colors": ["green", "red"],
            "xai_comparisons": [
                {
                    "output_xai_entry": "sq_patches16x16",
                    "xai_pairs": [
                        ("GLimeBinomial", "sq_patches16x16-kw0.67-ns2048-base"),
                    ],
                },
                {
                    "output_xai_entry": "ink_based-agg1.0-char",
                    "xai_pairs": [
                        ("GLimeBinomial", "ink_based-agg1.0-char-kw0.67-ns2048-new_mask"),
                    ],
                }
            ],
        },
    {
            "experiment_id": "VatLat5951_bs32_cos_1e-5_try3",
            "mask_ceil": 1.0,
            "mask_step": 0.05,
            "patches_colors": ["green", "red"],
            "xai_comparisons": [
                {
                    "output_xai_entry": "sq_patches16x16",
                    "xai_pairs": [
                        ("GLimeBinomial", "sq_patches16x16-kw0.67-ns2048-base"),
                    ],
                },
                {
                    "output_xai_entry": "ink_based-agg1.0-char",
                    "xai_pairs": [
                        ("GLimeBinomial", "ink_based-agg1.0-char-kw0.67-ns2048-new_mask"),
                    ],
                }
            ],
        },
    {
            "experiment_id": "VatLat5951_bs32_cos_5e-4_full_try2",
            "mask_ceil": 1.0,
            "mask_step": 0.05,
            "patches_colors": ["green", "red"],
            "xai_comparisons": [
                {
                    "output_xai_entry": "sq_patches16x16",
                    "xai_pairs": [
                        ("GLimeBinomial", "sq_patches16x16-kw0.67-ns2048-base"),
                    ],
                },
                {
                    "output_xai_entry": "ink_based-agg1.0-char",
                    "xai_pairs": [
                        ("GLimeBinomial", "ink_based-agg1.0-char-kw0.67-ns2048-new_mask"),
                    ],
                }
            ],
        },
]


if __name__ == "__main__":
    validate_faithfulness_figures_configs(FAITHFULNESS_FIGURES_CONFIGS)

    for config in FAITHFULNESS_FIGURES_CONFIGS:
        EXPERIMENT_ID = config["experiment_id"]
        MASK_CEIL, MASK_STEP = config["mask_ceil"], config["mask_step"]
        PATCHES_COLORS = config["patches_colors"]

        for comparison in config["xai_comparisons"]:
            OUTPUT_XAI_ENTRY = comparison["output_xai_entry"]
            XAI_PAIRS = comparison["xai_pairs"]

            faithfulness_figures_generator = FaithfulnessFiguresGenerator(EXPERIMENT_ID, OUTPUT_XAI_ENTRY, XAI_PAIRS, MASK_CEIL, MASK_STEP, PATCHES_COLORS)
            faithfulness_figures_generator()

    logger.info("*** ALL REQUESTED FAITHFULNESS FIGURES HAVE BEEN GENERATED ***\n")
