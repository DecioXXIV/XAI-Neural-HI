# Faithfulness Evaluation

The faithfulness stage tests whether the regions identified by an explanation
have a measurable effect on classifier behavior. It progressively masks page
segments, rebuilds crop-level test sets, and tracks classification performance
as masking increases.

Entry point:

```text
src/faithfulness.py
```

Run it from the repository root:

```bash
python -m src.faithfulness [arguments]
```

## Prerequisites

The selected XAI entry must exist in:

```text
metadata/<experiment_id>/xai-metadata.json
```

Its runtime directory must contain:

```text
experiments/<experiment_id>/xai/<algorithm>/<xai_entry>/
├── xai_instances_metadata.json
└── <page>/
    ├── <page>_forexp.png
    ├── segments.npy
    └── aggregated_scores.json
```

Faithfulness only evaluates explained instances that belong to the dataset's
test split. The supported workflow is to explain the complete test split before
running this stage.

## CLI reference

| Argument | Type | Required | Default / choices | Description |
|---|---|---:|---|---|
| `-experiment_id` | string | Yes | — | Experiment containing the trained model |
| `-xai_algorithm` | string | Yes | `Occlusion`, `Lime`, `GLimeBinomial` | Explanation method |
| `-xai_entry` | string | Yes | Existing entry | Exact explanation configuration |
| `-mask_ceil` | float | Yes | `(0, 1]` | Highest requested mask rate |
| `-mask_step` | float | Yes | `(0, 1]`, strictly below `mask_ceil` | Increment between mask rates |
| `-mask_rule` | string | Yes | `saliency`, `random` | Segment selection strategy |
| `-patches_color` | string | Yes | `green`, `red` | Attribution sign of interest |
| `-keep_masked_pages` | boolean | No | `false` | Accepted but currently not used for cleanup |
| `-keep_test_sets` | boolean | No | `false` | Retain generated crop test sets |
| `-random_seed` | integer | Conditional | Randomly generated for `random` when absent | Global random-baseline seed |

## Examples

### Saliency deletion of positive regions

```bash
python -m src.faithfulness \
  -experiment_id vat653_resnet_run1 \
  -xai_algorithm Occlusion \
  -xai_entry sq_patches32x32-base \
  -mask_ceil 1.0 \
  -mask_step 0.05 \
  -mask_rule saliency \
  -patches_color green \
  -keep_test_sets false
```

### Random baseline

```bash
python -m src.faithfulness \
  -experiment_id vat653_resnet_run1 \
  -xai_algorithm Occlusion \
  -xai_entry sq_patches32x32-base \
  -mask_ceil 1.0 \
  -mask_step 0.05 \
  -mask_rule random \
  -patches_color green \
  -random_seed 24 \
  -keep_test_sets false
```

Run comparable saliency and random configurations with the same color to
evaluate whether attribution ordering is more destructive than chance.

## Faithfulness entry names

The configuration identifier is:

```text
<mask_rule>-ceil<mask_ceil>-step<mask_step>-<patches_color>
```

Random configurations append:

```text
-seed<random_seed>
```

Examples:

```text
saliency-ceil1.0-step0.05-green
random-ceil1.0-step0.05-green-seed24
```

Completion timestamps are recorded in:

```text
metadata/<experiment_id>/faithfulness-metadata.json
```

An entry already present in that metadata is treated as complete and is not
rerun.

## Mask rates

`compute_mask_rates()` starts at `mask_step` and repeatedly adds the step while
the value is no greater than `mask_ceil`. Values are rounded to five decimal
places when appended.

For example:

```text
mask_step = 0.05
mask_ceil = 0.20
```

produces:

```text
0.05, 0.10, 0.15, 0.20
```

The unmasked baseline `0.0` is added later during test-set construction and
evaluation.

## Explained test-instance retrieval

The stage intersects the page names in `xai_instances_metadata.json` with the
actual test pages configured for the experiment. It then uses the padded
`<page>_forexp.png` image generated during explainability.

Explanations generated only for training pages are ignored by this retrieval
step.

## Segment mapping

For every page, a masking-results CSV combines:

- segment ID;
- normalized page-level attribution;
- bounding-box coordinates;
- area information;
- for ink segments, the local replacement RGB color.

The files are cached under:

```text
experiments/<experiment_id>/masked_images/<algorithm>/<xai_entry>/masking_results/
```

## Attribution color

`patches_color` selects the sign whose population determines the saliency
deletion schedule:

- `green`: positive attribution scores;
- `red`: negative attribution scores.

For square patches, green filtering includes zero-valued scores. For ink-based
segments, background ID `0` is excluded and green filtering is strictly
positive.

The color name refers to explanation semantics and visualization, not the
replacement color used in the image.

## Masking rules

### Saliency

Candidate regions are ordered by attribution:

- green: descending score;
- red: ascending score, so the most negative region comes first.

Masking is cumulative across rates. Once a region has been removed, it remains
removed for all higher mask rates. At rate `r`, the target is approximately:

```text
r * number of sign-filtered regions
```

The loop stops after the corresponding number of regions has been selected.

### Random

The random baseline derives a stable per-page seed from:

```text
global random seed + page name
```

This makes concurrent page processing reproducible. Each mask rate starts from
the original image and draws a new random permutation. Random mask sets are
therefore independent across rates rather than nested.

The number of regions to mask is still based on the number of regions matching
the selected sign, but the random regions themselves are drawn from the full
segment population. This preserves a masking-count baseline for the selected
positive or negative population without using its attribution order.

## Replacement behavior

### Square patches

The rectangular bounding box of each selected patch is filled with the
training-set mean RGB color.

### Ink-based segments

Only the selected ink pixels and a one-pixel dilation into background are
replaced. The erase color is the median local background color estimated around
the segment. Neighboring unselected ink labels are protected.

The stored ink `masked_area_ratio` is based on the actual replacement mask. The
square-patch ratio is based on selected rectangular patch areas.

## Masking outputs

Masked pages are saved as:

```text
experiments/<experiment_id>/masked_images/
└── <algorithm>/<xai_entry>/<faithfulness_entry>/
    ├── mask_rate0.05/<page>.png
    ├── mask_rate0.10/<page>.png
    └── ...
```

Per-page metadata records:

- realized masked-area ratio;
- number of masked regions;
- selected segment IDs.

It is stored under:

```text
masked_images/<algorithm>/<xai_entry>/metadata/<faithfulness_entry>/
```

`-keep_masked_pages` currently has no effect: masked pages are retained.

## Test-set construction

For every rate, including `0.0`, each page is passed through the same
`CropRetriever` used during fine-tuning. The resulting ImageFolder-compatible
layout is:

```text
experiments/<experiment_id>/faithfulness/
└── <algorithm>/<xai_entry>/<faithfulness_entry>/test_sets/
    ├── 0.0/<class>/
    ├── 0.05/<class>/
    ├── 0.10/<class>/
    └── ...
```

The `0.0` set is rebuilt from the unmasked padded explanation pages. Other
rates use their corresponding masked pages.

## Model evaluation

The fine-tuned best validation checkpoint is evaluated independently at every
mask rate. Images use the same model input size and RGB normalization as the
original experiment.

The evaluator computes:

- crop-level accuracy;
- crop-level macro-F1;
- page-level accuracy;
- page-level macro-F1;
- every crop's class logits;
- every crop's class probabilities.

Page predictions use majority vote over crop predictions.

The current implementation reuses the complete fine-tuning
`n_crops_per_instance.json` when grouping crop predictions by page. For this
reason, partial test subsamples can produce incorrect page grouping; explain
the complete test split for supported faithfulness evaluation.

## Evaluation outputs

```text
experiments/<experiment_id>/faithfulness/
└── <algorithm>/<xai_entry>/<faithfulness_entry>/
    ├── faithfulness_crop_level.json
    ├── faithfulness_page_level.json
    ├── faithfulness_logits_report.csv
    ├── faithfulness_probs_report.csv
    └── test_sets/                         # optional after completion
```

The JSON files map every mask rate to accuracy and macro-F1. CSV rows identify
the mask rate, page, crop, true class, and class-specific model output.

The stage produces performance curves as raw data; it does not currently
reduce them to a deletion AUC or another single faithfulness score.

If `-keep_test_sets false`, the temporary `test_sets/` tree is removed after
evaluation. Masked pages and reports remain available.
