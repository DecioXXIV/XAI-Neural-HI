# Cross-Model Explanation Agreement

Cross-model agreement compares page-level explanations produced by two trained
experiments. It reports rank correlation over all shared segments and over the
most positive or most negative regions selected by either model.

Entry point:

```text
src/cross_model_agreement.py
```

Run it from the repository root:

```bash
python -m src.cross_model_agreement [arguments]
```

## Prerequisites

Both experiments must have generated explanations with the same XAI algorithm.
The requested entries must exist in each experiment's `xai-metadata.json`.

Each runtime entry must contain:

```text
experiments/<experiment_id>/xai/<algorithm>/<entry>/
├── xai_instances_metadata.json
└── <page>/aggregated_scores.json
```

The CLI validates entry existence but does not verify that datasets,
segmentation configurations, classes, or page meanings are compatible. Those
conditions must be established by the experiment design.

## CLI reference

| Argument | Type | Required | Default | Description |
|---|---|---:|---|---|
| `-exp_id1` | string | Yes | — | First experiment |
| `-exp_id2` | string | Yes | — | Second experiment |
| `-xai_algorithm` | string | Yes | — | `Occlusion`, `Lime`, or `GLimeBinomial` |
| `-xai_entry1` | string | Yes | — | XAI entry from the first experiment |
| `-xai_entry2` | string | Yes | — | XAI entry from the second experiment |
| `-n_bootstrap` | integer | No | `20000` | Bootstrap replications; must be at least 2 |
| `-random_seed` | integer | No | Randomly generated | Seed for bootstrap sampling |

For reproducible confidence summaries, always provide `-random_seed`.

## Example

```bash
python -m src.cross_model_agreement \
  -exp_id1 vat653_resnet_run1 \
  -exp_id2 vat653_swin_run1 \
  -xai_algorithm Occlusion \
  -xai_entry1 sq_patches32x32-base \
  -xai_entry2 sq_patches32x32-base \
  -n_bootstrap 20000 \
  -random_seed 24
```

## Common pages

The evaluator reads the completed page names from both
`xai_instances_metadata.json` files and compares only their intersection.

No result is produced if there are no common explained pages.

## Score-key alignment

Unlike stability, cross-model agreement uses an intersection policy.

For each common page:

1. segment keys are loaded from both `aggregated_scores.json` files;
2. only keys present in both are retained;
3. keys are sorted and both score vectors are aligned to that order;
4. columns containing non-finite values are removed.

At least two usable shared segments are required for a finite correlation.

Key intersection does not establish semantic equivalence. Segment ID `42` must
represent the same region in both explanations for the result to be meaningful.
The safest configurations use the same deterministic square-patch grid or an
identical deterministic ink segmentation on the same page images.

## Spearman correlation

Spearman correlation is implemented as Pearson correlation between
average-tie ranks. A comparison returns `NaN` when:

- fewer than two values are available;
- either rank vector is constant;
- the result is non-finite.

### Overall agreement

The `overall` column uses every aligned shared segment.

### Top and bottom subsets

The evaluator also considers thresholds:

```text
5%, 10%, 25%
```

For each threshold and each model:

- `top` orders segments from highest to lowest attribution;
- `bottom` orders them from lowest to highest attribution.

The number selected per model is:

```text
max(1, int(threshold * number_of_shared_segments))
```

For a given rule and threshold, the evaluator takes the union of the segments
selected by either model, then computes Spearman correlation on that union.

The output columns are:

```text
top_5%
top_10%
top_25%
bottom_5%
bottom_10%
bottom_25%
overall
```

Using a union avoids evaluating only the regions that both models already
agree are important.

## Bootstrap confidence summaries

For each correlation column independently:

1. non-finite page correlations are discarded;
2. the arithmetic mean across pages is computed;
3. pages are sampled with replacement `n_bootstrap` times;
4. a mean is computed for every bootstrap sample;
5. the sample standard deviation of bootstrap means is used as the uncertainty
   estimate.

The reported summaries are:

```text
95%: mean +/- 1.96 * bootstrap_std
99%: mean +/- 2.576 * bootstrap_std
```

These are normal-style intervals based on bootstrap standard error, not
percentile bootstrap bounds.

## Output locations

Results are written from both experiments' perspectives.

First experiment:

```text
experiments/<exp_id1>/cross_model_agreement/<algorithm>/
└── <entry1>_vs_<exp_id2>__<entry2>/
    ├── spearman_correlations.csv
    ├── confidence_intervals.json
    └── score_cache/
```

Mirrored second experiment:

```text
experiments/<exp_id2>/cross_model_agreement/<algorithm>/
└── <entry2>_vs_<exp_id1>__<entry1>/
    ├── spearman_correlations.csv
    └── confidence_intervals.json
```

Path separators in identifiers are replaced with `__`.

## Interpretation

- `overall` measures global agreement in attribution ranking.
- `top_*` focuses on evidence supporting the target class.
- `bottom_*` focuses on evidence opposing the target class.
- A high subset correlation does not mean the models selected exactly the same
  segments; it means their ranks correlate on the union of their selections.

The evaluator uses normalized page-level scores. It does not compare predicted
classes, model accuracy, raw logits, or crop-level explanations.

## Reproducibility and reruns

The stage does not record its random seed in metadata or include it in the
output directory name. Reusing the same directory overwrites CSV and JSON
summaries. Provide a fixed seed and preserve outputs externally if multiple
bootstrap conditions must coexist.

The compatibility wrapper in
`src/utils/cross_model_agreement/cross_model_agreement_evaluator.py` re-exports
the active implementation from `src/utils/exp_comparison/evaluators.py`.
