# Explanation Stability

The stability stage measures agreement among multiple explanation runs for the
same trained experiment. It is intended for repeated stochastic runs or
controlled variants that share the same interpretable segment IDs.

Entry point:

```text
src/stability.py
```

Run it from the repository root:

```bash
python -m src.stability [arguments]
```

## Prerequisites

At least two completed XAI entries must exist under the same algorithm in:

```text
metadata/<experiment_id>/xai-metadata.json
```

An entry is considered completed only if it contains `END_TIMESTAMP`.

Every selected entry must also have:

```text
experiments/<experiment_id>/xai/<algorithm>/<entry>/
├── xai_instances_metadata.json
└── <page>/aggregated_scores.json
```

## CLI reference

| Argument | Type | Required | Description |
|---|---|---:|---|
| `-experiment_id` | string | Yes | Experiment whose explanation runs are compared |
| `-xai_algorithm` | string | Yes | `Occlusion`, `Lime`, or `GLimeBinomial` |
| `-xai_root_entry` | string | Yes | Prefix shared by the XAI entries to include |

## Selecting entries by prefix

The stage reads all entries for the selected algorithm and keeps those that:

1. start with `xai_root_entry`;
2. contain an `END_TIMESTAMP`.

For example, these completed entries:

```text
sq_patches32x32-kw0.67-ns2048-run1
sq_patches32x32-kw0.67-ns2048-run2
sq_patches32x32-kw0.67-ns2048-run3
```

can be selected with:

```text
sq_patches32x32-kw0.67-ns2048-
```

At least two matching entries are required.

## Example

```bash
python -m src.stability \
  -experiment_id vat653_resnet_run1 \
  -xai_algorithm Lime \
  -xai_root_entry sq_patches32x32-kw0.67-ns2048-
```

The ordering of entries follows their ordering in `xai-metadata.json`. A
separate output file records the index assigned to each entry.

## Common-instance selection

For every selected entry, the evaluator loads the page names from
`xai_instances_metadata.json`. Only their set intersection is compared.

This permits entries with different page coverage, but pages that are not
present in every run are excluded from the result.

## Strict score alignment

For each common page, `aggregated_scores.json` is loaded from every entry.
Segment keys are sorted and must be exactly identical across all runs.

The strict policy is appropriate when the segmentation is deterministic and
shared in meaning, for example:

- repeated LIME runs over the same square-patch grid;
- repeated G-LIME runs over the same saved ink segmentation;
- otherwise identical entries separated only by `xai_details`.

If any entry has different keys, the stage raises a score-key mismatch error.
It does not attempt spatial interpolation or key intersection.

## Per-page Pearson correlations

The aligned score vectors form a matrix with:

```text
rows    = XAI entries
columns = interpretable segments
```

For each page, the evaluator computes the Pearson correlation for every pair of
entries.

A pair returns `NaN` when:

- fewer than two finite segments remain;
- either score vector is constant;
- the resulting correlation is non-finite.

Columns containing a non-finite value in any entry are discarded before
correlation.

The raw table uses columns such as:

```text
instance,0_vs_1,0_vs_2,1_vs_2
```

## Aggregate stability estimate

For each page:

1. finite pairwise correlations are clipped just inside `[-1, 1]`;
2. Fisher's `atanh` transform converts them to z-values;
3. available pairwise z-values are averaged;
4. `tanh` converts the page mean back to a Pearson value.

The final mean and sample standard deviation are calculated across the finite
page-level means.

The standard error is:

```text
sample standard deviation / sqrt(number of finite pages)
```

The reported intervals are normal-style summaries:

```text
95%: mean +/- 1.976 * standard error
99%: mean +/- 2.576 * standard error
```

They are stored as formatted strings along with numeric `mean` and `std`.

## Score cache

Parsed JSON scores are cached as NumPy `.npz` data under the stability output
directory. A cache file is reused when it is at least as new as the source
`aggregated_scores.json`.

This avoids repeatedly parsing large JSON dictionaries on later runs.

## Outputs

```text
experiments/<experiment_id>/stability/
└── <algorithm>/<xai_root_entry>/
    ├── xai_entry_index.csv
    ├── pearson_correlations.csv
    ├── confidence_intervals.json
    └── score_cache/
```

`xai_entry_index.csv` is required to interpret numeric pair names in the raw
correlation table.

## Interpretation

High positive correlation means segment relevance varies similarly across
runs. Values around zero indicate unstable ordering or magnitude, while
negative values indicate systematic reversal.

The comparison operates on normalized `aggregated_scores.json` values, not on
raw crop scores or raw page scores. It assesses attribution agreement only; it
does not incorporate crop-level LIME R² into the final stability metric.

## Important assumptions

- Entries must describe the same model and experiment.
- Segment IDs must have the same spatial or semantic meaning across entries.
- The entry prefix should vary only the experimental factor whose stability is
  being assessed.
- The stage writes no completion timestamp and recomputes its summary when
  invoked, while reusing score caches when current.
