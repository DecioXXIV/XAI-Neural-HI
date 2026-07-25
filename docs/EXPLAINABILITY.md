# Explainability

The explainability stage produces attribution scores for manuscript pages using
the fine-tuned classifier. A page is represented by interpretable segments,
evaluated through crop-level perturbations, and reconstructed as a page-level
explanation.

Entry point:

```text
src/explain.py
```

Run it from the repository root:

```bash
python -m src.explain [arguments]
```

## Prerequisites

The experiment must have completed fine-tuning and testing. At minimum, the
following artifacts are required:

```text
metadata/<experiment_id>/general-metadata.json
metadata/<experiment_id>/ft-metadata.json
experiments/<experiment_id>/fine_tuning/checkpoints/val_best_model.pth
experiments/<experiment_id>/fine_tuning/class_to_idx.json
experiments/<experiment_id>/fine_tuning/rgb_train_stats.pkl
```

The source pages must still be present under `data/<dataset>/`.

## CLI reference

| Argument | Type | Required | Default / choices | Description |
|---|---|---:|---|---|
| `-experiment_id` | string | Yes | — | Fine-tuned experiment |
| `-xai_algorithm` | string | Yes | `Occlusion`, `Lime`, `GLimeBinomial` | Explanation method |
| `-xai_details` | string | No | `base` | Free-form suffix used to distinguish runs |
| `-subsample` | string | No | `test` | `test`, `train`, `all`, or a custom path-list name |
| `-save_samples` | boolean | No | `false` | Accepted but currently does not save perturbed images |
| `-seg_type` | string | Yes | `sq_patches`, `ink_based` | Interpretable segmentation |
| `-patch_dim` | integer | Conditional | Required for `sq_patches` | Square patch side in pixels |
| `-aggressiveness` | float | No | `1.0` | Positive scale for ink grouping or character splitting |
| `-granularity` | string | No | `char`; choices: `raw`, `char`, `word`, `custom` | Ink segment granularity |
| `-grouping_method` | string | No | `auto`; choices: `auto`, `hull`, `dilation` | Grouping for `word` and `custom` |
| `-num_samples` | integer | No | `1024` | Perturbations for LIME or G-LIME |
| `-kernel_width` | float | Operationally required for LIME/G-LIME | `None` in the parser | Positive locality or sampling width |

Although the parser permits a missing `-kernel_width`, both LIME variants use
it in arithmetic. Supply a positive value when selecting `Lime` or
`GLimeBinomial`.

## Examples

### Square-patch Occlusion

```bash
python -m src.explain \
  -experiment_id vat653_resnet_run1 \
  -xai_algorithm Occlusion \
  -xai_details base \
  -subsample test \
  -seg_type sq_patches \
  -patch_dim 32
```

### Ink-based Occlusion

```bash
python -m src.explain \
  -experiment_id vat653_resnet_run1 \
  -xai_algorithm Occlusion \
  -xai_details ink_char_v1 \
  -subsample test \
  -seg_type ink_based \
  -aggressiveness 1.0 \
  -granularity char
```

### LIME

```bash
python -m src.explain \
  -experiment_id vat653_resnet_run1 \
  -xai_algorithm Lime \
  -xai_details run1 \
  -subsample test \
  -seg_type sq_patches \
  -patch_dim 32 \
  -num_samples 2048 \
  -kernel_width 0.67
```

## XAI entry names

Every configuration is assigned a deterministic entry name and stored under
`xai-metadata.json`.

Square patches begin with:

```text
sq_patches<patch_dim>x<patch_dim>
```

Ink-based configurations begin with:

```text
ink_based-agg<aggressiveness>-<granularity>
```

For `word` and `custom`, the grouping method is appended. LIME and G-LIME then
append:

```text
-kw<kernel_width>-ns<num_samples>
```

Finally, all entries append:

```text
-<xai_details>
```

Examples:

```text
sq_patches32x32-base
sq_patches16x16-kw0.67-ns2048-run1
ink_based-agg1.0-char-ink_char_v1
ink_based-agg1.5-word-hull-base
```

The runtime directory is:

```text
experiments/<experiment_id>/xai/<algorithm>/<xai_entry>/
```

## Instance selection

`InstanceToExplainRetriever` supports:

- `test`: all configured test pages;
- `train`: all configured training pages;
- `all`: both splits;
- any other string: read paths from
  `experiments/<experiment_id>/<subsample>.txt`.

A custom list must contain one image path per line. The parent directory name
of every image must match a class in `class_to_idx.json`.

Pages are explained for their ground-truth class index, not for the model's
predicted class.

## Page preprocessing

Preprocessing runs concurrently across pages and produces all inputs needed by
the explainer.

### Crop coordinates

The same `CropRetriever` grid used during fine-tuning is recomputed from the
original page with the stored `crop_size`. Coordinates and padding are saved in:

```text
crop_coordinates.csv
```

Each crop is stored under:

```text
<page>/crops/<crop_id>/<crop_id>.png
```

### Padded page

Right and bottom edge padding is applied so the crop coordinate system has
complete fixed-size crops. The padded image is saved as:

```text
<page>/<page>_forexp.png
```

### Segment map

The page-level integer label map is saved as:

```text
<page>/segments.npy
```

Every pixel with the same segment ID is treated as one interpretable feature.

## Square-patch segmentation

`SquarePatchesSegmentsHandler` creates a regular grid of
`patch_dim × patch_dim` regions. The grid is clipped to the padded page and
segment IDs are remapped to consecutive integers starting from zero.

The same page segmentation is sliced for every crop, ensuring that overlapping
crops refer to the same page-level segment IDs.

## Ink-based segmentation

Ink-based segmentation reserves ID `0` for the background and assigns positive
IDs to ink regions.

### 1. Foreground estimation

The page is converted to OpenCV BGR format. A scale-dependent median blur
estimates the local background. The foreground score combines:

- darkness relative to the local background;
- chromatic deviation in LAB space;
- optional HSV saturation contribution.

The current constants use chromatic deviation and disable the saturation term.
Otsu thresholding converts the score into a binary ink mask.

### 2. Component cleaning

Connected components are filtered to remove:

- very small regions;
- border-touching regions;
- long, thin horizontal structures near the top;
- optionally oversized blobs.

### 3. Text-scale estimation

The median width, height, and area of plausible components define a robust
page-specific text scale. Grouping and splitting parameters are derived from
this scale.

### 4. Granularity

- `raw`: return cleaned connected components.
- `char`: split wide components at valleys in their vertical projection, then
  reassign removed foreground pixels to the nearest surviving label.
- `word`: merge nearby strokes using padded convex hulls (`auto` and `hull`) or
  anisotropic dilation (`dilation`).
- `custom`: apply the selected grouping method with the current neutral
  defaults.

An additional colored segment overlay is saved for inspection:

```text
<page>/<page>_segments_overlay.png
```

## Perturbation and inference

The best validation checkpoint is loaded in evaluation mode. Perturbed crop
images are resized to the model input size, normalized with the training RGB
statistics, and evaluated in batches. Explainers operate on raw target-class
logits.

### Segment replacement

For square patches, removed pixels are replaced with the training-set RGB mean.

For ink segments, replacement is local:

1. the segment bounding box is expanded;
2. nearby background pixels with label `0` are collected;
3. their median RGB value is used as the erase color;
4. the replacement mask is dilated by one pixel only into background, avoiding
   neighboring ink segments.

The local replacement data is cached per crop and segment.

## Explainers

### Occlusion

For `n` segments, Occlusion creates:

- one all-active reference vector;
- one vector per segment with that segment disabled.

For segment \(j\), the attribution is:

```text
reference target logit - target logit with segment j removed
```

A positive score means removing the segment lowers the correct-class logit.
The ink background is never disabled.

### LIME

LIME generates `num_samples` random binary vectors and forces the first vector
to all ones. Ink background ID `0` remains active in every sample.

Euclidean distance from the all-active vector is converted to a locality
weight:

```text
sqrt(exp(-(distance^2) / kernel_width^2))
```

Both the target logits and binary vectors are centered around the original
sample. A no-intercept Ridge regressor with `alpha=1.0` is fitted using the
locality weights. Its coefficients are the segment attributions, and weighted
R² records surrogate fidelity for the crop.

### G-LIME Binomial

G-LIME samples the number of active segments from a binomial-derived
distribution adjusted by `kernel_width`, then selects that many segments
without replacement. The first vector is all-active and ink background remains
active.

It fits the same centered Ridge surrogate as LIME without sample weights.

Neither LIME implementation currently exposes a random seed through the CLI.

## Crop-to-page aggregation

Every crop writes its raw segment scores to:

```text
<page>/crops/<crop_id>/scores.json
```

Occlusion aggregates a page segment by taking the mean of its scores from
overlapping crops.

LIME and G-LIME clamp negative crop R² values to zero and use positive R² as
crop weights. If no crop has a positive R², uniform weights are used. The
weighted contributions are divided by the total crop weight for the page.

The page directory receives:

```text
all_raw_scores.json
aggregated_raw_scores.json
aggregated_scores.json
crop_r2s.json                 # LIME and G-LIME
```

The current normalization sorts raw signed scores, accumulates their absolute
magnitudes until 99% of total magnitude is reached, and divides every score by
the signed score at that threshold.

## Visualizations

Two forms are generated:

- a static red-white-green attribution overlay;
- an interactive HTML visualization with segment hover, rank, thresholding,
  zoom, pan, and selection.

The interactive representation stores segment IDs in a 24-bit RGB map and
writes page-specific HTML, CSS, JavaScript, score, and data files. Visualization
workers run in separate spawned processes.

## Resume and completion behavior

The entry-level file:

```text
xai_instances_metadata.json
```

maps completed page names to timestamps.

- A page already listed there is not explained again.
- Existing crop `scores.json` files are reused during an incomplete page retry.
- Missing or outdated visualization files can be rebuilt without recomputing
  attribution scores.
- After all requested pages finish, `END_TIMESTAMP` is written to the
  configuration in `metadata/<experiment_id>/xai-metadata.json`.

Use a distinct `-xai_details` value when intentionally creating another run
with otherwise identical parameters.

## Next stages

- Use test-page explanations for [faithfulness](FAITHFULNESS.md).
- Generate multiple compatible entries for [stability](STABILITY.md).
- Generate the same kind of explanation for another model before
  [cross-model agreement](CROSS_MODEL_AGREEMENT.md).
- Explain training pages before XAI-guided [retraining](RETRAINING.md).
