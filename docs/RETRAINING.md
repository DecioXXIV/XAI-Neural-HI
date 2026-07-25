# XAI-Guided Retraining

The retraining stage constructs a second training set and trains another model
for the same writer-identification task. The new set can be sampled randomly or
assembled from crops scored with the first fine-tuned model and its
explanations.

Entry point:

```text
src/retrain.py
```

Run it from the repository root:

```bash
python -m src.retrain [arguments]
```

## Prerequisites

The original experiment must have completed fine-tuning. Both selection rules
load the padded training pages from an XAI entry, so the training split must
have been processed by `src.explain`, not only the test split. The `saliency`
rule additionally requires segment scores and, for LIME variants, crop R².

Required base artifacts include:

```text
metadata/<experiment_id>/general-metadata.json
metadata/<experiment_id>/ft-metadata.json
experiments/<experiment_id>/fine_tuning/checkpoints/val_best_model.pth
experiments/<experiment_id>/fine_tuning/n_crops_per_instance.json
experiments/<experiment_id>/fine_tuning/rgb_train_stats.pkl
```

For every selection rule:

```text
experiments/<experiment_id>/xai/<algorithm>/<xai_entry>/<training_page>/
└── <training_page>_forexp.png
```

For saliency selection, the same directory also requires:

```text
segments.npy
aggregated_scores.json
crop_r2s.json                  # LIME and G-LIME
```

## CLI reference

| Argument | Type | Required | Default / choices | Description |
|---|---|---:|---|---|
| `-experiment_id` | string | Yes | — | Original fine-tuned experiment |
| `-xai_algorithm` | string | Yes | `Occlusion`, `Lime`, `GLimeBinomial` | Explanation method |
| `-xai_entry` | string | Yes | Existing entry | Explanation configuration |
| `-original_ts_ratio` | float | Yes | `[0, 1]` | Target fraction of original fixed-grid crops |
| `-new_ts_ratio` | float | Yes | `(0, 1]` | Target fraction of newly proposed crops |
| `-selection_rule` | string | Yes | `saliency`, `random` | Training-set construction strategy |
| `-start_point` | string | Yes | `from_zero`, `from_ft1` | Initial checkpoint policy |
| `-batch_size` | integer | Yes | Positive | Retraining batch size |
| `-weight_decay` | float | No | `0.0001` | Optimizer weight decay |
| `-lr` | float | Yes | Positive | Retraining learning rate |
| `-lr_scheduler` | string | Yes | `CosineAnnealingLR` | Scheduler |
| `-lr_final_decay_ratio` | float | No | `0.01` | Cosine minimum relative to `lr` |
| `-label_smoothing` | float | No | `0.0` | Cross-entropy smoothing |
| `-early_stopping` | boolean | No | `true` | Enable post-trigger early stopping |
| `-early_stopping_patience` | integer | Conditional | Required when early stopping is enabled | Non-improving epochs after trigger |
| `-random_seed` | integer | No | Randomly generated | Sampling, augmentation, and training seed |
| `-epochs` | integer | No | `50` | Early-stopping trigger or hard epoch count |
| `-train_img_transforms` | string | Yes | Fine-tuning augmentation choices | Retraining augmentation preset |
| `-ft_mode` | string | Yes | `frozen`, `full` | Retrained encoder mode |
| `-keep_crops` | boolean | No | `false` | Retain the configuration-specific training crops |

The optimizer type, steering metric, classification-head layout, and crop size
are inherited from the first fine-tuning metadata. Only the listed retraining
hyperparameters can be changed through this CLI.

`original_ts_ratio + new_ts_ratio` is not constrained to be at most one.

## Example: XAI-guided retraining

```bash
python -m src.retrain \
  -experiment_id vat653_resnet_run1 \
  -xai_algorithm Occlusion \
  -xai_entry sq_patches32x32-base \
  -original_ts_ratio 0.5 \
  -new_ts_ratio 0.5 \
  -selection_rule saliency \
  -start_point from_ft1 \
  -batch_size 32 \
  -lr 0.00005 \
  -lr_scheduler CosineAnnealingLR \
  -lr_final_decay_ratio 0.1 \
  -weight_decay 0.01 \
  -label_smoothing 0.05 \
  -early_stopping true \
  -early_stopping_patience 5 \
  -random_seed 24 \
  -epochs 15 \
  -train_img_transforms moderate \
  -ft_mode full \
  -keep_crops false
```

## Runtime directory structure

Shared data for one explanation entry is stored under:

```text
experiments/<experiment_id>/retraining/<algorithm>/<xai_entry>/
```

One retraining condition is stored under:

```text
<ft_mode>-<start_point>/
└── <selection_rule>/
    └── original<original_ratio>-new<new_ratio>/
        └── random_seed<seed>/
```

The full configuration directory contains its training set, checkpoints,
history, and test outputs.

Retraining metadata is stored separately:

```text
metadata/<experiment_id>/retraining/
└── <ft_mode>-<start_point>/<selection_rule>/
    └── original<original_ratio>-new<new_ratio>/random_seed<seed>/
        └── retrain-metadata.json
```

The current metadata path does not include the XAI algorithm or XAI entry.
Using the same mode, start point, ratios, rule, and seed for different XAI
entries would therefore address the same metadata file. Use distinct condition
values or archive metadata when comparing XAI entries.

## Shared original crop dataset

The first invocation creates a fixed-grid crop dataset at the XAI-entry root:

```text
retraining/<algorithm>/<xai_entry>/
├── train/<class>/
├── val/<class>/
├── test/<class>/
└── n_crops_per_instance.json
```

It uses one training replica and the crop size inherited from the original
fine-tuning stage. This shared dataset supplies:

- candidate original crops;
- validation crops;
- test crops;
- page-level crop counts.

The stage uses the original fine-tuning RGB statistics instead of recomputing
normalization for the new set.

## Target crop counts

For each class, the number of files in the shared retraining `train/` directory
defines the reference count.

For saliency selection:

```text
memory target = ceil(reference_count * original_ts_ratio)
new target    = ceil(reference_count * new_ts_ratio)
```

The two populations are scored and selected separately.

For random selection, the stage instead extracts:

```text
ceil(reference_count * (original_ts_ratio + new_ts_ratio))
```

random crops per class.

## Saliency selection: memory crops

Memory crops are fixed-grid crops from the original training pages. They are
intended to preserve examples the first model already understands and explains
positively.

### Coordinate mapping

The stage reconstructs the fixed crop grid for every training page and maps
each crop path to page coordinates in:

```text
coords_to_ft1_train_crop.json
```

### Classification confidence

For each crop, the first model compares the true-class logit with the largest
other-class logit:

```text
margin = true_logit - max_other_logit
```

Negative margins are clamped to zero. The confidence transform is:

```text
1 - exp(-0.5 * margin^2)
```

Zero confidences can be replaced with a small value derived from the two
smallest positive confidences so the final product is not automatically zero.

### Positive XAI evidence

The crop's page-level segments are retrieved from the explanation. Positive and
negative attribution magnitudes are averaged over the unique segments
intersecting the crop.

Positive evidence is:

```text
green^2 / (green + red + 1e-8)
```

### Surrogate R²

- Occlusion uses a fixed value of `1.0`.
- LIME and G-LIME retrieve the crop R² saved during explanation.
- A missing crop R² contributes `0.0`.

### Positive ink fraction

The crop is converted to grayscale and binarized with Otsu's method. The score
is the fraction of ink pixels inside regions belonging to positive-attribution
segments.

### Memory score

```text
memory =
    adjusted_confidence
    * positive_xai_evidence
    * crop_r2
    * positive_ink_fraction
```

All factors and the product are written to `memory_scores.csv`.

### Memory selection

Candidates are sorted by memory score. For each class, the selector first
allocates an equal base number of crops to every page, then fills the remainder
with at most one additional crop per page in score order.

Selected files are copied into the configuration-specific training directory.

## Saliency selection: new crops

New crops are centered on positive-attribution page segments rather than on the
original fixed grid.

For every positive score:

1. retrieve the segment bounding box;
2. take its center;
3. create a crop of the inherited crop size around that center;
4. permit up to 10% overflow beyond page boundaries;
5. fill accepted overflow with the training RGB mean;
6. save the crop as `<page>_patch<segment_id>.png`.

Coordinates are saved in:

```text
coords_to_xai_crop.json
```

### Difficulty

The current difficulty transform first computes:

```text
max_other_logit - true_logit
```

and then:

```text
exp(-0.5 * margin^2)
```

It is largest near the class decision boundary and decreases as the absolute
logit margin grows.

### Openness score

New-crop positive XAI evidence and positive ink fraction are computed as for
memory crops. The final score is:

```text
openness =
    difficulty
    * positive_xai_evidence
    * positive_ink_fraction
```

The component factors and product are saved to `openness_scores.csv`.

### Spatially diverse selection

Positive page segments are connected when their pixels are 8-neighbors across
a segment boundary. This creates connected components of positive evidence.

For every page, candidates are grouped by connected component and selected in
round-robin order, prioritizing components whose best crop has higher openness.
The remainder phase adds at most one extra crop per page.

This reduces concentration on many nearby segments from the same evidence
region.

## Random selection

The random rule does not use attribution scores to select locations.

For every class:

1. training pages are sampled with replacement;
2. crop top-left coordinates are sampled from a range that permits 10% border
   overflow;
3. overflow is padded with the training RGB mean;
4. crops are saved as `<page>_random<index>.png`.

The NumPy generator is initialized with `-random_seed`, making page and
coordinate sampling reproducible.

The XAI entry is still used to locate each padded training page.

## Model starting point

### `from_zero`

The model is instantiated through the normal model loader without loading the
first fine-tuned classifier. This does not mean that every encoder is randomly
initialized:

- ResNet18 still loads its repository base checkpoint;
- Swin models still load torchvision ImageNet weights;
- the new classification head is freshly initialized.

### `from_ft1`

On the first run, the original fine-tuning
`checkpoints/val_best_model.pth` initializes the retrained model.

If retraining has already completed epochs, its own
`checkpoints/last_checkpoint.pth` is loaded instead.

## Training and testing

The configuration-specific `train/` directory is used for optimization. The
shared XAI-entry root supplies validation and test crops.

Training reuses the same `ModelTrainer` as fine-tuning:

- deterministic augmented loader;
- cross-entropy and label smoothing;
- inherited optimizer type;
- cosine scheduling;
- optional early stopping;
- AMP and multi-GPU support;
- resumable checkpoints and history.

Testing loads the best retraining checkpoint. Before page-level evaluation, the
shared `n_crops_per_instance.json` is copied into the condition directory.
Results have the same crop-level and page-level structure as the original
fine-tuning output.

## Idempotency and cleanup

The stage reuses existing coordinate maps and score CSVs. Training and testing
are skipped when their corresponding timestamps exist in
`retrain-metadata.json`.

When `-keep_crops false`, the condition-specific `train/`, `val/`, and `test/`
subdirectories are removed after testing. Shared crops, score tables,
checkpoints, history, and evaluation outputs remain.

## Naming constraint

Several retraining utilities recover the source page name by splitting crop
filenames at underscores. Page stems containing underscores can therefore be
misidentified. The current supported data naming convention uses page stems
without underscores.
