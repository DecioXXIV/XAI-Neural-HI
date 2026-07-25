# Fine-Tuning and Classifier Evaluation

The fine-tuning stage converts manuscript pages into fixed-size crops, trains a
writer classifier, and reports both crop-level and page-level performance.

Entry point:

```text
src/fine_tuning.py
```

Run it from the repository root:

```bash
python -m src.fine_tuning [arguments]
```

## Prerequisites

The experiment must already have:

```text
metadata/<experiment_id>/general-metadata.json
experiments/<experiment_id>/
```

The selected dataset must contain train and test pages for all configured
classes.

## CLI reference

| Argument | Type | Required | Default / choices | Description |
|---|---|---:|---|---|
| `-experiment_id` | string | Yes | — | Registered experiment |
| `-steering_metric` | string | No | `loss`; choices: `loss`, `accuracy`, `macrof1`, `weightedf1` | Metric used for the best checkpoint and early stopping |
| `-ch_layers` | comma-separated string | Yes | — | Classification-head specification |
| `-crop_size` | integer | Yes | Positive | Side of source crops |
| `-batch_size` | integer | Yes | Positive | Training, validation, and test batch size |
| `-opt` | string | Yes | `SGD`, `Adam`, `AdamW` | Optimizer |
| `-weight_decay` | float | No | `0.0001` | Optimizer weight decay in `[0, 1)` |
| `-lr` | float | Yes | Positive | Initial learning rate |
| `-lr_scheduler` | string | Yes | `CosineAnnealingLR` | Learning-rate scheduler |
| `-lr_final_decay_ratio` | float | No | `0.01` | Initial cosine minimum as a fraction of `lr` |
| `-label_smoothing` | float | No | `0.0` | Cross-entropy label smoothing in `[0, 1)` |
| `-early_stopping` | boolean | No | `true` | Enable the second early-stopping phase |
| `-early_stopping_patience` | integer | Conditional | Required and positive when early stopping is enabled | Epochs without improvement after the trigger |
| `-train_replicas` | integer | No | `1` | Saved copies of every training crop |
| `-random_seed` | integer | No | Randomly generated when missing or negative | Training and augmentation seed |
| `-epochs` | integer | No | `50` | Normal training phase length |
| `-train_img_transforms` | string | Yes | `legacy`, `moderate`, `backgroundrobust`, `aggressive`, `xaggressive` | Augmentation preset |
| `-ft_mode` | string | Yes | `frozen`, `full` | Freeze the encoder or train the entire network |
| `-keep_crops` | boolean | No | `false` | Retain generated `train/`, `val/`, and `test/` crop directories |

Boolean values accept forms such as `true`, `false`, `yes`, `no`, `1`, and
`0`.

## Example

```bash
python -m src.fine_tuning \
  -experiment_id vat653_resnet_run1 \
  -steering_metric macrof1 \
  -ch_layers fc:128,bn,act:relu \
  -crop_size 380 \
  -batch_size 32 \
  -opt AdamW \
  -lr 0.0005 \
  -lr_scheduler CosineAnnealingLR \
  -lr_final_decay_ratio 0.125 \
  -weight_decay 0.01 \
  -label_smoothing 0.1 \
  -early_stopping true \
  -early_stopping_patience 10 \
  -train_replicas 1 \
  -random_seed 24 \
  -epochs 20 \
  -train_img_transforms moderate \
  -ft_mode full \
  -keep_crops false
```

## Metadata initialization

On the first run, the CLI configuration is saved in:

```text
metadata/<experiment_id>/ft-metadata.json
```

The file contains:

- `HYPERPARAMETERS`: the effective training configuration;
- `FINE_TUNING_DETAILS`: completed epoch count;
- `TIMESTAMPS`: completion markers for training and testing.

If `HYPERPARAMETERS` already exists, new CLI values do not replace it. The
stored metadata remains authoritative.

## Phase 1: dataset creation

### Crop grid

`CropRetriever` builds a crop grid independently on the horizontal and vertical
axes.

For an axis of length `dim`:

- if `dim <= crop_size`, the only start coordinate is `0`;
- otherwise, the number of crops is `ceil(dim / crop_size)`;
- crop starts are evenly distributed between `0` and
  `dim - crop_size`.

This creates overlapping crops when necessary while ensuring coverage of both
ends of the page. Pages smaller than a crop are padded on the right or bottom
by replicating edge pixels.

### Generated subsets

Training and test pages are cropped first:

```text
experiments/<experiment_id>/fine_tuning/
├── train/<class>/
├── val/<class>/
└── test/<class>/
```

Training filenames include a replica identifier:

```text
<page>_cp<replica>_crop<index>.png
```

Test filenames use:

```text
<page>_crop<index>.png
```

For each test page, at least one crop and approximately 25% of its crops are
sampled with the fixed seed `24` and copied into the validation directory.
These crops are not removed from the test directory.

The number of source crops for every train and test page is saved in:

```text
experiments/<experiment_id>/fine_tuning/n_crops_per_instance.json
```

### RGB normalization

Mean and standard deviation are computed over all pixels of all saved training
crops and cached in:

```text
experiments/<experiment_id>/fine_tuning/rgb_train_stats.pkl
```

If this file already exists, its values are reused.

## Models and classification head

### ResNet18

`ResNet18` initializes a custom feature encoder from:

```text
src/models/cp/Test_3_TL_val_best_model.pth
```

The encoder expands its 512-dimensional backbone output to 1024 features. Its
input size is `380 × 380`.

### Swin models

`SwinTiny` and `SwinSmall` use torchvision ImageNet-1K weights. Both expose a
768-dimensional feature representation and use `224 × 224` inputs.

### Fine-tuning mode

- `frozen`: encoder parameters have `requires_grad = false`; only the
  classification head is trained.
- `full`: encoder and classification head are both trainable.

### Head syntax

`-ch_layers` is parsed from left to right. Supported tokens are:

| Token | Layer |
|---|---|
| `fc:<units>` | Linear layer with Xavier initialization |
| `act:relu` | ReLU |
| `act:leakyrelu` | LeakyReLU |
| `act:gelu` | GELU |
| `act:celu` | CELU |
| `act:silu` or `act:swish` | SiLU |
| `act:mish` | Mish |
| `dropout:<p>` | Dropout |
| `bn` | Batch normalization |
| `ln` | Layer normalization |

The final class projection is appended automatically.

## Deterministic training data

The training loader precomputes a new index permutation for every epoch using
the experiment seed and epoch number. Each random transform is then seeded from
the global seed, epoch, batch position, sample position, and transform index.

The augmentation sequence can include:

- color jitter;
- affine and perspective transforms;
- Gaussian blur and grayscale conversion;
- random erasing and inversion;
- Gaussian noise;
- normalization.

Validation and test data are only resized, converted to tensors, and
normalized.

## Phase 2: model training

The loss is cross entropy with the configured label smoothing. Accuracy,
macro-F1, weighted-F1, and average loss are recorded for both training and
validation after every epoch.

On CUDA, automatic mixed precision is enabled through `torch.amp`. Multiple
GPUs are handled with `nn.DataParallel`.

### Scheduler and early stopping

The initial `CosineAnnealingLR` phase lasts `-epochs` epochs and decays toward:

```text
lr * lr_final_decay_ratio
```

Without early stopping, training ends after `-epochs`.

With early stopping:

1. training always runs through the configured `-epochs`;
2. the best validation metric is tracked during that phase;
3. the scheduler is reset after the configured epoch count;
4. training may continue up to epoch 100;
5. it stops after `early_stopping_patience` non-improving epochs.

Consequently, `-epochs` is the early-stopping trigger rather than a hard maximum
when early stopping is enabled.

### Checkpoints and resume

The trainer writes:

```text
checkpoints/val_best_model.pth
checkpoints/last_checkpoint.pth
```

Checkpoints include:

- model state;
- optimizer state;
- scheduler state;
- early-stopping state;
- AMP scaler state.

`FINE_TUNING_DETAILS.EPOCHS_COMPLETED` causes a training invocation to load
`last_checkpoint.pth` and resume at the next epoch. The entire model,
optimizer, and scheduler configuration must remain compatible with the stored
checkpoint.

When `TIMESTAMPS.MODEL_FINE_TUNING` exists, the entry point skips training
entirely.

## Phase 3: testing

The best validation checkpoint is loaded and evaluated on every test crop.
For each crop, the stage records:

- ground-truth label;
- predicted label;
- logits;
- softmax probabilities.

Crop predictions are grouped by page using `n_crops_per_instance.json`. The
page prediction is the majority class among its crops; ties follow NumPy's
lowest-index `argmax` behavior.

The stage reports accuracy, macro-F1, weighted-F1, classification reports, and
confusion matrices at both levels.

## Outputs

```text
experiments/<experiment_id>/fine_tuning/
├── checkpoints/
│   ├── last_checkpoint.pth
│   └── val_best_model.pth
├── history/
│   ├── *.pkl
│   ├── *.png
│   └── training_recap.json
├── output/
│   ├── classification_metrics.json
│   ├── confusion_matrix_crop_level.png
│   ├── confusion_matrix_page_level.png
│   ├── crop_logits.csv
│   ├── crop_probs.csv
│   └── crop_preds_per_page.csv
├── class_to_idx.json
├── n_crops_per_instance.json
└── rgb_train_stats.pkl
```

If `-keep_crops false`, the generated `train/`, `val/`, and `test/`
subdirectories are removed after testing. Persistent statistics and reports are
retained.

## Next stage

The trained checkpoint, class mapping, crop size, and RGB statistics are
required by the [explainability stage](EXPLAINABILITY.md).
