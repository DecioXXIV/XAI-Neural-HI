![Python](https://img.shields.io/badge/Python-3.11.2-3776AB?logo=python&logoColor=white)
# Page-level Explanations for Neural Handwriting Identification

This repository implements an experiment-oriented pipeline for neural writer
identification on manuscript pages. It trains a crop-level image classifier,
builds page-level explanations, evaluates whether the explanations are
faithful, and compares explanations across repeated runs or different models.

The project is organized as a set of independent command-line stages. State is
persisted through JSON metadata and artifacts on disk, so later stages consume
the outputs of earlier ones.

## Pipeline overview

```text
setup_experiment
        |
        v
   fine_tuning
        |
        v
      explain
        |
        +----------------+----------------+
        |                |                |
        v                v                v
  faithfulness       stability     cross_model_agreement
```

The comparison stages depend on completed explanations:

- **Stability** compares multiple explanation runs for the same trained model.
- **Cross-model agreement** compares explanations produced by two experiments.

## Main stages

| Stage | Entry point | Purpose | Detailed guide |
|---|---|---|---|
| Experiment setup | `src/setup_experiment.py` | Register the model, dataset, and writer classes | [Setup](docs/SETUP_EXPERIMENT.md) |
| Fine-tuning | `src/fine_tuning.py` | Create crops, train the classifier, and evaluate it | [Fine-tuning](docs/FINE_TUNING.md) |
| Explainability | `src/explain.py` | Generate crop- and page-level attribution scores | [Explainability](docs/EXPLAINABILITY.md) |
| Faithfulness | `src/faithfulness.py` | Measure performance as salient regions are removed | [Faithfulness](docs/FAITHFULNESS.md) |
| Stability | `src/stability.py` | Compare repeated explanation configurations | [Stability](docs/STABILITY.md) |
| Cross-model agreement | `src/cross_model_agreement.py` | Compare explanations from two trained experiments | [Cross-model agreement](docs/CROSS_MODEL_AGREEMENT.md) |

## Repository layout

```text
.
├── cli/
│   └── arg_parsers.py          # CLI definitions and validation
├── data/                       # Source manuscript pages
├── metadata/                   # Experiment configuration and stage status
├── experiments/                # Runtime crops, checkpoints, and results
├── scripts/                    # Maintenance and checkpoint migration tools
├── src/
│   ├── models/                 # ResNet and Swin classifiers
│   ├── explainers/             # Occlusion, LIME, and G-LIME
│   ├── maskers/                # Faithfulness masking strategies
│   └── utils/                  # Data, training, XAI, and evaluation utilities
└── docs/                       # Detailed stage documentation
```

The paths in `src/utils/constants.py` are relative to the current working
directory. Run all commands from the repository root.

## Dataset layout

Pages are loaded from:

```text
data/<dataset>/train/<class>/<page image>
data/<dataset>/test/<class>/<page image>
```

Supported datasets and class identifiers are declared in `data/__init__.py`:

| Dataset | Available classes |
|---|---|
| `Vat.lat.653` | `1`, `2`, `3`, `4` |
| `Vat.lat.5951` | `1`, `2`, `3` |
| `Vat.lat.4221` | `1`, `4`, `6`, `8` |
| `Chelles` | `01` through `10` |

## Quick start

The following commands illustrate a complete experiment using ResNet18 and
square-patch Occlusion explanations.

### 1. Register an experiment

```bash
python -m src.setup_experiment \
  -experiment_id demo_resnet \
  -model_name ResNet18 \
  -dataset Vat.lat.653 \
  -classes 1,2,3,4
```

This creates:

```text
metadata/demo_resnet/general-metadata.json
experiments/demo_resnet/
```

### 2. Fine-tune and test the classifier

```bash
python -m src.fine_tuning \
  -experiment_id demo_resnet \
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

The stage writes its main artifacts under:

```text
experiments/demo_resnet/fine_tuning/
├── checkpoints/
├── history/
└── output/
```

### 3. Generate explanations

```bash
python -m src.explain \
  -experiment_id demo_resnet \
  -xai_algorithm Occlusion \
  -xai_details base \
  -subsample test \
  -seg_type sq_patches \
  -patch_dim 32
```

This configuration is stored with the XAI entry name:

```text
sq_patches32x32-base
```

### 4. Evaluate faithfulness

```bash
python -m src.faithfulness \
  -experiment_id demo_resnet \
  -xai_algorithm Occlusion \
  -xai_entry sq_patches32x32-base \
  -mask_ceil 1.0 \
  -mask_step 0.05 \
  -mask_rule saliency \
  -patches_color green \
  -keep_test_sets false
```

For a meaningful baseline, repeat the command with `-mask_rule random` and a
fixed `-random_seed`.

## Models

| CLI name | Encoder initialization | Model input |
|---|---|---:|
| `ResNet18` | Local checkpoint at `src/models/cp/Test_3_TL_val_best_model.pth` | `380 × 380` |
| `SwinTiny` | torchvision ImageNet-1K weights | `224 × 224` |
| `SwinSmall` | torchvision ImageNet-1K weights | `224 × 224` |

The classification head is built from the comma-separated `-ch_layers`
specification. Supported tokens include:

```text
fc:<units>, act:<name>, dropout:<probability>, bn, ln
```

A final linear layer with one output per writer class is always appended.

## Explanation methods

| Method | Behavior |
|---|---|
| `Occlusion` | Removes one interpretable segment at a time and measures the target-logit change |
| `Lime` | Fits a locally weighted Ridge surrogate to random binary perturbations |
| `GLimeBinomial` | Fits an unweighted Ridge surrogate to binomially sampled perturbations |

Two segmentation families are available:

- `sq_patches`: a regular grid controlled by `-patch_dim`.
- `ink_based`: foreground-aware segments controlled by aggressiveness,
  granularity, and grouping method.

Explanations target the ground-truth writer label and are first computed for
each crop. Overlapping crop scores are then aggregated into a page-level score
for every segment.

## Metadata and runtime artifacts

Configuration and execution status are stored separately from large artifacts:

```text
metadata/<experiment_id>/
├── general-metadata.json
├── ft-metadata.json
├── xai-metadata.json
└── faithfulness-metadata.json
```

```text
experiments/<experiment_id>/
├── fine_tuning/
├── xai/
├── masked_images/
├── faithfulness/
├── stability/
├── cross_model_agreement/
└── retraining/
```

Several stages are resumable. Fine-tuning uses checkpoints and
`FINE_TUNING_DETAILS.EPOCHS_COMPLETED`; completed stages use timestamps or
existing result files to skip work. Once a metadata configuration exists,
rerunning a command with different CLI values does not necessarily replace the
stored configuration. Use a new experiment or XAI entry when changing an
experimental condition.

## Reproducibility

The training entry point enables deterministic PyTorch algorithms and the
training data loader derives epoch-, batch-, sample-, and transform-specific
seeds from `-random_seed`. Random faithfulness baselines use a deterministic
per-instance seed derived from their global seed.

LIME and G-LIME perturbation generation currently has no CLI seed, so separate
explanation runs are stochastic even when model inference is deterministic.

## Current implementation constraints

- The validation set is copied from test-page crops; those crops remain in the
  test set.
- Faithfulness page-level aggregation assumes that the explained test
  instances match the fine-tuning test-page inventory. Explaining the complete
  test split is the supported workflow.
- `-save_samples` in the explainability CLI is accepted but does not currently
  save perturbed images.
- `-keep_masked_pages` in the faithfulness CLI is accepted but masked pages are
  currently retained regardless of its value.
- LIME and G-LIME operationally require a positive `-kernel_width`, even though
  the parser permits it to be omitted.

See the stage guides in [`docs/`](docs/) for complete CLI tables, algorithms,
directory layouts, and resume behavior.
