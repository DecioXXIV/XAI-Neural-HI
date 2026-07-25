# Experiment Setup

The setup stage registers the immutable identity of an experiment: its model
architecture, source dataset, and writer classes. It does not load images or
instantiate a neural network.

Entry point:

```text
src/setup_experiment.py
```

Run it from the repository root:

```bash
python -m src.setup_experiment [arguments]
```

## Inputs and prerequisites

The source dataset must already follow the expected directory structure:

```text
data/<dataset>/train/<class>/
data/<dataset>/test/<class>/
```

Dataset names and valid class identifiers are defined in `data/__init__.py`.
The setup command rejects any class that is not registered for the selected
dataset.

## CLI reference

| Argument | Type | Required | Allowed values | Description |
|---|---|---:|---|---|
| `-experiment_id` | string | Yes | Any unique directory-safe identifier | Name used under `metadata/` and `experiments/` |
| `-model_name` | string | Yes | `ResNet18`, `SwinTiny`, `SwinSmall` | Classifier architecture |
| `-dataset` | string | Yes | `Vat.lat.653`, `Vat.lat.5951`, `Vat.lat.4221`, `Chelles` | Source dataset |
| `-classes` | comma-separated string | Yes | Dataset-specific class identifiers | Writers included in the classification task |

Do not insert spaces in `-classes`:

```text
1,2,3,4
```

## Example

```bash
python -m src.setup_experiment \
  -experiment_id vat653_resnet_run1 \
  -model_name ResNet18 \
  -dataset Vat.lat.653 \
  -classes 1,2,3,4
```

## Execution flow

1. `cli.arg_parsers.get_setup_exp_args()` parses the command.
2. The comma-separated class string is converted to a list.
3. Every class is checked against `SCRIBES_TO_DATASET`.
4. `create_exp_metadata()` creates the experiment metadata directory.
5. The general metadata is saved.
6. The runtime experiment directory is created.

The resulting metadata has this shape:

```json
{
  "EXPERIMENT_ID": "vat653_resnet_run1",
  "MODEL_NAME": "ResNet18",
  "DATASET": "Vat.lat.653",
  "CLASSES": ["1", "2", "3", "4"]
}
```

## Outputs

```text
metadata/<experiment_id>/general-metadata.json
experiments/<experiment_id>/
```

The general metadata is consumed by every subsequent stage. In particular:

- fine-tuning uses the model, dataset, and classes;
- explainability uses the dataset and class-to-index mapping;
- faithfulness uses the test split and classes;
- comparison stages use the experiment identifier to locate explanations.

## Uniqueness and failure behavior

If `metadata/<experiment_id>/general-metadata.json` already exists, setup logs a
warning and exits without overwriting it. An experiment identifier should
therefore describe one fixed combination of:

- model architecture;
- dataset;
- class set.

To change any of these fields, create a new experiment identifier. Removing or
editing the metadata manually can make existing checkpoints and artifacts
inconsistent with the declared experiment.

## Next stage

After setup, run the [fine-tuning stage](FINE_TUNING.md). It creates
`ft-metadata.json`, the cropped datasets, trained checkpoints, and evaluation
reports required by the rest of the pipeline.
