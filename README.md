# Lambda-Psi Neuron Field Experiments

This repository contains the experimental code for evaluating Lambda-Psi Neuron Fields with both traditional affine neurons and APTx Neurons across classification and regression benchmarks.

The code is intentionally organized into a small set of modules so that model definitions, experiment utilities, and experiment execution remain easy to inspect independently.

## Project structure

```text
lambda_psi_modular_codebase/
├── lambda_psi_neuron_fields/
│   └── __init__.py
├── models.py
├── utils.py
├── run.py
├── run.sh
├── design.md
├── requirements.txt
└── README.md
```

### `lambda_psi_neuron_fields/__init__.py`

Contains the reusable neural primitives:

- APTx activation function
- APTx Neuron
- Vectorized APTx Neuron layer
- Generic Lambda-Psi field layer
- Traditional Neuron + Lambda-Psi field layer
- APTx Neuron + Lambda-Psi field layer

The generic Lambda-Psi field operates on the pre-field neuron outputs `y` and does not contain an internal ReLU. This keeps the field independent of the base neuron and allows activation-order experiments to remain explicit.

### `models.py`

Contains the model definitions used in the comparison study:

- architecture specifications for all comparison variants
- construction of traditional and APTx base-neuron layers
- `HiddenBlock`
- `ComparisonNetwork`
- extraction of learned Lambda/Psi values

### `utils.py`

Contains the reusable experiment infrastructure:

- dataset configurations
- run settings
- deterministic seeding
- train/validation/test data loading
- California Housing preprocessing
- classification and regression metrics
- training and evaluation functions
- AdamW optimizer construction
- run/config fingerprinting
- CSV result storage
- mean/std aggregation across seeds
- validation-based learning-rate selection

### `run.py`

Contains the executable experiment workflow:

1. Parse command-line arguments.
2. Build the dataset × variant × learning-rate × seed grid.
3. Run each experiment.
4. Select checkpoints using validation performance.
5. Restore the best checkpoint.
6. Evaluate the final test set.
7. Store run-level metrics and learned Lambda/Psi values.
8. Aggregate results across seeds.
9. Select the best learning rate using validation metrics only.

### `run.sh`

A minimal shell entry point:

```bash
./run.sh [arguments]
```

It forwards all arguments directly to `run.py`.

### `design.md`

Provides a detailed description of the software architecture, experiment lifecycle, data flow, model construction, metrics, checkpointing, reproducibility strategy, and result aggregation.

## Lambda-Psi field

For a vector of pre-field neuron outputs `y`, the field computes

```text
m_j = lambda * [
          (1 - psi) * sigmoid(y_j)
          + psi * softmax(y)_j
      ]
      + (1 - lambda) * [
          (1 - psi) * y_j
          + psi * mean(y)
      ]
```

The four boundary regimes are:

| lambda | psi | Field behaviour |
|---:|---:|---|
| 0 | 0 | Identity |
| 0 | 1 | Global Context / mean |
| 1 | 0 | Sigmoid |
| 1 | 1 | Softmax |

For trainable fields, Lambda and Psi are represented by unconstrained raw parameters and mapped to `(0, 1)` with the sigmoid function.

## Comparison variants

The experiment defines ten architectures:

| Variant | Hidden-block operation |
|---|---|
| `traditional` | Traditional neuron |
| `traditional_relu` | Traditional neuron → ReLU |
| `traditional_field` | Traditional neuron → Lambda-Psi |
| `traditional_field_relu` | Traditional neuron → Lambda-Psi → ReLU |
| `traditional_relu_field` | Traditional neuron → ReLU → Lambda-Psi |
| `aptx` | APTx Neuron |
| `aptx_relu` | APTx Neuron → ReLU |
| `aptx_field` | APTx Neuron → Lambda-Psi |
| `aptx_field_relu` | APTx Neuron → Lambda-Psi → ReLU |
| `aptx_relu_field` | APTx Neuron → ReLU → Lambda-Psi |

By default, the selected ordering is applied to the three hidden blocks and the output head remains the corresponding base-neuron layer. `--field-on-output` enables an additional field on the output head for field variants.

## Datasets

The default experiment includes:

- MNIST
- Fashion-MNIST
- CIFAR-10
- CIFAR-100
- California Housing

For classification datasets, the official training set is divided deterministically into training and validation subsets. The official test set is retained for final evaluation.

For California Housing, only the input features are standardized. The target is left in the dataset's original scale, so MSE, RMSE, MAE, and R² are computed on the original target values. Convenience RMSE/MAE columns in US-dollar units are also recorded.

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The required packages are:

```text
torch
torchvision
numpy
scikit-learn
```

## Running experiments

Make the shell script executable if required:

```bash
chmod +x run.sh
```

Run the full default grid and allow missing datasets to be downloaded:

```bash
./run.sh --download-dataset
```

The default grid contains:

```text
5 datasets × 10 variants × 5 learning rates × 5 seeds = 1250 runs
```

Default training seeds:

```text
0 1 2 3 4
```

Default initial learning rates:

```text
0.0003 0.0005 0.001 0.002 0.003
```

## Useful examples

Run only MNIST:

```bash
./run.sh --datasets mnist --download-dataset
```

Run selected variants:

```bash
./run.sh \
  --datasets mnist \
  --variants traditional traditional_field aptx aptx_field \
  --download-dataset
```

Run one seed and one learning rate:

```bash
./run.sh \
  --datasets mnist \
  --variants aptx_field \
  --seeds 0 \
  --lrs 1e-3 \
  --download-dataset
```

Run a short smoke test:

```bash
./run.sh \
  --datasets mnist \
  --variants traditional_field aptx_field \
  --seeds 0 \
  --lrs 1e-3 \
  --epochs 2 \
  --patience 2 \
  --max-runs 2 \
  --download-dataset
```

Apply the Lambda-Psi field to the output head as an explicit ablation:

```bash
./run.sh --field-on-output --download-dataset
```

Use fixed APTx alpha values:

```bash
./run.sh --no-alpha-trainable --download-dataset
```

Disable APTx delta / traditional-neuron bias:

```bash
./run.sh --no-delta --download-dataset
```

Use automatic mixed precision on CUDA:

```bash
./run.sh --amp --download-dataset
```

See all available command-line options:

```bash
python3 run.py --help
```

## Experiment protocol

Each individual run corresponds to one:

```text
dataset + variant + seed + initial learning rate + experiment configuration
```

During training:

- classification models are selected by validation accuracy;
- regression models are selected by validation RMSE;
- early stopping uses the configured validation metric and patience;
- AdamW is used as the optimizer;
- Lambda/Psi raw parameters are excluded from weight decay so weight decay does not implicitly push them toward `0.5`;
- a cosine-annealing learning-rate scheduler is used;
- optional gradient clipping is supported;
- optional AMP is supported on CUDA.

After training, the best validation checkpoint is restored before the validation and test metrics are recorded.

The test set is not used to select the checkpoint or the learning rate.

## Reproducibility

The experiment separates two kinds of randomness:

- `split_seed` controls dataset splitting;
- `seed` controls model initialization, training randomness, and training-loader shuffling.

This allows all model variants and learning rates to use the same validation split while still measuring variability across independent training seeds.

Each run ID contains a configuration hash. The hash includes the experiment settings and a fingerprint of the scientific source files, which prevents resume mode from silently treating results produced by different source/configuration states as the same experiment.

## Result files

By default results are written to:

```text
./results/lambda_psi_comparison/
```

The main files are:

### `all_runs.csv`

Contains one row for every individual run, including:

- dataset and architecture
- seed and initial learning rate
- hyperparameters
- parameter counts
- best epoch
- validation metrics
- final test metrics
- training duration
- checkpoint path
- learned Lambda/Psi values for every field layer
- software versions and source/configuration fingerprints

### `summary_by_lr.csv`

Groups runs by dataset, architecture, configuration, and learning rate, then reports mean and standard deviation across seeds.

### `selected_lr_summary.csv`

Contains the selected learning rate for each complete experimental condition.

Learning-rate selection uses only validation performance:

- maximum mean validation accuracy for classification;
- minimum mean validation RMSE for regression.

A selected learning rate is written only after all requested learning rates have complete seed sets.

### `checkpoints/`

Stores the best validation checkpoint for each run.

## Reading the code

A convenient reading order is:

```text
lambda_psi_neuron_fields/__init__.py
        ↓
models.py
        ↓
utils.py
        ↓
run.py
        ↓
design.md
```

This follows the same conceptual progression as execution: mathematical primitives → model composition → experiment utilities → experiment orchestration → detailed architecture documentation.
