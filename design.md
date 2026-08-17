# Lambda-Psi Experiment Codebase Design

## 1. Purpose

This codebase evaluates the Lambda-Psi Neuron Field formulation with two base neuron families—traditional affine neurons and APTx Neurons—while keeping ReLU placement explicit. The experimental design compares ten architectures over MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100, and California Housing, using multiple initial learning rates and multiple random seeds.

The modularization does **not** change the scientific experiment. It separates the previous monolithic `experiment.py` into modules with clear responsibilities:

```text
project/
├── lambda_psi_neuron_fields/
│   └── __init__.py       # APTx primitives + Lambda-Psi mathematical layer
├── models.py             # the ten model variants and network composition
├── utils.py              # data, reproducibility, training, metrics, CSV summaries
├── run.py                # CLI, one-run lifecycle, and experiment-grid orchestration
├── run.sh                # optional single shell entry point
├── design.md             # this document
└── requirements.txt
```

## 2. Dependency direction

The dependency direction is intentionally one-way:

```text
lambda_psi_neuron_fields/__init__.py
             │
             ▼
         models.py
             │
             ├──────────────┐
             ▼              ▼
          run.py  ◄────── utils.py
```

More precisely:

- `lambda_psi_neuron_fields/__init__.py` is the reusable mathematical package. It does not depend on the experiment code.
- `models.py` imports `aptx_neuron_layer` and `lambda_psi_field_layer` from that package.
- `utils.py` contains generic experiment machinery and does not depend on `models.py`.
- `run.py` imports the architecture from `models.py` and the experiment utilities from `utils.py`.

This avoids circular imports and keeps the paper's field equation in one implementation.

## 3. `lambda_psi_neuron_fields/__init__.py`

The package defines the APTx activation/neuron primitives and the neuron-agnostic Lambda-Psi field.

### 3.1 APTx primitives

The package contains:

- `aptx_activation_function`
- `aptx_neuron`
- `aptx_neuron_layer`

The vectorized `aptx_neuron_layer` produces the pre-field neuron outputs used by the experiment.

### 3.2 Generic Lambda-Psi field

`lambda_psi_field_layer` accepts a 2-D tensor of pre-field neuron outputs `y` and applies the paper equation:

\[
m_j = \lambda\left[(1-\psi)\sigma(y_j) + \psi\,\mathrm{softmax}(\mathbf{y})_j\right]
+ (1-\lambda)\left[(1-\psi)y_j + \psi\bar{y}\right].
\]

Its four fixed boundary regimes are:

| Lambda | Psi | Behavior |
|---:|---:|---|
| 0 | 0 | Identity |
| 0 | 1 | Global Context / layer mean |
| 1 | 0 | Sigmoid |
| 1 | 1 | Softmax |

For a trainable field, raw scalar parameters are passed through sigmoid so learned Lambda and Psi remain in `[0, 1]`. ReLU is not embedded inside the field, which is important for distinguishing `field -> ReLU` from `ReLU -> field`.

### 3.3 Renamed composite layer classes

The updated package names are:

- `traditional_neuron_lambda_psi_field_layer`
- `aptx_neuron_lambda_psi_field_layer`

These names make the composition explicit. The experiment itself imports the generic `lambda_psi_field_layer` and composes it externally, because the ten ablations require control over ReLU placement. Therefore the renamed composite classes do not break the experiment runner.

## 4. `models.py`

`models.py` contains architecture definitions only.

### 4.1 `VARIANT_SPECS`

This dictionary is the single source of truth for all ten comparisons:

| Variant key | Hidden-block execution |
|---|---|
| `traditional` | Traditional neuron |
| `traditional_relu` | Traditional neuron -> ReLU |
| `traditional_field` | Traditional neuron -> Lambda-Psi |
| `traditional_field_relu` | Traditional neuron -> Lambda-Psi -> ReLU |
| `traditional_relu_field` | Traditional neuron -> ReLU -> Lambda-Psi |
| `aptx` | APTx Neuron |
| `aptx_relu` | APTx Neuron -> ReLU |
| `aptx_field` | APTx Neuron -> Lambda-Psi |
| `aptx_field_relu` | APTx Neuron -> Lambda-Psi -> ReLU |
| `aptx_relu_field` | APTx Neuron -> ReLU -> Lambda-Psi |

### 4.2 `build_base_neuron()`

This factory creates one transformation layer:

- `traditional` -> `nn.Linear`
- `aptx` -> `aptx_neuron_layer`

This is where the base neuron family changes; the remaining network composition is shared.

### 4.3 `HiddenBlock`

`HiddenBlock` performs one hidden-layer transformation. Its forward flow begins with:

```text
input -> base neuron -> y
```

Then the `relu_position` and `use_field` settings determine what happens to `y`:

```text
after_neuron: y -> ReLU
before_field: y -> ReLU -> Lambda-Psi
after_field:  y -> Lambda-Psi -> ReLU
none + field: y -> Lambda-Psi
none:         y
```

No ordering is implicit. This is essential for a clean ablation study.

### 4.4 `ComparisonNetwork`

The network stacks three hidden blocks:

```text
input
  -> hidden block 1
  -> hidden block 2
  -> hidden block 3
  -> output base neuron
  -> task output
```

By default, the output head is the corresponding base neuron and has no field. This keeps the primary comparison focused on hidden representation layers and gives `CrossEntropyLoss` raw class scores and California Housing an unrestricted continuous scalar prediction.

`--field-on-output` is retained as an explicit additional ablation. For a field variant only, it adds:

```text
output base neuron -> Lambda-Psi field
```

`field_values()` reads the learned Lambda/Psi values from every field-bearing hidden layer and, when enabled, the output field.

## 5. `utils.py`

`utils.py` contains reusable experiment infrastructure.

### 5.1 Dataset configuration

`DATASET_CONFIGS` defines task type and default hyperparameters for each dataset:

- input/output dimensions
- normalization values for image datasets
- validation size
- epochs
- batch size
- three hidden-layer widths
- early-stopping patience
- weight decay
- label smoothing

`RunSettings` represents one exact grid entry and carries both fixed defaults and optional CLI overrides.

### 5.2 Reproducibility

Two seeds intentionally serve different purposes:

- `split_seed`: fixes train/validation/test partitioning.
- `seed`: controls model initialization, training RNG, and training DataLoader shuffle order.

This means every architecture, learning rate, and model seed is compared on the same underlying data split while still measuring stochastic training variation.

`set_seed()` sets Python, NumPy, PyTorch, and CUDA RNGs and requests deterministic cuDNN behavior.

`seed_worker()` initializes DataLoader worker-side Python/NumPy randomness from PyTorch's deterministic worker seed.

### 5.3 Run identity and resume safety

Each run ID contains dataset, variant, seed, LR, and a configuration hash.

The configuration hash includes important non-seed/non-LR settings such as:

- split seed
- field-on-output flag
- APTx alpha trainability
- delta/bias setting
- Lambda/Psi initial values
- gradient clipping
- early-stopping threshold
- AMP state
- workers/backend
- resolved training hyperparameters
- a source-code fingerprint

After modularization, `source_fingerprint()` hashes:

```text
utils.py
models.py
run.py
lambda_psi_neuron_fields/__init__.py
```

This is an intentional adaptation of the previous single-file fingerprint. Without it, resume mode could mistakenly treat results produced by old module code as current results.

### 5.4 Classification data flow

For MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100:

```text
official training set
  -> deterministic index split
     -> training subset with augmentation
     -> validation subset with evaluation transform

official test set
  -> evaluation transform only
```

MNIST/Fashion-MNIST training uses random crop. CIFAR additionally uses random horizontal flip. Validation and test sets use normalization only.

The official test set is not used for early stopping or LR selection.

### 5.5 California Housing data flow

California Housing follows:

```text
full data
  -> deterministic train+validation / test split
  -> deterministic train / validation split
```

Only the features `X` are standardized using a scaler fit on the training subset. The target `y` is deliberately left in the dataset's original units.

Therefore reported MSE, RMSE, MAE, and R2 correspond to unnormalized target values. The CSV additionally multiplies RMSE and MAE by 100,000 for convenience because the dataset target is conventionally expressed in units of 100,000 USD.

### 5.6 Metrics

Classification accumulates across the full loader:

- mean loss
- top-1 accuracy
- top-5 accuracy

Regression accumulates sufficient statistics across the full loader:

- SSE
- SAE
- target sum
- squared-target sum

The final metrics are then calculated once over the entire dataset:

- MSE
- RMSE
- MAE
- R2

This avoids the statistical error of averaging independently calculated per-batch RMSE or R2 values.

### 5.7 Training and evaluation

`train_one_epoch()` performs:

```text
batch
 -> device
 -> zero gradients
 -> forward
 -> loss
 -> finite-loss check
 -> backward
 -> optional gradient clipping
 -> optimizer step
 -> metric accumulation
```

If AMP is active on CUDA, gradient scaling/autocast is used. A non-finite loss raises before the optimizer step.

`evaluate()` performs the same forward/metric path under `torch.no_grad()` without updating model parameters.

### 5.8 Optimizer design

The optimizer is AdamW, but Lambda/Psi raw field parameters are placed in a zero-weight-decay parameter group.

This matters because:

```text
raw = 0 -> sigmoid(raw) = 0.5
```

Applying weight decay to `lambda_raw` and `psi_raw` would pull the fields toward 0.5 and impose an unintended prior on the learned field regime. Base-neuron parameters retain the configured weight decay.

### 5.9 CSV persistence and summaries

Every finished run is appended immediately to `all_runs.csv`, making long experiment grids recoverable after interruption.

The utilities generate:

```text
all_runs.csv
summary_by_lr.csv
selected_lr_summary.csv
```

`summary_by_lr.csv` groups by dataset + variant + configuration + LR and reports mean/std across seeds.

`selected_lr_summary.csv` is produced only after every requested LR has a complete requested seed set. LR selection uses validation data only:

- classification: maximize mean best-validation accuracy
- regression: minimize mean best-validation RMSE

Test metrics never participate in LR selection.

## 6. `run.py`

`run.py` is the orchestration layer.

### 6.1 `run_single_experiment()`

One call corresponds to one exact:

```text
dataset + model variant + training seed + initial learning rate
```

Its lifecycle is:

```text
resolve settings
  -> set training seed
  -> choose CPU/CUDA
  -> build DataLoaders
  -> construct ComparisonNetwork
  -> count parameters
  -> build criterion / AdamW / CosineAnnealingLR / optional AMP
  -> build run ID and checkpoint path
  -> train epoch
  -> validate epoch
  -> select best checkpoint from validation metric
  -> early stop when patience is exhausted
  -> restore best checkpoint
  -> re-evaluate validation with restored checkpoint
  -> evaluate official test set
  -> read learned Lambda/Psi values
  -> return one CSV-ready dictionary
```

Checkpoint selection is:

- classification: higher validation accuracy is better
- regression: lower validation RMSE is better

The test set is evaluated only after the selected checkpoint is restored.

### 6.2 Grid construction

Default values are:

```text
Datasets: 5
Variants: 10
Learning rates: 5
Seeds: 5
```

Thus the complete default grid contains:

```text
5 x 10 x 5 x 5 = 1250 runs
```

Duplicate seed/LR arguments are removed before the grid is constructed.

### 6.3 Resume behavior

Unless `--no-resume` is supplied, successful run IDs already present in `all_runs.csv` are skipped.

Failed runs are recorded with `status=error`, so failures remain inspectable but can be retried later because only successful run IDs count as complete.

### 6.4 Failure handling

A failed run records an error row and the grid continues. `--fail-fast` changes this behavior and raises immediately after the first failed run.

## 7. Primary output files

For the default results directory:

```text
results/lambda_psi_comparison/
├── all_runs.csv
├── summary_by_lr.csv
├── selected_lr_summary.csv
└── checkpoints/
    └── <dataset>/<variant>/<run-id>.pt
```

`all_runs.csv` contains reproducibility metadata, configuration, parameter counts, runtime, best validation metrics, final test metrics, and learned Lambda/Psi values.

## 8. Execution examples

The simplest direct Python command is:

```bash
python3 run.py --download-dataset
```

The single shell wrapper is equivalent to calling `run.py` and forwards every argument:

```bash
./run.sh --download-dataset
```

A one-run smoke test:

```bash
./run.sh \
  --download-dataset \
  --datasets mnist \
  --variants traditional_field \
  --seeds 0 \
  --lrs 1e-3 \
  --epochs 2 \
  --patience 2 \
  --max-runs 1
```

MNIST across all ten variants for one seed/LR:

```bash
./run.sh \
  --download-dataset \
  --datasets mnist \
  --seeds 0 \
  --lrs 1e-3
```

Full default grid:

```bash
./run.sh --download-dataset
```

## 9. Scientific separation of responsibilities

The important architectural principle is that the code mirrors the paper's two-stage idea:

```text
x -> base neuron -> y -> optional Lambda-Psi -> m
```

ReLU is an explicit experiment operator around this sequence rather than hidden inside the field implementation. Consequently, `Neuron -> Field -> ReLU` and `Neuron -> ReLU -> Field` are truly different computational graphs.

The field remains neuron-agnostic: the same `lambda_psi_field_layer` receives `y` whether `y` is generated by `nn.Linear` or by `aptx_neuron_layer`.

## 10. Notes on the updated package names

The updated `__init__.py` is internally consistent and the renamed composite layer classes are exported in `__all__`. The modular experiment does not depend on the old composite names, so no compatibility problem is introduced by the rename.

The experimental code intentionally uses `aptx_neuron_layer` + `lambda_psi_field_layer` as separate components because this is required to express all ten ReLU/field orderings without embedding extra activation behavior inside package classes.
