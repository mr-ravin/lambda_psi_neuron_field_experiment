#!/usr/bin/env python3
"""Generate the CSV data tables used in the Lambda-Psi paper.

Expected repository layout
--------------------------
results/
├── lambda_psi_comparison/
│   ├── all_runs.csv
│   └── selected_lr_summary.csv
└── lambda_psi_hidden_only/
    ├── all_runs.csv
    └── selected_lr_summary.csv

The script writes table1.csv ... table8.csv in paper_tables/ by default.
The numbering follows the order of tables in the manuscript:

  table1.csv  Boundary regimes of the Lambda-Psi field
  table2.csv  Dataset-specific network/training configuration
  table3.csv  Architectural variants
  table4.csv  Primary classification results
  table5.csv  Primary California Housing regression results
  table6.csv  Hidden-only ablation for APTx Neuron + Field
  table7.csv  Learned hidden-field Lambda/Psi parameters
  table8.csv  Learned output-field Lambda/Psi parameters

Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


DATASET_ORDER = [
    "mnist",
    "fashion_mnist",
    "cifar10",
    "cifar100",
    "california_housing",
]

DATASET_LABELS = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "cifar10": "CIFAR-10",
    "cifar100": "CIFAR-100",
    "california_housing": "California Housing",
}

CLASSIFICATION_DATASETS = DATASET_ORDER[:4]

VARIANT_ORDER = [
    "traditional",
    "traditional_relu",
    "traditional_field",
    "traditional_field_relu",
    "traditional_relu_field",
    "aptx",
    "aptx_relu",
    "aptx_field",
    "aptx_field_relu",
    "aptx_relu_field",
]

VARIANT_LABELS = {
    "traditional": "Traditional",
    "traditional_relu": "Traditional + ReLU",
    "traditional_field": "Traditional + Field",
    "traditional_field_relu": "Traditional + Field + ReLU",
    "traditional_relu_field": "Traditional + ReLU + Field",
    "aptx": "APTx Neuron",
    "aptx_relu": "APTx Neuron + ReLU",
    "aptx_field": "APTx Neuron + Field",
    "aptx_field_relu": "APTx Neuron + Field + ReLU",
    "aptx_relu_field": "APTx Neuron + ReLU + Field",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate table1.csv ... table8.csv for the Lambda-Psi paper."
    )
    parser.add_argument(
        "--primary-dir",
        type=Path,
        default=Path("results/lambda_psi_comparison"),
        help="Primary hidden+output result directory.",
    )
    parser.add_argument(
        "--hidden-only-dir",
        type=Path,
        default=Path("results/lambda_psi_hidden_only"),
        help="Hidden-only ablation result directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_tables"),
        help="Directory in which table1.csv ... table8.csv are written.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def as_float(row: Mapping[str, str], key: str) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        raise ValueError(f"Missing numeric value for column '{key}' in row: {row}")
    return float(value)


def as_int(row: Mapping[str, str], key: str) -> int:
    return int(round(as_float(row, key)))


def fmt_number(value: float, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def fmt_pm(row: Mapping[str, str], mean_key: str, std_key: str, decimals: int) -> str:
    mean = as_float(row, mean_key)
    std = as_float(row, std_key)
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def index_selected(rows: Sequence[Mapping[str, str]]) -> Dict[tuple[str, str], Mapping[str, str]]:
    out: Dict[tuple[str, str], Mapping[str, str]] = {}
    for row in rows:
        key = (row["dataset"], row["variant"])
        if key in out:
            raise ValueError(f"Duplicate selected-LR row for {key}")
        if row.get("complete_seed_set") not in ("1", "1.0", "True", "true"):
            raise ValueError(f"Incomplete seed set for selected-LR row {key}")
        out[key] = row
    return out


def require_row(index: Mapping[tuple[str, str], Mapping[str, str]], dataset: str, variant: str) -> Mapping[str, str]:
    key = (dataset, variant)
    if key not in index:
        raise KeyError(f"Missing selected-LR result for dataset={dataset}, variant={variant}")
    return index[key]


def unique_value(rows: Sequence[Mapping[str, str]], dataset: str, key: str) -> str:
    values = {row.get(key, "") for row in rows if row.get("dataset") == dataset and row.get("status") == "ok"}
    values.discard("")
    if len(values) != 1:
        raise ValueError(f"Expected one unique '{key}' for {dataset}, found: {sorted(values)}")
    return next(iter(values))


def validate_primary(primary_all: Sequence[Mapping[str, str]], primary_selected: Sequence[Mapping[str, str]]) -> None:
    ok_rows = [r for r in primary_all if r.get("status") == "ok"]
    if len(ok_rows) != 1250:
        raise ValueError(f"Primary experiment should contain 1250 successful runs, found {len(ok_rows)}")

    grid = {(r["dataset"], r["variant"], r["lr_initial"], r["seed"]) for r in ok_rows}
    if len(grid) != 1250:
        raise ValueError("Primary experiment contains duplicate dataset/variant/LR/seed runs")

    selected_index = index_selected(primary_selected)
    expected = {(d, v) for d in DATASET_ORDER for v in VARIANT_ORDER}
    missing = sorted(expected - set(selected_index))
    if missing:
        raise ValueError(f"Primary selected-LR summary is missing conditions: {missing}")


def validate_hidden(hidden_all: Sequence[Mapping[str, str]], hidden_selected: Sequence[Mapping[str, str]]) -> None:
    ok_rows = [r for r in hidden_all if r.get("status") == "ok"]
    if len(ok_rows) != 750:
        raise ValueError(f"Hidden-only experiment should contain 750 successful runs, found {len(ok_rows)}")

    if any(r.get("field_on_output") not in ("0", "0.0", "False", "false") for r in ok_rows):
        raise ValueError("Hidden-only results contain a run with field_on_output enabled")

    field_variants = [v for v in VARIANT_ORDER if "field" in v]
    selected_index = index_selected(hidden_selected)
    expected = {(d, v) for d in DATASET_ORDER for v in field_variants}
    missing = sorted(expected - set(selected_index))
    if missing:
        raise ValueError(f"Hidden-only selected-LR summary is missing conditions: {missing}")


def make_table1() -> tuple[List[str], List[Dict[str, object]]]:
    fields = ["lambda", "psi", "resulting_equation", "regime"]
    rows = [
        {"lambda": 0, "psi": 0, "resulting_equation": "m_j = y_j", "regime": "Identity"},
        {"lambda": 0, "psi": 1, "resulting_equation": "m_j = mean(y)", "regime": "Global Context"},
        {"lambda": 1, "psi": 0, "resulting_equation": "m_j = sigmoid(y_j)", "regime": "Sigmoid"},
        {"lambda": 1, "psi": 1, "resulting_equation": "m_j = softmax(y)_j", "regime": "Softmax"},
    ]
    return fields, rows


def make_table2(primary_all: Sequence[Mapping[str, str]]) -> tuple[List[str], List[Dict[str, object]]]:
    fields = ["Dataset", "Hidden dimensions", "Max epochs", "Batch", "Patience", "Weight decay"]
    rows: List[Dict[str, object]] = []
    for dataset in DATASET_ORDER:
        h1 = int(float(unique_value(primary_all, dataset, "hidden_1")))
        h2 = int(float(unique_value(primary_all, dataset, "hidden_2")))
        h3 = int(float(unique_value(primary_all, dataset, "hidden_3")))
        rows.append(
            {
                "Dataset": DATASET_LABELS[dataset],
                "Hidden dimensions": f"{h1}-{h2}-{h3}",
                "Max epochs": int(float(unique_value(primary_all, dataset, "epochs_requested"))),
                "Batch": int(float(unique_value(primary_all, dataset, "batch_size"))),
                "Patience": int(float(unique_value(primary_all, dataset, "patience"))),
                "Weight decay": unique_value(primary_all, dataset, "weight_decay"),
            }
        )
    return fields, rows


def hidden_operation(row: Mapping[str, str]) -> str:
    base = "Traditional neuron" if row["base_neuron"] == "traditional" else "APTx Neuron"
    uses_field = row.get("uses_field") in ("1", "1.0", "True", "true")
    relu_position = row.get("relu_position", "none")

    if not uses_field:
        if relu_position == "after_neuron":
            return f"{base} -> ReLU"
        return base

    if relu_position == "after_field":
        return f"{base} -> Field -> ReLU"
    if relu_position == "before_field":
        return f"{base} -> ReLU -> Field"
    return f"{base} -> Field"


def make_table3(primary_all: Sequence[Mapping[str, str]]) -> tuple[List[str], List[Dict[str, object]]]:
    fields = ["Variant", "Hidden-block operation"]
    first_by_variant: Dict[str, Mapping[str, str]] = {}
    for row in primary_all:
        if row.get("status") == "ok" and row["variant"] not in first_by_variant:
            first_by_variant[row["variant"]] = row

    rows = []
    for variant in VARIANT_ORDER:
        row = first_by_variant.get(variant)
        if row is None:
            raise KeyError(f"Variant '{variant}' not found in primary all_runs.csv")
        rows.append(
            {
                "Variant": VARIANT_LABELS[variant],
                "Hidden-block operation": hidden_operation(row),
            }
        )
    return fields, rows


def make_table4(primary_index: Mapping[tuple[str, str], Mapping[str, str]]) -> tuple[List[str], List[Dict[str, object]]]:
    fields = ["Architecture"] + [DATASET_LABELS[d] for d in CLASSIFICATION_DATASETS]
    rows: List[Dict[str, object]] = []
    for variant in VARIANT_ORDER:
        out: Dict[str, object] = {"Architecture": VARIANT_LABELS[variant]}
        for dataset in CLASSIFICATION_DATASETS:
            row = require_row(primary_index, dataset, variant)
            out[DATASET_LABELS[dataset]] = fmt_pm(
                row, "test_accuracy_pct_mean", "test_accuracy_pct_std", 2
            )
        rows.append(out)
    return fields, rows


def make_table5(primary_index: Mapping[tuple[str, str], Mapping[str, str]]) -> tuple[List[str], List[Dict[str, object]]]:
    fields = ["Architecture", "Selected LR", "RMSE", "MAE", "R2"]
    rows: List[Dict[str, object]] = []
    for variant in VARIANT_ORDER:
        row = require_row(primary_index, "california_housing", variant)
        rows.append(
            {
                "Architecture": VARIANT_LABELS[variant],
                "Selected LR": fmt_number(as_float(row, "selected_lr"), 4).rstrip("0").rstrip("."),
                "RMSE": fmt_pm(row, "test_rmse_mean", "test_rmse_std", 4),
                "MAE": fmt_pm(row, "test_mae_mean", "test_mae_std", 4),
                "R2": fmt_pm(row, "test_r2_mean", "test_r2_std", 4),
            }
        )
    return fields, rows


def make_table6(
    primary_index: Mapping[tuple[str, str], Mapping[str, str]],
    hidden_index: Mapping[tuple[str, str], Mapping[str, str]],
) -> tuple[List[str], List[Dict[str, object]]]:
    fields = ["Dataset / metric", "Primary: hidden + output", "Ablation: hidden only"]
    rows: List[Dict[str, object]] = []
    for dataset in DATASET_ORDER:
        p = require_row(primary_index, dataset, "aptx_field")
        h = require_row(hidden_index, dataset, "aptx_field")
        if dataset == "california_housing":
            label = "California Housing RMSE"
            primary_value = fmt_pm(p, "test_rmse_mean", "test_rmse_std", 5)
            hidden_value = fmt_pm(h, "test_rmse_mean", "test_rmse_std", 5)
        else:
            label = f"{DATASET_LABELS[dataset]} accuracy"
            primary_value = fmt_pm(p, "test_accuracy_pct_mean", "test_accuracy_pct_std", 3)
            hidden_value = fmt_pm(h, "test_accuracy_pct_mean", "test_accuracy_pct_std", 3)
        rows.append(
            {
                "Dataset / metric": label,
                "Primary: hidden + output": primary_value,
                "Ablation: hidden only": hidden_value,
            }
        )
    return fields, rows


def make_table7(primary_index: Mapping[tuple[str, str], Mapping[str, str]]) -> tuple[List[str], List[Dict[str, object]]]:
    fields = ["Dataset", "lambda_1", "psi_1", "lambda_2", "psi_2", "lambda_3", "psi_3"]
    rows: List[Dict[str, object]] = []
    mapping = [
        ("lambda_1", "field1_lambda_mean", "field1_lambda_std"),
        ("psi_1", "field1_psi_mean", "field1_psi_std"),
        ("lambda_2", "field2_lambda_mean", "field2_lambda_std"),
        ("psi_2", "field2_psi_mean", "field2_psi_std"),
        ("lambda_3", "field3_lambda_mean", "field3_lambda_std"),
        ("psi_3", "field3_psi_mean", "field3_psi_std"),
    ]
    for dataset in DATASET_ORDER:
        row = require_row(primary_index, dataset, "aptx_field")
        out: Dict[str, object] = {"Dataset": DATASET_LABELS[dataset]}
        for label, mean_key, std_key in mapping:
            out[label] = fmt_pm(row, mean_key, std_key, 3)
        rows.append(out)
    return fields, rows


def make_table8(primary_index: Mapping[tuple[str, str], Mapping[str, str]]) -> tuple[List[str], List[Dict[str, object]]]:
    fields = ["Dataset", "lambda_out", "psi_out"]
    rows: List[Dict[str, object]] = []
    for dataset in DATASET_ORDER:
        row = require_row(primary_index, dataset, "aptx_field")
        rows.append(
            {
                "Dataset": DATASET_LABELS[dataset],
                "lambda_out": fmt_pm(row, "output_lambda_mean", "output_lambda_std", 3),
                "psi_out": fmt_pm(row, "output_psi_mean", "output_psi_std", 3),
            }
        )
    return fields, rows


def main() -> None:
    args = parse_args()

    primary_all = read_csv(args.primary_dir / "all_runs.csv")
    primary_selected = read_csv(args.primary_dir / "selected_lr_summary.csv")
    hidden_all = read_csv(args.hidden_only_dir / "all_runs.csv")
    hidden_selected = read_csv(args.hidden_only_dir / "selected_lr_summary.csv")

    validate_primary(primary_all, primary_selected)
    validate_hidden(hidden_all, hidden_selected)

    primary_index = index_selected(primary_selected)
    hidden_index = index_selected(hidden_selected)

    table_builders = [
        lambda: make_table1(),
        lambda: make_table2(primary_all),
        lambda: make_table3(primary_all),
        lambda: make_table4(primary_index),
        lambda: make_table5(primary_index),
        lambda: make_table6(primary_index, hidden_index),
        lambda: make_table7(primary_index),
        lambda: make_table8(primary_index),
    ]

    descriptions = [
        "Boundary regimes",
        "Dataset/network configuration",
        "Architectural variants",
        "Primary classification results",
        "Primary California Housing results",
        "Hidden-only APTx Neuron + Field ablation",
        "Learned hidden-field Lambda/Psi values",
        "Learned output-field Lambda/Psi values",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for i, (builder, description) in enumerate(zip(table_builders, descriptions), start=1):
        fields, rows = builder()
        path = args.output_dir / f"table{i}.csv"
        write_csv(path, fields, rows)
        print(f"Wrote {path}  ({description})")


if __name__ == "__main__":
    main()
