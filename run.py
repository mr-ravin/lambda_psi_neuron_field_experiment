"""Command-line runner for the Lambda-Psi comparison study.

Execution flow:
1. Parse and validate CLI arguments.
2. Build the dataset x variant x learning-rate x seed grid.
3. Execute one train/validation/test lifecycle per grid entry.
4. Append every completed/error run to all_runs.csv immediately.
5. Refresh cross-seed summaries and validation-selected learning-rate results.

Architecture code lives in models.py and reusable experiment utilities live in
utils.py, keeping this file focused on orchestration.
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
from datetime import datetime, timezone
from typing import Dict

import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from models import ComparisonNetwork, VARIANT_SPECS
from utils import (
    DATASET_CONFIGS,
    RESULT_COLUMNS,
    RunSettings,
    _format_metrics,
    _metric_is_better,
    append_result,
    build_dataloaders,
    build_optimizer,
    build_run_id,
    completed_run_ids,
    resolve_run_hyperparameters,
    run_config_hash,
    set_seed,
    source_fingerprint,
    train_one_epoch,
    evaluate,
    validate_choices,
    write_summaries,
)

# -----------------------------------------------------------------------------
# One complete experiment run
# -----------------------------------------------------------------------------
def run_single_experiment(settings: RunSettings) -> Dict[str, object]:
    # One call represents exactly one dataset + variant + seed + initial-LR run.
    # The function resolves settings, builds data/model/optimizer, trains while
    # selecting checkpoints on validation data, restores the best checkpoint,
    # performs final validation/test evaluation, and returns one CSV-ready record.
    if settings.dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset {settings.dataset_name!r}")
    if settings.variant not in VARIANT_SPECS:
        raise ValueError(f"Unknown variant {settings.variant!r}")

    config = DATASET_CONFIGS[settings.dataset_name]
    spec = VARIANT_SPECS[settings.variant]
    task_type = str(config["task_type"])

    resolved = resolve_run_hyperparameters(settings)
    epochs = int(resolved["epochs"])
    batch_size = int(resolved["batch_size"])
    hidden_dims = tuple(resolved["hidden_dims"])
    patience = int(resolved["patience"])
    weight_decay = float(resolved["weight_decay"])
    label_smoothing = float(resolved["label_smoothing"])

    set_seed(settings.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = build_dataloaders(settings, batch_size)

    model = ComparisonNetwork(
        input_dim=int(config["input_dim"]),
        output_dim=int(config["output_dim"]),
        hidden_dims=hidden_dims,
        variant=settings.variant,
        lambda_init=settings.lambda_init,
        psi_init=settings.psi_init,
        field_on_output=settings.field_on_output,
        is_alpha_trainable=settings.is_alpha_trainable,
        use_delta=settings.use_delta,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if task_type == "classification":
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        best_metric = -float("inf")
    else:
        criterion = nn.MSELoss()
        best_metric = float("inf")

    optimizer = build_optimizer(
        model=model,
        lr=settings.lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = (
        torch.cuda.amp.GradScaler()
        if settings.amp and device.type == "cuda"
        else None
    )

    run_id = build_run_id(settings)
    config_hash = run_config_hash(settings)
    checkpoint_dir = os.path.join(settings.results_dir, "checkpoints", settings.dataset_name, settings.variant)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"{run_id}.pt")

    best_epoch = 0
    epochs_without_improvement = 0
    epochs_ran = 0
    start = time.perf_counter()

    print("\n" + "=" * 100)
    print(f"Run: {run_id}")
    print(f"Variant: {spec['label']}")
    print(f"Device: {device}")
    print(
        f"epochs={epochs}, batch={batch_size}, hidden={hidden_dims}, "
        f"lr={settings.lr:g}, seed={settings.seed}, split_seed={settings.split_seed}"
    )
    print(f"parameters={total_params:,}, trainable={trainable_params:,}")

    for epoch in range(1, epochs + 1):
        epochs_ran = epoch
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            task_type=task_type,
            scaler=scaler,
            grad_clip=settings.grad_clip,
        )
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            task_type=task_type,
        )
        scheduler.step()

        current_metric = (
            val_metrics["accuracy"]
            if task_type == "classification"
            else val_metrics["rmse"]
        )
        is_best = _metric_is_better(
            task_type=task_type,
            current=current_metric,
            best=best_metric,
            min_delta=settings.min_delta,
            first_epoch=(epoch == 1),
        )

        if is_best:
            best_metric = current_metric
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_metric": best_metric,
                    "run_id": run_id,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch [{epoch:03d}/{epochs}] "
            f"train({_format_metrics(task_type, train_metrics)}) | "
            f"val({_format_metrics(task_type, val_metrics)}) | "
            f"best_epoch={best_epoch}"
        )

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch}; best epoch={best_epoch}; "
                f"patience={patience}."
            )
            break

    train_seconds = time.perf_counter() - start

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Re-evaluate validation using exactly the restored best model. This gives all
    # validation metrics corresponding to the checkpoint used for final testing.
    best_val_metrics = evaluate(model, val_loader, criterion, device, task_type)
    test_metrics = evaluate(model, test_loader, criterion, device, task_type)
    field_values = model.field_values()

    print(f"Best validation: {_format_metrics(task_type, best_val_metrics)}")
    print(f"Final test:     {_format_metrics(task_type, test_metrics)}")
    if field_values:
        print("Best-checkpoint Lambda/Psi:")
        for name, (lam, psi) in field_values.items():
            print(f"  {name}: lambda={lam:.6f}, psi={psi:.6f}")

    result: Dict[str, object] = {
        "run_id": run_id,
        "config_hash": config_hash,
        "source_fingerprint": source_fingerprint(),
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "amp": int(settings.amp),
        "num_workers": settings.num_workers,
        "timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "status": "ok",
        "error": "",
        "dataset": settings.dataset_name,
        "task_type": task_type,
        "variant": settings.variant,
        "variant_label": spec["label"],
        "base_neuron": spec["base"],
        "uses_field": int(bool(spec["use_field"])),
        "relu_position": spec["relu_position"],
        "field_on_output": int(settings.field_on_output and bool(spec["use_field"])),
        "is_alpha_trainable": int(settings.is_alpha_trainable),
        "use_delta_or_bias": int(settings.use_delta),
        "lambda_init": settings.lambda_init,
        "psi_init": settings.psi_init,
        "seed": settings.seed,
        "split_seed": settings.split_seed,
        "lr_initial": settings.lr,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "epochs_requested": epochs,
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch,
        "patience": patience,
        "min_delta": settings.min_delta,
        "batch_size": batch_size,
        "hidden_1": hidden_dims[0],
        "hidden_2": hidden_dims[1],
        "hidden_3": hidden_dims[2],
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "train_seconds": train_seconds,
        "device": str(device),
        "checkpoint_path": checkpoint_path,
        "best_val_loss": best_val_metrics["loss"],
        "best_val_accuracy_pct": "",
        "best_val_top5_accuracy_pct": "",
        "best_val_mse": "",
        "best_val_rmse": "",
        "best_val_rmse_usd": "",
        "best_val_mae": "",
        "best_val_mae_usd": "",
        "best_val_r2": "",
        "test_loss": test_metrics["loss"],
        "test_accuracy_pct": "",
        "test_top5_accuracy_pct": "",
        "test_mse": "",
        "test_rmse": "",
        "test_rmse_usd": "",
        "test_mae": "",
        "test_mae_usd": "",
        "test_r2": "",
        "field1_lambda": "",
        "field1_psi": "",
        "field2_lambda": "",
        "field2_psi": "",
        "field3_lambda": "",
        "field3_psi": "",
        "output_lambda": "",
        "output_psi": "",
    }

    if task_type == "classification":
        result.update(
            {
                "best_val_accuracy_pct": best_val_metrics["accuracy"] * 100.0,
                "best_val_top5_accuracy_pct": best_val_metrics["top5_accuracy"] * 100.0,
                "test_accuracy_pct": test_metrics["accuracy"] * 100.0,
                "test_top5_accuracy_pct": test_metrics["top5_accuracy"] * 100.0,
            }
        )
    else:
        result.update(
            {
                "best_val_mse": best_val_metrics["mse"],
                "best_val_rmse": best_val_metrics["rmse"],
                "best_val_rmse_usd": best_val_metrics["rmse"] * 100000.0,
                "best_val_mae": best_val_metrics["mae"],
                "best_val_mae_usd": best_val_metrics["mae"] * 100000.0,
                "best_val_r2": best_val_metrics["r2"],
                "test_mse": test_metrics["mse"],
                "test_rmse": test_metrics["rmse"],
                "test_rmse_usd": test_metrics["rmse"] * 100000.0,
                "test_mae": test_metrics["mae"],
                "test_mae_usd": test_metrics["mae"] * 100000.0,
                "test_r2": test_metrics["r2"],
            }
        )

    for field_name, (lam, psi) in field_values.items():
        if field_name == "field1":
            result["field1_lambda"] = lam
            result["field1_psi"] = psi
        elif field_name == "field2":
            result["field2_lambda"] = lam
            result["field2_psi"] = psi
        elif field_name == "field3":
            result["field3_lambda"] = lam
            result["field3_psi"] = psi
        elif field_name == "output":
            result["output_lambda"] = lam
            result["output_psi"] = psi

    del model, optimizer, scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result

# -----------------------------------------------------------------------------
# Default experiment grid
# -----------------------------------------------------------------------------
DEFAULT_DATASETS = [
    "mnist",
    "fashion_mnist",
    "cifar10",
    "cifar100",
    "california_housing",
]
DEFAULT_VARIANTS = list(VARIANT_SPECS.keys())
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_LRS = [3e-4, 5e-4, 1e-3, 2e-3, 3e-3]

# -----------------------------------------------------------------------------
# CLI and grid orchestration
# -----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Lambda-Psi comparisons across datasets, variants, seeds, and "
            "initial learning rates. Results are stored as CSV files."
        )
    )
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--lrs", nargs="+", type=float, default=DEFAULT_LRS)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=2026,
        help="Fixed train/validation/test split seed, separate from training seeds.",
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--results-dir", default="./results/lambda_psi_comparison")
    parser.add_argument("--download-dataset", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--lambda-init", type=float, default=0.5)
    parser.add_argument("--psi-init", type=float, default=0.5)
    parser.add_argument(
        "--field-on-output",
        action="store_true",
        help=(
                "Also apply Lambda-Psi to the task output head for field variants. "
                "CLI default: fields are applied only to hidden representation layers. "
                "The paper's primary experiment enables this option."
        ),
    )
    parser.add_argument(
        "--no-alpha-trainable",
        action="store_true",
        help="Keep APTx alpha fixed at 1 instead of training it.",
    )
    parser.add_argument(
        "--no-delta",
        action="store_true",
        help="Disable APTx delta and traditional-neuron bias.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Rerun run IDs already completed in all_runs.csv.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the grid immediately if any run fails.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit the number of grid entries; useful for smoke tests.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--hidden-1", type=int, default=None)
    parser.add_argument("--hidden-2", type=int, default=None)
    parser.add_argument("--hidden-3", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    return parser

def main() -> None:
    # Program entry point: validate CLI input, build the Cartesian experiment grid,
    # skip already-completed run IDs when resume is enabled, execute each run,
    # persist success/error rows, and continuously regenerate aggregate summaries.
    args = build_parser().parse_args()
    validate_choices(args.datasets, DATASET_CONFIGS.keys(), "dataset")
    validate_choices(args.variants, VARIANT_SPECS.keys(), "variant")

    if any(value is not None for value in (args.hidden_1, args.hidden_2, args.hidden_3)):
        if any(value is None for value in (args.hidden_1, args.hidden_2, args.hidden_3)):
            raise SystemExit(
                "If hidden sizes are overridden, provide --hidden-1, --hidden-2, "
                "and --hidden-3 together."
            )
        hidden_override = (args.hidden_1, args.hidden_2, args.hidden_3)
    else:
        hidden_override = None

    if not 0.0 < args.lambda_init < 1.0 or not 0.0 < args.psi_init < 1.0:
        raise SystemExit(
            "Trainable --lambda-init and --psi-init must be strictly between 0 and 1."
        )

    os.makedirs(args.results_dir, exist_ok=True)
    all_runs_csv = os.path.join(args.results_dir, "all_runs.csv")
    done = set() if args.no_resume else completed_run_ids(all_runs_csv)

    seeds = list(dict.fromkeys(args.seeds))
    lrs = list(dict.fromkeys(args.lrs))

    grid = [
        (dataset, variant, seed, lr)
        for dataset in args.datasets
        for variant in args.variants
        for lr in lrs
        for seed in seeds
    ]
    if args.max_runs is not None:
        grid = grid[: args.max_runs]

    print(f"Planned grid entries: {len(grid)}")
    print(f"Results directory: {args.results_dir}")
    print(f"Resume mode: {'off' if args.no_resume else 'on'}")

    successful = 0
    skipped = 0
    failed = 0

    for index, (dataset, variant, seed, lr) in enumerate(grid, start=1):
        settings = RunSettings(
            dataset_name=dataset,
            variant=variant,
            seed=seed,
            split_seed=args.split_seed,
            lr=lr,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            download_dataset=args.download_dataset,
            num_workers=args.num_workers,
            amp=args.amp,
            grad_clip=args.grad_clip,
            min_delta=args.min_delta,
            lambda_init=args.lambda_init,
            psi_init=args.psi_init,
            field_on_output=args.field_on_output,
            is_alpha_trainable=not args.no_alpha_trainable,
            use_delta=not args.no_delta,
            epochs_override=args.epochs,
            batch_size_override=args.batch_size,
            hidden_dims_override=hidden_override,
            patience_override=args.patience,
            weight_decay_override=args.weight_decay,
            label_smoothing_override=args.label_smoothing,
        )

        run_id = build_run_id(settings)
        print(f"\nGRID {index}/{len(grid)}: {run_id}")

        if run_id in done:
            print("Already completed; skipping.")
            skipped += 1
            continue

        try:
            row = run_single_experiment(settings)
            append_result(all_runs_csv, row)
            done.add(run_id)
            successful += 1
        except Exception as exc:
            failed += 1
            error_text = f"{type(exc).__name__}: {exc}"
            print(error_text)
            traceback.print_exc()

            error_row = {column: "" for column in RESULT_COLUMNS}
            error_row.update(
                {
                    "run_id": run_id,
                    "config_hash": run_config_hash(settings),
                    "source_fingerprint": source_fingerprint(),
                    "torch_version": torch.__version__,
                    "torchvision_version": torchvision.__version__,
                    "sklearn_version": sklearn.__version__,
                    "numpy_version": np.__version__,
                    "amp": int(args.amp),
                    "num_workers": args.num_workers,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "error",
                    "error": error_text,
                    "dataset": dataset,
                    "task_type": DATASET_CONFIGS[dataset]["task_type"],
                    "variant": variant,
                    "variant_label": VARIANT_SPECS[variant]["label"],
                    "base_neuron": VARIANT_SPECS[variant]["base"],
                    "uses_field": int(bool(VARIANT_SPECS[variant]["use_field"])),
                    "relu_position": VARIANT_SPECS[variant]["relu_position"],
                    "field_on_output": int(
                        args.field_on_output and bool(VARIANT_SPECS[variant]["use_field"])
                    ),
                    "is_alpha_trainable": int(not args.no_alpha_trainable),
                    "use_delta_or_bias": int(not args.no_delta),
                    "lambda_init": args.lambda_init,
                    "psi_init": args.psi_init,
                    "seed": seed,
                    "split_seed": args.split_seed,
                    "lr_initial": lr,
                }
            )
            append_result(all_runs_csv, error_row)
            if args.fail_fast:
                raise

        write_summaries(
            all_runs_csv=all_runs_csv,
            results_dir=args.results_dir,
            expected_seeds=len(seeds),
            expected_lrs=len(lrs),
        )

    write_summaries(
        all_runs_csv=all_runs_csv,
        results_dir=args.results_dir,
        expected_seeds=len(seeds),
        expected_lrs=len(lrs),
    )

    print("\n" + "=" * 100)
    print("Grid finished")
    print(f"successful={successful}, skipped={skipped}, failed={failed}")
    print(f"all runs:            {all_runs_csv}")
    print(f"summary by LR:       {os.path.join(args.results_dir, 'summary_by_lr.csv')}")
    print(f"selected LR summary: {os.path.join(args.results_dir, 'selected_lr_summary.csv')}")
    print(f"finished UTC:        {datetime.now(timezone.utc).isoformat()}")

if __name__ == "__main__":
    main()
