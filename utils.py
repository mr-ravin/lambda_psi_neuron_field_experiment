"""Shared experiment utilities for the Lambda-Psi study.

This module provides dataset configuration/loading, reproducibility, run IDs,
metrics, training/evaluation helpers, optimizer construction, CSV persistence,
and cross-seed / cross-learning-rate summaries. It intentionally contains no CLI
loop and no model architecture definitions.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import json
import math
import os
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset, TensorDataset

# -----------------------------------------------------------------------------
# Dataset defaults and per-run settings
# -----------------------------------------------------------------------------
DATASET_CONFIGS: Dict[str, Dict[str, object]] = {
    "mnist": {
        "task_type": "classification",
        "dataset_class": torchvision.datasets.MNIST,
        "input_dim": 1 * 28 * 28,
        "output_dim": 10,
        "num_classes": 10,
        "mean": (0.1307,),
        "std": (0.3081,),
        "image_size": 28,
        "val_size": 5000,
        "epochs": 50,
        "batch_size": 128,
        "hidden_dims": (256, 128, 64),
        "patience": 5,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
    },
    "fashion_mnist": {
        "task_type": "classification",
        "dataset_class": torchvision.datasets.FashionMNIST,
        "input_dim": 1 * 28 * 28,
        "output_dim": 10,
        "num_classes": 10,
        "mean": (0.2860,),
        "std": (0.3530,),
        "image_size": 28,
        "val_size": 5000,
        "epochs": 100,
        "batch_size": 128,
        "hidden_dims": (256, 192, 128),
        "patience": 10,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
    },
    "cifar10": {
        "task_type": "classification",
        "dataset_class": torchvision.datasets.CIFAR10,
        "input_dim": 3 * 32 * 32,
        "output_dim": 10,
        "num_classes": 10,
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "image_size": 32,
        "val_size": 5000,
        "epochs": 100,
        "batch_size": 64,
        "hidden_dims": (384, 256, 192),
        "patience": 15,
        "weight_decay": 2e-4,
        "label_smoothing": 0.05,
    },
    "cifar100": {
        "task_type": "classification",
        "dataset_class": torchvision.datasets.CIFAR100,
        "input_dim": 3 * 32 * 32,
        "output_dim": 100,
        "num_classes": 100,
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "image_size": 32,
        "val_size": 5000,
        "epochs": 100,
        "batch_size": 64,
        "hidden_dims": (384, 256, 192),
        "patience": 20,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
    },
    "california_housing": {
        "task_type": "regression",
        "input_dim": 8,
        "output_dim": 1,
        "num_classes": None,
        "val_size": 0.10,
        "test_size": 0.20,
        "epochs": 200,
        "batch_size": 128,
        "hidden_dims": (128, 64, 32),
        "patience": 20,
        "weight_decay": 1e-4,
        "label_smoothing": 0.0,
    },
}

@dataclass
class RunSettings:
    dataset_name: str
    variant: str
    seed: int
    split_seed: int
    lr: float
    data_dir: str
    results_dir: str
    download_dataset: bool = False
    num_workers: int = 4
    amp: bool = False
    grad_clip: Optional[float] = 1.0
    min_delta: float = 1e-4
    lambda_init: float = 0.5
    psi_init: float = 0.5
    field_on_output: bool = False
    is_alpha_trainable: bool = True
    use_delta: bool = True
    epochs_override: Optional[int] = None
    batch_size_override: Optional[int] = None
    hidden_dims_override: Optional[Tuple[int, int, int]] = None
    patience_override: Optional[int] = None
    weight_decay_override: Optional[float] = None
    label_smoothing_override: Optional[float] = None

# -----------------------------------------------------------------------------
# Reproducibility and run identity
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id: int) -> None:
    # DataLoader gives each worker a deterministic torch seed based on its generator.
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def resolve_run_hyperparameters(settings: RunSettings) -> Dict[str, object]:
    """Resolve dataset defaults plus optional CLI overrides into one configuration."""
    if settings.dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset {settings.dataset_name!r}")

    config = DATASET_CONFIGS[settings.dataset_name]
    return {
        "epochs": (
            int(settings.epochs_override)
            if settings.epochs_override is not None
            else int(config["epochs"])
        ),
        "batch_size": (
            int(settings.batch_size_override)
            if settings.batch_size_override is not None
            else int(config["batch_size"])
        ),
        "hidden_dims": (
            tuple(settings.hidden_dims_override)
            if settings.hidden_dims_override is not None
            else tuple(config["hidden_dims"])
        ),
        "patience": (
            int(settings.patience_override)
            if settings.patience_override is not None
            else int(config["patience"])
        ),
        "weight_decay": (
            float(settings.weight_decay_override)
            if settings.weight_decay_override is not None
            else float(config["weight_decay"])
        ),
        "label_smoothing": (
            float(settings.label_smoothing_override)
            if settings.label_smoothing_override is not None
            else float(config["label_smoothing"])
        ),
    }

@functools.lru_cache(maxsize=1)
def source_fingerprint() -> str:
    """Fingerprint all scientific Python code that defines an experiment.

    In the original single-file runner this fingerprint covered experiment.py and
    the field package. After modularization the same protection must cover
    utils.py, models.py, run.py, and lambda_psi_neuron_fields/__init__.py.
    This prevents resume mode from silently mixing results from different code.
    """
    hasher = hashlib.sha256()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(base_dir, "utils.py"),
        os.path.join(base_dir, "models.py"),
        os.path.join(base_dir, "run.py"),
        os.path.join(base_dir, "lambda_psi_neuron_fields", "__init__.py"),
    ]
    for path in paths:
        with open(path, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()[:16]


def runtime_backend() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def run_config_hash(settings: RunSettings) -> str:
    """Hash all non-seed/non-LR settings that define an experimental condition."""
    resolved = resolve_run_hyperparameters(settings)
    payload = {
        "dataset": settings.dataset_name,
        "variant": settings.variant,
        "split_seed": settings.split_seed,
        "field_on_output": bool(settings.field_on_output),
        "is_alpha_trainable": bool(settings.is_alpha_trainable),
        "use_delta": bool(settings.use_delta),
        "lambda_init": float(settings.lambda_init),
        "psi_init": float(settings.psi_init),
        "grad_clip": None if settings.grad_clip is None else float(settings.grad_clip),
        "min_delta": float(settings.min_delta),
        "amp": bool(settings.amp),
        "num_workers": int(settings.num_workers),
        "runtime_backend": runtime_backend(),
        "source_fingerprint": source_fingerprint(),
        **resolved,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]

def build_run_id(settings: RunSettings) -> str:
    lr_tag = f"{settings.lr:.12g}".replace("+", "").replace("-", "m").replace(".", "p")
    return (
        f"{settings.dataset_name}__{settings.variant}__seed{settings.seed}"
        f"__lr{lr_tag}__cfg{run_config_hash(settings)}"
    )

# -----------------------------------------------------------------------------
# Data transforms and DataLoaders
# -----------------------------------------------------------------------------
def _classification_transforms(dataset_name: str, config: Dict[str, object]):
    mean = config["mean"]
    std = config["std"]
    image_size = int(config["image_size"])

    if dataset_name in {"mnist", "fashion_mnist"}:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(image_size, padding=2),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform

def build_classification_dataloaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    run_seed: int,
    split_seed: int,
    download_dataset: bool,
):
    config = DATASET_CONFIGS[dataset_name]
    dataset_class = config["dataset_class"]
    train_transform, eval_transform = _classification_transforms(dataset_name, config)

    train_dataset_aug = dataset_class(
        root=data_dir,
        train=True,
        download=download_dataset,
        transform=train_transform,
    )
    train_dataset_eval = dataset_class(
        root=data_dir,
        train=True,
        download=download_dataset,
        transform=eval_transform,
    )
    test_set = dataset_class(
        root=data_dir,
        train=False,
        download=download_dataset,
        transform=eval_transform,
    )

    num_train = len(train_dataset_aug)
    val_size = int(config["val_size"])
    if not 0 < val_size < num_train:
        raise ValueError(f"Invalid val_size={val_size} for dataset size {num_train}")

    # Split seed is deliberately separate from training seed so all architectures,
    # LRs, and model seeds can be compared on exactly the same validation split.
    split_rng = np.random.default_rng(split_seed)
    indices = np.arange(num_train)
    split_rng.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_set = Subset(train_dataset_aug, train_indices)
    val_set = Subset(train_dataset_eval, val_indices)

    train_generator = torch.Generator()
    train_generator.manual_seed(run_seed)

    pin_memory = torch.cuda.is_available()
    common = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
    }

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=train_generator,
        **common,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        **common,
    )

    return train_loader, val_loader, test_loader

def build_california_housing_dataloaders(
    batch_size: int,
    num_workers: int,
    run_seed: int,
    split_seed: int,
    download_dataset: bool,
):
    config = DATASET_CONFIGS["california_housing"]
    data = fetch_california_housing(download_if_missing=download_dataset)

    X = data.data.astype(np.float32)
    # IMPORTANT: y stays in the dataset's original target units.
    # Only X is standardized. Therefore MSE/RMSE/MAE are never normalized.
    y = data.target.astype(np.float32).reshape(-1, 1)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=float(config["test_size"]),
        random_state=split_seed,
    )

    val_relative_size = float(config["val_size"]) / (1.0 - float(config["test_size"]))
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_relative_size,
        random_state=split_seed,
    )

    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train).astype(np.float32)
    X_val = x_scaler.transform(X_val).astype(np.float32)
    X_test = x_scaler.transform(X_test).astype(np.float32)

    train_set = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_set = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_set = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_generator = torch.Generator()
    train_generator.manual_seed(run_seed)

    pin_memory = torch.cuda.is_available()
    common = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker if num_workers > 0 else None,
    }

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=train_generator,
        **common,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        **common,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, test_loader

def build_dataloaders(settings: RunSettings, batch_size: int):
    config = DATASET_CONFIGS[settings.dataset_name]
    if config["task_type"] == "classification":
        return build_classification_dataloaders(
            dataset_name=settings.dataset_name,
            data_dir=settings.data_dir,
            batch_size=batch_size,
            num_workers=settings.num_workers,
            run_seed=settings.seed,
            split_seed=settings.split_seed,
            download_dataset=settings.download_dataset,
        )
    return build_california_housing_dataloaders(
        batch_size=batch_size,
        num_workers=settings.num_workers,
        run_seed=settings.seed,
        split_seed=settings.split_seed,
        download_dataset=settings.download_dataset,
    )

# -----------------------------------------------------------------------------
# Metric accumulation
# -----------------------------------------------------------------------------
def _empty_metric_state(task_type: str) -> Dict[str, float]:
    if task_type == "classification":
        return {
            "loss_sum": 0.0,
            "sample_count": 0.0,
            "correct1": 0.0,
            "correct5": 0.0,
        }
    return {
        "loss_sum": 0.0,
        "sample_count": 0.0,
        "element_count": 0.0,
        "sse": 0.0,
        "sae": 0.0,
        "sum_y": 0.0,
        "sum_y2": 0.0,
    }

def _update_classification_state(
    state: Dict[str, float],
    outputs: torch.Tensor,
    targets: torch.Tensor,
    loss: torch.Tensor,
) -> None:
    batch = targets.size(0)
    state["loss_sum"] += float(loss.detach()) * batch
    state["sample_count"] += batch

    preds = outputs.argmax(dim=1)
    state["correct1"] += float((preds == targets).sum().item())

    k = min(5, outputs.size(1))
    topk = outputs.topk(k=k, dim=1).indices
    correct5 = topk.eq(targets.view(-1, 1)).any(dim=1).sum().item()
    state["correct5"] += float(correct5)

def _update_regression_state(
    state: Dict[str, float],
    outputs: torch.Tensor,
    targets: torch.Tensor,
    loss: torch.Tensor,
) -> None:
    batch = targets.size(0)
    errors = (outputs.detach() - targets).double().view(-1)
    y = targets.detach().double().view(-1)

    state["loss_sum"] += float(loss.detach()) * batch
    state["sample_count"] += batch
    state["element_count"] += y.numel()
    state["sse"] += float(torch.sum(errors * errors).item())
    state["sae"] += float(torch.sum(torch.abs(errors)).item())
    state["sum_y"] += float(torch.sum(y).item())
    state["sum_y2"] += float(torch.sum(y * y).item())

def _finalize_metrics(state: Dict[str, float], task_type: str) -> Dict[str, float]:
    if state["sample_count"] <= 0:
        raise RuntimeError("No samples were processed.")

    loss = state["loss_sum"] / state["sample_count"]
    if task_type == "classification":
        n = state["sample_count"]
        return {
            "loss": loss,
            "accuracy": state["correct1"] / n,
            "top5_accuracy": state["correct5"] / n,
        }

    n = state["element_count"]
    mse = state["sse"] / n
    rmse = math.sqrt(mse)
    mae = state["sae"] / n
    ss_tot = state["sum_y2"] - (state["sum_y"] ** 2) / n
    r2 = 0.0 if ss_tot <= 0 else 1.0 - state["sse"] / ss_tot
    return {
        "loss": loss,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

# -----------------------------------------------------------------------------
# Training, evaluation, and optimizer helpers
# -----------------------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    task_type: str,
    scaler,
    grad_clip: Optional[float],
) -> Dict[str, float]:
    model.train()
    state = _empty_metric_state(task_type)

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                if task_type == "regression":
                    outputs = outputs.view_as(targets)
                loss = criterion(outputs, targets)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss encountered before optimizer step: {float(loss.detach())}"
                )
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            if task_type == "regression":
                outputs = outputs.view_as(targets)
            loss = criterion(outputs, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss encountered before optimizer step: {float(loss.detach())}"
                )
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if task_type == "classification":
            _update_classification_state(state, outputs, targets, loss)
        else:
            _update_regression_state(state, outputs, targets, loss)

    return _finalize_metrics(state, task_type)

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    task_type: str,
) -> Dict[str, float]:
    model.eval()
    state = _empty_metric_state(task_type)

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        if task_type == "regression":
            outputs = outputs.view_as(targets)
        loss = criterion(outputs, targets)

        if task_type == "classification":
            _update_classification_state(state, outputs, targets, loss)
        else:
            _update_regression_state(state, outputs, targets, loss)

    return _finalize_metrics(state, task_type)

def _metric_is_better(
    task_type: str,
    current: float,
    best: float,
    min_delta: float,
    first_epoch: bool,
) -> bool:
    if first_epoch:
        return True
    if task_type == "classification":
        return current > best + min_delta
    return current < best - min_delta

def _format_metrics(task_type: str, metrics: Dict[str, float]) -> str:
    if task_type == "classification":
        return (
            f"loss={metrics['loss']:.4f}, "
            f"acc={metrics['accuracy'] * 100:.2f}%, "
            f"top5={metrics['top5_accuracy'] * 100:.2f}%"
        )
    return (
        f"loss/MSE={metrics['loss']:.4f}, "
        f"RMSE={metrics['rmse']:.4f}, "
        f"MAE={metrics['mae']:.4f}, "
        f"R2={metrics['r2']:.4f}"
    )

def _safe_float(value: Optional[float]):
    return "" if value is None else float(value)

def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
    """Build AdamW without weight-decaying Lambda/Psi raw field parameters.

    Weight decay on lambda_raw/psi_raw would pull those logits toward zero, which
    corresponds to lambda=psi=0.5. That would impose an unintended prior on the
    learned field regime. Base-neuron parameters retain the configured decay.
    """
    field_params = []
    regular_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("lambda_raw") or name.endswith("psi_raw"):
            field_params.append(parameter)
        else:
            regular_params.append(parameter)

    groups = []
    if regular_params:
        groups.append({"params": regular_params, "weight_decay": weight_decay})
    if field_params:
        groups.append({"params": field_params, "weight_decay": 0.0})

    return optim.AdamW(groups, lr=lr)

# -----------------------------------------------------------------------------
# Per-run CSV schema
# -----------------------------------------------------------------------------
RESULT_COLUMNS = [
    "run_id",
    "config_hash",
    "source_fingerprint",
    "torch_version",
    "torchvision_version",
    "sklearn_version",
    "numpy_version",
    "amp",
    "num_workers",
    "timestamp_utc",
    "status",
    "error",
    "dataset",
    "task_type",
    "variant",
    "variant_label",
    "base_neuron",
    "uses_field",
    "relu_position",
    "field_on_output",
    "is_alpha_trainable",
    "use_delta_or_bias",
    "lambda_init",
    "psi_init",
    "seed",
    "split_seed",
    "lr_initial",
    "weight_decay",
    "label_smoothing",
    "epochs_requested",
    "epochs_ran",
    "best_epoch",
    "patience",
    "min_delta",
    "batch_size",
    "hidden_1",
    "hidden_2",
    "hidden_3",
    "total_parameters",
    "trainable_parameters",
    "train_seconds",
    "device",
    "checkpoint_path",
    "best_val_loss",
    "best_val_accuracy_pct",
    "best_val_top5_accuracy_pct",
    "best_val_mse",
    "best_val_rmse",
    "best_val_rmse_usd",
    "best_val_mae",
    "best_val_mae_usd",
    "best_val_r2",
    "test_loss",
    "test_accuracy_pct",
    "test_top5_accuracy_pct",
    "test_mse",
    "test_rmse",
    "test_rmse_usd",
    "test_mae",
    "test_mae_usd",
    "test_r2",
    "field1_lambda",
    "field1_psi",
    "field2_lambda",
    "field2_psi",
    "field3_lambda",
    "field3_psi",
    "output_lambda",
    "output_psi",
]

# -----------------------------------------------------------------------------
# Summary metrics and CSV helpers
# -----------------------------------------------------------------------------
SUMMARY_METRICS = [
    "best_val_loss",
    "best_val_accuracy_pct",
    "best_val_top5_accuracy_pct",
    "best_val_mse",
    "best_val_rmse",
    "best_val_rmse_usd",
    "best_val_mae",
    "best_val_mae_usd",
    "best_val_r2",
    "test_loss",
    "test_accuracy_pct",
    "test_top5_accuracy_pct",
    "test_mse",
    "test_rmse",
    "test_rmse_usd",
    "test_mae",
    "test_mae_usd",
    "test_r2",
    "field1_lambda",
    "field1_psi",
    "field2_lambda",
    "field2_psi",
    "field3_lambda",
    "field3_psi",
    "output_lambda",
    "output_psi",
    "train_seconds",
]

def canonical_lr(lr: float) -> str:
    return f"{float(lr):.12g}"

def validate_choices(values: Sequence[str], allowed: Iterable[str], label: str) -> None:
    allowed_set = set(allowed)
    bad = [value for value in values if value not in allowed_set]
    if bad:
        raise SystemExit(
            f"Unknown {label}: {bad}. Allowed values: {', '.join(sorted(allowed_set))}"
        )

def read_existing_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def completed_run_ids(path: str) -> set:
    return {
        row["run_id"]
        for row in read_existing_rows(path)
        if row.get("status") == "ok"
    }

def append_result(path: str, row: Dict[str, object]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in RESULT_COLUMNS})
        f.flush()
        os.fsync(f.fileno())

def parse_float(value: object):
    if value is None or value == "":
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if math.isfinite(value_f) else None

def mean_std(values: List[float]) -> Tuple[object, object]:
    if not values:
        return "", ""
    mean_value = statistics.fmean(values)
    std_value = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean_value, std_value

def write_summaries(
    all_runs_csv: str,
    results_dir: str,
    expected_seeds: int,
    expected_lrs: int,
) -> None:
    # Summary flow:
    #   1. Deduplicate successful runs by run_id.
    #   2. Group by dataset + variant + config + LR.
    #   3. Compute mean/std across seeds and write summary_by_lr.csv.
    #   4. Once every requested LR has every requested seed, choose the winning
    #      LR from validation metrics only and write selected_lr_summary.csv.
    # Keep the latest successful row per run_id. This makes resume/retry safe.
    successful_by_id = {}
    for row in read_existing_rows(all_runs_csv):
        if row.get("status") == "ok":
            successful_by_id[row["run_id"]] = row
    rows = list(successful_by_id.values())
    if not rows:
        return

    groups = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["variant"],
            row.get("config_hash", ""),
            canonical_lr(float(row["lr_initial"])),
        )
        groups[key].append(row)

    summary_columns = [
        "dataset",
        "task_type",
        "variant",
        "variant_label",
        "config_hash",
        "field_on_output",
        "is_alpha_trainable",
        "use_delta_or_bias",
        "lambda_init",
        "psi_init",
        "epochs_requested",
        "batch_size",
        "hidden_1",
        "hidden_2",
        "hidden_3",
        "weight_decay",
        "label_smoothing",
        "lr_initial",
        "n_runs",
        "n_unique_seeds",
        "expected_seeds",
        "complete_seed_set",
    ]
    for metric in SUMMARY_METRICS:
        summary_columns.extend([f"{metric}_mean", f"{metric}_std"])

    summary_rows: List[Dict[str, object]] = []
    for (dataset, variant, config_hash, lr), group in sorted(groups.items()):
        first = group[0]
        out: Dict[str, object] = {
            "dataset": dataset,
            "task_type": first["task_type"],
            "variant": variant,
            "variant_label": first["variant_label"],
            "config_hash": config_hash,
            "field_on_output": first.get("field_on_output", ""),
            "is_alpha_trainable": first.get("is_alpha_trainable", ""),
            "use_delta_or_bias": first.get("use_delta_or_bias", ""),
            "lambda_init": first.get("lambda_init", ""),
            "psi_init": first.get("psi_init", ""),
            "epochs_requested": first.get("epochs_requested", ""),
            "batch_size": first.get("batch_size", ""),
            "hidden_1": first.get("hidden_1", ""),
            "hidden_2": first.get("hidden_2", ""),
            "hidden_3": first.get("hidden_3", ""),
            "weight_decay": first.get("weight_decay", ""),
            "label_smoothing": first.get("label_smoothing", ""),
            "lr_initial": lr,
            "n_runs": len(group),
            "n_unique_seeds": len({row["seed"] for row in group}),
            "expected_seeds": expected_seeds,
            "complete_seed_set": int(
                len({row["seed"] for row in group}) >= expected_seeds
            ),
        }
        for metric in SUMMARY_METRICS:
            values = [
                v
                for v in (parse_float(row.get(metric)) for row in group)
                if v is not None
            ]
            metric_mean, metric_std = mean_std(values)
            out[f"{metric}_mean"] = metric_mean
            out[f"{metric}_std"] = metric_std
        summary_rows.append(out)

    summary_by_lr_path = os.path.join(results_dir, "summary_by_lr.csv")
    with open(summary_by_lr_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_columns)
        writer.writeheader()
        writer.writerows(summary_rows)

    # Select learning rate only from validation metrics; test results are never
    # used to choose the winning LR.
    variant_groups = defaultdict(list)
    for row in summary_rows:
        variant_groups[(row["dataset"], row["variant"], row.get("config_hash", ""))].append(row)

    selected_rows: List[Dict[str, object]] = []
    for (_, _, _), candidates in sorted(variant_groups.items()):
        complete = [row for row in candidates if int(row["complete_seed_set"]) == 1]
        complete_lrs = {canonical_lr(float(row["lr_initial"])) for row in complete}

        # Do not publish a "selected" LR from a partial sweep.  This prevents a
        # half-finished grid from looking like a final model-selection result.
        if len(complete_lrs) < expected_lrs:
            continue

        pool = complete
        task_type = str(pool[0]["task_type"])

        if task_type == "classification":
            valid = [
                row
                for row in pool
                if parse_float(row["best_val_accuracy_pct_mean"]) is not None
            ]
            if not valid:
                continue
            selected = max(
                valid,
                key=lambda row: float(row["best_val_accuracy_pct_mean"]),
            )
            selection_metric = "best_val_accuracy_pct_mean"
            selection_direction = "max"
        else:
            valid = [
                row
                for row in pool
                if parse_float(row["best_val_rmse_mean"]) is not None
            ]
            if not valid:
                continue
            selected = min(valid, key=lambda row: float(row["best_val_rmse_mean"]))
            selection_metric = "best_val_rmse_mean"
            selection_direction = "min"

        out = dict(selected)
        out["selection_metric"] = selection_metric
        out["selection_direction"] = selection_direction
        out["selected_lr"] = selected["lr_initial"]
        selected_rows.append(out)

    selected_columns = [
        "dataset",
        "task_type",
        "variant",
        "variant_label",
        "config_hash",
        "field_on_output",
        "is_alpha_trainable",
        "use_delta_or_bias",
        "lambda_init",
        "psi_init",
        "epochs_requested",
        "batch_size",
        "hidden_1",
        "hidden_2",
        "hidden_3",
        "weight_decay",
        "label_smoothing",
        "selected_lr",
        "selection_metric",
        "selection_direction",
        "n_runs",
        "n_unique_seeds",
        "expected_seeds",
        "complete_seed_set",
    ]
    for metric in SUMMARY_METRICS:
        selected_columns.extend([f"{metric}_mean", f"{metric}_std"])

    selected_path = os.path.join(results_dir, "selected_lr_summary.csv")
    with open(selected_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=selected_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(selected_rows)
