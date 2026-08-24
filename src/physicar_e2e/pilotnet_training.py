"""Gated PilotNet V1 training, validation, ONNX export, and CPU benchmark."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import time
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from .pilotnet import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    PILOTNET_PARAMETER_COUNT,
    build_pilotnet,
    preprocess_png,
    steering_normalized_to_rad,
)


class GateFailure(RuntimeError):
    """A hard pipeline gate failed and later stages must not run."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    required = {
        "seed", "train_episodes", "validation_episodes", "image_width", "image_height",
        "max_steering_rad", "near_zero_steering_rad", "learning_rate", "batch_size",
        "max_epochs", "early_stopping_patience", "tiny_overfit_samples",
        "tiny_overfit_epochs", "tiny_overfit_required_loss_ratio", "onnx_opset",
        "onnx_equivalence_samples", "benchmark_warmup_iterations", "benchmark_iterations",
    }
    missing = required - config.keys()
    if missing:
        raise ValueError(f"training config missing fields: {sorted(missing)}")
    if config["image_width"] != IMAGE_WIDTH or config["image_height"] != IMAGE_HEIGHT:
        raise ValueError("training image dimensions do not match PilotNet V1")
    train = tuple(config["train_episodes"])
    validation = tuple(config["validation_episodes"])
    if set(train) & set(validation):
        raise ValueError("train and validation episodes overlap")
    return config


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def read_episode_rows(dataset_root: str | Path, episodes: Sequence[str]) -> list[dict[str, Any]]:
    root = Path(dataset_root)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for episode in episodes:
        manifest = root / "manifests" / f"{episode}.csv"
        if not manifest.is_file():
            raise GateFailure(f"dataset integrity: missing manifest {manifest}")
        with manifest.open(newline="", encoding="utf-8") as stream:
            episode_rows = list(csv.DictReader(stream))
        if not episode_rows:
            raise GateFailure(f"dataset integrity: empty episode {episode}")
        for raw in episode_rows:
            if raw.get("episode_id") != episode:
                raise GateFailure(f"dataset integrity: episode mismatch in {manifest}")
            try:
                sample_index = int(raw["sample_index"])
                steering = float(raw["steering_rad"])
            except (KeyError, TypeError, ValueError) as exc:
                raise GateFailure(f"dataset integrity: malformed row in {manifest}: {exc}") from exc
            key = (episode, sample_index)
            if key in seen:
                raise GateFailure(f"dataset integrity: duplicate sample {key}")
            seen.add(key)
            rows.append({
                "episode_id": episode,
                "sample_index": sample_index,
                "image_path": root / raw["image_path"],
                "steering_rad": steering,
            })
    return rows


def validate_dataset_integrity(dataset_root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    root = Path(dataset_root)
    metadata_path = root / "dataset_metadata.json"
    if not metadata_path.is_file():
        raise GateFailure("dataset integrity: dataset_metadata.json is missing")
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    train_episodes = tuple(config["train_episodes"])
    validation_episodes = tuple(config["validation_episodes"])
    if train_episodes != ("episode_001", "episode_002") or validation_episodes != ("episode_003",):
        raise GateFailure("dataset integrity: canonical V1 episode split changed")
    train_rows = read_episode_rows(root, train_episodes)
    validation_rows = read_episode_rows(root, validation_episodes)
    paths: set[Path] = set()
    maximum = float(config["max_steering_rad"])
    for row in (*train_rows, *validation_rows):
        path = row["image_path"]
        if path in paths:
            raise GateFailure(f"dataset integrity: image leakage/duplicate path {path}")
        paths.add(path)
        if not path.is_file():
            raise GateFailure(f"dataset integrity: missing image {path}")
        steering = row["steering_rad"]
        if not math.isfinite(steering):
            raise GateFailure(f"dataset integrity: non-finite label in {path}")
        if abs(steering) > maximum + 1e-6:
            raise GateFailure(f"dataset integrity: steering {steering} outside ±{maximum}")
        try:
            with Image.open(path) as image:
                if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT) or image.mode != "RGB":
                    raise GateFailure(
                        f"dataset integrity: expected RGB {IMAGE_WIDTH}x{IMAGE_HEIGHT}, "
                        f"got {image.mode} {image.size} at {path}"
                    )
                image.verify()
        except GateFailure:
            raise
        except Exception as exc:
            raise GateFailure(f"dataset integrity: unreadable image {path}: {exc}") from exc
    if set(train_episodes) & set(validation_episodes):
        raise GateFailure("dataset integrity: frame-level leakage through episode overlap")
    return {
        "result": "PASS",
        "dataset_metadata_sha256": sha256_file(metadata_path),
        "train_episodes": list(train_episodes),
        "validation_episodes": list(validation_episodes),
        "training_samples": len(train_rows),
        "validation_samples": len(validation_rows),
        "missing_images": 0,
        "non_finite_labels": 0,
        "episode_level_separation": True,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
    }


class PilotDataset(Dataset):
    def __init__(self, rows: Sequence[dict[str, Any]], max_steering_rad: float) -> None:
        self.rows = list(rows)
        self.max_steering_rad = max_steering_rad

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        image = torch.from_numpy(preprocess_png(row["image_path"]))
        target = torch.tensor([row["steering_rad"] / self.max_steering_rad], dtype=torch.float32)
        return image, target


def tiny_overfit_sanity(rows: Sequence[dict[str, Any]], config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    count = int(config["tiny_overfit_samples"])
    fixed_rows = list(rows[:count])
    if len(fixed_rows) != count:
        raise GateFailure("tiny-overfit: insufficient fixed samples")
    set_reproducible_seed(int(config["seed"]) + 1)
    model = build_pilotnet().to(device)
    dataset = PilotDataset(fixed_rows, float(config["max_steering_rad"]))
    images, targets = next(iter(DataLoader(dataset, batch_size=count, shuffle=False)))
    images, targets = images.to(device), targets.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    criterion = torch.nn.MSELoss()
    losses: list[float] = []
    model.train()
    for _ in range(int(config["tiny_overfit_epochs"])):
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = criterion(predictions, targets)
        if not torch.isfinite(loss):
            raise GateFailure("tiny-overfit: non-finite loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    initial = losses[0]
    final = losses[-1]
    ratio = final / initial if initial else 0.0
    required = float(config["tiny_overfit_required_loss_ratio"])
    if not final < initial or ratio > required:
        raise GateFailure(
            f"tiny-overfit: loss did not decrease enough: initial={initial:.8g}, "
            f"final={final:.8g}, ratio={ratio:.6f}, required<={required}"
        )
    return {
        "result": "PASS", "samples": count, "epochs": len(losses),
        "initial_normalized_mse": initial, "final_normalized_mse": final,
        "final_to_initial_ratio": ratio, "required_maximum_ratio": required,
    }


def _epoch_loss(model, loader, criterion, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            loss = criterion(predictions, targets)
            if not torch.isfinite(loss):
                raise GateFailure("full training: encountered non-finite loss")
            if training:
                loss.backward()
                optimizer.step()
            batch = images.shape[0]
            total += float(loss.detach().cpu()) * batch
            count += batch
    return total / count


def train_baseline(
    train_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
    checkpoint_path: Path,
) -> tuple[Any, dict[str, Any], list[dict[str, float]]]:
    set_reproducible_seed(int(config["seed"]))
    model = build_pilotnet().to(device)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        PilotDataset(train_rows, float(config["max_steering_rad"])),
        batch_size=int(config["batch_size"]), shuffle=True, generator=generator, num_workers=0,
    )
    validation_loader = DataLoader(
        PilotDataset(validation_rows, float(config["max_steering_rad"])),
        batch_size=int(config["batch_size"]), shuffle=False, num_workers=0,
    )
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]))
    best_loss = math.inf
    best_epoch = 0
    best_train_loss = math.inf
    stale = 0
    history: list[dict[str, float]] = []
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, int(config["max_epochs"]) + 1):
        train_loss = _epoch_loss(model, train_loader, criterion, device, optimizer)
        validation_loss = _epoch_loss(model, validation_loader, criterion, device)
        history.append({"epoch": epoch, "train_normalized_mse": train_loss, "validation_normalized_mse": validation_loss})
        print(json.dumps(history[-1], sort_keys=True), flush=True)
        improvement = best_loss - validation_loss
        if improvement > float(config.get("minimum_improvement", 0.0)):
            best_loss = validation_loss
            best_train_loss = train_loss
            best_epoch = epoch
            stale = 0
            torch.save({
                "model_state_dict": model.state_dict(), "epoch": epoch,
                "train_normalized_mse": train_loss, "validation_normalized_mse": validation_loss,
                "parameter_count": PILOTNET_PARAMETER_COUNT, "training_config": config,
            }, checkpoint_path)
        else:
            stale += 1
            if stale >= int(config["early_stopping_patience"]):
                break
    if not checkpoint_path.is_file() or best_epoch == 0:
        raise GateFailure("full training: no best checkpoint was produced")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, {
        "result": "PASS", "epochs_completed": len(history), "best_epoch": best_epoch,
        "train_normalized_mse_at_best": best_train_loss,
        "validation_normalized_mse_at_best": best_loss,
        "early_stopped": len(history) < int(config["max_epochs"]),
    }, history


def predict_rows(model, rows: Sequence[dict[str, Any]], config: dict[str, Any], device: torch.device):
    loader = DataLoader(
        PilotDataset(rows, float(config["max_steering_rad"])),
        batch_size=int(config["batch_size"]), shuffle=False, num_workers=0,
    )
    predictions: list[float] = []
    labels: list[float] = []
    model.eval()
    with torch.no_grad():
        for images, normalized_targets in loader:
            output = model(images.to(device)).cpu().numpy().reshape(-1)
            predictions.extend(steering_normalized_to_rad(output, float(config["max_steering_rad"])).tolist())
            labels.extend(steering_normalized_to_rad(normalized_targets.numpy().reshape(-1), float(config["max_steering_rad"])).tolist())
    return np.asarray(predictions, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def error_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    if predictions.shape != labels.shape or labels.size == 0:
        raise ValueError("error metrics require equal, non-empty prediction and label arrays")
    errors = predictions - labels
    absolute = np.abs(errors)
    return {
        "sample_count": int(labels.size),
        "mae_rad": float(np.mean(absolute)),
        "rmse_rad": float(np.sqrt(np.mean(errors * errors))),
        "bias_mean_signed_error_rad": float(np.mean(errors)),
        "max_absolute_error_rad": float(np.max(absolute)),
    }


def validate_offline(model, rows, config, device):
    predictions, labels = predict_rows(model, rows, config, device)
    overall = error_metrics(predictions, labels)
    overall["normalized_mae"] = overall["mae_rad"] / float(config["max_steering_rad"])
    threshold = float(config["near_zero_steering_rad"])
    masks = {
        "negative": labels < -threshold,
        "near_zero": np.abs(labels) <= threshold,
        "positive": labels > threshold,
    }
    groups = {name: error_metrics(predictions[mask], labels[mask]) for name, mask in masks.items()}
    return {"result": "PASS", "overall": overall, "groups": groups}, predictions, labels


def export_onnx(model, path: Path, config: dict[str, Any]) -> None:
    import onnx
    path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu").eval()
    example = torch.zeros((1, 3, IMAGE_HEIGHT, IMAGE_WIDTH), dtype=torch.float32)
    torch.onnx.export(
        model_cpu, example, path, export_params=True, opset_version=int(config["onnx_opset"]),
        do_constant_folding=True, input_names=["camera_yuv"], output_names=["steering_normalized"],
        dynamic_axes={"camera_yuv": {0: "batch"}, "steering_normalized": {0: "batch"}},
        dynamo=False,
    )
    checked = onnx.load(path)
    onnx.checker.check_model(checked)
    input_dims = checked.graph.input[0].type.tensor_type.shape.dim
    fixed = [dimension.dim_value for dimension in input_dims[1:]]
    if fixed != [3, IMAGE_HEIGHT, IMAGE_WIDTH]:
        raise GateFailure(f"ONNX export: invalid input shape {fixed}")
    output_dims = checked.graph.output[0].type.tensor_type.shape.dim
    if output_dims[-1].dim_value != 1:
        raise GateFailure("ONNX export: output shape is not [batch, 1]")


def validate_onnx_equivalence(model, rows, path: Path, config: dict[str, Any]) -> dict[str, Any]:
    import onnxruntime as ort
    count = min(int(config["onnx_equivalence_samples"]), len(rows))
    inputs = np.stack([preprocess_png(row["image_path"]) for row in rows[:count]])
    with torch.no_grad():
        pytorch = model.to("cpu")(torch.from_numpy(inputs)).numpy().reshape(-1)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_predictions = session.run(["steering_normalized"], {"camera_yuv": inputs})[0].reshape(-1)
    difference = np.abs(pytorch - onnx_predictions)
    mean = float(np.mean(difference))
    maximum = float(np.max(difference))
    if mean > float(config["onnx_mean_abs_difference_limit"]) or maximum > float(config["onnx_max_abs_difference_limit"]):
        raise GateFailure(f"ONNX equivalence: mean={mean:.8g}, max={maximum:.8g}")
    return {
        "result": "PASS", "samples": count, "mean_absolute_difference_normalized": mean,
        "max_absolute_difference_normalized": maximum,
        "mean_absolute_difference_rad": mean * float(config["max_steering_rad"]),
        "max_absolute_difference_rad": maximum * float(config["max_steering_rad"]),
    }


def _latency_summary(values_s: Iterable[float]) -> dict[str, float | int]:
    values = np.asarray(list(values_s), dtype=np.float64) * 1000.0
    return {
        "iterations": int(values.size), "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
    }


def benchmark_host_cpu(path: Path, sample_row: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    image = np.expand_dims(preprocess_png(sample_row["image_path"]), 0)
    for _ in range(int(config["benchmark_warmup_iterations"])):
        session.run(["steering_normalized"], {"camera_yuv": image})
    inference: list[float] = []
    for _ in range(int(config["benchmark_iterations"])):
        started = time.perf_counter()
        session.run(["steering_normalized"], {"camera_yuv": image})
        inference.append(time.perf_counter() - started)
    preprocessing: list[float] = []
    for _ in range(max(100, int(config["benchmark_iterations"]) // 2)):
        started = time.perf_counter()
        preprocess_png(sample_row["image_path"])
        preprocessing.append(time.perf_counter() - started)
    return {
        "label": "HOST CPU BENCHMARK — NOT Raspberry Pi 5 performance",
        "provider": session.get_providers()[0],
        "warmup_iterations": int(config["benchmark_warmup_iterations"]),
        "onnx_inference": _latency_summary(inference),
        "png_preprocessing": _latency_summary(preprocessing),
        "camera_period_ms": 66.0,
    }


def create_plots(history, predictions, labels, plots_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot([row["epoch"] for row in history], [row["train_normalized_mse"] for row in history], label="train")
    axis.plot([row["epoch"] for row in history], [row["validation_normalized_mse"] for row in history], label="validation")
    axis.set(xlabel="epoch", ylabel="normalized MSE", title="PilotNet V1 loss")
    axis.legend()
    fig.tight_layout()
    loss_path = plots_dir / "loss_curve.png"
    fig.savefig(loss_path, dpi=120)
    plt.close(fig)
    paths.append(str(loss_path))
    fig, axis = plt.subplots(figsize=(9, 3))
    axis.plot(labels, label="ground truth", linewidth=1)
    axis.plot(predictions, label="prediction", linewidth=1)
    axis.set(xlabel="episode_003 frame", ylabel="steering (rad)", title="Validation steering trace")
    axis.legend()
    fig.tight_layout()
    trace_path = plots_dir / "validation_trace.png"
    fig.savefig(trace_path, dpi=120)
    plt.close(fig)
    paths.append(str(trace_path))
    fig, axis = plt.subplots(figsize=(4, 4))
    axis.scatter(labels, predictions, s=5, alpha=0.4)
    bounds = (-0.36, 0.36)
    axis.plot(bounds, bounds, color="black", linewidth=1)
    axis.set(xlim=bounds, ylim=bounds, xlabel="label (rad)", ylabel="prediction (rad)", title="Prediction vs label")
    fig.tight_layout()
    scatter_path = plots_dir / "validation_scatter.png"
    fig.savefig(scatter_path, dpi=120)
    plt.close(fig)
    paths.append(str(scatter_path))
    return paths


def run_offline_pipeline(config_path: Path, dataset_root: Path, artifact_root: Path, result_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    artifact_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report: dict[str, Any] = {
        "version": "pilotnet_training_v1", "generated_utc": datetime.now(timezone.utc).isoformat(),
        "gate_reached": "environment", "result": "FAIL",
        "environment": {
            "python": __import__("platform").python_version(), "torch": torch.__version__,
            "device": str(device), "cuda_available": torch.cuda.is_available(),
            "numpy": np.__version__,
        },
        "architecture": {
            "input_shape": [3, IMAGE_HEIGHT, IMAGE_WIDTH], "output": "steering_normalized",
            "parameter_count": PILOTNET_PARAMETER_COUNT,
            "feature_shape": [64, 1, 18],
        },
        "training_config_sha256": sha256_file(config_path),
    }
    try:
        integrity = validate_dataset_integrity(dataset_root, config)
        train_rows = integrity.pop("train_rows")
        validation_rows = integrity.pop("validation_rows")
        report["dataset_integrity"] = integrity
        report["gate_reached"] = "dataset_integrity"
        tiny = tiny_overfit_sanity(train_rows, config, device)
        report["tiny_overfit"] = tiny
        report["gate_reached"] = "tiny_overfit"
        checkpoint_path = artifact_root / "checkpoints" / "pilotnet_v1_best.pt"
        model, training, history = train_baseline(train_rows, validation_rows, config, device, checkpoint_path)
        report["training"] = training
        maximum = float(config["max_steering_rad"])
        report["training"]["train_mse_rad2_at_best"] = training["train_normalized_mse_at_best"] * maximum**2
        report["training"]["validation_mse_rad2_at_best"] = training["validation_normalized_mse_at_best"] * maximum**2
        report["gate_reached"] = "full_training"
        validation, predictions, labels = validate_offline(model, validation_rows, config, device)
        report["offline_validation"] = validation
        report["gate_reached"] = "offline_validation"
        onnx_path = artifact_root / "onnx" / "pilotnet_v1.onnx"
        export_onnx(model, onnx_path, config)
        report["gate_reached"] = "onnx_export"
        equivalence = validate_onnx_equivalence(model, validation_rows, onnx_path, config)
        report["onnx_equivalence"] = equivalence
        report["gate_reached"] = "onnx_equivalence"
        benchmark = benchmark_host_cpu(onnx_path, validation_rows[0], config)
        report["host_cpu_benchmark"] = benchmark
        report["plots"] = create_plots(history, predictions, labels, artifact_root / "plots")
        report["artifacts"] = {
            "checkpoint": {"path": str(checkpoint_path), "size_bytes": checkpoint_path.stat().st_size, "sha256": sha256_file(checkpoint_path)},
            "onnx": {"path": str(onnx_path), "size_bytes": onnx_path.stat().st_size, "sha256": sha256_file(onnx_path)},
        }
        report["gate_reached"] = "host_cpu_benchmark"
        report["result"] = "PASS"
    except Exception as exc:
        report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_offline_pipeline(args.config, args.dataset_root, args.artifact_root, args.result)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"HARD GATE FAILURE: {exc}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
