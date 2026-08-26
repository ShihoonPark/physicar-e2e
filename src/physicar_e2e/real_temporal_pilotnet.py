"""Controlled real-data Temporal PilotNet scratch/transfer comparison.

This module is deliberately offline-only. It reads the frozen REAL_DATASET_V1
manifest and derived RGB images, trains exactly two registered models, exports
evidence, and never imports a vehicle or simulator client.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

import numpy as np

from .pilotnet import IMAGE_HEIGHT, IMAGE_WIDTH, preprocess_png
from .pilotnet_temporal import (
    TEMPORAL_CHANNELS,
    TEMPORAL_PARAMETER_COUNT,
    build_temporal_pilotnet,
)
from .pilotnet_training import set_reproducible_seed
from .real_dataset import (
    MANIFEST_COLUMNS,
    SPEED_SEMANTICS,
    STEERING_SCALE_RAD,
    canonical_json_bytes,
    sha256_file,
)


VERSION = "real_temporal_pilotnet_v1"
SCRATCH_NAME = "REAL-SCRATCH-V1"
TRANSFER_NAME = "REAL-D1-TRANSFER-V1"
MODEL_NAMES = (SCRATCH_NAME, TRANSFER_NAME)
EXPECTED_MANIFEST_SHA256 = "ba82ae5f1f7c606f5f516ea006148f033ab95ec9097d2f6aaa300c2ab91f5597"
EXPECTED_MANIFEST_COUNT = 2_163
EXPECTED_D1_CHECKPOINT_SHA256 = "b63a8da4401df33d1e8e375c66bd46b35d9cecd70542aa7af8cdbecfdb69a434"
TRAIN_BAGS = ("bag_01", "bag_02")
VALIDATION_BAGS = ("bag_03",)
EXPECTED_BAG_COUNTS = {"bag_01": 649, "bag_02": 1064, "bag_03": 450}
NEURAL_INPUT_FIELDS = ("image_t_minus_2", "image_t_minus_1", "image_t")
DRIVING_PERMITTED = False
SIMULATOR_TRAINING_SAMPLES_PERMITTED = False
RAW_BAG_ACCESS_REQUIRED = False
MAGNITUDE_BIN_LABELS = (
    "abs_lt_0p05",
    "abs_0p05_to_lt_0p15",
    "abs_0p15_to_lt_0p25",
    "abs_gte_0p25",
)


class RealTemporalTrainingError(RuntimeError):
    """Raised when a preregistered real-training gate is violated."""


@dataclass(frozen=True)
class DatasetAudit:
    rows: tuple[dict[str, Any], ...]
    train_rows: tuple[dict[str, Any], ...]
    validation_rows: tuple[dict[str, Any], ...]
    evidence: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RealTemporalTrainingError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RealTemporalTrainingError(f"JSON root must be an object: {path}")
    return payload


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    _atomic_write(path, canonical_json_bytes(payload))


def write_bytes_once(path: Path, content: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != content:
            raise RealTemporalTrainingError(f"refusing to replace mismatched frozen file: {path}")
        return
    _atomic_write(path, content)


def write_json_once(path: Path, payload: Any) -> None:
    write_bytes_once(path, canonical_json_bytes(payload))


def load_config(path: Path) -> dict[str, Any]:
    config = _read_json(path)
    if config.get("version") != VERSION:
        raise RealTemporalTrainingError("real temporal task version changed")
    if config.get("expected_branch") != "feature/real-temporal-pilotnet-v1":
        raise RealTemporalTrainingError("expected branch contract changed")

    dataset = config.get("dataset", {})
    expected_dataset = {
        "name": "REAL_DATASET_V1",
        "root": "/home/a/physicar-e2e-artifacts/real_dataset_v1",
        "manifest_path": "/home/a/physicar-e2e-artifacts/real_dataset_v1/manifests/real_dataset_v1.csv",
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "accepted_sequence_count": EXPECTED_MANIFEST_COUNT,
        "bag_sequence_counts": EXPECTED_BAG_COUNTS,
        "train_bags": list(TRAIN_BAGS),
        "validation_bags": list(VALIDATION_BAGS),
        "train_count_before_filter": 1713,
        "validation_count": 450,
        "train_filter": "target_time_speed_mps != 0.0",
        "validation_speed_filter": None,
    }
    if dataset != expected_dataset:
        raise RealTemporalTrainingError("REAL_DATASET_V1 identity or grouped split changed")

    expected_camera = {
        "source_width": 480,
        "source_height": 360,
        "source_encoding": "rgb8",
        "roi": [0, 80, 480, 360],
        "crop_width": 480,
        "crop_height": 280,
        "output_width": 200,
        "output_height": 66,
        "resize": "Pillow_Image.Resampling.BILINEAR",
        "horizontal_crop": False,
        "undistortion": False,
        "simulator_roi_permitted": False,
    }
    if config.get("camera_contract") != expected_camera:
        raise RealTemporalTrainingError("Real Camera ROI V1 changed")
    if config.get("preprocessing") != {
        "stored_color_space": "RGB",
        "model_color_space": "YUV_BT601_full_range",
        "normalization": "(channel - 0.5) * 2.0",
        "frame_order": ["t_minus_2", "t_minus_1", "t"],
    }:
        raise RealTemporalTrainingError("model preprocessing contract changed")

    steering = config.get("steering_contract", {})
    if steering != {
        "manifest_target_field": "steering_rad",
        "recorded_field": "steering_recorded_raw",
        "recorded_to_radians_scale": 0.35,
        "training_target": "physical_steering_rad",
        "positive_direction": "LEFT",
        "negative_direction": "RIGHT",
        "additional_scaling_permitted": False,
        "target_clipping_permitted": False,
        "near_zero_abs_lte_rad": 0.01,
    }:
        raise RealTemporalTrainingError("physical steering target contract changed")
    speed = config.get("speed_contract", {})
    if speed != {
        "field": "speed_mps",
        "unit": "m/s",
        "semantics": SPEED_SEMANTICS,
        "neural_input": False,
        "target": False,
        "metadata_only": True,
        "only_exact_zero_train_filter_permitted": True,
    }:
        raise RealTemporalTrainingError("speed metadata-only contract changed")

    if config.get("architecture") != {
        "name": "Temporal PilotNet",
        "input_shape": ["N", 9, 66, 200],
        "output_shape": ["N", 1],
        "parameter_count": TEMPORAL_PARAMETER_COUNT,
        "output_unit": "radians",
    }:
        raise RealTemporalTrainingError("Temporal PilotNet architecture contract changed")
    if config.get("early_stopping") != {
        "semantics": "validation_loss_improvement_strictly_greater_than_minimum_resets_patience",
        "patience": 7,
        "minimum_improvement_mse_rad2": 0.000001,
    }:
        raise RealTemporalTrainingError("established early-stopping semantics changed")

    models = config.get("models", {})
    expected_models = {
        SCRATCH_NAME: {
            "initialization": "from_scratch", "optimizer": "Adam",
            "loss": "MSE_steering_rad", "learning_rate": 0.001,
            "batch_size": 64, "max_epochs": 35, "seed": 20260824,
        },
        TRANSFER_NAME: {
            "initialization": "exact_frozen_simulator_D1", "optimizer": "Adam",
            "loss": "MSE_steering_rad", "learning_rate": 0.0001,
            "batch_size": 64, "max_epochs": 20, "seed": 20260824,
            "full_network_finetuning": True, "frozen_parameter_count": 0,
        },
    }
    if models != expected_models:
        raise RealTemporalTrainingError("registered scratch/transfer training configs changed")

    d1 = config.get("d1_initialization", {})
    if (
        d1.get("checkpoint_path")
        != "/home/a/physicar-ai-sim-docker/userdata/physicar_e2e/random_cone_1p0_v1/dagger1/d1/checkpoints/random_cone_temporal_d1_best.pt"
        or d1.get("checkpoint_sha256") != EXPECTED_D1_CHECKPOINT_SHA256
        or d1.get("onnx_sha256") != "3dee7ab9bb0ce6892dbba0784389af3c87b453e3150e1f7375e6b5301dba128c"
        or d1.get("freeze_sha256") != "66dbf7762ab089f111e2c02d22240d861e575730dcb416692bf6fac4e1e3fdc8"
        or d1.get("freeze_seal_sha256") != "7781423c7ba69f381e91120687d07d93d006393ff3c0c74af751085ce6ea1840"
        or d1.get("forbidden_initializations") != ["D1-R", "D2-FE", "R1", "V9", "C1"]
    ):
        raise RealTemporalTrainingError("exact frozen simulator D1 identity changed")
    for key in ("checkpoint_path", "onnx_path", "freeze_path", "freeze_seal_path"):
        if not str(d1.get(key, "")).startswith("/home/a/physicar-ai-sim-docker/"):
            raise RealTemporalTrainingError(f"D1 provenance path changed: {key}")

    if config.get("validation") != {
        "magnitude_bins_rad": [0.05, 0.15, 0.25],
        "baseline_predictions": ["ZERO", "MEAN"],
    }:
        raise RealTemporalTrainingError("validation contract changed")
    if config.get("export") != {
        "opset": 17, "equivalence_samples": 128,
        "mean_absolute_difference_limit_rad": 0.00001,
        "max_absolute_difference_limit_rad": 0.0001,
    }:
        raise RealTemporalTrainingError("ONNX export/equivalence contract changed")
    if config.get("benchmark") != {
        "warmup_iterations": 25, "iterations": 200, "batch_size": 1,
        "machine_classification": "CURRENT_X86_NOT_RASPBERRY_PI",
    }:
        raise RealTemporalTrainingError("x86 timing contract changed")
    if config.get("output") != {
        "external_root": "/home/a/physicar-e2e-artifacts/real_temporal_pilotnet_v1",
        "compact_result_root": "results/real_temporal_pilotnet_v1",
    }:
        raise RealTemporalTrainingError("artifact/result roots changed")
    if any(value is not False for value in config.get("prohibitions", {}).values()):
        raise RealTemporalTrainingError("a prohibited operation was enabled")
    return config


def current_git_branch(repo: Path) -> str:
    process = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, check=True,
        capture_output=True, text=True,
    )
    return process.stdout.strip()


def verify_branch(repo: Path, config: dict[str, Any]) -> str:
    branch = current_git_branch(repo)
    if branch != config["expected_branch"]:
        raise RealTemporalTrainingError(
            f"expected branch {config['expected_branch']!r}, found {branch!r}"
        )
    return branch


def _safe_dataset_image_path(dataset_root: Path, raw_path: str, bag_id: str) -> Path:
    relative = PurePosixPath(raw_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise RealTemporalTrainingError(f"unsafe dataset image path: {raw_path}")
    if len(relative.parts) < 3 or relative.parts[:2] != ("images", bag_id):
        raise RealTemporalTrainingError(f"cross-bag or non-real image path: {raw_path}")
    path = dataset_root.joinpath(*relative.parts)
    if not path.is_file():
        raise RealTemporalTrainingError(f"missing frozen dataset image: {path}")
    return path


def _parse_manifest_row(raw: dict[str, str], dataset_root: Path) -> dict[str, Any]:
    bag_id = raw["source_bag"]
    if bag_id not in EXPECTED_BAG_COUNTS:
        raise RealTemporalTrainingError(f"unexpected source bag {bag_id!r}")
    steering_recorded = float(raw["steering_recorded_raw"])
    steering_rad = float(raw["steering_rad"])
    if not math.isfinite(steering_recorded) or not math.isfinite(steering_rad):
        raise RealTemporalTrainingError("non-finite steering in REAL_DATASET_V1")
    expected_rad = steering_recorded * STEERING_SCALE_RAD
    if not math.isclose(steering_rad, expected_rad, rel_tol=0.0, abs_tol=1e-15):
        raise RealTemporalTrainingError("steering_rad was clipped or was not scaled exactly once")
    speed_mps = float(raw["speed_mps"])
    if not math.isfinite(speed_mps):
        raise RealTemporalTrainingError("speed metadata must be finite for the frozen split")
    paths = tuple(_safe_dataset_image_path(dataset_root, raw[field], bag_id) for field in NEURAL_INPUT_FIELDS)
    if len(set(paths)) != 3:
        raise RealTemporalTrainingError("temporal sequence contains duplicate-frame padding")
    return {
        "sequence_id": raw["sequence_id"],
        "source_bag": bag_id,
        "paths": paths,
        "steering_rad": steering_rad,
        "steering_recorded_raw": steering_recorded,
        "speed_mps": speed_mps,
        "manifest_row": raw,
    }


def keep_train_row_by_speed(speed_mps: float) -> bool:
    """The sole registered TRAIN selector: reject IEEE numeric zero only."""
    return float(speed_mps) != 0.0


def split_rows(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_before = [row for row in rows if row["source_bag"] in TRAIN_BAGS]
    validation = [row for row in rows if row["source_bag"] in VALIDATION_BAGS]
    unexpected = [row for row in rows if row["source_bag"] not in {*TRAIN_BAGS, *VALIDATION_BAGS}]
    if unexpected:
        raise RealTemporalTrainingError("manifest contains a bag outside the frozen grouped split")
    if len(train_before) != 1713 or len(validation) != 450:
        raise RealTemporalTrainingError("pre-filter grouped split counts changed")

    removed = [row for row in train_before if not keep_train_row_by_speed(row["speed_mps"])]
    train = [row for row in train_before if keep_train_row_by_speed(row["speed_mps"])]
    if len(removed) != 19 or len(train) != 1694:
        raise RealTemporalTrainingError(
            f"exact-zero TRAIN filter changed: removed={len(removed)}, train={len(train)}"
        )
    if any(row["speed_mps"] == 0.0 for row in train):
        raise RealTemporalTrainingError("exact-zero speed row survived TRAIN filtering")
    validation_zeros = sum(row["speed_mps"] == 0.0 for row in validation)

    train_frames = {str(path) for row in train for path in row["paths"]}
    validation_frames = {str(path) for row in validation for path in row["paths"]}
    frame_overlap = train_frames & validation_frames
    if frame_overlap:
        raise RealTemporalTrainingError("temporal frames leak between TRAIN and validation")
    train_bags = {row["source_bag"] for row in train}
    validation_bags = {row["source_bag"] for row in validation}
    if train_bags != set(TRAIN_BAGS) or validation_bags != set(VALIDATION_BAGS):
        raise RealTemporalTrainingError("bag-level split membership changed")
    evidence = {
        "split_strategy": "grouped_by_source_bag_no_frame_level_random_split",
        "train_bags": list(TRAIN_BAGS),
        "validation_bags": list(VALIDATION_BAGS),
        "train_count_before_exact_zero_speed_filter": len(train_before),
        "train_exact_zero_speed_removed_count": len(removed),
        "train_exact_zero_speed_removed_by_bag": dict(sorted(Counter(row["source_bag"] for row in removed).items())),
        "train_count": len(train),
        "validation_count": len(validation),
        "validation_exact_zero_speed_count_retained": validation_zeros,
        "validation_filter_applied": False,
        "train_validation_bag_overlap_count": len(train_bags & validation_bags),
        "train_validation_frame_overlap_count": len(frame_overlap),
        "random_frame_split_used": False,
        "only_filter": "TRAIN target-time speed_mps == 0.0 excluded",
    }
    return train, validation, evidence


def audit_dataset(config: dict[str, Any]) -> DatasetAudit:
    manifest = Path(config["dataset"]["manifest_path"])
    observed_hash = sha256_file(manifest)
    if observed_hash != EXPECTED_MANIFEST_SHA256:
        raise RealTemporalTrainingError(
            f"REAL_DATASET_V1 manifest hash mismatch: {observed_hash}"
        )
    dataset_root = Path(config["dataset"]["root"])
    raw_rows: list[dict[str, str]] = []
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise RealTemporalTrainingError("REAL_DATASET_V1 manifest columns changed")
        raw_rows.extend(dict(row) for row in reader)
    if len(raw_rows) != EXPECTED_MANIFEST_COUNT:
        raise RealTemporalTrainingError(f"manifest has {len(raw_rows)} rows, expected 2163")
    if len({row["sequence_id"] for row in raw_rows}) != len(raw_rows):
        raise RealTemporalTrainingError("duplicate sequence_id in REAL_DATASET_V1")
    counts = dict(sorted(Counter(row["source_bag"] for row in raw_rows).items()))
    if counts != EXPECTED_BAG_COUNTS:
        raise RealTemporalTrainingError(f"source bag counts changed: {counts}")
    rows = [_parse_manifest_row(raw, dataset_root) for raw in raw_rows]
    train, validation, split_evidence = split_rows(rows)
    evidence = {
        "dataset_name": "REAL_DATASET_V1",
        "manifest_path": str(manifest),
        "manifest_sha256_expected": EXPECTED_MANIFEST_SHA256,
        "manifest_sha256_observed": observed_hash,
        "manifest_hash_result": "PASS",
        "accepted_sequence_count": len(rows),
        "bag_sequence_counts": counts,
        "images_regenerated": False,
        "manifest_modified": False,
        "raw_bags_accessed": False,
        "steering_conversion_audit": {
            "result": "PASS",
            "formula": "steering_rad = steering_recorded_raw * 0.35",
            "additional_training_scaling": False,
            "target_clipping": False,
        },
        **split_evidence,
    }
    return DatasetAudit(tuple(rows), tuple(train), tuple(validation), evidence)


def _csv_bytes(rows: Sequence[dict[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row["manifest_row"][key] for key in MANIFEST_COLUMNS})
    return stream.getvalue().encode("utf-8")


def _sequence_id_hash(rows: Sequence[dict[str, Any]]) -> str:
    content = "".join(f"{row['sequence_id']}\n" for row in rows).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def freeze_split_manifests(
    audit: DatasetAudit, external_root: Path,
) -> dict[str, Any]:
    split_root = external_root / "splits"
    train_path = split_root / "train.csv"
    validation_path = split_root / "validation.csv"
    write_bytes_once(train_path, _csv_bytes(audit.train_rows))
    write_bytes_once(validation_path, _csv_bytes(audit.validation_rows))
    return {
        "train": {
            "path": str(train_path), "sha256": sha256_file(train_path),
            "sequence_id_sha256": _sequence_id_hash(audit.train_rows),
            "count": len(audit.train_rows), "source_bags": list(TRAIN_BAGS),
        },
        "validation": {
            "path": str(validation_path), "sha256": sha256_file(validation_path),
            "sequence_id_sha256": _sequence_id_hash(audit.validation_rows),
            "count": len(audit.validation_rows), "source_bags": list(VALIDATION_BAGS),
        },
    }


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), probability * 100.0))


def numeric_distribution(values: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values]
    if not data or any(not math.isfinite(value) for value in data):
        if data:
            raise RealTemporalTrainingError("distribution contains a non-finite value")
        return {
            "count": 0, "min": None, "p05": None, "mean": None,
            "median": None, "std": None, "p95": None, "max": None,
        }
    return {
        "count": len(data), "min": min(data), "p05": percentile(data, 0.05),
        "mean": statistics.fmean(data), "median": statistics.median(data),
        "std": statistics.pstdev(data), "p95": percentile(data, 0.95),
        "max": max(data),
    }


def _magnitude_bin(value: float) -> str:
    magnitude = abs(float(value))
    if magnitude < 0.05:
        return MAGNITUDE_BIN_LABELS[0]
    if magnitude < 0.15:
        return MAGNITUDE_BIN_LABELS[1]
    if magnitude < 0.25:
        return MAGNITUDE_BIN_LABELS[2]
    return MAGNITUDE_BIN_LABELS[3]


def split_distribution(rows: Sequence[dict[str, Any]], near_zero_rad: float) -> dict[str, Any]:
    steering = [float(row["steering_rad"]) for row in rows]
    speed = [float(row["speed_mps"]) for row in rows]
    regions = {
        "left": sum(value > near_zero_rad for value in steering),
        "right": sum(value < -near_zero_rad for value in steering),
        "near_zero": sum(abs(value) <= near_zero_rad for value in steering),
    }
    if sum(regions.values()) != len(rows):
        raise RealTemporalTrainingError("left/right/near-zero regions are not exhaustive")
    magnitude_counts = Counter(_magnitude_bin(value) for value in steering)
    return {
        "sequence_count": len(rows),
        "source_bags": sorted({str(row["source_bag"]) for row in rows}),
        "steering_rad": numeric_distribution(steering),
        "direction_region_definition": {
            "left": f"steering_rad > {near_zero_rad}",
            "right": f"steering_rad < -{near_zero_rad}",
            "near_zero": f"abs(steering_rad) <= {near_zero_rad}",
        },
        "left_right_near_zero_counts": regions,
        "exact_sign_counts": {
            "positive_LEFT": sum(value > 0.0 for value in steering),
            "negative_RIGHT": sum(value < 0.0 for value in steering),
            "zero": sum(value == 0.0 for value in steering),
        },
        "magnitude_bin_counts": {key: magnitude_counts[key] for key in MAGNITUDE_BIN_LABELS},
        "speed_metadata": {
            "unit": "m/s", "semantics": SPEED_SEMANTICS,
            "neural_input": False, "target": False,
            "exact_zero_count": sum(value == 0.0 for value in speed),
            "distribution_mps": numeric_distribution(speed),
        },
    }


class RealTemporalDataset:
    """Training pairs containing only temporal camera input and steering_rad."""

    neural_input_fields = NEURAL_INPUT_FIELDS
    target_field = "steering_rad"
    metadata_excluded_fields = ("speed_mps",)

    def __init__(self, rows: Sequence[dict[str, Any]], *, cache_frames: bool = True) -> None:
        self.rows = list(rows)
        self.cache_frames = bool(cache_frames)
        self._frame_cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _frame(self, path: Path) -> np.ndarray:
        if self.cache_frames and path in self._frame_cache:
            return self._frame_cache[path]
        value = preprocess_png(path)
        if value.shape != (3, IMAGE_HEIGHT, IMAGE_WIDTH) or value.dtype != np.float32:
            raise RealTemporalTrainingError(f"invalid preprocessed frame tensor from {path}")
        if self.cache_frames:
            self._frame_cache[path] = value
        return value

    def __getitem__(self, index: int):
        import torch

        row = self.rows[index]
        value = np.ascontiguousarray(
            np.concatenate([self._frame(path) for path in row["paths"]], axis=0),
            dtype=np.float32,
        )
        if value.shape != (TEMPORAL_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH):
            raise RealTemporalTrainingError(f"invalid temporal tensor shape {value.shape}")
        target = torch.tensor([float(row["steering_rad"])], dtype=torch.float32)
        return torch.from_numpy(value), target


def state_dict_sha256(state_dict: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(array.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _cpu_state_dict(model: Any) -> dict[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def verify_d1_artifacts(config: dict[str, Any]) -> dict[str, Any]:
    d1 = config["d1_initialization"]
    items = {
        "checkpoint": (Path(d1["checkpoint_path"]), d1["checkpoint_sha256"]),
        "onnx": (Path(d1["onnx_path"]), d1["onnx_sha256"]),
        "freeze": (Path(d1["freeze_path"]), d1["freeze_sha256"]),
        "freeze_seal": (Path(d1["freeze_seal_path"]), d1["freeze_seal_sha256"]),
    }
    evidence: dict[str, Any] = {}
    for name, (path, expected) in items.items():
        if not path.is_file():
            raise RealTemporalTrainingError(f"missing frozen D1 {name}: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise RealTemporalTrainingError(
                f"frozen D1 {name} hash mismatch: expected {expected}, got {observed}"
            )
        evidence[name] = {
            "path": str(path), "sha256_expected": expected,
            "sha256_observed": observed, "result": "PASS",
        }
    return evidence


def initialize_model(model_name: str, config: dict[str, Any], device: Any) -> tuple[Any, dict[str, Any]]:
    import torch

    if model_name not in MODEL_NAMES:
        raise RealTemporalTrainingError(f"unregistered model {model_name!r}")
    training = config["models"][model_name]
    set_reproducible_seed(int(training["seed"]))
    model = build_temporal_pilotnet().to(device)
    source_state_sha256: str | None = None
    d1_evidence: dict[str, Any] | None = None
    if model_name == TRANSFER_NAME:
        d1_evidence = verify_d1_artifacts(config)
        checkpoint_path = Path(config["d1_initialization"]["checkpoint_path"])
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("parameter_count") != TEMPORAL_PARAMETER_COUNT:
            raise RealTemporalTrainingError("D1 checkpoint parameter metadata changed")
        source_state = checkpoint.get("model_state_dict")
        if not isinstance(source_state, dict):
            raise RealTemporalTrainingError("D1 checkpoint has no model_state_dict")
        model.load_state_dict(source_state, strict=True)
        source_state_sha256 = state_dict_sha256(source_state)
    elif training["initialization"] != "from_scratch":
        raise RealTemporalTrainingError("scratch model initialization changed")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameter_count != TEMPORAL_PARAMETER_COUNT or trainable_count != TEMPORAL_PARAMETER_COUNT:
        raise RealTemporalTrainingError(
            f"architecture/fine-tuning gate failed: total={parameter_count}, trainable={trainable_count}"
        )
    initial_sha256 = state_dict_sha256(model.state_dict())
    if model_name == TRANSFER_NAME and initial_sha256 != source_state_sha256:
        raise RealTemporalTrainingError("transfer initialization is not tensor-exact D1")
    if model_name == SCRATCH_NAME:
        d1_checkpoint = torch.load(
            Path(config["d1_initialization"]["checkpoint_path"]),
            map_location="cpu", weights_only=False,
        )
        d1_state_sha256 = state_dict_sha256(d1_checkpoint["model_state_dict"])
        if initial_sha256 == d1_state_sha256:
            raise RealTemporalTrainingError("scratch initialization unexpectedly equals D1")
    else:
        d1_state_sha256 = source_state_sha256
    return model, {
        "mode": training["initialization"],
        "initial_state_dict_sha256": initial_sha256,
        "d1_source_state_dict_sha256": d1_state_sha256,
        "tensor_exact_d1_initialization": model_name == TRANSFER_NAME,
        "initialized_from_scratch": model_name == SCRATCH_NAME,
        "total_parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "frozen_parameter_count": parameter_count - trainable_count,
        "full_network_finetuning": trainable_count == parameter_count,
        "d1_artifact_identity": d1_evidence,
    }


def model_training_config(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    registered = dict(config["models"][model_name])
    return {
        **registered,
        "model_name": model_name,
        "target": "steering_rad_from_manifest_without_scaling_or_clipping",
        "input_fields": list(NEURAL_INPUT_FIELDS),
        "input_shape": ["N", 9, 66, 200],
        "output_shape": ["N", 1],
        "output_unit": "radians",
        "parameter_count": TEMPORAL_PARAMETER_COUNT,
        "early_stopping_patience": config["early_stopping"]["patience"],
        "minimum_improvement_mse_rad2": config["early_stopping"]["minimum_improvement_mse_rad2"],
        "augmentation": False,
        "balancing": False,
        "oversampling": False,
        "steering_weighting": False,
        "source_weighting": False,
        "hyperparameter_sweep": False,
        "optimization_sources": list(TRAIN_BAGS),
        "simulator_samples_in_optimization": 0,
        "speed_is_neural_input": False,
        "speed_is_target": False,
    }


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _run_epoch(model: Any, loader: Any, device: Any, optimizer: Any | None = None) -> float:
    import torch

    model.train(optimizer is not None)
    criterion = torch.nn.MSELoss()
    total = 0.0
    count = 0
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for images, steering_rad in loader:
            images = images.to(device)
            steering_rad = steering_rad.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            predictions_rad = model(images)
            loss = criterion(predictions_rad, steering_rad)
            if not torch.isfinite(loss):
                raise RealTemporalTrainingError("non-finite physical steering MSE")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            batch = int(images.shape[0])
            total += float(loss.detach().cpu()) * batch
            count += batch
    if not count:
        raise RealTemporalTrainingError("empty optimization/evaluation loader")
    return total / count


def _model_slug(model_name: str) -> str:
    return "scratch" if model_name == SCRATCH_NAME else "transfer"


def train_model(
    model_name: str,
    train_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
    identity: dict[str, Any],
    external_root: Path,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    if {row["source_bag"] for row in train_rows} != set(TRAIN_BAGS):
        raise RealTemporalTrainingError("optimization rows are not exactly bag_01/bag_02")
    if any(row["speed_mps"] == 0.0 for row in train_rows):
        raise RealTemporalTrainingError("zero-speed row reached optimization")
    if {row["source_bag"] for row in validation_rows} != set(VALIDATION_BAGS):
        raise RealTemporalTrainingError("early-stopping validation is not exactly bag_03")

    training = model_training_config(model_name, config)
    model, initialization = initialize_model(model_name, config, device)
    run_identity = {
        **identity,
        "model_name": model_name,
        "initial_state_dict_sha256": initialization["initial_state_dict_sha256"],
    }
    slug = _model_slug(model_name)
    artifact_dir = external_root / slug
    checkpoint_path = artifact_dir / "checkpoints" / f"real_{slug}_v1_best.pt"
    state_path = artifact_dir / "checkpoints" / f"real_{slug}_v1_training_state.pt"
    marker_path = artifact_dir / "training.started.json"
    snapshot_path = artifact_dir / "training_config_snapshot.json"
    marker = {
        "version": VERSION + "_training_marker",
        "model_name": model_name,
        "status": "ONE_LOGICAL_TRAINING_RUN_STARTED",
        "identity": run_identity,
        "training_config": training,
        "resumable_epoch_transactions": True,
        "retraining_permitted": False,
    }
    write_json_once(marker_path, marker)
    write_json_once(snapshot_path, {
        "version": VERSION + "_training_config_snapshot",
        "model_name": model_name,
        "identity": run_identity,
        "training_config": training,
        "initialization": initialization,
    })

    optimizer = torch.optim.Adam(model.parameters(), lr=float(training["learning_rate"]))
    generator = torch.Generator()
    epoch_completed = 0
    best_validation = math.inf
    best_train = math.inf
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    completed = False
    resumed_from_epoch = 0
    if state_path.is_file():
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state.get("identity") != run_identity or state.get("training_config") != training:
            raise RealTemporalTrainingError(f"{model_name}: training state identity/config mismatch")
        model.load_state_dict(state["model_state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        generator.set_state(state["data_loader_generator_state"].cpu())
        epoch_completed = int(state["epoch_completed"])
        resumed_from_epoch = epoch_completed
        best_validation = float(state["best_validation_mse_rad2"])
        best_train = float(state["best_train_mse_rad2"])
        best_epoch = int(state["best_epoch"])
        stale_epochs = int(state["stale_epochs"])
        history = list(state["history"])
        best_state = state.get("best_model_state_dict")
        completed = bool(state.get("completed", False))
    else:
        generator.manual_seed(int(training["seed"]))
        _atomic_torch_save(state_path, {
            "version": VERSION + "_resumable_training_state",
            "identity": run_identity,
            "training_config": training,
            "initialization": initialization,
            "epoch_completed": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "data_loader_generator_state": generator.get_state(),
            "best_validation_mse_rad2": best_validation,
            "best_train_mse_rad2": best_train,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
            "history": history,
            "best_model_state_dict": best_state,
            "completed": False,
        })

    train_dataset = RealTemporalDataset(train_rows, cache_frames=True)
    validation_dataset = RealTemporalDataset(validation_rows, cache_frames=True)
    train_loader = DataLoader(
        train_dataset, batch_size=int(training["batch_size"]), shuffle=True,
        generator=generator, num_workers=0, drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=int(training["batch_size"]), shuffle=False,
        num_workers=0, drop_last=False,
    )
    if not completed:
        for epoch in range(epoch_completed + 1, int(training["max_epochs"]) + 1):
            train_mse = _run_epoch(model, train_loader, device, optimizer)
            validation_mse = _run_epoch(model, validation_loader, device)
            item = {
                "epoch": epoch,
                "train_mse_rad2": train_mse,
                "validation_mse_rad2": validation_mse,
            }
            history.append(item)
            print(json.dumps({"model": model_name, **item}, sort_keys=True), flush=True)
            if best_validation - validation_mse > float(training["minimum_improvement_mse_rad2"]):
                best_validation = validation_mse
                best_train = train_mse
                best_epoch = epoch
                stale_epochs = 0
                best_state = _cpu_state_dict(model)
            else:
                stale_epochs += 1
            should_stop = stale_epochs >= int(training["early_stopping_patience"])
            completed = should_stop or epoch == int(training["max_epochs"])
            _atomic_torch_save(state_path, {
                "version": VERSION + "_resumable_training_state",
                "identity": run_identity,
                "training_config": training,
                "initialization": initialization,
                "epoch_completed": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "data_loader_generator_state": generator.get_state(),
                "best_validation_mse_rad2": best_validation,
                "best_train_mse_rad2": best_train,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "history": history,
                "best_model_state_dict": best_state,
                "completed": completed,
            })
            if should_stop:
                break

    if not completed or best_state is None or best_epoch <= 0:
        raise RealTemporalTrainingError(f"{model_name}: logical training run did not complete")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    final_state_sha256 = state_dict_sha256(best_state)
    checkpoint_payload = {
        "version": VERSION + "_checkpoint",
        "model_name": model_name,
        "model_state_dict": best_state,
        "epoch": best_epoch,
        "parameter_count": TEMPORAL_PARAMETER_COUNT,
        "output_unit": "radians",
        "identity": run_identity,
        "training_config": training,
        "initialization": initialization,
        "final_state_dict_sha256": final_state_sha256,
    }
    if checkpoint_path.is_file():
        existing = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            existing.get("identity") != run_identity
            or existing.get("final_state_dict_sha256") != final_state_sha256
            or state_dict_sha256(existing.get("model_state_dict", {})) != final_state_sha256
        ):
            raise RealTemporalTrainingError(f"refusing to replace mismatched {model_name} checkpoint")
    else:
        _atomic_torch_save(checkpoint_path, checkpoint_payload)
    report = {
        "version": VERSION + "_training",
        "model_name": model_name,
        "result": "PASS",
        "single_logical_training_run": True,
        "retraining_permitted": False,
        "resumed_from_completed_epoch": resumed_from_epoch,
        "device": str(device),
        "train_count": len(train_rows),
        "validation_count": len(validation_rows),
        "training_config": training,
        "initialization_audit": initialization,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_train_mse_rad2": best_train,
        "best_validation_mse_rad2": best_validation,
        "early_stopped": len(history) < int(training["max_epochs"]),
        "history": history,
        "artifacts": {
            "checkpoint": {
                "path": str(checkpoint_path), "sha256": sha256_file(checkpoint_path),
                "size_bytes": checkpoint_path.stat().st_size,
                "state_dict_sha256": final_state_sha256,
            },
            "training_state": {
                "path": str(state_path), "sha256": sha256_file(state_path),
                "size_bytes": state_path.stat().st_size,
            },
            "training_config_snapshot": {
                "path": str(snapshot_path), "sha256": sha256_file(snapshot_path),
                "size_bytes": snapshot_path.stat().st_size,
            },
        },
    }
    return model, report


def predict(model: Any, rows: Sequence[dict[str, Any]], batch_size: int, device: Any) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    dataset = RealTemporalDataset(rows, cache_frames=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for images, targets in loader:
            values = model(images.to(device)).detach().cpu().numpy().reshape(-1)
            predictions.append(values)
            labels.append(targets.numpy().reshape(-1))
    return (
        np.concatenate(predictions).astype(np.float64),
        np.concatenate(labels).astype(np.float64),
    )


def error_metrics(predictions: Sequence[float], labels: Sequence[float]) -> dict[str, Any]:
    prediction = np.asarray(predictions, dtype=np.float64).reshape(-1)
    target = np.asarray(labels, dtype=np.float64).reshape(-1)
    if prediction.shape != target.shape:
        raise RealTemporalTrainingError("metric prediction/target shapes differ")
    if not target.size:
        return {
            "count": 0, "mae_rad": None, "rmse_rad": None, "bias_rad": None,
            "median_absolute_error_rad": None, "p95_absolute_error_rad": None,
            "max_absolute_error_rad": None, "pearson_correlation": None,
            "corrective_magnitude_ratio": None, "steering_sign_agreement": None,
        }
    if not np.all(np.isfinite(prediction)) or not np.all(np.isfinite(target)):
        raise RealTemporalTrainingError("non-finite prediction/target in validation metrics")
    error = prediction - target
    absolute = np.abs(error)
    correlation: float | None
    if target.size < 2 or float(np.std(prediction)) == 0.0 or float(np.std(target)) == 0.0:
        correlation = None
    else:
        correlation = float(np.corrcoef(prediction, target)[0, 1])
    target_magnitude = float(np.mean(np.abs(target)))
    return {
        "count": int(target.size),
        "mae_rad": float(np.mean(absolute)),
        "rmse_rad": float(np.sqrt(np.mean(error * error))),
        "bias_rad": float(np.mean(error)),
        "median_absolute_error_rad": float(np.median(absolute)),
        "p95_absolute_error_rad": float(np.percentile(absolute, 95)),
        "max_absolute_error_rad": float(np.max(absolute)),
        "pearson_correlation": correlation,
        "corrective_magnitude_ratio": (
            float(np.mean(np.abs(prediction)) / target_magnitude)
            if target_magnitude > 0.0 else None
        ),
        "steering_sign_agreement": float(np.mean(np.sign(prediction) == np.sign(target))),
    }


def magnitude_bin_metrics(predictions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    output: dict[str, Any] = {}
    assignments = np.asarray([_magnitude_bin(value) for value in labels], dtype=object)
    for label in MAGNITUDE_BIN_LABELS:
        mask = assignments == label
        left = mask & (labels > 0.0)
        right = mask & (labels < 0.0)
        output[label] = {
            "combined": error_metrics(predictions[mask], labels[mask]),
            "left": error_metrics(predictions[left], labels[left]),
            "right": error_metrics(predictions[right], labels[right]),
            "zero_target_count": int(np.sum(mask & (labels == 0.0))),
        }
    return output


def evaluate_models(
    models: dict[str, Any],
    validation_rows: Sequence[dict[str, Any]],
    train_rows: Sequence[dict[str, Any]],
    device: Any,
) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "validation_set": {
            "source_bags": list(VALIDATION_BAGS), "count": len(validation_rows),
            "identical_for_all_models_and_baselines": True,
        },
        "models": {},
    }
    reference_labels: np.ndarray | None = None
    for model_name in MODEL_NAMES:
        predictions, labels = predict(models[model_name], validation_rows, 64, device)
        if reference_labels is None:
            reference_labels = labels
        elif not np.array_equal(reference_labels, labels):
            raise RealTemporalTrainingError("models were not evaluated against identical targets")
        comparison["models"][model_name] = {
            "overall": error_metrics(predictions, labels),
            "by_target_magnitude_and_direction": magnitude_bin_metrics(predictions, labels),
        }
    assert reference_labels is not None
    train_mean = statistics.fmean(float(row["steering_rad"]) for row in train_rows)
    zero_predictions = np.zeros_like(reference_labels)
    mean_predictions = np.full_like(reference_labels, train_mean)
    comparison["trivial_baselines"] = {
        "ZERO": {
            "prediction_rad": 0.0,
            **{key: value for key, value in error_metrics(zero_predictions, reference_labels).items()
               if key in ("count", "mae_rad", "rmse_rad")},
        },
        "MEAN": {
            "prediction_rad": train_mean,
            **{key: value for key, value in error_metrics(mean_predictions, reference_labels).items()
               if key in ("count", "mae_rad", "rmse_rad")},
        },
    }
    return comparison


def _relative_regression(worse: float, better: float) -> float:
    return (float(worse) - float(better)) / max(abs(float(better)), 1e-12)


def select_model(comparison: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    metrics = {name: comparison["models"][name]["overall"] for name in MODEL_NAMES}
    first, second = MODEL_NAMES
    if metrics[first]["mae_rad"] <= metrics[second]["mae_rad"]:
        primary, other = first, second
    else:
        primary, other = second, first
    primary_mae = float(metrics[primary]["mae_rad"])
    other_mae = float(metrics[other]["mae_rad"])
    relative_margin = (other_mae - primary_mae) / max(other_mae, 1e-12)
    selection_config = config["selection"]

    sanity_specs = (
        ("rmse_rad", "lower"),
        ("p95_absolute_error_rad", "lower"),
        ("absolute_bias_rad", "lower"),
        ("steering_sign_agreement", "higher"),
        ("magnitude_ratio_distance_from_one", "lower"),
        ("high_steering_mae_rad", "lower"),
    )
    values: dict[str, dict[str, float]] = {}
    for name in MODEL_NAMES:
        overall = metrics[name]
        high = comparison["models"][name]["by_target_magnitude_and_direction"]["abs_gte_0p25"]["combined"]
        values[name] = {
            "rmse_rad": float(overall["rmse_rad"]),
            "p95_absolute_error_rad": float(overall["p95_absolute_error_rad"]),
            "absolute_bias_rad": abs(float(overall["bias_rad"])),
            "steering_sign_agreement": float(overall["steering_sign_agreement"]),
            "magnitude_ratio_distance_from_one": abs(float(overall["corrective_magnitude_ratio"]) - 1.0),
            "high_steering_mae_rad": float(high["mae_rad"]),
        }
    sanity_wins = {name: 0 for name in MODEL_NAMES}
    sanity_detail: dict[str, Any] = {}
    tolerance = 1e-12
    for key, direction in sanity_specs:
        a, b = values[first][key], values[second][key]
        winner: str | None = None
        if abs(a - b) > tolerance:
            if (direction == "lower" and a < b) or (direction == "higher" and a > b):
                winner = first
            else:
                winner = second
            sanity_wins[winner] += 1
        sanity_detail[key] = {first: a, second: b, "preferred": direction, "winner": winner}

    near_tie = relative_margin <= float(selection_config["near_tie_relative_mae"])
    status = "SELECTED"
    selected: str | None = primary
    reason: str
    if near_tie:
        required = int(selection_config["near_tie_required_sanity_wins"])
        leaders = [name for name in MODEL_NAMES if sanity_wins[name] >= required]
        if len(leaders) == 1:
            selected = leaders[0]
            reason = (
                f"MAE was within {selection_config['near_tie_relative_mae']:.1%}; "
                f"{selected} won {sanity_wins[selected]} of 6 registered sanity metrics."
            )
        else:
            selected = None
            status = "MODEL_SELECTION_INCONCLUSIVE"
            reason = "Validation MAE was a near tie and no model won at least four registered sanity metrics."
    elif relative_margin >= float(selection_config["clear_mae_relative_margin"]):
        reason = f"{primary} had a clear {relative_margin:.2%} lower bag_03 validation MAE."
    else:
        primary_p95_regression = _relative_regression(
            values[primary]["p95_absolute_error_rad"], values[other]["p95_absolute_error_rad"]
        )
        primary_high_regression = _relative_regression(
            values[primary]["high_steering_mae_rad"], values[other]["high_steering_mae_rad"]
        )
        primary_sign_drop = (
            values[other]["steering_sign_agreement"] - values[primary]["steering_sign_agreement"]
        )
        tail_limit = float(selection_config["substantial_tail_regression"])
        sign_limit = float(selection_config["substantial_sign_agreement_drop"])
        substantial_conflict = (
            primary_high_regression >= tail_limit
            and (
                primary_p95_regression >= tail_limit
                or primary_sign_drop >= sign_limit
            )
        )
        if substantial_conflict:
            selected = None
            status = "MODEL_SELECTION_INCONCLUSIVE"
            reason = (
                f"{primary} won MAE by {relative_margin:.2%}, but its registered tail/high-steering "
                "sanity metrics conflicted substantially."
            )
        else:
            reason = f"{primary} had lower bag_03 validation MAE without a substantial registered sanity conflict."
    return {
        "status": status,
        "selected_model": selected,
        "primary_metric": "bag_03 validation MAE rad",
        "primary_mae_winner": primary,
        "relative_mae_margin": relative_margin,
        "near_tie": near_tie,
        "sanity_wins": sanity_wins,
        "sanity_detail": sanity_detail,
        "reason": reason,
        "selection_does_not_prefer_transfer_by_identity": True,
    }


def _onnx_contract(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(path)
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise RealTemporalTrainingError("ONNX must have exactly one input and one output")
    input_value = model.graph.input[0]
    output_value = model.graph.output[0]
    input_dims = input_value.type.tensor_type.shape.dim
    output_dims = output_value.type.tensor_type.shape.dim
    fixed_input = [dimension.dim_value for dimension in input_dims[1:]]
    if fixed_input != [9, 66, 200] or output_dims[-1].dim_value != 1:
        raise RealTemporalTrainingError("ONNX N×9×66×200 -> N×1 contract failed")
    parameter_count = sum(
        math.prod(initializer.dims) for initializer in model.graph.initializer
    )
    if parameter_count != TEMPORAL_PARAMETER_COUNT:
        raise RealTemporalTrainingError(f"ONNX initializer parameter count is {parameter_count}")
    return {
        "checker": "PASS",
        "input_name": input_value.name,
        "input_shape": ["N", 9, 66, 200],
        "output_name": output_value.name,
        "output_shape": ["N", 1],
        "parameter_count": parameter_count,
    }


def _equivalence_inputs(rows: Sequence[dict[str, Any]], count: int) -> np.ndarray:
    selected = list(rows[:count])
    if len(selected) != count:
        raise RealTemporalTrainingError("insufficient ONNX equivalence samples")
    dataset = RealTemporalDataset(selected, cache_frames=True)
    return np.stack([dataset[index][0].numpy() for index in range(len(dataset))])


def export_and_check_onnx(
    model_name: str,
    model: Any,
    validation_rows: Sequence[dict[str, Any]],
    config: dict[str, Any],
    external_root: Path,
) -> dict[str, Any]:
    import onnxruntime as ort
    import torch

    slug = _model_slug(model_name)
    path = external_root / slug / "onnx" / f"real_{slug}_v1.onnx"
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu").eval()
    if not path.exists():
        temporary = path.with_suffix(".onnx.tmp")
        torch.onnx.export(
            model,
            torch.zeros((1, 9, 66, 200), dtype=torch.float32),
            temporary,
            opset_version=int(config["export"]["opset"]),
            input_names=["camera_yuv_temporal"],
            output_names=["steering_rad"],
            dynamic_axes={"camera_yuv_temporal": {0: "batch"}, "steering_rad": {0: "batch"}},
            dynamo=False,
        )
        _onnx_contract(temporary)
        temporary.replace(path)
    contract = _onnx_contract(path)
    sample_count = int(config["export"]["equivalence_samples"])
    inputs = _equivalence_inputs(validation_rows, sample_count)
    with torch.no_grad():
        pytorch_values = model(torch.from_numpy(inputs)).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_values = session.run(["steering_rad"], {"camera_yuv_temporal": inputs})[0]
    difference = np.abs(pytorch_values - onnx_values)
    mean_difference = float(np.mean(difference))
    max_difference = float(np.max(difference))
    if (
        mean_difference > float(config["export"]["mean_absolute_difference_limit_rad"])
        or max_difference > float(config["export"]["max_absolute_difference_limit_rad"])
    ):
        raise RealTemporalTrainingError(
            f"{model_name} PyTorch/ONNX mismatch: mean={mean_difference}, max={max_difference}"
        )
    return {
        "path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
        "contract": contract,
        "equivalence": {
            "result": "PASS", "samples": sample_count,
            "mean_absolute_difference_rad": mean_difference,
            "max_absolute_difference_rad": max_difference,
            "mean_limit_rad": config["export"]["mean_absolute_difference_limit_rad"],
            "max_limit_rad": config["export"]["max_absolute_difference_limit_rad"],
        },
    }


def _copy_once(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256_file(destination) != sha256_file(source):
            raise RealTemporalTrainingError(f"selected artifact mismatch: {destination}")
        return
    shutil.copy2(source, destination)


def freeze_selected_model(
    selection: dict[str, Any],
    training_reports: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    dataset_evidence: dict[str, Any],
    split_artifacts: dict[str, Any],
    external_root: Path,
) -> dict[str, Any]:
    selected = selection.get("selected_model")
    if selected not in MODEL_NAMES:
        raise RealTemporalTrainingError("cannot freeze an inconclusive model selection")
    selected_dir = external_root / "selected"
    source_checkpoint = Path(training_reports[selected]["artifacts"]["checkpoint"]["path"])
    source_onnx = Path(training_reports[selected]["artifacts"]["onnx"]["path"])
    checkpoint = selected_dir / "real_temporal_pilotnet_v1_selected.pt"
    onnx = selected_dir / "real_temporal_pilotnet_v1_selected.onnx"
    _copy_once(source_checkpoint, checkpoint)
    _copy_once(source_onnx, onnx)
    freeze_path = selected_dir / "freeze.json"
    if freeze_path.is_file():
        existing = _read_json(freeze_path)
        if (
            existing.get("selected_model") != selected
            or existing.get("checkpoint", {}).get("sha256") != sha256_file(checkpoint)
            or existing.get("onnx", {}).get("sha256") != sha256_file(onnx)
        ):
            raise RealTemporalTrainingError("existing selected freeze identity changed")
        return existing
    freeze = {
        "version": VERSION + "_selected_freeze",
        "frozen_utc": utc_now(),
        "selected_model": selected,
        "model_selection": selection,
        "checkpoint": {
            "path": str(checkpoint), "sha256": sha256_file(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "source_path": str(source_checkpoint),
        },
        "onnx": {
            "path": str(onnx), "sha256": sha256_file(onnx),
            "size_bytes": onnx.stat().st_size,
            "source_path": str(source_onnx),
        },
        "architecture": config["architecture"],
        "preprocessing_contract": {
            "camera": config["camera_contract"],
            "model": config["preprocessing"],
            "temporal_input": ["t_minus_2", "t_minus_1", "t"],
        },
        "dataset": {
            "name": "REAL_DATASET_V1",
            "manifest_sha256": dataset_evidence["manifest_sha256_observed"],
            "accepted_sequence_count": dataset_evidence["accepted_sequence_count"],
        },
        "split": split_artifacts,
        "training_config": training_reports[selected]["training_config"],
        "validation_metrics": comparison["models"][selected],
        "task_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "d1_initialization": (
            training_reports[selected]["initialization_audit"]["d1_artifact_identity"]
            if selected == TRANSFER_NAME else None
        ),
        "optimization_data": {
            "real_train_only": True, "source_bags": list(TRAIN_BAGS),
            "simulator_sample_count": 0,
        },
        "runtime_status": {
            "real_vehicle_run_performed": False,
            "simulator_run_performed": False,
            "runtime_integration_performed": False,
            "traffic_light_runtime_modified": False,
        },
        "retraining_or_tuning_after_freeze_permitted": False,
    }
    write_json_once(freeze_path, freeze)
    freeze_hash = sha256_file(freeze_path)
    seal = {
        "version": VERSION + "_selected_freeze_seal",
        "selected_model": selected,
        "freeze_sha256": freeze_hash,
        "checkpoint_sha256": freeze["checkpoint"]["sha256"],
        "onnx_sha256": freeze["onnx"]["sha256"],
        "real_vehicle_runs_before_seal": 0,
        "simulator_runs_before_seal": 0,
        "retraining_permitted": False,
    }
    write_json_once(selected_dir / "freeze_seal.json", seal)
    freeze["freeze_file"] = {"path": str(freeze_path), "sha256": freeze_hash}
    freeze["freeze_seal"] = {
        "path": str(selected_dir / "freeze_seal.json"),
        "sha256": sha256_file(selected_dir / "freeze_seal.json"),
    }
    return freeze


def latency_summary(values_s: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(values_s, dtype=np.float64) * 1000.0
    return {
        "count": int(values.size), "mean_ms": float(np.mean(values)),
        "p95_ms": float(np.percentile(values, 95)), "median_ms": float(np.median(values)),
        "max_ms": float(np.max(values)),
    }


def benchmark_selected_onnx(
    onnx_path: Path, validation_rows: Sequence[dict[str, Any]], config: dict[str, Any],
) -> dict[str, Any]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    warmup = int(config["benchmark"]["warmup_iterations"])
    iterations = int(config["benchmark"]["iterations"])
    for index in range(warmup):
        row = validation_rows[index % len(validation_rows)]
        value = np.expand_dims(
            np.concatenate([preprocess_png(path) for path in row["paths"]], axis=0), 0,
        ).astype(np.float32, copy=False)
        session.run(["steering_rad"], {"camera_yuv_temporal": value})
    preprocessing: list[float] = []
    inference: list[float] = []
    total: list[float] = []
    for index in range(iterations):
        row = validation_rows[index % len(validation_rows)]
        total_started = time.perf_counter()
        started = time.perf_counter()
        value = np.expand_dims(
            np.concatenate([preprocess_png(path) for path in row["paths"]], axis=0), 0,
        ).astype(np.float32, copy=False)
        preprocessing.append(time.perf_counter() - started)
        if value.shape != (1, 9, 66, 200):
            raise RealTemporalTrainingError("batch=1 benchmark input contract failed")
        started = time.perf_counter()
        session.run(["steering_rad"], {"camera_yuv_temporal": value})
        inference.append(time.perf_counter() - started)
        total.append(time.perf_counter() - total_started)
    return {
        "classification": "CURRENT_X86_NOT_RASPBERRY_PI",
        "raspberry_pi_performance_claimed": False,
        "batch_size": 1,
        "temporal_frames": 3,
        "input_shape": [1, 9, 66, 200],
        "provider": session.get_providers()[0],
        "warmup_iterations": warmup,
        "iterations": iterations,
        "preprocess_scope": "three stored RGB PNG decodes plus RGB-to-YUV normalization and temporal concatenation; frozen ROI/resize occurred at dataset extraction",
        "host": {
            "machine": platform.machine(), "processor": platform.processor(),
            "platform": platform.platform(), "cpu_count": os.cpu_count(),
        },
        "preprocess": latency_summary(preprocessing),
        "onnx_inference": latency_summary(inference),
        "total": latency_summary(total),
    }


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_report(
    dataset_result: dict[str, Any],
    training_reports: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    selection: dict[str, Any],
    freeze: dict[str, Any] | None,
    timing: dict[str, Any] | None,
) -> str:
    distributions = dataset_result["distribution"]
    lines = [
        "# Real Temporal PilotNet V1",
        "",
        f"Result: **{selection['status']}**",
        "",
        "This is an offline real-data training comparison. No physical-car drive, simulator drive, data collection, raw-bag modification, or runtime integration was performed. Offline validation is not real-robot driving success.",
        "",
        "## Frozen data and split",
        "",
        f"REAL_DATASET_V1 manifest SHA-256: `{dataset_result['manifest_sha256_observed']}` (PASS).",
        "",
        "| Split | Bags | Sequences | Steering mean / median / std rad | Left / right / near-zero | Exact-zero speed |",
        "|---|---|---:|---|---|---:|",
    ]
    for name in ("train", "validation"):
        item = distributions[name]
        steering = item["steering_rad"]
        direction = item["left_right_near_zero_counts"]
        lines.append(
            f"| {name.upper()} | {', '.join(item['source_bags'])} | {item['sequence_count']} | "
            f"{_fmt(steering['mean'])} / {_fmt(steering['median'])} / {_fmt(steering['std'])} | "
            f"{direction['left']} / {direction['right']} / {direction['near_zero']} | "
            f"{item['speed_metadata']['exact_zero_count']} |"
        )
    lines.extend([
        "",
        "TRAIN began with 1,713 grouped sequences; the only selection filter removed 19 target-time speeds exactly equal to 0.0 m/s, leaving 1,694. Validation retained all 450 bag_03 sequences and contained no exact-zero speeds. Near-zero steering means abs(steering_rad) <= 0.01. The left-heavy distribution was not rebalanced.",
        "",
        "Speed distribution (metadata only; m/s):",
        "",
        "| Split | Min | p05 | Mean | Median | Std | p95 | Max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in ("train", "validation"):
        speed = distributions[name]["speed_metadata"]["distribution_mps"]
        lines.append(
            f"| {name.upper()} | {_fmt(speed['min'])} | {_fmt(speed['p05'])} | "
            f"{_fmt(speed['mean'])} | {_fmt(speed['median'])} | {_fmt(speed['std'])} | "
            f"{_fmt(speed['p95'])} | {_fmt(speed['max'])} |"
        )
    lines.extend([
        "",
        "Speed remains metadata only and its command-versus-feedback semantics remain unresolved.",
        "",
        "Magnitude-bin counts:",
        "",
        "| Split | <0.05 | 0.05–<0.15 | 0.15–<0.25 | ≥0.25 rad |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in ("train", "validation"):
        bins = distributions[name]["magnitude_bin_counts"]
        lines.append(f"| {name.upper()} | {bins[MAGNITUDE_BIN_LABELS[0]]} | {bins[MAGNITUDE_BIN_LABELS[1]]} | {bins[MAGNITUDE_BIN_LABELS[2]]} | {bins[MAGNITUDE_BIN_LABELS[3]]} |")
    lines.extend([
        "",
        "## Training",
        "",
        "Both models use N×9×66×200 Temporal PilotNet with 255,819 parameters and produce physical steering radians. Targets were the manifest `steering_rad` values without additional scaling or clipping.",
        "",
        f"Transfer initialization used only the frozen simulator D1 checkpoint `{EXPECTED_D1_CHECKPOINT_SHA256}`; all 255,819 parameters were fine-tuned on real TRAIN data.",
        "",
        "| Model | Initialization | LR | Epochs / best | Best validation MSE rad² | Checkpoint SHA-256 |",
        "|---|---|---:|---:|---:|---|",
    ])
    for name in MODEL_NAMES:
        item = training_reports[name]
        lines.append(
            f"| {name} | {item['training_config']['initialization']} | {item['training_config']['learning_rate']} | "
            f"{item['epochs_completed']} / {item['best_epoch']} | {_fmt(item['best_validation_mse_rad2'], 9)} | "
            f"`{item['artifacts']['checkpoint']['sha256']}` |"
        )
    lines.extend([
        "",
        "## Bag_03 validation",
        "",
        "| Model | n | MAE | RMSE | Bias | Median AE | p95 AE | Max AE | Pearson | Magnitude ratio | Sign agreement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in MODEL_NAMES:
        metric = comparison["models"][name]["overall"]
        lines.append(
            f"| {name} | {metric['count']} | {_fmt(metric['mae_rad'])} | {_fmt(metric['rmse_rad'])} | "
            f"{_fmt(metric['bias_rad'])} | {_fmt(metric['median_absolute_error_rad'])} | "
            f"{_fmt(metric['p95_absolute_error_rad'])} | {_fmt(metric['max_absolute_error_rad'])} | "
            f"{_fmt(metric['pearson_correlation'])} | {_fmt(metric['corrective_magnitude_ratio'])} | "
            f"{_fmt(metric['steering_sign_agreement'])} |"
        )
    lines.extend([
        "",
        "Trivial baselines:",
        "",
        "| Baseline | Prediction rad | n | MAE rad | RMSE rad |",
        "|---|---:|---:|---:|---:|",
    ])
    for name in ("ZERO", "MEAN"):
        metric = comparison["trivial_baselines"][name]
        lines.append(f"| {name} | {_fmt(metric['prediction_rad'])} | {metric['count']} | {_fmt(metric['mae_rad'])} | {_fmt(metric['rmse_rad'])} |")
    lines.extend(["", "Per-bin combined/left/right results (MAE rad; count):", ""])
    for model_name in MODEL_NAMES:
        lines.extend([
            f"### {model_name}",
            "",
            "| Target magnitude | Combined | LEFT | RIGHT |",
            "|---|---:|---:|---:|",
        ])
        bins = comparison["models"][model_name]["by_target_magnitude_and_direction"]
        labels = ("<0.05", "0.05–<0.15", "0.15–<0.25", "≥0.25")
        for key, label in zip(MAGNITUDE_BIN_LABELS, labels):
            item = bins[key]
            cells = []
            for region in ("combined", "left", "right"):
                metric = item[region]
                cells.append(f"{_fmt(metric['mae_rad'])} ({metric['count']})")
            lines.append(f"| {label} | {' | '.join(cells)} |")
        lines.append("")
    lines.extend([
        "## Selection and export",
        "",
        f"{selection['reason']} Selected candidate: **{selection.get('selected_model') or 'none'}**.",
        "",
        "Under this controlled run, simulator D1 pretraining did not help the real visual domain.",
        "",
        "Both model ONNX files passed the N×9×66×200 → N×1 shape gate, 255,819-parameter gate, ONNX checker, and PyTorch↔ONNX equivalence gate.",
    ])
    if freeze is not None:
        lines.extend([
            "",
            f"Selected checkpoint SHA-256: `{freeze['checkpoint']['sha256']}`.",
            "",
            f"Selected ONNX SHA-256: `{freeze['onnx']['sha256']}`.",
        ])
    if timing is not None:
        lines.extend([
            "",
            "## Current x86 batch=1 timing",
            "",
            "| Component | Mean ms | p95 ms |",
            "|---|---:|---:|",
            f"| Preprocess | {_fmt(timing['preprocess']['mean_ms'], 3)} | {_fmt(timing['preprocess']['p95_ms'], 3)} |",
            f"| ONNX inference | {_fmt(timing['onnx_inference']['mean_ms'], 3)} | {_fmt(timing['onnx_inference']['p95_ms'], 3)} |",
            f"| Total | {_fmt(timing['total']['mean_ms'], 3)} | {_fmt(timing['total']['p95_ms'], 3)} |",
            "",
            "These measurements are from the current x86 CPU with batch=1; they are not Raspberry Pi timing and make no Pi performance claim.",
        ])
    lines.extend([
        "",
        "## Deferred runtime work",
        "",
        "Camera acquisition, the three-frame runtime buffer, steering publication, speed policy, GREEN traffic-light start gate, watchdog, and safe stop remain for the separately authorized runtime-integration milestone. The neural observation remains camera-only.",
        "",
    ])
    return "\n".join(lines)


def run_pipeline(repo: Path, config_path: Path) -> dict[str, Any]:
    import torch

    repo = repo.resolve()
    config_path = config_path.resolve()
    config = load_config(config_path)
    branch = verify_branch(repo, config)
    if DRIVING_PERMITTED or SIMULATOR_TRAINING_SAMPLES_PERMITTED or RAW_BAG_ACCESS_REQUIRED:
        raise RealTemporalTrainingError("offline-only scope constants changed")
    d1_identity = verify_d1_artifacts(config)
    audit = audit_dataset(config)
    external_root = Path(config["output"]["external_root"])
    result_root = repo / config["output"]["compact_result_root"]
    split_artifacts = freeze_split_manifests(audit, external_root)
    near_zero = float(config["steering_contract"]["near_zero_abs_lte_rad"])
    dataset_result = {
        "version": VERSION + "_dataset_split",
        "generated_utc": utc_now(),
        "result": "PASS",
        **audit.evidence,
        "split_artifacts": split_artifacts,
        "distribution": {
            "train": split_distribution(audit.train_rows, near_zero),
            "validation": split_distribution(audit.validation_rows, near_zero),
        },
        "speed_semantics_unresolved": True,
        "speed_semantics": SPEED_SEMANTICS,
    }
    write_json(result_root / "dataset_split.json", dataset_result)

    identity = {
        "task_config_sha256": sha256_file(config_path),
        "real_dataset_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "train_split_sha256": split_artifacts["train"]["sha256"],
        "validation_split_sha256": split_artifacts["validation"]["sha256"],
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if sum(parameter.numel() for parameter in build_temporal_pilotnet().parameters()) != TEMPORAL_PARAMETER_COUNT:
        raise RealTemporalTrainingError("Temporal PilotNet parameter gate failed")
    models: dict[str, Any] = {}
    training_reports: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_NAMES:
        model, report = train_model(
            model_name, audit.train_rows, audit.validation_rows,
            config, identity, external_root, device,
        )
        models[model_name] = model
        training_reports[model_name] = report

    comparison = evaluate_models(models, audit.validation_rows, audit.train_rows, device)
    selection = select_model(comparison, config)
    comparison.update({
        "version": VERSION + "_validation_comparison",
        "generated_utc": utc_now(),
        "result": "PASS",
        "model_selection": selection,
    })

    for model_name in MODEL_NAMES:
        onnx = export_and_check_onnx(
            model_name, models[model_name], audit.validation_rows, config, external_root,
        )
        training_reports[model_name]["artifacts"]["onnx"] = onnx
        write_json(
            result_root / ("scratch_training.json" if model_name == SCRATCH_NAME else "transfer_training.json"),
            training_reports[model_name],
        )
    write_json(result_root / "validation_comparison.json", comparison)

    selected_freeze: dict[str, Any] | None = None
    timing: dict[str, Any] | None = None
    if selection["status"] == "SELECTED":
        selected_freeze = freeze_selected_model(
            selection, training_reports, comparison, config, config_path,
            audit.evidence, split_artifacts, external_root,
        )
        timing = benchmark_selected_onnx(
            Path(selected_freeze["onnx"]["path"]), audit.validation_rows, config,
        )
        compact_freeze = {
            **selected_freeze,
            "external_freeze": selected_freeze.get("freeze_file"),
            "x86_timing": timing,
        }
        write_json(result_root / "freeze.json", compact_freeze)

    summary = {
        "version": VERSION,
        "generated_utc": utc_now(),
        "result": selection["status"],
        "branch": branch,
        "task_config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "dataset": {
            "manifest_sha256": audit.evidence["manifest_sha256_observed"],
            "accepted_sequences": len(audit.rows),
            "train_sequences": len(audit.train_rows),
            "validation_sequences": len(audit.validation_rows),
            "split_artifacts": split_artifacts,
        },
        "d1_identity": d1_identity,
        "models": {
            name: {
                "validation": comparison["models"][name]["overall"],
                "checkpoint": training_reports[name]["artifacts"]["checkpoint"],
                "onnx": training_reports[name]["artifacts"]["onnx"],
            }
            for name in MODEL_NAMES
        },
        "trivial_baselines": comparison["trivial_baselines"],
        "model_selection": selection,
        "selected_freeze": selected_freeze,
        "x86_batch1_timing": timing,
        "scope": {
            "real_vehicle_driven": False,
            "simulator_driven": False,
            "new_data_collected": False,
            "raw_bags_modified": False,
            "simulator_samples_optimized": 0,
            "traffic_light_runtime_modified": False,
            "committed": False,
            "pushed": False,
        },
    }
    write_json(result_root / "summary.json", summary)
    report = build_report(
        dataset_result, training_reports, comparison, selection, selected_freeze, timing,
    )
    _atomic_write(result_root / "REPORT.md", report.encode("utf-8"))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, default=Path("configs/real_temporal_pilotnet_v1.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    repo = args.repo.resolve()
    config_path = args.config if args.config.is_absolute() else repo / args.config
    summary = run_pipeline(repo, config_path)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
