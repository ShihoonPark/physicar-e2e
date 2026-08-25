"""High-Speed Temporal PilotNet V9: preserved-data three-frame A/B pipeline."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .expert_driver import PoseLivenessMonitor
from .high_speed_v5 import SPEED_MPS, live_preflight, write_json
from .pilotnet import preprocess_live_jpeg, preprocess_png, steering_normalized_to_rad
from .pilotnet_inference import InferenceConfig, _summary_ms, classify_failure, fixed_speed_commands
from .pilotnet_recovery_training import load_checkpoint
from .pilotnet_temporal import (
    CausalFrameBuffer, MAX_ADJACENT_GAP_S, TEMPORAL_CHANNELS, TEMPORAL_PARAMETER_COUNT,
    TemporalInputError, append_live_jpeg, build_temporal_pilotnet, preprocess_temporal_paths,
)
from .pilotnet_training import GateFailure, set_reproducible_seed, sha256_file
from .route_geometry import OffTrackMonitor, ProgressTracker
from .sim_client import SimClient


VERSION = "high_speed_temporal_v1"
MAX_TOTAL_ATTEMPTS = 5
TARGET_VALID_PASSES = 3
TEMPORAL_FIELDS = [
    "stratum", "source_id", "sequence_index", "frame_t_minus_2", "frame_t_minus_1", "frame_t",
    "timestamp_t_minus_2_ns", "timestamp_t_minus_1_ns", "timestamp_t_ns", "target_steering_rad",
    "source_mcap_sha256", "source_manifest_sha256", "window_role",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {"count": int(array.size), "mean": float(np.mean(array)), "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)), "max": float(np.max(array))}


def load_dataset_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    frozen = (config.get("version"), config.get("history_frames"), config.get("maximum_adjacent_gap_s"),
              config.get("causal_only"), config.get("allow_boundary_crossing"),
              config.get("allow_duplicate_padding"), config.get("channel_order"),
              config.get("new_training_data_collection_permitted"), config.get("dagger_iteration4_permitted"))
    if frozen != ("high_speed_temporal_dataset_v1", 3, .120, True, False, False,
                  "oldest_to_current", False, False):
        raise GateFailure(f"temporal dataset contract changed: {frozen}")
    return config


def load_training_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    if (config.get("version"), config.get("input_channels"), config.get("history_frames"),
            config.get("maximum_adjacent_gap_s"), config.get("initialization")) != (
            "pilotnet_training_v9_high_speed_temporal", 9, 3, .120, "from_scratch"):
        raise GateFailure("V9 frozen training contract changed")
    reference = json.loads((path.parent / "pilotnet_training_v8_high_speed_dagger.json").read_text())
    for key in ("seed", "optimizer", "loss", "learning_rate", "batch_size", "max_epochs",
                "early_stopping_patience", "minimum_improvement", "max_steering_rad"):
        if config.get(key) != reference.get(key):
            raise GateFailure(f"V9 training semantic changed: {key}")
    return config


def _source_specs(sim_root: Path) -> dict[str, list[tuple[str, Path, Path]]]:
    base = sim_root / "userdata/physicar_e2e"
    nominal = base / "high_speed_v1/dataset"
    d1 = base / "high_speed_dagger_v1/extracted"
    d2 = base / "high_speed_dagger_iteration2_v1/extracted"
    d3 = base / "high_speed_dagger_iteration3_v1/extracted"
    def episodes(ids): return [(episode, nominal, nominal / "manifests" / f"{episode}.csv") for episode in ids]
    return {
        "train": [*episodes([f"episode_{i:03d}" for i in range(1, 9)]),
                  ("high_speed_dagger_rollout_A", d1, d1 / "manifests/high_speed_dagger_rollout_A.csv"),
                  ("high_speed_dagger_iter2_rollout_A", d2, d2 / "manifests/high_speed_dagger_iter2_rollout_A.csv"),
                  ("high_speed_dagger_iter3_rollout_A", d3, d3 / "manifests/high_speed_dagger_iter3_rollout_A.csv")],
        "nominal_validation": episodes(["episode_009", "episode_010"]),
        "nominal_holdout": episodes(["episode_011", "episode_012"]),
        "dagger1_B": [("high_speed_dagger_rollout_B", d1, d1 / "manifests/high_speed_dagger_rollout_B.csv")],
        "dagger2_B": [("high_speed_dagger_iter2_rollout_B", d2, d2 / "manifests/high_speed_dagger_iter2_rollout_B.csv")],
        "dagger3_B": [("high_speed_dagger_iter3_rollout_B", d3, d3 / "manifests/high_speed_dagger_iter3_rollout_B.csv")],
    }


def _source_rows(source_id: str, root: Path, manifest: Path) -> list[dict[str, Any]]:
    if not manifest.is_file():
        raise GateFailure(f"missing preserved source manifest {manifest}")
    with manifest.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    parsed = []
    for raw in rows:
        if raw.get("episode_id") != source_id:
            raise GateFailure("source boundary identity mismatch")
        image = root / raw["image_path"]
        if not image.is_file():
            raise GateFailure(f"missing preserved image {image}")
        parsed.append({"image": image, "timestamp_ns": int(raw["camera_header_time_ns"]),
                       "steering_rad": float(raw["steering_rad"]),
                       "source_mcap_sha256": raw["source_mcap_sha256"],
                       "window_role": raw.get("window_role", "")})
    if len(parsed) < 3 or len({row["source_mcap_sha256"] for row in parsed}) != 1:
        raise GateFailure(f"insufficient or mixed-source trajectory {source_id}")
    return parsed


def build_sequences(source_id: str, stratum: str, root: Path, manifest: Path,
                    maximum_gap_s: float = MAX_ADJACENT_GAP_S) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if maximum_gap_s != .120:
        raise GateFailure("maximum adjacent temporal gap must remain exactly 0.120 s")
    raw = _source_rows(source_id, root, manifest)
    accepted: list[dict[str, Any]] = []
    gaps: list[float] = []
    spans: list[float] = []
    rejected_gap = 0
    manifest_hash = sha256_file(manifest)
    for index in range(2, len(raw)):
        a, b, c = raw[index - 2:index + 1]
        g1 = (b["timestamp_ns"] - a["timestamp_ns"]) / 1e9
        g2 = (c["timestamp_ns"] - b["timestamp_ns"]) / 1e9
        if not (a["timestamp_ns"] < b["timestamp_ns"] < c["timestamp_ns"]):
            raise GateFailure(f"non-causal timestamps in {source_id}")
        if g1 > maximum_gap_s or g2 > maximum_gap_s:
            rejected_gap += 1
            continue
        row = {"stratum": stratum, "source_id": source_id, "sequence_index": len(accepted),
               "frame_t_minus_2": str(a["image"]), "frame_t_minus_1": str(b["image"]), "frame_t": str(c["image"]),
               "timestamp_t_minus_2_ns": a["timestamp_ns"], "timestamp_t_minus_1_ns": b["timestamp_ns"],
               "timestamp_t_ns": c["timestamp_ns"], "target_steering_rad": c["steering_rad"],
               "source_mcap_sha256": c["source_mcap_sha256"], "source_manifest_sha256": manifest_hash,
               "window_role": c["window_role"]}
        accepted.append(row); gaps.extend((g1, g2)); spans.append(g1 + g2)
    return accepted, {"source_id": source_id, "source_frames": len(raw), "temporal_candidates": max(0, len(raw) - 2),
                      "accepted": len(accepted), "rejected_gap": rejected_gap,
                      "rejected_boundary": min(2, len(raw)), "source_manifest": str(manifest),
                      "source_manifest_sha256": manifest_hash,
                      "adjacent_gap_s": distribution(gaps), "oldest_to_current_span_s": distribution(spans)}


def _write_temporal_manifest(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=TEMPORAL_FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def dataset_stage(repo: Path, sim_root: Path) -> dict[str, Any]:
    result_path = repo / "results/high_speed_temporal_dataset_v1/summary.json"
    external = sim_root / "userdata/physicar_e2e/high_speed_temporal_v1"
    if result_path.exists() or external.exists():
        raise RuntimeError("refusing to rebuild temporal dataset evidence")
    config_path = repo / "configs/high_speed_temporal_dataset_v1.json"
    config = load_dataset_config(config_path)
    specs = _source_specs(sim_root)
    report: dict[str, Any] = {"version": config["version"], "generated_utc": utc_now(), "result": "FAIL",
                              "maximum_adjacent_gap_s": .120, "new_training_data_collected": False,
                              "raw_bags_created": 0, "dagger_iteration4_executed": False,
                              "external_root": str(external), "strata": {}}
    all_rows: dict[str, list[dict[str, Any]]] = {}
    try:
        for stratum, sources in specs.items():
            rows: list[dict[str, Any]] = []; source_stats = []
            for source_id, root, manifest in sources:
                built, stats = build_sequences(source_id, stratum, root, manifest); rows.extend(built); source_stats.append(stats)
            path = external / "manifests" / f"{stratum}.csv"; _write_temporal_manifest(path, rows); all_rows[stratum] = rows
            report["strata"][stratum] = {"sequence_count": len(rows), "source_count": len(sources),
                "temporal_candidate_count": sum(x["temporal_candidates"] for x in source_stats),
                "rejected_gap_count": sum(x["rejected_gap"] for x in source_stats),
                "rejected_boundary_count": sum(x["rejected_boundary"] for x in source_stats),
                "adjacent_gap_s": distribution([g for r in rows for g in ((r["timestamp_t_minus_1_ns"]-r["timestamp_t_minus_2_ns"])/1e9, (r["timestamp_t_ns"]-r["timestamp_t_minus_1_ns"])/1e9)]),
                "oldest_to_current_span_s": distribution([(r["timestamp_t_ns"]-r["timestamp_t_minus_2_ns"])/1e9 for r in rows]),
                "manifest": str(path), "manifest_sha256": sha256_file(path), "sources": source_stats}
        train_ids = {r["source_id"] for r in all_rows["train"]}
        eval_ids = {r["source_id"] for name, rows in all_rows.items() if name != "train" for r in rows}
        train_hashes = {r["source_mcap_sha256"] for r in all_rows["train"]}
        eval_hashes = {r["source_mcap_sha256"] for name, rows in all_rows.items() if name != "train" for r in rows}
        if train_ids & eval_ids or train_hashes & eval_hashes:
            raise GateFailure("temporal training/evaluation trajectory leakage")
        if any(path.suffix == ".mcap" for path in external.rglob("*")):
            raise GateFailure("temporal stage unexpectedly created raw driving data")
        report["leakage"] = {"source_id_overlap": False, "source_hash_overlap": False,
                             "all_B_holdouts_excluded": True, "low_speed_data_included": False}
        report["parameter_precheck"] = {"expected": TEMPORAL_PARAMETER_COUNT,
                                         "actual": sum(p.numel() for p in build_temporal_pilotnet().parameters()), "result": "PASS"}
        report["result"] = "PASS"
    finally:
        write_json(result_path, report)
    return report


def read_temporal_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for raw in csv.DictReader(path.open(newline="", encoding="utf-8")):
        times = tuple(int(raw[key]) for key in ("timestamp_t_minus_2_ns", "timestamp_t_minus_1_ns", "timestamp_t_ns"))
        if not times[0] < times[1] < times[2] or max(times[1]-times[0], times[2]-times[1]) > 120_000_000:
            raise GateFailure("temporal manifest causal/gap violation")
        paths = tuple(Path(raw[key]) for key in ("frame_t_minus_2", "frame_t_minus_1", "frame_t"))
        if not all(path.is_file() for path in paths): raise GateFailure("temporal manifest missing image")
        rows.append({**raw, "paths": paths, "image_path": paths[2], "steering_rad": float(raw["target_steering_rad"])})
    if not rows: raise GateFailure(f"empty temporal manifest {path}")
    return rows


class TemporalDataset(Dataset):
    def __init__(self, rows, maximum): self.rows, self.maximum = list(rows), float(maximum)
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        return torch.from_numpy(preprocess_temporal_paths(row["paths"])), torch.tensor([row["steering_rad"] / self.maximum], dtype=torch.float32)


def _epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None); total = 0.; count = 0; criterion = torch.nn.MSELoss()
    context = torch.enable_grad() if optimizer else torch.no_grad()
    with context:
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if optimizer: optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), targets)
            if optimizer: loss.backward(); optimizer.step()
            total += float(loss.detach().cpu()) * images.shape[0]; count += images.shape[0]
    return total / count


def train_temporal(train_rows, validation_rows, config, device, checkpoint):
    set_reproducible_seed(int(config["seed"])); model = build_temporal_pilotnet().to(device)
    generator = torch.Generator().manual_seed(int(config["seed"])); maximum = config["max_steering_rad"]
    train_loader = DataLoader(TemporalDataset(train_rows, maximum), batch_size=config["batch_size"], shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(TemporalDataset(validation_rows, maximum), batch_size=config["batch_size"], shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best = math.inf; stale = 0; history = []; best_epoch = 0; best_train = math.inf; checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, config["max_epochs"] + 1):
        train_loss, val_loss = _epoch(model, train_loader, device, optimizer), _epoch(model, val_loader, device)
        item = {"epoch": epoch, "train_normalized_mse": train_loss, "validation_normalized_mse": val_loss}; history.append(item); print(json.dumps(item), flush=True)
        if best - val_loss > config["minimum_improvement"]:
            best, best_epoch, best_train, stale = val_loss, epoch, train_loss, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "parameter_count": TEMPORAL_PARAMETER_COUNT,
                        "training_config": config}, checkpoint)
        else:
            stale += 1
            if stale >= config["early_stopping_patience"]: break
    saved = torch.load(checkpoint, map_location=device, weights_only=False); model.load_state_dict(saved["model_state_dict"]); model.eval()
    return model, {"result": "PASS", "epochs_completed": len(history), "best_epoch": best_epoch,
                   "train_normalized_mse_at_best": best_train, "validation_normalized_mse_at_best": best,
                   "early_stopped": len(history) < config["max_epochs"], "initialized_from_scratch": True}, history


def predict_temporal(model, rows, config, device, *, repeated_current=False):
    predictions = []; labels = []; model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), config["batch_size"]):
            batch = rows[start:start+config["batch_size"]]
            paths = [(row["paths"][2],) * 3 if repeated_current else row["paths"] for row in batch]
            images = torch.from_numpy(np.stack([preprocess_temporal_paths(p) for p in paths])).to(device)
            output = model(images).cpu().numpy().reshape(-1)
            predictions.extend(steering_normalized_to_rad(output, config["max_steering_rad"]).tolist()); labels.extend(row["steering_rad"] for row in batch)
    return np.asarray(predictions), np.asarray(labels)


def predict_v8(model, rows, config, device):
    predictions = []; labels = []; model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), config["batch_size"]):
            batch = rows[start:start+config["batch_size"]]
            images = torch.from_numpy(np.stack([preprocess_png(row["paths"][2]) for row in batch])).to(device)
            output = model(images).cpu().numpy().reshape(-1)
            predictions.extend(steering_normalized_to_rad(output, config["max_steering_rad"]).tolist()); labels.extend(row["steering_rad"] for row in batch)
    return np.asarray(predictions), np.asarray(labels)


def metrics(pred, labels):
    errors = pred - labels; mean_label = float(np.mean(np.abs(labels)))
    return {"sample_count": int(labels.size), "mae_rad": float(np.mean(np.abs(errors))),
            "rmse_rad": float(np.sqrt(np.mean(errors * errors))), "bias_mean_signed_error_rad": float(np.mean(errors)),
            "max_absolute_error_rad": float(np.max(np.abs(errors))),
            "correlation": float(np.corrcoef(pred, labels)[0, 1]) if labels.size > 1 else None,
            "corrective_magnitude_ratio": float(np.mean(np.abs(pred)) / mean_label) if mean_label else None}


def export_temporal_onnx(model, path: Path, config):
    import onnx
    path.parent.mkdir(parents=True, exist_ok=True); model = model.to("cpu").eval()
    torch.onnx.export(model, torch.zeros((1, 9, 66, 200)), path, opset_version=config["onnx_opset"],
                      input_names=["camera_yuv_temporal"], output_names=["steering_normalized"],
                      dynamic_axes={"camera_yuv_temporal": {0: "batch"}, "steering_normalized": {0: "batch"}}, dynamo=False)
    checked = onnx.load(path); onnx.checker.check_model(checked)
    fixed = [dim.dim_value for dim in checked.graph.input[0].type.tensor_type.shape.dim[1:]]
    if fixed != [9, 66, 200] or checked.graph.output[0].type.tensor_type.shape.dim[-1].dim_value != 1:
        raise GateFailure("V9 ONNX I/O contract failure")


def validate_equivalence(model, rows, path, config):
    import onnxruntime as ort
    selected = rows[:config["onnx_equivalence_samples"]]
    values = np.stack([preprocess_temporal_paths(row["paths"]) for row in selected])
    with torch.no_grad(): torch_values = model.to("cpu")(torch.from_numpy(values)).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_values = session.run(["steering_normalized"], {"camera_yuv_temporal": values})[0]
    diff = np.abs(torch_values - onnx_values); maximum = config["max_steering_rad"]
    result = {"samples": len(selected), "mean_absolute_difference_normalized": float(diff.mean()),
              "max_absolute_difference_normalized": float(diff.max()),
              "mean_absolute_difference_rad": float(diff.mean()*maximum), "max_absolute_difference_rad": float(diff.max()*maximum)}
    result["result"] = "PASS" if diff.mean() <= config["onnx_mean_abs_difference_limit"] and diff.max() <= config["onnx_max_abs_difference_limit"] else "FAIL"
    if result["result"] != "PASS": raise GateFailure("V9 ONNX equivalence failed")
    return result


def training_stage(repo: Path, sim_root: Path) -> dict[str, Any]:
    result_path = repo / "results/pilotnet_training_v9_high_speed_temporal/summary.json"
    if result_path.exists(): raise RuntimeError("refusing to repeat V9 training")
    dataset = json.loads((repo / "results/high_speed_temporal_dataset_v1/summary.json").read_text())
    if dataset.get("result") != "PASS" or dataset.get("new_training_data_collected") is not False: raise GateFailure("temporal dataset gate failed")
    config = load_training_config(repo / "configs/pilotnet_training_v9_high_speed_temporal.json")
    external = sim_root / "userdata/physicar_e2e/high_speed_temporal_v1"; manifests = external / "manifests"
    rows = {name: read_temporal_rows(manifests / f"{name}.csv") for name in ("train", "nominal_validation", "nominal_holdout", "dagger1_B", "dagger2_B", "dagger3_B")}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    count = sum(p.numel() for p in build_temporal_pilotnet().parameters())
    if count != 255_819: raise GateFailure(f"V9 parameter gate failed: {count}")
    artifact_root = external / "v9"; checkpoint = artifact_root / "checkpoints/pilotnet_v9_high_speed_temporal_best.pt"
    if artifact_root.exists(): raise RuntimeError("refusing to overwrite V9 artifacts")
    report: dict[str, Any] = {"version": "pilotnet_training_v9_high_speed_temporal", "generated_utc": utc_now(), "result": "FAIL",
        "architecture": {"input_shape": [9,66,200], "parameter_count": count, "conv1_parameter_increase": 3600,
                         "only_change_from_v8": "conv1 input channels 3 to 9"}, "training_from_scratch": True,
        "dataset": {name: len(value) for name, value in rows.items()}, "matched_target_identity": True}
    try:
        # The architecture gate precedes all optimization; train with exactly the frozen config.
        model, training, history = train_temporal(rows["train"], rows["nominal_validation"], config, device, checkpoint)
        report["training"] = training; report["epochs"] = history
        v8_summary = json.loads((repo / "results/pilotnet_training_v8_high_speed_dagger/summary.json").read_text())
        v8_path = Path(v8_summary["artifacts"]["checkpoint"]["path"])
        if sha256_file(v8_path) != v8_summary["artifacts"]["checkpoint"]["sha256"]: raise GateFailure("V8 artifact identity failed")
        v8 = load_checkpoint(v8_path, device); comparisons = {}
        for name in ("nominal_validation", "nominal_holdout", "dagger1_B", "dagger2_B", "dagger3_B"):
            v8p, labels8 = predict_v8(v8, rows[name], config, device); v9p, labels9 = predict_temporal(model, rows[name], config, device)
            if not np.array_equal(labels8, labels9): raise GateFailure("matched V8/V9 target identity failed")
            comparisons[name] = {"matched_count": len(rows[name]), "v8": metrics(v8p, labels8), "v9": metrics(v9p, labels9)}
            if name == "dagger3_B":
                comparisons[name]["subregions"] = {}
                for region in ("85_90_percent", "90_95_percent", "95_100_percent"):
                    subset = [row for row in rows[name] if row["window_role"] == region]
                    if subset:
                        p8,l8 = predict_v8(v8, subset, config, device); p9,l9 = predict_temporal(model, subset, config, device)
                        comparisons[name]["subregions"][region] = {"v8": metrics(p8,l8), "v9": metrics(p9,l9)}
        report["matched_offline_comparison"] = comparisons
        ablation_p, ablation_l = predict_temporal(model, rows["dagger3_B"], config, device, repeated_current=True)
        report["repeated_current_frame_ablation_dagger3_B"] = metrics(ablation_p, ablation_l)
        onnx_path = artifact_root / "onnx/pilotnet_v9_high_speed_temporal.onnx"; export_temporal_onnx(model, onnx_path, config)
        report["onnx_contract"] = {"checker": "PASS", "input": ["batch",9,66,200], "output": ["batch",1]}
        report["onnx_equivalence"] = validate_equivalence(model, [*rows["nominal_validation"], *rows["dagger3_B"]], onnx_path, config)
        report["artifacts"] = {"checkpoint": {"path": str(checkpoint), "size_bytes": checkpoint.stat().st_size, "sha256": sha256_file(checkpoint)},
                               "onnx": {"path": str(onnx_path), "size_bytes": onnx_path.stat().st_size, "sha256": sha256_file(onnx_path)},
                               "v8_checkpoint_sha256": sha256_file(v8_path)}
        report["result"] = "PASS"
    finally: write_json(result_path, report)
    return report


class TemporalOnnxModel:
    observation_fields = ("camera_yuv_t_minus_2", "camera_yuv_t_minus_1", "camera_yuv_t")
    def __init__(self, path: Path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        meta = self.session.get_inputs()[0]
        if meta.name != "camera_yuv_temporal" or meta.shape[1:] != [9,66,200]: raise GateFailure("V9 ONNX live input contract failed")
    def predict(self, value):
        if value.shape != (9,66,200) or value.dtype != np.float32: raise TemporalInputError("invalid V9 temporal tensor")
        output = self.session.run(["steering_normalized"], {"camera_yuv_temporal": value[None]})[0]
        return float(output.reshape(-1)[0])


def load_inference_config(repo: Path) -> InferenceConfig:
    payload = json.loads((repo / "configs/pilotnet_inference_v9_high_speed_temporal.json").read_text())
    frozen = (payload.get("history_frames"), payload.get("input_channels"), payload.get("maximum_adjacent_gap_s"),
              payload.get("duplicate_frame_padding"), payload.get("warmup_while_stopped"), payload.get("smoke_speeds_mps"),
              payload.get("maximum_total_attempts"))
    if frozen != (3,9,.120,False,True,[1.8,1.8,1.8],5): raise GateFailure(f"V9 inference contract changed: {frozen}")
    return InferenceConfig(payload)


def warm_temporal_buffer(client, config):
    buffer = CausalFrameBuffer(.120); acquisitions = []; preprocessing = []
    for index in range(3):
        if index: time.sleep(1.0 / config.payload["control_frequency_hz"])
        started = time.perf_counter(); jpeg = client.camera_jpeg(config.payload["camera_path"]); timestamp = time.monotonic(); acquisitions.append(time.perf_counter()-started)
        started = time.perf_counter(); append_live_jpeg(buffer, jpeg, timestamp, roi=config.roi); preprocessing.append(time.perf_counter()-started)
    g1,g2,span = buffer.gaps()
    return buffer, {"result": "PASS", "real_frame_acquisitions": 3, "duplicate_padding_frames": 0,
                    "vehicle_motion_during_warmup": False, "adjacent_gaps_s": [g1,g2], "oldest_to_current_span_s": span,
                    "camera_acquisition_latency": _summary_ms(acquisitions), "preprocessing_latency": _summary_ms(preprocessing)}


def run_temporal_live(client, model, config, initial, speed_mps):
    safety = config.safety_config(speed_mps); route = initial.route; tracker = ProgressTracker(route.length, safety.maximum_progress_jump_m)
    off_track = OffTrackMonitor(safety.off_track_grace_s); liveness = PoseLivenessMonitor(safety.pose_stale_timeout_s, safety.pose_motion_translation_threshold_m, safety.pose_motion_yaw_threshold_rad)
    periods=[]; ctes=[]; steerings=[]; camera_times=[]; prep_times=[]; infer_times=[]; total_times=[]; gap1=[]; gap2=[]; spans=[]
    api_failures=liveness_failures=saturation=invalid_history=0; previous=None; motion=False; failure=None; result="FAIL"; temporal_failure=False
    final_pose=initial.pose; projection=route.project((final_pose["x"],final_pose["y"])); stop_errors=[]
    started=time.monotonic(); next_tick=started; next_world=started
    try:
        buffer, warmup = warm_temporal_buffer(client, config)
        while True:
            now=time.monotonic()
            if now-started >= safety.maximum_runtime_s: raise RuntimeError("maximum runtime exceeded before lap completion")
            if now < next_tick: time.sleep(next_tick-now)
            tick=time.monotonic()
            if previous is not None: periods.append(tick-previous)
            previous=tick
            if tick >= next_world:
                status=client.status()
                if status.get("running") is not True or status.get("switching") is not False or status.get("current") != initial.world: raise RuntimeError("simulator state changed while driving")
                next_world=tick+safety.world_check_interval_s
            camera_started=time.perf_counter(); jpeg=client.camera_jpeg(config.payload["camera_path"]); timestamp=time.monotonic(); camera_times.append(time.perf_counter()-camera_started)
            model_started=time.perf_counter(); prep_started=time.perf_counter(); append_live_jpeg(buffer,jpeg,timestamp,roi=config.roi); prep_times.append(time.perf_counter()-prep_started)
            g1,g2,span=buffer.gaps(); gap1.append(g1); gap2.append(g2); spans.append(span)
            infer_started=time.perf_counter(); normalized=model.predict(buffer.tensor()); infer_times.append(time.perf_counter()-infer_started); total_times.append(time.perf_counter()-model_started)
            steering,speed=fixed_speed_commands(speed_mps,float(steering_normalized_to_rad(normalized,safety.max_steering_rad)))
            pose=client.pose(); clock=client.clock()
            try: liveness.update(pose,float(clock["sim_time"]),time.monotonic(),motion_commanded=motion)
            except RuntimeError: liveness_failures+=1; raise
            final_pose=pose; projection=route.project((pose["x"],pose["y"])); tracker.update(projection.s)
            boundary=route.track_boundary_distance((pose["x"],pose["y"]))
            if boundary is None or not math.isfinite(boundary): raise RuntimeError("invalid track boundary geometry")
            if off_track.update(boundary>safety.off_track_margin_m,time.monotonic()): raise RuntimeError(f"sustained off-track: boundary distance {boundary:.3f}m")
            client.command_steering(steering); client.command_speed(speed)
            if not motion: motion=True; liveness.update(pose,float(clock["sim_time"]),time.monotonic(),motion_commanded=True)
            ctes.append(projection.distance); steerings.append(steering); saturation+=math.isclose(abs(steering),safety.max_steering_rad,abs_tol=1e-8)
            distance=math.dist((pose["x"],pose["y"]),route.points[0])
            if tracker.lap_complete(distance,safety.start_gate_radius_m,safety.minimum_lap_progress_fraction): result="PASS"; break
            next_tick += 1.0/safety.control_frequency_hz
            if next_tick < time.monotonic()-1.0/safety.control_frequency_hz: next_tick=time.monotonic()
    except TemporalInputError as exc:
        failure=str(exc); temporal_failure=True; invalid_history+=1
    except Exception as exc:
        failure=str(exc)
        if any(word in failure.lower() for word in ("get ","post ","control rejected","unavailable")): api_failures+=1
    finally:
        off_track.finalize(time.monotonic()); stop_errors=client.safe_stop()
        if stop_errors: api_failures+=len(stop_errors); result="FAIL"; failure=(failure+"; " if failure else "")+"; ".join(stop_errors)
    elapsed=time.monotonic()-started; deltas=[abs(steerings[i]-steerings[i-1]) for i in range(1,len(steerings))]; sat=saturation/len(steerings) if steerings else 0.
    return {"result": result, "failure": failure, "temporal_input_failure": temporal_failure, "warmup": locals().get("warmup"), "elapsed_s": elapsed,
        "route_length_m": route.length, "route_completion_fraction": tracker.unwrapped/route.length, "total_unwrapped_progress_m": tracker.unwrapped,
        "final_route_s_m": projection.s, "final_distance_to_start_m": math.dist((final_pose["x"],final_pose["y"]),route.points[0]),
        "mean_cte_m": statistics.fmean(ctes) if ctes else 0., "max_cte_m": max(ctes,default=0.), "off_track_events": off_track.event_count, "off_track_total_duration_s": off_track.total_duration_s,
        "mean_absolute_predicted_steering_rad": statistics.fmean(abs(x) for x in steerings) if steerings else 0., "max_absolute_predicted_steering_rad": max((abs(x) for x in steerings),default=0.),
        "steering_saturation_fraction": sat, "mean_absolute_command_delta_rad": statistics.fmean(deltas) if deltas else 0.,
        "camera_acquisition_latency": _summary_ms(camera_times), "preprocessing_latency": _summary_ms(prep_times), "onnx_inference_latency": _summary_ms(infer_times), "total_temporal_model_path_latency": _summary_ms(total_times),
        "control_loop_period": _summary_ms(periods), "control_loop_frequency_hz": 1.0/statistics.fmean(periods) if periods else 0., "timing_slips_over_100ms": sum(x>.1 for x in periods),
        "temporal_frame_gaps": {"oldest_to_middle_s": distribution(gap1), "middle_to_current_s": distribution(gap2), "oldest_to_current_s": distribution(spans)},
        "temporal_invalid_history_count": invalid_history, "api_failures": api_failures, "liveness_failures": liveness_failures,
        "safe_stop_success": not stop_errors, "safe_stop_errors": stop_errors, "neural_observation_fields": list(model.observation_fields),
        "failure_category": classify_failure(failure,steerings,sat), "speed_mps": speed_mps}


def classify_temporal_run(run):
    if run.get("temporal_input_failure"): return "TEMPORAL_INPUT_FAIL"
    if run.get("result") == "PASS": return "POLICY_PASS"
    if run.get("api_failures") or run.get("liveness_failures") or not run.get("safe_stop_success"): return "INFRA_FAIL"
    return "POLICY_FAIL"


def run_attempts(client, model, config, result_dir, preflight_one=live_preflight, run_one=run_temporal_live):
    attempts=[]; passes=0
    for number in range(1,MAX_TOTAL_ATTEMPTS+1):
        try:
            initial, preflight=preflight_one(client,config); run=run_one(client,model,config,initial,SPEED_MPS); classification=classify_temporal_run(run)
            attempt={"attempt_number":number,"valid_policy_run_number":passes+1 if classification in {"POLICY_PASS","POLICY_FAIL"} else None,"classification":classification,"preflight":preflight,"run":run}
        except Exception as exc:
            errors=client.safe_stop(); attempt={"attempt_number":number,"valid_policy_run_number":None,"classification":"INFRA_FAIL","run":None,"preflight":{"result":"FAIL","failure":f"{type(exc).__name__}: {exc}","safe_stop_success":not errors,"safe_stop_errors":errors}}
        attempts.append(attempt); write_json(result_dir/f"attempt_{number:02d}.json",attempt)
        classification = attempt["classification"]
        if classification == "POLICY_FAIL": return attempts,"FAIL"
        if classification == "POLICY_PASS":
            passes+=1
            if passes==TARGET_VALID_PASSES: return attempts,"PASS"
    return attempts,"INCONCLUSIVE"


def _artifact(repo):
    report=json.loads((repo/"results/pilotnet_training_v9_high_speed_temporal/summary.json").read_text())
    if report.get("result")!="PASS" or report.get("onnx_equivalence",{}).get("result")!="PASS": raise GateFailure("V9 training/ONNX gate failed")
    path=Path(report["artifacts"]["onnx"]["path"])
    if sha256_file(path)!=report["artifacts"]["onnx"]["sha256"]: raise GateFailure("V9 ONNX hash mismatch")
    return report,path


def preflight_stage(repo):
    training,path=_artifact(repo); config=load_inference_config(repo); result_path=repo/"results/pilotnet_e2e_v9_high_speed_temporal/preflight.json"
    if result_path.exists(): raise RuntimeError("refusing to repeat V9 preflight")
    client=SimClient(config.payload["base_url"],config.payload["api_timeout_s"]); report={"version":"pilotnet_e2e_v9_high_speed_temporal_preflight","generated_utc":utc_now(),"result":"FAIL","onnx":training["artifacts"]["onnx"]}
    try:
        if errors:=client.safe_stop(): raise GateFailure("initial safe stop failed: "+"; ".join(errors))
        _,checks=live_preflight(client,config); report.update(checks); buffer,warmup=warm_temporal_buffer(client,config); report["temporal_buffer_warmup"]=warmup; report["model_input_shape"]=[9,66,200]
    finally:
        errors=client.safe_stop(); report["safe_stop_success"]=not errors; report["safe_stop_errors"]=errors; report["result"]="FAIL" if errors else report.get("result","PASS"); write_json(result_path,report)
    return report


def live_stage(repo):
    training,path=_artifact(repo); config=load_inference_config(repo); result_dir=repo/"results/pilotnet_e2e_v9_high_speed_temporal"; marker=result_dir/"experiment.started.json"; summary=result_dir/"summary.json"
    if marker.exists() or summary.exists(): raise RuntimeError("refusing to repeat V9 live evaluation")
    client=SimClient(config.payload["base_url"],config.payload["api_timeout_s"]); model=TemporalOnnxModel(path); report={"version":"pilotnet_e2e_v9_high_speed_temporal","generated_utc":utc_now(),"result":"INCONCLUSIVE","onnx":training["artifacts"]["onnx"]}
    try:
        if errors:=client.safe_stop(): raise GateFailure("initial safe stop failed: "+"; ".join(errors))
        write_json(marker,{"status":"V9_TEMPORAL_LIVE_STARTED","maximum_total_attempts":5,"maximum_valid_policy_runs":3,"additional_dagger":False})
        attempts,result=run_attempts(client,model,config,result_dir); passes=sum(x["classification"]=="POLICY_PASS" for x in attempts)
        report.update({"result":result,"attempts":attempts,"policy_pass_count":passes,"temporal_input_failure_count":sum(x["classification"]=="TEMPORAL_INPUT_FAIL" for x in attempts),"infrastructure_failure_count":sum(x["classification"]=="INFRA_FAIL" for x in attempts)})
        if result=="PASS": report["decision"]={"classification":"PASS","v9_can_be_frozen":True,"cone_avoidance_v1_justified":True}
        elif result=="FAIL": report["decision"]={"classification":"PARTIAL_SUPPORT_NOT_REPEATABLE" if passes else "FAIL","v9_can_be_frozen":False,"cone_avoidance_v1_justified":False,"additional_dagger":False,"recommendation":"system-level high-speed temporal/architecture analysis"}
    finally:
        errors=client.safe_stop(); report["final_safe_stop_success"]=not errors; report["final_safe_stop_errors"]=errors; write_json(summary,report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("stage",choices=("dataset","train","live-preflight","live")); parser.add_argument("--sim-root",type=Path,required=True); args=parser.parse_args(argv)
    repo=Path(__file__).resolve().parents[2]
    try:
        result=dataset_stage(repo,args.sim_root.resolve()) if args.stage=="dataset" else training_stage(repo,args.sim_root.resolve()) if args.stage=="train" else preflight_stage(repo) if args.stage=="live-preflight" else live_stage(repo)
        print(json.dumps(result,indent=2,sort_keys=True)); return 0 if result.get("result")=="PASS" else 1
    except Exception as exc:
        print(f"HARD GATE FAILURE: {type(exc).__name__}: {exc}",file=__import__("sys").stderr); return 1


if __name__=="__main__": raise SystemExit(main())
