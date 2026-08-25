"""Audit and hard gate for High-Speed Temporal PilotNet late-region balance V1."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Sequence

from .high_speed_v5 import write_json
from .pilotnet_temporal import TEMPORAL_PARAMETER_COUNT, build_temporal_pilotnet
from .pilotnet_training import GateFailure, sha256_file


VERSION = "high_speed_temporal_balanced_v1"
BIN_NAMES = ("85_90_percent", "90_95_percent", "95_100_percent")
MINIMUM_PER_BIN = 20
V9_DAGGER3_A = "high_speed_dagger_iter3_rollout_A"
V9_DAGGER3_B = "high_speed_dagger_iter3_rollout_B"
MAJOR_STRATA = ("nominal_validation", "nominal_holdout", "dagger1_B", "dagger2_B")


class InsufficientLateRegionDiversity(GateFailure):
    """The pre-registered minimum equal-bin count was not available."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def route_bin(completion_fraction: float) -> str | None:
    """Return the exact pre-registered late-route bin."""
    value = float(completion_fraction)
    if 0.85 <= value < 0.90:
        return "85_90_percent"
    if 0.90 <= value < 0.95:
        return "90_95_percent"
    if 0.95 <= value <= 1.00:
        return "95_100_percent"
    return None


def evenly_spaced_indices(population: int, count: int) -> list[int]:
    """Choose deterministic ordered indices spanning the full population."""
    if count < 1 or population < count:
        raise ValueError("selection requires 1 <= count <= population")
    if count == 1:
        return [(population - 1) // 2]
    indices = [round(index * (population - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count or indices != sorted(indices):
        raise RuntimeError("evenly spaced selection produced duplicate or unordered indices")
    return indices


def balanced_subset(groups: dict[str, Sequence[dict[str, Any]]],
                    minimum_per_bin: int = MINIMUM_PER_BIN) -> tuple[int, list[dict[str, Any]]]:
    """Apply the frozen K gate, then deterministically undersample each bin."""
    if tuple(groups) != BIN_NAMES:
        raise ValueError("balanced groups must use the three frozen bins in order")
    k = min(len(groups[name]) for name in BIN_NAMES)
    if k < minimum_per_bin:
        raise InsufficientLateRegionDiversity(
            f"K={k} is below the pre-registered minimum {minimum_per_bin}"
        )
    selected: list[dict[str, Any]] = []
    for name in BIN_NAMES:
        ordered = list(groups[name])
        selected.extend(ordered[index] for index in evenly_spaced_indices(len(ordered), k))
    if len(selected) != 3 * k or len({item["identity"] for item in selected}) != 3 * k:
        raise GateFailure("balanced selection duplicated a temporal sequence")
    return k, selected


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    bins = tuple(
        (item.get("name"), item.get("lower"), item.get("upper"), item.get("upper_inclusive"))
        for item in config.get("route_bins", [])
    )
    expected_bins = (
        ("85_90_percent", 0.85, 0.90, False),
        ("90_95_percent", 0.90, 0.95, False),
        ("95_100_percent", 0.95, 1.00, True),
    )
    frozen = (
        config.get("version"), bins, config.get("minimum_per_bin"),
        config.get("new_data_collection_permitted"),
        config.get("automatic_followup_optimization_permitted"),
    )
    if frozen != (VERSION, expected_bins, MINIMUM_PER_BIN, False, False):
        raise GateFailure(f"late-balance experiment contract changed: {frozen}")
    selection = config.get("selection", {})
    if selection != {
        "method": "deterministic_evenly_spaced_ordered_indices",
        "random_selection": False,
        "oversampling": False,
        "duplicate_selection": False,
        "error_based_selection": False,
    }:
        raise GateFailure("late-balance selection contract changed")
    return config


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise GateFailure(f"missing preserved manifest {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _source_index(source_root: Path, source_manifest: Path,
                  source_id: str) -> dict[tuple[int, str], dict[str, str]]:
    index: dict[tuple[int, str], dict[str, str]] = {}
    for row in _read_csv(source_manifest):
        if row.get("episode_id") != source_id or row.get("rollout_id") != source_id:
            raise GateFailure(f"mixed source identity in {source_manifest}")
        image = str((source_root / row["image_path"]).resolve())
        key = (int(row["camera_header_time_ns"]), image)
        if key in index:
            raise GateFailure(f"duplicate source timestamp/image identity in {source_manifest}")
        if not Path(image).is_file():
            raise GateFailure(f"missing preserved image {image}")
        index[key] = row
    return index


def audit_temporal_source(temporal_manifest: Path, source_root: Path,
                          source_manifest: Path, source_id: str) -> dict[str, Any]:
    """Join each V9 temporal current frame to preserved route-progress metadata."""
    source_hash = sha256_file(source_manifest)
    source = _source_index(source_root, source_manifest, source_id)
    temporal_rows = [row for row in _read_csv(temporal_manifest) if row.get("source_id") == source_id]
    if not temporal_rows:
        raise GateFailure(f"no V9 temporal rows for {source_id}")
    groups: dict[str, list[dict[str, Any]]] = {name: [] for name in BIN_NAMES}
    previous_timestamp = -1
    for row in temporal_rows:
        timestamps = [int(row[name]) for name in (
            "timestamp_t_minus_2_ns", "timestamp_t_minus_1_ns", "timestamp_t_ns"
        )]
        if not timestamps[0] < timestamps[1] < timestamps[2]:
            raise GateFailure(f"non-causal V9 temporal row in {source_id}")
        gaps = [(timestamps[1] - timestamps[0]) / 1e9, (timestamps[2] - timestamps[1]) / 1e9]
        if any(gap <= 0 or gap > 0.120 for gap in gaps):
            raise GateFailure(f"V9 accepted row violates the 0.120 s gap gate in {source_id}")
        if timestamps[2] <= previous_timestamp:
            raise GateFailure(f"V9 temporal rows are not trajectory ordered in {source_id}")
        previous_timestamp = timestamps[2]
        frame_t = str(Path(row["frame_t"]).resolve())
        key = (timestamps[2], frame_t)
        metadata = source.get(key)
        if metadata is None:
            raise GateFailure(f"cannot join V9 current frame to route metadata: {key}")
        if row.get("source_manifest_sha256") != source_hash:
            raise GateFailure(f"V9 source-manifest hash mismatch for {source_id}")
        if row.get("source_mcap_sha256") != metadata.get("source_mcap_sha256"):
            raise GateFailure(f"V9 source MCAP hash mismatch for {source_id}")
        if not math.isclose(float(row["target_steering_rad"]), float(metadata["steering_rad"]),
                            rel_tol=0.0, abs_tol=1e-12):
            raise GateFailure(f"V9 target mismatch for {source_id}")
        completion = float(metadata["completion_fraction"])
        name = route_bin(completion)
        if name is None or metadata.get("window_role") != name or row.get("window_role") != name:
            raise GateFailure(f"route-bin metadata mismatch at completion {completion}")
        groups[name].append({
            "identity": f"{source_id}:{timestamps[2]}:{frame_t}",
            "source_id": source_id,
            "sequence_index": int(row["sequence_index"]),
            "timestamp_t_ns": timestamps[2],
            "frame_t": frame_t,
            "completion_fraction": completion,
            "route_s_m": float(metadata["route_s_m"]),
            "window_role": name,
        })
    return {
        "source_id": source_id,
        "temporal_manifest": str(temporal_manifest),
        "temporal_manifest_sha256": sha256_file(temporal_manifest),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": source_hash,
        "joined_temporal_sequence_count": len(temporal_rows),
        "bin_counts": {name: len(groups[name]) for name in BIN_NAMES},
        "groups": groups,
        "route_progress_metadata": "preserved completion_fraction joined by current timestamp and image path",
        "causal_contract_verified": True,
    }


def _metric_subset(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": metrics["sample_count"],
        "mae_rad": metrics["mae_rad"],
        "rmse_rad": metrics["rmse_rad"],
        "bias_mean_signed_error_rad": metrics["bias_mean_signed_error_rad"],
        "max_absolute_error_rad": metrics["max_absolute_error_rad"],
        "correlation": metrics["correlation"],
        "corrective_magnitude_ratio": metrics["corrective_magnitude_ratio"],
    }


def preserved_v9_evidence(repo: Path, config: dict[str, Any]) -> dict[str, Any]:
    training_path = repo / "results/pilotnet_training_v9_high_speed_temporal/summary.json"
    live_path = repo / "results/pilotnet_e2e_v9_high_speed_temporal/summary.json"
    dataset_path = repo / "results/high_speed_temporal_dataset_v1/summary.json"
    training = json.loads(training_path.read_text())
    live = json.loads(live_path.read_text())
    dataset = json.loads(dataset_path.read_text())
    expected = config["preserved_v9"]
    checkpoint = training.get("artifacts", {}).get("checkpoint", {})
    onnx = training.get("artifacts", {}).get("onnx", {})
    checks = (
        training.get("result") == "PASS",
        training.get("architecture", {}).get("parameter_count") == TEMPORAL_PARAMETER_COUNT,
        training.get("training_from_scratch") is True,
        live.get("result") == "PASS",
        live.get("policy_pass_count") == expected["required_policy_passes"],
        dataset.get("new_training_data_collected") is False,
        checkpoint.get("sha256") == expected["checkpoint_sha256"],
        onnx.get("sha256") == expected["onnx_sha256"],
        sha256_file(Path(checkpoint["path"])) == expected["checkpoint_sha256"],
        sha256_file(Path(onnx["path"])) == expected["onnx_sha256"],
    )
    if not all(checks):
        raise GateFailure("preserved V9 identity/evidence gate failed")
    return {
        "training_result": training["result"],
        "parameter_count": training["architecture"]["parameter_count"],
        "trained_from_scratch": training["training_from_scratch"],
        "checkpoint": checkpoint,
        "onnx": onnx,
        "live_result": live["result"],
        "policy_pass_count": live["policy_pass_count"],
        "dataset_new_training_data_collected": dataset["new_training_data_collected"],
        "training_summary_sha256": sha256_file(training_path),
        "live_summary_sha256": sha256_file(live_path),
        "dataset_summary_sha256": sha256_file(dataset_path),
    }


def v9_dagger3_b_metrics(repo: Path, b_audit: dict[str, Any]) -> dict[str, Any]:
    training = json.loads(
        (repo / "results/pilotnet_training_v9_high_speed_temporal/summary.json").read_text()
    )
    preserved = training["matched_offline_comparison"]["dagger3_B"]
    bins: dict[str, Any] = {}
    for name in BIN_NAMES:
        values = _metric_subset(preserved["subregions"][name]["v9"])
        if values["sample_count"] != b_audit["bin_counts"][name]:
            raise GateFailure(f"V9 DAgger3-B metric subset mismatch for {name}")
        bins[name] = values
    late_count = sum(bins[name]["sample_count"] for name in BIN_NAMES[1:])
    late_mae = sum(
        bins[name]["sample_count"] * bins[name]["mae_rad"] for name in BIN_NAMES[1:]
    ) / late_count
    return {
        "overall": _metric_subset(preserved["v9"]),
        "bins": bins,
        "combined_90_100": {"sample_count": late_count, "mae_rad": late_mae},
        "final_to_early_mae_ratio": bins["95_100_percent"]["mae_rad"] / bins["85_90_percent"]["mae_rad"],
    }


def offline_live_gate(v9_major: dict[str, float], v10_major: dict[str, float],
                      v9_late_mae: float, v10_late_mae: float,
                      catastrophic_ratio: float = 1.50) -> dict[str, Any]:
    if set(v9_major) != set(MAJOR_STRATA) or set(v10_major) != set(MAJOR_STRATA):
        raise ValueError("offline gate requires the four frozen major strata")
    reasons: list[str] = []
    if v10_late_mae > v9_late_mae:
        reasons.append("DAGGER3_B_90_100_MAE_WORSE")
    for name in MAJOR_STRATA:
        if v10_major[name] > catastrophic_ratio * v9_major[name]:
            reasons.append(f"CATASTROPHIC_MAE_REGRESSION:{name}")
    return {"result": "PASS" if not reasons else "FAIL", "reasons": reasons}


def _without_groups(audit: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in audit.items() if key != "groups"}


def audit_stage(repo: Path, sim_root: Path) -> dict[str, Any]:
    result_path = repo / "results/high_speed_temporal_balanced_v1/summary.json"
    if result_path.exists():
        raise RuntimeError("refusing to overwrite late-balance audit evidence")
    for path in (
        repo / "results/pilotnet_training_v10_high_speed_temporal_balanced",
        repo / "results/pilotnet_e2e_v10_high_speed_temporal_balanced",
        sim_root / "userdata/physicar_e2e/high_speed_temporal_balanced_v1",
    ):
        if path.exists():
            raise GateFailure(f"unexpected pre-existing V10 artifact path {path}")
    config_path = repo / "configs/high_speed_temporal_balanced_v1.json"
    config = load_config(config_path)
    base = sim_root / "userdata/physicar_e2e"
    temporal = base / "high_speed_temporal_v1/manifests"
    dagger3 = base / "high_speed_dagger_iteration3_v1/extracted"
    a = audit_temporal_source(
        temporal / "train.csv", dagger3,
        dagger3 / "manifests/high_speed_dagger_iter3_rollout_A.csv", V9_DAGGER3_A,
    )
    b = audit_temporal_source(
        temporal / "dagger3_B.csv", dagger3,
        dagger3 / "manifests/high_speed_dagger_iter3_rollout_B.csv", V9_DAGGER3_B,
    )
    v9 = preserved_v9_evidence(repo, config)
    b_metrics = v9_dagger3_b_metrics(repo, b)
    counts = a["bin_counts"]
    k = min(counts.values())
    ratio = max(counts.values()) / min(counts.values())
    report: dict[str, Any] = {
        "version": VERSION,
        "generated_utc": utc_now(),
        "result": "STOP_INSUFFICIENT_LATE_REGION_DIVERSITY",
        "hard_gate": {
            "name": "minimum_equal_dagger3_A_bin_count",
            "required_minimum": MINIMUM_PER_BIN,
            "actual_k": k,
            "result": "FAIL" if k < MINIMUM_PER_BIN else "PASS",
            "reason": f"K={k} is below the pre-registered minimum {MINIMUM_PER_BIN}" if k < MINIMUM_PER_BIN else None,
        },
        "preserved_v9": v9,
        "dagger3_A_audit": _without_groups(a),
        "dagger3_B_audit": _without_groups(b),
        "dagger3_B_v9_offline_metrics": b_metrics,
        "original_imbalance": {
            "max_to_min_ratio": ratio,
            "85_90_to_90_95_ratio": counts["85_90_percent"] / counts["90_95_percent"],
            "85_90_to_95_100_ratio": counts["85_90_percent"] / counts["95_100_percent"],
        },
        "selection": {
            "method": config["selection"]["method"],
            "performed": False,
            "balanced_manifest_created": False,
            "selected_count_per_bin": None,
            "oversampling": False,
            "reason": "selection prohibited because K < 20",
        },
        "no_new_data": {
            "training_data_collected": False,
            "raw_bags_created": 0,
            "expert_laps_created": 0,
            "neural_rollouts_created": 0,
            "dagger_iterations_created": 0,
            "synthetic_samples_created": 0,
        },
        "v10": {
            "training_manifest_created": False,
            "train_sequence_count": None,
            "training_executed": False,
            "checkpoint_created": False,
            "onnx_created": False,
            "offline_evaluation_executed": False,
            "live_preflight_executed": False,
            "live_attempts": 0,
        },
        "decision": {
            "v9_remains_canonical": True,
            "v10_beats_v9": False,
            "user_visual_comparison_required": False,
            "cone_avoidance_v1_status": "preserved V9 already satisfies its 3/3 simulator repeatability gate; this stopped experiment adds no V10 evidence",
            "automatic_followup_optimization": False,
        },
        "config_sha256": sha256_file(config_path),
    }
    if k >= MINIMUM_PER_BIN:
        raise GateFailure("audit unexpectedly passed; bounded implementation is intentionally not implicit")
    write_json(result_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    try:
        report = audit_stage(repo, args.sim_root.resolve())
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"AUDIT FAILURE: {type(exc).__name__}: {exc}", file=__import__("sys").stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
