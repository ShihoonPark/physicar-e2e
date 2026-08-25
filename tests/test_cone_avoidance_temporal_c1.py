from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from physicar_e2e.cone_avoidance_temporal_c1 import (
    C1_EVAL_STRATA,
    C1_TRAIN_STRATA,
    EPISODES,
    HOLDOUT_EPISODES,
    MAXIMUM_LIVE_ATTEMPTS,
    REQUIRED_CLEARANCE_M,
    REQUIRED_TOPICS,
    RETURN_CTE_M,
    RETURN_DURATION_S,
    TRAIN_EPISODES,
    VALIDATION_EPISODES,
    WORLD,
    _attach_evaluation_route_s,
    audit_frozen,
    classify_cone_policy_run,
    load_c1_inference_config,
    load_c1_training_config,
    load_collection_config,
    run_c1_attempts,
    transform_odom_to_world,
)
from physicar_e2e.dataset_extractor import ScalarRecord, latest_causal
from physicar_e2e.high_speed_temporal import build_sequences
from physicar_e2e.pilotnet_temporal import CausalFrameBuffer, build_temporal_pilotnet
from physicar_e2e.pilotnet_training import sha256_file


REPO = Path(__file__).resolve().parents[1]
SIM_ROOT = Path("/home/a/physicar-ai-sim-docker")


def test_exactly_twelve_frozen_cone_episode_ids_and_split() -> None:
    assert EPISODES == tuple(f"cone_episode_{index:03d}" for index in range(1, 13))
    assert TRAIN_EPISODES == EPISODES[:8]
    assert VALIDATION_EPISODES == EPISODES[8:10]
    assert HOLDOUT_EPISODES == EPISODES[10:12]
    assert not (set(TRAIN_EPISODES) & (set(VALIDATION_EPISODES) | set(HOLDOUT_EPISODES)))


def test_frozen_world_pose_expert_and_v9_identities() -> None:
    if not SIM_ROOT.is_dir():
        pytest.skip("simulator asset checkout unavailable")
    audit = audit_frozen(REPO, SIM_ROOT)
    assert audit["result"] == "PASS"
    assert audit["world"] == WORLD
    assert audit["cone"]["route_s_m"] == pytest.approx(6.9)
    assert audit["cone"]["x_m"] == pytest.approx(6.165700204, abs=1e-9)
    assert audit["cone"]["y_m"] == pytest.approx(1.229802786, abs=1e-9)
    assert audit["avoidance"]["side"] == "right"
    assert min(audit["expert_clearances_m"]) >= .05


def test_collection_contract_is_exact_and_disk_gated() -> None:
    payload, collector = load_collection_config(REPO / "configs/cone_avoidance_collection_v1.json")
    assert payload["episode_count"] == 12
    assert tuple(collector.required_topics) == REQUIRED_TOPICS
    assert payload["minimum_free_bytes_before_collection"] == 8 * 1024**3
    assert payload["minimum_free_bytes_before_training"] == 5 * 1024**3
    assert payload["retry_valid_expert_failure"] is False


def test_causal_zoh_never_selects_future_record() -> None:
    records = [ScalarRecord(100, .1), ScalarRecord(200, .2)]
    assert latest_causal(records, 99) is None
    assert latest_causal(records, 100) == records[0]
    assert latest_causal(records, 199) == records[0]
    assert latest_causal(records, 200) == records[1]


def test_evaluation_route_s_attachment_is_also_causal() -> None:
    rows = [{"camera_record_time_ns": 150}, {"camera_record_time_ns": 250}]
    diagnostics = _attach_evaluation_route_s(rows, [(100, 1.0), (200, 2.0), (300, 3.0)])
    assert [row["route_s_m"] for row in rows] == [1.0, 2.0]
    assert diagnostics["future_violations"] == 0


def test_odom_frame_is_rigidly_aligned_to_frozen_world_spawn() -> None:
    # A +X odom displacement maps to world -Y at the frozen -pi/2 spawn yaw.
    x_m, y_m = transform_odom_to_world(
        1.0, 0.0, initial_odom=(0.0, 0.0, 0.0),
        initial_world=(1.4, 3.394607, -np.pi / 2),
    )
    assert x_m == pytest.approx(1.4)
    assert y_m == pytest.approx(2.394607)


def _write_source_manifest(root: Path, timestamps: list[int]) -> Path:
    manifest = root / "manifest.csv"
    fields = ["episode_id", "image_path", "camera_header_time_ns", "steering_rad", "source_mcap_sha256", "window_role"]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, timestamp in enumerate(timestamps):
            image = root / f"frame_{index}.png"
            image.touch()
            writer.writerow({
                "episode_id": "cone_episode_001", "image_path": image.name,
                "camera_header_time_ns": timestamp, "steering_rad": 0.0,
                "source_mcap_sha256": "source", "window_role": "",
            })
    return manifest


def test_three_frame_sequences_are_causal_gap_gated_and_unpadded(tmp_path: Path) -> None:
    manifest = _write_source_manifest(tmp_path, [0, 60_000_000, 120_000_000, 300_000_000])
    rows, stats = build_sequences("cone_episode_001", "cone_train", tmp_path, manifest)
    assert len(rows) == 1
    assert [rows[0][key] for key in ("timestamp_t_minus_2_ns", "timestamp_t_minus_1_ns", "timestamp_t_ns")] == [0, 60_000_000, 120_000_000]
    assert stats["rejected_boundary"] == 2
    assert stats["rejected_gap"] == 1
    buffer = CausalFrameBuffer(.120)
    frame = np.zeros((3, 66, 200), dtype=np.float32)
    buffer.append(0.0, frame); buffer.append(.06, frame); buffer.append(.12, frame)
    assert buffer.tensor().shape == (9, 66, 200)


def test_c1_architecture_training_semantics_and_source_roles_match_v9() -> None:
    config = load_c1_training_config(REPO)
    assert sum(parameter.numel() for parameter in build_temporal_pilotnet().parameters()) == 255_819
    assert config["initialization"] == "from_scratch"
    assert config["sample_weighting"] is False and config["resampling"] is False
    assert C1_TRAIN_STRATA == ("v9_train", "cone_train")
    assert C1_EVAL_STRATA == ("nominal_validation", "nominal_holdout", "cone_validation", "cone_holdout")
    assert not (set(C1_TRAIN_STRATA) & set(C1_EVAL_STRATA))
    v9_train = SIM_ROOT / "userdata/physicar_e2e/high_speed_temporal_v1/manifests/train.csv"
    if v9_train.is_file():
        assert sha256_file(v9_train) == config["preserved_v9_train_manifest_sha256"]


def test_camera_only_live_contract_and_frozen_safety_thresholds() -> None:
    config = load_c1_inference_config(REPO)
    forbidden = set(config.payload["camera_only_forbidden_inputs"])
    assert forbidden == {"route", "gt_pose", "cone_coordinates", "cte", "bypass_reference", "expert_steering", "obstacle_distance", "simulator_object_state"}
    assert config.payload["expected_world"] == WORLD
    assert REQUIRED_CLEARANCE_M == config.payload["required_cone_clearance_m"] == .05
    assert RETURN_CTE_M == config.payload["return_maximum_absolute_nominal_cte_m"] == .05
    assert RETURN_DURATION_S == config.payload["return_minimum_stable_duration_s"] == .50
    assert config.payload["smoke_speeds_mps"] == [1.8, 1.8, 1.8]
    assert config.payload["control_frequency_hz"] == 15.0


def _policy_run(classification: str) -> dict:
    return {"classification": classification}


class FakeClient:
    def __init__(self) -> None:
        self.safe_stop_calls = 0

    def safe_stop(self):
        self.safe_stop_calls += 1
        return []


def test_primary_cone_policy_failure_stops_repeatability(tmp_path: Path) -> None:
    calls = []

    def preflight(*args):
        return object(), {"result": "PASS"}

    def run(*args):
        calls.append(1)
        return _policy_run("CONE_POLICY_FAIL")

    attempts, result = run_c1_attempts(
        FakeClient(), object(), object(), object(), object(), {}, Path("/sim"), tmp_path,
        preflight_one=preflight, run_one=run,
    )
    assert result == "FAIL"
    assert len(attempts) == len(calls) == 1


def test_conditional_three_pass_and_bounded_infrastructure_replacements(tmp_path: Path) -> None:
    preflight_calls = 0

    def preflight(*args):
        nonlocal preflight_calls
        preflight_calls += 1
        if preflight_calls <= 2:
            raise RuntimeError("temporary infrastructure failure")
        return object(), {"result": "PASS"}

    def run(*args):
        return _policy_run("CONE_POLICY_PASS")

    attempts, result = run_c1_attempts(
        FakeClient(), object(), object(), object(), object(), {}, Path("/sim"), tmp_path,
        preflight_one=preflight, run_one=run,
    )
    assert result == "PASS"
    assert len(attempts) == MAXIMUM_LIVE_ATTEMPTS
    assert [item["classification"] for item in attempts[:2]] == ["INFRA_FAIL", "INFRA_FAIL"]
    assert [item["classification"] for item in attempts[2:]] == ["CONE_POLICY_PASS"] * 3


def test_clearance_collision_recovery_temporal_and_safe_stop_classification() -> None:
    passing = {
        "result": "PASS", "temporal_input_failure": False, "api_failures": 0,
        "liveness_failures": 0, "safe_stop_success": True,
        "minimum_footprint_to_cone_clearance_m": .05,
        "footprint_cone_intersection_occurred": False, "recovery_success": True,
    }
    assert classify_cone_policy_run(passing) == "CONE_POLICY_PASS"
    assert classify_cone_policy_run({**passing, "minimum_footprint_to_cone_clearance_m": .0499}) == "CONE_POLICY_FAIL"
    assert classify_cone_policy_run({**passing, "footprint_cone_intersection_occurred": True}) == "CONE_POLICY_FAIL"
    assert classify_cone_policy_run({**passing, "recovery_success": False}) == "CONE_POLICY_FAIL"
    assert classify_cone_policy_run({**passing, "temporal_input_failure": True}) == "TEMPORAL_INPUT_FAIL"
    assert classify_cone_policy_run({**passing, "safe_stop_success": False}) == "INFRA_FAIL"
