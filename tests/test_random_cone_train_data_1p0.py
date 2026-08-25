from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from physicar_e2e.dataset_extractor import (
    DriveWindow,
    ScalarRecord,
    detect_dominant_drive_window,
    latest_causal,
    synchronize_frame,
)
from physicar_e2e.random_cone_train_data import (
    COLLECTION_VERSION,
    EPISODE_ORDER,
    MAXIMUM_ADJACENT_GAP_S,
    REPEAT_IDS,
    REQUIRED_TOPICS,
    TRAIN_SCENARIOS,
    EpisodeSpec,
    TaskConfig,
    TrainDataGateError,
    build_temporal_sequences,
    collect_one_episode,
    episode_specs,
    load_task_config,
    region_coverage,
    retry_decision,
    validate_collection_gate,
    validate_existing_episode,
)
from physicar_e2e.rosbag_collector import BagInfo, RecorderHandle, RecorderStopResult


REPO = Path(__file__).resolve().parents[1]
TASK_PATH = REPO / "configs/random_cone_train_data_1p0_v1.json"


@pytest.fixture(scope="module")
def task() -> TaskConfig:
    return load_task_config(TASK_PATH, REPO)


def test_exact_train_scenarios_two_round_order_and_no_09_through_12(task: TaskConfig) -> None:
    assert TRAIN_SCENARIOS == ("01", "02", "03", "04", "05", "06", "07", "08")
    assert REPEAT_IDS == ("R01", "R02")
    assert len(EPISODE_ORDER) == len(set(EPISODE_ORDER)) == 16
    assert EPISODE_ORDER[:8] == tuple(f"train_s{value}_r01" for value in TRAIN_SCENARIOS)
    assert EPISODE_ORDER[8:] == tuple(f"train_s{value}_r02" for value in TRAIN_SCENARIOS)
    assert [spec.episode_id for spec in episode_specs()] == list(EPISODE_ORDER)
    assert all(spec.role == "TRAIN" and spec.scenario_id in TRAIN_SCENARIOS for spec in episode_specs())
    assert not any(f"s{scenario}" in episode for episode in EPISODE_ORDER for scenario in ("09", "10", "11", "12"))
    assert task.payload["scenario_roles"]["VALIDATION"] == ["09", "10"]
    assert task.payload["scenario_roles"]["UNSEEN_HOLDOUT"] == ["11", "12"]


def test_exact_topics_disk_retry_and_no_training_permissions(task: TaskConfig) -> None:
    assert tuple(task.collection["required_topics"]) == REQUIRED_TOPICS
    assert task.collection["minimum_free_bytes_before_collection"] == 8 * 1024**3
    assert task.collection["minimum_projected_free_bytes"] == 6 * 1024**3
    assert task.collection["infrastructure_replacement_attempts_per_episode"] == 1
    assert task.collection["retry_genuine_policy_failure"] is False
    assert task.dataset["history_frames"] == 3
    assert task.dataset["maximum_adjacent_gap_s"] == 0.120
    assert task.dataset["causal_only"] is True
    assert task.dataset["allow_episode_boundary_crossing"] is False
    assert task.dataset["allow_reset_boundary_crossing"] is False
    assert task.dataset["allow_duplicate_padding"] is False
    assert task.payload["permissions"]["validation_bag_collection_permitted"] is False
    assert task.payload["permissions"]["holdout_bag_collection_permitted"] is False
    assert task.payload["permissions"]["neural_training_permitted"] is False


def test_genuine_failure_stops_and_infrastructure_replacement_is_bounded() -> None:
    assert retry_decision("GENUINE_EXPERT_FAIL", 1) == "STOP_GENUINE_FAILURE"
    assert retry_decision("INFRA_FAIL", 1) == "RETRY_INFRA"
    assert retry_decision("INFRA_FAIL", 2) == "STOP_INFRA"
    assert retry_decision("TRAIN_EPISODE_PASS", 1) == "CONTINUE"


def _extractor_config() -> dict:
    return {
        "minimum_drive_speed_mps": 0.10,
        "maximum_steering_age_s": 0.15,
        "maximum_speed_age_s": 0.15,
    }


def test_causal_zoh_and_active_window_semantics_are_reused() -> None:
    steering = [ScalarRecord(100, 0.1), ScalarRecord(200, 0.2)]
    assert latest_causal(steering, 99) is None
    assert latest_causal(steering, 199) == steering[0]
    window = detect_dominant_drive_window(
        [ScalarRecord(0, 0.0), ScalarRecord(100, 1.0), ScalarRecord(200, 1.0), ScalarRecord(300, 0.0)],
        0.10,
    )
    assert window == DriveWindow(100, 200, 2)
    reason, selected, _ = synchronize_frame(
        199, steering, [ScalarRecord(100, 1.0)], window, _extractor_config(),
    )
    assert reason is None and selected == steering[0]


def _frame_row(spec: EpisodeSpec, index: int, time_ns: int) -> dict:
    return {
        "episode_id": spec.episode_id, "scenario_id": spec.scenario_id,
        "repeat_id": spec.repeat_id, "scenario_role": "TRAIN",
        "image_path": f"images/{spec.episode_id}/frame_{index:06d}.png",
        "image_sha256": f"{index:064x}", "camera_record_time_ns": time_ns,
        "steering_record_time_ns": time_ns - 1, "steering_age_ms": 0.000001,
        "steering_rad": 0.01 * index, "speed_record_time_ns": time_ns - 1,
        "speed_age_ms": 0.000001, "speed_mps": 1.0, "route_s_m": float(index),
        "source_mcap_sha256": "a" * 64,
    }


def test_temporal_sequences_are_strictly_causal_gap_gated_unpadded_and_train_only() -> None:
    spec = episode_specs()[0]
    rows = [
        _frame_row(spec, 0, 0), _frame_row(spec, 1, 60_000_000),
        _frame_row(spec, 2, 120_000_000), _frame_row(spec, 3, 300_000_000),
    ]
    temporal, stats = build_temporal_sequences(rows, spec, "b" * 64)
    assert len(temporal) == 1
    assert stats == {
        **stats,
        "temporal_candidate_sequences": 2,
        "accepted_temporal_sequences": 1,
        "gap_rejects": 1,
        "boundary_rejects": 2,
        "future_label_violations": 0,
    }
    sequence = temporal[0]
    assert sequence["scenario_role"] == "TRAIN"
    assert sequence["scenario_id"] == sequence["cone_scenario_id"] == "01"
    assert (
        sequence["camera_timestamp_t_minus_2_ns"]
        < sequence["camera_timestamp_t_minus_1_ns"]
        < sequence["camera_timestamp_t_ns"]
    )
    assert sequence["adjacent_gap_1_s"] <= MAXIMUM_ADJACENT_GAP_S
    assert sequence["adjacent_gap_2_s"] <= MAXIMUM_ADJACENT_GAP_S
    assert len({sequence["frame_t_minus_2"], sequence["frame_t_minus_1"], sequence["frame_t"]}) == 3


def test_temporal_builder_rejects_episode_crossing_duplicate_padding_and_future_label() -> None:
    spec = episode_specs()[0]
    rows = [_frame_row(spec, index, index * 60_000_000) for index in range(3)]
    crossed = [dict(row) for row in rows]
    crossed[1]["episode_id"] = "train_s02_r01"
    with pytest.raises(TrainDataGateError, match="episode boundary"):
        build_temporal_sequences(crossed, spec, "b" * 64)
    padded = [dict(row) for row in rows]
    padded[1]["image_path"] = padded[0]["image_path"]
    with pytest.raises(TrainDataGateError, match="padding"):
        build_temporal_sequences(padded, spec, "b" * 64)
    future = [dict(row) for row in rows]
    future[2]["steering_record_time_ns"] = future[2]["camera_record_time_ns"] + 1
    with pytest.raises(TrainDataGateError, match="future steering"):
        build_temporal_sequences(future, spec, "b" * 64)


def test_cone_region_gate_requires_all_four_maneuver_phases() -> None:
    regions = {
        "approach": (0.0, 1.0), "avoidance": (1.0, 2.0),
        "pass_return": (2.0, 3.0), "post_recovery": (3.0, 4.0),
    }
    rows = [{"route_progress_m": value} for value in (0.5, 1.5, 2.5, 3.5)]
    assert region_coverage(rows, regions)["result"] == "PASS"
    assert region_coverage(rows[:-1], regions)["result"] == "FAIL"


class FakeScenario:
    def __init__(self, scenario_id: str = "01") -> None:
        self.scenario_id = scenario_id
        self.chosen_side = "right"

    def to_dict(self) -> dict:
        return {"scenario_id": self.scenario_id, "role": "TRAIN", "chosen_side": self.chosen_side}


class FakeExpert:
    @staticmethod
    def world_name(scenario_id: str) -> str:
        return f"world_{scenario_id}"


def _fake_bundle() -> SimpleNamespace:
    return SimpleNamespace(scenario=FakeScenario(), geometry={"result": "PASS"})


def _passing_metadata(task: TaskConfig, bag: Path, size: int) -> dict:
    spec = episode_specs()[0]
    bundle = _fake_bundle()
    counts = {topic: 20 for topic in REQUIRED_TOPICS}
    counts["/camera/image_raw"] = 200
    digest = hashlib.sha256(bag.read_bytes()).hexdigest()
    return {
        "version": COLLECTION_VERSION + "_episode", "episode_id": spec.episode_id,
        "scenario_id": spec.scenario_id, "repeat_id": spec.repeat_id,
        "scenario_role": "TRAIN", "collection_order_index": 0,
        "world": "world_01", "frozen_scenario": bundle.scenario.to_dict(),
        "frozen_scenario_sha256": hashlib.sha256(
            ("{\"chosen_side\":\"right\",\"role\":\"TRAIN\",\"scenario_id\":\"01\"}\n").encode()
        ).hexdigest(),
        "task_config_sha256": task.sha256,
        "frozen_expert_config_sha256": task.frozen["config_sha256"],
        "frozen_expert_result_manifest_sha256": task.frozen["result_manifest_sha256"],
        "result": "PASS", "classification": "TRAIN_EPISODE_PASS",
        "expert_classification": "RANDOM_CONE_EXPERT_PASS",
        "required_topics": list(REQUIRED_TOPICS), "actual_topic_message_counts": counts,
        "bag_duration_s": 10.0, "bag_size_bytes": size, "bag_mcap_sha256": digest,
        "recorder_graceful_shutdown": True, "recorder_orphaned": False,
        "orphan_process_check_pass": True, "post_run_safe_stop_success": True,
        "final_safe_stop_success": True,
        "expert_result_metrics": {
            "result": "PASS", "minimum_footprint_to_cone_clearance_m": 0.001,
            "cone_contact_or_intersection_occurred": False, "recovery_success": True,
            "api_failures": 0, "pose_failures": 0, "clock_failures": 0,
            "safe_stop_success": True,
        },
    }


class ResumeBackend:
    def __init__(self, root: Path) -> None:
        self.host_data_root = root
        self.container_data_root = "/data"

    def bag_info(self, handle: RecorderHandle) -> BagInfo:
        counts = {topic: 20 for topic in REQUIRED_TOPICS}
        counts["/camera/image_raw"] = 200
        return BagInfo(10.0, sum(counts.values()), counts)


def test_finalized_valid_episode_is_skipped_but_partial_is_not_valid(tmp_path: Path, task: TaskConfig) -> None:
    spec = episode_specs()[0]
    episode_root = tmp_path / spec.episode_id / "bag"
    episode_root.mkdir(parents=True)
    mcap = episode_root / "data.mcap"
    mcap.write_bytes(b"valid mcap")
    metadata = _passing_metadata(task, mcap, len(b"valid mcap"))
    result_path = tmp_path / "result.json"
    from physicar_e2e.random_cone_train_data import write_json
    write_json(result_path, metadata)
    status, _, errors = validate_existing_episode(
        spec, result_path=result_path, task=task, expert=FakeExpert(),
        bundle=_fake_bundle(), backend=ResumeBackend(tmp_path),
    )
    assert status == "VALID" and not errors
    result_path.unlink()
    status, _, _ = validate_existing_episode(
        spec, result_path=result_path, task=task, expert=FakeExpert(),
        bundle=_fake_bundle(), backend=ResumeBackend(tmp_path),
    )
    assert status == "PARTIAL"


def _collection_row(spec: EpisodeSpec) -> dict:
    return {
        "episode_id": spec.episode_id, "scenario_id": spec.scenario_id,
        "repeat_id": spec.repeat_id, "scenario_role": "TRAIN", "result": "PASS",
        "classification": "TRAIN_EPISODE_PASS",
        "preflight": {"fixed_control": {
            "speed_mps": 1.0, "lookahead_m": 0.9, "control_frequency_hz": 15.0,
            "steering_limit_rad": 0.349066, "wheelbase_m": 0.18,
        }},
        "post_run_safe_stop_success": True, "final_safe_stop_success": True,
        "expert_result_metrics": {"safe_stop_success": True},
    }


def test_collection_gate_demands_exact_16_unique_train_ids() -> None:
    rows = [_collection_row(spec) for spec in episode_specs()]
    assert validate_collection_gate(rows)["result"] == "PASS"
    assert validate_collection_gate(rows[:-1])["result"] == "FAIL"
    assert validate_collection_gate([*rows[:-1], rows[0]])["result"] == "FAIL"


class LifecycleClient:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def safe_stop(self) -> list[str]:
        self.events.append("safe_stop")
        return []


class LifecycleBackend:
    def __init__(self, root: Path, events: list[str]) -> None:
        self.host_data_root = root
        self.container_data_root = "/data"
        self.host_userdata_root = root
        self.events = events

    def start_recorder(self, episode_id: str, topics) -> RecorderHandle:
        self.events.append("start_recorder")
        episode = self.host_data_root / episode_id
        bag = episode / "bag"
        bag.mkdir(parents=True)
        (bag / "data.mcap").write_bytes(b"bag")
        return RecorderHandle(episode_id, episode, bag, "/data/e", "/data/e/bag", "/data/e/pid", "/data/e/log")

    def stop_recorder(self, handle: RecorderHandle) -> RecorderStopResult:
        self.events.append("stop_recorder")
        return RecorderStopResult(True, False)

    def _alive(self, handle: RecorderHandle) -> bool:
        self.events.append("orphan_check")
        return False

    def bag_info(self, handle: RecorderHandle) -> BagInfo:
        counts = {topic: 20 for topic in REQUIRED_TOPICS}
        counts["/camera/image_raw"] = 200
        return BagInfo(10.0, sum(counts.values()), counts)


def test_episode_lifecycle_safe_stops_before_recorder_termination(
    tmp_path: Path, task: TaskConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    monkeypatch.setattr(
        "physicar_e2e.random_cone_train_data.disk_state",
        lambda _path: {"available_bytes": 7 * 1024**3},
    )

    def prepare(*args):
        events.append("preflight")
        return object(), {"result": "PASS"}, {"result": "PASS", "fixed_control": {
            "speed_mps": 1.0, "lookahead_m": 0.9, "control_frequency_hz": 15.0,
            "steering_limit_rad": 0.349066, "wheelbase_m": 0.18,
        }}

    def run(*args):
        events.append("expert")
        return {
            "classification": "RANDOM_CONE_EXPERT_PASS", "result": "PASS",
            "minimum_footprint_to_cone_clearance_m": 0.001,
            "cone_contact_or_intersection_occurred": False, "recovery_success": True,
            "api_failures": 0, "pose_failures": 0, "clock_failures": 0,
            "safe_stop_success": True, "elapsed_s": 10.0,
        }

    result = collect_one_episode(
        episode_specs()[0], task=task, repo=REPO, sim_root=Path("/sim"),
        expert=FakeExpert(), bundle=_fake_bundle(),
        backend=LifecycleBackend(tmp_path, events), client=LifecycleClient(events),
        attempt_number=1, result_path=tmp_path / "attempt.json",
        prepare=prepare, run_expert=run,
    )
    assert result["result"] == "PASS"
    stop_index = events.index("stop_recorder")
    assert events[stop_index - 1] == "safe_stop"
    assert result["post_run_safe_stop_success"] is True
    assert result["final_safe_stop_success"] is True


def test_pipeline_module_has_no_neural_training_or_export_call() -> None:
    source = (REPO / "src/physicar_e2e/random_cone_train_data.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    called = {
        node.func.id if isinstance(node.func, ast.Name) else node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute))
    }
    assert not called.intersection({
        "train_temporal", "train_pilotnet", "export_temporal_onnx", "export_onnx",
        "run_temporal_live", "run_neural_policy",
    })
