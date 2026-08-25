import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from physicar_e2e.real_bag_audit import (
    CameraSample,
    RealBagAuditError,
    ScalarSample,
    analyze_speed,
    analyze_steering,
    camera_timestamp_pair,
    causal_sync_metrics,
    cross_bag_consistency,
    decode_float64_cdr,
    latest_causal_index,
    load_config,
    run_audit,
    scalar_sample_from_decoded,
    simulator_camera_comparison,
    temporal_readiness,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "real_bag_audit_v1.json"
V2_CONFIG_PATH = ROOT / "configs" / "real_bag_audit_v2.json"


def _camera(index: int, time_ns: int) -> CameraSample:
    return CameraSample(
        index=index,
        record_time_ns=time_ns,
        publish_time_ns=time_ns + 1,
        header_time_ns=time_ns - 50,
        width=480,
        height=360,
        encoding="rgb8",
        is_bigendian=0,
        step=1440,
        data_length=518400,
        frame_id="camera",
    )


def _bag_result(bag_id: str, width: int = 480) -> dict:
    contract = {
        "width": width,
        "height": 360,
        "encoding": "rgb8",
        "is_bigendian": 0,
        "step": width * 3,
        "frame_id": "camera",
        "count": 3,
    }
    timing = {"mean": 0.01, "median": 0.01, "p95": 0.02, "max": 0.02}
    return {
        "bag_id": bag_id,
        "integrity": {"result": "PASS"},
        "camera": {"uniform_contract": contract, "fps": 15.0, "inter_frame_gap_s": timing},
        "steering": {
            "distribution_rad": {"min": -0.1, "max": 0.1, "median": 0.0},
            "range_check": {
                "below_confirmed_range_count": 0,
                "above_confirmed_range_count": 0,
            },
            "timestamp": {"measured_rate_hz": 15.0},
        },
        "speed": {
            "distribution_raw": {"min": 0.0, "max": 1.0, "median": 0.8},
            "sign_and_stationary_counts": {"exact_zero": 1},
            "timestamp": {"measured_rate_hz": 15.0},
        },
        "causal_synchronization": {"steering_age_ms": timing, "speed_age_ms": timing},
    }


def test_three_actual_bag_mappings_are_exact() -> None:
    config = load_config(CONFIG_PATH)
    assert config["bags"] == [
        {"id": "bag_01", "path": "/home/a/bag_01"},
        {"id": "bag_02", "path": "/home/a/bag_02"},
        {"id": "bag_03", "path": "/home/a/bag_03"},
    ]


def test_v2_maps_only_the_complete_retransferred_bags() -> None:
    config = load_config(V2_CONFIG_PATH)
    assert config["bags"] == [
        {"id": "bag_01", "path": "/home/a/output_bag/bag_01"},
        {"id": "bag_02", "path": "/home/a/output_bag/bag_02"},
        {"id": "bag_03", "path": "/home/a/output_bag/bag_03"},
    ]
    assert config["historical_v1"]["preserve_unchanged"] is True
    assert config["steering_contract"]["recorded_numeric_range"] == [-1.0, 1.0]
    assert config["steering_contract"]["recorded_to_radians_scale"] == pytest.approx(0.35)
    assert config["steering_contract"]["command_or_feedback"] == "command"
    assert config["steering_contract"]["left_right_sign_convention"] == "positive_left_negative_right"
    assert config["speed_contract"]["unit"] == "meters_per_second"
    assert config["speed_contract"]["unit_symbol"] == "m/s"
    assert config["speed_contract"]["meaning"] == "unresolved_command_or_actual_feedback_measurement"


def test_float64_uses_mcap_log_time_not_publish_time() -> None:
    record = SimpleNamespace(log_time=100, publish_time=999)
    decoded = SimpleNamespace(data=0.25)
    sample = scalar_sample_from_decoded(record, decoded)
    assert sample.record_time_ns == 100
    assert sample.publish_time_ns == 999
    assert sample.value == pytest.approx(0.25)


def test_float64_cdr_crosscheck_decodes_ros_little_endian_payload() -> None:
    payload = b"\x00\x01\x00\x00" + struct.pack("<d", 0.9049272093225353)
    assert decode_float64_cdr(payload) == pytest.approx(0.9049272093225353)


def test_image_header_and_bag_timestamps_remain_distinct() -> None:
    stamp = SimpleNamespace(sec=1, nanosec=5)
    pair = camera_timestamp_pair(2_000_000_010, stamp)
    assert pair == {
        "record_time_ns": 2_000_000_010,
        "header_time_ns": 1_000_000_005,
        "record_minus_header_ns": 1_000_000_005,
    }


def test_latest_causal_index_is_zero_order_hold() -> None:
    assert latest_causal_index([100, 200, 200, 300], 99) is None
    assert latest_causal_index([100, 200, 200, 300], 200) == 2
    assert latest_causal_index([100, 200, 200, 300], 250) == 2


def test_causal_sync_has_zero_future_labels() -> None:
    cameras = [_camera(0, 150), _camera(1, 250), _camera(2, 350)]
    steering = [ScalarSample(0, 100, 101, -0.1), ScalarSample(1, 300, 301, 0.1)]
    speed = [ScalarSample(0, 200, 201, 1.0)]
    metrics, assignments = causal_sync_metrics(cameras, steering, speed)
    assert metrics["future_label_violations"] == 0
    assert metrics["missing_causal_speed_count"] == 1
    assert assignments[1][0].record_time_ns == 100
    assert assignments[1][1].record_time_ns == 200
    assert all(
        scalar is None or scalar.record_time_ns <= camera.record_time_ns
        for camera, pair in zip(cameras, assignments)
        for scalar in pair
    )


def test_temporal_triplets_are_strict_and_gate_is_diagnostic_only() -> None:
    config = load_config(CONFIG_PATH)
    cameras = [_camera(0, 0), _camera(1, 66_000_000), _camera(2, 132_000_000)]
    scalar = ScalarSample(0, 0, 0, 0.0)
    assignments = [(scalar, scalar)] * 3
    result = temporal_readiness(
        cameras, assignments, config["simulator_temporal_reference"], incomplete=False
    )
    assert result["candidate_three_frame_sequences"] == 1
    assert result["valid_strict_causal_sequences"] == 1
    assert result["valid_strict_causal_sequences_with_current_causal_labels"] == 1
    assert result["timestamp_order_failure_count"] == 0
    assert result["simulator_comparison"]["existing_0p120_s_adjacent_gate_applied"] is False


def test_temporal_reports_adjacent_gap_count_without_enforcing_gate() -> None:
    config = load_config(V2_CONFIG_PATH)
    cameras = [_camera(0, 0), _camera(1, 130_000_000), _camera(2, 200_000_000)]
    scalar = ScalarSample(0, 0, 0, 0.0)
    result = temporal_readiness(
        cameras, [(scalar, scalar)] * 3, config["simulator_temporal_reference"], incomplete=False
    )
    assert result["valid_strict_causal_sequences"] == 1
    assert result["simulator_comparison"]["adjacent_gap_over_0p120_s_count"] == 1
    assert result["simulator_comparison"]["existing_0p120_s_adjacent_gate_applied"] is False


def test_out_of_range_reconciliation_keeps_values_and_adds_speed_phase_context() -> None:
    config = load_config(V2_CONFIG_PATH)
    steering = [
        ScalarSample(index, time, time, value)
        for index, (time, value) in enumerate(
            [(100, 0.4), (200, 0.4), (300, 0.2), (400, -0.5), (500, -0.5), (600, 0.36)]
        )
    ]
    speed = [
        ScalarSample(0, 50, 50, 0.0),
        ScalarSample(1, 250, 250, 1.0),
        ScalarSample(2, 550, 550, 0.0),
    ]
    result = analyze_steering(
        steering, config, speed_samples=speed, metadata_start_ns=0
    )
    detail = result["out_of_range_reconciliation"]
    assert result["distribution_recorded_raw"]["max"] == pytest.approx(0.4)
    assert result["distribution_recorded_raw"]["min"] == pytest.approx(-0.5)
    assert result["distribution_rad"]["max"] == pytest.approx(0.14)
    assert result["distribution_rad"]["min"] == pytest.approx(-0.175)
    assert result["recorded_to_radians"] == {
        "required": True,
        "scale": 0.35,
        "offset": 0.0,
        "formula": "steering_rad = steering_recorded * 0.35",
        "clipping_applied": False,
        "applies_to": "every finite steering sample",
        "evidence": config["steering_contract"]["recorded_scale_evidence"],
    }
    assert result["range_check"]["outside_confirmed_range_count"] == 0
    assert result["semantics"]["command_or_actual_actuator_feedback"] == "command"
    assert result["semantics"]["positive_direction"] == "LEFT"
    assert result["semantics"]["negative_direction"] == "RIGHT"
    assert result["semantics"]["selective_clipping_permitted"] is False
    assert detail["clipping_applied"] is False
    assert detail["outside_count"] == 5
    assert detail["outside_fraction"] == pytest.approx(5 / 6)
    assert detail["resolved_by_confirmed_rescaling"] is True
    assert detail["physical_outside_count_after_rescaling"] == 0
    assert detail["temporal_location_episode_count"] == 3
    assert detail["repeated_consecutive_plateau_count"] == 2
    assert detail["causal_speed_relation"]["exact_zero_count"] == 3
    assert detail["causal_speed_relation"]["nonzero_count"] == 2
    assert detail["driving_phase_counts"] == {
        "active_nonzero_speed": 2,
        "stationary_prefix": 2,
        "stationary_suffix": 1,
    }


def test_speed_reports_all_nonzero_windows_and_longest() -> None:
    samples = [
        ScalarSample(index, index * 100, index * 100, value)
        for index, value in enumerate([0.0, 1.0, 1.0, 0.0, 2.0, 0.0])
    ]
    result = analyze_speed(samples, metadata_start_ns=0, incomplete=False)
    assert [item["record_count"] for item in result["active_driving_candidate_windows"]] == [2, 1]
    assert result["active_driving_candidate_window"]["record_count"] == 2


def test_speed_uses_confirmed_mps_unit_without_guessing_command_or_feedback() -> None:
    config = load_config(V2_CONFIG_PATH)
    samples = [ScalarSample(0, 0, 0, 0.5), ScalarSample(1, 100, 100, 1.0)]
    result = analyze_speed(
        samples,
        metadata_start_ns=0,
        incomplete=False,
        speed_contract=config["speed_contract"],
    )
    assert result["unit"] == "meters_per_second"
    assert result["unit_symbol"] == "m/s"
    assert result["distribution_mps"] == result["distribution_raw"]
    assert result["meaning"] == "unresolved_command_or_actual_feedback_measurement"


def test_v2_refuses_to_overwrite_historical_v1_results() -> None:
    with pytest.raises(RealBagAuditError, match="must not overwrite"):
        run_audit(V2_CONFIG_PATH, ROOT / "results" / "real_bag_audit_v1")


def test_cross_bag_consistency_checks_all_three_contracts() -> None:
    bags = [_bag_result(f"bag_0{index}") for index in range(1, 4)]
    assert cross_bag_consistency(bags)["camera"]["same_contract_on_readable_messages"] is True
    bags[2] = _bag_result("bag_03", width=640)
    assert cross_bag_consistency(bags)["camera"]["same_contract_on_readable_messages"] is False


def test_simulator_comparison_ignores_per_bag_contract_counts() -> None:
    config = load_config(CONFIG_PATH)
    bags = [_bag_result(f"bag_0{index}") for index in range(1, 4)]
    for count, bag in enumerate(bags, start=1):
        bag["camera"]["uniform_contract"]["count"] = count
    result = simulator_camera_comparison(bags, config)
    assert result["resolution_match"] is True
    assert result["aspect_ratio_match"] is True
    assert result["encoding_match"] is True


def test_no_odom_roi_training_or_simulator_side_effects_are_configured() -> None:
    config = load_config(CONFIG_PATH)
    required_names = {item["name"] for item in config["required_topics"].values()}
    assert required_names == {"/camera/image_raw", "/steering", "/speed"}
    assert "/odom" not in required_names
    assert config["real_camera_roi"]["auto_apply_simulator_roi"] is False
    assert config["real_camera_roi"]["status"].startswith("unresolved")
    assert config["scope_guards"] == {
        "real_data_audit_only": True,
        "require_odometry": False,
        "generate_training_dataset": False,
        "invoke_training": False,
        "drive_simulator": False,
        "modify_docker": False,
        "modify_bags": False,
        "auto_apply_simulator_roi": False,
    }


def test_v2_real_camera_roi_v1_matches_human_approval() -> None:
    config = load_config(V2_CONFIG_PATH)
    roi = config["real_camera_roi"]
    assert roi["version"] == "real_camera_roi_v1"
    assert roi["status"] == "approved"
    assert roi["source"] == {
        "width": 480,
        "height": 360,
        "color_space": "RGB",
        "ros_encoding": "rgb8",
    }
    assert roi["crop"] == {
        "x_start": 0,
        "x_end": 480,
        "y_start": 80,
        "y_end": 360,
        "end_coordinates_exclusive": True,
    }
    assert (roi["cropped_width"], roi["cropped_height"]) == (480, 280)
    assert roi["resize"]["output_width"] == 200
    assert roi["resize"]["output_height"] == 66
    assert roi["resize"]["interpolation"] == "bilinear"
    assert roi["post_resize_color_conversion"] == "existing RGB_to_YUV preprocessing"
    assert roi["temporal_input"] == {
        "frame_order": ["t_minus_2", "t_minus_1", "t"],
        "frame_count": 3,
        "causal": True,
    }
    assert roi["horizontal_crop_applied"] is False
    assert roi["camera_undistortion_applied"] is False
    assert roi["simulator_y_160_360_crop_used"] is False
    assert roi["apply_to_audit_previews"] is False


def test_v2_rejects_simulator_or_drifted_real_roi(tmp_path: Path) -> None:
    payload = json.loads(V2_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["real_camera_roi"]["crop"]["y_start"] = 160
    path = tmp_path / "bad_roi.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RealBagAuditError, match="Real Camera ROI V1 contract"):
        load_config(path)


def test_v2_rejects_unconfirmed_speed_command_guess(tmp_path: Path) -> None:
    payload = json.loads(V2_CONFIG_PATH.read_text(encoding="utf-8"))
    payload["speed_contract"]["meaning"] = "command"
    path = tmp_path / "guessed_speed_semantics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RealBagAuditError, match="steering or speed contract"):
        load_config(path)


def test_runner_imports_only_the_audit_entrypoint() -> None:
    source = (ROOT / "scripts" / "run_real_bag_audit_v1.py").read_text(encoding="utf-8")
    assert "physicar_e2e.real_bag_audit import main" in source
    assert "pilotnet_training" not in source
    assert "dataset_extractor" not in source
    assert "sim_client" not in source

    v2_source = (ROOT / "scripts" / "run_real_bag_audit_v2.py").read_text(encoding="utf-8")
    assert "physicar_e2e.real_bag_audit import main" in v2_source
    assert "real_bag_audit_v2.json" in v2_source


def test_config_json_is_compact_git_ready_evidence() -> None:
    parsed = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert parsed["steering_contract"]["unit"] == "radians"
    assert parsed["steering_contract"]["confirmed_numeric_range_rad"] == [-0.35, 0.35]
