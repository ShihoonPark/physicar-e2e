"""Focused guards for Real Camera–Steering Phase Audit V1."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from physicar_e2e import real_camera_steering_phase_audit as audit


def commands() -> list[audit.ScalarCommand]:
    return [
        audit.ScalarCommand(100, -1.0, 0),
        audit.ScalarCommand(200, 0.25, 1),
        audit.ScalarCommand(300, 1.0, 2),
    ]


def test_previous_lookup_is_latest_lte_and_handles_boundary() -> None:
    assert audit.previous_lookup(commands(), 99) is None
    assert audit.previous_lookup(commands(), 100).index == 0
    assert audit.previous_lookup(commands(), 299).index == 1
    assert audit.previous_lookup(commands(), 300).index == 2


def test_next_lookup_is_first_gte_and_handles_boundary() -> None:
    assert audit.next_lookup(commands(), 100).index == 0
    assert audit.next_lookup(commands(), 101).index == 1
    assert audit.next_lookup(commands(), 300).index == 2
    assert audit.next_lookup(commands(), 301) is None


def test_nearest_lookup_minimizes_absolute_offset_and_tie_uses_previous() -> None:
    assert audit.nearest_lookup(commands(), 140).index == 0
    assert audit.nearest_lookup(commands(), 160).index == 1
    assert audit.nearest_lookup(commands(), 150).index == 0
    assert audit.nearest_lookup([], 150) is None


def test_physical_steering_scaling_occurs_exactly_once_without_clipping() -> None:
    assert audit.STEERING_SCALE_RAD == 0.35
    assert audit.scale_steering(1.0) == pytest.approx(0.35)
    assert audit.scale_steering(-0.5) == pytest.approx(-0.175)
    assert audit.scale_steering(2.0) == pytest.approx(0.70)
    with pytest.raises(audit.PhaseAuditError):
        audit.scale_steering(float("nan"))


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"exact_deployed_publisher_attributed": False}, "INCONCLUSIVE"),
        ({
            "exact_deployed_publisher_attributed": True,
            "command_computed_in_camera_callback": True,
        }, "POST_CAMERA_COMMAND_CORRECT"),
        ({
            "exact_deployed_publisher_attributed": True,
            "independent_command_source": True,
        }, "INDEPENDENT_CONTROL_STREAM"),
        ({
            "exact_deployed_publisher_attributed": True,
            "active_command_is_intended_target": True,
        }, "PREVIOUS_CAUSAL_COMMAND_CORRECT"),
        ({
            "exact_deployed_publisher_attributed": True,
            "command_computed_in_camera_callback": True,
            "independent_command_source": True,
        }, "INCONCLUSIVE"),
    ],
)
def test_semantic_classification_uses_unambiguous_code_provenance(
    kwargs: dict[str, bool], expected: str
) -> None:
    assert audit.classify_semantics(**kwargs) == expected
    assert expected in audit.SEMANTIC_DECISIONS


def test_source_semantics_report_is_evidence_backed_and_inconclusive() -> None:
    evidence = audit.audit_source_semantics()
    assert evidence["requested_source_present"] is False
    assert evidence["recovered_track_drive"]["archive_present"] is True
    assert evidence["recovered_track_drive"]["data_logger"]["finding"].endswith(
        "It creates no publisher."
    )
    assert evidence["recovered_track_drive"]["topic_match"]["exact_match"] is False
    assert evidence["exact_deployed_steering_publisher_attributed"] is False
    assert evidence["semantic_decision"] == "INCONCLUSIVE"
    assert evidence["platform_source"]["external_dependency_read_only"] is True
    answers = evidence["requested_trace_answers"]
    assert set(answers) == {
        "camera_publisher", "steering_publisher", "speed_publisher",
        "steering_computation_function", "image_compute_publish_order",
        "data_logger_role", "teleoperation_role", "command_hold",
        "representation_mismatch",
    }
    assert "does not republish" in answers["data_logger_role"]
    assert "holding steering" in answers["command_hold"]


def test_audit_source_contains_no_training_or_driving_implementation() -> None:
    source = inspect.getsource(audit)
    tree = ast.parse(source)
    forbidden_attributes = {"backward", "step", "zero_grad", "fit", "train"}
    forbidden_names = {"DataLoader", "Optimizer", "Adam", "SGD"}
    called_attributes = {
        node.func.attr for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    loaded_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    assert called_attributes.isdisjoint(forbidden_attributes)
    assert loaded_names.isdisjoint(forbidden_names)
    assert audit.TRAINING_PERMITTED is False
    assert audit.DRIVING_PERMITTED is False
    assert audit.DATASET_MODIFICATION_PERMITTED is False
    assert "subprocess" not in source
    assert "rclpy" not in source


def test_report_generation_mentions_source_lines_and_no_training() -> None:
    source = audit.audit_source_semantics()
    assert source["recovered_track_drive"]["camera_callback_e2e"]["callback_order"]["lines"]
    assert source["platform_source"]["exact_topic_publishers"]["lines"]
    assert source["platform_source"]["driver_subscribers"]["lines"]
    assert source["platform_source"]["held_steering_evidence"]["lines"]


def test_manifest_identity_is_the_frozen_real_dataset_v1() -> None:
    assert audit.sha256_file(audit.MANIFEST_PATH) == audit.MANIFEST_SHA256


def test_module_does_not_open_bags_or_frozen_artifacts_for_writing() -> None:
    source = Path(audit.__file__).read_text(encoding="utf-8")
    assert '.open("wb")' not in source
    assert '.open("w")' not in source
    assert "write_bytes" in source  # compact result JSON only
    assert "output_dir" in source
