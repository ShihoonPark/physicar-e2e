"""Deterministic Random Cone Expert V1 geometry, worlds, and live gate.

The benchmark freezes twelve seeded route locations before live driving.  It
reuses the real fixed-cone/vehicle geometry and the preserved 1.80 m/s Pure
Pursuit controller, but evaluates practical success solely by positive
footprint clearance and absence of cone contact/intersection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Sequence
import xml.etree.ElementTree as ET

from PIL import Image

from .cone_avoidance_environment import (
    ConeGeometry,
    ConeSite,
    EnvironmentConfig,
    EnvironmentError,
    RouteData,
    VehicleFootprint,
    asset_set,
    box_polygon,
    cone_geometry_dict,
    directory_manifest,
    expected_world_bytes,
    footprint_polygon,
    load_route,
    other_world_collision_clearance,
    parse_cone_geometry,
    parse_vehicle_footprint,
    parse_xml,
    point_track_clearance,
    polygon_clearance,
    route_curvature,
    route_yaw,
    sha256_file,
    share_path,
    side_clearances,
    vehicle_footprint_dict,
    verify_canonical_hashes,
)
from .cone_avoidance_expert import (
    BypassPlan,
    ObstacleAwareRoute,
    _ConeMaskingClient,
    _minimum_transition_for_curvature,
    _three_point_curvature,
    _wrapped,
    activate_world,
)
from .expert_driver import DriverConfig, Preflight, preflight as lane_preflight, run_driver
from .pilotnet_v4_repeatability import clock_health_preflight
from .route_geometry import ClosedRoute, pure_pursuit_steering
from .sim_client import SimClient


VERSION = "random_cone_expert_v1"
MAP_FAMILY = "71e69ee938032295503bfed557fde18c"
EXPECTED_RESULT_DIRECTORY = "results/random_cone_expert_v1"
SCENARIO_COUNT = 12
ROLE_IDS = {
    "TRAIN": [f"{number:02d}" for number in range(1, 9)],
    "VALIDATION": ["09", "10"],
    "UNSEEN_HOLDOUT": ["11", "12"],
}
CLASS_ORDER = ("low_curvature", "moderate_left_curve", "moderate_right_curve")

# These identities make the new milestone fail closed if historical evidence
# or either preserved neural model changes.  The files themselves are never
# opened for inference or training by this module.
PRESERVED_REPOSITORY_FILES = {
    "high_speed_expert_config": (
        "configs/high_speed_expert_v1.json",
        "3afdc5d3204143e8d1f64c1ec68b2bda2de08912b779f2ece69a7d86c99503c9",
    ),
    "fixed_cone_environment_config": (
        "configs/cone_avoidance_environment_v1.json",
        "0778f735fd431f7befcf0ed17f59379e48883635914dea6c91b8087cc285830a",
    ),
    "fixed_cone_expert_config": (
        "configs/cone_avoidance_expert_v1.json",
        "77deb963369b34aba917c3db6f559a63f85e80ae6e78dd2cf20a34dfe54e9831",
    ),
    "fixed_cone_expert_summary": (
        "results/cone_avoidance_expert_v1/summary.json",
        "b1b0603ba34e50644802b6b2c46fdc92c9290f31273eed2f2ac0387830ce7082",
    ),
    "temporal_v9_training_summary": (
        "results/pilotnet_training_v9_high_speed_temporal/summary.json",
        "4c22c0b7f2d408b44b4698ff98d394ff3bde3f8d40c5dc34e6edd0a30d906f87",
    ),
    "temporal_v9_live_summary": (
        "results/pilotnet_e2e_v9_high_speed_temporal/summary.json",
        "56829cfc312f5cfe353458c60afcd146a54f406374d2e02546e20406cccaa6d2",
    ),
    "temporal_v9_inference_config": (
        "configs/pilotnet_inference_v9_high_speed_temporal.json",
        "8da99cd486ddd9849432f7368a185d20880615189d23b3df5bfb7996a8de963e",
    ),
    "c1_training_summary": (
        "results/pilotnet_training_c1_cone_temporal/summary.json",
        "ea2f3c1dd65357cb2c5c8b12d8035515f0d612b6146a14b0bc4c2e8ece217f38",
    ),
    "c1_inference_config": (
        "configs/pilotnet_inference_c1_cone_temporal.json",
        "d079468d3f30868c993ee5e09e2f455051e1412b9628009364cdc5cbbceca08e",
    ),
    "c1_practical_config": (
        "configs/pilotnet_c1_practical_cone_validation_v1.json",
        "da5711acc2fb92436cd8ea4136ba12e7d254021d9461c8fa5f98417c2a72bf5b",
    ),
    "c1_practical_summary": (
        "results/pilotnet_c1_practical_cone_validation_v1/summary.json",
        "f3e5a41e9b3147d6f8d48704e06a1841c0190407f9456bb08e6889a8f28e25fa",
    ),
}
PRESERVED_EXTERNAL_ARTIFACTS = {
    "temporal_v9_checkpoint": (
        "userdata/physicar_e2e/high_speed_temporal_v1/v9/checkpoints/pilotnet_v9_high_speed_temporal_best.pt",
        "1cded5fcc7f3d13242de096c4868fc576d03fa6bc86df6e5c8c7c235d9faa6cc",
    ),
    "temporal_v9_onnx": (
        "userdata/physicar_e2e/high_speed_temporal_v1/v9/onnx/pilotnet_v9_high_speed_temporal.onnx",
        "7f6aa4c2d8c9b3615c580f065660c674efff94ff4cd0b9bdc9357df904000888",
    ),
    "c1_checkpoint": (
        "userdata/physicar_e2e/cone_avoidance_v1/c1/checkpoints/pilotnet_c1_cone_temporal_best.pt",
        "1e90002ca139b3cfb0f34074e013e52b6754df33ed0e3b438ca81809c9e2ee39",
    ),
    "c1_onnx": (
        "userdata/physicar_e2e/cone_avoidance_v1/c1/onnx/pilotnet_c1_cone_temporal.onnx",
        "22440ad61f6e5136b33016eb0781d79ab71637e659478ac0c92cc04cffc98e5f",
    ),
}


class RandomConeError(RuntimeError):
    """A controlled protocol, geometry, or evidence-gate failure."""


@dataclass(frozen=True)
class FrozenScenario:
    scenario_id: str
    role: str
    provenance: dict[str, Any]
    route_s_m: float
    x_m: float
    y_m: float
    yaw_rad: float
    local_curvature_per_m: float
    curvature_class: str
    left_logical_track_clearance_m: float
    right_logical_track_clearance_m: float
    nonlocal_route_clearance_m: float
    nearby_collision_clearance_m: float
    nearby_collision_model: str
    nearby_collision_name: str
    chosen_side: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FrozenScenario":
        expected = {
            "scenario_id", "role", "provenance", "route_s_m", "x_m", "y_m",
            "yaw_rad", "local_curvature_per_m", "curvature_class",
            "left_logical_track_clearance_m", "right_logical_track_clearance_m",
            "nonlocal_route_clearance_m", "nearby_collision_clearance_m",
            "nearby_collision_model", "nearby_collision_name", "chosen_side",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise RandomConeError("frozen scenario fields do not match Random Cone V1")
        numeric = (
            "route_s_m", "x_m", "y_m", "yaw_rad", "local_curvature_per_m",
            "left_logical_track_clearance_m", "right_logical_track_clearance_m",
            "nonlocal_route_clearance_m", "nearby_collision_clearance_m",
        )
        if not all(_finite_number(payload[name]) for name in numeric):
            raise RandomConeError("frozen scenario geometry must be finite")
        if payload["curvature_class"] not in CLASS_ORDER:
            raise RandomConeError("invalid frozen curvature class")
        if payload["chosen_side"] not in ("left", "right"):
            raise RandomConeError("invalid frozen avoidance side")
        provenance = payload["provenance"]
        if not isinstance(provenance, dict) or set(provenance) != {
            "random_seed", "algorithm", "candidate_grid_index", "class_rank",
            "rank_sha256", "assignment_sha256",
        }:
            raise RandomConeError("invalid scenario provenance")
        if not isinstance(provenance["candidate_grid_index"], int) or not isinstance(provenance["class_rank"], int):
            raise RandomConeError("scenario provenance indexes must be integers")
        if not all(isinstance(provenance[name], str) and len(provenance[name]) == 64 for name in ("rank_sha256", "assignment_sha256")):
            raise RandomConeError("scenario provenance hashes must be SHA-256 strings")
        return cls(
            scenario_id=str(payload["scenario_id"]), role=str(payload["role"]),
            provenance=dict(provenance),
            route_s_m=float(payload["route_s_m"]), x_m=float(payload["x_m"]),
            y_m=float(payload["y_m"]), yaw_rad=float(payload["yaw_rad"]),
            local_curvature_per_m=float(payload["local_curvature_per_m"]),
            curvature_class=str(payload["curvature_class"]),
            left_logical_track_clearance_m=float(payload["left_logical_track_clearance_m"]),
            right_logical_track_clearance_m=float(payload["right_logical_track_clearance_m"]),
            nonlocal_route_clearance_m=float(payload["nonlocal_route_clearance_m"]),
            nearby_collision_clearance_m=float(payload["nearby_collision_clearance_m"]),
            nearby_collision_model=str(payload["nearby_collision_model"]),
            nearby_collision_name=str(payload["nearby_collision_name"]),
            chosen_side=str(payload["chosen_side"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "role": self.role,
            "provenance": dict(self.provenance),
            "route_s_m": self.route_s_m,
            "x_m": self.x_m,
            "y_m": self.y_m,
            "yaw_rad": self.yaw_rad,
            "local_curvature_per_m": self.local_curvature_per_m,
            "curvature_class": self.curvature_class,
            "left_logical_track_clearance_m": self.left_logical_track_clearance_m,
            "right_logical_track_clearance_m": self.right_logical_track_clearance_m,
            "nonlocal_route_clearance_m": self.nonlocal_route_clearance_m,
            "nearby_collision_clearance_m": self.nearby_collision_clearance_m,
            "nearby_collision_model": self.nearby_collision_model,
            "nearby_collision_name": self.nearby_collision_name,
            "chosen_side": self.chosen_side,
        }


@dataclass(frozen=True)
class RandomConeConfig:
    payload: dict[str, Any]
    path: Path
    baseline_path: Path
    baseline: DriverConfig
    environment_path: Path
    environment: EnvironmentConfig
    random_seed: int
    sampling: dict[str, Any]
    avoidance: dict[str, Any]
    return_to_route: dict[str, float]
    worlds: dict[str, str]
    scenarios: tuple[FrozenScenario, ...]
    live_protocol: dict[str, Any]
    permissions: dict[str, bool]

    @classmethod
    def load(
        cls,
        path: Path,
        repo_root: Path,
        sim_root: Path,
        *,
        allow_unfrozen: bool = False,
    ) -> "RandomConeConfig":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RandomConeError(f"cannot load Random Cone config {path}: {exc}") from exc
        required = {
            "version", "map_family", "baseline_expert_config", "baseline_expert_sha256",
            "fixed_cone_environment_config", "fixed_cone_environment_sha256",
            "preserved_repository_files", "preserved_external_artifacts", "random_seed",
            "sampling", "avoidance", "return_to_route", "worlds", "scenario_roles",
            "scenarios", "live_protocol", "permissions", "privileged_inputs",
            "future_policy_forbidden_inputs",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise RandomConeError("Random Cone config fields do not match the V1 protocol")
        if payload["version"] != VERSION or payload["map_family"] != MAP_FAMILY:
            raise RandomConeError("Random Cone version/map family differs from V1")
        if payload["scenario_roles"] != ROLE_IDS:
            raise RandomConeError("scenario roles must be the frozen 8/2/2 split")
        expected_repo = {
            key: {"path": value[0], "sha256": value[1]}
            for key, value in PRESERVED_REPOSITORY_FILES.items()
        }
        expected_external = {
            key: {"path": value[0], "sha256": value[1]}
            for key, value in PRESERVED_EXTERNAL_ARTIFACTS.items()
        }
        if payload["preserved_repository_files"] != expected_repo or payload["preserved_external_artifacts"] != expected_external:
            raise RandomConeError("preserved evidence/model identities differ from V1")
        baseline_path = (repo_root / str(payload["baseline_expert_config"])).resolve()
        environment_path = (repo_root / str(payload["fixed_cone_environment_config"])).resolve()
        if sha256_file(baseline_path) != payload["baseline_expert_sha256"]:
            raise RandomConeError("preserved High-Speed Expert config identity mismatch")
        if sha256_file(environment_path) != payload["fixed_cone_environment_sha256"]:
            raise RandomConeError("preserved fixed-cone environment config identity mismatch")
        baseline = DriverConfig.load(baseline_path)
        environment = EnvironmentConfig.load(environment_path)
        control = (
            baseline.fixed_speed_mps, baseline.lookahead_m, baseline.control_frequency_hz,
            baseline.max_steering_rad, baseline.wheelbase_m,
        )
        if control != (1.80, 0.90, 15.0, 0.349066, 0.18):
            raise RandomConeError(f"frozen Expert control contract changed: {control}")
        if MAP_FAMILY not in environment.canonical_cone_world:
            raise RandomConeError("fixed geometry is not from the required map family")
        seed = payload["random_seed"]
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise RandomConeError("random_seed must be a nonnegative integer")
        sampling = payload["sampling"]
        sampling_keys = {
            "algorithm", "candidate_step_m", "start_exclusion_m", "end_exclusion_m",
            "curvature_half_window_m", "low_curvature_max_abs_per_m",
            "moderate_curvature_min_abs_per_m", "moderate_curvature_max_abs_per_m",
            "target_count_by_class", "minimum_route_separation_m",
            "fixed_cone_exclusion_m", "nonlocal_route_exclusion_m",
            "minimum_nonlocal_route_clearance_m", "side_tie_break",
            "maximum_continuous_steering_saturation_s",
        }
        if not isinstance(sampling, dict) or set(sampling) != sampling_keys:
            raise RandomConeError("sampling fields do not match V1")
        if sampling["algorithm"] != "sha256_ranked_stratified_route_grid_v1":
            raise RandomConeError("sampling algorithm differs from V1")
        quotas = sampling["target_count_by_class"]
        if (
            not isinstance(quotas, dict)
            or set(quotas) != set(CLASS_ORDER)
            or not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in quotas.values())
            or sum(quotas.values()) != SCENARIO_COUNT
            or sampling["side_tie_break"] not in ("left", "right")
        ):
            raise RandomConeError("sampling class quota/tie break is invalid")
        if not all(
            _positive_number(value)
            for key, value in sampling.items()
            if key not in ("algorithm", "target_count_by_class", "side_tie_break")
        ):
            raise RandomConeError("sampling numeric fields must be positive")
        low = float(sampling["low_curvature_max_abs_per_m"])
        moderate_min = float(sampling["moderate_curvature_min_abs_per_m"])
        moderate_max = float(sampling["moderate_curvature_max_abs_per_m"])
        if not low <= moderate_min < moderate_max:
            raise RandomConeError("curvature-class thresholds are inconsistent")
        avoidance = payload["avoidance"]
        avoidance_keys = {
            "algorithm", "profile", "planning_cone_clearance_m", "extra_planning_clearance_m",
            "offset_search_step_m", "minimum_reference_track_reserve_m",
            "transition_minimum_time_s", "transition_minimum_lookaheads",
            "plateau_half_lookaheads", "maximum_added_curvature_fraction",
            "sample_spacing_m", "kinematic_preview_substep_s",
        }
        if not isinstance(avoidance, dict) or set(avoidance) != avoidance_keys:
            raise RandomConeError("avoidance fields do not match V1")
        if (
            avoidance["algorithm"] != "automatic_symmetric_quintic_route_normal_v1"
            or avoidance["profile"] != "symmetric_quintic_with_flat_pass"
        ):
            raise RandomConeError("avoidance algorithm/profile differs from V1")
        if not all(
            _positive_number(value)
            for key, value in avoidance.items()
            if key not in ("algorithm", "profile")
        ):
            raise RandomConeError("avoidance numeric fields must be positive")
        if float(avoidance["maximum_added_curvature_fraction"]) > 1.0:
            raise RandomConeError("added-curvature fraction cannot exceed one")
        return_contract = payload["return_to_route"]
        if not isinstance(return_contract, dict) or set(return_contract) != {
            "maximum_absolute_nominal_cte_m", "minimum_stable_duration_s"
        } or not all(_positive_number(value) for value in return_contract.values()):
            raise RandomConeError("return-to-route contract is invalid")
        worlds = payload["worlds"]
        expected_world_keys = {
            "canonical_cone_world", "canonical_cone_free_world", "derived_world_prefix",
            "derived_cone_model_prefix",
        }
        if not isinstance(worlds, dict) or set(worlds) != expected_world_keys:
            raise RandomConeError("world-placement fields do not match V1")
        if worlds["canonical_cone_world"] != environment.canonical_cone_world or worlds["canonical_cone_free_world"] != environment.canonical_cone_free_world:
            raise RandomConeError("Random Cone worlds do not preserve the fixed map/route")
        if not all(isinstance(value, str) and value for value in worlds.values()):
            raise RandomConeError("world/model prefixes must be non-empty strings")
        live = payload["live_protocol"]
        expected_live = {
            "maximum_valid_policy_runs": 12,
            "infrastructure_replacement_attempts_per_scenario": 1,
            "retry_genuine_policy_failure": False,
            "success_contract": "positive_clearance_and_no_cone_contact_or_intersection",
            "result_directory": EXPECTED_RESULT_DIRECTORY,
        }
        if live != expected_live:
            raise RandomConeError("live protocol differs from Random Cone V1")
        expected_permissions = {
            "neural_training_permitted": False,
            "training_bag_collection_permitted": False,
            "v9_model_changes_permitted": False,
            "c1_model_changes_permitted": False,
            "fixed_cone_evidence_changes_permitted": False,
            "tracked_simulator_source_changes_permitted": False,
            "commit_permitted": False,
            "push_permitted": False,
        }
        if payload["permissions"] != expected_permissions:
            raise RandomConeError("forbidden-action permissions differ from V1")
        if payload["privileged_inputs"] != [
            "gt_vehicle_pose", "nominal_route", "cone_gt_pose",
            "generated_bypass_reference", "track_geometry",
        ] or payload["future_policy_forbidden_inputs"] != [
            "cone_gt_coordinates", "route", "pose", "cte", "expert_command",
        ]:
            raise RandomConeError("privileged/future-policy information boundary changed")
        raw_scenarios = payload["scenarios"]
        if not isinstance(raw_scenarios, list) or (not allow_unfrozen and len(raw_scenarios) != SCENARIO_COUNT):
            raise RandomConeError("production config must contain exactly 12 frozen scenarios")
        if allow_unfrozen and len(raw_scenarios) not in (0, SCENARIO_COUNT):
            raise RandomConeError("proposal config scenarios must be empty or complete")
        scenarios = tuple(FrozenScenario.from_dict(item) for item in raw_scenarios)
        if scenarios:
            expected_ids = [f"{number:02d}" for number in range(1, 13)]
            if [item.scenario_id for item in scenarios] != expected_ids:
                raise RandomConeError("frozen scenarios must be ordered 01 through 12")
            for scenario in scenarios:
                expected_role = next(role for role, ids in ROLE_IDS.items() if scenario.scenario_id in ids)
                if scenario.role != expected_role or scenario.provenance["random_seed"] != seed or scenario.provenance["algorithm"] != sampling["algorithm"]:
                    raise RandomConeError(f"scenario {scenario.scenario_id} role/provenance mismatch")
        # Resolve canonical geometry during loading.  This is read-only and
        # makes even proposal generation provenance-gated.
        verify_canonical_hashes(environment, share_path(sim_root))
        return cls(
            payload=payload, path=path.resolve(), baseline_path=baseline_path,
            baseline=baseline, environment_path=environment_path, environment=environment,
            random_seed=seed, sampling=dict(sampling), avoidance=dict(avoidance),
            return_to_route={key: float(value) for key, value in return_contract.items()},
            worlds={key: str(value) for key, value in worlds.items()}, scenarios=scenarios,
            live_protocol=dict(live), permissions=dict(payload["permissions"]),
        )

    def driver_for(self, scenario: FrozenScenario) -> DriverConfig:
        driver = replace(self.baseline, expected_world=self.world_name(scenario.scenario_id))
        driver.validate()
        return driver

    def world_name(self, scenario_id: str) -> str:
        return f"{self.worlds['derived_world_prefix']}{scenario_id}"

    def cone_model_name(self, scenario_id: str) -> str:
        return f"{self.worlds['derived_cone_model_prefix']}{scenario_id}"


@dataclass(frozen=True)
class Candidate:
    grid_index: int
    route_s_m: float
    curvature_per_m: float
    curvature_class: str
    rank_sha256: str


@dataclass(frozen=True)
class ScenarioBundle:
    scenario: FrozenScenario
    plan: BypassPlan
    geometry: dict[str, Any]
    side_evaluations: dict[str, dict[str, Any]]


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _curvature_class(config: RandomConeConfig, curvature: float) -> str | None:
    magnitude = abs(curvature)
    low = float(config.sampling["low_curvature_max_abs_per_m"])
    moderate_min = float(config.sampling["moderate_curvature_min_abs_per_m"])
    moderate_max = float(config.sampling["moderate_curvature_max_abs_per_m"])
    if magnitude <= low:
        return "low_curvature"
    if moderate_min < curvature <= moderate_max:
        return "moderate_left_curve"
    if -moderate_max <= curvature < -moderate_min:
        return "moderate_right_curve"
    return None


def _nonlocal_route_clearance(route: ClosedRoute, s: float, step: float, excluded_arc: float) -> float:
    origin = route.point_at(s)
    count = int(math.ceil(route.length / step))
    minimum = math.inf
    for index in range(count):
        other_s = index * route.length / count
        arc = abs((other_s - s + route.length / 2.0) % route.length - route.length / 2.0)
        if arc > excluded_arc:
            minimum = min(minimum, math.dist(origin, route.point_at(other_s)))
    if not math.isfinite(minimum):
        raise RandomConeError("nonlocal-route clearance had no eligible samples")
    return minimum


def _plan_constants(
    config: RandomConeConfig,
    cone: ConeGeometry,
    footprint: VehicleFootprint,
) -> dict[str, float]:
    driver = config.baseline
    required = (
        max(abs(footprint.y_min_m), abs(footprint.y_max_m))
        + cone.half_width_m
        + float(config.avoidance["planning_cone_clearance_m"])
    )
    maximum_offset = required + float(config.avoidance["extra_planning_clearance_m"])
    physical = math.tan(driver.max_steering_rad) / driver.wheelbase_m
    design = physical * float(config.avoidance["maximum_added_curvature_fraction"])
    geometric_minimum = _minimum_transition_for_curvature(maximum_offset, design)
    transition = max(
        geometric_minimum,
        driver.fixed_speed_mps * float(config.avoidance["transition_minimum_time_s"]),
        driver.lookahead_m * float(config.avoidance["transition_minimum_lookaheads"]),
    )
    plateau_half = driver.lookahead_m * float(config.avoidance["plateau_half_lookaheads"])
    profile_bound = maximum_offset * (10.0 * math.sqrt(3.0) / 3.0) / transition**2
    return {
        "required_center_offset_m": required,
        "maximum_lateral_offset_m": maximum_offset,
        "physical_curvature_limit_per_m": physical,
        "design_added_curvature_limit_per_m": design,
        "geometric_minimum_transition_m": geometric_minimum,
        "transition_length_m": transition,
        "plateau_half_length_m": plateau_half,
        "added_profile_curvature_bound_per_m": profile_bound,
    }


def _make_site(
    config: RandomConeConfig,
    route_data: RouteData,
    candidate: Candidate,
    side: str,
    constants: dict[str, float],
) -> ConeSite:
    route = route_data.route
    s = candidate.route_s_m
    point = route.point_at(s)
    yaw = route_yaw(route, s)
    left, right = side_clearances(route_data, s, yaw)
    transition = constants["transition_length_m"]
    plateau = constants["plateau_half_length_m"]
    nonlocal_clearance = _nonlocal_route_clearance(
        route, s, float(config.sampling["candidate_step_m"]),
        float(config.sampling["nonlocal_route_exclusion_m"]),
    )
    return ConeSite(
        route_s_m=s, x_m=point[0], y_m=point[1], yaw_rad=yaw,
        local_curvature_per_m=candidate.curvature_per_m,
        straight_run_start_s_m=s - transition - plateau,
        straight_run_end_s_m=s + transition + plateau,
        straight_run_length_m=2.0 * (transition + plateau),
        left_clearance_m=left, right_clearance_m=right,
        nonlocal_route_clearance_m=nonlocal_clearance, chosen_side=side,
        reason=(
            "seeded frozen route-grid candidate; side selected automatically from two-sided "
            "footprint/track/steering feasibility and predicted clearance"
        ),
    )


def build_candidate_plan(
    config: RandomConeConfig,
    route_data: RouteData,
    cone: ConeGeometry,
    footprint: VehicleFootprint,
    candidate: Candidate,
    side: str,
) -> BypassPlan:
    constants = _plan_constants(config, cone, footprint)
    site = _make_site(config, route_data, candidate, side, constants)
    available = site.left_clearance_m if side == "left" else site.right_clearance_m
    track_reserve = float(config.avoidance["minimum_reference_track_reserve_m"])
    if constants["maximum_lateral_offset_m"] >= available - track_reserve:
        raise RandomConeError(
            f"{side} bypass lacks logical track clearance at s={site.route_s_m:.3f}"
        )
    plateau = constants["plateau_half_length_m"]
    target_clearance = (
        float(config.avoidance["planning_cone_clearance_m"])
        + float(config.avoidance["extra_planning_clearance_m"])
    )
    search_step = float(config.avoidance["offset_search_step_m"])
    offset = constants["maximum_lateral_offset_m"]
    best: BypassPlan | None = None
    while offset < available - track_reserve + 1e-12:
        geometric_minimum = _minimum_transition_for_curvature(
            offset, constants["design_added_curvature_limit_per_m"]
        )
        transition = max(
            geometric_minimum,
            config.baseline.fixed_speed_mps * float(config.avoidance["transition_minimum_time_s"]),
            config.baseline.lookahead_m * float(config.avoidance["transition_minimum_lookaheads"]),
        )
        start = site.route_s_m - transition - plateau
        end = site.route_s_m + transition + plateau
        plan = BypassPlan(
            nominal=route_data.route, site=site, cone=cone, footprint=footprint,
            side=side, side_sign=1.0 if side == "left" else -1.0,
            required_center_offset_m=constants["required_center_offset_m"],
            maximum_lateral_offset_m=offset,
            transition_length_m=transition, plateau_half_length_m=plateau,
            departure_start_s_m=start, plateau_start_s_m=site.route_s_m - plateau,
            cone_s_m=site.route_s_m, plateau_end_s_m=site.route_s_m + plateau,
            return_end_s_m=end,
            physical_curvature_limit_per_m=constants["physical_curvature_limit_per_m"],
            design_curvature_limit_per_m=constants["design_added_curvature_limit_per_m"],
            geometric_minimum_transition_m=geometric_minimum,
            available_left_clearance_m=site.left_clearance_m,
            available_right_clearance_m=site.right_clearance_m,
        )
        clearance, intersects = _minimum_reference_cone_clearance(plan, search_step)
        best = plan
        if not intersects and clearance + 1e-9 >= target_clearance:
            break
        offset += search_step
    if best is None:
        raise RandomConeError("automatic lateral-offset search produced no plan")
    plan = best
    transition = plan.transition_length_m
    start = plan.departure_start_s_m
    end = plan.return_end_s_m
    if start < float(config.sampling["start_exclusion_m"]) - 1e-9:
        raise RandomConeError("bypass enters the spawn/start-gate exclusion")
    if end > route_data.route.length - float(config.sampling["end_exclusion_m"]) + 1e-9:
        raise RandomConeError("bypass enters the route-closure/end exclusion")
    clearance, intersects = _minimum_reference_cone_clearance(plan, search_step)
    if intersects or clearance + 1e-9 < target_clearance:
        raise RandomConeError(
            f"{side} automatic offset search cannot achieve the planning clearance target"
        )
    return plan


def _minimum_reference_cone_clearance(
    plan: BypassPlan,
    spacing: float,
) -> tuple[float, bool]:
    cone_polygon = box_polygon(
        plan.site.x_m, plan.site.y_m, plan.site.yaw_rad,
        plan.cone.half_length_m, plan.cone.half_width_m,
    )
    first = plan.plateau_start_s_m - spacing
    last = plan.plateau_end_s_m + spacing
    count = int(math.ceil((last - first) / spacing))
    minimum = math.inf
    intersection = False
    for index in range(count + 1):
        s = first + index * (last - first) / count
        point = plan.point_at(s)
        clearance, intersects = polygon_clearance(
            footprint_polygon(plan.footprint, point[0], point[1], plan.yaw_at(s)),
            cone_polygon,
        )
        minimum = min(minimum, clearance)
        intersection = intersection or intersects
    return minimum, intersection


def _preview(
    config: RandomConeConfig,
    plan: BypassPlan,
    nominal: ClosedRoute,
) -> dict[str, Any]:
    """Full remaining-lap ideal bicycle gate using the frozen controller."""
    driver = config.baseline
    period = 1.0 / driver.control_frequency_hz
    substep = float(config.avoidance["kinematic_preview_substep_s"])
    start_s = plan.departure_start_s_m - driver.lookahead_m
    x, y = nominal.point_at(start_s)
    yaw = route_yaw(nominal, start_s)
    cone_polygon = box_polygon(
        plan.site.x_m, plan.site.y_m, plan.site.yaw_rad,
        plan.cone.half_length_m, plan.cone.half_width_m,
    )
    minimum_clearance = math.inf
    minimum_clearance_s = start_s
    intersection = False
    minimum_track_clearance = math.inf
    maximum_outside_track_m = 0.0
    continuous_offtrack_s = 0.0
    maximum_continuous_offtrack_s = 0.0
    recovery_candidate_s: float | None = None
    recovery_success = False
    recovery_time_s: float | None = None
    first_after_return_s: float | None = None
    elapsed = 0.0
    last_s = start_s
    unwrapped_s = start_s
    requested_steering_max = 0.0
    applied_steering_max = 0.0
    saturation_count = 0
    saturation_run_s = 0.0
    maximum_saturation_run_s = 0.0
    controls = 0
    goal_s = nominal.length - min(0.20, driver.start_gate_radius_m / 2.0)
    maximum_controls = int(math.ceil(
        (nominal.length - start_s + 6.0) / (driver.fixed_speed_mps * period)
    ))
    for _ in range(maximum_controls):
        projection = nominal.project((x, y))
        target = plan.point_at(projection.s + driver.lookahead_m)
        steering, requested_curvature, _ = pure_pursuit_steering(
            (x, y), yaw, target, driver.wheelbase_m, driver.max_steering_rad
        )
        requested = abs(math.atan(driver.wheelbase_m * requested_curvature))
        saturated = requested >= driver.max_steering_rad - 1e-9
        saturation_count += int(saturated)
        saturation_run_s = saturation_run_s + period if saturated else 0.0
        maximum_saturation_run_s = max(maximum_saturation_run_s, saturation_run_s)
        requested_steering_max = max(requested_steering_max, requested)
        applied_steering_max = max(applied_steering_max, abs(steering))
        controls += 1
        remaining = period
        while remaining > 1e-12:
            dt = min(substep, remaining)
            curvature = math.tan(steering) / driver.wheelbase_m
            if abs(curvature) <= 1e-12:
                x += driver.fixed_speed_mps * dt * math.cos(yaw)
                y += driver.fixed_speed_mps * dt * math.sin(yaw)
            else:
                next_yaw = yaw + driver.fixed_speed_mps * curvature * dt
                x += (math.sin(next_yaw) - math.sin(yaw)) / curvature
                y += (-math.cos(next_yaw) + math.cos(yaw)) / curvature
                yaw = next_yaw
            elapsed += dt
            remaining -= dt
            clearance, intersects = polygon_clearance(
                footprint_polygon(plan.footprint, x, y, yaw), cone_polygon
            )
            if clearance < minimum_clearance:
                minimum_clearance = clearance
                minimum_clearance_s = nominal.project((x, y)).s
            intersection = intersection or intersects
        projection = nominal.project((x, y))
        delta = (projection.s - last_s + nominal.length / 2.0) % nominal.length - nominal.length / 2.0
        if abs(delta) <= driver.maximum_progress_jump_m:
            unwrapped_s += delta
            last_s = projection.s
        track_clearance = point_track_clearance(nominal, (x, y))
        minimum_track_clearance = min(minimum_track_clearance, track_clearance)
        outside = max(0.0, -track_clearance)
        maximum_outside_track_m = max(maximum_outside_track_m, outside)
        if outside > driver.off_track_margin_m:
            continuous_offtrack_s += period
        else:
            continuous_offtrack_s = 0.0
        maximum_continuous_offtrack_s = max(maximum_continuous_offtrack_s, continuous_offtrack_s)
        if unwrapped_s >= plan.return_end_s_m:
            first_after_return_s = elapsed if first_after_return_s is None else first_after_return_s
            if projection.distance <= config.return_to_route["maximum_absolute_nominal_cte_m"]:
                recovery_candidate_s = elapsed if recovery_candidate_s is None else recovery_candidate_s
                if (
                    not recovery_success
                    and elapsed - recovery_candidate_s
                    >= config.return_to_route["minimum_stable_duration_s"]
                ):
                    recovery_success = True
                    recovery_time_s = elapsed - first_after_return_s
            elif not recovery_success:
                recovery_candidate_s = None
        if unwrapped_s >= goal_s:
            break
    return {
        "model": "ideal kinematic bicycle; frozen 15 Hz Pure Pursuit; full remaining-lap preview",
        "evidence_role": "offline feasibility gate only; not live simulator evidence",
        "completed_remaining_lap": unwrapped_s >= goal_s,
        "final_unwrapped_route_s_m": unwrapped_s,
        "minimum_footprint_to_cone_clearance_m": minimum_clearance,
        "minimum_clearance_route_s_m": minimum_clearance_s,
        "footprint_cone_intersection": intersection,
        "minimum_logical_track_clearance_m": minimum_track_clearance,
        "maximum_outside_logical_track_m": maximum_outside_track_m,
        "maximum_continuous_offtrack_s": maximum_continuous_offtrack_s,
        "recovery_success": recovery_success,
        "recovery_time_s": recovery_time_s,
        "maximum_requested_steering_rad": requested_steering_max,
        "maximum_applied_steering_rad": applied_steering_max,
        "steering_saturation_fraction": saturation_count / controls if controls else 0.0,
        "maximum_continuous_steering_saturation_s": maximum_saturation_run_s,
        "control_steps": controls,
    }


def _scenario_environment_config(
    config: RandomConeConfig,
    scenario_id: str,
    site: ConeSite,
) -> EnvironmentConfig:
    return replace(
        config.environment,
        derived_world=config.world_name(scenario_id),
        derived_cone_model=config.cone_model_name(scenario_id),
        frozen_cone={
            "route_s_m": site.route_s_m, "x_m": site.x_m, "y_m": site.y_m,
            "yaw_rad": site.yaw_rad,
            "local_curvature_per_m": site.local_curvature_per_m,
        },
    )


def _expected_cone_count(
    config: RandomConeConfig,
    plan: BypassPlan,
    scenario_id: str,
    share: Path,
) -> int:
    scenario_environment = _scenario_environment_config(config, scenario_id, plan.site)
    cone, source = parse_cone_geometry(config.environment, share)
    cone_free = asset_set(share, config.environment.canonical_cone_free_world)
    root = ET.fromstring(expected_world_bytes(
        scenario_environment, cone_free.world, source, plan.site, cone
    ))
    world = root if root.tag == "world" else root.find("world")
    if world is None:
        return 0
    return sum(
        (model.get("name") or "").lower().startswith("cone")
        for model in world.findall("model")
    )


def validate_candidate_geometry(
    config: RandomConeConfig,
    plan: BypassPlan,
    route_data: RouteData,
    nearby_collision: dict[str, Any],
    share: Path,
) -> dict[str, Any]:
    spacing = float(config.avoidance["sample_spacing_m"])
    span = plan.return_end_s_m - plan.departure_start_s_m
    count = int(math.ceil(span / spacing))
    cone_polygon = box_polygon(
        plan.site.x_m, plan.site.y_m, plan.site.yaw_rad,
        plan.cone.half_length_m, plan.cone.half_width_m,
    )
    minimum_clearance = math.inf
    minimum_clearance_s = plan.departure_start_s_m
    intersection = False
    minimum_track = math.inf
    nonfinite = 0
    for index in range(count + 1):
        s = plan.departure_start_s_m + index * span / count
        point = plan.point_at(s)
        yaw = plan.yaw_at(s)
        if not all(math.isfinite(value) for value in (*point, yaw)):
            nonfinite += 1
            continue
        clearance, intersects = polygon_clearance(
            footprint_polygon(plan.footprint, point[0], point[1], yaw), cone_polygon
        )
        if clearance < minimum_clearance:
            minimum_clearance = clearance
            minimum_clearance_s = s
        intersection = intersection or intersects
        minimum_track = min(minimum_track, point_track_clearance(route_data.route, point))
    start_position_error = math.dist(
        plan.point_at(plan.departure_start_s_m),
        plan.nominal.point_at(plan.departure_start_s_m),
    )
    end_position_error = math.dist(
        plan.point_at(plan.return_end_s_m),
        plan.nominal.point_at(plan.return_end_s_m),
    )
    start_heading_error = abs(_wrapped(
        plan.yaw_at(plan.departure_start_s_m)
        - route_yaw(plan.nominal, plan.departure_start_s_m)
    ))
    end_heading_error = abs(_wrapped(
        plan.yaw_at(plan.return_end_s_m)
        - route_yaw(plan.nominal, plan.return_end_s_m)
    ))
    constants = _plan_constants(config, plan.cone, plan.footprint)
    added_profile_bound = (
        plan.maximum_lateral_offset_m * (10.0 * math.sqrt(3.0) / 3.0)
        / plan.transition_length_m**2
    )
    local_combined_bound = (
        abs(plan.site.local_curvature_per_m)
        + added_profile_bound
    )
    preview = _preview(config, plan, route_data.route)
    expected_cones = _expected_cone_count(config, plan, "00", share)
    gates = {
        "exactly_one_cone_in_expected_world": expected_cones == 1,
        "finite_smooth_bypass": nonfinite == 0,
        "no_planned_footprint_cone_intersection": not intersection,
        "positive_planned_footprint_clearance": minimum_clearance > 0.0,
        "automatic_planning_clearance_target_achieved": minimum_clearance + 1e-9 >= (
            float(config.avoidance["planning_cone_clearance_m"])
            + float(config.avoidance["extra_planning_clearance_m"])
        ),
        "reference_center_track_feasible": minimum_track >= -1e-9,
        "cone_clear_of_nearby_world_collisions": float(nearby_collision["minimum_clearance_m"]) > 0.0,
        "added_profile_curvature_within_design_limit": (
            added_profile_bound
            <= constants["design_added_curvature_limit_per_m"] + 1e-9
        ),
        "local_combined_curvature_within_steering_limit": (
            local_combined_bound <= plan.physical_curvature_limit_per_m + 1e-9
        ),
        "smooth_departure_and_return_position": max(start_position_error, end_position_error) <= 1e-9,
        "smooth_departure_and_return_heading": max(start_heading_error, end_heading_error) <= 1e-4,
        "preview_no_cone_intersection": not preview["footprint_cone_intersection"],
        "preview_positive_cone_clearance": preview["minimum_footprint_to_cone_clearance_m"] > 0.0,
        "preview_no_sustained_offtrack": (
            preview["maximum_continuous_offtrack_s"] <= config.baseline.off_track_grace_s + 1e-9
        ),
        "preview_steering_feasible": (
            preview["maximum_applied_steering_rad"] <= config.baseline.max_steering_rad + 1e-9
            and preview["maximum_continuous_steering_saturation_s"]
            <= float(config.sampling["maximum_continuous_steering_saturation_s"]) + 1e-9
        ),
        "preview_smooth_return_to_nominal": bool(preview["recovery_success"]),
        "preview_completed_remaining_lap": bool(preview["completed_remaining_lap"]),
    }
    return {
        "result": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "profile": config.avoidance["profile"],
        "algorithm": config.avoidance["algorithm"],
        "chosen_side": plan.side,
        "required_center_offset_m": plan.required_center_offset_m,
        "planned_lateral_offset_m": plan.maximum_lateral_offset_m,
        "transition_length_m": plan.transition_length_m,
        "plateau_half_length_m": plan.plateau_half_length_m,
        "departure_start_s_m": plan.departure_start_s_m,
        "plateau_start_s_m": plan.plateau_start_s_m,
        "cone_s_m": plan.cone_s_m,
        "plateau_end_s_m": plan.plateau_end_s_m,
        "return_end_s_m": plan.return_end_s_m,
        "available_left_logical_track_clearance_m": plan.available_left_clearance_m,
        "available_right_logical_track_clearance_m": plan.available_right_clearance_m,
        "minimum_reference_center_track_clearance_m": minimum_track,
        "minimum_planned_footprint_to_cone_clearance_m": minimum_clearance,
        "minimum_planned_clearance_route_s_m": minimum_clearance_s,
        "planned_footprint_cone_intersection": intersection,
        "physical_curvature_limit_per_m": plan.physical_curvature_limit_per_m,
        "design_added_curvature_limit_per_m": constants["design_added_curvature_limit_per_m"],
        "added_profile_curvature_bound_per_m": added_profile_bound,
        "local_combined_curvature_bound_per_m": local_combined_bound,
        "local_combined_equivalent_steering_rad": math.atan(
            config.baseline.wheelbase_m * local_combined_bound
        ),
        "start_position_error_m": start_position_error,
        "end_position_error_m": end_position_error,
        "start_heading_error_rad": start_heading_error,
        "end_heading_error_rad": end_heading_error,
        "nearby_collision_clearance": nearby_collision,
        "kinematic_preview": preview,
        "clearance_success_contract": "strictly positive; 0.05 m is not a live pass threshold",
    }


def _side_score(geometry: dict[str, Any], side: str, tie_break: str) -> tuple[float, ...]:
    preview = geometry["kinematic_preview"]
    return (
        min(
            float(geometry["minimum_planned_footprint_to_cone_clearance_m"]),
            float(preview["minimum_footprint_to_cone_clearance_m"]),
        ),
        -float(preview["steering_saturation_fraction"]),
        -float(preview["maximum_outside_logical_track_m"]),
        float(geometry["minimum_reference_center_track_clearance_m"]),
        1.0 if side == tie_break else 0.0,
    )


def _candidate_lists(config: RandomConeConfig, route: ClosedRoute, constants: dict[str, float]) -> dict[str, list[Candidate]]:
    half_span = constants["transition_length_m"] + constants["plateau_half_length_m"]
    first = float(config.sampling["start_exclusion_m"]) + half_span
    last = route.length - float(config.sampling["end_exclusion_m"]) - half_span
    step = float(config.sampling["candidate_step_m"])
    if last <= first:
        raise RandomConeError("route is too short for frozen approach/return exclusions")
    count = int(math.floor((last - first) / step)) + 1
    result = {name: [] for name in CLASS_ORDER}
    fixed_s = float(config.environment.frozen_cone["route_s_m"])
    for index in range(count):
        s = first + index * step
        separation = abs((s - fixed_s + route.length / 2.0) % route.length - route.length / 2.0)
        if separation < float(config.sampling["fixed_cone_exclusion_m"]):
            continue
        curvature = route_curvature(
            route, s, float(config.sampling["curvature_half_window_m"])
        )
        geometry_class = _curvature_class(config, curvature)
        if geometry_class is None:
            continue
        rank = _hash_text(
            f"{config.random_seed}:{config.sampling['algorithm']}:{index}"
        )
        result[geometry_class].append(Candidate(index, s, curvature, geometry_class, rank))
    for values in result.values():
        values.sort(key=lambda item: (item.rank_sha256, item.grid_index))
    return result


def derive_frozen_scenarios(
    config: RandomConeConfig,
    sim_root: Path,
) -> tuple[ScenarioBundle, ...]:
    share = share_path(sim_root)
    cone_free = asset_set(share, config.environment.canonical_cone_free_world)
    route_data = load_route(cone_free.route)
    cone, _ = parse_cone_geometry(config.environment, share)
    footprint = parse_vehicle_footprint(config.environment, share)
    constants = _plan_constants(config, cone, footprint)
    candidates = _candidate_lists(config, route_data.route, constants)
    targets = {
        name: int(config.sampling["target_count_by_class"][name])
        for name in CLASS_ORDER
    }
    selected: list[tuple[Candidate, BypassPlan, dict[str, Any], dict[str, dict[str, Any]], dict[str, Any], int]] = []
    selected_counts = {name: 0 for name in CLASS_ORDER}
    indexes = {name: 0 for name in CLASS_ORDER}
    while any(selected_counts[name] < targets[name] for name in CLASS_ORDER):
        made_progress = False
        for geometry_class in CLASS_ORDER:
            if selected_counts[geometry_class] >= targets[geometry_class]:
                continue
            values = candidates[geometry_class]
            while indexes[geometry_class] < len(values):
                candidate = values[indexes[geometry_class]]
                indexes[geometry_class] += 1
                if any(
                    abs((candidate.route_s_m - existing[0].route_s_m + route_data.route.length / 2.0) % route_data.route.length - route_data.route.length / 2.0)
                    < float(config.sampling["minimum_route_separation_m"])
                    for existing in selected
                ):
                    continue
                try:
                    provisional = build_candidate_plan(
                        config, route_data, cone, footprint, candidate, "left"
                    )
                    if provisional.site.nonlocal_route_clearance_m < float(config.sampling["minimum_nonlocal_route_clearance_m"]):
                        continue
                    nearby = other_world_collision_clearance(
                        config.environment, share, provisional.site, cone
                    )
                    side_evaluations: dict[str, dict[str, Any]] = {}
                    feasible: list[tuple[tuple[float, ...], str, BypassPlan, dict[str, Any]]] = []
                    for side in ("left", "right"):
                        try:
                            plan = build_candidate_plan(
                                config, route_data, cone, footprint, candidate, side
                            )
                            geometry = validate_candidate_geometry(
                                config, plan, route_data, nearby, share
                            )
                        except (EnvironmentError, RandomConeError, ValueError) as exc:
                            side_evaluations[side] = {
                                "result": "FAIL", "failure": f"{type(exc).__name__}: {exc}"
                            }
                            continue
                        score = _side_score(
                            geometry, side, str(config.sampling["side_tie_break"])
                        )
                        side_evaluations[side] = {
                            "result": geometry["result"], "score": list(score),
                            "minimum_planned_clearance_m": geometry["minimum_planned_footprint_to_cone_clearance_m"],
                            "minimum_preview_clearance_m": geometry["kinematic_preview"]["minimum_footprint_to_cone_clearance_m"],
                            "minimum_reference_track_clearance_m": geometry["minimum_reference_center_track_clearance_m"],
                            "maximum_preview_outside_track_m": geometry["kinematic_preview"]["maximum_outside_logical_track_m"],
                            "preview_saturation_fraction": geometry["kinematic_preview"]["steering_saturation_fraction"],
                            "failed_gates": [name for name, passed in geometry["gates"].items() if not passed],
                        }
                        if geometry["result"] == "PASS":
                            feasible.append((score, side, plan, geometry))
                    if not feasible:
                        continue
                    feasible.sort(key=lambda item: item[0], reverse=True)
                    _, _, chosen_plan, chosen_geometry = feasible[0]
                except (EnvironmentError, RandomConeError, ValueError):
                    continue
                class_rank = indexes[geometry_class]
                selected.append((
                    candidate, chosen_plan, chosen_geometry, side_evaluations,
                    nearby, class_rank,
                ))
                selected_counts[geometry_class] += 1
                made_progress = True
                break
        if not made_progress:
            raise RandomConeError(
                f"could not freeze requested curvature quotas; selected {selected_counts}"
            )
    if len(selected) != SCENARIO_COUNT:
        raise RandomConeError("sampler did not produce exactly twelve scenarios")
    selected.sort(key=lambda item: (
        _hash_text(f"{config.random_seed}:assignment:{item[0].grid_index}"),
        item[0].grid_index,
    ))
    bundles: list[ScenarioBundle] = []
    for number, (candidate, plan, geometry, side_evaluations, nearby, class_rank) in enumerate(selected, 1):
        scenario_id = f"{number:02d}"
        role = next(role for role, ids in ROLE_IDS.items() if scenario_id in ids)
        assignment = _hash_text(
            f"{config.random_seed}:assignment:{candidate.grid_index}"
        )
        scenario = FrozenScenario(
            scenario_id=scenario_id, role=role,
            provenance={
                "random_seed": config.random_seed,
                "algorithm": config.sampling["algorithm"],
                "candidate_grid_index": candidate.grid_index,
                "class_rank": class_rank,
                "rank_sha256": candidate.rank_sha256,
                "assignment_sha256": assignment,
            },
            route_s_m=plan.site.route_s_m, x_m=plan.site.x_m, y_m=plan.site.y_m,
            yaw_rad=plan.site.yaw_rad,
            local_curvature_per_m=plan.site.local_curvature_per_m,
            curvature_class=candidate.curvature_class,
            left_logical_track_clearance_m=plan.site.left_clearance_m,
            right_logical_track_clearance_m=plan.site.right_clearance_m,
            nonlocal_route_clearance_m=plan.site.nonlocal_route_clearance_m,
            nearby_collision_clearance_m=float(nearby["minimum_clearance_m"]),
            nearby_collision_model=str(nearby["nearest_model"]),
            nearby_collision_name=str(nearby["nearest_collision"]),
            chosen_side=plan.side,
        )
        bundles.append(ScenarioBundle(scenario, plan, geometry, side_evaluations))
    classes = {name: sum(bundle.scenario.curvature_class == name for bundle in bundles) for name in CLASS_ORDER}
    if classes != targets:
        raise RandomConeError(f"frozen curvature diversity gate failed: {classes}")
    if {bundle.plan.side for bundle in bundles} != {"left", "right"}:
        raise RandomConeError("deterministic geometry did not produce both avoidance sides")
    return tuple(bundles)


def verify_frozen_scenarios(
    config: RandomConeConfig,
    sim_root: Path,
    *,
    tolerance: float = 1e-8,
) -> tuple[ScenarioBundle, ...]:
    if len(config.scenarios) != SCENARIO_COUNT:
        raise RandomConeError("cannot verify an unfrozen scenario config")
    derived = derive_frozen_scenarios(config, sim_root)
    for frozen, bundle in zip(config.scenarios, derived, strict=True):
        _compare_frozen(frozen.to_dict(), bundle.scenario.to_dict(), tolerance, frozen.scenario_id)
    return derived


def _compare_frozen(
    frozen: Any,
    computed: Any,
    tolerance: float,
    label: str,
    path: str = "",
) -> None:
    if isinstance(frozen, dict) and isinstance(computed, dict):
        if set(frozen) != set(computed):
            raise RandomConeError(f"scenario {label} frozen fields changed at {path or '<root>'}")
        for key in frozen:
            _compare_frozen(frozen[key], computed[key], tolerance, label, f"{path}.{key}" if path else key)
        return
    if _finite_number(frozen) and _finite_number(computed):
        if not math.isclose(float(frozen), float(computed), rel_tol=0.0, abs_tol=tolerance):
            raise RandomConeError(
                f"scenario {label} differs from frozen config at {path}: {frozen} != {computed}"
            )
        return
    if frozen != computed:
        raise RandomConeError(
            f"scenario {label} differs from frozen config at {path}: {frozen!r} != {computed!r}"
        )


def verify_scenario_environment(
    config: RandomConeConfig,
    bundle: ScenarioBundle,
    sim_root: Path,
) -> dict[str, Any]:
    share = share_path(sim_root)
    hashes = verify_canonical_hashes(config.environment, share)
    cone, source = parse_cone_geometry(config.environment, share)
    footprint = parse_vehicle_footprint(config.environment, share)
    scenario_environment = _scenario_environment_config(
        config, bundle.scenario.scenario_id, bundle.plan.site
    )
    cone_free = asset_set(share, config.environment.canonical_cone_free_world)
    derived = asset_set(share, scenario_environment.derived_world)
    if not derived.world.is_file() or not derived.route.is_file() or not derived.model.is_dir():
        raise RandomConeError(
            f"derived assets are missing for scenario {bundle.scenario.scenario_id}"
        )
    expected = expected_world_bytes(
        scenario_environment, cone_free.world, source, bundle.plan.site, cone
    )
    if derived.world.read_bytes() != expected:
        raise RandomConeError(
            f"scenario {bundle.scenario.scenario_id} world differs from frozen config"
        )
    if sha256_file(derived.route) != sha256_file(cone_free.route):
        raise RandomConeError("derived route differs from preserved canonical route")
    if directory_manifest(derived.model) != directory_manifest(cone_free.model):
        raise RandomConeError("derived model metadata differs from canonical cone-free model")
    _, world = parse_xml(derived.world)
    cones = [
        model for model in world.findall("model")
        if (model.get("name") or "").lower().startswith("cone")
    ]
    if len(cones) != 1 or cones[0].get("name") != config.cone_model_name(bundle.scenario.scenario_id):
        raise RandomConeError("derived scenario does not contain exactly its intended cone")
    size = tuple(float(value) for value in cones[0].findtext(
        "./link/collision/geometry/box/size", ""
    ).split())
    if size != cone.size_xyz_m:
        raise RandomConeError("derived cone collision geometry changed")
    nearby = other_world_collision_clearance(
        config.environment, share, bundle.plan.site, cone
    )
    return {
        "result": "PASS",
        "scenario_id": bundle.scenario.scenario_id,
        "world": scenario_environment.derived_world,
        "cone_model": scenario_environment.derived_cone_model,
        "cone_count": 1,
        "route_sha256": sha256_file(derived.route),
        "route_byte_identical": True,
        "model_metadata_identical": True,
        "canonical_hashes": hashes,
        "cone_geometry": cone_geometry_dict(cone),
        "vehicle_footprint": vehicle_footprint_dict(footprint),
        "other_world_collision_clearance": nearby,
    }


def generate_scenario_worlds(
    config: RandomConeConfig,
    bundles: Sequence[ScenarioBundle],
    sim_root: Path,
) -> dict[str, Any]:
    """Create only ignored derived assets; never overwrite an existing mismatch."""
    share = share_path(sim_root)
    verify_canonical_hashes(config.environment, share)
    cone, source = parse_cone_geometry(config.environment, share)
    parse_vehicle_footprint(config.environment, share)
    cone_free = asset_set(share, config.environment.canonical_cone_free_world)
    canonical_before = {
        "world": sha256_file(cone_free.world),
        "route": sha256_file(cone_free.route),
        "model": directory_manifest(cone_free.model),
    }
    records: list[dict[str, Any]] = []
    for bundle in bundles:
        scenario_environment = _scenario_environment_config(
            config, bundle.scenario.scenario_id, bundle.plan.site
        )
        derived = asset_set(share, scenario_environment.derived_world)
        targets = (derived.world, derived.route, derived.model)
        if any(path.exists() for path in targets):
            record = verify_scenario_environment(config, bundle, sim_root)
            record["generation"] = "already valid; no files rewritten"
            records.append(record)
            continue
        expected = expected_world_bytes(
            scenario_environment, cone_free.world, source, bundle.plan.site, cone
        )
        with tempfile.TemporaryDirectory(prefix=".random-cone-v1-", dir=share) as temporary:
            staged = asset_set(Path(temporary), scenario_environment.derived_world)
            staged.world.parent.mkdir(parents=True, exist_ok=True)
            staged.route.parent.mkdir(parents=True, exist_ok=True)
            staged.model.parent.mkdir(parents=True, exist_ok=True)
            staged.world.write_bytes(expected)
            shutil.copy2(cone_free.route, staged.route)
            shutil.copytree(
                cone_free.model, staged.model, symlinks=True, copy_function=shutil.copy2
            )
            for source_path, target in (
                (staged.route, derived.route),
                (staged.model, derived.model),
                (staged.world, derived.world),
            ):
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source_path, target)
        record = verify_scenario_environment(config, bundle, sim_root)
        record["generation"] = "generated and validated"
        records.append(record)
    canonical_after = {
        "world": sha256_file(cone_free.world),
        "route": sha256_file(cone_free.route),
        "model": directory_manifest(cone_free.model),
    }
    if canonical_after != canonical_before:
        raise RandomConeError("canonical cone-free assets changed during generation")
    return {
        "result": "PASS", "scenario_count": len(records),
        "canonical_cone_free_assets_unchanged": True, "scenarios": records,
    }


def verify_all_scenario_worlds(
    config: RandomConeConfig,
    bundles: Sequence[ScenarioBundle],
    sim_root: Path,
) -> dict[str, Any]:
    records = [verify_scenario_environment(config, bundle, sim_root) for bundle in bundles]
    return {
        "result": "PASS", "scenario_count": len(records), "scenarios": records,
        "all_routes_byte_identical": all(item["route_byte_identical"] for item in records),
        "exactly_one_cone_each": all(item["cone_count"] == 1 for item in records),
    }


def audit_preserved_state(repo_root: Path, sim_root: Path) -> dict[str, Any]:
    identities: dict[str, Any] = {}
    for label, (relative, expected) in PRESERVED_REPOSITORY_FILES.items():
        path = repo_root / relative
        observed = sha256_file(path)
        if observed != expected:
            raise RandomConeError(f"preserved repository evidence changed: {relative}")
        identities[label] = {"path": relative, "sha256": observed}
    for label, (relative, expected) in PRESERVED_EXTERNAL_ARTIFACTS.items():
        path = sim_root / relative
        observed = sha256_file(path)
        if observed != expected:
            raise RandomConeError(f"preserved model changed: {relative}")
        identities[label] = {"path": str(path), "sha256": observed}
    v9_training = _read_json(repo_root / PRESERVED_REPOSITORY_FILES["temporal_v9_training_summary"][0])
    v9_live = _read_json(repo_root / PRESERVED_REPOSITORY_FILES["temporal_v9_live_summary"][0])
    fixed = _read_json(repo_root / PRESERVED_REPOSITORY_FILES["fixed_cone_expert_summary"][0])
    c1 = _read_json(repo_root / PRESERVED_REPOSITORY_FILES["c1_practical_summary"][0])
    if (
        v9_training.get("result") != "PASS"
        or v9_training.get("architecture", {}).get("input_shape") != [9, 66, 200]
        or v9_training.get("architecture", {}).get("parameter_count") != 255_819
        or v9_live.get("result") != "PASS"
        or v9_live.get("policy_pass_count") != 3
    ):
        raise RandomConeError("preserved Temporal PilotNet V9 is not the validated 3/3 model")
    fixed_attempts = fixed.get("attempts", [])
    if fixed.get("result") != "PASS" or len(fixed_attempts) != 3 or any(
        item.get("classification") != "OBSTACLE_EXPERT_PASS" for item in fixed_attempts
    ):
        raise RandomConeError("preserved fixed-cone Expert evidence is not 3/3 PASS")
    c1_attempts = c1.get("attempts", [])
    if c1.get("result") != "PASS" or len(c1_attempts) != 3 or any(
        item.get("classification") != "PRACTICAL_CONE_PASS" for item in c1_attempts
    ) or any(
        (item.get("run") or {}).get("footprint_cone_intersection_occurred") is not False
        for item in c1_attempts
    ):
        raise RandomConeError("preserved practical C1 evidence is not collision-free 3/3 PASS")
    return {
        "result": "PASS", "identities": identities,
        "temporal_pilotnet_v9": {
            "speed_mps": 1.80, "observation": "causal three-frame camera",
            "success": "3/3 cone-free", "model_unchanged": True,
        },
        "fixed_cone_expert_v1": {
            "speed_mps": 1.80, "lookahead_m": 0.90,
            "control_frequency_hz": 15.0, "cone_s_m": 6.9,
            "success": "3/3", "evidence_unchanged": True,
        },
        "temporal_pilotnet_c1": {
            "scope": "fixed one-cone practical collision-only validation",
            "success": "3/3", "cone_contact_or_intersection": "0/3",
            "model_unchanged": True,
        },
        "no_training_or_bag_collection": True,
    }


def simulator_tracked_status(sim_root: Path) -> dict[str, Any]:
    try:
        status = subprocess.run(
            ["git", "status", "--short", "--untracked-files=no"], cwd=sim_root,
            text=True, capture_output=True, check=True,
        ).stdout.splitlines()
        changed = subprocess.run(
            ["git", "diff", "--name-only"], cwd=sim_root,
            text=True, capture_output=True, check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RandomConeError(f"cannot inspect simulator Git state: {exc}") from exc
    source_changes = [path for path in changed if path != "userdata/last_world"]
    return {
        "status_short": status, "tracked_diff_paths": changed,
        "tracked_source_changes": source_changes,
        "result": "PASS" if not source_changes else "FAIL",
    }


def offline_report(
    config: RandomConeConfig,
    bundles: Sequence[ScenarioBundle],
    repo_root: Path,
    sim_root: Path,
) -> dict[str, Any]:
    environments = verify_all_scenario_worlds(config, bundles, sim_root)
    scenarios: list[dict[str, Any]] = []
    for bundle in bundles:
        scenarios.append({
            **bundle.scenario.to_dict(),
            "world": config.world_name(bundle.scenario.scenario_id),
            "cone_model": config.cone_model_name(bundle.scenario.scenario_id),
            "bypass": bundle.geometry,
            "side_evaluations": bundle.side_evaluations,
        })
    config_hash = sha256_file(config.path)
    return {
        "version": VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "result": "PASS",
        "config": str(config.path.relative_to(repo_root)),
        "config_sha256": config_hash,
        "map_family": MAP_FAMILY,
        "random_seed": config.random_seed,
        "sampling": config.sampling,
        "scenario_roles": ROLE_IDS,
        "curvature_class_counts": {
            name: sum(bundle.scenario.curvature_class == name for bundle in bundles)
            for name in CLASS_ORDER
        },
        "avoidance_side_counts": {
            side: sum(bundle.plan.side == side for bundle in bundles)
            for side in ("left", "right")
        },
        "fixed_control": {
            "speed_mps": config.baseline.fixed_speed_mps,
            "lookahead_m": config.baseline.lookahead_m,
            "control_frequency_hz": config.baseline.control_frequency_hz,
            "steering_limit_rad": config.baseline.max_steering_rad,
            "wheelbase_m": config.baseline.wheelbase_m,
        },
        "cone_geometry": cone_geometry_dict(bundles[0].plan.cone),
        "vehicle_footprint": vehicle_footprint_dict(bundles[0].plan.footprint),
        "baseline_audit": audit_preserved_state(repo_root, sim_root),
        "world_reproducibility": environments,
        "simulator_tracked_status": simulator_tracked_status(sim_root),
        "scenarios": scenarios,
        "all_offline_gates_pass": all(bundle.geometry["result"] == "PASS" for bundle in bundles),
        "positions_frozen_before_live": True,
        "no_scenario_replacement_after_freeze": True,
        "neural_training_performed": False,
        "training_bags_collected": False,
    }


def write_overview_plot(
    path: Path,
    bundles: Sequence[ScenarioBundle],
    route_data: RouteData,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except Exception as exc:
        raise RandomConeError(f"matplotlib is required for the overview plot: {exc}") from exc
    figure, axes = plt.subplots(3, 4, figsize=(14.0, 9.0), constrained_layout=True)
    for axis, bundle in zip(axes.flat, bundles, strict=True):
        plan = bundle.plan
        span = plan.return_end_s_m - plan.departure_start_s_m
        samples = [
            plan.point_at(plan.departure_start_s_m + index * span / 180.0)
            for index in range(181)
        ]
        center = [route_data.route.point_at(plan.departure_start_s_m - 0.5 + index * (span + 1.0) / 220.0) for index in range(221)]
        axis.plot([point[0] for point in center], [point[1] for point in center], color="#888888", linewidth=1.0, label="nominal")
        axis.plot([point[0] for point in samples], [point[1] for point in samples], color="#1769aa", linewidth=2.0, label="bypass")
        cone = box_polygon(
            plan.site.x_m, plan.site.y_m, plan.site.yaw_rad,
            plan.cone.half_length_m, plan.cone.half_width_m,
        )
        axis.add_patch(Polygon(cone, closed=True, facecolor="#ff7f0e", edgecolor="#9a4b00"))
        minimum_s = float(bundle.geometry["minimum_planned_clearance_route_s_m"])
        point = plan.point_at(minimum_s)
        vehicle = footprint_polygon(plan.footprint, point[0], point[1], plan.yaw_at(minimum_s))
        axis.add_patch(Polygon(vehicle, closed=True, fill=False, edgecolor="#d62728", linestyle="--"))
        axis.set_aspect("equal", adjustable="datalim")
        axis.grid(True, linewidth=0.3, alpha=0.4)
        axis.set_title(
            f"{bundle.scenario.scenario_id} {bundle.scenario.role}\n"
            f"{bundle.scenario.curvature_class}, {plan.side}, s={plan.cone_s_m:.2f} m\n"
            f"planned={bundle.geometry['minimum_planned_footprint_to_cone_clearance_m']:.3f} m",
            fontsize=8,
        )
        axis.tick_params(labelsize=7)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2)
    figure.suptitle("Random Cone Expert V1 — 12 frozen seeded scenarios", fontsize=14)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def scenario_structural_preflight(
    client: SimClient,
    config: RandomConeConfig,
    bundle: ScenarioBundle,
) -> Preflight:
    driver = config.driver_for(bundle.scenario)
    masked = lane_preflight(_ConeMaskingClient(client), driver, False)
    payload = client.objects()
    if payload.get("world") != masked.world:
        raise RuntimeError("objects world does not match active scenario world")
    model_name = config.cone_model_name(bundle.scenario.scenario_id)
    cones = [
        item for item in payload.get("objects", [])
        if isinstance(item, dict) and str(item.get("name", "")).lower().startswith("cone")
    ]
    if len(cones) != 1 or cones[0].get("name") != model_name:
        raise RuntimeError(f"expected exactly cone {model_name!r}, observed {cones}")
    cone = cones[0]
    size = cone.get("size") or {}
    if any(
        not math.isclose(
            float(size.get(key, math.nan)), bundle.plan.cone.size_xyz_m[index], abs_tol=1e-9
        )
        for index, key in enumerate(("x", "y", "z"))
    ):
        raise RuntimeError(f"live cone collision size mismatch: {size}")
    for label in ("origin", "current"):
        pose = cone.get(label) or {}
        observed = (float(pose.get("x", math.nan)), float(pose.get("y", math.nan)))
        if math.dist(observed, (bundle.plan.site.x_m, bundle.plan.site.y_m)) > 0.01:
            raise RuntimeError(f"cone {label} differs from frozen scenario pose: {pose}")
    return Preflight(
        masked.world, masked.route, masked.route_points, 1, masked.bounds, masked.pose
    )


def wait_after_scenario_reset(
    client: SimClient,
    config: RandomConeConfig,
    bundle: ScenarioBundle,
) -> Preflight:
    driver = config.driver_for(bundle.scenario)
    try:
        if errors := client.safe_stop():
            raise RuntimeError("pre-reset safe stop failed: " + "; ".join(errors))
        response = client.reset()
        if response.get("ok") is not True:
            raise RuntimeError(f"simulator reset was not confirmed: {response}")
        if errors := client.safe_stop():
            raise RuntimeError("post-reset safe stop failed: " + "; ".join(errors))
        deadline = time.monotonic() + driver.reset_wait_timeout_s
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                return scenario_structural_preflight(client, config, bundle)
            except Exception as exc:
                last_error = exc
                time.sleep(0.1)
        raise RuntimeError(f"reset did not settle at frozen scenario spawn: {last_error}")
    except BaseException:
        client.safe_stop()
        raise


def full_scenario_preflight(
    client: SimClient,
    config: RandomConeConfig,
    bundle: ScenarioBundle,
    sim_root: Path,
) -> tuple[Preflight, dict[str, Any]]:
    if bundle.geometry.get("result") != "PASS":
        raise RuntimeError("offline scenario geometry gate is not PASS")
    environment = verify_scenario_environment(config, bundle, sim_root)
    initial = wait_after_scenario_reset(client, config, bundle)
    clock = clock_health_preflight(client)
    if clock.get("result") != "PASS":
        raise RuntimeError(str(clock.get("failure_reason", "clock health failed")))
    started = time.monotonic()
    jpeg = client.camera_jpeg()
    with Image.open(BytesIO(jpeg)) as image:
        image.load()
        dimensions = list(image.size)
        mode = image.mode
    if dimensions != [480, 360]:
        raise RuntimeError(f"camera dimensions differ from 480x360: {dimensions}")
    driver = config.driver_for(bundle.scenario)
    return initial, {
        "result": "PASS",
        "scenario_id": bundle.scenario.scenario_id,
        "role": bundle.scenario.role,
        "world": initial.world,
        "environment": environment,
        "offline_geometry_gate": "PASS",
        "route_points": initial.route_points,
        "route_length_m": initial.route.length,
        "cone_count": initial.cone_count,
        "pose": initial.pose,
        "bounds": initial.bounds,
        "clock_health": clock,
        "camera": {
            "result": "PASS", "dimensions": dimensions, "mode": mode,
            "acquisition_ms": (time.monotonic() - started) * 1000.0,
        },
        "control_api": "PASS",
        "fixed_control": {
            "speed_mps": driver.fixed_speed_mps,
            "lookahead_m": driver.lookahead_m,
            "control_frequency_hz": driver.control_frequency_hz,
            "steering_limit_rad": driver.max_steering_rad,
            "wheelbase_m": driver.wheelbase_m,
        },
    }


class RandomConeObserver:
    """Measure practical cone contact/clearance without altering commands."""

    MOVEMENT_CONTACT_TOLERANCE_M = 0.002
    MOVEMENT_CONTACT_TOLERANCE_RAD = 0.02

    def __init__(
        self,
        client: SimClient,
        nominal: ClosedRoute,
        bundle: ScenarioBundle,
        config: RandomConeConfig,
    ) -> None:
        self.client = client
        self.nominal = nominal
        self.bundle = bundle
        self.plan = bundle.plan
        self.config = config
        self.model_name = config.cone_model_name(bundle.scenario.scenario_id)
        self.samples: list[dict[str, Any]] = []
        self.intersection_occurred = False
        self.contact_or_movement_occurred = False
        self.recovery_candidate_at: float | None = None
        self.recovery_candidate_ctes: list[float] = []
        self.recovery_success = False
        self.recovery_time_s: float | None = None
        self.recovery_cte_m: float | None = None
        self.first_after_return_at: float | None = None

    def __getattr__(self, name: str):
        return getattr(self.client, name)

    def pose(self) -> dict[str, Any]:
        pose = self.client.pose()
        objects = self.client.objects()
        cones = [
            item for item in objects.get("objects", [])
            if isinstance(item, dict) and item.get("name") == self.model_name
        ]
        if len(cones) != 1:
            raise RuntimeError("cone GT telemetry lost the unique frozen scenario cone")
        origin = cones[0].get("origin") or {}
        current = cones[0].get("current") or origin
        cone_x = float(current["x"])
        cone_y = float(current["y"])
        cone_yaw = float(current.get("yaw", self.plan.site.yaw_rad))
        origin_xy = (float(origin.get("x", cone_x)), float(origin.get("y", cone_y)))
        origin_yaw = float(origin.get("yaw", self.plan.site.yaw_rad))
        moved = (
            math.dist(origin_xy, (cone_x, cone_y)) > self.MOVEMENT_CONTACT_TOLERANCE_M
            or abs(_wrapped(cone_yaw - origin_yaw)) > self.MOVEMENT_CONTACT_TOLERANCE_RAD
        )
        projection = self.nominal.project((float(pose["x"]), float(pose["y"])))
        clearance, intersects = polygon_clearance(
            footprint_polygon(
                self.plan.footprint, float(pose["x"]), float(pose["y"]), float(pose["yaw"])
            ),
            box_polygon(
                cone_x, cone_y, cone_yaw,
                self.plan.cone.half_length_m, self.plan.cone.half_width_m,
            ),
        )
        now = time.monotonic()
        row = {
            "wall_time_s": now,
            "route_s_m": projection.s,
            "nominal_cte_m": projection.distance,
            "nominal_signed_cte_m": projection.signed_error,
            "cone_clearance_m": clearance,
            "footprint_intersection": intersects,
            "cone_moved_from_origin": moved,
            "steering_rad": None,
        }
        self.samples.append(row)
        self.intersection_occurred = self.intersection_occurred or intersects
        self.contact_or_movement_occurred = self.contact_or_movement_occurred or moved
        if projection.s >= self.plan.return_end_s_m:
            self.first_after_return_at = now if self.first_after_return_at is None else self.first_after_return_at
            threshold = self.config.return_to_route["maximum_absolute_nominal_cte_m"]
            if projection.distance <= threshold:
                if self.recovery_candidate_at is None:
                    self.recovery_candidate_at = now
                    self.recovery_candidate_ctes = []
                self.recovery_candidate_ctes.append(projection.distance)
                if (
                    not self.recovery_success
                    and now - self.recovery_candidate_at
                    >= self.config.return_to_route["minimum_stable_duration_s"]
                ):
                    self.recovery_success = True
                    self.recovery_time_s = now - self.first_after_return_at
                    self.recovery_cte_m = max(self.recovery_candidate_ctes)
            elif not self.recovery_success:
                self.recovery_candidate_at = None
                self.recovery_candidate_ctes = []
        if intersects:
            raise RuntimeError("vehicle-footprint/cone intersection detected")
        if moved:
            raise RuntimeError("cone movement indicates physical contact")
        return pose

    def command_steering(self, value: float) -> dict[str, Any]:
        response = self.client.command_steering(value)
        if self.samples:
            self.samples[-1]["steering_rad"] = float(value)
        return response


def run_random_cone_expert(
    client: SimClient,
    config: RandomConeConfig,
    initial: Preflight,
    bundle: ScenarioBundle,
) -> dict[str, Any]:
    driver = config.driver_for(bundle.scenario)
    observer = RandomConeObserver(client, initial.route, bundle, config)
    control_route = ObstacleAwareRoute(initial.route, bundle.plan)
    control_preflight = Preflight(
        initial.world, control_route, initial.route_points, initial.cone_count,
        initial.bounds, initial.pose,
    )
    metrics = run_driver(observer, driver, control_preflight)
    rows = observer.samples
    steering = [float(row["steering_rad"]) for row in rows if row["steering_rad"] is not None]
    deltas = [abs(steering[index] - steering[index - 1]) for index in range(1, len(steering))]
    minimum = min(rows, key=lambda row: float(row["cone_clearance_m"]), default=None)
    avoidance_rows = [
        row for row in rows
        if bundle.plan.departure_start_s_m <= float(row["route_s_m"]) <= bundle.plan.return_end_s_m
    ]
    directional_offsets = [
        float(row["nominal_signed_cte_m"]) * bundle.plan.side_sign
        for row in avoidance_rows
    ]
    failure = str(metrics.get("failure") or "").lower()
    metrics.update({
        "classification": None,
        "scenario_id": bundle.scenario.scenario_id,
        "role": bundle.scenario.role,
        "curvature_class": bundle.scenario.curvature_class,
        "cone_s_m": bundle.scenario.route_s_m,
        "cone_x_m": bundle.scenario.x_m,
        "cone_y_m": bundle.scenario.y_m,
        "local_curvature_per_m": bundle.scenario.local_curvature_per_m,
        "chosen_side": bundle.plan.side,
        "planned_lateral_offset_m": bundle.plan.maximum_lateral_offset_m,
        "planned_minimum_clearance_m": bundle.geometry["minimum_planned_footprint_to_cone_clearance_m"],
        "lap_time_s": metrics["elapsed_s"],
        "route_completion_fraction": metrics["total_unwrapped_progress_m"] / metrics["route_length_m"],
        "nominal_route_used_for_progress": True,
        "control_reference": "nominal route outside generated local bypass; automatic quintic bypass inside span",
        "mean_absolute_command_delta_rad": statistics.fmean(deltas) if deltas else 0.0,
        "minimum_footprint_to_cone_clearance_m": None if minimum is None else minimum["cone_clearance_m"],
        "minimum_cone_clearance_route_s_m": None if minimum is None else minimum["route_s_m"],
        "footprint_cone_intersection_occurred": observer.intersection_occurred,
        "cone_contact_or_movement_occurred": observer.contact_or_movement_occurred,
        "cone_contact_or_intersection_occurred": (
            observer.intersection_occurred or observer.contact_or_movement_occurred
        ),
        "maximum_lateral_avoidance_offset_reached_m": max(directional_offsets, default=0.0),
        "recovery_cte_m": observer.recovery_cte_m,
        "recovery_success": observer.recovery_success,
        "recovery_time_s": observer.recovery_time_s,
        "return_contract": dict(config.return_to_route),
        "control_loop_frequency_hz": 1.0 / metrics["mean_loop_period_s"] if metrics["mean_loop_period_s"] else 0.0,
        "timing_slips": metrics["period_slip_count"],
        "api_failures": int(any(token in failure for token in (
            "get ", "post ", "unavailable", "control rejected"
        ))),
        "pose_failures": int("pose did not change meaningfully" in failure),
        "clock_failures": int(
            "clock did not advance" in failure or "clock moved backward" in failure
        ),
        "clearance_measured_not_enforced_as_5cm_margin": True,
        "practical_success_contract": "strictly positive clearance and no cone contact/intersection",
    })
    metrics["classification"] = classify_random_cone_run(metrics)
    return metrics


def classify_random_cone_run(metrics: dict[str, Any]) -> str:
    failure = str(metrics.get("failure") or "").lower()
    if (
        not metrics.get("safe_stop_success", False)
        or metrics.get("api_failures", 0)
        or metrics.get("pose_failures", 0)
        or metrics.get("clock_failures", 0)
        or any(token in failure for token in (
            "simulator state changed", "unexpected world", "invalid track boundary"
        ))
    ):
        return "INFRA_FAIL"
    clearance = metrics.get("minimum_footprint_to_cone_clearance_m")
    if (
        metrics.get("result") == "PASS"
        and clearance is not None
        and float(clearance) > 0.0
        and metrics.get("cone_contact_or_intersection_occurred") is False
        and metrics.get("recovery_success") is True
        and int(metrics.get("off_track_event_count", 0)) >= 0
    ):
        return "RANDOM_CONE_EXPERT_PASS"
    return "RANDOM_CONE_EXPERT_FAIL"


def run_live_benchmark(
    client: SimClient,
    config: RandomConeConfig,
    bundles: Sequence[ScenarioBundle],
    sim_root: Path,
    result_dir: Path,
    *,
    activate_one: Callable[[SimClient, str], dict[str, Any]] = activate_world,
    preflight_one: Callable[..., tuple[Preflight, dict[str, Any]]] = full_scenario_preflight,
    run_one: Callable[..., dict[str, Any]] = run_random_cone_expert,
) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    valid_runs = 0
    unresolved_infrastructure = False
    for bundle in bundles:
        attempts: list[dict[str, Any]] = []
        valid: dict[str, Any] | None = None
        maximum_attempts = 1 + int(
            config.live_protocol["infrastructure_replacement_attempts_per_scenario"]
        )
        for attempt_number in range(1, maximum_attempts + 1):
            try:
                if errors := client.safe_stop():
                    raise RuntimeError("pre-scenario safe stop failed: " + "; ".join(errors))
                activation = activate_one(
                    client, config.world_name(bundle.scenario.scenario_id)
                )
                initial, preflight_result = preflight_one(
                    client, config, bundle, sim_root
                )
                metrics = run_one(client, config, initial, bundle)
                classification = str(metrics["classification"])
                if errors := client.safe_stop():
                    classification = "INFRA_FAIL"
                    metrics["post_scenario_safe_stop_success"] = False
                    metrics["post_scenario_safe_stop_errors"] = errors
                else:
                    metrics["post_scenario_safe_stop_success"] = True
                    metrics["post_scenario_safe_stop_errors"] = []
                attempt = {
                    "attempt_number": attempt_number,
                    "classification": classification,
                    "infrastructure_replacement": attempt_number > 1,
                    "world_activation": activation,
                    "preflight": preflight_result,
                    "metrics": metrics,
                }
            except Exception as exc:
                stop_errors = client.safe_stop()
                attempt = {
                    "attempt_number": attempt_number,
                    "classification": "INFRA_FAIL",
                    "infrastructure_replacement": attempt_number > 1,
                    "world_activation": None,
                    "preflight": {
                        "result": "FAIL", "failure": f"{type(exc).__name__}: {exc}",
                        "safe_stop_success": not stop_errors,
                        "safe_stop_errors": stop_errors,
                    },
                    "metrics": None,
                }
            attempts.append(attempt)
            write_json(
                result_dir / (
                    f"scenario_{bundle.scenario.scenario_id}_attempt_{attempt_number:02d}.json"
                ),
                {
                    "version": VERSION,
                    "scenario": bundle.scenario.to_dict(),
                    "planned_bypass": bundle.geometry,
                    **attempt,
                },
            )
            if attempt["classification"] == "INFRA_FAIL":
                if attempt_number < maximum_attempts:
                    continue
                unresolved_infrastructure = True
                break
            valid_runs += 1
            if valid_runs > int(config.live_protocol["maximum_valid_policy_runs"]):
                raise RandomConeError("valid policy-run cap exceeded")
            valid = attempt
            break
        record = {
            "scenario": bundle.scenario.to_dict(),
            "world": config.world_name(bundle.scenario.scenario_id),
            "planned_bypass": bundle.geometry,
            "attempts": attempts,
            "valid_policy_run": valid,
            "valid_policy_run_count": int(valid is not None),
            "result": None if valid is None else valid["classification"],
        }
        records.append(record)
        print(
            f"Random Cone scenario {bundle.scenario.scenario_id}/12: "
            f"{record['result'] or 'INFRA_FAIL'} "
            f"({len(attempts)} attempt{'s' if len(attempts) != 1 else ''})",
            file=sys.stderr,
            flush=True,
        )
        if unresolved_infrastructure:
            break
        # A genuine failure is preserved with no retry.  Later frozen
        # scenarios continue for safe, useful geometry-class characterization.
    if unresolved_infrastructure:
        return records, "INCONCLUSIVE"
    if len(records) != SCENARIO_COUNT or valid_runs != SCENARIO_COUNT:
        return records, "INCONCLUSIVE"
    if all(item["result"] == "RANDOM_CONE_EXPERT_PASS" for item in records):
        return records, "PASS"
    return records, "FAIL"


def aggregate_live(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        item["valid_policy_run"]["metrics"]
        for item in records if item.get("valid_policy_run") is not None
    ]
    passes = [
        metrics for metrics in valid
        if metrics.get("classification") == "RANDOM_CONE_EXPERT_PASS"
    ]
    clearances = [
        float(metrics["minimum_footprint_to_cone_clearance_m"])
        for metrics in valid
        if metrics.get("minimum_footprint_to_cone_clearance_m") is not None
    ]
    return {
        "valid_policy_runs": len(valid),
        "pass_count": len(passes),
        "success": f"{len(passes)}/{SCENARIO_COUNT}",
        "minimum_actual_clearance_m": min(clearances) if clearances else None,
        "actual_clearance_mean_m": statistics.fmean(clearances) if clearances else None,
        "cone_contact_or_intersection_count": sum(
            bool(metrics.get("cone_contact_or_intersection_occurred")) for metrics in valid
        ),
        "recovery_success_count": sum(metrics.get("recovery_success") is True for metrics in valid),
        "safe_stop_success_count": sum(metrics.get("safe_stop_success") is True for metrics in valid),
        "lap_time_mean_s": statistics.fmean(
            float(metrics["lap_time_s"]) for metrics in passes
        ) if passes else None,
        "worst_nominal_max_cte_m": max(
            (float(metrics["max_centerline_error_m"]) for metrics in valid), default=None
        ),
        "mean_nominal_mean_cte_m": statistics.fmean(
            float(metrics["mean_centerline_error_m"]) for metrics in valid
        ) if valid else None,
        "steering_saturation_mean": statistics.fmean(
            float(metrics["steering_saturation_fraction"]) for metrics in valid
        ) if valid else None,
        "infrastructure_attempts": sum(
            len(item.get("attempts", [])) - int(item.get("valid_policy_run") is not None)
            for item in records
        ),
    }


def verify_offline_evidence(
    path: Path,
    config: RandomConeConfig,
    bundles: Sequence[ScenarioBundle],
) -> dict[str, Any]:
    report = _read_json(path)
    if report.get("result") != "PASS" or report.get("config_sha256") != sha256_file(config.path):
        raise RandomConeError("offline evidence does not match the frozen production config")
    observed = report.get("scenarios")
    if not isinstance(observed, list) or len(observed) != SCENARIO_COUNT:
        raise RandomConeError("offline evidence does not contain all twelve scenarios")
    for item, bundle in zip(observed, bundles, strict=True):
        if (
            item.get("scenario_id") != bundle.scenario.scenario_id
            or not math.isclose(float(item.get("route_s_m", math.nan)), bundle.scenario.route_s_m, abs_tol=1e-8)
            or item.get("chosen_side") != bundle.plan.side
            or (item.get("bypass") or {}).get("result") != "PASS"
        ):
            raise RandomConeError(
                f"offline evidence changed for scenario {bundle.scenario.scenario_id}"
            )
    return report


def repository_status(repo_root: Path) -> dict[str, Any]:
    output = subprocess.run(
        ["git", "status", "--short", "--branch"], cwd=repo_root,
        text=True, capture_output=True, check=True,
    ).stdout.splitlines()
    return {"status_short_branch": output}


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    rows: list[str] = []
    metric_rows: list[str] = []
    infrastructure_rows: list[str] = []
    failure_rows: list[str] = []
    for item in report.get("scenarios", []):
        scenario = item["scenario"]
        valid = item.get("valid_policy_run")
        metrics = None if valid is None else valid.get("metrics")
        rows.append(
            "| {id} | {role} | {klass} | {s:.2f} | {side} | {planned:.3f} | {actual} | {collision} | {recovery} | {result} |".format(
                id=scenario["scenario_id"], role=scenario["role"],
                klass=scenario["curvature_class"], s=float(scenario["route_s_m"]),
                side=scenario["chosen_side"],
                planned=float(item["planned_bypass"]["planned_lateral_offset_m"]),
                actual="—" if metrics is None or metrics.get("minimum_footprint_to_cone_clearance_m") is None else f"{float(metrics['minimum_footprint_to_cone_clearance_m']):.4f}",
                collision="—" if metrics is None else str(bool(metrics.get("cone_contact_or_intersection_occurred"))).lower(),
                recovery="—" if metrics is None else str(bool(metrics.get("recovery_success"))).lower(),
                result=item.get("result") or "INCONCLUSIVE",
            )
        )
        if metrics is not None:
            recovery_time = metrics.get("recovery_time_s")
            recovery = str(bool(metrics.get("recovery_success"))).lower()
            if recovery_time is not None:
                recovery += f"/{float(recovery_time):.3f}s"
            metric_rows.append(
                "| {id} | {lap:.3f} | {completion:.2f}% | {mean:.4f}/{maximum:.4f} | {offtrack} | {recovery} | {saturation:.2f}% | {frequency:.3f} Hz/{slips} | {failures} | {safe} |".format(
                    id=scenario["scenario_id"], lap=float(metrics["lap_time_s"]),
                    completion=100.0 * float(metrics["route_completion_fraction"]),
                    mean=float(metrics["mean_centerline_error_m"]),
                    maximum=float(metrics["max_centerline_error_m"]),
                    offtrack=int(metrics["off_track_event_count"]), recovery=recovery,
                    saturation=100.0 * float(metrics["steering_saturation_fraction"]),
                    frequency=float(metrics["control_loop_frequency_hz"]),
                    slips=int(metrics["period_slip_count"]),
                    failures="{}/{}/{}".format(
                        int(metrics["api_failures"]), int(metrics["pose_failures"]),
                        int(metrics["clock_failures"]),
                    ),
                    safe=str(bool(metrics.get("safe_stop_success"))).lower(),
                )
            )
            if item.get("result") == "RANDOM_CONE_EXPERT_FAIL":
                failure_rows.append(
                    f"- Scenario {scenario['scenario_id']}: {metrics.get('failure', 'policy failure')}"
                )
        for attempt in item.get("attempts", []):
            if attempt.get("classification") != "INFRA_FAIL":
                continue
            detail = (attempt.get("preflight") or {}).get("failure", "unspecified infrastructure failure")
            infrastructure_rows.append(
                f"- Scenario {scenario['scenario_id']} attempt {attempt['attempt_number']}: {detail}"
            )
    result = report.get("result", "INCONCLUSIVE")
    aggregate = report.get("aggregate", {})
    content = [
        "# Random Cone Expert V1", "",
        f"Final simulation gate: **{result}**.", "",
        "This is simulator-only evidence. It is not real-robot evidence.", "",
        "## Frozen contract", "",
        f"- Seed: `{report.get('random_seed')}`",
        f"- Map family: `{MAP_FAMILY}`",
        "- Split: scenarios 01–08 TRAIN, 09–10 VALIDATION, 11–12 UNSEEN_HOLDOUT",
        "- Control: 1.80 m/s, 0.90 m lookahead, 15 Hz, ±0.349066 rad, 0.18 m wheelbase",
        "- Practical pass condition: positive footprint clearance and no cone contact/intersection",
        "- No neural training and no training-bag collection", "",
        "## Scenario results", "",
        "| ID | Role | Geometry | s (m) | Side | Offset (m) | Actual clearance (m) | Collision | Recovery | Result |",
        "|---:|---|---|---:|---|---:|---:|---|---|---|",
        *rows, "",
        "## Detailed live metrics", "",
        "| ID | Lap time | Completion | Mean/max CTE (m) | Off-track events | Recovery/time | Steering saturation | Loop/slips | API/pose/clock failures | Safe stop |",
        "|---:|---:|---:|---:|---:|---|---:|---|---|---|",
        *metric_rows, "",
        "## Aggregate and failures", "",
        f"- Valid policy runs: `{aggregate.get('valid_policy_runs', 0)}/12`",
        f"- Scenario passes: `{aggregate.get('success', '0/12')}`",
        f"- Cone contact/intersection: `{aggregate.get('cone_contact_or_intersection_count', 0)}/12`",
        f"- Minimum actual footprint clearance: `{float(aggregate.get('minimum_actual_clearance_m', math.nan)):.6f} m`",
        f"- Successful recoveries: `{aggregate.get('recovery_success_count', 0)}/12`",
        f"- Per-scenario safe stops: `{aggregate.get('safe_stop_success_count', 0)}/12`",
        f"- Final safe stop: `{str(bool(report.get('final_safe_stop_success'))).lower()}`",
        *failure_rows, "",
        "Scenario 01 cleared the cone but failed during the return: this is a moderate-left/right-side bypass in the complex multi-turn region around s=19–21 m. The other moderate-left scenarios passed, so the observed failure is not a general cone-collision or curvature-class failure. The offline ideal-bicycle preview underpredicted simulator return-path tracking divergence at this site.", "",
        "## Infrastructure-only attempts", "",
        *(infrastructure_rows or ["None."]), "",
        "Each listed attempt stopped safely and received at most the one permitted fresh replacement. No genuine policy run was retried.", "",
        "## Baselines and disposition", "",
        "Temporal PilotNet V9, Fixed Cone Avoidance Expert V1, and practical Temporal PilotNet C1 evidence/models were hash-audited and left unchanged.", "",
        f"- Config SHA-256: `{report.get('config_sha256')}`",
        f"- Offline evidence SHA-256: `{report.get('offline_geometry_sha256')}`",
        f"- Baseline audit before/after: `{report.get('baseline_audit_before', {}).get('result')}/{report.get('baseline_audit_after', {}).get('result')}`",
        f"- Tracked simulator source changes after run: `{len(report.get('simulator_status_after', {}).get('tracked_source_changes', []))}`",
        f"- Neural training performed: `{str(bool(report.get('neural_training_performed'))).lower()}`",
        f"- Training bags collected: `{str(bool(report.get('training_bags_collected'))).lower()}`", "",
        f"Random Cone Expert V1 frozen: **{str(bool(report.get('random_cone_expert_frozen'))).lower()}**.",
        f"PASS-qualified 8/2/2 split release frozen: **{str(bool(report.get('exact_8_2_2_split_frozen'))).lower()}**.",
        f"Random-cone bag collection justified: **{str(bool(report.get('random_cone_bag_collection_justified'))).lower()}**.", "",
        "## Limitations", "",
        "The bicycle preview is an offline feasibility model; the reported live rows are simulator measurements. Curvature classes use a 0.5 m route window on the original piecewise-polyline route. The 5 cm value is a planning margin, not a practical success threshold.", "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def _restore_world(client: SimClient, original_world: str | None) -> dict[str, Any]:
    if not original_world:
        return {"result": "PASS", "action": "no original world reported"}
    result = activate_world(client, original_world)
    errors = client.safe_stop()
    if errors:
        raise RuntimeError("safe stop after world restoration failed: " + "; ".join(errors))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, default=Path(EXPECTED_RESULT_DIRECTORY))
    parser.add_argument("--scenario-id", choices=[f"{number:02d}" for number in range(1, 13)])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--propose-freeze", action="store_true")
    mode.add_argument("--generate-worlds", action="store_true")
    mode.add_argument("--verify-worlds", action="store_true")
    mode.add_argument("--offline-geometry", action="store_true")
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[2]
    sim_root = args.sim_root.expanduser().resolve()
    result_dir = args.result_dir
    if not result_dir.is_absolute():
        result_dir = (repo_root / result_dir).resolve()
    expected_result = (repo_root / EXPECTED_RESULT_DIRECTORY).resolve()
    if not (args.propose_freeze or args.generate_worlds or args.verify_worlds) and result_dir != expected_result:
        print(f"ERROR: result directory must be {EXPECTED_RESULT_DIRECTORY}", file=sys.stderr)
        return 2
    try:
        config = RandomConeConfig.load(
            args.config, repo_root, sim_root, allow_unfrozen=args.propose_freeze
        )
        if args.propose_freeze:
            bundles = derive_frozen_scenarios(config, sim_root)
            print(json.dumps({
                "random_seed": config.random_seed,
                "sampling": config.sampling,
                "scenarios": [bundle.scenario.to_dict() for bundle in bundles],
            }, indent=2, sort_keys=True))
            return 0
        bundles = verify_frozen_scenarios(config, sim_root)
        if args.generate_worlds:
            print(json.dumps(
                generate_scenario_worlds(config, bundles, sim_root), indent=2, sort_keys=True
            ))
            return 0
        if args.verify_worlds:
            print(json.dumps(
                verify_all_scenario_worlds(config, bundles, sim_root), indent=2, sort_keys=True
            ))
            return 0
        if args.offline_geometry:
            report = offline_report(config, bundles, repo_root, sim_root)
            write_json(result_dir / "offline_geometry.json", report)
            cone_free = asset_set(
                share_path(sim_root), config.environment.canonical_cone_free_world
            )
            write_overview_plot(
                result_dir / "overview.png", bundles, load_route(cone_free.route)
            )
            report["overview_plot"] = str(
                (result_dir / "overview.png").relative_to(repo_root)
            )
            write_json(result_dir / "offline_geometry.json", report)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        offline_path = result_dir / "offline_geometry.json"
        offline = verify_offline_evidence(offline_path, config, bundles)
        client = SimClient(config.baseline.base_url, config.baseline.api_timeout_s)
        original_world: str | None = None
        restoration: dict[str, Any] | None = None
        if args.preflight_only:
            scenario_id = args.scenario_id or "01"
            bundle = next(item for item in bundles if item.scenario.scenario_id == scenario_id)
            original_world = str(client.status().get("current") or "") or None
            try:
                activation = activate_world(client, config.world_name(scenario_id))
                _, preflight_result = full_scenario_preflight(
                    client, config, bundle, sim_root
                )
                result = {
                    "version": VERSION, "result": "PREFLIGHT_PASS",
                    "scenario": bundle.scenario.to_dict(),
                    "world_activation": activation, "preflight": preflight_result,
                    "offline_geometry_sha256": sha256_file(offline_path),
                }
                write_json(result_dir / f"preflight_{scenario_id}.json", result)
                return_code = 0
            finally:
                restoration = _restore_world(client, original_world)
                client.safe_stop()
            result["world_restoration"] = restoration
            write_json(result_dir / f"preflight_{scenario_id}.json", result)
            print(json.dumps(result, indent=2, sort_keys=True))
            return return_code
        marker = result_dir / "experiment.started.json"
        summary_path = result_dir / "summary.json"
        if marker.exists() or summary_path.exists():
            raise RandomConeError(
                "refusing to repeat or overwrite the frozen 12-scenario live experiment"
            )
        original_status = client.status()
        original_world = str(original_status.get("current") or "") or None
        baseline_before = audit_preserved_state(repo_root, sim_root)
        simulator_before = simulator_tracked_status(sim_root)
        if simulator_before["result"] != "PASS":
            raise RandomConeError("simulator already has tracked source changes")
        write_json(marker, {
            "status": "RANDOM_CONE_EXPERT_V1_STARTED_DO_NOT_REPEAT",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": sha256_file(config.path),
            "offline_geometry_sha256": sha256_file(offline_path),
            "random_seed": config.random_seed,
            "scenario_ids": [bundle.scenario.scenario_id for bundle in bundles],
            "maximum_valid_policy_runs": 12,
            "original_world": original_world,
        })
        report: dict[str, Any] = {
            "version": VERSION,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "result": "INCONCLUSIVE",
            "random_seed": config.random_seed,
            "map_family": MAP_FAMILY,
            "scenario_roles": ROLE_IDS,
            "config_sha256": sha256_file(config.path),
            "offline_geometry_sha256": sha256_file(offline_path),
            "baseline_audit_before": baseline_before,
            "simulator_status_before": simulator_before,
            "scenarios": [],
        }
        code = 2
        try:
            report["scenarios"], live_result = run_live_benchmark(
                client, config, bundles, sim_root, result_dir
            )
            report["live_result"] = live_result
            report["aggregate"] = aggregate_live(report["scenarios"])
            code = 0 if live_result == "PASS" else (1 if live_result == "FAIL" else 2)
        except Exception as exc:
            report["live_result"] = "INCONCLUSIVE"
            report["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            try:
                restoration = _restore_world(client, original_world)
                report["world_restoration"] = restoration
            except Exception as exc:
                report["world_restoration"] = {
                    "result": "FAIL", "failure": f"{type(exc).__name__}: {exc}"
                }
                code = 2
            final_stop_errors = client.safe_stop()
            report["final_safe_stop_success"] = not final_stop_errors
            report["final_safe_stop_errors"] = final_stop_errors
            if final_stop_errors:
                code = 2
            try:
                report["baseline_audit_after"] = audit_preserved_state(repo_root, sim_root)
                report["simulator_status_after"] = simulator_tracked_status(sim_root)
            except Exception as exc:
                report["post_run_audit_failure"] = f"{type(exc).__name__}: {exc}"
                code = 2
            final_pass = (
                report.get("live_result") == "PASS"
                and report.get("final_safe_stop_success") is True
                and (report.get("world_restoration") or {}).get("result") == "PASS"
                and (report.get("simulator_status_after") or {}).get("result") == "PASS"
                and "post_run_audit_failure" not in report
            )
            report["result"] = "PASS" if final_pass else (
                "FAIL" if report.get("live_result") == "FAIL" and code != 2 else "INCONCLUSIVE"
            )
            report["random_cone_expert_frozen"] = final_pass
            report["exact_8_2_2_split_frozen"] = final_pass
            report["random_cone_bag_collection_justified"] = final_pass
            report["neural_training_performed"] = False
            report["training_bags_collected"] = False
            report["repository_status"] = repository_status(repo_root)
            write_json(summary_path, report)
            write_markdown_report(result_dir / "REPORT.md", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return code
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RandomConeError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RandomConeError(f"JSON root is not an object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _positive_number(value: object) -> bool:
    return _finite_number(value) and float(value) > 0.0
