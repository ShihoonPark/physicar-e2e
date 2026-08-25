from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import pytest

from physicar_e2e.cone_avoidance_environment import (
    EnvironmentConfig,
    asset_set,
    cone_geometry_dict,
    generate_environment,
    load_route,
    parse_cone_geometry,
    parse_vehicle_footprint,
    route_curvature,
    select_cone_site,
    sha256_file,
    share_path,
    verify_derived_environment,
)


REPO = Path(__file__).resolve().parents[1]
SIM_ROOT = Path("/home/a/physicar-ai-sim-docker")
CONFIG = REPO / "configs" / "cone_avoidance_environment_v1.json"


def _require_assets() -> tuple[EnvironmentConfig, Path]:
    config = EnvironmentConfig.load(CONFIG)
    share = share_path(SIM_ROOT)
    if not asset_set(share, config.canonical_cone_world).world.is_file():
        pytest.skip("read-only simulator asset checkout is unavailable")
    return config, share


def test_production_environment_contract_is_frozen() -> None:
    config = EnvironmentConfig.load(CONFIG)
    assert config.canonical_cone_free_world.endswith("_e2e_lane_follow_v1")
    assert config.derived_world.endswith("_e2e_cone_avoidance_v1")
    assert config.canonical_cone_free_world != config.derived_world
    assert config.required_cone_clearance_m == 0.05
    assert config.extra_cone_clearance_m == 0.005
    assert config.footprint_steering_limit_rad == 0.349066


def test_real_cone_model_provenance_and_collision_geometry() -> None:
    config, share = _require_assets()
    geometry, _ = parse_cone_geometry(config, share)
    payload = cone_geometry_dict(geometry)
    assert geometry.source_world == config.canonical_cone_world
    assert geometry.source_model == "cone2"
    assert geometry.collision_name == "cone"
    assert geometry.size_xyz_m == (0.18, 0.18, 0.38)
    assert payload["collision_2d_half_extents_m"] == [0.09, 0.09]
    assert geometry.visual_uri.endswith("/cone2.dae")


def test_vehicle_footprint_is_parsed_from_all_real_collision_groups() -> None:
    config, share = _require_assets()
    footprint = parse_vehicle_footprint(config, share)
    assert footprint.source_sdf.endswith("/models/physicar/model.sdf")
    assert footprint.length_m == pytest.approx(0.27)
    assert footprint.width_m == pytest.approx(0.21854076122953137)
    assert len(footprint.collision_elements) == 6
    assert {"front_left_wheel_link_collision", "front_right_wheel_link_collision"} <= set(footprint.collision_elements)
    assert footprint.y_max_m > 0.0975  # includes front-wheel articulation, not neutral geometry only


def test_cone_location_is_deterministic_on_route_and_low_curvature() -> None:
    config, share = _require_assets()
    route_data = load_route(asset_set(share, config.canonical_cone_free_world).route)
    first = select_cone_site(config, route_data)
    second = select_cone_site(config, route_data)
    assert first == second
    assert first.route_s_m == pytest.approx(6.9)
    assert math.dist((first.x_m, first.y_m), route_data.route.point_at(first.route_s_m)) < 1e-10
    assert abs(first.local_curvature_per_m) < config.selector["maximum_abs_curvature_per_m"]
    assert abs(route_curvature(route_data.route, first.route_s_m, 0.5)) < 1e-10
    assert first.straight_run_length_m >= config.selector["minimum_straight_run_m"]
    assert first.nonlocal_route_clearance_m >= config.selector["minimum_nonlocal_clearance_m"]


def test_avoidance_side_uses_greater_available_clearance() -> None:
    config, share = _require_assets()
    site = select_cone_site(config, load_route(asset_set(share, config.canonical_cone_free_world).route))
    assert site.right_clearance_m > site.left_clearance_m
    assert site.chosen_side == "right"


def test_generation_leaves_canonical_cone_free_world_untouched_and_has_one_cone(tmp_path: Path) -> None:
    config, source_share = _require_assets()
    target_root = tmp_path / "sim"
    target_share = share_path(target_root)
    for world in (config.canonical_cone_world, config.canonical_cone_free_world):
        source = asset_set(source_share, world)
        target = asset_set(target_share, world)
        target.world.parent.mkdir(parents=True, exist_ok=True)
        target.route.parent.mkdir(parents=True, exist_ok=True)
        target.model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source.world, target.world)
        shutil.copy2(source.route, target.route)
        shutil.copytree(source.model, target.model)
    source_mesh = source_share / "meshes" / config.canonical_cone_world / "cone2.dae"
    target_mesh = target_share / "meshes" / config.canonical_cone_world / "cone2.dae"
    target_mesh.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_mesh, target_mesh)
    vehicle_source = source_share / "models" / "physicar" / "model.sdf"
    vehicle_target = target_share / "models" / "physicar" / "model.sdf"
    vehicle_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(vehicle_source, vehicle_target)
    cone_free = asset_set(target_share, config.canonical_cone_free_world)
    before = (cone_free.world.read_bytes(), cone_free.route.read_bytes())
    result = generate_environment(config, target_share)
    after = (cone_free.world.read_bytes(), cone_free.route.read_bytes())
    assert result["result"] == "PASS"
    assert before == after
    derived = asset_set(target_share, config.derived_world)
    world = ET.parse(derived.world).getroot().find("world")
    assert world is not None
    cones = [m for m in world.findall("model") if (m.get("name") or "").startswith("cone")]
    assert [model.get("name") for model in cones] == [config.derived_cone_model]
    assert sha256_file(derived.route) == sha256_file(cone_free.route)
    verification = verify_derived_environment(config, target_share)
    assert verification["cone_count"] == 1
    assert verification["other_world_collision_clearance"]["result"] == "PASS"
    assert verification["other_world_collision_clearance"]["minimum_clearance_m"] > 0.0
