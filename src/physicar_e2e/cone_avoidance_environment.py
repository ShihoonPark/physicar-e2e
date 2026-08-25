"""Reproducible one-cone environment and simulator-geometry inspection.

This module is deliberately specific to Cone Avoidance Expert V1.  It derives
one ignored custom simulator world from the preserved cone-free world and
copies one real cone model block from the original custom world.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Sequence
import xml.etree.ElementTree as ET

from .route_geometry import ClosedRoute


VERSION = "cone_avoidance_environment_v1"
Point = tuple[float, float]


class EnvironmentError(RuntimeError):
    """A controlled geometry, provenance, or asset-integrity failure."""


@dataclass(frozen=True)
class EnvironmentConfig:
    payload: dict[str, Any]
    canonical_cone_world: str
    canonical_cone_free_world: str
    derived_world: str
    source_cone_model: str
    derived_cone_model: str
    canonical_sha256: dict[str, str]
    selector: dict[str, Any]
    frozen_cone: dict[str, float]
    required_cone_clearance_m: float
    extra_cone_clearance_m: float
    footprint_steering_limit_rad: float

    @classmethod
    def load(cls, path: Path) -> "EnvironmentConfig":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EnvironmentError(f"cannot load environment config {path}: {exc}") from exc
        required = {
            "version", "canonical_cone_world", "canonical_cone_free_world",
            "derived_world", "source_cone_model", "derived_cone_model",
            "canonical_sha256", "selector", "frozen_cone",
            "required_cone_clearance_m", "extra_cone_clearance_m",
            "footprint_steering_limit_rad",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise EnvironmentError("environment config fields do not match the V1 contract")
        if payload["version"] != VERSION:
            raise EnvironmentError(f"environment version must be {VERSION}")
        names = [
            payload["canonical_cone_world"], payload["canonical_cone_free_world"],
            payload["derived_world"], payload["source_cone_model"],
            payload["derived_cone_model"],
        ]
        if not all(isinstance(value, str) and value and Path(value).name == value for value in names):
            raise EnvironmentError("world/model identities must be non-empty basenames")
        if len(set(names[:3])) != 3:
            raise EnvironmentError("canonical and derived world names must be distinct")
        hashes = payload["canonical_sha256"]
        expected_hash_keys = {
            "canonical_cone_world", "canonical_cone_free_world", "canonical_route",
            "cone_mesh", "vehicle_sdf",
        }
        if not isinstance(hashes, dict) or set(hashes) != expected_hash_keys:
            raise EnvironmentError("canonical_sha256 fields do not match the V1 contract")
        if not all(isinstance(value, str) and len(value) == 64 for value in hashes.values()):
            raise EnvironmentError("canonical SHA-256 values must be 64-character strings")
        selector = payload["selector"]
        selector_keys = {
            "sample_step_m", "start_exclusion_m", "end_exclusion_m",
            "curvature_half_window_m", "curvature_neighborhood_m",
            "maximum_abs_curvature_per_m", "minimum_straight_run_m",
            "nonlocal_route_exclusion_m", "minimum_nonlocal_clearance_m",
            "side_tie_tolerance_m", "tie_break_side",
        }
        if not isinstance(selector, dict) or set(selector) != selector_keys:
            raise EnvironmentError("selector fields do not match the V1 contract")
        numeric_selector = [value for key, value in selector.items() if key != "tie_break_side"]
        if not all(_positive_number(value) for value in numeric_selector):
            raise EnvironmentError("selector numeric values must be finite and positive")
        if selector["tie_break_side"] not in ("left", "right"):
            raise EnvironmentError("tie_break_side must be left or right")
        frozen = payload["frozen_cone"]
        if not isinstance(frozen, dict) or set(frozen) != {
            "route_s_m", "x_m", "y_m", "yaw_rad", "local_curvature_per_m"
        } or not all(_finite_number(value) for value in frozen.values()):
            raise EnvironmentError("frozen_cone must contain finite V1 geometry")
        for key in ("required_cone_clearance_m", "extra_cone_clearance_m"):
            if not _positive_number(payload[key]):
                raise EnvironmentError(f"{key} must be finite and positive")
        if not _positive_number(payload["footprint_steering_limit_rad"]):
            raise EnvironmentError("footprint steering limit must be finite and positive")
        return cls(
            payload=payload,
            canonical_cone_world=names[0],
            canonical_cone_free_world=names[1],
            derived_world=names[2],
            source_cone_model=names[3],
            derived_cone_model=names[4],
            canonical_sha256=dict(hashes),
            selector=dict(selector),
            frozen_cone={key: float(value) for key, value in frozen.items()},
            required_cone_clearance_m=float(payload["required_cone_clearance_m"]),
            extra_cone_clearance_m=float(payload["extra_cone_clearance_m"]),
            footprint_steering_limit_rad=float(payload["footprint_steering_limit_rad"]),
        )


@dataclass(frozen=True)
class AssetSet:
    world: Path
    route: Path
    model: Path


@dataclass(frozen=True)
class RouteData:
    route: ClosedRoute
    center: tuple[Point, ...]
    inner: tuple[Point, ...]
    outer: tuple[Point, ...]


@dataclass(frozen=True)
class ConeGeometry:
    source_world: str
    source_model: str
    collision_name: str
    size_xyz_m: tuple[float, float, float]
    visual_uri: str
    source_pose_z_m: float

    @property
    def half_length_m(self) -> float:
        return self.size_xyz_m[0] / 2.0

    @property
    def half_width_m(self) -> float:
        return self.size_xyz_m[1] / 2.0


@dataclass(frozen=True)
class VehicleFootprint:
    source_sdf: str
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    steering_envelope_rad: float
    collision_elements: tuple[str, ...]
    simplification: str

    @property
    def length_m(self) -> float:
        return self.x_max_m - self.x_min_m

    @property
    def width_m(self) -> float:
        return self.y_max_m - self.y_min_m

    @property
    def vertices(self) -> tuple[Point, ...]:
        return (
            (self.x_min_m, self.y_min_m), (self.x_max_m, self.y_min_m),
            (self.x_max_m, self.y_max_m), (self.x_min_m, self.y_max_m),
        )


@dataclass(frozen=True)
class ConeSite:
    route_s_m: float
    x_m: float
    y_m: float
    yaw_rad: float
    local_curvature_per_m: float
    straight_run_start_s_m: float
    straight_run_end_s_m: float
    straight_run_length_m: float
    left_clearance_m: float
    right_clearance_m: float
    nonlocal_route_clearance_m: float
    chosen_side: str
    reason: str


def share_path(sim_root: Path) -> Path:
    return sim_root.expanduser().resolve() / "src" / "physicar-sim" / "share"


def asset_set(share: Path, world: str) -> AssetSet:
    return AssetSet(
        world=share / "worlds" / f"{world}.world",
        route=share / "routes" / f"{world}.npy",
        model=share / "models" / world,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise EnvironmentError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def directory_manifest(root: Path) -> dict[str, tuple[str, str]]:
    if not root.is_dir():
        raise EnvironmentError(f"model directory is missing: {root}")
    result: dict[str, tuple[str, str]] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            result[relative] = ("symlink", os.readlink(path))
        elif path.is_dir():
            result[relative] = ("directory", "")
        elif path.is_file():
            result[relative] = ("file", sha256_file(path))
    return result


def canonical_paths(config: EnvironmentConfig, share: Path) -> dict[str, Path]:
    cone_world = asset_set(share, config.canonical_cone_world)
    cone_free = asset_set(share, config.canonical_cone_free_world)
    return {
        "canonical_cone_world": cone_world.world,
        "canonical_cone_free_world": cone_free.world,
        "canonical_route": cone_free.route,
        "cone_mesh": share / "meshes" / config.canonical_cone_world / f"{config.source_cone_model}.dae",
        "vehicle_sdf": share / "models" / "physicar" / "model.sdf",
    }


def verify_canonical_hashes(config: EnvironmentConfig, share: Path) -> dict[str, str]:
    observed = {key: sha256_file(path) for key, path in canonical_paths(config, share).items()}
    mismatches = {
        key: {"expected": config.canonical_sha256[key], "observed": value}
        for key, value in observed.items() if config.canonical_sha256[key] != value
    }
    if mismatches:
        raise EnvironmentError(f"canonical asset identity mismatch: {mismatches}")
    cone_route = asset_set(share, config.canonical_cone_world).route
    if sha256_file(cone_route) != observed["canonical_route"]:
        raise EnvironmentError("cone and cone-free routes are not byte-identical")
    return observed


def parse_xml(path: Path) -> tuple[ET.Element, ET.Element]:
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise EnvironmentError(f"cannot parse simulator XML {path}: {exc}") from exc
    worlds = [root] if root.tag == "world" else root.findall("world")
    if len(worlds) != 1:
        raise EnvironmentError(f"expected exactly one world element in {path}")
    return root, worlds[0]


def _world_cone_models(world: ET.Element) -> list[ET.Element]:
    return [model for model in world.findall("model") if (model.get("name") or "").lower().startswith("cone")]


def parse_cone_geometry(config: EnvironmentConfig, share: Path) -> tuple[ConeGeometry, ET.Element]:
    source_path = asset_set(share, config.canonical_cone_world).world
    _, world = parse_xml(source_path)
    all_cones = _world_cone_models(world)
    if len(all_cones) != 6:
        raise EnvironmentError(f"original world must contain exactly six real cone models, found {len(all_cones)}")
    matches = [model for model in all_cones if model.get("name") == config.source_cone_model]
    if len(matches) != 1:
        raise EnvironmentError(f"source cone {config.source_cone_model!r} is not unique")
    model = matches[0]
    collisions = model.findall("./link/collision")
    if len(collisions) != 1:
        raise EnvironmentError("source cone must have exactly one collision element")
    size_element = collisions[0].find("./geometry/box/size")
    if size_element is None or not size_element.text:
        raise EnvironmentError("source cone collision is not an explicit box")
    size = _float_tuple(size_element.text, 3, "cone collision size")
    visual = model.find("./link/visual/geometry/mesh/uri")
    if visual is None or not visual.text or not visual.text.startswith("model://meshes/"):
        raise EnvironmentError("source cone visual does not reference a simulator mesh")
    mesh_relative = visual.text.removeprefix("model://meshes/")
    mesh = share / "meshes" / mesh_relative
    if not mesh.is_file():
        raise EnvironmentError(f"source cone visual mesh is missing: {mesh}")
    pose = _float_tuple(model.findtext("pose", ""), 6, "source cone pose")
    return ConeGeometry(
        source_world=config.canonical_cone_world,
        source_model=config.source_cone_model,
        collision_name=collisions[0].get("name") or "",
        size_xyz_m=(size[0], size[1], size[2]),
        visual_uri=visual.text.strip(),
        source_pose_z_m=pose[2],
    ), model


def parse_vehicle_footprint(config: EnvironmentConfig, share: Path) -> VehicleFootprint:
    path = share / "models" / "physicar" / "model.sdf"
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError) as exc:
        raise EnvironmentError(f"cannot parse PhysiCar SDF {path}: {exc}") from exc
    model = root.find("model") if root.tag != "model" else root
    if model is None or model.get("name") != "physicar":
        raise EnvironmentError("vehicle SDF does not contain the PhysiCar model")
    base = model.find("./link[@name='base_footprint']")
    if base is None:
        raise EnvironmentError("PhysiCar base_footprint link is missing")
    body = base.find("./collision[@name='base_footprint_fixed_joint_lump__base_collision_collision']")
    lidar = base.find("./collision[@name='base_footprint_fixed_joint_lump__lidar_collision_collision_1']")
    if body is None or lidar is None:
        raise EnvironmentError("required PhysiCar body/lidar collisions are missing")
    body_size = _float_tuple(body.findtext("./geometry/box/size", ""), 3, "vehicle body size")
    body_pose = _float_tuple(body.findtext("pose", "0 0 0 0 0 0"), 6, "vehicle body pose")
    if abs(body_pose[5]) > 1e-9:
        raise EnvironmentError("V1 footprint parser requires an axis-aligned body collision")
    x_min = body_pose[0] - body_size[0] / 2.0
    x_max = body_pose[0] + body_size[0] / 2.0
    y_min = body_pose[1] - body_size[1] / 2.0
    y_max = body_pose[1] + body_size[1] / 2.0

    lidar_pose = _float_tuple(lidar.findtext("pose", ""), 6, "lidar collision pose")
    lidar_radius = _single_float(lidar.findtext("./geometry/cylinder/radius", ""), "lidar radius")
    x_min = min(x_min, lidar_pose[0] - lidar_radius)
    x_max = max(x_max, lidar_pose[0] + lidar_radius)
    y_min = min(y_min, lidar_pose[1] - lidar_radius)
    y_max = max(y_max, lidar_pose[1] + lidar_radius)

    collision_names = [body.get("name") or "", lidar.get("name") or ""]
    wheel_names = (
        "front_left_wheel_link", "front_right_wheel_link",
        "rear_left_wheel_link", "rear_right_wheel_link",
    )
    for wheel_name in wheel_names:
        link = model.find(f"./link[@name='{wheel_name}']")
        if link is None:
            raise EnvironmentError(f"vehicle wheel link is missing: {wheel_name}")
        collision = link.find("collision")
        if collision is None:
            raise EnvironmentError(f"vehicle wheel collision is missing: {wheel_name}")
        radius = _single_float(collision.findtext("./geometry/cylinder/radius", ""), f"{wheel_name} radius")
        length = _single_float(collision.findtext("./geometry/cylinder/length", ""), f"{wheel_name} length")
        collision_pose = _float_tuple(collision.findtext("pose", ""), 6, f"{wheel_name} collision pose")
        if not math.isclose(abs(collision_pose[3]), math.pi / 2.0, abs_tol=1e-6):
            raise EnvironmentError(f"{wheel_name} cylinder is not oriented along the lateral axis")
        joint_name = wheel_name.replace("_wheel_link", "_steering_joint") if wheel_name.startswith("front") else wheel_name.replace("_link", "_joint")
        joint = model.find(f"./joint[@name='{joint_name}']")
        if joint is None or joint.find("pose") is None:
            raise EnvironmentError(f"vehicle wheel anchor joint is missing: {joint_name}")
        anchor = _float_tuple(joint.findtext("pose", ""), 6, f"{joint_name} pose")
        steering = config.footprint_steering_limit_rad if wheel_name.startswith("front") else 0.0
        # The wheel-cylinder ground projection is a rectangle with radius in
        # rolling x and half cylinder length in lateral y.  Enclose every
        # articulation angle in [-steering,+steering].
        half_x = radius * math.cos(steering) + (length / 2.0) * math.sin(steering)
        half_y = radius * math.sin(steering) + (length / 2.0) * math.cos(steering)
        center_x = anchor[0] + collision_pose[0]
        center_y = anchor[1] + collision_pose[1]
        x_min, x_max = min(x_min, center_x - half_x), max(x_max, center_x + half_x)
        y_min, y_max = min(y_min, center_y - half_y), max(y_max, center_y + half_y)
        collision_names.append(collision.get("name") or "")
    values = (x_min, x_max, y_min, y_max)
    if not all(math.isfinite(value) for value in values) or x_min >= x_max or y_min >= y_max:
        raise EnvironmentError("derived vehicle footprint is invalid")
    return VehicleFootprint(
        source_sdf=str(path),
        x_min_m=x_min,
        x_max_m=x_max,
        y_min_m=y_min,
        y_max_m=y_max,
        steering_envelope_rad=config.footprint_steering_limit_rad,
        collision_elements=tuple(collision_names),
        simplification=(
            "axis-aligned base-frame rectangle enclosing the body box, lidar cylinder, "
            "rear wheel cylinders, and both front wheel cylinders for every steering "
            "angle within the Expert command clamp"
        ),
    )


def load_route(path: Path) -> RouteData:
    try:
        import numpy as np
        array = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise EnvironmentError(f"cannot load route array {path}: {exc}") from exc
    if array.ndim != 2 or array.shape[1] != 6 or array.shape[0] < 20:
        raise EnvironmentError(f"route must be an N x 6 numeric array, got {array.shape}")
    if not bool(np.isfinite(array).all()):
        raise EnvironmentError("route array contains non-finite geometry")
    center = _drop_closed_duplicate(tuple((float(row[0]), float(row[1])) for row in array), "center")
    inner = _drop_closed_duplicate(tuple((float(row[2]), float(row[3])) for row in array), "inner")
    outer = _drop_closed_duplicate(tuple((float(row[4]), float(row[5])) for row in array), "outer")
    return RouteData(ClosedRoute(center, inner, outer), center, inner, outer)


def route_yaw(route: ClosedRoute, s: float, half_window_m: float = 0.1) -> float:
    before, after = route.point_at(s - half_window_m), route.point_at(s + half_window_m)
    return math.atan2(after[1] - before[1], after[0] - before[0])


def route_curvature(route: ClosedRoute, s: float, half_window_m: float) -> float:
    first, middle, last = route.point_at(s - half_window_m), route.point_at(s), route.point_at(s + half_window_m)
    a, b, c = _distance(first, middle), _distance(middle, last), _distance(first, last)
    denominator = a * b * c
    if denominator <= 1e-12:
        raise EnvironmentError("curvature window contains degenerate route geometry")
    cross = (middle[0] - first[0]) * (last[1] - middle[1]) - (middle[1] - first[1]) * (last[0] - middle[0])
    return 2.0 * cross / denominator


def side_clearances(route_data: RouteData, s: float, yaw: float) -> tuple[float, float]:
    origin = route_data.route.point_at(s)
    left = (-math.sin(yaw), math.cos(yaw))
    right = (-left[0], -left[1])
    boundaries = (route_data.inner, route_data.outer)
    left_distance = _ray_boundary_distance(origin, left, boundaries)
    right_distance = _ray_boundary_distance(origin, right, boundaries)
    return left_distance, right_distance


def select_cone_site(config: EnvironmentConfig, route_data: RouteData) -> ConeSite:
    route = route_data.route
    selector = config.selector
    step = float(selector["sample_step_m"])
    first = float(selector["start_exclusion_m"])
    last = route.length - float(selector["end_exclusion_m"])
    if last <= first:
        raise EnvironmentError("route is too short for the selector exclusions")
    count = int(math.floor((last - first) / step)) + 1
    samples = [first + index * step for index in range(count)]
    curvature_half = float(selector["curvature_half_window_m"])
    neighborhood = float(selector["curvature_neighborhood_m"])
    neighborhood_count = int(math.ceil(neighborhood / step))

    def is_straight(s: float) -> bool:
        maximum = max(
            abs(route_curvature(route, s + offset * step, curvature_half))
            for offset in range(-neighborhood_count, neighborhood_count + 1)
        )
        return maximum <= float(selector["maximum_abs_curvature_per_m"])

    flags = [is_straight(s) for s in samples]
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate([*flags, False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            runs.append((start, index - 1))
            start = None
    eligible = [run for run in runs if samples[run[1]] - samples[run[0]] >= float(selector["minimum_straight_run_m"])]
    if not eligible:
        raise EnvironmentError("no route section satisfies the deterministic straight selector")
    chosen_run = sorted(eligible, key=lambda run: (-(samples[run[1]] - samples[run[0]]), samples[run[0]]))[0]
    midpoint = (samples[chosen_run[0]] + samples[chosen_run[1]]) / 2.0
    site_index = min(range(chosen_run[0], chosen_run[1] + 1), key=lambda index: (abs(samples[index] - midpoint), index))
    s = samples[site_index]
    point = route.point_at(s)
    yaw = route_yaw(route, s)
    curvature = route_curvature(route, s, curvature_half)
    left, right = side_clearances(route_data, s, yaw)
    tie_tolerance = float(selector["side_tie_tolerance_m"])
    if left > right + tie_tolerance:
        side = "left"
    elif right > left + tie_tolerance:
        side = "right"
    else:
        side = str(selector["tie_break_side"])
    nonlocal_clearance = _nonlocal_route_clearance(
        route,
        s,
        step,
        float(selector["nonlocal_route_exclusion_m"]),
    )
    if nonlocal_clearance < float(selector["minimum_nonlocal_clearance_m"]):
        raise EnvironmentError(
            f"selected site is ambiguous with a nonlocal route section ({nonlocal_clearance:.3f}m)"
        )
    site = ConeSite(
        route_s_m=s,
        x_m=point[0],
        y_m=point[1],
        yaw_rad=yaw,
        local_curvature_per_m=curvature,
        straight_run_start_s_m=samples[chosen_run[0]],
        straight_run_end_s_m=samples[chosen_run[1]],
        straight_run_length_m=samples[chosen_run[1]] - samples[chosen_run[0]],
        left_clearance_m=left,
        right_clearance_m=right,
        nonlocal_route_clearance_m=nonlocal_clearance,
        chosen_side=side,
        reason=(
            "midpoint sample of the longest eligible low-curvature run after spawn/final-region "
            "exclusions; deterministic side uses greater ray clearance with a frozen tie-break"
        ),
    )
    verify_frozen_site(config, site)
    return site


def verify_frozen_site(config: EnvironmentConfig, site: ConeSite, tolerance: float = 1e-6) -> None:
    observed = {
        "route_s_m": site.route_s_m, "x_m": site.x_m, "y_m": site.y_m,
        "yaw_rad": site.yaw_rad, "local_curvature_per_m": site.local_curvature_per_m,
    }
    mismatches = {
        key: {"frozen": config.frozen_cone[key], "computed": value}
        for key, value in observed.items()
        if not math.isclose(config.frozen_cone[key], value, rel_tol=0.0, abs_tol=tolerance)
    }
    if mismatches:
        raise EnvironmentError(f"deterministic cone site differs from frozen config: {mismatches}")


def derive_world_root(
    config: EnvironmentConfig,
    cone_free_path: Path,
    source_cone: ET.Element,
    site: ConeSite,
    cone_geometry: ConeGeometry,
) -> ET.Element:
    root, world = parse_xml(cone_free_path)
    if world.get("name") != config.canonical_cone_free_world:
        raise EnvironmentError("cone-free source has the wrong internal world name")
    if _world_cone_models(world):
        raise EnvironmentError("preserved cone-free world unexpectedly contains a cone")
    world.set("name", config.derived_world)
    cone = copy.deepcopy(source_cone)
    cone.set("name", config.derived_cone_model)
    pose = cone.find("pose")
    if pose is None:
        raise EnvironmentError("source cone has no model pose")
    pose.text = (
        f"{site.x_m:.9f} {site.y_m:.9f} {cone_geometry.source_pose_z_m:.9f} "
        f"0 0 {site.yaw_rad:.9f}"
    )
    children = list(world)
    first_model = next((index for index, child in enumerate(children) if child.tag == "model"), len(children))
    world.insert(first_model, cone)
    ET.indent(root, space="  ")
    return root


def expected_world_bytes(
    config: EnvironmentConfig,
    cone_free_path: Path,
    source_cone: ET.Element,
    site: ConeSite,
    cone_geometry: ConeGeometry,
) -> bytes:
    root = derive_world_root(config, cone_free_path, source_cone, site, cone_geometry)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True, short_empty_elements=True)


def verify_derived_environment(config: EnvironmentConfig, share: Path) -> dict[str, Any]:
    hashes = verify_canonical_hashes(config, share)
    cone_geometry, source_cone = parse_cone_geometry(config, share)
    footprint = parse_vehicle_footprint(config, share)
    cone_free = asset_set(share, config.canonical_cone_free_world)
    derived = asset_set(share, config.derived_world)
    route_data = load_route(cone_free.route)
    site = select_cone_site(config, route_data)
    if not derived.world.is_file() or not derived.route.is_file() or not derived.model.is_dir():
        raise EnvironmentError("one or more derived one-cone assets are missing")
    expected = expected_world_bytes(config, cone_free.world, source_cone, site, cone_geometry)
    if derived.world.read_bytes() != expected:
        raise EnvironmentError("derived world bytes differ from the deterministic transformation")
    if sha256_file(derived.route) != sha256_file(cone_free.route):
        raise EnvironmentError("derived route is not byte-identical to the preserved route")
    if directory_manifest(derived.model) != directory_manifest(cone_free.model):
        raise EnvironmentError("derived model metadata differs from the cone-free source")
    _, world = parse_xml(derived.world)
    cones = _world_cone_models(world)
    if len(cones) != 1 or cones[0].get("name") != config.derived_cone_model:
        raise EnvironmentError("derived world does not contain exactly the intended cone")
    size = _float_tuple(cones[0].findtext("./link/collision/geometry/box/size", ""), 3, "derived cone size")
    if tuple(size) != cone_geometry.size_xyz_m:
        raise EnvironmentError("derived cone collision differs from its source geometry")
    visual_uri = cones[0].findtext("./link/visual/geometry/mesh/uri", "").strip()
    if visual_uri != cone_geometry.visual_uri:
        raise EnvironmentError("derived cone visual provenance differs from the source cone")
    other_collisions = other_world_collision_clearance(config, share, site, cone_geometry)
    return {
        "result": "PASS",
        "canonical_hashes": hashes,
        "canonical_cone_free_world_unchanged": True,
        "derived_world": config.derived_world,
        "cone_count": 1,
        "route_sha256": sha256_file(derived.route),
        "route_byte_identical": True,
        "model_metadata_identical": True,
        "cone_geometry": cone_geometry_dict(cone_geometry),
        "vehicle_footprint": vehicle_footprint_dict(footprint),
        "cone_site": cone_site_dict(site),
        "other_world_collision_clearance": other_collisions,
    }


def generate_environment(config: EnvironmentConfig, share: Path, force: bool = False) -> dict[str, Any]:
    verify_canonical_hashes(config, share)
    cone_geometry, source_cone = parse_cone_geometry(config, share)
    parse_vehicle_footprint(config, share)
    cone_free = asset_set(share, config.canonical_cone_free_world)
    derived = asset_set(share, config.derived_world)
    site = select_cone_site(config, load_route(cone_free.route))
    expected = expected_world_bytes(config, cone_free.world, source_cone, site, cone_geometry)
    fingerprint_before = {
        "world": sha256_file(cone_free.world),
        "route": sha256_file(cone_free.route),
        "model": directory_manifest(cone_free.model),
    }
    targets = (derived.world, derived.route, derived.model)
    if any(path.exists() for path in targets):
        try:
            result = verify_derived_environment(config, share)
        except EnvironmentError:
            if not force:
                raise EnvironmentError(
                    "derived assets exist but fail verification; use --force to replace only the derived targets"
                )
        else:
            result["generation"] = "already valid; no files rewritten"
            return result
    with tempfile.TemporaryDirectory(prefix=".cone-avoidance-v1-", dir=share) as temporary:
        staged = asset_set(Path(temporary), config.derived_world)
        staged.world.parent.mkdir(parents=True, exist_ok=True)
        staged.route.parent.mkdir(parents=True, exist_ok=True)
        staged.model.parent.mkdir(parents=True, exist_ok=True)
        staged.world.write_bytes(expected)
        shutil.copy2(cone_free.route, staged.route)
        shutil.copytree(cone_free.model, staged.model, symlinks=True, copy_function=shutil.copy2)
        if force:
            for target in targets:
                _remove_exact_target(target)
        for source, target in ((staged.route, derived.route), (staged.model, derived.model), (staged.world, derived.world)):
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(source, target)
    fingerprint_after = {
        "world": sha256_file(cone_free.world),
        "route": sha256_file(cone_free.route),
        "model": directory_manifest(cone_free.model),
    }
    if fingerprint_after != fingerprint_before:
        raise EnvironmentError("preserved cone-free assets changed during generation")
    result = verify_derived_environment(config, share)
    result["generation"] = "generated and validated"
    return result


def cone_geometry_dict(geometry: ConeGeometry) -> dict[str, Any]:
    return {
        "source_world": geometry.source_world,
        "source_model": geometry.source_model,
        "collision_name": geometry.collision_name,
        "collision_size_xyz_m": list(geometry.size_xyz_m),
        "collision_2d_half_extents_m": [geometry.half_length_m, geometry.half_width_m],
        "visual_uri": geometry.visual_uri,
        "source_pose_z_m": geometry.source_pose_z_m,
    }


def vehicle_footprint_dict(footprint: VehicleFootprint) -> dict[str, Any]:
    return {
        "source_sdf": footprint.source_sdf,
        "length_m": footprint.length_m,
        "width_m": footprint.width_m,
        "x_extent_m": [footprint.x_min_m, footprint.x_max_m],
        "y_extent_m": [footprint.y_min_m, footprint.y_max_m],
        "vertices_base_frame_m": [list(point) for point in footprint.vertices],
        "steering_envelope_rad": footprint.steering_envelope_rad,
        "collision_elements": list(footprint.collision_elements),
        "conservative_simplification": footprint.simplification,
    }


def cone_site_dict(site: ConeSite) -> dict[str, Any]:
    return {
        "route_s_m": site.route_s_m, "x_m": site.x_m, "y_m": site.y_m,
        "yaw_rad": site.yaw_rad, "local_curvature_per_m": site.local_curvature_per_m,
        "straight_run_start_s_m": site.straight_run_start_s_m,
        "straight_run_end_s_m": site.straight_run_end_s_m,
        "straight_run_length_m": site.straight_run_length_m,
        "left_clearance_m": site.left_clearance_m,
        "right_clearance_m": site.right_clearance_m,
        "nonlocal_route_clearance_m": site.nonlocal_route_clearance_m,
        "chosen_side": site.chosen_side, "reason": site.reason,
    }


def other_world_collision_clearance(
    config: EnvironmentConfig,
    share: Path,
    site: ConeSite,
    cone: ConeGeometry,
) -> dict[str, Any]:
    """Conservatively check the cone against every explicit 2D world box."""
    _, world = parse_xml(asset_set(share, config.canonical_cone_free_world).world)
    cone_polygon = box_polygon(
        site.x_m, site.y_m, site.yaw_rad, cone.half_length_m, cone.half_width_m
    )
    records: list[dict[str, Any]] = []
    for model in world.findall("model"):
        model_name = model.get("name") or ""
        model_pose = _float_tuple(model.findtext("pose", "0 0 0 0 0 0"), 6, f"{model_name} pose")
        cosine, sine = math.cos(model_pose[5]), math.sin(model_pose[5])
        for collision in model.findall("./link/collision"):
            size_text = collision.findtext("./geometry/box/size")
            if not size_text:
                continue
            size = _float_tuple(size_text, 3, f"{model_name} collision size")
            local = _float_tuple(collision.findtext("pose", "0 0 0 0 0 0"), 6, f"{model_name} collision pose")
            center_x = model_pose[0] + cosine * local[0] - sine * local[1]
            center_y = model_pose[1] + sine * local[0] + cosine * local[1]
            polygon = box_polygon(
                center_x, center_y, model_pose[5] + local[5], size[0] / 2.0, size[1] / 2.0
            )
            clearance, intersects = polygon_clearance(cone_polygon, polygon)
            if intersects:
                raise EnvironmentError(
                    f"frozen cone intersects explicit world collision {model_name}/{collision.get('name')}"
                )
            records.append({
                "model": model_name,
                "collision": collision.get("name") or "",
                "clearance_m": clearance,
            })
    if not records:
        raise EnvironmentError("cone-free source exposes no explicit world collision boxes")
    nearest = min(records, key=lambda record: float(record["clearance_m"]))
    return {
        "result": "PASS",
        "checked_explicit_box_collisions": len(records),
        "nearest_model": nearest["model"],
        "nearest_collision": nearest["collision"],
        "minimum_clearance_m": nearest["clearance_m"],
    }


def footprint_polygon(footprint: VehicleFootprint, x: float, y: float, yaw: float) -> tuple[Point, ...]:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return tuple(
        (x + cosine * px - sine * py, y + sine * px + cosine * py)
        for px, py in footprint.vertices
    )


def box_polygon(x: float, y: float, yaw: float, half_length: float, half_width: float) -> tuple[Point, ...]:
    cosine, sine = math.cos(yaw), math.sin(yaw)
    local = ((-half_length, -half_width), (half_length, -half_width),
             (half_length, half_width), (-half_length, half_width))
    return tuple((x + cosine * px - sine * py, y + sine * px + cosine * py) for px, py in local)


def polygon_clearance(first: Sequence[Point], second: Sequence[Point]) -> tuple[float, bool]:
    if _polygons_intersect(first, second):
        return 0.0, True
    distance = min(
        _segment_distance(a, b, c, d)
        for a, b in _closed_segments(first)
        for c, d in _closed_segments(second)
    )
    return distance, False


def point_track_clearance(route: ClosedRoute, point: Point) -> float:
    if route.inner is None or route.outer is None:
        raise EnvironmentError("track boundaries are unavailable")
    if route.is_off_track(point, 0.0):
        return -route.track_boundary_distance(point)  # type: ignore[operator]
    return min(_distance_to_polyline(point, route.inner), _distance_to_polyline(point, route.outer))


def _xml_signature(element: ET.Element) -> tuple[Any, ...]:
    return (
        element.tag, tuple(sorted(element.attrib.items())), (element.text or "").strip(),
        tuple(_xml_signature(child) for child in element),
    )


def _ray_boundary_distance(origin: Point, direction: Point, boundaries: Iterable[Sequence[Point]]) -> float:
    distances: list[float] = []
    for polygon in boundaries:
        for start, end in _closed_segments(polygon):
            segment = (end[0] - start[0], end[1] - start[1])
            denominator = _cross(direction, segment)
            if abs(denominator) <= 1e-12:
                continue
            relative = (start[0] - origin[0], start[1] - origin[1])
            ray_t = _cross(relative, segment) / denominator
            segment_t = _cross(relative, direction) / denominator
            if ray_t >= -1e-9 and -1e-9 <= segment_t <= 1.0 + 1e-9:
                distances.append(max(0.0, ray_t))
    positive = [value for value in distances if value > 1e-7]
    if not positive:
        raise EnvironmentError("could not intersect a route-normal ray with track boundaries")
    return min(positive)


def _nonlocal_route_clearance(route: ClosedRoute, s: float, step: float, excluded_arc: float) -> float:
    point = route.point_at(s)
    result = math.inf
    count = int(math.ceil(route.length / step))
    for index in range(count):
        other_s = index * route.length / count
        arc = abs((other_s - s + route.length / 2.0) % route.length - route.length / 2.0)
        if arc > excluded_arc:
            result = min(result, _distance(point, route.point_at(other_s)))
    if not math.isfinite(result):
        raise EnvironmentError("nonlocal route-clearance computation had no eligible samples")
    return result


def _polygons_intersect(first: Sequence[Point], second: Sequence[Point]) -> bool:
    if any(_segments_intersect(a, b, c, d) for a, b in _closed_segments(first) for c, d in _closed_segments(second)):
        return True
    return _point_in_polygon(first[0], second) or _point_in_polygon(second[0], first)


def _segment_distance(a: Point, b: Point, c: Point, d: Point) -> float:
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(_point_segment_distance(a, c, d), _point_segment_distance(b, c, d),
               _point_segment_distance(c, a, b), _point_segment_distance(d, a, b))


def _point_segment_distance(point: Point, start: Point, end: Point) -> float:
    vx, vy = end[0] - start[0], end[1] - start[1]
    length_sq = vx * vx + vy * vy
    if length_sq <= 1e-18:
        return _distance(point, start)
    fraction = max(0.0, min(1.0, ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / length_sq))
    return _distance(point, (start[0] + fraction * vx, start[1] + fraction * vy))


def _distance_to_polyline(point: Point, polygon: Sequence[Point]) -> float:
    return min(_point_segment_distance(point, start, end) for start, end in _closed_segments(polygon))


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    def orientation(p: Point, q: Point, r: Point) -> float:
        return _cross((q[0] - p[0], q[1] - p[1]), (r[0] - p[0], r[1] - p[1]))
    values = orientation(a, b, c), orientation(a, b, d), orientation(c, d, a), orientation(c, d, b)
    if ((values[0] > 1e-10 and values[1] < -1e-10) or (values[0] < -1e-10 and values[1] > 1e-10)) and (
        (values[2] > 1e-10 and values[3] < -1e-10) or (values[2] < -1e-10 and values[3] > 1e-10)
    ):
        return True
    return (
        abs(values[0]) <= 1e-10 and _on_segment(c, a, b)
        or abs(values[1]) <= 1e-10 and _on_segment(d, a, b)
        or abs(values[2]) <= 1e-10 and _on_segment(a, c, d)
        or abs(values[3]) <= 1e-10 and _on_segment(b, c, d)
    )


def _point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    inside = False
    for start, end in _closed_segments(polygon):
        if _on_segment(point, start, end):
            return True
        if (start[1] > point[1]) != (end[1] > point[1]):
            crossing_x = start[0] + (point[1] - start[1]) * (end[0] - start[0]) / (end[1] - start[1])
            if point[0] < crossing_x:
                inside = not inside
    return inside


def _on_segment(point: Point, start: Point, end: Point) -> bool:
    return abs(_cross((end[0] - start[0], end[1] - start[1]), (point[0] - start[0], point[1] - start[1]))) <= 1e-10 and (
        min(start[0], end[0]) - 1e-10 <= point[0] <= max(start[0], end[0]) + 1e-10
        and min(start[1], end[1]) - 1e-10 <= point[1] <= max(start[1], end[1]) + 1e-10
    )


def _closed_segments(points: Sequence[Point]):
    for index, start in enumerate(points):
        yield start, points[(index + 1) % len(points)]


def _drop_closed_duplicate(points: tuple[Point, ...], name: str) -> tuple[Point, ...]:
    result = points[:-1] if len(points) > 1 and _distance(points[0], points[-1]) <= 1e-9 else points
    if len(result) < 3:
        raise EnvironmentError(f"{name} route geometry has fewer than three points")
    return result


def _float_tuple(text: str, count: int, label: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value) for value in text.split())
    except ValueError as exc:
        raise EnvironmentError(f"{label} is not numeric") from exc
    if len(values) < count or not all(math.isfinite(value) for value in values[:count]):
        raise EnvironmentError(f"{label} must contain {count} finite values")
    return values[:count]


def _single_float(text: str, label: str) -> float:
    values = _float_tuple(text, 1, label)
    if values[0] <= 0:
        raise EnvironmentError(f"{label} must be positive")
    return values[0]


def _cross(first: Point, second: Point) -> float:
    return first[0] * second[1] - first[1] * second[0]


def _distance(first: Point, second: Point) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _positive_number(value: object) -> bool:
    return _finite_number(value) and float(value) > 0.0


def _remove_exact_target(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--generate", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only and args.force:
        print("ERROR: --force requires --generate", file=sys.stderr)
        return 2
    try:
        config = EnvironmentConfig.load(args.config)
        share = share_path(args.sim_root)
        result = (
            generate_environment(config, share, force=args.force)
            if args.generate else verify_derived_environment(config, share)
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (EnvironmentError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
