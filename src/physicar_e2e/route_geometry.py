"""Closed-polyline geometry and deterministic Pure Pursuit helpers."""

from __future__ import annotations

from dataclasses import dataclass
import bisect
import math
from typing import Sequence

Point = tuple[float, float]


@dataclass(frozen=True)
class Projection:
    point: Point
    segment_index: int
    segment_fraction: float
    s: float
    distance: float
    signed_error: float


class ClosedRoute:
    """A closed centerline with an optional outer shell and inner hole."""

    def __init__(
        self,
        waypoints: Sequence[Sequence[float]],
        inner: Sequence[Sequence[float]] | None = None,
        outer: Sequence[Sequence[float]] | None = None,
    ) -> None:
        raw = list(_closed_points(waypoints, "centerline"))
        if len(raw) < 3:
            raise ValueError("closed route needs at least three distinct points")
        self.points = tuple(raw)
        if (inner is None) != (outer is None):
            raise ValueError("inner and outer boundaries must be provided together")
        self.inner = _optional_boundary(inner, "inner")
        self.outer = _optional_boundary(outer, "outer")
        if self.inner is not None and self.outer is not None:
            _validate_track_ring(self.inner, self.outer)
        self.segment_lengths: list[float] = []
        self.cumulative = [0.0]
        for index, point in enumerate(self.points):
            length = _distance(point, self.points[(index + 1) % len(self.points)])
            if length <= 1e-9:
                raise ValueError(f"zero-length route segment at index {index}")
            self.segment_lengths.append(length)
            self.cumulative.append(self.cumulative[-1] + length)
        self.length = self.cumulative[-1]

    def project(self, position: Sequence[float]) -> Projection:
        px, py = _point(position)
        best: Projection | None = None
        for index, start in enumerate(self.points):
            end = self.points[(index + 1) % len(self.points)]
            vx, vy = end[0] - start[0], end[1] - start[1]
            length_sq = vx * vx + vy * vy
            t = max(0.0, min(1.0, ((px - start[0]) * vx + (py - start[1]) * vy) / length_sq))
            qx, qy = start[0] + t * vx, start[1] + t * vy
            dx, dy = px - qx, py - qy
            distance = math.hypot(dx, dy)
            cross = vx * (py - qy) - vy * (px - qx)
            signed = math.copysign(distance, cross) if distance else 0.0
            candidate = Projection(
                point=(qx, qy),
                segment_index=index,
                segment_fraction=t,
                s=(self.cumulative[index] + t * self.segment_lengths[index]) % self.length,
                distance=distance,
                signed_error=signed,
            )
            if best is None or candidate.distance < best.distance:
                best = candidate
        assert best is not None
        return best

    def point_at(self, s: float) -> Point:
        wrapped = s % self.length
        index = min(bisect.bisect_right(self.cumulative, wrapped) - 1, len(self.points) - 1)
        fraction = (wrapped - self.cumulative[index]) / self.segment_lengths[index]
        return _lerp(self.points[index], self.points[(index + 1) % len(self.points)], fraction)

    def track_boundary_distance(self, position: Sequence[float]) -> float | None:
        """Return zero in the track band, otherwise distance to its real boundary."""
        if self.inner is None or self.outer is None:
            return None
        point = _point(position)
        inside_outer = _point_in_polygon(point, self.outer, include_boundary=True)
        inside_inner_hole = _point_in_polygon(point, self.inner, include_boundary=False)
        if inside_outer and not inside_inner_hole:
            return 0.0
        return min(
            _distance_to_closed_polyline(point, self.inner),
            _distance_to_closed_polyline(point, self.outer),
        )

    def is_off_track(self, position: Sequence[float], margin_m: float) -> bool:
        if not math.isfinite(margin_m) or margin_m < 0:
            raise ValueError("off-track margin must be finite and nonnegative")
        distance = self.track_boundary_distance(position)
        if distance is None:
            raise ValueError("track boundaries are unavailable")
        return distance > margin_m


def pure_pursuit_steering(
    position: Sequence[float],
    yaw: float,
    target: Sequence[float],
    wheelbase_m: float,
    max_steering_rad: float,
) -> tuple[float, float, float]:
    """Return clamped steering, curvature, and actual target distance.

    The actual Euclidean vehicle-to-target distance is used in the curvature
    denominator; arc-length lookahead can differ near sharp corners.
    """
    x, y = _point(position)
    tx, ty = _point(target)
    dx, dy = tx - x, ty - y
    actual = math.hypot(dx, dy)
    if actual <= 1e-9:
        raise ValueError("Pure Pursuit target is coincident with vehicle")
    lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
    curvature = 2.0 * lateral / (actual * actual)
    steering = math.atan(wheelbase_m * curvature)
    steering = max(-max_steering_rad, min(max_steering_rad, steering))
    if not all(math.isfinite(v) for v in (steering, curvature, actual)):
        raise ValueError("non-finite Pure Pursuit result")
    return steering, curvature, actual


class ProgressTracker:
    def __init__(self, route_length: float, maximum_jump_m: float) -> None:
        if route_length <= 0 or maximum_jump_m <= 0:
            raise ValueError("route length and maximum jump must be positive")
        self.route_length = route_length
        self.maximum_jump_m = maximum_jump_m
        self.last_s: float | None = None
        self.unwrapped = 0.0
        self.rejected_jumps = 0

    def update(self, s: float) -> float:
        s %= self.route_length
        if self.last_s is None:
            self.last_s = s
            return self.unwrapped
        delta = (s - self.last_s + self.route_length / 2.0) % self.route_length - self.route_length / 2.0
        if abs(delta) > self.maximum_jump_m:
            self.rejected_jumps += 1
            return self.unwrapped
        self.unwrapped += delta
        self.last_s = s
        return self.unwrapped

    def lap_complete(
        self,
        distance_to_start_m: float,
        start_gate_radius_m: float,
        minimum_progress_fraction: float,
    ) -> bool:
        return (
            self.unwrapped >= minimum_progress_fraction * self.route_length
            and distance_to_start_m <= start_gate_radius_m
        )


class OffTrackMonitor:
    def __init__(self, grace_s: float) -> None:
        if grace_s < 0:
            raise ValueError("off-track grace cannot be negative")
        self.grace_s = grace_s
        self.started_at: float | None = None
        self.event_count = 0
        self.total_duration_s = 0.0

    def update(self, is_off_track: bool, now: float) -> bool:
        if is_off_track:
            if self.started_at is None:
                self.started_at = now
                self.event_count += 1
            return now - self.started_at >= self.grace_s
        if self.started_at is not None:
            self.total_duration_s += max(0.0, now - self.started_at)
            self.started_at = None
        return False

    def finalize(self, now: float) -> None:
        if self.started_at is not None:
            self.total_duration_s += max(0.0, now - self.started_at)
            self.started_at = None


def _point(value: Sequence[float]) -> Point:
    if len(value) < 2:
        raise ValueError("point requires x and y")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(v) for v in result):
        raise ValueError("point coordinates must be finite")
    return result


def _optional_boundary(values: Sequence[Sequence[float]] | None, name: str) -> tuple[Point, ...] | None:
    if values is None:
        return None
    return _closed_points(values, name)


def _closed_points(values: Sequence[Sequence[float]], name: str) -> tuple[Point, ...]:
    points: list[Point] = []
    for value in values:
        point = _point(value)
        if not points or _distance(point, points[-1]) > 1e-9:
            points.append(point)
    if len(points) >= 2 and _distance(points[0], points[-1]) <= 1e-9:
        points.pop()
    if len(points) < 3:
        raise ValueError(f"{name} needs at least three distinct points")
    return tuple(points)


def _validate_track_ring(inner: tuple[Point, ...], outer: tuple[Point, ...]) -> None:
    _validate_simple_polygon(inner, "inner")
    _validate_simple_polygon(outer, "outer")
    for a1, a2 in _closed_segments(inner):
        for b1, b2 in _closed_segments(outer):
            if _segments_intersect(a1, a2, b1, b2):
                raise ValueError("inner and outer track boundaries intersect")
    if not all(_point_in_polygon(point, outer, include_boundary=False) for point in inner):
        raise ValueError("inner boundary must be strictly enclosed by outer boundary")


def _validate_simple_polygon(polygon: tuple[Point, ...], name: str) -> None:
    segments = list(_closed_segments(polygon))
    count = len(segments)
    for i, (a1, a2) in enumerate(segments):
        for j in range(i + 1, count):
            if j == i + 1 or (i == 0 and j == count - 1):
                continue
            if _segments_intersect(a1, a2, *segments[j]):
                raise ValueError(f"{name} boundary self-intersects")
    area_twice = sum(a[0] * b[1] - b[0] * a[1] for a, b in segments)
    if abs(area_twice) <= 1e-9:
        raise ValueError(f"{name} boundary has zero area")


def _point_in_polygon(point: Point, polygon: tuple[Point, ...], *, include_boundary: bool) -> bool:
    inside = False
    px, py = point
    for start, end in _closed_segments(polygon):
        if _point_on_segment(point, start, end):
            return include_boundary
        if (start[1] > py) != (end[1] > py):
            crossing_x = start[0] + (py - start[1]) * (end[0] - start[0]) / (end[1] - start[1])
            if px < crossing_x:
                inside = not inside
    return inside


def _distance_to_closed_polyline(point: Point, polygon: tuple[Point, ...]) -> float:
    return min(_distance_to_segment(point, start, end) for start, end in _closed_segments(polygon))


def _distance_to_segment(point: Point, start: Point, end: Point) -> float:
    vx, vy = end[0] - start[0], end[1] - start[1]
    length_sq = vx * vx + vy * vy
    t = max(
        0.0,
        min(1.0, ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / length_sq),
    )
    nearest = (start[0] + t * vx, start[1] + t * vy)
    return _distance(point, nearest)


def _closed_segments(polygon: tuple[Point, ...]):
    for index, start in enumerate(polygon):
        yield start, polygon[(index + 1) % len(polygon)]


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    cross = (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])
    if abs(cross) > 1e-9:
        return False
    return (
        min(start[0], end[0]) - 1e-9 <= point[0] <= max(start[0], end[0]) + 1e-9
        and min(start[1], end[1]) - 1e-9 <= point[1] <= max(start[1], end[1]) + 1e-9
    )


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    def orientation(p: Point, q: Point, r: Point) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    o1, o2 = orientation(a, b, c), orientation(a, b, d)
    o3, o4 = orientation(c, d, a), orientation(c, d, b)
    if ((o1 > 1e-9 and o2 < -1e-9) or (o1 < -1e-9 and o2 > 1e-9)) and (
        (o3 > 1e-9 and o4 < -1e-9) or (o3 < -1e-9 and o4 > 1e-9)
    ):
        return True
    return (
        (abs(o1) <= 1e-9 and _point_on_segment(c, a, b))
        or (abs(o2) <= 1e-9 and _point_on_segment(d, a, b))
        or (abs(o3) <= 1e-9 and _point_on_segment(a, c, d))
        or (abs(o4) <= 1e-9 and _point_on_segment(b, c, d))
    )


def _lerp(a: Point, b: Point, fraction: float) -> Point:
    return (a[0] + fraction * (b[0] - a[0]), a[1] + fraction * (b[1] - a[1]))


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
