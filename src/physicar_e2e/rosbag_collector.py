"""Small, fail-fast ROS 2 bag collector around the canonical expert driver."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Protocol, Sequence

from .expert_driver import DriverConfig, preflight, run_driver, wait_after_reset
from .sim_client import SimClient


ROS_SETUP = """source /opt/ros/jazzy/setup.bash
source /opt/physicar/install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///opt/physicar/src/physicar-ros/deploy/cyclonedds.xml
"""


class CollectorError(RuntimeError):
    pass


@dataclass(frozen=True)
class CollectorConfig:
    expected_world: str
    required_topics: tuple[str, ...]
    container_name: str
    compose_service: str
    container_userdata_root: str
    data_relative_root: str
    storage_id: str
    recorder_startup_timeout_s: float
    recorder_shutdown_timeout_s: float
    settle_duration_s: float
    pilot_episode_count: int
    minimum_free_bytes: int
    minimum_camera_messages: int

    @classmethod
    def load(cls, path: str | Path) -> "CollectorConfig":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load collector configuration: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("collector configuration root must be a JSON object")
        payload = dict(payload)
        if isinstance(payload.get("required_topics"), list):
            payload["required_topics"] = tuple(payload["required_topics"])
        try:
            config = cls(**payload)
        except TypeError as exc:
            raise ValueError(f"invalid collector configuration fields: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.expected_world, str) or not self.expected_world:
            raise ValueError("expected_world must be a non-empty string")
        if not self.required_topics or any(
            not isinstance(topic, str) or not topic.startswith("/") for topic in self.required_topics
        ):
            raise ValueError("required_topics must contain absolute ROS topic names")
        if len(set(self.required_topics)) != len(self.required_topics):
            raise ValueError("required_topics must not contain duplicates")
        required_v1 = {
            "/camera/image_raw", "/steering", "/speed", "/cmd_vel",
            "/odom", "/clock", "/tf", "/tf_static",
        }
        if set(self.required_topics) != required_v1:
            raise ValueError("required_topics must exactly match the Rosbag Collector V1 baseline")
        for name in ("container_name", "compose_service", "container_userdata_root", "data_relative_root"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if not self.container_userdata_root.startswith("/"):
            raise ValueError("container_userdata_root must be absolute")
        relative = PurePosixPath(self.data_relative_root)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("data_relative_root must be a safe relative path")
        if self.storage_id not in {"mcap", "sqlite3"}:
            raise ValueError("storage_id must be mcap or sqlite3")
        for name in ("recorder_startup_timeout_s", "recorder_shutdown_timeout_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.settle_duration_s, (int, float)) or isinstance(self.settle_duration_s, bool) \
                or not math.isfinite(self.settle_duration_s) or self.settle_duration_s < 0:
            raise ValueError("settle_duration_s must be finite and nonnegative")
        for name in ("pilot_episode_count", "minimum_free_bytes", "minimum_camera_messages"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class BagInfo:
    duration_s: float
    total_messages: int
    topic_counts: dict[str, int]
    raw_output: str = ""


@dataclass(frozen=True)
class RecorderHandle:
    episode_id: str
    host_episode_path: Path
    host_bag_path: Path
    container_episode_path: str
    container_bag_path: str
    container_pid_path: str
    container_log_path: str


@dataclass(frozen=True)
class RecorderStopResult:
    graceful: bool
    orphaned: bool
    detail: str | None = None


class Backend(Protocol):
    host_userdata_root: Path

    def preflight(self, required_topics: Sequence[str]) -> dict[str, str]: ...
    def start_recorder(self, episode_id: str, required_topics: Sequence[str]) -> RecorderHandle: ...
    def stop_recorder(self, handle: RecorderHandle) -> RecorderStopResult: ...
    def bag_info(self, handle: RecorderHandle) -> BagInfo: ...


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_topic_list(output: str) -> dict[str, str]:
    topics: dict[str, str] = {}
    for line in output.splitlines():
        match = re.fullmatch(r"\s*(/\S+)\s+\[([^]]+)]\s*", line)
        if match:
            topics[match.group(1)] = match.group(2)
    return topics


def parse_bag_info(output: str) -> BagInfo:
    duration = re.search(r"^Duration:\s+([0-9]+(?:\.[0-9]+)?)s\s*$", output, re.MULTILINE)
    messages = re.search(r"^Messages:\s+([0-9]+)\s*$", output, re.MULTILINE)
    if duration is None or messages is None:
        raise CollectorError("ros2 bag info output lacks duration or total message count")
    topic_counts: dict[str, int] = {}
    pattern = re.compile(r"Topic:\s+(\S+)\s+\|.*?\|\s+Count:\s+([0-9]+)\s+\|", re.MULTILINE)
    for match in pattern.finditer(output):
        topic_counts[match.group(1)] = int(match.group(2))
    if not topic_counts:
        raise CollectorError("ros2 bag info output contains no topic information")
    return BagInfo(float(duration.group(1)), int(messages.group(1)), topic_counts, output)


def verify_bag(info: BagInfo, required_topics: Sequence[str], minimum_camera_messages: int) -> None:
    if not math.isfinite(info.duration_s) or info.duration_s <= 0:
        raise CollectorError("bag duration is not positive")
    missing = sorted(set(required_topics) - set(info.topic_counts))
    if missing:
        raise CollectorError("bag is missing required topics: " + ", ".join(missing))
    empty = sorted(topic for topic in required_topics if info.topic_counts[topic] <= 0)
    if empty:
        raise CollectorError("required topics have zero messages: " + ", ".join(empty))
    if info.topic_counts["/camera/image_raw"] < minimum_camera_messages:
        raise CollectorError(
            f"camera message count {info.topic_counts['/camera/image_raw']} is below "
            f"minimum {minimum_camera_messages}"
        )


def directory_size(path: Path) -> int:
    if not path.is_dir():
        raise CollectorError(f"finalized bag directory does not exist: {path}")
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


class DockerRosBackend:
    def __init__(
        self,
        config: CollectorConfig,
        sim_root: Path,
        *,
        run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.sim_root = sim_root.expanduser().resolve()
        self._run = run
        self._monotonic = monotonic
        self._sleep = sleep
        self.host_userdata_root = self._discover_userdata_mount()
        expected = (self.sim_root / "userdata").resolve()
        if self.host_userdata_root != expected:
            raise CollectorError(
                f"container userdata mount source {self.host_userdata_root} does not match --sim-root {expected}"
            )
        self.host_data_root = self.host_userdata_root / config.data_relative_root
        self.container_data_root = str(PurePosixPath(config.container_userdata_root) / config.data_relative_root)

    def _command(self, args: Sequence[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = self._run(list(args), text=True, capture_output=True, check=False)
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise CollectorError(f"command failed ({' '.join(args[:3])}): {detail}")
        return result

    def _docker_shell(self, body: str, *, detached: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
        args = ["docker", "exec"]
        if detached:
            args.append("-d")
        args.extend([self.config.container_name, "bash", "-lc", body])
        return self._command(args, check=check)

    def _ros(self, command: str, *, check: bool = True) -> subprocess.CompletedProcess[str]:
        return self._docker_shell(ROS_SETUP + command, check=check)

    def _discover_userdata_mount(self) -> Path:
        result = self._command([
            "docker", "inspect", "--format", "{{json .Mounts}}", self.config.container_name,
        ])
        try:
            mounts = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise CollectorError(f"cannot parse docker mounts: {exc}") from exc
        matches = [item for item in mounts if item.get("Destination") == self.config.container_userdata_root]
        if len(matches) != 1 or matches[0].get("Type") != "bind" or matches[0].get("RW") is not True:
            raise CollectorError("expected exactly one writable bind mount for container userdata")
        return Path(matches[0]["Source"]).resolve()

    def preflight(self, required_topics: Sequence[str]) -> dict[str, str]:
        inspect = self._command([
            "docker", "inspect", "--format",
            "{{.State.Running}} {{.State.Health.Status}} {{index .Config.Labels \"com.docker.compose.service\"}}",
            self.config.container_name,
        ]).stdout.strip()
        if inspect != f"true healthy {self.config.compose_service}":
            raise CollectorError(f"unexpected container state/service: {inspect!r}")
        self._ros("ros2 bag record --help >/dev/null\nros2 bag info --help >/dev/null")
        topics = parse_topic_list(self._ros("ros2 topic list -t").stdout)
        missing = sorted(set(required_topics) - set(topics))
        if missing:
            raise CollectorError("live ROS graph lacks required topics: " + ", ".join(missing))
        return {topic: topics[topic] for topic in required_topics}

    def start_recorder(self, episode_id: str, required_topics: Sequence[str]) -> RecorderHandle:
        host_episode = self.host_data_root / episode_id
        if host_episode.exists():
            raise CollectorError(f"refusing to overwrite existing episode: {host_episode}")
        host_episode.mkdir(parents=True)
        container_episode = str(PurePosixPath(self.container_data_root) / episode_id)
        bag = str(PurePosixPath(container_episode) / "bag")
        pid = str(PurePosixPath(container_episode) / ".rosbag_pid")
        log = str(PurePosixPath(container_episode) / "recorder.log")
        handle = RecorderHandle(episode_id, host_episode, host_episode / "bag", container_episode, bag, pid, log)
        quoted_topics = " ".join(shlex.quote(topic) for topic in required_topics)
        body = ROS_SETUP + (
            f"echo $$ > {shlex.quote(pid)}\n"
            f"exec ros2 bag record --output {shlex.quote(bag)} --storage {shlex.quote(self.config.storage_id)} "
            f"--disable-keyboard-controls --topics {quoted_topics} > {shlex.quote(log)} 2>&1"
        )
        try:
            self._docker_shell(body, detached=True)
            deadline = self._monotonic() + self.config.recorder_startup_timeout_s
            while self._monotonic() < deadline:
                probe = self._docker_shell(
                    f"test -s {shlex.quote(pid)} && kill -0 \"$(cat {shlex.quote(pid)})\" "
                    f"&& test -d {shlex.quote(bag)}",
                    check=False,
                )
                if probe.returncode == 0:
                    return handle
                early = self._docker_shell(
                    f"test -s {shlex.quote(pid)} && ! kill -0 \"$(cat {shlex.quote(pid)})\"",
                    check=False,
                )
                if early.returncode == 0:
                    raise CollectorError("rosbag recorder exited during startup; see recorder.log")
                self._sleep(0.1)
            raise CollectorError("rosbag recorder startup timed out")
        except BaseException:
            self._stop_exact_pid(handle, signal="INT", tolerate_missing=True)
            raise

    def _alive(self, handle: RecorderHandle) -> bool:
        result = self._docker_shell(
            f"test -s {shlex.quote(handle.container_pid_path)} && "
            f"kill -0 \"$(cat {shlex.quote(handle.container_pid_path)})\"",
            check=False,
        )
        return result.returncode == 0

    def _stop_exact_pid(self, handle: RecorderHandle, *, signal: str, tolerate_missing: bool = False) -> bool:
        result = self._docker_shell(
            f"test -s {shlex.quote(handle.container_pid_path)} && "
            f"kill -{signal} \"$(cat {shlex.quote(handle.container_pid_path)})\"",
            check=False,
        )
        if result.returncode != 0 and not tolerate_missing:
            raise CollectorError(f"could not send SIG{signal} to scoped recorder PID")
        return result.returncode == 0

    def _wait_dead(self, handle: RecorderHandle, timeout_s: float) -> bool:
        deadline = self._monotonic() + timeout_s
        while self._monotonic() < deadline:
            if not self._alive(handle):
                return True
            self._sleep(0.1)
        return not self._alive(handle)

    def stop_recorder(self, handle: RecorderHandle) -> RecorderStopResult:
        if not self._alive(handle):
            return RecorderStopResult(False, False, "recorder exited before shutdown")
        if not self._stop_exact_pid(handle, signal="INT"):
            return RecorderStopResult(False, self._alive(handle), "failed to send SIGINT")
        if self._wait_dead(handle, self.config.recorder_shutdown_timeout_s):
            return RecorderStopResult(True, False)
        detail = "graceful recorder shutdown timed out; sent scoped SIGTERM"
        self._stop_exact_pid(handle, signal="TERM", tolerate_missing=True)
        if not self._wait_dead(handle, 2.0):
            detail += "; sent scoped SIGKILL"
            self._stop_exact_pid(handle, signal="KILL", tolerate_missing=True)
            self._wait_dead(handle, 1.0)
        return RecorderStopResult(False, self._alive(handle), detail)

    def bag_info(self, handle: RecorderHandle) -> BagInfo:
        result = self._ros(f"ros2 bag info {shlex.quote(handle.container_bag_path)}")
        return parse_bag_info(result.stdout)


def _topic_metrics(info: BagInfo) -> dict[str, dict[str, float | int]]:
    return {
        topic: {
            "message_count": count,
            "average_recorded_rate_hz": count / info.duration_s,
        }
        for topic, count in sorted(info.topic_counts.items())
    }


def _base_metadata(
    episode_id: str,
    config: CollectorConfig,
    expert_config: DriverConfig,
    expert_config_path: Path,
    git_commit: str,
    handle: RecorderHandle | None,
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "world": config.expected_world,
        "canonical_expert_config_path": str(expert_config_path),
        "canonical_expert_config_sha256": sha256_file(expert_config_path),
        "canonical_expert_config": asdict(expert_config),
        "physicar_e2e_git_commit": git_commit,
        "bag_host_path": str(handle.host_bag_path) if handle else None,
        "bag_container_path": handle.container_bag_path if handle else None,
        "required_topics": list(config.required_topics),
        "recording_start_utc": None,
        "expert_driving_start_utc": None,
        "expert_driving_end_utc": None,
        "recording_end_utc": None,
        "expert_result_metrics": None,
        "actual_topic_message_counts": {},
        "topic_metrics": {},
        "bag_duration_s": None,
        "bag_size_bytes": None,
        "result": "FAIL",
        "failure_reason": None,
        "recorder_graceful_shutdown": False,
        "recorder_orphaned": False,
    }


def collect_episode(
    episode_number: int,
    config: CollectorConfig,
    expert_config: DriverConfig,
    expert_config_path: Path,
    git_commit: str,
    backend: Backend,
    client: SimClient,
    result_path: Path,
    *,
    driver: Callable[..., dict[str, Any]] = run_driver,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    episode_id = f"episode_{episode_number:03d}"
    handle: RecorderHandle | None = None
    metadata = _base_metadata(episode_id, config, expert_config, expert_config_path, git_commit, handle)
    stop_result: RecorderStopResult | None = None
    try:
        initial = wait_after_reset(client, expert_config, False)
        sleeper(config.settle_duration_s)
        initial = preflight(client, expert_config, False)
        if shutil.disk_usage(backend.host_userdata_root).free < config.minimum_free_bytes:
            raise CollectorError("insufficient free space on simulator userdata filesystem")
        handle = backend.start_recorder(episode_id, config.required_topics)
        metadata.update({
            "bag_host_path": str(handle.host_bag_path),
            "bag_container_path": handle.container_bag_path,
            "recording_start_utc": utc_now(),
        })
        metadata["expert_driving_start_utc"] = utc_now()
        metrics = driver(client, expert_config, initial)
        metadata["expert_driving_end_utc"] = utc_now()
        metadata["expert_result_metrics"] = metrics
        if metrics.get("result") != "PASS":
            raise CollectorError(f"canonical expert failed: {metrics.get('failure') or metrics.get('result')}")
    except BaseException as exc:
        metadata["failure_reason"] = str(exc)
    finally:
        if handle is not None:
            try:
                stop_result = backend.stop_recorder(handle)
                metadata["recorder_graceful_shutdown"] = stop_result.graceful
                metadata["recorder_orphaned"] = stop_result.orphaned
                if not stop_result.graceful:
                    reason = stop_result.detail or "recorder did not shut down gracefully"
                    metadata["failure_reason"] = "; ".join(filter(None, [metadata["failure_reason"], reason]))
            except BaseException as exc:
                metadata["recorder_orphaned"] = True
                metadata["failure_reason"] = "; ".join(
                    filter(None, [metadata["failure_reason"], f"recorder cleanup failed: {exc}"])
                )
            metadata["recording_end_utc"] = utc_now()
        stop_errors = client.safe_stop()
        if stop_errors:
            metadata["failure_reason"] = "; ".join(
                filter(None, [metadata["failure_reason"], "final safe-stop failed: " + "; ".join(stop_errors)])
            )
    if handle is not None and stop_result is not None and stop_result.graceful:
        try:
            info = backend.bag_info(handle)
            verify_bag(info, config.required_topics, config.minimum_camera_messages)
            size = directory_size(handle.host_bag_path)
            metadata.update({
                "actual_topic_message_counts": dict(sorted(info.topic_counts.items())),
                "topic_metrics": _topic_metrics(info),
                "bag_duration_s": info.duration_s,
                "bag_size_bytes": size,
            })
        except BaseException as exc:
            metadata["failure_reason"] = "; ".join(
                filter(None, [metadata["failure_reason"], f"bag integrity failed: {exc}"])
            )
    metrics = metadata.get("expert_result_metrics") or {}
    if (
        metadata["failure_reason"] is None
        and metrics.get("result") == "PASS"
        and metrics.get("safe_stop_success") is True
        and metadata["recorder_graceful_shutdown"] is True
        and metadata["recorder_orphaned"] is False
        and metadata["bag_size_bytes"] is not None
    ):
        metadata["result"] = "PASS"
    elif metadata["failure_reason"] is None:
        metadata["failure_reason"] = "episode did not satisfy all success gates"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def summarize(episodes: Sequence[dict[str, Any]], requested_count: int) -> dict[str, Any]:
    sizes = [int(item["bag_size_bytes"]) for item in episodes if item.get("bag_size_bytes") is not None]
    durations = [float(item["bag_duration_s"]) for item in episodes if item.get("bag_duration_s")]
    total_size = sum(sizes)
    total_duration = sum(durations)
    mean_size = total_size / len(sizes) if sizes else None
    return {
        "collector": "rosbag_collector_v1",
        "requested_episode_count": requested_count,
        "completed_episode_count": len(episodes),
        "passed_episode_count": sum(item.get("result") == "PASS" for item in episodes),
        "episode_results": [{"episode_id": item["episode_id"], "result": item["result"]} for item in episodes],
        "bag_sizes_bytes": sizes,
        "total_bag_size_bytes": total_size,
        "mean_bag_size_bytes": mean_size,
        "mean_storage_rate_bytes_per_s": total_size / total_duration if total_duration else None,
        "projected_50_episode_storage_bytes": mean_size * 50 if mean_size is not None else None,
        "result": "PASS" if len(episodes) == requested_count and all(
            item.get("result") == "PASS" for item in episodes
        ) else "FAIL",
        "failure_reason": next(
            (item.get("failure_reason") for item in episodes if item.get("result") != "PASS"), None
        ),
        "generated_utc": utc_now(),
    }


def collect_sequence(
    count: int,
    config: CollectorConfig,
    expert_config: DriverConfig,
    expert_config_path: Path,
    git_commit: str,
    backend: Backend,
    client: SimClient,
    results_dir: Path,
    **episode_kwargs: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for number in range(1, count + 1):
        result = collect_episode(
            number, config, expert_config, expert_config_path, git_commit, backend, client,
            results_dir / f"episode_{number:03d}.json", **episode_kwargs,
        )
        episodes.append(result)
        if result["result"] != "PASS":
            break
    return episodes, summarize(episodes, count)


def git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=False,
    )
    if result.returncode != 0:
        raise CollectorError(f"cannot determine Git commit: {result.stderr.strip()}")
    return result.stdout.strip()


def verify_environment(root: Path, sim_root: Path) -> str:
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "setup_lane_follow_environment_v1.py"),
            "--sim-root", str(sim_root.expanduser().resolve()),
            "--verify-only",
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise CollectorError(f"lane-follow environment verification failed: {result.stderr.strip()}")
    return result.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expert-config", type=Path, required=True)
    parser.add_argument("--sim-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    client: SimClient | None = None
    try:
        config = CollectorConfig.load(args.config)
        expert_config = DriverConfig.load(args.expert_config)
        if config.expected_world != expert_config.expected_world:
            raise CollectorError("collector and canonical expert expected_world differ")
        count = args.episodes if args.episodes is not None else config.pilot_episode_count
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise CollectorError("--episodes must be a positive integer")
        environment_diagnostics = verify_environment(root, args.sim_root)
        backend = DockerRosBackend(config, args.sim_root)
        topic_types = backend.preflight(config.required_topics)
        client = SimClient(expert_config.base_url, expert_config.api_timeout_s)
        initial_stop_errors = client.safe_stop()
        if initial_stop_errors:
            raise CollectorError("initial safe-stop failed: " + "; ".join(initial_stop_errors))
        initial = preflight(client, expert_config, False)
        preflight_result = {
            "result": "PASS", "world": initial.world, "route_length_m": initial.route.length,
            "route_points": initial.route_points, "cone_count": initial.cone_count,
            "topic_types": topic_types, "host_userdata_root": str(backend.host_userdata_root),
            "container_userdata_root": config.container_userdata_root,
            "environment_verification": environment_diagnostics,
        }
        if args.preflight_only:
            print(json.dumps(preflight_result, indent=2, sort_keys=True))
            return 0
        episodes, summary = collect_sequence(
            count, config, expert_config, args.expert_config.resolve(), git_commit(root), backend, client,
            args.results_dir,
        )
        summary["preflight"] = preflight_result
        summary_path = args.results_dir.parent / "rosbag_collector_v1_pilot_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["result"] == "PASS" else 1
    except KeyboardInterrupt:
        print("ERROR: interrupted by user", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    finally:
        if client is not None:
            errors = client.safe_stop()
            if errors:
                print("ERROR: final safe-stop failed: " + "; ".join(errors), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
