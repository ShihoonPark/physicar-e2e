from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
from physicar_e2e.expert_driver import DriverConfig
from physicar_e2e.rosbag_collector import (
    BagInfo,
    CollectorConfig,
    CollectorError,
    DockerRosBackend,
    RecorderHandle,
    RecorderStopResult,
    collect_episode,
    collect_sequence,
    parse_bag_info,
    parse_topic_list,
    summarize,
    verify_bag,
)


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "/camera/image_raw", "/steering", "/speed", "/cmd_vel",
    "/odom", "/clock", "/tf", "/tf_static",
)


def collector_config(**changes):
    base = CollectorConfig(
        expected_world="world", required_topics=REQUIRED, container_name="container",
        compose_service="sim", container_userdata_root="/userdata", data_relative_root="pilot",
        storage_id="mcap", recorder_startup_timeout_s=1.0, recorder_shutdown_timeout_s=1.0,
        settle_duration_s=0.0, pilot_episode_count=3, minimum_free_bytes=1,
        minimum_camera_messages=10,
    )
    return replace(base, **changes)


def expert_config():
    payload = json.loads((ROOT / "configs" / "expert_driver_v1.json").read_text())
    payload["expected_world"] = "world"
    return DriverConfig(**payload)


def valid_info(**counts):
    topics = {topic: 20 for topic in REQUIRED}
    topics["/camera/image_raw"] = 200
    topics.update(counts)
    return BagInfo(10.0, sum(topics.values()), topics)


class FakeClient:
    def __init__(self):
        self.stop_calls = 0

    def safe_stop(self):
        self.stop_calls += 1
        return []


class FakeBackend:
    def __init__(self, root, *, start_error=None, stop=None, info=None):
        self.host_userdata_root = Path(root)
        self.start_error = start_error
        self.stop = stop or RecorderStopResult(True, False)
        self.info = info or valid_info()
        self.started = []
        self.stopped = []

    def preflight(self, required_topics):
        return {topic: "type" for topic in required_topics}

    def start_recorder(self, episode_id, required_topics):
        self.started.append((episode_id, tuple(required_topics)))
        if self.start_error:
            raise self.start_error
        episode = self.host_userdata_root / episode_id
        bag = episode / "bag"
        bag.mkdir(parents=True)
        (bag / "data.mcap").write_bytes(b"x" * 100)
        return RecorderHandle(episode_id, episode, bag, f"/data/{episode_id}", f"/data/{episode_id}/bag",
                              f"/data/{episode_id}/pid", f"/data/{episode_id}/log")

    def stop_recorder(self, handle):
        self.stopped.append(handle.episode_id)
        return self.stop

    def bag_info(self, handle):
        return self.info


def passing_metrics():
    return {"result": "PASS", "failure": None, "safe_stop_success": True, "elapsed_s": 10.0}


class ConfigAndParsingTests(unittest.TestCase):
    def test_config_load_and_required_topic_construction(self):
        config = CollectorConfig.load(ROOT / "configs" / "rosbag_collector_v1.json")
        self.assertEqual(config.required_topics, REQUIRED)
        self.assertEqual(config.pilot_episode_count, 3)

    def test_config_validation_rejects_wrong_topics_and_bad_values(self):
        for config in (
            replace(collector_config(), required_topics=("/camera/image_raw",)),
            replace(collector_config(), required_topics=REQUIRED + ("/camera/image_raw",)),
            replace(collector_config(), recorder_startup_timeout_s=0),
            replace(collector_config(), pilot_episode_count=True),
            replace(collector_config(), data_relative_root="../escape"),
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()

    def test_topic_list_parsing(self):
        parsed = parse_topic_list("/camera/image_raw [sensor_msgs/msg/Image]\n/speed [std_msgs/msg/Float64]\n")
        self.assertEqual(parsed["/camera/image_raw"], "sensor_msgs/msg/Image")

    def test_installed_jazzy_bag_info_parsing_shape(self):
        output = """Files:             bag_0.mcap
Bag size:          1.0 MiB
Storage id:        mcap
Duration:          10.500s
Messages:          210
Topic information: Topic: /camera/image_raw | Type: sensor_msgs/msg/Image | Count: 200 | Serialization Format: cdr
                   Topic: /steering | Type: std_msgs/msg/Float64 | Count: 10 | Serialization Format: cdr
"""
        info = parse_bag_info(output)
        self.assertEqual(info.duration_s, 10.5)
        self.assertEqual(info.topic_counts["/camera/image_raw"], 200)

    def test_missing_required_topic_fails(self):
        info = valid_info()
        del info.topic_counts["/tf_static"]
        with self.assertRaisesRegex(CollectorError, "missing required"):
            verify_bag(info, REQUIRED, 10)

    def test_zero_message_required_topic_fails(self):
        with self.assertRaisesRegex(CollectorError, "zero messages"):
            verify_bag(valid_info(**{"/speed": 0}), REQUIRED, 10)

    def test_insubstantial_camera_fails(self):
        with self.assertRaisesRegex(CollectorError, "camera message count"):
            verify_bag(valid_info(**{"/camera/image_raw": 9}), REQUIRED, 10)

    def test_valid_bag_info_passes(self):
        verify_bag(valid_info(), REQUIRED, 10)


class DockerProcessTests(unittest.TestCase):
    @staticmethod
    def completed(code=0, stdout="", stderr=""):
        return subprocess.CompletedProcess([], code, stdout, stderr)

    def make_backend(self, temporary, responses=None):
        responses = list(responses or [])

        def run(args, **kwargs):
            if args[:3] == ["docker", "inspect", "--format"]:
                mounts = [{"Destination": "/userdata", "Source": temporary, "Type": "bind", "RW": True}]
                return self.completed(stdout=json.dumps(mounts))
            if responses:
                return responses.pop(0)
            return self.completed()

        sim_root = Path(temporary).parent
        Path(temporary).mkdir()
        return DockerRosBackend(collector_config(), sim_root, run=run, monotonic=Mock(return_value=0.0))

    def test_episode_path_creation_and_refusal_to_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            userdata = Path(root) / "userdata"
            backend = self.make_backend(str(userdata))
            # Detached launch, then successful alive/directory probe.
            backend._monotonic = Mock(side_effect=[0.0, 0.0])
            handle = backend.start_recorder("episode_001", REQUIRED)
            self.assertTrue(handle.host_episode_path.is_dir())
            with self.assertRaisesRegex(CollectorError, "refusing to overwrite"):
                backend.start_recorder("episode_001", REQUIRED)

    def test_graceful_stop_signals_scoped_pid(self):
        backend = object.__new__(DockerRosBackend)
        backend.config = collector_config()
        backend._alive = Mock(side_effect=[True, False])
        backend._stop_exact_pid = Mock(return_value=True)
        backend._wait_dead = Mock(return_value=True)
        result = backend.stop_recorder(Mock())
        self.assertTrue(result.graceful)
        backend._stop_exact_pid.assert_called_once_with(unittest.mock.ANY, signal="INT")

    def test_shutdown_timeout_reports_and_forces_only_scoped_pid(self):
        backend = object.__new__(DockerRosBackend)
        backend.config = collector_config()
        backend._alive = Mock(side_effect=[True, False])
        backend._stop_exact_pid = Mock(return_value=True)
        backend._wait_dead = Mock(side_effect=[False, True])
        result = backend.stop_recorder(Mock())
        self.assertFalse(result.graceful)
        self.assertFalse(result.orphaned)
        signals = [call.kwargs["signal"] for call in backend._stop_exact_pid.call_args_list]
        self.assertEqual(signals, ["INT", "TERM"])


class EpisodeLifecycleTests(unittest.TestCase):
    def run_episode(self, backend, result_path, driver=passing_metrics):
        client = FakeClient()
        expert = expert_config()
        config_path = ROOT / "configs" / "expert_driver_v1.json"
        with (
            patch("physicar_e2e.rosbag_collector.wait_after_reset", return_value=Mock()),
            patch("physicar_e2e.rosbag_collector.preflight", return_value=Mock()),
            patch("physicar_e2e.rosbag_collector.shutil.disk_usage", return_value=Mock(free=10**12)),
        ):
            result = collect_episode(
                1, collector_config(), expert, config_path, "abc", backend, client, result_path,
                driver=lambda *args: driver(), sleeper=lambda _: None,
            )
        return result, client

    def test_recorder_startup_failure_prevents_expert_launch(self):
        with tempfile.TemporaryDirectory() as root:
            driver = Mock(return_value=passing_metrics())
            backend = FakeBackend(root, start_error=CollectorError("startup failed"))
            result, _ = self.run_episode(backend, Path(root) / "result.json", driver=driver)
            self.assertEqual(result["result"], "FAIL")
            driver.assert_not_called()
            self.assertEqual(backend.stopped, [])

    def test_expert_failure_stops_and_finalizes_recorder(self):
        with tempfile.TemporaryDirectory() as root:
            backend = FakeBackend(root)
            result, client = self.run_episode(
                backend, Path(root) / "result.json",
                driver=lambda: {"result": "FAIL", "failure": "off track", "safe_stop_success": True},
            )
            self.assertEqual(result["result"], "FAIL")
            self.assertEqual(backend.stopped, ["episode_001"])
            self.assertGreater(client.stop_calls, 0)

    def test_shutdown_timeout_makes_episode_fail_without_orphan(self):
        with tempfile.TemporaryDirectory() as root:
            backend = FakeBackend(root, stop=RecorderStopResult(False, False, "shutdown timed out"))
            result, _ = self.run_episode(backend, Path(root) / "result.json")
            self.assertEqual(result["result"], "FAIL")
            self.assertIn("shutdown timed out", result["failure_reason"])
            self.assertFalse(result["recorder_orphaned"])

    def test_valid_episode_writes_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "result.json"
            result, _ = self.run_episode(FakeBackend(root), path)
            self.assertEqual(result["result"], "PASS")
            saved = json.loads(path.read_text())
            self.assertEqual(saved["actual_topic_message_counts"]["/camera/image_raw"], 200)
            self.assertEqual(saved["bag_size_bytes"], 100)
            self.assertTrue(saved["recorder_graceful_shutdown"])

    def test_failed_episode_stops_sequence_and_no_fourth_when_three(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []

            def fake_collect(number, *args, **kwargs):
                calls.append(number)
                return {"episode_id": f"episode_{number:03d}", "result": "FAIL", "failure_reason": "x",
                        "bag_size_bytes": None, "bag_duration_s": None}

            with patch("physicar_e2e.rosbag_collector.collect_episode", side_effect=fake_collect):
                episodes, summary = collect_sequence(
                    3, collector_config(), expert_config(), ROOT / "configs" / "expert_driver_v1.json",
                    "abc", FakeBackend(root), FakeClient(), Path(root),
                )
            self.assertEqual(calls, [1])
            self.assertEqual(len(episodes), 1)
            self.assertEqual(summary["result"], "FAIL")

    def test_exactly_three_successes_never_launches_fourth(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []

            def fake_collect(number, *args, **kwargs):
                calls.append(number)
                return {"episode_id": f"episode_{number:03d}", "result": "PASS", "failure_reason": None,
                        "bag_size_bytes": 100, "bag_duration_s": 10.0}

            with patch("physicar_e2e.rosbag_collector.collect_episode", side_effect=fake_collect):
                episodes, summary = collect_sequence(
                    3, collector_config(), expert_config(), ROOT / "configs" / "expert_driver_v1.json",
                    "abc", FakeBackend(root), FakeClient(), Path(root),
                )
            self.assertEqual(calls, [1, 2, 3])
            self.assertEqual(summary["completed_episode_count"], 3)
            self.assertEqual(summary["result"], "PASS")

    def test_summary_storage_aggregation_and_projection(self):
        episodes = [
            {"episode_id": "episode_001", "result": "PASS", "failure_reason": None,
             "bag_size_bytes": 100, "bag_duration_s": 10},
            {"episode_id": "episode_002", "result": "PASS", "failure_reason": None,
             "bag_size_bytes": 200, "bag_duration_s": 10},
            {"episode_id": "episode_003", "result": "PASS", "failure_reason": None,
             "bag_size_bytes": 300, "bag_duration_s": 10},
        ]
        result = summarize(episodes, 3)
        self.assertEqual(result["total_bag_size_bytes"], 600)
        self.assertEqual(result["mean_bag_size_bytes"], 200)
        self.assertEqual(result["mean_storage_rate_bytes_per_s"], 20)
        self.assertEqual(result["projected_50_episode_storage_bytes"], 10_000)


if __name__ == "__main__":
    unittest.main()
