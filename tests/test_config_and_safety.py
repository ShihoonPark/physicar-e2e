from dataclasses import replace
import contextlib
import io
import unittest
from unittest.mock import patch

import _bootstrap  # noqa: F401
from physicar_e2e.expert_driver import (
    DriverConfig,
    PoseLivenessMonitor,
    Preflight,
    main,
    run_driver,
    wait_after_reset,
)
from physicar_e2e.route_geometry import ClosedRoute
from physicar_e2e.sim_client import SimClient, verify_control_schema


def valid_config():
    return DriverConfig(
        base_url="http://test", expected_world="world", wheelbase_m=0.18,
        max_steering_rad=0.349066, fixed_speed_mps=0.5, control_frequency_hz=15,
        lookahead_m=0.45, start_gate_radius_m=0.3, minimum_lap_progress_fraction=0.9,
        off_track_margin_m=0.05, off_track_grace_s=0.5, api_timeout_s=0.5,
        pose_stale_timeout_s=0.75, pose_motion_translation_threshold_m=0.005,
        pose_motion_yaw_threshold_rad=0.01, maximum_runtime_s=1.0, closed_route_tolerance_m=0.15,
        spawn_route_tolerance_m=0.5, minimum_route_points=3, maximum_progress_jump_m=1.0,
        world_check_interval_s=1.0, reset_wait_timeout_s=1.0,
    )


class ConfigTests(unittest.TestCase):
    def test_invalid_wheelbase_rejected(self):
        with self.assertRaises(ValueError):
            replace(valid_config(), wheelbase_m=0).validate()

    def test_invalid_frequency_rejected(self):
        with self.assertRaises(ValueError):
            replace(valid_config(), control_frequency_hz=-1).validate()

    def test_malformed_fixed_speed_rejected_as_value_error(self):
        for value in (None, "0.5", True, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(valid_config(), fixed_speed_mps=value).validate()

    def test_invalid_steering_limit_rejected(self):
        with self.assertRaises(ValueError):
            replace(valid_config(), max_steering_rad=0).validate()
        with self.assertRaises(ValueError):
            replace(valid_config(), max_steering_rad=2).validate()

    def test_non_finite_off_track_margin_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(valid_config(), off_track_margin_m=value).validate()

    def test_non_finite_off_track_grace_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(valid_config(), off_track_grace_s=value).validate()

    def test_non_finite_minimum_route_points_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(valid_config(), minimum_route_points=value).validate()

    def test_invalid_pose_freshness_configuration_rejected(self):
        names = (
            "pose_stale_timeout_s",
            "pose_motion_translation_threshold_m",
            "pose_motion_yaw_threshold_rad",
        )
        for name in names:
            for value in (float("nan"), float("inf"), float("-inf"), 0.0, -0.01):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    replace(valid_config(), **{name: value}).validate()


class PoseLivenessTests(unittest.TestCase):
    @staticmethod
    def pose(x=0.0, y=0.0, yaw=0.0):
        return {"x": x, "y": y, "yaw": yaw}

    def test_changing_pose_remains_live(self):
        monitor = PoseLivenessMonitor(0.75, 0.005, 0.01)
        monitor.update(self.pose(), 1.0, 0.0, motion_commanded=True)
        for index in range(1, 11):
            monitor.update(
                self.pose(x=index * 0.01),
                1.0 + index * 0.1,
                index * 0.1,
                motion_commanded=True,
            )

    def test_intentional_stop_disables_pose_motion_requirement(self):
        monitor = PoseLivenessMonitor(0.1, 0.005, 0.01)
        monitor.update(self.pose(), 1.0, 0.0, motion_commanded=True)
        monitor.update(self.pose(), 1.0, 10.0, motion_commanded=False)
        monitor.update(self.pose(), 1.0, 20.0, motion_commanded=False)

    def test_frozen_clock_is_distinguished(self):
        monitor = PoseLivenessMonitor(0.1, 0.005, 0.01)
        monitor.update(self.pose(), 1.0, 0.0, motion_commanded=True)
        with self.assertRaisesRegex(RuntimeError, "simulator clock did not advance"):
            monitor.update(self.pose(), 1.0, 0.11, motion_commanded=True)


class FakeFailureClient:
    def __init__(self):
        self.stop_calls = 0

    def status(self):
        return {"running": True, "switching": False, "current": "world"}

    def pose(self):
        raise RuntimeError("pose unavailable")

    def safe_stop(self):
        self.stop_calls += 1
        return []


class OrderedLifecycleClient:
    def __init__(self, *, reset_error=None, stop_results=None):
        self.events = []
        self.reset_error = reset_error
        self.stop_results = list(stop_results or [])

    def safe_stop(self):
        self.events.extend((("speed", 0.0), ("steering", 0.0)))
        return self.stop_results.pop(0) if self.stop_results else []

    def reset(self):
        self.events.append(("reset", None))
        if self.reset_error is not None:
            raise self.reset_error
        return {"success": True}


def reset_preflight_result():
    route = ClosedRoute([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    return Preflight("world", route, 5, 0, {}, {"x": 0, "y": 0, "yaw": 0})


class LifecycleTests(unittest.TestCase):
    def test_reset_stops_before_and_after_reset(self):
        client = OrderedLifecycleClient()
        with patch("physicar_e2e.expert_driver.preflight", return_value=reset_preflight_result()):
            wait_after_reset(client, valid_config(), False)
        self.assertEqual(
            client.events,
            [
                ("speed", 0.0), ("steering", 0.0),
                ("reset", None),
                ("speed", 0.0), ("steering", 0.0),
            ],
        )

    def test_failed_pre_reset_stop_prevents_reset(self):
        client = OrderedLifecycleClient(stop_results=[["speed stop failed"]])
        with self.assertRaisesRegex(RuntimeError, "pre-reset safe-stop failed"):
            wait_after_reset(client, valid_config(), False)
        self.assertNotIn(("reset", None), client.events)
        self.assertGreaterEqual(client.events.count(("speed", 0.0)), 2)

    def test_reset_failure_attempts_safe_stop(self):
        client = OrderedLifecycleClient(reset_error=RuntimeError("reset failed"))
        with self.assertRaisesRegex(RuntimeError, "reset failed"):
            wait_after_reset(client, valid_config(), False)
        reset_index = client.events.index(("reset", None))
        self.assertIn(("speed", 0.0), client.events[reset_index + 1:])
        self.assertIn(("steering", 0.0), client.events[reset_index + 1:])

    def test_post_reset_preflight_failure_attempts_safe_stop(self):
        client = OrderedLifecycleClient()
        config = replace(valid_config(), reset_wait_timeout_s=1.0)
        with (
            patch("physicar_e2e.expert_driver.preflight", side_effect=RuntimeError("bad spawn")),
            patch("physicar_e2e.expert_driver.time.monotonic", side_effect=[0.0, 0.0, 2.0]),
            patch("physicar_e2e.expert_driver.time.sleep"),
        ):
            with self.assertRaisesRegex(RuntimeError, "bad spawn"):
                wait_after_reset(client, config, False)
        self.assertGreaterEqual(client.events.count(("speed", 0.0)), 3)
        self.assertEqual(client.events[-2:], [("speed", 0.0), ("steering", 0.0)])

    def test_ordinary_preflight_failure_attempts_safe_stop(self):
        client = OrderedLifecycleClient()
        with contextlib.redirect_stderr(io.StringIO()):
            with (
                patch("physicar_e2e.expert_driver.SimClient", return_value=client),
                patch("physicar_e2e.expert_driver.preflight", side_effect=RuntimeError("preflight failed")),
            ):
                result = main(["--config", "configs/expert_driver_v1.json", "--preflight-only"])
        self.assertEqual(result, 2)
        self.assertGreaterEqual(client.events.count(("speed", 0.0)), 2)
        self.assertEqual(client.events[-2:], [("speed", 0.0), ("steering", 0.0)])

    def test_malformed_api_exception_is_controlled_and_stops(self):
        client = OrderedLifecycleClient()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with (
                patch("physicar_e2e.expert_driver.SimClient", return_value=client),
                patch("physicar_e2e.expert_driver.preflight", side_effect=TypeError("bad API shape")),
            ):
                result = main(["--config", "configs/expert_driver_v1.json", "--preflight-only"])
        self.assertEqual(result, 2)
        self.assertIn("ERROR: bad API shape", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertEqual(client.events[-2:], [("speed", 0.0), ("steering", 0.0)])

    def test_failed_initial_stop_prevents_preflight(self):
        client = OrderedLifecycleClient(stop_results=[["speed stop failed"]])
        preflight_mock = patch("physicar_e2e.expert_driver.preflight")
        with contextlib.redirect_stderr(io.StringIO()):
            with patch("physicar_e2e.expert_driver.SimClient", return_value=client), preflight_mock as checked:
                result = main(["--config", "configs/expert_driver_v1.json", "--preflight-only"])
        self.assertEqual(result, 2)
        checked.assert_not_called()
        self.assertGreaterEqual(client.events.count(("speed", 0.0)), 2)

    def test_successful_preflight_sends_no_nonzero_command(self):
        client = OrderedLifecycleClient()
        with contextlib.redirect_stdout(io.StringIO()):
            with (
                patch("physicar_e2e.expert_driver.SimClient", return_value=client),
                patch("physicar_e2e.expert_driver.preflight", return_value=reset_preflight_result()),
            ):
                result = main(["--config", "configs/expert_driver_v1.json", "--preflight-only"])
        self.assertEqual(result, 0)
        numeric_commands = [event for event in client.events if event[0] in ("speed", "steering")]
        self.assertTrue(numeric_commands)
        self.assertTrue(all(value == 0.0 for _, value in numeric_commands))


class SafetyTests(unittest.TestCase):
    def test_fast_frozen_pose_triggers_safe_stop(self):
        class FrozenPoseClient:
            def __init__(self):
                self.sim_time = 0.0
                self.nonzero_speed_calls = 0
                self.stop_calls = 0

            def status(self):
                return {"running": True, "switching": False, "current": "world"}

            def pose(self):
                return {"x": 0.0, "y": -0.5, "yaw": 0.0}

            def clock(self):
                self.sim_time += 0.001
                return {"sim_time": self.sim_time, "paused": False}

            def command_steering(self, value):
                return {"success": True}

            def command_speed(self, value):
                if value > 0:
                    self.nonzero_speed_calls += 1
                return {"success": True}

            def safe_stop(self):
                self.stop_calls += 1
                return []

        center = [(-0.5, -0.5), (2.5, -0.5), (2.5, 2.5), (-0.5, 2.5), (-0.5, -0.5)]
        inner = [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]
        outer = [(-1, -1), (3, -1), (3, 3), (-1, 3), (-1, -1)]
        route = ClosedRoute(center, inner, outer)
        initial = Preflight("world", route, 5, 0, {}, {"x": 0, "y": -0.5, "yaw": 0})
        config = replace(
            valid_config(),
            control_frequency_hz=1000.0,
            pose_stale_timeout_s=0.005,
            maximum_runtime_s=0.05,
        )
        client = FrozenPoseClient()
        metrics = run_driver(client, config, initial)
        self.assertEqual(metrics["result"], "FAIL")
        self.assertIn("pose did not change meaningfully", metrics["failure"])
        self.assertGreater(client.nonzero_speed_calls, 0)
        self.assertEqual(client.stop_calls, 1)
        self.assertTrue(metrics["safe_stop_success"])

    def test_safe_stop_attempted_on_runtime_failure(self):
        client = FakeFailureClient()
        route = ClosedRoute([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        initial = Preflight("world", route, 5, 0, {}, {"x": 0, "y": 0, "yaw": 0})
        metrics = run_driver(client, valid_config(), initial)
        self.assertEqual(metrics["result"], "FAIL")
        self.assertIn("pose unavailable", metrics["failure"])
        self.assertEqual(client.stop_calls, 1)
        self.assertTrue(metrics["safe_stop_success"])

    def test_safe_stop_attempts_both_commands(self):
        class StopClient(SimClient):
            def __init__(self):
                self.calls = []
            def command_speed(self, value):
                self.calls.append(("speed", value))
                raise RuntimeError("speed failure")
            def command_steering(self, value):
                self.calls.append(("steering", value))
                return {"success": True}
        client = StopClient()
        errors = client.safe_stop()
        self.assertEqual(client.calls, [("speed", 0.0), ("steering", 0.0)])
        self.assertEqual(len(errors), 1)


class SchemaTests(unittest.TestCase):
    def test_verified_control_schema(self):
        schema = {"paths": {}, "components": {"schemas": {}}}
        for path, name in (("/speed", "SpeedRequest"), ("/steering", "SteeringRequest")):
            schema["paths"][path] = {"post": {"requestBody": {"content": {"application/json": {
                "schema": {"$ref": f"#/components/schemas/{name}"}
            }}}}}
            schema["components"]["schemas"][name] = {
                "required": ["value"], "properties": {"value": {"type": "number"}}
            }
        verify_control_schema(schema)


if __name__ == "__main__":
    unittest.main()
