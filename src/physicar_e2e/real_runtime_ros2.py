"""ROS 2 adapter for Real Runtime V1.

All ROS imports are intentionally inside ``build_node_class``/``main`` so the
inference core and its tests import on a host without ROS 2.  The adapter does
not guess a traffic-light topic.  A verified future adapter must call
``authorize_green_from_verified_adapter``.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any, Sequence

from .real_runtime import (
    ControlDispatcher,
    PublisherError,
    RealRuntimeCore,
    RealRuntimeError,
    SafetyState,
    SelectedOnnxModel,
    load_config,
    validate_config,
)


def ros2_available() -> bool:
    try:
        import rclpy  # noqa: F401
        import sensor_msgs.msg  # noqa: F401
        import std_msgs.msg  # noqa: F401
    except ImportError:
        return False
    return True


def build_node_class():
    """Create the node class only after ROS 2 is known to be installed."""
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image
    from std_msgs.msg import Float64

    class RealRuntimeNode(Node):
        def __init__(
            self,
            config: dict[str, Any],
            *,
            publish_control: bool,
            development_start_bypass: bool,
        ) -> None:
            super().__init__("physicar_real_runtime_v1")
            effective = copy.deepcopy(config)
            effective["safety"]["publish_control"] = bool(publish_control)
            self.runtime_config = validate_config(effective)
            self.model = SelectedOnnxModel(self.runtime_config)
            self.core = RealRuntimeCore(self.runtime_config, self.model)
            self._frame_index = 0
            self._steering_publisher = None
            self._speed_publisher = None

            if publish_control:
                self._steering_publisher = self.create_publisher(
                    Float64, self.runtime_config["steering"]["topic"], 10
                )
                self._speed_publisher = self.create_publisher(
                    Float64, self.runtime_config["speed"]["topic"], 10
                )

            def publish_steering(value: float) -> None:
                if self._steering_publisher is None:
                    raise PublisherError("steering publisher is unavailable")
                message = Float64()
                message.data = float(value)
                self._steering_publisher.publish(message)

            def publish_speed(value: float) -> None:
                if self._speed_publisher is None:
                    raise PublisherError("speed publisher is unavailable")
                message = Float64()
                message.data = float(value)
                self._speed_publisher.publish(message)

            self.dispatcher = ControlDispatcher(
                publish_control=publish_control,
                steering_publish=publish_steering,
                speed_publish=publish_speed,
            )
            self.core.activate(development_bypass=development_start_bypass)
            self._camera_subscription = self.create_subscription(
                Image,
                self.runtime_config["camera"]["topic"],
                self._on_camera,
                qos_profile_sensor_data,
            )
            watchdog_period = min(
                float(self.runtime_config["safety"]["camera_timeout_s"]) / 4.0,
                0.025,
            )
            self._watchdog_timer = self.create_timer(watchdog_period, self._on_watchdog)
            self.get_logger().info(
                "Real Runtime V1 initialized: publish_control=%s physical_motion_authorized=%s"
                % (
                    publish_control,
                    self.runtime_config["speed"]["physical_motion_authorized"],
                )
            )
            if (
                self.runtime_config["start_gate"]["required"]
                and not development_start_bypass
                and self.runtime_config["start_gate"]["adapter"] is None
            ):
                self.get_logger().warning(
                    "WAITING_FOR_START: the real GREEN-signal adapter/topic is unverified; "
                    "no topic was invented and motion cannot be authorized"
                )

        def authorize_green_from_verified_adapter(self, green: bool) -> None:
            """Boundary for a separately verified traffic-light adapter."""
            step = self.core.set_green_authorized(bool(green))
            self._dispatch(step)

        def _dispatch(self, step) -> None:
            try:
                self.dispatcher.dispatch(step)
            except PublisherError as exc:
                self.core.mark_publisher_failure(str(exc))
                self.get_logger().error(str(exc))

        def _on_camera(self, message: Any) -> None:
            # Live temporal causality uses callback arrival time, not a possibly
            # stale or differently-clocked message header stamp.
            arrival_time_ns = time.monotonic_ns()
            step = self.core.process_camera(
                message,
                arrival_time_ns=arrival_time_ns,
                frame_id=self._frame_index,
            )
            self._frame_index += 1
            self._dispatch(step)
            if step.model_steering_rad is not None:
                self.get_logger().debug(
                    "model_steering_rad=%.9f published_steering_normalized=%.9f "
                    "speed_command_mps=%.6f"
                    % (
                        step.model_steering_rad,
                        step.published_steering_normalized,
                        step.speed_command_mps,
                    )
                )
            elif step.state == SafetyState.FAULT:
                self.get_logger().error(step.reason)

        def _on_watchdog(self) -> None:
            step = self.core.check_watchdog(now_ns=time.monotonic_ns())
            if step is not None:
                self._dispatch(step)
                self.get_logger().error(step.reason)
            if (
                self.dispatcher.publish_control
                and self.core.state == SafetyState.RUNNING
                and not self.dispatcher.liveness_ok(
                    time.monotonic_ns(),
                    float(self.runtime_config["safety"]["publisher_liveness_timeout_s"]),
                )
            ):
                fault = self.core.mark_publisher_failure("control dispatcher liveness timeout")
                self._dispatch(fault)

        def safe_shutdown(self) -> None:
            self._dispatch(self.core.stop("ROS 2 node shutdown"))

    return RealRuntimeNode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--publish-control",
        action="store_true",
        help="explicitly opt in to /steering and /speed publication (default: no publishers)",
    )
    parser.add_argument(
        "--development-start-bypass",
        action="store_true",
        help="bench/dry-run bypass; rejected when --publish-control is enabled",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        config = load_config(arguments.config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 2
    requested_bypass = bool(
        arguments.development_start_bypass or config["start_gate"]["development_bypass"]
    )
    if arguments.publish_control and requested_bypass:
        print(
            "ERROR: development start bypass cannot be combined with control publication",
            file=__import__("sys").stderr,
        )
        return 2
    if not ros2_available():
        print(
            "ERROR: ROS 2 Python packages are unavailable; use real_runtime.py for offline replay",
            file=__import__("sys").stderr,
        )
        return 2

    import rclpy

    node = None
    try:
        rclpy.init(args=None)
        node_class = build_node_class()
        node = node_class(
            config,
            publish_control=arguments.publish_control,
            development_start_bypass=requested_bypass,
        )
        rclpy.spin(node)
        return 0
    except KeyboardInterrupt:
        return 0
    except RealRuntimeError as exc:
        print(f"ERROR: {exc}", file=__import__("sys").stderr)
        return 2
    finally:
        if node is not None:
            node.safe_shutdown()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
