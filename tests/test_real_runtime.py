import copy
import importlib
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from physicar_e2e.pilotnet import RGB_TO_YUV_BT601, preprocess_rgb
from physicar_e2e.real_dataset import load_config as load_dataset_config
from physicar_e2e.real_dataset import preprocess_real_camera_image
from physicar_e2e.real_runtime import (
    MAX_ADJACENT_GAP_NS,
    SAFE_SPEED_MPS,
    SELECTED_ONNX_SHA256,
    BufferStatus,
    CausalFrameBuffer,
    ControlDispatcher,
    ImageContractError,
    RealRuntimeCore,
    SafetyState,
    SelectedOnnxModel,
    audit_selected_model,
    load_config,
    preprocess_camera_message,
    replay_bag,
    resize_real_camera_rgb,
    steering_command_from_radians,
    unpack_rgb8_image,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "real_runtime_v1.json"
DATASET_CONFIG_PATH = ROOT / "configs" / "real_dataset_v1.json"
SELECTED_ONNX = Path(
    "/home/a/physicar-e2e-artifacts/real_temporal_pilotnet_v1/selected/"
    "real_temporal_pilotnet_v1_selected.onnx"
)
BAG_03 = Path("/home/a/output_bag/bag_03/bag_03_0.mcap")


def image_message(*, width=480, height=360, encoding="rgb8", step=1440, data=None):
    if data is None:
        row = bytes((index % 251 for index in range(step)))
        data = row * height
    return SimpleNamespace(width=width, height=height, encoding=encoding, step=step, data=data)


class ConstantModel:
    def __init__(self, value=0.1):
        self.value = value
        self.calls = []

    def infer(self, temporal_tensor):
        self.calls.append(np.asarray(temporal_tensor).copy())
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class RuntimeConfigAndModelAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG_PATH)

    def test_selected_onnx_identity_and_contract(self):
        audit = audit_selected_model(self.config)
        self.assertEqual(audit["result"], "PASS")
        self.assertEqual(audit["selected_model"], "REAL-SCRATCH-V1")
        self.assertEqual(audit["onnx"]["sha256_observed"], SELECTED_ONNX_SHA256)
        self.assertEqual(audit["onnx"]["input_shape"], ["N", 9, 66, 200])
        self.assertEqual(audit["onnx"]["output_shape"], ["N", 1])
        self.assertEqual(audit["onnx"]["parameter_count"], 255_819)
        self.assertFalse(audit["real_vehicle_success_claimed"])

    def test_default_deployment_is_non_publishing_non_moving_and_gate_required(self):
        self.assertFalse(self.config["safety"]["publish_control"])
        self.assertFalse(self.config["speed"]["physical_motion_authorized"])
        self.assertTrue(self.config["start_gate"]["required"])
        self.assertFalse(self.config["start_gate"]["development_bypass"])
        self.assertIsNone(self.config["start_gate"]["topic"])
        self.assertIsNone(self.config["start_gate"]["adapter"])

    def test_speed_is_not_a_neural_input(self):
        self.assertFalse(self.config["model_contract"]["speed_is_neural_input"])
        self.assertFalse(self.config["speed"]["neural_input"])
        self.assertEqual(self.config["model_contract"]["input_shape"], [1, 9, 66, 200])

    def test_confirmed_real_control_interface_is_direct_only(self):
        interface = self.config["control_interface"]
        self.assertEqual(interface["status"], "CONFIRMED_REAL_VEHICLE_DIRECT_TOPICS")
        self.assertEqual(interface["routing"], "direct")
        self.assertEqual(
            (interface["steering_topic"], interface["steering_type"]),
            ("/steering", "std_msgs/msg/Float64"),
        )
        self.assertEqual(
            interface["steering_conversion"],
            "steering_normalized = steering_rad / 0.35",
        )
        self.assertEqual(
            (interface["speed_topic"], interface["speed_type"]),
            ("/speed", "std_msgs/msg/Float64"),
        )
        self.assertEqual(
            interface["forbidden_topics"],
            ["/teleop/steering", "/teleop/speed", "/cmd_vel"],
        )


class RealCameraPreprocessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_config = load_dataset_config(DATASET_CONFIG_PATH)

    def test_exact_480x360_rgb8_validation(self):
        valid = unpack_rgb8_image(image_message())
        self.assertEqual(valid.shape, (360, 480, 3))
        self.assertEqual(valid.dtype, np.uint8)
        for message in (
            image_message(width=479),
            image_message(height=359),
            image_message(encoding="bgr8"),
            image_message(step=1439, data=bytes(1439 * 360)),
            image_message(data=bytes(1440 * 360 - 1)),
        ):
            with self.subTest(message=message):
                with self.assertRaises(ImageContractError):
                    unpack_rgb8_image(message)

    def test_row_padding_is_ignored_without_channel_or_pixel_shift(self):
        packed_row = bytes([11, 22, 33] * 480)
        padded_row = packed_row + bytes([240, 241, 242, 243])
        rgb = unpack_rgb8_image(image_message(step=1444, data=padded_row * 360))
        self.assertTrue(np.all(rgb == np.asarray([11, 22, 33], dtype=np.uint8)))

    def test_real_roi_is_y80_to_360_with_no_horizontal_crop(self):
        rgb = np.zeros((360, 480, 3), dtype=np.uint8)
        rgb[:80, :, :] = (255, 0, 0)
        rgb[80:, 0, :] = (0, 255, 0)
        rgb[80:, -1, :] = (0, 0, 255)
        result = resize_real_camera_rgb(rgb)
        self.assertEqual(result.shape, (66, 200, 3))
        self.assertFalse(np.any(result[:, :, 0] == 255))
        self.assertGreater(int(result[32, 0, 1]), 0)
        self.assertGreater(int(result[32, -1, 2]), 0)

    def test_resize_matches_real_dataset_extractor_exactly(self):
        rng = np.random.default_rng(20260826)
        raw = rng.integers(0, 256, size=(360, 480, 3), dtype=np.uint8)
        message = image_message(data=raw.tobytes())
        canonical_image = preprocess_real_camera_image(message, self.dataset_config)
        canonical_rgb = np.asarray(canonical_image, dtype=np.uint8)
        canonical_image.close()
        self.assertTrue(np.array_equal(resize_real_camera_rgb(raw), canonical_rgb))
        self.assertTrue(
            np.array_equal(preprocess_camera_message(message), preprocess_rgb(canonical_rgb))
        )

    def test_exact_rgb_to_yuv_normalization_path(self):
        raw = np.zeros((360, 480, 3), dtype=np.uint8)
        raw[80:, :, 0] = 255
        tensor = preprocess_camera_message(image_message(data=raw.tobytes()))
        rgb_unit = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        yuv = rgb_unit @ RGB_TO_YUV_BT601.T
        yuv[1:] += np.float32(0.5)
        expected = (yuv - np.float32(0.5)) * np.float32(2.0)
        self.assertTrue(np.allclose(tensor[:, 20, 20], expected, rtol=0.0, atol=1e-7))
        self.assertEqual(tensor.shape, (3, 66, 200))
        self.assertEqual(tensor.dtype, np.float32)


class TemporalBufferTests(unittest.TestCase):
    def setUp(self):
        self.frame_a = np.zeros((3, 66, 200), dtype=np.float32)
        self.frame_b = np.ones((3, 66, 200), dtype=np.float32)
        self.frame_c = np.full((3, 66, 200), 2, dtype=np.float32)

    def test_three_genuine_frames_in_t_minus_2_to_t_order_without_padding(self):
        buffer = CausalFrameBuffer()
        self.assertEqual(buffer.append(1_000_000_000, "a", self.frame_a).status, BufferStatus.WARMING)
        with self.assertRaises(Exception):
            buffer.tensor()
        self.assertEqual(buffer.append(1_060_000_000, "b", self.frame_b).status, BufferStatus.WARMING)
        self.assertEqual(buffer.append(1_120_000_000, "c", self.frame_c).status, BufferStatus.READY)
        value = buffer.tensor()
        self.assertTrue(np.array_equal(value[:3], self.frame_a))
        self.assertTrue(np.array_equal(value[3:6], self.frame_b))
        self.assertTrue(np.array_equal(value[6:9], self.frame_c))

    def test_exactly_120ms_is_valid_and_strictly_greater_resets(self):
        buffer = CausalFrameBuffer()
        buffer.append(0, "a", self.frame_a)
        accepted = buffer.append(MAX_ADJACENT_GAP_NS, "b", self.frame_b)
        self.assertEqual(accepted.status, BufferStatus.WARMING)
        reset = buffer.append(MAX_ADJACENT_GAP_NS * 2 + 1, "c", self.frame_c)
        self.assertEqual(reset.status, BufferStatus.RESET_GAP)
        self.assertEqual(buffer.size, 1)
        self.assertEqual(buffer.timestamps_ns, (MAX_ADJACENT_GAP_NS * 2 + 1,))

    def test_nonincreasing_and_duplicate_frame_ids_invalidate_history(self):
        buffer = CausalFrameBuffer()
        buffer.append(100, "a", self.frame_a)
        self.assertEqual(buffer.append(100, "b", self.frame_b).status, BufferStatus.INVALID_ORDER)
        self.assertEqual(buffer.size, 0)
        buffer.append(200, "a", self.frame_a)
        self.assertEqual(buffer.append(201, "a", self.frame_b).status, BufferStatus.DUPLICATE_FRAME)
        self.assertEqual(buffer.size, 0)


class SteeringConversionTests(unittest.TestCase):
    def test_radians_are_bounded_then_divided_by_point_35_once(self):
        command = steering_command_from_radians(0.175)
        self.assertEqual(command.model_steering_rad, 0.175)
        self.assertEqual(command.bounded_steering_rad, 0.175)
        self.assertEqual(command.published_steering_normalized, 0.5)
        self.assertFalse(command.saturated)
        self.assertNotEqual(command.published_steering_normalized, 0.175 * 0.35)

    def test_left_positive_right_negative_and_normalized_clamp(self):
        left = steering_command_from_radians(0.7)
        right = steering_command_from_radians(-0.7)
        self.assertEqual(left.bounded_steering_rad, 0.35)
        self.assertEqual(left.published_steering_normalized, 1.0)
        self.assertEqual(right.bounded_steering_rad, -0.35)
        self.assertEqual(right.published_steering_normalized, -1.0)
        self.assertTrue(left.saturated and right.saturated)

    def test_nan_and_inf_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaises(Exception):
                    steering_command_from_radians(value)


class SafetyStateMachineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG_PATH)
        cls.message = image_message()

    def core(self, model=None):
        return RealRuntimeCore(self.config, model or ConstantModel())

    def test_traffic_light_gate_waits_then_warms_then_runs(self):
        model = ConstantModel()
        core = self.core(model)
        core.activate(now_ns=1_000_000_000)
        self.assertEqual(core.state, SafetyState.WAITING_FOR_START)
        waiting = core.process_camera(self.message, arrival_time_ns=1_010_000_000, frame_id=0)
        self.assertTrue(waiting.safe_stop)
        self.assertEqual(len(model.calls), 0)
        core.set_green_authorized(True)
        first = core.process_camera(self.message, arrival_time_ns=1_020_000_000, frame_id=1)
        second = core.process_camera(self.message, arrival_time_ns=1_080_000_000, frame_id=2)
        third = core.process_camera(self.message, arrival_time_ns=1_140_000_000, frame_id=3)
        self.assertTrue(first.safe_stop and second.safe_stop)
        self.assertFalse(third.safe_stop)
        self.assertEqual(third.state, SafetyState.RUNNING)
        self.assertEqual(len(model.calls), 1)

    def test_development_bypass_only_warms_without_padding(self):
        model = ConstantModel()
        core = self.core(model)
        core.activate(development_bypass=True, now_ns=1_000_000_000)
        results = [
            core.process_camera(
                self.message, arrival_time_ns=1_000_000_000 + index * 60_000_000,
                frame_id=index,
            )
            for index in range(3)
        ]
        self.assertEqual([result.safe_stop for result in results], [True, True, False])
        self.assertEqual(core.statistics.warmup_frames, 2)
        self.assertEqual(core.statistics.predictions, 1)

    def test_dropout_safe_stops_and_rebuilds_from_current_plus_two_fresh_frames(self):
        model = ConstantModel()
        core = self.core(model)
        core.activate(development_bypass=True, now_ns=0)
        times = [0, 60_000_000, 120_000_000, 300_000_000, 360_000_000, 420_000_000]
        results = [
            core.process_camera(self.message, arrival_time_ns=stamp, frame_id=index)
            for index, stamp in enumerate(times)
        ]
        self.assertFalse(results[2].safe_stop)
        self.assertTrue(results[3].safe_stop)
        self.assertEqual(results[3].state, SafetyState.FAULT)
        self.assertTrue(results[4].safe_stop)
        self.assertFalse(results[5].safe_stop)
        self.assertEqual(core.statistics.dropouts, 1)
        self.assertEqual(core.statistics.invalid_temporal_buffers, 1)
        self.assertEqual(len(model.calls), 2)

    def test_camera_watchdog_clears_history_and_commands_safe_stop(self):
        core = self.core()
        core.activate(development_bypass=True, now_ns=1_000_000_000)
        core.process_camera(self.message, arrival_time_ns=1_000_000_000, frame_id=0)
        result = core.check_watchdog(now_ns=1_120_000_001)
        self.assertIsNotNone(result)
        self.assertEqual(result.state, SafetyState.FAULT)
        self.assertTrue(result.safe_stop)
        self.assertEqual(result.speed_command_mps, 0.0)
        self.assertEqual(result.published_steering_normalized, 0.0)
        self.assertEqual(core.buffer.size, 0)

    def test_invalid_image_onnx_exception_and_nonfinite_output_all_safe_stop(self):
        invalid_core = self.core()
        invalid_core.activate(development_bypass=True, now_ns=0)
        invalid = invalid_core.process_camera(
            image_message(encoding="bgr8"), arrival_time_ns=0, frame_id=0
        )
        self.assertTrue(invalid.safe_stop)
        self.assertEqual(invalid.state, SafetyState.FAULT)

        for model in (ConstantModel(RuntimeError("ORT failure")), ConstantModel(math.nan)):
            core = self.core(model)
            core.activate(development_bypass=True, now_ns=0)
            steps = [
                core.process_camera(self.message, arrival_time_ns=index * 60_000_000, frame_id=index)
                for index in range(3)
            ]
            self.assertTrue(steps[-1].safe_stop)
            self.assertEqual(steps[-1].speed_command_mps, SAFE_SPEED_MPS)
            self.assertEqual(steps[-1].published_steering_normalized, 0.0)
        self.assertEqual(core.statistics.nonfinite_model_outputs, 1)

    def test_safe_stop_never_holds_previous_prediction(self):
        core = self.core(ConstantModel(0.2))
        core.activate(development_bypass=True, now_ns=0)
        steps = [
            core.process_camera(self.message, arrival_time_ns=index * 60_000_000, frame_id=index)
            for index in range(3)
        ]
        self.assertGreater(steps[-1].published_steering_normalized, 0)
        stopped = core.stop()
        self.assertEqual(stopped.published_steering_normalized, 0.0)
        self.assertEqual(stopped.speed_command_mps, 0.0)
        self.assertEqual(stopped.state, SafetyState.SAFE_STOPPED)

    def test_default_runtime_speed_is_zero_and_not_passed_to_model(self):
        model = ConstantModel(0.1)
        core = self.core(model)
        core.activate(development_bypass=True, now_ns=0)
        result = None
        for index in range(3):
            result = core.process_camera(
                self.message, arrival_time_ns=index * 60_000_000, frame_id=index
            )
        self.assertEqual(result.speed_command_mps, 0.0)
        self.assertEqual(model.calls[0].shape, (9, 66, 200))


class PublisherSeparationTests(unittest.TestCase):
    def test_dry_run_dispatcher_never_invokes_control_callbacks(self):
        calls = []
        dispatcher = ControlDispatcher(
            publish_control=False,
            steering_publish=lambda value: calls.append(("steering", value)),
            speed_publish=lambda value: calls.append(("speed", value)),
        )
        config = load_config(CONFIG_PATH)
        core = RealRuntimeCore(config, ConstantModel())
        core.activate(development_bypass=True, now_ns=0)
        step = core.process_camera(image_message(), arrival_time_ns=0, frame_id=0)
        result = dispatcher.dispatch(step)
        self.assertFalse(result.published)
        self.assertEqual(calls, [])
        self.assertEqual(result.steering_messages, 0)
        self.assertEqual(result.speed_messages, 0)

    def test_core_and_ros_adapter_import_without_importing_ros(self):
        source = (ROOT / "src" / "physicar_e2e" / "real_runtime.py").read_text(encoding="utf-8")
        self.assertNotIn("import rclpy", source)
        before = set(sys.modules)
        adapter = importlib.import_module("physicar_e2e.real_runtime_ros2")
        newly_loaded = set(sys.modules) - before
        self.assertNotIn("rclpy", newly_loaded)
        self.assertFalse(adapter.ros2_available())

    def test_runtime_sources_have_no_simulator_or_training_surface(self):
        core_source = (ROOT / "src" / "physicar_e2e" / "real_runtime.py").read_text()
        ros_source = (ROOT / "src" / "physicar_e2e" / "real_runtime_ros2.py").read_text()
        self.assertNotIn("sim_client", core_source + ros_source)
        self.assertNotIn("train_model", core_source + ros_source)
        self.assertNotIn("/home/a/physicar-ai-sim-docker", core_source + ros_source)
        for forbidden_topic in ("/teleop/steering", "/teleop/speed", "/cmd_vel"):
            self.assertNotIn(forbidden_topic, ros_source)


@unittest.skipUnless(SELECTED_ONNX.is_file() and BAG_03.is_file(), "canonical Runtime V1 artifacts absent")
class BagReplayEquivalenceTests(unittest.TestCase):
    def test_complete_bag_03_matches_frozen_preprocessing_predictions_and_metrics(self):
        config = load_config(CONFIG_PATH)
        model = SelectedOnnxModel(config)
        result = replay_bag(config, model, "bag_03", verify_checkpoint_pipeline=True)
        self.assertEqual(result["result"], "PASS")
        self.assertEqual(result["source"]["camera_frame_count"], 454)
        self.assertEqual(result["statistics"]["warmup_frames"], 4)
        self.assertEqual(result["statistics"]["predictions"], 450)
        self.assertEqual(result["statistics"]["invalid_temporal_buffers"], 1)
        self.assertEqual(result["statistics"]["nan_inf_model_output_count"], 0)
        self.assertEqual(result["preprocessing_equivalence"]["mismatch_count"], 0)
        self.assertEqual(
            result["prediction_equivalence"]
            ["runtime_raw_camera_vs_stored_rgb_onnx_max_absolute_difference_rad"],
            0.0,
        )
        self.assertEqual(result["checkpoint_pipeline_equivalence"]["result"], "PASS")
        self.assertTrue(result["normalized_commands_within_bounds"])
        self.assertEqual(result["safety"]["steering_publish_count"], 0)
        self.assertEqual(result["safety"]["speed_publish_count"], 0)
        self.assertFalse(result["safety"]["real_vehicle_motion_performed"])


if __name__ == "__main__":
    unittest.main()
