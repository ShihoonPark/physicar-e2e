import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from physicar_e2e.real_dataset import (
    MANIFEST_COLUMNS,
    MAXIMUM_AGE_OR_GAP_S,
    NEURAL_INPUT_FIELDS,
    SPEED_SEMANTICS,
    TRAINING_INVOCATION_PERMITTED,
    CameraFrame,
    RealDatasetError,
    ScalarRecord,
    build_steering_target,
    evaluate_sequence_candidate,
    latest_causal,
    load_config,
    preprocess_real_camera_image,
    speed_metadata,
    steering_recorded_to_radians,
    write_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "real_dataset_v1.json"


def config():
    return load_config(CONFIG_PATH)


def frames(times=(0, 60_000_000, 120_000_000), bag_id="bag_01"):
    return [
        CameraFrame(
            bag_id=bag_id,
            index=index,
            time_ns=time_ns,
            image_path=f"images/{bag_id}/frame_{index:06d}.png",
            image_sha256="a" * 64,
            image_size_bytes=1,
        )
        for index, time_ns in enumerate(times)
    ]


def evaluate(
    *,
    camera_times=(0, 60_000_000, 120_000_000),
    steering=(ScalarRecord(110_000_000, 0.8),),
    speed=(ScalarRecord(110_000_000, 0.5),),
):
    return evaluate_sequence_candidate(
        frames=frames(camera_times),
        target_index=2,
        steering_records=steering,
        speed_records=speed,
        source_mcap_sha256="b" * 64,
        config=config(),
    )


class SteeringContractTests(unittest.TestCase):
    def test_whole_stream_scale_is_exactly_point_35(self):
        self.assertAlmostEqual(steering_recorded_to_radians(0.8), 0.28)
        self.assertAlmostEqual(steering_recorded_to_radians(-1.0), -0.35)
        with self.assertRaises(RealDatasetError):
            steering_recorded_to_radians(0.8, 1.0)

    def test_manifest_target_prevents_omission_or_double_scaling(self):
        row, rejection = evaluate()
        self.assertIsNone(rejection)
        self.assertEqual(row["steering_recorded_raw"], 0.8)
        self.assertAlmostEqual(row["steering_rad"], 0.28)
        self.assertNotAlmostEqual(row["steering_rad"], 0.8)
        self.assertNotAlmostEqual(row["steering_rad"], 0.8 * 0.35 * 0.35)

    def test_positive_remains_left_and_negative_remains_right(self):
        self.assertEqual(build_steering_target(0.2).direction, "LEFT")
        self.assertGreater(build_steering_target(0.2).radians, 0)
        self.assertEqual(build_steering_target(-0.2).direction, "RIGHT")
        self.assertLess(build_steering_target(-0.2).radians, 0)

    def test_no_selective_clip(self):
        self.assertAlmostEqual(build_steering_target(1.1).radians, 0.385)


class CausalTimingTests(unittest.TestCase):
    def test_mcap_log_time_causal_zoh_uses_latest_prior(self):
        records = [ScalarRecord(100, 1.0), ScalarRecord(200, 2.0)]
        self.assertEqual(latest_causal(records, 199), records[0])
        self.assertEqual(latest_causal(records, 200), records[1])
        self.assertIsNone(latest_causal(records, 99))

    def test_steering_age_strictly_over_120ms_rejected(self):
        row, rejection = evaluate(
            camera_times=(180_000_000, 240_000_000, 300_000_000),
            steering=(ScalarRecord(179_000_000, 0.2),),
            speed=(ScalarRecord(299_000_000, 0.5),),
        )
        self.assertIsNone(row)
        self.assertIn("steering_age_gt_0p120_s", rejection["reasons"])

    def test_steering_age_exactly_120ms_is_accepted(self):
        row, rejection = evaluate(
            camera_times=(180_000_000, 240_000_000, 300_000_000),
            steering=(ScalarRecord(180_000_000, 0.2),),
            speed=(ScalarRecord(299_000_000, 0.5),),
        )
        self.assertIsNone(rejection)
        self.assertAlmostEqual(row["steering_age_s"], MAXIMUM_AGE_OR_GAP_S)

    def test_adjacent_gap_strictly_over_120ms_rejected(self):
        row, rejection = evaluate(
            camera_times=(0, 60_000_000, 181_000_000),
            steering=(ScalarRecord(180_000_000, 0.2),),
            speed=(ScalarRecord(180_000_000, 0.5),),
        )
        self.assertIsNone(row)
        self.assertIn("adjacent_camera_gap_gt_0p120_s", rejection["reasons"])

    def test_adjacent_gap_exactly_120ms_is_accepted(self):
        row, rejection = evaluate(
            camera_times=(0, 60_000_000, 180_000_000),
            steering=(ScalarRecord(180_000_000, 0.2),),
            speed=(ScalarRecord(180_000_000, 0.5),),
        )
        self.assertIsNone(rejection)
        self.assertAlmostEqual(row["adjacent_gap_t_minus_1_to_t_s"], 0.120)

    def test_no_future_steering_label(self):
        row, rejection = evaluate(
            steering=(ScalarRecord(120_000_001, 0.5),),
            speed=(ScalarRecord(119_000_000, 0.5),),
        )
        self.assertIsNone(row)
        self.assertIn("no_causal_steering", rejection["reasons"])
        self.assertNotIn("future_steering_label", rejection["reasons"])


class SpeedMetadataTests(unittest.TestCase):
    def test_stale_speed_is_preserved_but_does_not_reject(self):
        row, rejection = evaluate(
            camera_times=(180_000_000, 240_000_000, 300_000_000),
            steering=(ScalarRecord(299_000_000, 0.2),),
            speed=(ScalarRecord(179_000_000, 0.5),),
        )
        self.assertIsNone(rejection)
        self.assertEqual(row["speed_mps"], 0.5)
        self.assertTrue(row["speed_stale"])
        self.assertFalse(row["speed_valid"])
        self.assertEqual(row["speed_state"], "STALE")
        self.assertEqual(row["speed_semantics"], SPEED_SEMANTICS)

    def test_missing_speed_does_not_block_extraction(self):
        row, rejection = evaluate(speed=())
        self.assertIsNone(rejection)
        self.assertFalse(row["speed_available"])
        self.assertFalse(row["speed_valid"])
        self.assertEqual(row["speed_state"], "MISSING")
        self.assertEqual(row["speed_semantics"], SPEED_SEMANTICS)

    def test_speed_age_boundary(self):
        fresh = speed_metadata(200_000_000, [ScalarRecord(80_000_000, 0.4)])
        stale = speed_metadata(200_000_001, [ScalarRecord(80_000_000, 0.4)])
        self.assertTrue(fresh["speed_valid"])
        self.assertFalse(fresh["speed_stale"])
        self.assertFalse(stale["speed_valid"])
        self.assertTrue(stale["speed_stale"])

    def test_speed_never_becomes_neural_input_or_target(self):
        cfg = config()
        self.assertEqual(NEURAL_INPUT_FIELDS, ("image_t_minus_2", "image_t_minus_1", "image_t"))
        self.assertNotIn("speed_mps", NEURAL_INPUT_FIELDS)
        self.assertFalse(cfg["speed_contract"]["neural_input"])
        self.assertFalse(cfg["speed_contract"]["steering_target"])
        self.assertFalse(cfg["speed_contract"]["active_driving_filter"])


class RealCameraContractTests(unittest.TestCase):
    def test_roi_is_exactly_real_v1_and_not_simulator_roi(self):
        camera = config()["camera_contract"]
        self.assertEqual(
            camera["roi"],
            {"x_start": 0, "x_end": 480, "y_start": 80, "y_end": 360, "end_coordinates_exclusive": True},
        )
        self.assertFalse(camera["simulator_y_160_360_crop"])
        self.assertFalse(camera["horizontal_crop"])
        self.assertFalse(camera["undistortion"])

    def test_crop_and_bilinear_resize_produce_rgb_200x66(self):
        red = bytes([255, 0, 0] * 480)
        green = bytes([0, 255, 0] * 480)
        payload = red * 80 + green * 280
        message = SimpleNamespace(
            width=480,
            height=360,
            encoding="rgb8",
            step=1440,
            data=payload,
        )
        image = preprocess_real_camera_image(message, config())
        self.assertEqual(image.size, (200, 66))
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.getpixel((0, 0)), (0, 255, 0))
        self.assertEqual(image.getpixel((199, 65)), (0, 255, 0))
        image.close()


class ManifestAndScopeTests(unittest.TestCase):
    def test_manifest_has_required_temporal_and_metadata_fields(self):
        required = {
            "sequence_id", "source_bag", "target_camera_log_time_ns",
            "image_t_minus_2", "image_t_minus_1", "image_t",
            "camera_t_minus_2_log_time_ns", "camera_t_minus_1_log_time_ns",
            "camera_t_log_time_ns", "adjacent_gap_t_minus_2_to_t_minus_1_s",
            "adjacent_gap_t_minus_1_to_t_s", "oldest_to_current_span_s",
            "steering_recorded_raw", "steering_rad", "steering_log_time_ns",
            "steering_age_s", "speed_mps", "speed_log_time_ns", "speed_age_s",
            "speed_valid", "speed_stale", "speed_semantics",
        }
        self.assertTrue(required.issubset(MANIFEST_COLUMNS))
        row, _ = evaluate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.csv"
            write_manifest(path, [row])
            with path.open(newline="", encoding="utf-8") as stream:
                loaded = list(csv.DictReader(stream))
        self.assertEqual(len(loaded), 1)
        self.assertEqual(tuple(loaded[0]), MANIFEST_COLUMNS)

    def test_real_sequence_has_three_distinct_same_bag_frames(self):
        row, rejection = evaluate()
        self.assertIsNone(rejection)
        paths = [row[field] for field in NEURAL_INPUT_FIELDS]
        self.assertEqual(len(set(paths)), 3)
        self.assertTrue(all(path.startswith("images/bag_01/") for path in paths))

    def test_config_rejects_guessed_speed_semantics(self):
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        raw["speed_contract"]["semantics"] = "COMMAND"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaises(RealDatasetError):
                load_config(path)

    def test_no_training_invocation_or_authorization(self):
        self.assertFalse(TRAINING_INVOCATION_PERMITTED)
        self.assertFalse(config()["scope_guards"]["training_permitted"])
        source = (ROOT / "src" / "physicar_e2e" / "real_dataset.py").read_text(encoding="utf-8")
        runner = (ROOT / "scripts" / "run_real_dataset_v1.py").read_text(encoding="utf-8")
        self.assertNotIn("pilotnet_training", source)
        self.assertNotIn("torch", source)
        self.assertNotIn("subprocess", source)
        self.assertNotIn("pilotnet_training", runner)


if __name__ == "__main__":
    unittest.main()
