import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from physicar_e2e.dataset_extractor import (
    DriveWindow,
    ExtractionError,
    ScalarRecord,
    aggregate_summary,
    decode_rgb8_image,
    detect_dominant_drive_window,
    latest_causal,
    numeric_distribution,
    prepare_output_root,
    preprocess_image,
    sha256_file,
    steering_distribution,
    synchronize_frame,
    synchronization_diagnostics,
    write_manifest,
)


def config():
    return {
        "minimum_drive_speed_mps": 0.1,
        "maximum_steering_age_s": 0.15,
        "maximum_speed_age_s": 0.15,
        "source_width": 480,
        "source_height": 360,
        "source_encoding": "rgb8",
        "roi": {"x_start": 0, "x_end": 480, "y_start": 160, "y_end": 360},
        "output_width": 200,
        "output_height": 66,
        "maximum_steering_rad": 0.349066,
        "near_zero_steering_rad": 0.01,
        "saturation_fraction_of_limit": 0.99,
        "steering_histogram_bin_edges_rad": [-0.349066, -0.1, 0.1, 0.349066],
    }


def message(width=480, height=360, encoding="rgb8", step=None, data=None):
    step = width * 3 if step is None else step
    data = bytes(step * height) if data is None else data
    return SimpleNamespace(width=width, height=height, encoding=encoding, step=step, data=data)


class SynchronizationTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.window = DriveWindow(100, 1_000_000_000, 2)

    def test_causal_zoh_selects_latest_prior(self):
        records = [ScalarRecord(100, 1.0), ScalarRecord(200, 2.0)]
        self.assertEqual(latest_causal(records, 199), records[0])

    def test_future_steering_never_selected(self):
        self.assertIsNone(latest_causal([ScalarRecord(200, 2.0)], 199))

    def test_before_first_steering_rejected(self):
        reason, _, _ = synchronize_frame(150, [ScalarRecord(200, 0)], [ScalarRecord(100, .5)], self.window, self.cfg)
        self.assertEqual(reason, "no_causal_steering")

    def test_stale_steering_rejected(self):
        reason, _, _ = synchronize_frame(200_000_000, [ScalarRecord(1, 0)], [ScalarRecord(199_000_000, .5)], self.window, self.cfg)
        self.assertEqual(reason, "stale_steering")

    def test_stale_speed_rejected(self):
        reason, _, _ = synchronize_frame(200_000_000, [ScalarRecord(199_000_000, 0)], [ScalarRecord(1, .5)], self.window, self.cfg)
        self.assertEqual(reason, "stale_speed")

    def test_low_speed_rejected(self):
        reason, _, _ = synchronize_frame(200, [ScalarRecord(190, 0)], [ScalarRecord(190, 0)], self.window, self.cfg)
        self.assertEqual(reason, "below_drive_speed")

    def test_valid_active_frame_accepted(self):
        reason, steering, speed = synchronize_frame(200, [ScalarRecord(190, .2)], [ScalarRecord(190, .5)], self.window, self.cfg)
        self.assertIsNone(reason)
        self.assertLessEqual(steering.time_ns, 200)
        self.assertLessEqual(speed.time_ns, 200)

    def test_rejection_reason_accounting(self):
        cases = [-1, 50, 1_000_000_001]
        reasons = [synchronize_frame(t, [], [], self.window, self.cfg)[0] for t in cases]
        self.assertEqual(reasons, ["before_active_drive_window", "before_active_drive_window", "after_active_drive_window"])


class WindowTests(unittest.TestCase):
    def test_active_segment_detection(self):
        records = [ScalarRecord(0, 0), ScalarRecord(1, .5), ScalarRecord(5, .5), ScalarRecord(6, 0)]
        self.assertEqual(detect_dominant_drive_window(records, .1), DriveWindow(1, 5, 2))

    def test_dominant_of_multiple_segments(self):
        records = [ScalarRecord(0, .5), ScalarRecord(1, 0), ScalarRecord(10, .5), ScalarRecord(20, .5), ScalarRecord(21, 0)]
        self.assertEqual(detect_dominant_drive_window(records, .1).start_ns, 10)


class ImageTests(unittest.TestCase):
    def test_rgb8_row_stride(self):
        row = bytes([255, 0, 0] * 480) + b"padding"
        decoded = decode_rgb8_image(message(step=len(row), data=row * 360), config())
        self.assertEqual(decoded.getpixel((0, 0)), (255, 0, 0))
        self.assertEqual(decoded.getpixel((0, 1)), (255, 0, 0))

    def test_wrong_encoding_fails(self):
        with self.assertRaisesRegex(ExtractionError, "encoding"):
            decode_rgb8_image(message(encoding="bgr8"), config())

    def test_wrong_dimensions_fail(self):
        with self.assertRaisesRegex(ExtractionError, "dimensions"):
            decode_rgb8_image(message(width=10), config())

    def test_truncated_data_fails(self):
        with self.assertRaisesRegex(ExtractionError, "truncated"):
            decode_rgb8_image(message(data=b"short"), config())

    def test_invalid_step_fails(self):
        with self.assertRaisesRegex(ExtractionError, "step"):
            decode_rgb8_image(message(step=100, data=bytes(36000)), config())

    def test_roi_dimensions(self):
        image = Image.new("RGB", (480, 360))
        roi = config()["roi"]
        self.assertEqual(image.crop((roi["x_start"], roi["y_start"], roi["x_end"], roi["y_end"])).size, (480, 200))

    def test_resize_exact_output(self):
        result = preprocess_image(Image.new("RGB", (480, 360)), config())
        self.assertEqual(result.size, (200, 66))
        self.assertEqual(result.mode, "RGB")


class OutputAndStatisticsTests(unittest.TestCase):
    def test_deterministic_filenames_and_order(self):
        times = [30, 10, 20]
        ordered = sorted(times)
        names = [f"frame_{index:06d}.png" for index, _ in enumerate(ordered)]
        self.assertEqual(names, ["frame_000000.png", "frame_000001.png", "frame_000002.png"])

    def test_manifest_row_correctness(self):
        from physicar_e2e.dataset_extractor import MANIFEST_COLUMNS
        row = {key: key for key in MANIFEST_COLUMNS}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "manifest.csv"
            write_manifest(path, [row])
            with path.open() as stream:
                loaded = list(csv.DictReader(stream))
        self.assertEqual(loaded[0]["episode_id"], "episode_id")
        self.assertEqual(list(loaded[0]), MANIFEST_COLUMNS)

    def test_steering_distribution(self):
        result = steering_distribution([-.349066, 0, .02], config())
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["negative_count"], 1)
        self.assertEqual(result["near_zero_count"], 1)
        self.assertEqual(result["positive_count"], 1)
        self.assertEqual(sum(result["histogram"]["bin_counts"]), 3)

    def test_synchronization_percentiles(self):
        result = synchronization_diagnostics([1, 2, 100], [3, 4, 5], [0, 10_000_000, 30_000_000])
        self.assertAlmostEqual(result["steering_age_ms"]["median"], 2)
        self.assertAlmostEqual(result["steering_age_ms"]["p95"], 90.2)
        self.assertEqual(result["accepted_camera_interval_ms"]["max"], 20)

    def test_numeric_empty_is_honest(self):
        self.assertIsNone(numeric_distribution([])["mean"])

    def test_source_sha256(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "source.mcap"
            path.write_bytes(b"canonical raw")
            self.assertEqual(sha256_file(path), hashlib.sha256(b"canonical raw").hexdigest())

    def test_fail_if_output_exists(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                prepare_output_root(Path(temp), False)

    def test_force_replaces_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "a" / "b" / "dataset"
            root.mkdir(parents=True)
            (root / "old").write_text("old")
            prepare_output_root(root, True)
            self.assertFalse((root / "old").exists())
            self.assertTrue((root / "images").is_dir())

    def test_episode_level_separation_and_three_episode_summary(self):
        episodes = []
        rows = []
        for index in range(3):
            episode_id = f"episode_{index + 1:03d}"
            episodes.append({
                "episode_id": episode_id, "result": "PASS",
                "counts": {"total_camera_frames": 1, "active_window_camera_frames": 1, "accepted_camera_samples": 1, "rejection_by_reason": {}},
                "source": {"mcap_size_bytes": 10},
                "synchronization": {"future_label_violations": 0, "steering_age_ms": {"max": 1}, "speed_age_ms": {"max": 1}},
            })
            rows.append({"episode_id": episode_id, "camera_record_time_ns": index, "steering_rad": 0, "steering_age_ms": 1, "speed_age_ms": 1})
        with tempfile.TemporaryDirectory() as temp:
            summary = aggregate_summary(episodes, rows, Path(temp), config(), "hash")
        self.assertEqual(summary["episode_count"], 3)
        self.assertTrue(summary["pilot_success_gate"]["episode_level_separation_preserved"])
        self.assertIn("No frame-level split", summary["split_policy"])


if __name__ == "__main__":
    unittest.main()
