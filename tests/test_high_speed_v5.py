import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from PIL import Image

import _bootstrap  # noqa: F401
from physicar_e2e.dataset_extractor import ScalarRecord, latest_causal
from physicar_e2e.high_speed_v5 import (
    EPISODES, HOLDOUT_EPISODES, LOOKAHEAD_M, REQUIRED_TOPICS, SPEED_MPS,
    TRAIN_EPISODES, VALIDATION_EPISODES, load_v5_inference, run_live_attempts,
    validate_v5_dataset, verify_frozen_expert,
)
from physicar_e2e.pilotnet import PILOTNET_PARAMETER_COUNT, build_pilotnet
from physicar_e2e.pilotnet_training import load_config, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def policy_result(result="PASS", failure=None):
    return {"result": result, "failure": failure, "api_failures": 0,
            "liveness_failures": 0, "safe_stop_success": True}


class Client:
    def __init__(self): self.stop_calls = 0
    def safe_stop(self): self.stop_calls += 1; return []


class HighSpeedV5Tests(unittest.TestCase):
    def test_frozen_expert_and_exact_collection_contract(self):
        expert = verify_frozen_expert(ROOT, ROOT / "configs/high_speed_expert_v1.json")
        self.assertEqual((expert.fixed_speed_mps, expert.lookahead_m), (1.8, 0.9))
        collector = json.loads((ROOT / "configs/high_speed_rosbag_v1.json").read_text())
        self.assertEqual(collector["pilot_episode_count"], 12)
        self.assertEqual(tuple(collector["required_topics"]), REQUIRED_TOPICS)

    def test_split_is_exactly_eight_two_two_without_overlap(self):
        self.assertEqual((len(TRAIN_EPISODES), len(VALIDATION_EPISODES), len(HOLDOUT_EPISODES)), (8, 2, 2))
        self.assertEqual((*TRAIN_EPISODES, *VALIDATION_EPISODES, *HOLDOUT_EPISODES), EPISODES)
        self.assertFalse(set(TRAIN_EPISODES) & set(VALIDATION_EPISODES))
        self.assertFalse(set(TRAIN_EPISODES) & set(HOLDOUT_EPISODES))

    def test_causal_labels_never_select_future(self):
        records = [ScalarRecord(10, 0.1), ScalarRecord(20, 0.2)]
        self.assertEqual(latest_causal(records, 19), records[0])
        self.assertLessEqual(latest_causal(records, 20).time_ns, 20)

    def test_training_is_from_scratch_and_excludes_v4_and_recovery(self):
        config = load_config(ROOT / "configs/pilotnet_training_v5_high_speed.json")
        self.assertEqual(config["initialization"], "from_scratch")
        serialized = json.dumps(config).lower()
        self.assertNotIn("0.50", serialized)
        self.assertNotIn("recovery", serialized)
        self.assertNotIn("pilotnet_v4", serialized)

    def test_v5_architecture_equals_v4_contract(self):
        self.assertEqual(sum(parameter.numel() for parameter in build_pilotnet().parameters()), 252_219)
        self.assertEqual(PILOTNET_PARAMETER_COUNT, 252_219)

    def test_v4_artifacts_untouched(self):
        self.assertEqual(sha256_file(ROOT / "configs/pilotnet_inference_v4_dagger.json"),
                         "5689447968dea75eed7771cd3751304da2df8aecb7269ded86e3f6ce422f0b45")
        self.assertEqual(sha256_file(Path("/home/a/physicar-ai-sim-docker/userdata/physicar_e2e/pilotnet_dagger_iteration2_v1/v4/onnx/pilotnet_v4_dagger.onnx")),
                         "5dd2b88b50c43aed44361229dea34e8981cb0a34d05b01eef81a9ccdf63f396a")

    def test_v5_live_speed_and_preprocessing_match_v4(self):
        config = load_v5_inference(ROOT)
        self.assertEqual(config.payload["smoke_speeds_mps"], [1.8, 1.8, 1.8])
        self.assertEqual(config.payload["roi"], {"x_start": 0, "x_end": 480, "y_start": 160, "y_end": 360})
        self.assertEqual(config.payload["control_frequency_hz"], 15.0)

    def execute_live(self, outcomes):
        run = Mock(side_effect=outcomes)
        preflight = Mock(return_value=(object(), {"result": "PASS"}))
        with tempfile.TemporaryDirectory() as directory:
            attempts, result = run_live_attempts(Client(), object(), object(), Path(directory),
                                                 preflight_one=preflight, run_one=run)
        return attempts, result, run.call_count

    def test_first_policy_failure_stops_without_retry_and_safe_stop_evidence(self):
        attempts, result, count = self.execute_live([policy_result("FAIL", "sustained off-track"), policy_result()])
        self.assertEqual((len(attempts), result, count), (1, "FAIL", 1))
        self.assertTrue(attempts[0]["run"]["safe_stop_success"])

    def test_conditional_three_run_logic(self):
        attempts, result, count = self.execute_live([policy_result(), policy_result(), policy_result(), policy_result()])
        self.assertEqual((len(attempts), result, count), (3, "PASS", 3))

    def test_dataset_integrity_rejects_frame_leakage(self):
        config = load_config(ROOT / "configs/pilotnet_training_v5_high_speed.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "manifests").mkdir(); (root / "images").mkdir()
            shared = root / "images/shared.png"; Image.new("RGB", (200, 66)).save(shared)
            for episode in EPISODES:
                with (root / "manifests" / f"{episode}.csv").open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["episode_id", "sample_index", "image_path", "steering_rad"])
                    writer.writeheader(); writer.writerow({"episode_id": episode, "sample_index": 0,
                                                           "image_path": "images/shared.png", "steering_rad": 0.0})
            with self.assertRaisesRegex(Exception, "duplicate/leaked"):
                validate_v5_dataset(root, config)


if __name__ == "__main__": unittest.main()
