import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
from physicar_e2e.pilotnet import PILOTNET_PARAMETER_COUNT, build_pilotnet
from physicar_e2e.pilotnet_dagger import (
    latest_causal_shadow, load_config, select_dagger_window, sim_time_ns, window_role,
)
from physicar_e2e.pilotnet_dagger_inference import aggregate, run_conditional_v3
from physicar_e2e.pilotnet_dagger_training import dagger_split, read_dagger_rows
from physicar_e2e.pilotnet_training import GateFailure


ROOT = Path(__file__).resolve().parents[1]


def telemetry():
    return [
        {"sim_time_s": 10.0, "elapsed_s": 0.0, "cte_m": 0.01, "shadow_expert_steering_rad": 0.1},
        {"sim_time_s": 10.1, "elapsed_s": 0.1, "cte_m": 0.02, "shadow_expert_steering_rad": 0.2},
        {"sim_time_s": 10.2, "elapsed_s": 0.2, "cte_m": 0.04, "shadow_expert_steering_rad": 0.3},
    ]


class DaggerAlignmentTests(unittest.TestCase):
    def test_shadow_expert_alignment_is_latest_causal(self):
        selected = latest_causal_shadow(telemetry(), 10_150_000_000)
        self.assertEqual(selected["shadow_expert_steering_rad"], 0.2)
        self.assertLessEqual(sim_time_ns(selected), 10_150_000_000)

    def test_no_future_label_when_camera_precedes_first_state(self):
        self.assertIsNone(latest_causal_shadow(telemetry(), 9_999_000_000))

    def test_objective_window_starts_before_divergence_and_ends_at_last_valid(self):
        result = select_dagger_window(telemetry(), 2, 0.1)
        self.assertEqual(result["start_sim_time_ns"], 10_100_000_000)
        self.assertEqual(result["divergence_sim_time_ns"], 10_200_000_000)
        self.assertEqual(result["end_sim_time_ns"], 10_200_000_000)

    def test_window_roles_are_fixed_by_time_not_model_error(self):
        self.assertEqual(window_role(99, 100, 1.0), "pre_divergence")
        self.assertEqual(window_role(100, 100, 1.0), "divergence")
        self.assertEqual(window_role(1_000_000_100, 100, 1.0), "late_failure")


class DaggerCompositionTests(unittest.TestCase):
    def test_rollout_assignment_is_frozen(self):
        config = load_config(ROOT / "configs/pilotnet_dagger_v1.json")
        self.assertEqual(config["rollout_assignment"], {"dagger_rollout_A": "training", "dagger_rollout_B": "holdout"})
        self.assertEqual(config["maximum_collection_rollouts"], 1)

    def test_split_has_no_holdout_leakage(self):
        metadata = {"episodes": [
            {"rollout_id": "dagger_rollout_A", "role": "training"},
            {"rollout_id": "dagger_rollout_B", "role": "holdout"},
        ]}
        config = {"dagger_training_rollout": "dagger_rollout_A", "dagger_holdout_rollout": "dagger_rollout_B"}
        self.assertEqual(dagger_split(metadata, config), {"training": "dagger_rollout_A", "holdout": "dagger_rollout_B"})

    def test_manifest_rejects_future_expert_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "manifests").mkdir(); (root / "images" / "dagger_rollout_A").mkdir(parents=True)
            image = root / "images/dagger_rollout_A/frame.png"; image.write_bytes(b"x")
            fields = ["episode_id", "sample_index", "image_path", "steering_rad", "expert_label_time_ns", "camera_header_time_ns", "window_role", "source_mcap_sha256"]
            with (root / "manifests/dagger_rollout_A.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerow({
                    "episode_id": "dagger_rollout_A", "sample_index": 0, "image_path": "images/dagger_rollout_A/frame.png",
                    "steering_rad": 0.1, "expert_label_time_ns": 11, "camera_header_time_ns": 10,
                    "window_role": "divergence", "source_mcap_sha256": "a",
                })
            with self.assertRaisesRegex(GateFailure, "future"):
                read_dagger_rows(root, "dagger_rollout_A")

    def test_v3_dataset_contract_excludes_recovery_and_nominal_validation(self):
        config = json.loads((ROOT / "configs/pilotnet_training_v3_dagger.json").read_text())
        self.assertEqual(config["train_episodes"], ["episode_001", "episode_002"])
        self.assertEqual(config["validation_episodes"], ["episode_003"])
        self.assertEqual(config["dagger_training_rollout"], "dagger_rollout_A")
        self.assertEqual(config["dagger_holdout_rollout"], "dagger_rollout_B")
        self.assertFalse(any("recovery" in str(value) for value in config.values()))

    def test_v1_v3_architecture_is_identical(self):
        v1, v3 = build_pilotnet(), build_pilotnet()
        self.assertEqual(sum(parameter.numel() for parameter in v3.parameters()), PILOTNET_PARAMETER_COUNT)
        self.assertEqual(list(v1.state_dict()), list(v3.state_dict()))


class V3ConditionalLiveTests(unittest.TestCase):
    class Config:
        payload = {"smoke_speeds_mps": [0.5, 0.5, 0.5]}
        def safety_config(self, speed): return speed

    def test_first_failure_stops_and_no_retry(self):
        with (patch("physicar_e2e.pilotnet_dagger_inference.wait_after_reset", return_value=object()),
              patch("physicar_e2e.pilotnet_dagger_inference.run_smoke", return_value={"result": "FAIL"}) as run):
            result = run_conditional_v3(object(), object(), self.Config())
        self.assertEqual(len(result), 1); self.assertEqual(run.call_count, 1)

    def test_maximum_three_runs(self):
        with (patch("physicar_e2e.pilotnet_dagger_inference.wait_after_reset", return_value=object()),
              patch("physicar_e2e.pilotnet_dagger_inference.run_smoke", return_value={"result": "PASS"}) as run):
            result = run_conditional_v3(object(), object(), self.Config())
        self.assertEqual(len(result), 3); self.assertEqual(run.call_count, 3)

    def test_repeatability_requires_exactly_three_passes(self):
        self.assertIsNone(aggregate([{"result": "PASS"}]))


if __name__ == "__main__":
    unittest.main()
