import unittest
from unittest.mock import patch
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from physicar_e2e.pilotnet import PILOTNET_PARAMETER_COUNT, build_pilotnet
from physicar_e2e.pilotnet_recovery_inference import aggregate_repeatability, run_conditional_v2
from physicar_e2e.pilotnet_recovery_training import recovery_split


def metadata():
    episodes = []
    for role in ("failure", "curvature_near", "curvature_far"):
        for suffix in ("lat_p10", "lat_m10", "yaw_p06", "yaw_m06"):
            episodes.append({"episode_id": f"recovery_{role}_{suffix}", "recovery": {"anchor_role": role}})
    return {"episodes": episodes}


class RecoverySplitTests(unittest.TestCase):
    def test_nominal_validation_split_remains_episode_003(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "configs" / "pilotnet_training_v2_recovery.json").read_text())
        self.assertEqual(config["train_episodes"], ["episode_001", "episode_002"])
        self.assertEqual(config["validation_episodes"], ["episode_003"])

    def test_frozen_train_holdout_split_and_no_anchor_leakage(self):
        config = {
            "recovery_training_anchor_roles": ["failure", "curvature_near"],
            "recovery_holdout_anchor_role": "curvature_far",
        }
        split = recovery_split(metadata(), config)
        self.assertEqual(len(split["training"]), 8)
        self.assertEqual(len(split["holdout"]), 4)
        self.assertFalse(set(split["training"]) & set(split["holdout"]))
        self.assertTrue(all("curvature_far" in item for item in split["holdout"]))

    def test_v1_v2_architecture_contract_is_identical(self):
        v1 = build_pilotnet()
        v2 = build_pilotnet()
        self.assertEqual(sum(item.numel() for item in v1.parameters()), PILOTNET_PARAMETER_COUNT)
        self.assertEqual(list(v1.state_dict()), list(v2.state_dict()))


class ConditionalInferenceTests(unittest.TestCase):
    class Config:
        payload = {"smoke_speeds_mps": [0.5, 0.5, 0.5]}

        def safety_config(self, speed):
            return speed

    def test_first_failure_prevents_second_and_third(self):
        with (
            patch("physicar_e2e.pilotnet_recovery_inference.wait_after_reset", return_value=object()),
            patch("physicar_e2e.pilotnet_recovery_inference.run_smoke", return_value={"result": "FAIL"}) as run,
        ):
            results = run_conditional_v2(object(), object(), self.Config())
        self.assertEqual(len(results), 1)
        self.assertEqual(run.call_count, 1)

    def test_maximum_three_runs(self):
        with (
            patch("physicar_e2e.pilotnet_recovery_inference.wait_after_reset", return_value=object()),
            patch("physicar_e2e.pilotnet_recovery_inference.run_smoke", return_value={"result": "PASS"}) as run,
        ):
            results = run_conditional_v2(object(), object(), self.Config())
        self.assertEqual(len(results), 3)
        self.assertEqual(run.call_count, 3)

    def test_repeatability_requires_three_passes(self):
        self.assertIsNone(aggregate_repeatability([{"result": "PASS"}]))


if __name__ == "__main__":
    unittest.main()
