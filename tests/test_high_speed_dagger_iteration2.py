import json
from pathlib import Path
import unittest

import _bootstrap  # noqa: F401
from physicar_e2e.high_speed_dagger_iteration2 import (
    MAX_LIVE_ATTEMPTS, ROLLOUTS, control_authority_contract, freeze_a_window,
    load_config, reproduction_gate, route_in_frozen_window,
)
from physicar_e2e.pilotnet import build_pilotnet

ROOT = Path(__file__).resolve().parents[1]


class Iteration2Contracts(unittest.TestCase):
    def test_v6_controls_and_shadow_expert(self):
        c = control_authority_contract()
        self.assertEqual(c["vehicle_controller"], "PilotNet V6")
        self.assertFalse(c["shadow_expert_control_authority"])
        self.assertEqual(c["neural_observation"], ["camera"])

    def test_frozen_speed_expert_and_exactly_two_roles(self):
        c = load_config(ROOT / "configs/high_speed_dagger_iteration2_v1.json")
        self.assertEqual((c["diagnostic_speed_mps"], c["lookahead_m"], c["control_frequency_hz"]), (1.8, .9, 15.0))
        self.assertEqual(ROLLOUTS, {"high_speed_dagger_iter2_rollout_A": "training", "high_speed_dagger_iter2_rollout_B": "holdout"})
        self.assertEqual(c["maximum_collection_rollouts"], 2)

    def test_reproduction_gate_and_frozen_route_interval(self):
        self.assertEqual(reproduction_gate("training", {"result": "FAIL", "route_completion_fraction": .60})["result"], "PASS")
        self.assertEqual(reproduction_gate("training", {"result": "FAIL", "route_completion_fraction": .59})["result"], "FAIL")
        rows = [{"sim_time_s": i * .1, "unwrapped_progress_m": 18 + i * .2} for i in range(30)]
        w = freeze_a_window(rows, 20, 30.0)
        self.assertTrue(route_in_frozen_window(w["route_s_start_m"], w))
        self.assertFalse(route_in_frozen_window(w["route_s_end_m"] + .01, w))

    def test_architecture_and_from_scratch(self):
        self.assertEqual(sum(p.numel() for p in build_pilotnet().parameters()), 252219)
        c = json.loads((ROOT / "configs/pilotnet_training_v7_high_speed_dagger.json").read_text())
        self.assertEqual(c["initialization"], "from_scratch")
        self.assertEqual(c["train_episodes"], [f"episode_{i:03d}" for i in range(1, 9)])
        self.assertEqual(c["dagger1_training_rollout"], "high_speed_dagger_rollout_A")
        self.assertEqual(c["dagger2_training_rollout"], "high_speed_dagger_iter2_rollout_A")
        self.assertNotIn("v4", json.dumps(c).lower())
        self.assertNotIn("low_speed", json.dumps(c).lower())

    def test_live_budget_and_no_iteration3(self):
        c = json.loads((ROOT / "configs/pilotnet_inference_v7_high_speed_dagger.json").read_text())
        self.assertEqual(MAX_LIVE_ATTEMPTS, 5)
        self.assertEqual(c["smoke_speeds_mps"], [1.8, 1.8, 1.8])
        self.assertEqual(c["maximum_smoke_runs"], 3)


if __name__ == "__main__":
    unittest.main()
