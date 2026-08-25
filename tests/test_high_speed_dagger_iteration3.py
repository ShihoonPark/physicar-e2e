import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import _bootstrap  # noqa: F401
from physicar_e2e.high_speed_dagger import run_v6_attempts
from physicar_e2e.high_speed_dagger_iteration3 import (
    MAX_LIVE_ATTEMPTS, ROLLOUTS, control_authority_contract,
    fixed_target_region, in_fixed_target_region, late_region_name,
    load_config, reproduction_gate, validate_cumulative_composition,
)
from physicar_e2e.pilotnet import build_pilotnet
from physicar_e2e.pilotnet_dagger import latest_causal_shadow, sim_time_ns
from physicar_e2e.pilotnet_training import GateFailure

ROOT = Path(__file__).resolve().parents[1]


def row(episode, image, source):
    return {"episode_id": episode, "image_path": Path(image), "source_mcap_sha256": source,
            "window_role": "95_100_percent"}


class Client:
    def __init__(self): self.stop_calls = 0
    def safe_stop(self): self.stop_calls += 1; return []


class LiveConfig:
    payload = {"smoke_speeds_mps": [1.8, 1.8, 1.8]}


def policy(result="PASS"):
    return {"result": result, "api_failures": 0, "liveness_failures": 0,
            "safe_stop_success": True}


class Iteration3ContractTests(unittest.TestCase):
    def test_v7_controls_and_expert_is_shadow_only(self):
        contract = control_authority_contract()
        self.assertEqual(contract["vehicle_controller"], "PilotNet V7")
        self.assertFalse(contract["shadow_expert_control_authority"])
        self.assertEqual(contract["neural_observation"], ["camera"])

    def test_frozen_physical_contract_and_exactly_two_rollouts(self):
        config = load_config(ROOT / "configs/high_speed_dagger_iteration3_v1.json")
        self.assertEqual((config["diagnostic_speed_mps"], config["lookahead_m"],
                          config["control_frequency_hz"], config["max_steering_rad"]),
                         (1.8, .9, 15.0, .349066))
        self.assertEqual(config["maximum_collection_rollouts"], 2)
        self.assertEqual(config["rollout_assignment"], ROLLOUTS)

    def test_fixed_85_percent_region_accepts_pass_or_late_failure(self):
        self.assertEqual(fixed_target_region()["minimum_completion_fraction"], .85)
        self.assertFalse(in_fixed_target_region(.849999))
        self.assertTrue(in_fixed_target_region(.85))
        self.assertEqual(reproduction_gate("training", {"result": "PASS", "route_completion_fraction": .99})["result"], "PASS")
        self.assertEqual(reproduction_gate("holdout", {"result": "FAIL", "route_completion_fraction": .90})["result"], "PASS")
        self.assertEqual(reproduction_gate("training", {"result": "FAIL", "route_completion_fraction": .84})["result"], "FAIL")
        self.assertEqual([late_region_name(x) for x in (.85, .90, .95)],
                         ["85_90_percent", "90_95_percent", "95_100_percent"])

    def test_causal_shadow_label_never_uses_future(self):
        telemetry = [{"sim_time_s": 1.0, "shadow_expert_steering_rad": .1},
                     {"sim_time_s": 1.1, "shadow_expert_steering_rad": .2}]
        selected = latest_causal_shadow(telemetry, 1_050_000_000)
        self.assertLessEqual(sim_time_ns(selected), 1_050_000_000)

    def test_cumulative_config_from_scratch_and_no_low_speed(self):
        config = json.loads((ROOT / "configs/pilotnet_training_v8_high_speed_dagger.json").read_text())
        self.assertEqual(config["initialization"], "from_scratch")
        self.assertEqual(config["dagger1_training_rollout"], "high_speed_dagger_rollout_A")
        self.assertEqual(config["dagger2_training_rollout"], "high_speed_dagger_iter2_rollout_A")
        self.assertEqual(config["dagger3_training_rollout"], "high_speed_dagger_iter3_rollout_A")
        serialized = json.dumps(config).lower()
        self.assertNotIn("pilotnet_v4", serialized)
        self.assertNotIn("low_speed", serialized)
        self.assertNotIn("recovery", serialized)

    def test_architecture_is_unchanged(self):
        self.assertEqual(sum(p.numel() for p in build_pilotnet().parameters()), 252_219)

    def test_ab_hash_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [root / str(i) for i in range(9)]
            for index, path in enumerate(files): path.write_bytes(bytes([index]))
            nominal = [row("episode_001", files[0], "nominal")]
            d1a, d1b = [row("high_speed_dagger_rollout_A", files[1], "d1a")], [row("high_speed_dagger_rollout_B", files[2], "d1b")]
            d2a, d2b = [row("high_speed_dagger_iter2_rollout_A", files[3], "d2a")], [row("high_speed_dagger_iter2_rollout_B", files[4], "d2b")]
            d3a = [row("high_speed_dagger_iter3_rollout_A", files[5], "d3a")]
            d3b = [row("high_speed_dagger_iter3_rollout_B", files[6], "d3b")]
            result = validate_cumulative_composition(nominal, [row("episode_009", files[7], "nv")],
                                                     [row("episode_011", files[8], "nh")],
                                                     d1a, d1b, d2a, d2b, d3a, d3b)
            self.assertFalse(result["image_hash_overlap"])
            files[6].write_bytes(files[5].read_bytes())
            with self.assertRaisesRegex(GateFailure, "image hash leakage"):
                validate_cumulative_composition(nominal, [], [], d1a, d1b, d2a, d2b, d3a, d3b)

    def test_minimum_samples_and_no_iteration4_are_frozen(self):
        config = load_config(ROOT / "configs/high_speed_dagger_iteration3_v1.json")
        self.assertEqual(config["minimum_selected_samples"], 20)
        source = (ROOT / "src/physicar_e2e/high_speed_dagger_iteration3.py").read_text()
        self.assertIn('"automatic_iteration4": False', source)
        self.assertNotIn("Iteration 4 collection", source)


class Iteration3LiveGateTests(unittest.TestCase):
    def execute(self, outcomes):
        run = Mock(side_effect=outcomes)
        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            attempts, result = run_v6_attempts(client, object(), LiveConfig(), Path(directory),
                                               preflight_one=Mock(return_value=(object(), {"result": "PASS"})),
                                               run_one=run)
        return attempts, result, run, client

    def test_first_failure_stops(self):
        attempts, result, run, client = self.execute([policy("FAIL"), policy()])
        self.assertEqual((len(attempts), result, run.call_count), (1, "FAIL", 1))
        self.assertTrue(attempts[0]["run"]["safe_stop_success"])

    def test_second_failure_stops(self):
        attempts, result, run, _ = self.execute([policy(), policy("FAIL"), policy()])
        self.assertEqual((len(attempts), result, run.call_count), (2, "FAIL", 2))

    def test_three_valid_passes_and_five_total_attempt_budget(self):
        attempts, result, run, _ = self.execute([policy(), policy(), policy(), policy()])
        self.assertEqual((len(attempts), result, run.call_count), (3, "PASS", 3))
        self.assertEqual(MAX_LIVE_ATTEMPTS, 5)


if __name__ == "__main__": unittest.main()
