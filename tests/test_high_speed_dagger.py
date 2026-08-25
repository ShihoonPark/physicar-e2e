import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import _bootstrap  # noqa: F401
from physicar_e2e.high_speed_dagger import (
    ROLLOUTS, control_authority_contract, load_config, load_v6_inference,
    passes_progress_gate, reproduction_gate, run_v6_attempts, select_objective_window,
    validate_composition,
)
from physicar_e2e.pilotnet import PILOTNET_PARAMETER_COUNT, build_pilotnet
from physicar_e2e.pilotnet_dagger import latest_causal_shadow, sim_time_ns
from physicar_e2e.pilotnet_training import GateFailure


ROOT = Path(__file__).resolve().parents[1]


def row(episode, path, source="source"):
    return {"episode_id": episode, "image_path": Path(path), "source_mcap_sha256": source,
            "window_role": "divergence"}


def policy_result(result="PASS"):
    return {"result": result, "api_failures": 0, "liveness_failures": 0, "safe_stop_success": True}


class Client:
    def __init__(self):
        self.stop_calls = 0

    def safe_stop(self):
        self.stop_calls += 1
        return []


class Config:
    payload = {"smoke_speeds_mps": [1.8, 1.8, 1.8]}


class HighSpeedDaggerContractTests(unittest.TestCase):
    def test_v5_controls_and_expert_is_shadow_only(self):
        contract = control_authority_contract()
        self.assertEqual(contract["vehicle_controller"], "PilotNet V5")
        self.assertFalse(contract["shadow_expert_control_authority"])
        self.assertEqual(contract["neural_observation"], ["camera"])

    def test_speed_lookahead_rate_and_limit_are_frozen(self):
        config = load_config(ROOT / "configs/high_speed_dagger_v1.json")
        self.assertEqual((config["diagnostic_speed_mps"], config["lookahead_m"],
                          config["control_frequency_hz"], config["max_steering_rad"]),
                         (1.8, 0.9, 15.0, 0.349066))

    def test_exactly_two_rollouts_and_frozen_roles(self):
        config = load_config(ROOT / "configs/high_speed_dagger_v1.json")
        self.assertEqual(config["maximum_collection_rollouts"], 2)
        self.assertEqual(config["rollout_assignment"], ROLLOUTS)
        self.assertEqual(ROLLOUTS["high_speed_dagger_rollout_A"], "training")
        self.assertEqual(ROLLOUTS["high_speed_dagger_rollout_B"], "holdout")

    def test_latest_shadow_label_is_causal_and_never_future(self):
        telemetry = [{"sim_time_s": 1.0, "shadow_expert_steering_rad": 0.1},
                     {"sim_time_s": 1.1, "shadow_expert_steering_rad": 0.2}]
        selected = latest_causal_shadow(telemetry, 1_050_000_000)
        self.assertEqual(selected["shadow_expert_steering_rad"], 0.1)
        self.assertLessEqual(sim_time_ns(selected), 1_050_000_000)

    def test_window_is_two_seconds_and_progress_gate_is_30_percent(self):
        telemetry = [{"sim_time_s": 10.0}, {"sim_time_s": 12.0}, {"sim_time_s": 13.0}]
        selected = select_objective_window(telemetry, 1, 30.0)
        self.assertEqual(selected["start_sim_time_ns"], 10_000_000_000)
        self.assertEqual(selected["end_sim_time_ns"], 13_000_000_000)
        self.assertEqual(selected["minimum_route_progress_m"], 9.0)
        self.assertFalse(passes_progress_gate(8.99, 30.0))
        self.assertTrue(passes_progress_gate(9.0, 30.0))

    def test_rollout_a_reproduction_gate_stops_early_and_full_lap(self):
        early = reproduction_gate("training", {"result": "FAIL", "route_completion_fraction": 0.299})
        success = reproduction_gate("training", {"result": "PASS", "route_completion_fraction": 1.0})
        self.assertEqual((early["result"], success["result"]), ("FAIL", "FAIL"))

    def test_composition_has_no_nominal_holdout_or_rollout_b_leakage(self):
        train = [row(f"episode_{index:03d}", f"train/{index}") for index in range(1, 9)]
        validation = [row(f"episode_{index:03d}", f"validation/{index}") for index in range(9, 11)]
        holdout = [row(f"episode_{index:03d}", f"holdout/{index}") for index in range(11, 13)]
        dagger_a = [row("high_speed_dagger_rollout_A", "dagger/A", "A")]
        dagger_b = [row("high_speed_dagger_rollout_B", "dagger/B", "B")]
        result = validate_composition(train, validation, holdout, dagger_a, dagger_b)
        self.assertFalse(result["holdout_leakage"])

    def test_composition_rejects_rollout_b_as_training(self):
        train = [row(f"episode_{index:03d}", f"train/{index}") for index in range(1, 9)]
        validation = [row(f"episode_{index:03d}", f"validation/{index}") for index in range(9, 11)]
        holdout = [row(f"episode_{index:03d}", f"holdout/{index}") for index in range(11, 13)]
        with self.assertRaisesRegex(GateFailure, "non-A"):
            validate_composition(train, validation, holdout,
                                 [row("high_speed_dagger_rollout_B", "dagger/B", "B")],
                                 [row("high_speed_dagger_rollout_B", "dagger/B2", "B2")])

    def test_training_config_excludes_v4_low_speed_dagger_and_recovery(self):
        config = json.loads((ROOT / "configs/pilotnet_training_v6_high_speed_dagger.json").read_text())
        serialized = json.dumps(config).lower()
        self.assertNotIn("pilotnet_v4", serialized)
        self.assertNotIn(config["dagger_training_rollout"], {"dagger_rollout_A", "dagger_iter2_rollout_A"})
        self.assertNotIn(config["dagger_holdout_rollout"], {"dagger_rollout_B", "dagger_iter2_rollout_B"})
        self.assertNotIn("recovery", serialized)

    def test_v6_architecture_identical_to_v5(self):
        v5, v6 = build_pilotnet(), build_pilotnet()
        self.assertEqual(sum(parameter.numel() for parameter in v6.parameters()), 252_219)
        self.assertEqual(PILOTNET_PARAMETER_COUNT, 252_219)
        self.assertEqual(list(v5.state_dict()), list(v6.state_dict()))

    def test_v6_trains_from_scratch_with_unchanged_hyperparameters(self):
        v5 = json.loads((ROOT / "configs/pilotnet_training_v5_high_speed.json").read_text())
        v6 = json.loads((ROOT / "configs/pilotnet_training_v6_high_speed_dagger.json").read_text())
        self.assertEqual(v6["initialization"], "from_scratch")
        for key in ("optimizer", "loss", "learning_rate", "batch_size", "max_epochs",
                    "early_stopping_patience", "roi" if "roi" in v5 else "image_width"):
            self.assertEqual(v6[key], v5[key])

    def test_v6_inference_is_camera_only_exactly_1p8(self):
        config = load_v6_inference(ROOT)
        self.assertTrue(config.payload["camera_only_model_observation"])
        self.assertEqual(config.payload["smoke_speeds_mps"], [1.8, 1.8, 1.8])


class HighSpeedDaggerLiveGateTests(unittest.TestCase):
    def execute(self, outcomes=None, preflight=None):
        run = Mock(side_effect=outcomes or [])
        before = preflight or Mock(return_value=(object(), {"result": "PASS"}))
        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            attempts, result = run_v6_attempts(client, object(), Config(), Path(directory),
                                               preflight_one=before, run_one=run)
        return attempts, result, run, client

    def test_first_v6_policy_failure_stops(self):
        attempts, result, run, _ = self.execute([policy_result("FAIL"), policy_result("PASS")])
        self.assertEqual((len(attempts), result, run.call_count), (1, "FAIL", 1))

    def test_three_passes_use_exactly_three_valid_runs(self):
        attempts, result, run, _ = self.execute([policy_result(), policy_result(), policy_result(), policy_result()])
        self.assertEqual((len(attempts), result, run.call_count), (3, "PASS", 3))

    def test_infrastructure_exception_safe_stops(self):
        before = Mock(side_effect=RuntimeError("clock unavailable"))
        attempts, result, run, client = self.execute([], before)
        self.assertEqual(len(attempts), 5)
        self.assertEqual(result, "INCONCLUSIVE")
        self.assertEqual(run.call_count, 0)
        self.assertEqual(client.stop_calls, 5)
        self.assertTrue(all(item["classification"] == "INFRA_FAIL" for item in attempts))


if __name__ == "__main__":
    unittest.main()
