import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import _bootstrap  # noqa: F401
from physicar_e2e.expert_driver import Preflight
from physicar_e2e.pilotnet import PILOTNET_PARAMETER_COUNT, build_pilotnet
from physicar_e2e.pilotnet_dagger import latest_causal_shadow
from physicar_e2e.pilotnet_dagger_iteration2 import (
    detect_final_failure_divergence, enforce_reproduction_gate, filter_progress_rows, load_config, select_iteration2_window,
)
from physicar_e2e.pilotnet_dagger_iteration2_inference import aggregate, run_conditional_v4
from physicar_e2e.pilotnet_dagger_iteration2_training import iteration2_split, validate_cumulative_composition
from physicar_e2e.pilotnet_failure_diagnosis import run_live_loop
from physicar_e2e.pilotnet_training import GateFailure
from physicar_e2e.route_geometry import ClosedRoute


ROOT = Path(__file__).resolve().parents[1]


def telemetry():
    return [
        {"sim_time_s": 100.0, "elapsed_s": 0.0, "cte_m": 0.01, "unwrapped_progress_m": 9.0},
        {"sim_time_s": 101.0, "elapsed_s": 1.0, "cte_m": 0.02, "unwrapped_progress_m": 10.0},
        {"sim_time_s": 103.0, "elapsed_s": 3.0, "cte_m": 0.05, "unwrapped_progress_m": 12.0},
        {"sim_time_s": 104.0, "elapsed_s": 4.0, "cte_m": 0.20, "unwrapped_progress_m": 13.0},
    ]


def initial_state():
    center = [(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]
    inner = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8), (0.2, 0.2)]
    outer = [(-0.2, -0.2), (1.2, -0.2), (1.2, 1.2), (-0.2, 1.2), (-0.2, -0.2)]
    return Preflight("world", ClosedRoute(center, inner, outer), 5, 0, {}, {"x": 0, "y": 0, "yaw": 0})


class Iteration2CollectionTests(unittest.TestCase):
    def test_v3_rollout_ab_assignment_and_count_are_frozen(self):
        config = load_config(ROOT / "configs/pilotnet_dagger_iteration2_v1.json")
        self.assertEqual(config["rollout_assignment"], {"dagger_iter2_rollout_A": "training", "dagger_iter2_rollout_B": "holdout"})
        self.assertEqual(config["maximum_collection_rollouts"], 2)

    def test_stop_when_later_failure_distribution_not_reproduced(self):
        result = enforce_reproduction_gate("training", {"result": "FAIL", "route_progress_m": 9.99})
        self.assertEqual(result["result"], "FAIL")
        self.assertIn("not reproduced", result["reason"])

    def test_stop_when_v3_unexpectedly_completes_lap(self):
        result = enforce_reproduction_gate("training", {"result": "PASS", "route_progress_m": 30.5})
        self.assertEqual(result["result"], "FAIL")
        self.assertIn("unexpectedly completed", result["reason"])

    def test_later_failure_at_or_beyond_10m_passes(self):
        result = enforce_reproduction_gate("training", {"result": "FAIL", "route_progress_m": 19.0})
        self.assertEqual(result["result"], "PASS")

    def test_holdout_does_not_control_training_reproduction_gate(self):
        self.assertEqual(enforce_reproduction_gate("holdout", {"result": "FAIL", "route_progress_m": 2.0})["result"], "NOT_APPLICABLE")

    def test_iteration2_window_is_two_seconds_before_divergence(self):
        result = select_iteration2_window(telemetry(), 2)
        self.assertEqual(result["start_sim_time_ns"], 101_000_000_000)
        self.assertEqual(result["divergence_sim_time_ns"], 103_000_000_000)
        self.assertEqual(result["end_sim_time_ns"], 104_000_000_000)

    def test_progress_filter_excludes_early_solved_states(self):
        selected = filter_progress_rows(telemetry(), 10.0)
        self.assertEqual([row["unwrapped_progress_m"] for row in selected], [10.0, 12.0, 13.0])

    def test_final_failure_divergence_ignores_recovered_early_excursion(self):
        times = [index * 0.1 for index in range(15)]
        ctes = [0.01, 0.04, 0.05, 0.06, 0.02, 0.01, 0.01, 0.02, 0.01, 0.02, 0.04, 0.06, 0.10, 0.20, 0.30]
        result = detect_final_failure_divergence(times, ctes, stable_window_s=0.0, cte_floor_m=0.03, persistence_samples=5)
        self.assertEqual(result["divergence_index"], 10)
        self.assertIn("final continuous", result["method"])

    def test_shadow_alignment_still_never_uses_future_label(self):
        rows = [{"sim_time_s": 1.0, "shadow_expert_steering_rad": 0.1}, {"sim_time_s": 1.1, "shadow_expert_steering_rad": 0.2}]
        selected = latest_causal_shadow(rows, 1_050_000_000)
        self.assertEqual(selected["shadow_expert_steering_rad"], 0.1)

    def test_v3_policy_runtime_failure_safe_stops(self):
        class Client:
            def camera_jpeg(self, path="/camera"): raise RuntimeError("camera unavailable")
            def safe_stop(self): self.stopped = True; return []
        config = load_config(ROOT / "configs/pilotnet_dagger_iteration2_v1.json"); config["expected_world"] = "world"
        client = Client()
        with tempfile.TemporaryDirectory() as directory:
            result, _ = run_live_loop(client, object(), config, initial_state(), Path(directory) / "frames", policy_name="PilotNet V3")
        self.assertEqual(result["result"], "FAIL"); self.assertTrue(result["safe_stop_success"]); self.assertTrue(client.stopped)
        self.assertEqual(result["policy_controlling_vehicle"], "PilotNet V3 only")


class V4CompositionTests(unittest.TestCase):
    def test_iteration2_split_and_no_role_swap(self):
        metadata = {"episodes": [{"rollout_id": "dagger_iter2_rollout_A", "role": "training"}, {"rollout_id": "dagger_iter2_rollout_B", "role": "holdout"}]}
        config = {"dagger2_training_rollout": "dagger_iter2_rollout_A", "dagger2_holdout_rollout": "dagger_iter2_rollout_B"}
        self.assertEqual(iteration2_split(metadata, config), {"training": "dagger_iter2_rollout_A", "holdout": "dagger_iter2_rollout_B"})

    def test_cumulative_composition_retains_dagger1_and_excludes_v2_and_holdouts(self):
        nominal = [{"episode_id": "episode_001"}]
        d1 = [{"episode_id": "dagger_rollout_A", "source_mcap_sha256": "a"}]
        d2 = [{"episode_id": "dagger_iter2_rollout_A", "source_mcap_sha256": "b"}]
        holdout = [{"episode_id": "dagger_iter2_rollout_B", "source_mcap_sha256": "c"}]
        result = validate_cumulative_composition(nominal, d1, d2, holdout)
        self.assertTrue(result["dagger1_retained"]); self.assertTrue(result["v2_recovery_excluded"]); self.assertFalse(result["holdout_leakage"])

    def test_holdout_source_leakage_is_rejected(self):
        row = {"episode_id": "dagger_iter2_rollout_A", "source_mcap_sha256": "same"}
        with self.assertRaisesRegex(GateFailure, "leaked"):
            validate_cumulative_composition([], [], [row], [{"episode_id": "dagger_iter2_rollout_B", "source_mcap_sha256": "same"}])

    def test_v4_config_preserves_nominal_split_and_excludes_recovery(self):
        config = json.loads((ROOT / "configs/pilotnet_training_v4_dagger.json").read_text())
        self.assertEqual(config["train_episodes"], ["episode_001", "episode_002"])
        self.assertEqual(config["validation_episodes"], ["episode_003"])
        self.assertEqual(config["dagger1_training_rollout"], "dagger_rollout_A")
        self.assertFalse(any("recovery" in str(value) for value in config.values()))

    def test_v1_v3_v4_architecture_contract_is_equal(self):
        models = [build_pilotnet() for _ in range(3)]
        self.assertTrue(all(sum(parameter.numel() for parameter in model.parameters()) == PILOTNET_PARAMETER_COUNT for model in models))
        self.assertEqual(list(models[0].state_dict()), list(models[2].state_dict()))


class V4LiveGateTests(unittest.TestCase):
    class Config:
        payload = {"smoke_speeds_mps": [0.5, 0.5, 0.5]}
        def safety_config(self, speed): return speed

    def test_first_v4_failure_stops_all_later_runs(self):
        with (patch("physicar_e2e.pilotnet_dagger_iteration2_inference.wait_after_reset", return_value=object()),
              patch("physicar_e2e.pilotnet_dagger_iteration2_inference.run_smoke", return_value={"result": "FAIL"}) as run):
            results = run_conditional_v4(object(), object(), self.Config())
        self.assertEqual(len(results), 1); self.assertEqual(run.call_count, 1)

    def test_v4_maximum_is_three_runs(self):
        with (patch("physicar_e2e.pilotnet_dagger_iteration2_inference.wait_after_reset", return_value=object()),
              patch("physicar_e2e.pilotnet_dagger_iteration2_inference.run_smoke", return_value={"result": "PASS"}) as run):
            results = run_conditional_v4(object(), object(), self.Config())
        self.assertEqual(len(results), 3); self.assertEqual(run.call_count, 3)

    def test_second_v4_failure_prevents_third_run(self):
        outcomes = iter([{"result": "PASS"}, {"result": "FAIL"}])
        with (patch("physicar_e2e.pilotnet_dagger_iteration2_inference.wait_after_reset", return_value=object()),
              patch("physicar_e2e.pilotnet_dagger_iteration2_inference.run_smoke", side_effect=lambda *_: next(outcomes)) as run):
            results = run_conditional_v4(object(), object(), self.Config())
        self.assertEqual([result["result"] for result in results], ["PASS", "FAIL"])
        self.assertEqual(run.call_count, 2)

    def test_repeatability_requires_three_passes(self):
        self.assertIsNone(aggregate([{"result": "PASS"}]))


if __name__ == "__main__": unittest.main()
