import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import _bootstrap  # noqa: F401
from physicar_e2e.high_speed_temporal import MAX_TOTAL_ATTEMPTS, run_attempts
from physicar_e2e.high_speed_temporal_balanced import (
    BIN_NAMES,
    InsufficientLateRegionDiversity,
    MAJOR_STRATA,
    balanced_subset,
    evenly_spaced_indices,
    load_config,
    offline_live_gate,
    route_bin,
)
from physicar_e2e.pilotnet_temporal import build_temporal_pilotnet


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/high_speed_temporal_balanced_v1.json"


def record(name, index):
    return {"identity": f"{name}:{index}", "window_role": name, "index": index}


def live_result(result="PASS"):
    return {
        "result": result,
        "temporal_input_failure": False,
        "api_failures": 0,
        "liveness_failures": 0,
        "safe_stop_success": True,
    }


class Client:
    def __init__(self):
        self.safe_stops = 0

    def safe_stop(self):
        self.safe_stops += 1
        return []


class LateBalanceContracts(unittest.TestCase):
    def test_v9_preserved_hashes_and_three_pass_evidence(self):
        config = load_config(CONFIG)
        training = json.loads(
            (ROOT / "results/pilotnet_training_v9_high_speed_temporal/summary.json").read_text()
        )
        live = json.loads(
            (ROOT / "results/pilotnet_e2e_v9_high_speed_temporal/summary.json").read_text()
        )
        self.assertEqual(
            training["artifacts"]["checkpoint"]["sha256"],
            config["preserved_v9"]["checkpoint_sha256"],
        )
        self.assertEqual(
            training["artifacts"]["onnx"]["sha256"],
            config["preserved_v9"]["onnx_sha256"],
        )
        self.assertEqual((live["result"], live["policy_pass_count"]), ("PASS", 3))

    def test_no_new_data_collection_or_followup_optimization(self):
        config = load_config(CONFIG)
        self.assertFalse(config["new_data_collection_permitted"])
        self.assertFalse(config["automatic_followup_optimization_permitted"])
        source = (ROOT / "src/physicar_e2e/high_speed_temporal_balanced.py").read_text()
        self.assertNotIn("DockerRosBackend", source)
        self.assertNotIn("start_recorder", source)

    def test_route_bins_are_exact(self):
        self.assertIsNone(route_bin(0.849999))
        self.assertEqual(route_bin(0.85), "85_90_percent")
        self.assertEqual(route_bin(0.899999), "85_90_percent")
        self.assertEqual(route_bin(0.90), "90_95_percent")
        self.assertEqual(route_bin(0.949999), "90_95_percent")
        self.assertEqual(route_bin(0.95), "95_100_percent")
        self.assertEqual(route_bin(1.0), "95_100_percent")
        self.assertIsNone(route_bin(1.000001))

    def test_k_is_minimum_and_below_twenty_stops(self):
        groups = {
            BIN_NAMES[0]: [record(BIN_NAMES[0], index) for index in range(1474)],
            BIN_NAMES[1]: [record(BIN_NAMES[1], index) for index in range(15)],
            BIN_NAMES[2]: [record(BIN_NAMES[2], index) for index in range(6)],
        }
        with self.assertRaisesRegex(InsufficientLateRegionDiversity, "K=6"):
            balanced_subset(groups)

    def test_evenly_spaced_selection_is_deterministic_and_unique(self):
        first = evenly_spaced_indices(1474, 20)
        second = evenly_spaced_indices(1474, 20)
        self.assertEqual(first, second)
        self.assertEqual((len(first), len(set(first)), first[0], first[-1]), (20, 20, 0, 1473))
        self.assertEqual(first, sorted(first))

    def test_balancing_undersamples_without_oversampling(self):
        groups = {
            BIN_NAMES[0]: [record(BIN_NAMES[0], index) for index in range(30)],
            BIN_NAMES[1]: [record(BIN_NAMES[1], index) for index in range(20)],
            BIN_NAMES[2]: [record(BIN_NAMES[2], index) for index in range(25)],
        }
        k, selected = balanced_subset(groups)
        self.assertEqual((k, len(selected), len({row["identity"] for row in selected})), (20, 60, 60))
        self.assertEqual(
            [sum(row["window_role"] == name for row in selected) for name in BIN_NAMES],
            [20, 20, 20],
        )

    def test_training_sources_exclude_all_b_and_low_speed_data(self):
        config = load_config(CONFIG)
        self.assertEqual(
            config["training_sources"],
            [
                "nominal_episode_001_008",
                "high_speed_dagger_rollout_A",
                "high_speed_dagger_iter2_rollout_A",
                "balanced_high_speed_dagger_iter3_rollout_A",
            ],
        )
        self.assertIn("all_B_holdouts", config["excluded_sources"])
        self.assertIn("low_speed_v4", config["excluded_sources"])
        self.assertTrue(all(source.endswith("_B") or source.startswith("nominal_")
                            for source in config["evaluation_sources"]))

    def test_temporal_contract_and_parameter_count_match_v9(self):
        config = load_config(CONFIG)
        v9 = json.loads((ROOT / "configs/high_speed_temporal_dataset_v1.json").read_text())
        temporal = config["temporal_contract"]
        self.assertEqual(
            (temporal["history_frames"], temporal["input_channels"], temporal["frame_order"],
             temporal["maximum_adjacent_gap_s"], temporal["future_frames_permitted"],
             temporal["boundary_crossing_permitted"], temporal["duplicate_history_padding"]),
            (v9["history_frames"], 9, v9["channel_order"], v9["maximum_adjacent_gap_s"],
             False, False, False),
        )
        self.assertEqual(sum(parameter.numel() for parameter in build_temporal_pilotnet().parameters()), 255_819)

    def test_training_from_scratch_and_hyperparameters_match_v9(self):
        config = load_config(CONFIG)["training_contract"]
        v9 = json.loads((ROOT / "configs/pilotnet_training_v9_high_speed_temporal.json").read_text())
        self.assertEqual(config["initialization"], "from_scratch")
        for key in ("loss", "optimizer", "learning_rate", "batch_size", "max_epochs", "seed"):
            self.assertEqual(config[key], v9[key])
        self.assertEqual(config["max_epochs"], 35)

    def test_matched_v9_v10_evaluation_sources_are_frozen(self):
        config = load_config(CONFIG)
        self.assertEqual(
            config["evaluation_sources"],
            [
                "nominal_episode_009_010",
                "nominal_episode_011_012",
                "high_speed_dagger_rollout_B",
                "high_speed_dagger_iter2_rollout_B",
                "high_speed_dagger_iter3_rollout_B",
            ],
        )

    def test_offline_gate_requires_late_non_regression(self):
        v9 = {name: 1.0 for name in MAJOR_STRATA}
        self.assertEqual(offline_live_gate(v9, dict(v9), 0.02, 0.02)["result"], "PASS")
        failed = offline_live_gate(v9, dict(v9), 0.02, 0.020001)
        self.assertIn("DAGGER3_B_90_100_MAE_WORSE", failed["reasons"])

    def test_offline_gate_blocks_catastrophic_major_regression(self):
        v9 = {name: 1.0 for name in MAJOR_STRATA}
        v10 = dict(v9)
        v10["dagger2_B"] = 1.500001
        failed = offline_live_gate(v9, v10, 0.02, 0.019)
        self.assertIn("CATASTROPHIC_MAE_REGRESSION:dagger2_B", failed["reasons"])


class LateBalanceLiveContracts(unittest.TestCase):
    def execute(self, outcomes):
        client = Client()
        remaining = iter(outcomes)

        def run_one(*_args):
            client.safe_stop()
            return next(remaining)

        before = Mock(return_value=(object(), {"result": "PASS"}))
        with tempfile.TemporaryDirectory() as directory:
            attempts, result = run_attempts(
                client, object(), object(), Path(directory),
                preflight_one=before, run_one=run_one,
            )
        return client, attempts, result

    def test_first_policy_failure_stops_and_safe_stops(self):
        client, attempts, result = self.execute([live_result("FAIL"), live_result()])
        self.assertEqual((len(attempts), result, client.safe_stops), (1, "FAIL", 1))

    def test_three_valid_passes_stop_within_attempt_budget(self):
        client, attempts, result = self.execute([live_result(), live_result(), live_result(), live_result()])
        self.assertEqual((len(attempts), result, client.safe_stops), (3, "PASS", 3))
        self.assertEqual(MAX_TOTAL_ATTEMPTS, 5)
        self.assertEqual(load_config(CONFIG)["live_contract"]["maximum_valid_policy_evaluations"], 3)


if __name__ == "__main__":
    unittest.main()
