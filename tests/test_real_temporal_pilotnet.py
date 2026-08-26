import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from physicar_e2e.real_temporal_pilotnet import (
    DRIVING_PERMITTED,
    EXPECTED_D1_CHECKPOINT_SHA256,
    EXPECTED_MANIFEST_SHA256,
    MAGNITUDE_BIN_LABELS,
    MODEL_NAMES,
    RAW_BAG_ACCESS_REQUIRED,
    SCRATCH_NAME,
    SIMULATOR_TRAINING_SAMPLES_PERMITTED,
    TRANSFER_NAME,
    RealTemporalDataset,
    RealTemporalTrainingError,
    audit_dataset,
    export_and_check_onnx,
    initialize_model,
    keep_train_row_by_speed,
    load_config,
    model_training_config,
    select_model,
    sha256_file,
    state_dict_sha256,
    verify_d1_artifacts,
)
from physicar_e2e.pilotnet_temporal import (
    TEMPORAL_PARAMETER_COUNT,
    build_temporal_pilotnet,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "real_temporal_pilotnet_v1.json"
MANIFEST_PATH = Path(
    "/home/a/physicar-e2e-artifacts/real_dataset_v1/manifests/real_dataset_v1.csv"
)


class RealTemporalDatasetContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG_PATH)
        cls.audit = audit_dataset(cls.config)

    def test_exact_real_dataset_manifest_hash_and_count(self):
        self.assertEqual(sha256_file(MANIFEST_PATH), EXPECTED_MANIFEST_SHA256)
        self.assertEqual(self.audit.evidence["accepted_sequence_count"], 2163)
        self.assertEqual(
            self.audit.evidence["bag_sequence_counts"],
            {"bag_01": 649, "bag_02": 1064, "bag_03": 450},
        )

    def test_grouped_bag_split_and_no_frame_leakage(self):
        self.assertEqual({row["source_bag"] for row in self.audit.train_rows}, {"bag_01", "bag_02"})
        self.assertEqual({row["source_bag"] for row in self.audit.validation_rows}, {"bag_03"})
        train_frames = {path for row in self.audit.train_rows for path in row["paths"]}
        validation_frames = {path for row in self.audit.validation_rows for path in row["paths"]}
        self.assertFalse(train_frames & validation_frames)
        self.assertEqual(self.audit.evidence["train_validation_frame_overlap_count"], 0)
        self.assertFalse(self.audit.evidence["random_frame_split_used"])

    def test_exact_zero_speed_filter_only(self):
        self.assertFalse(keep_train_row_by_speed(0.0))
        self.assertFalse(keep_train_row_by_speed(-0.0))
        for value in (-0.3, -0.1, 0.1, 0.2, 0.3):
            self.assertTrue(keep_train_row_by_speed(value))
        self.assertEqual(len(self.audit.train_rows), 1694)
        self.assertEqual(self.audit.evidence["train_exact_zero_speed_removed_count"], 19)
        self.assertEqual(
            self.audit.evidence["train_exact_zero_speed_removed_by_bag"],
            {"bag_01": 17, "bag_02": 2},
        )
        self.assertEqual(len(self.audit.validation_rows), 450)
        self.assertFalse(self.audit.evidence["validation_filter_applied"])

    def test_speed_is_never_neural_input_or_target(self):
        dataset = RealTemporalDataset(self.audit.train_rows[:1], cache_frames=False)
        image, target = dataset[0]
        self.assertEqual(dataset.neural_input_fields, ("image_t_minus_2", "image_t_minus_1", "image_t"))
        self.assertEqual(dataset.target_field, "steering_rad")
        self.assertEqual(dataset.metadata_excluded_fields, ("speed_mps",))
        self.assertEqual(tuple(image.shape), (9, 66, 200))
        self.assertEqual(tuple(target.shape), (1,))
        self.assertFalse(self.config["speed_contract"]["neural_input"])
        self.assertFalse(self.config["speed_contract"]["target"])

    def test_steering_is_already_scaled_exactly_once_and_not_clipped(self):
        row = self.audit.train_rows[0]
        self.assertAlmostEqual(
            row["steering_rad"], row["steering_recorded_raw"] * 0.35, places=15
        )
        _, target = RealTemporalDataset([row], cache_frames=False)[0]
        self.assertEqual(float(target.item()), float(np.float32(row["steering_rad"])))
        self.assertFalse(self.config["steering_contract"]["additional_scaling_permitted"])
        self.assertFalse(self.config["steering_contract"]["target_clipping_permitted"])

    def test_real_roi_and_preprocessing_contract(self):
        camera = self.config["camera_contract"]
        self.assertEqual(camera["roi"], [0, 80, 480, 360])
        self.assertEqual((camera["crop_width"], camera["crop_height"]), (480, 280))
        self.assertEqual((camera["output_width"], camera["output_height"]), (200, 66))
        self.assertFalse(camera["horizontal_crop"])
        self.assertFalse(camera["undistortion"])
        self.assertFalse(camera["simulator_roi_permitted"])
        self.assertEqual(self.config["preprocessing"]["frame_order"], ["t_minus_2", "t_minus_1", "t"])


class RealTemporalModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG_PATH)

    def test_temporal_input_and_exact_parameter_count(self):
        model = build_temporal_pilotnet()
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 255819)
        output = model(torch.zeros((2, 9, 66, 200), dtype=torch.float32))
        self.assertEqual(tuple(output.shape), (2, 1))
        self.assertEqual(TEMPORAL_PARAMETER_COUNT, 255819)

    def test_scratch_initialization_is_deterministic_and_not_d1(self):
        first, first_audit = initialize_model(SCRATCH_NAME, self.config, torch.device("cpu"))
        second, second_audit = initialize_model(SCRATCH_NAME, self.config, torch.device("cpu"))
        self.assertTrue(first_audit["initialized_from_scratch"])
        self.assertFalse(first_audit["tensor_exact_d1_initialization"])
        self.assertEqual(first_audit["initial_state_dict_sha256"], second_audit["initial_state_dict_sha256"])
        self.assertEqual(state_dict_sha256(first.state_dict()), state_dict_sha256(second.state_dict()))
        self.assertNotEqual(
            first_audit["initial_state_dict_sha256"], first_audit["d1_source_state_dict_sha256"]
        )

    def test_transfer_initialization_is_exact_d1_and_full_network_trainable(self):
        model, audit = initialize_model(TRANSFER_NAME, self.config, torch.device("cpu"))
        self.assertTrue(audit["tensor_exact_d1_initialization"])
        self.assertFalse(audit["initialized_from_scratch"])
        self.assertEqual(audit["initial_state_dict_sha256"], audit["d1_source_state_dict_sha256"])
        self.assertEqual(audit["total_parameter_count"], 255819)
        self.assertEqual(audit["trainable_parameter_count"], 255819)
        self.assertEqual(audit["frozen_parameter_count"], 0)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        identity = verify_d1_artifacts(self.config)
        self.assertEqual(identity["checkpoint"]["sha256_observed"], EXPECTED_D1_CHECKPOINT_SHA256)

    def test_d1_r_d2_fe_and_other_initializations_are_forbidden(self):
        self.assertEqual(
            self.config["d1_initialization"]["forbidden_initializations"],
            ["D1-R", "D2-FE", "R1", "V9", "C1"],
        )
        changed = copy.deepcopy(self.config)
        changed["d1_initialization"]["checkpoint_path"] = "/tmp/D1-R.pt"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(RealTemporalTrainingError):
                load_config(path)

    def test_optimization_is_real_train_only_with_no_speed_or_sim_samples(self):
        for model_name in MODEL_NAMES:
            training = model_training_config(model_name, self.config)
            self.assertEqual(training["optimization_sources"], ["bag_01", "bag_02"])
            self.assertEqual(training["simulator_samples_in_optimization"], 0)
            self.assertFalse(training["speed_is_neural_input"])
            self.assertFalse(training["speed_is_target"])
            self.assertEqual(training["target"], "steering_rad_from_manifest_without_scaling_or_clipping")

    def test_onnx_checker_shape_parameter_and_equivalence(self):
        audit = audit_dataset(self.config)
        model, _ = initialize_model(SCRATCH_NAME, self.config, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as directory:
            evidence = export_and_check_onnx(
                SCRATCH_NAME, model, audit.validation_rows, self.config, Path(directory)
            )
        self.assertEqual(evidence["contract"]["input_shape"], ["N", 9, 66, 200])
        self.assertEqual(evidence["contract"]["output_shape"], ["N", 1])
        self.assertEqual(evidence["contract"]["parameter_count"], 255819)
        self.assertEqual(evidence["contract"]["checker"], "PASS")
        self.assertEqual(evidence["equivalence"]["result"], "PASS")


def comparison_fixture(
    scratch_mae=0.05,
    transfer_mae=0.06,
    scratch_rmse=0.07,
    transfer_rmse=0.08,
    scratch_p95=0.12,
    transfer_p95=0.14,
    scratch_high=0.08,
    transfer_high=0.10,
    scratch_sign=0.8,
    transfer_sign=0.75,
):
    def model(mae, rmse, p95, high, sign):
        overall = {
            "mae_rad": mae,
            "rmse_rad": rmse,
            "p95_absolute_error_rad": p95,
            "bias_rad": 0.002,
            "steering_sign_agreement": sign,
            "corrective_magnitude_ratio": 0.95,
        }
        bins = {
            key: {"combined": {"mae_rad": high if key == "abs_gte_0p25" else mae}}
            for key in MAGNITUDE_BIN_LABELS
        }
        return {"overall": overall, "by_target_magnitude_and_direction": bins}

    return {
        "models": {
            SCRATCH_NAME: model(scratch_mae, scratch_rmse, scratch_p95, scratch_high, scratch_sign),
            TRANSFER_NAME: model(transfer_mae, transfer_rmse, transfer_p95, transfer_high, transfer_sign),
        }
    }


class RealTemporalSelectionAndSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(CONFIG_PATH)

    def test_model_selection_uses_mae_without_transfer_preference(self):
        result = select_model(comparison_fixture(), self.config)
        self.assertEqual(result["status"], "SELECTED")
        self.assertEqual(result["selected_model"], SCRATCH_NAME)
        self.assertTrue(result["selection_does_not_prefer_transfer_by_identity"])

    def test_model_selection_can_be_inconclusive_on_near_tie_conflict(self):
        fixture = comparison_fixture(
            scratch_mae=0.0500,
            transfer_mae=0.0502,
            scratch_rmse=0.080,
            transfer_rmse=0.070,
            scratch_p95=0.15,
            transfer_p95=0.12,
            scratch_high=0.12,
            transfer_high=0.08,
            scratch_sign=0.80,
            transfer_sign=0.80,
        )
        # Split the remaining two metrics; the sign metric ties, so neither reaches four wins.
        fixture["models"][SCRATCH_NAME]["overall"]["bias_rad"] = 0.001
        fixture["models"][TRANSFER_NAME]["overall"]["bias_rad"] = 0.002
        fixture["models"][SCRATCH_NAME]["overall"]["corrective_magnitude_ratio"] = 0.99
        fixture["models"][TRANSFER_NAME]["overall"]["corrective_magnitude_ratio"] = 0.90
        result = select_model(fixture, self.config)
        self.assertEqual(result["status"], "MODEL_SELECTION_INCONCLUSIVE")
        self.assertIsNone(result["selected_model"])

    def test_no_real_or_simulator_driving_surface(self):
        self.assertFalse(DRIVING_PERMITTED)
        self.assertFalse(SIMULATOR_TRAINING_SAMPLES_PERMITTED)
        self.assertFalse(RAW_BAG_ACCESS_REQUIRED)
        self.assertTrue(all(value is False for value in self.config["prohibitions"].values()))
        source = (ROOT / "src" / "physicar_e2e" / "real_temporal_pilotnet.py").read_text()
        self.assertNotIn("from .sim_client import", source)
        self.assertNotIn("DockerRosBackend", source)
        self.assertNotIn("publish_steering", source)
        self.assertNotIn("start_recorder", source)


if __name__ == "__main__":
    unittest.main()
