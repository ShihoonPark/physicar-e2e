import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import _bootstrap  # noqa: F401
from physicar_e2e.pilotnet import build_pilotnet
from physicar_e2e.pilotnet_training import (
    GateFailure,
    PilotDataset,
    export_onnx,
    load_config,
    validate_dataset_integrity,
    validate_onnx_equivalence,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "pilotnet_training_v1.json"


class DatasetAndTrainingTests(unittest.TestCase):
    def _dataset(self, root: Path, duplicate_path: bool = False):
        (root / "manifests").mkdir()
        (root / "images").mkdir()
        (root / "dataset_metadata.json").write_text("{}\n", encoding="utf-8")
        for index, episode in enumerate(("episode_001", "episode_002", "episode_003")):
            image_name = "shared.png" if duplicate_path else f"{episode}.png"
            image_path = root / "images" / image_name
            if not image_path.exists():
                Image.new("RGB", (200, 66), (index * 40, 20, 10)).save(image_path)
            with (root / "manifests" / f"{episode}.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["episode_id", "sample_index", "image_path", "steering_rad"])
                writer.writeheader()
                writer.writerow({
                    "episode_id": episode, "sample_index": 0,
                    "image_path": f"images/{image_name}", "steering_rad": 0.01 * index,
                })

    def test_canonical_episode_split_has_no_frame_level_leakage(self):
        config = load_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._dataset(root)
            result = validate_dataset_integrity(root, config)
        self.assertEqual(result["training_samples"], 2)
        self.assertEqual(result["validation_samples"], 1)
        self.assertEqual(result["train_episodes"], ["episode_001", "episode_002"])
        self.assertEqual(result["validation_episodes"], ["episode_003"])
        self.assertTrue(result["episode_level_separation"])

    def test_duplicate_image_across_episodes_is_rejected(self):
        config = load_config(CONFIG)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._dataset(root, duplicate_path=True)
            with self.assertRaisesRegex(GateFailure, "leakage/duplicate"):
                validate_dataset_integrity(root, config)

    def test_training_batch_shape_and_finite_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            Image.new("RGB", (200, 66), (30, 60, 90)).save(path)
            dataset = PilotDataset([{"image_path": path, "steering_rad": 0.1}], 0.349066)
            image, target = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 66, 200))
        self.assertEqual(tuple(target.shape), (1,))
        prediction = build_pilotnet()(image.unsqueeze(0))
        loss = torch.nn.functional.mse_loss(prediction, target.unsqueeze(0))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_onnx_contract_and_pytorch_equivalence_helper(self):
        config = load_config(CONFIG)
        model = build_pilotnet().eval()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "frame.png"
            Image.new("RGB", (200, 66), (12, 34, 56)).save(image_path)
            onnx_path = root / "model.onnx"
            export_onnx(model, onnx_path, config)
            result = validate_onnx_equivalence(
                model, [{"image_path": image_path, "steering_rad": 0.0}], onnx_path, config
            )
        self.assertEqual(result["result"], "PASS")
        self.assertLessEqual(result["max_absolute_difference_normalized"], config["onnx_max_abs_difference_limit"])


if __name__ == "__main__":
    unittest.main()
