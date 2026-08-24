import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import _bootstrap  # noqa: F401
from physicar_e2e.pilotnet import (
    MAX_STEERING_RAD,
    PILOTNET_PARAMETER_COUNT,
    build_pilotnet,
    clamp_steering_rad,
    preprocess_live_jpeg,
    preprocess_rgb,
    rgb_to_yuv_bt601,
    steering_normalized_to_rad,
    steering_rad_to_normalized,
)


class PilotNetContractTests(unittest.TestCase):
    def test_exact_input_output_and_parameter_count(self):
        model = build_pilotnet().eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 3, 66, 200))
        self.assertEqual(tuple(output.shape), (2, 1))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 252_219)
        self.assertEqual(PILOTNET_PARAMETER_COUNT, 252_219)

    def test_preprocessing_is_deterministic_float32_chw(self):
        rgb = np.arange(66 * 200 * 3, dtype=np.uint8).reshape(66, 200, 3)
        first = preprocess_rgb(rgb)
        second = preprocess_rgb(rgb.copy())
        self.assertEqual(first.shape, (3, 66, 200))
        self.assertEqual(first.dtype, np.float32)
        np.testing.assert_array_equal(first, second)

    def test_rgb_to_yuv_known_primary_values(self):
        rgb = np.asarray([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
        actual = rgb_to_yuv_bt601(rgb)
        expected = np.asarray(
            [[[0.299, 0.35287, 1.115], [0.587, 0.21114, -0.01499], [0.114, 0.936, 0.39999]]],
            dtype=np.float32,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-6)

    def test_steering_normalization_round_trip_and_runtime_clamp(self):
        radians = np.asarray([-MAX_STEERING_RAD, -0.1, 0.0, 0.2, MAX_STEERING_RAD], dtype=np.float32)
        np.testing.assert_allclose(
            steering_normalized_to_rad(steering_rad_to_normalized(radians)), radians, atol=1e-7
        )
        self.assertEqual(clamp_steering_rad(1.0), MAX_STEERING_RAD)
        self.assertEqual(clamp_steering_rad(-1.0), -MAX_STEERING_RAD)

    def test_live_jpeg_crop_resize_contract(self):
        image = Image.new("RGB", (480, 360), (20, 80, 140))
        encoded = io.BytesIO()
        image.save(encoded, format="JPEG", quality=90)
        tensor = preprocess_live_jpeg(encoded.getvalue())
        self.assertEqual(tensor.shape, (3, 66, 200))
        self.assertEqual(tensor.dtype, np.float32)

    def test_checkpoint_round_trip(self):
        source = build_pilotnet()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"model_state_dict": source.state_dict()}, path)
            restored = build_pilotnet()
            restored.load_state_dict(torch.load(path, weights_only=True)["model_state_dict"])
        for left, right in zip(source.parameters(), restored.parameters()):
            torch.testing.assert_close(left, right)


if __name__ == "__main__":
    unittest.main()
