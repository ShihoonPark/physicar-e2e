"""PilotNet V1 architecture and deterministic image/steering contracts."""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    import torch

IMAGE_HEIGHT = 66
IMAGE_WIDTH = 200
MAX_STEERING_RAD = 0.349066
PILOTNET_PARAMETER_COUNT = 252_219

# Full-range analog YUV coefficients applied to RGB values in [0, 1]. U and V
# are biased by 0.5 for storage as numeric channels. This is not JPEG YCbCr.
RGB_TO_YUV_BT601 = np.asarray(
    [
        [0.29900, 0.58700, 0.11400],
        [-0.14713, -0.28886, 0.43600],
        [0.61500, -0.51499, -0.10001],
    ],
    dtype=np.float32,
)


def rgb_to_yuv_bt601(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8/float RGB HWC to full-range BT.601-style YUV float32."""
    values = np.asarray(rgb)
    if values.ndim != 3 or values.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image, got {values.shape}")
    values = values.astype(np.float32)
    if values.size and values.max() > 1.0:
        values /= 255.0
    yuv = values @ RGB_TO_YUV_BT601.T
    yuv[..., 1:] += 0.5
    return yuv.astype(np.float32, copy=False)


def preprocess_rgb(rgb: np.ndarray) -> np.ndarray:
    """Return deterministic normalized CHW float32 input for a 200x66 RGB image."""
    values = np.asarray(rgb)
    if values.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise ValueError(f"expected RGB shape {(IMAGE_HEIGHT, IMAGE_WIDTH, 3)}, got {values.shape}")
    normalized = (rgb_to_yuv_bt601(values) - np.float32(0.5)) * np.float32(2.0)
    return np.ascontiguousarray(normalized.transpose(2, 0, 1), dtype=np.float32)


def preprocess_png(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return preprocess_rgb(rgb)


def preprocess_live_jpeg(
    jpeg: bytes,
    *,
    roi: tuple[int, int, int, int] = (0, 160, 480, 360),
) -> np.ndarray:
    """Decode HTTP JPEG, apply the extractor ROI/resize, then shared preprocessing."""
    with Image.open(io.BytesIO(jpeg)) as image:
        rgb_image = image.convert("RGB")
        if rgb_image.size != (480, 360):
            raise ValueError(f"live camera must be 480x360 before ROI, got {rgb_image.size}")
        resized = rgb_image.crop(roi).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.Resampling.BILINEAR)
        rgb = np.asarray(resized, dtype=np.uint8)
    return preprocess_rgb(rgb)


def steering_rad_to_normalized(value: float | np.ndarray, maximum: float = MAX_STEERING_RAD):
    return np.asarray(value, dtype=np.float32) / np.float32(maximum)


def steering_normalized_to_rad(value: float | np.ndarray, maximum: float = MAX_STEERING_RAD):
    return np.asarray(value, dtype=np.float32) * np.float32(maximum)


def clamp_steering_rad(value: float, maximum: float = MAX_STEERING_RAD) -> float:
    if not np.isfinite(value):
        raise ValueError("steering output must be finite")
    return float(max(-maximum, min(maximum, value)))


def build_pilotnet():
    """Build the exact NVIDIA PilotNet/DAVE-2-style V1 regressor."""
    import torch.nn as nn

    class PilotNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ReLU(),
                nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
                nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
                nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            )
            self.regressor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 1 * 18, 100), nn.ReLU(),
                nn.Linear(100, 50), nn.ReLU(),
                nn.Linear(50, 10), nn.ReLU(),
                nn.Linear(10, 1),
            )

        def forward(self, image: "torch.Tensor") -> "torch.Tensor":
            return self.regressor(self.features(image))

    model = PilotNet()
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != PILOTNET_PARAMETER_COUNT:
        raise RuntimeError(f"PilotNet parameter contract changed: {count}")
    return model
