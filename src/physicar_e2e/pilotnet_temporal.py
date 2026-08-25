"""Minimal causal three-frame PilotNet extension."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .pilotnet import IMAGE_HEIGHT, IMAGE_WIDTH, preprocess_live_jpeg, preprocess_png


TEMPORAL_FRAMES = 3
TEMPORAL_CHANNELS = 9
TEMPORAL_PARAMETER_COUNT = 255_819
MAX_ADJACENT_GAP_S = 0.120


def build_temporal_pilotnet():
    import torch.nn as nn

    class TemporalPilotNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(9, 24, kernel_size=5, stride=2), nn.ReLU(),
                nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
                nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
                nn.Conv2d(48, 64, kernel_size=3), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3), nn.ReLU(),
            )
            self.regressor = nn.Sequential(
                nn.Flatten(), nn.Linear(64 * 1 * 18, 100), nn.ReLU(),
                nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10), nn.ReLU(), nn.Linear(10, 1),
            )

        def forward(self, image):
            return self.regressor(self.features(image))

    model = TemporalPilotNet()
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != TEMPORAL_PARAMETER_COUNT:
        raise RuntimeError(f"Temporal PilotNet parameter contract changed: {count}")
    return model


def preprocess_temporal_paths(paths: Sequence[str | Path]) -> np.ndarray:
    if len(paths) != TEMPORAL_FRAMES:
        raise ValueError("temporal input requires exactly three real frames")
    value = np.concatenate([preprocess_png(path) for path in paths], axis=0)
    if value.shape != (TEMPORAL_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH):
        raise ValueError(f"invalid temporal tensor shape {value.shape}")
    return np.ascontiguousarray(value, dtype=np.float32)


@dataclass(frozen=True)
class BufferedFrame:
    timestamp_s: float
    tensor: np.ndarray


class CausalFrameBuffer:
    """Three independent acquisitions; no duplicate-frame startup padding."""

    def __init__(self, maximum_adjacent_gap_s: float = MAX_ADJACENT_GAP_S) -> None:
        if maximum_adjacent_gap_s != MAX_ADJACENT_GAP_S:
            raise ValueError("temporal gap gate must remain exactly 0.120 s")
        self.maximum_adjacent_gap_s = maximum_adjacent_gap_s
        self._frames: deque[BufferedFrame] = deque(maxlen=TEMPORAL_FRAMES)

    def append(self, timestamp_s: float, tensor: np.ndarray) -> None:
        value = np.asarray(tensor, dtype=np.float32)
        if value.shape != (3, IMAGE_HEIGHT, IMAGE_WIDTH):
            raise ValueError("buffer frame must be 3x66x200")
        if self._frames:
            gap = float(timestamp_s) - self._frames[-1].timestamp_s
            if gap <= 0 or gap > self.maximum_adjacent_gap_s:
                raise TemporalInputError(f"adjacent camera gap {gap:.6f}s outside (0, 0.120]")
        self._frames.append(BufferedFrame(float(timestamp_s), np.ascontiguousarray(value)))

    @property
    def ready(self) -> bool:
        return len(self._frames) == TEMPORAL_FRAMES

    def tensor(self) -> np.ndarray:
        if not self.ready:
            raise TemporalInputError("three real camera frames are not available")
        return np.ascontiguousarray(np.concatenate([frame.tensor for frame in self._frames], axis=0))

    def gaps(self) -> tuple[float, float, float]:
        if not self.ready:
            raise TemporalInputError("three real camera frames are not available")
        times = [frame.timestamp_s for frame in self._frames]
        return times[1] - times[0], times[2] - times[1], times[2] - times[0]


class TemporalInputError(RuntimeError):
    pass


def append_live_jpeg(buffer: CausalFrameBuffer, jpeg: bytes, timestamp_s: float,
                     roi=(0, 160, 480, 360)) -> np.ndarray:
    frame = preprocess_live_jpeg(jpeg, roi=roi)
    buffer.append(timestamp_s, frame)
    return frame
