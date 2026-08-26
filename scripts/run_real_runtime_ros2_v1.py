#!/usr/bin/env python3
"""Launch the ROS 2 adapter with the canonical Runtime V1 config."""

from __future__ import annotations

import sys
from pathlib import Path

from physicar_e2e.real_runtime_ros2 import main


def source_checkout_args(argv: list[str]) -> list[str]:
    if "--config" in argv:
        return argv
    config = Path(__file__).resolve().parents[1] / "configs" / "real_runtime_v1.json"
    return ["--config", str(config), *argv]


if __name__ == "__main__":
    raise SystemExit(main(source_checkout_args(sys.argv[1:])))
