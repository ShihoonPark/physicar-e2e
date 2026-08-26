#!/usr/bin/env python3
"""Run the read-only Real PhysiCar camera/steering phase audit V1."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from physicar_e2e.real_camera_steering_phase_audit import main


if __name__ == "__main__":
    raise SystemExit(main())
