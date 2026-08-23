#!/usr/bin/env python3
"""Source-tree launcher for Automated Rosbag Collector V1."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physicar_e2e.rosbag_collector import main  # noqa: E402


def source_checkout_args(args: list[str]) -> list[str]:
    defaults: list[str] = []
    if "--config" not in args and not any(arg.startswith("--config=") for arg in args):
        defaults += ["--config", str(ROOT / "configs" / "rosbag_collector_v1.json")]
    if "--expert-config" not in args and not any(arg.startswith("--expert-config=") for arg in args):
        defaults += ["--expert-config", str(ROOT / "configs" / "expert_driver_v1.json")]
    if "--results-dir" not in args and not any(arg.startswith("--results-dir=") for arg in args):
        defaults += ["--results-dir", str(ROOT / "results" / "rosbag_collector_v1_pilot")]
    return [*defaults, *args]


if __name__ == "__main__":
    raise SystemExit(main(source_checkout_args(sys.argv[1:])))
