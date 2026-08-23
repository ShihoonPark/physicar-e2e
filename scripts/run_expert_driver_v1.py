#!/usr/bin/env python3
"""Source-tree launcher for Expert Driver V1."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from physicar_e2e.expert_driver import main  # noqa: E402


def source_checkout_args(args: list[str]) -> list[str]:
    if "--config" in args or any(arg.startswith("--config=") for arg in args):
        return args
    return ["--config", str(ROOT / "configs" / "expert_driver_v1.json"), *args]


if __name__ == "__main__":
    raise SystemExit(main(source_checkout_args(sys.argv[1:])))
