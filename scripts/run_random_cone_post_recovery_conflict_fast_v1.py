#!/usr/bin/env python3
"""Execute the bounded, read-only post-recovery conflict diagnosis."""

from __future__ import annotations

import argparse
from pathlib import Path

from physicar_e2e.random_cone_post_recovery_conflict_fast import load_config, run_diagnosis


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path)
    arguments = parser.parse_args()
    config = load_config(arguments.repo, arguments.config)
    summary = run_diagnosis(config)
    return 0 if summary["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
