#!/usr/bin/env python3
"""Run the read-only Real PhysiCar Bag Audit V2."""

from pathlib import Path
import sys

from physicar_e2e.real_bag_audit import main


if __name__ == "__main__":
    config = Path(__file__).resolve().parents[1] / "configs" / "real_bag_audit_v2.json"
    raise SystemExit(main(["--config", str(config), *sys.argv[1:]]))
