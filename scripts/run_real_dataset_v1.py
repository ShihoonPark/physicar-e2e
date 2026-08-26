#!/usr/bin/env python3
"""Run the approved Real Dataset Extraction V1 only; never invoke training."""

from physicar_e2e.real_dataset import main


if __name__ == "__main__":
    raise SystemExit(main())
