#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from physicar_e2e.pilotnet_dagger_iteration2_training import main
if __name__ == "__main__": raise SystemExit(main())
