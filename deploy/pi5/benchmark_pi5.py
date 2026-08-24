#!/usr/bin/env python3
from pilotnet_pi5 import main
import sys

if __name__ == "__main__":
    raise SystemExit(main([*sys.argv[1:], "--benchmark"]))
