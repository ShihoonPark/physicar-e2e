"""Standalone entry point; delegates to the adjacent self-contained deployment core."""
from pilotnet_pi5_core import *  # noqa: F401,F403
from pilotnet_pi5_core import main

if __name__ == "__main__":
    raise SystemExit(main())
