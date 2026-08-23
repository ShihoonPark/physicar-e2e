# Lane-Follow Environment V1

Lane-Follow Environment V1 is a simulation-only derivative of the published
canonical world `custom_71e69ee938032295503bfed557fde18c`. It removes only the
six top-level models `cone1` through `cone6` for the lane-following E2E
baseline. The route, walls, light, track geometry, vehicle configuration, and
canonical source assets remain unchanged.

The simulator intentionally ignores custom world, route, and model assets as
generated/user data. Consequently, the derived assets are not force-added to
the simulator repository. This script and its explicit configuration are the
reproducible source of truth for recreating the runtime-validated environment:

```bash
python3 scripts/setup_lane_follow_environment_v1.py \
  --sim-root ~/physicar-ai-sim-docker
```

The command validates the canonical world, copies its route byte-for-byte,
copies its model metadata directory as-is, and creates a byte-preserving world
edit with the derived internal name and the six cone blocks removed. If valid
derived assets already exist, it exits successfully without rewriting them. If
existing derived assets are inconsistent, it fails unless `--force` is given;
that option replaces only the three derived asset targets.

Read-only verification is available separately:

```bash
python3 scripts/setup_lane_follow_environment_v1.py \
  --sim-root ~/physicar-ai-sim-docker \
  --verify-only
```

Verification checks both identities, canonical cone presence, zero derived
cones, the exact expected world transformation, route byte equality, and the
complete model-directory manifest. It does not run the vehicle and does not
claim real-robot validation.
