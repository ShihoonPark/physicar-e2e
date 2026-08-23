# PhysiCar E2E

Reproducible end-to-end autonomous-driving research for PhysiCar. The first
milestone is a privileged geometric expert used only for simulator data
generation and evaluation; future camera models must not receive route or GT
pose inputs.

## Expert Driver V1

The source-checkout launcher supplies the repository's canonical configuration:

```bash
python3 scripts/run_expert_driver_v1.py --preflight-only
python3 scripts/run_expert_driver_v1.py --dry-run 3
python3 scripts/run_expert_driver_v1.py --result results/expert_driver_v1_latest.json
```

The installed console command deliberately requires an explicit configuration
path so it never depends on the caller's working directory:

```bash
physicar-expert-v1 --config /path/to/expert_driver_v1.json --preflight-only
```

The driving command verifies the expected derived world, runs one lap at the
configured low speed, and best-effort publishes zero speed and steering from a
`finally` block. See `docs/expert_driver_v1.md` for safety and metric details.

`results/expert_driver_v1_pre_safety_run.json` is historical evidence from the
earlier successful lap. It predates the pose/clock liveness watchdog and is not
the canonical result for the current implementation. A new canonical result is
intentionally pending final review and an explicitly authorized lap.

Unit tests require no simulator or ROS installation:

```bash
python3 -m unittest discover -s tests -v
```

## Automated Rosbag Collector V1

The collector records one external MCAP bag per canonical expert lap. It uses
the ROS 2 Jazzy installation inside the existing simulator container and
discovers the userdata host bind mount before writing. Raw bags and recorder
logs remain under simulator `userdata`, while compact JSON evidence is written
under `results/rosbag_collector_v1_pilot/`.

Run its read-only asset/runtime/topic preflight before collection:

```bash
python3 scripts/run_rosbag_collector_v1.py \
  --sim-root ~/physicar-ai-sim-docker --preflight-only
```

The explicitly authorized V1 pilot command is:

```bash
python3 scripts/run_rosbag_collector_v1.py \
  --sim-root ~/physicar-ai-sim-docker --episodes 3
```

See `docs/rosbag_collector_v1.md` for lifecycle, failure, storage, and metadata
details. The collector performs no image preprocessing or synchronization.

## Lane-Follow Environment V1

The cone-free, simulation-only baseline environment is reproducibly generated
from its preserved canonical simulator world. See
`docs/lane_follow_environment_v1.md` for generation and read-only verification
commands; the generated simulator assets remain intentionally ignored there.
