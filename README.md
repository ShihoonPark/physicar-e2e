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
