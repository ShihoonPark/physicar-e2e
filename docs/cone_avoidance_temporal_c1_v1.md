# Cone Avoidance Temporal PilotNet C1 V1

This milestone is a simulation-only, fixed-one-cone experiment. It does not
establish real-robot safety or general cone avoidance.

## Frozen inputs

- Temporal PilotNet V9 is preserved at 1.80 m/s with causal `[t-2,t-1,t]`
  camera input and 255,819 parameters.
- Cone Avoidance Expert V1, its right-side bypass, the cone pose at route
  `s=6.9 m`, and the derived world are unchanged.
- The neural boundary accepts only three causal camera tensors. Pose, route,
  cone state, CTE, clearance, and recovery are evaluation-only.

## Pipeline

The integrated runner exposes explicit hard-gated stages:

```bash
cd ~/physicar-e2e

uv run python scripts/run_cone_avoidance_temporal_c1_v1.py \
  --sim-root ~/physicar-ai-sim-docker --stage audit

uv run python scripts/run_cone_avoidance_temporal_c1_v1.py \
  --sim-root ~/physicar-ai-sim-docker --stage collect

uv run python scripts/run_cone_avoidance_temporal_c1_v1.py \
  --sim-root ~/physicar-ai-sim-docker --stage dataset

uv run python scripts/run_cone_avoidance_temporal_c1_v1.py \
  --sim-root ~/physicar-ai-sim-docker --stage train

uv run python scripts/run_cone_avoidance_temporal_c1_v1.py \
  --sim-root ~/physicar-ai-sim-docker --stage live
```

`--stage reanalyze` exists only to reproducibly correct evaluation-only
odom-to-world route-s alignment. It does not modify raw bags, images, labels,
splits, checkpoint, ONNX, or training. Fresh pipeline execution applies the
correct transform during initial extraction.

Raw bags, extracted images, checkpoint, ONNX, and plots are stored under:

```text
/home/a/physicar-ai-sim-docker/userdata/physicar_e2e/cone_avoidance_v1/
```

They remain outside Git. The runner refuses to overwrite completed collection,
training, or bounded live evidence.

## Result

Collection was 12/12 Expert PASS and the offline C1 gate passed. The first
valid C1 live run was `CONE_POLICY_FAIL`: no physical intersection occurred,
but measured footprint clearance was 0.043583 m, below the frozen 0.050 m
contract. The bounded runner stopped after that one valid failure. C1 is not
frozen, runs two and three were not performed, and no DAgger or retraining was
started.

See `results/pilotnet_e2e_c1_cone_temporal/summary.json` and
`results/pilotnet_training_c1_cone_temporal/summary.json` for compact evidence.
