# PilotNet Failure Diagnosis V1

This milestone diagnoses—without retraining—why canonical PilotNet V1 fails near the first 2–3 m of the route at 0.50 m/s despite low nominal offline error and a successful 0.30 m/s lap. Existing V1 and V2 evidence is immutable input to the experiment.

## Fixed contracts

- PilotNet remains the 252,219-parameter `3×66×200` model.
- Training/offline and live paths share the direct full-range BT.601-style RGB→YUV matrix and `(channel - 0.5) * 2` normalization in `pilotnet.py`.
- Live input is camera-only. GT pose, route, CTE, boundaries, and simulator clock are used only for safety and diagnostics.
- PilotNet V1 is the only steering authority. The canonical Pure Pursuit command is computed in shadow mode and cannot reach the command function.
- Exactly one 0.50 m/s live diagnostic run is permitted. An existing live result file blocks another run.
- The diagnostic bag contains only `/camera/image_raw` and `/clock`; it and HTTP frames remain outside Git and are never training data.

## Offline diagnostics

The ordered `episode_003` validation sequence is evaluated for linear calibration, signed bias, four absolute-steering bins, and fixed temporal shifts from −200 to +200 ms. Positive shift means a prediction at `t` is compared with the recorded expert label at `t + shift`. Labels are linearly interpolated using actual camera record timestamps; this diagnostic does not change training labels. The nominal manifests have no pose or route-progress association, so an offline 0–5 m section is not fabricated.

The final convolutional output (`64×1×18`) is the fixed feature representation. All nominal training frames are embedded with V1, L2-normalized, and compared to live features by nearest cosine distance. No pass threshold is assigned to that distance.

## Live and transport diagnostics

The one live run saves each HTTP JPEG with request/receive timestamps and logs V1 and shadow-expert steering. Divergence onset is the first persistent above-baseline CTE window with positive slope; the rule is fixed before examining the result. Raw ROS RGB frames are associated one-to-one with HTTP receive wall times within the configured tolerance. Exact frame identity is not claimed. Both transports then use the same ROI, bilinear resize, RGB→YUV conversion, normalization, model, and steering scale.

Run the offline gate first:

```bash
scripts/run_pilotnet_failure_diagnosis_v1.py --offline \
  --config configs/pilotnet_failure_diagnosis_v1.json \
  --dataset-root /path/to/dataset_extractor_v1_pilot \
  --v1-checkpoint /path/to/pilotnet_v1_best.pt \
  --v2-checkpoint /path/to/pilotnet_v2_recovery_best.pt \
  --artifact-root /external/physicar_e2e/pilotnet_failure_diagnosis_v1 \
  --result results/pilotnet_failure_diagnosis_v1/offline.json
```

The live mode requires explicit simulator/external-data authorization. It resets once, records once, drives V1 once, safe-stops, finalizes the bag, and performs the post-run analyses. It must never be retried in this milestone.

Simulator findings remain simulator-only evidence and do not establish Raspberry Pi or real-robot behavior.
