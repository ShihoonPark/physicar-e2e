# Real PhysiCar Bag Audit V1

Overall result: **FAIL_INCOMPLETE_INPUTS**. The local MCAP files are not complete, so every numeric finding below is explicitly limited to strictly readable prefixes. No result is presented as a complete-bag or real-robot performance result.

## 1. Integrity of all three bags

| Bag | Metadata duration | Metadata messages | Readable messages | Readable span | Closing magic | Strict scan | Result |
|---|---:|---:|---:|---:|---|---|---|
| bag_01 | 43.447676 s | 3043 | 412 | 5.876564 s | no | EndOfFile | **FAIL** |
| bag_02 | 72.767350 s | 4989 | 864 | 12.297079 s | no | EndOfFile | **FAIL** |
| bag_03 | 30.397140 s | 2120 | 616 | 8.735710 s | no | EndOfFile | **FAIL** |

All three files begin with valid MCAP magic. All end inside data rather than with closing MCAP magic; CRC checking succeeds for completed records and then raises `EndOfFile`. Metadata therefore cannot be verified against all messages.

### Paths and timestamps

- `bag_01`: `/home/a/bag_01/bag_01_0.mcap`
  - metadata: 1787684938669069319 (2026-08-25T19:08:58.669069319Z) to 1787684982116744998 (2026-08-25T19:09:42.116744998Z)
  - readable: 1787684938669069319 (2026-08-25T19:08:58.669069319Z) to 1787684944545633115 (2026-08-25T19:09:04.545633115Z)
- `bag_02`: `/home/a/bag_02/bag_02_0.mcap`
  - metadata: 1787685013611973884 (2026-08-25T19:10:13.611973884Z) to 1787685086379324121 (2026-08-25T19:11:26.379324121Z)
  - readable: 1787685013611973884 (2026-08-25T19:10:13.611973884Z) to 1787685025909053329 (2026-08-25T19:10:25.909053329Z)
- `bag_03`: `/home/a/bag_03/bag_03_0.mcap`
  - metadata: 1787685107512840513 (2026-08-25T19:11:47.512840513Z) to 1787685137909980841 (2026-08-25T19:12:17.909980841Z)
  - readable: 1787685107512840513 (2026-08-25T19:11:47.512840513Z) to 1787685116248550622 (2026-08-25T19:11:56.248550622Z)

### Topic counts and measured rates (readable prefixes)

| Bag | Camera expected/readable/rate | Steering expected/readable/rate | Speed expected/readable/rate |
|---|---:|---:|---:|
| bag_01 | 651/87/15.017 Hz | 653/89/14.997 Hz | 653/89/14.997 Hz |
| bag_02 | 1068/185/15.007 Hz | 1070/185/15.005 Hz | 1070/185/15.005 Hz |
| bag_03 | 454/132/15.007 Hz | 454/132/15.015 Hz | 454/132/15.015 Hz |

## 2. Exact camera contract

Every readable camera message in every bag is `480x360`, `rgb8`, `is_bigendian=0`, `step=1440`, `frame_id=camera`. Header stamps are present on every readable frame. This is a readable-prefix finding until complete MCAPs are supplied.

## 3. Camera rate and timestamp statistics

| Bag | Frames | FPS | gap mean | median | p95 | max | record monotonic | duplicate record/header | decode failures |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| bag_01 | 87 | 15.017 | 0.066593 s | 0.066687 s | 0.069727 s | 0.085031 s | yes | 0/0 | 0 |
| bag_02 | 185 | 15.007 | 0.066638 s | 0.066773 s | 0.069463 s | 0.074212 s | yes | 0/0 | 0 |
| bag_03 | 132 | 15.007 | 0.066634 s | 0.066652 s | 0.070125 s | 0.072461 s | yes | 0/0 | 0 |

## 4. Representative preview paths

- `bag_01` readable-prefix contact sheet: `/home/a/physicar-e2e-artifacts/real_bag_audit_v1/previews/bag_01/bag_01_contact_sheet.png`
- `bag_02` readable-prefix contact sheet: `/home/a/physicar-e2e-artifacts/real_bag_audit_v1/previews/bag_02/bag_02_contact_sheet.png`
- `bag_03` readable-prefix contact sheet: `/home/a/physicar-e2e-artifacts/real_bag_audit_v1/previews/bag_03/bag_03_contact_sheet.png`

The sheets contain uncropped real frames: first/early/middle/late/last within the readable prefix plus minimum/maximum observed steering frames. They do not claim to show the true middle or end of the metadata-declared bag.

Turn-like views are visible at numeric steering extrema (including a right-curving view near bag_03's negative extreme and left-curving views near positive extremes), but these visual associations do not establish the vehicle's left/right sign convention or whether the topic is command or feedback.

## 5. Steering numeric distributions

Unit and approximate numeric range are confirmed as radians and `[-0.35,+0.35] rad` by the user. Near-zero uses `abs(steering)<=0.01 rad`. The readable data conflicts with the confirmed approximate range, as shown below.

| Bag | Count/rate | min | p01 | p05 | p25 | mean | median | std | p75 | p95 | p99 | max | neg/zero/pos |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bag_01 | 89/14.997 Hz | -0.001336 | -0.000160 | 0.000000 | 0.198679 | 0.385808 | 0.326348 | 0.240144 | 0.651411 | 0.801825 | 0.890981 | 0.904927 | 0/7/82 |
| bag_02 | 185/15.005 Hz | -0.068039 | -0.064339 | -0.045952 | 0.025323 | 0.114940 | 0.052025 | 0.215605 | 0.069822 | 0.735348 | 0.805080 | 0.805080 | 26/11/148 |
| bag_03 | 132/15.015 Hz | -0.293672 | -0.262945 | -0.159494 | 0.055363 | 0.128290 | 0.134024 | 0.158479 | 0.211658 | 0.390903 | 0.476745 | 0.508245 | 21/3/108 |

Out-of-range readable samples: bag_01 35/89 outside, min/max=-0.001336/0.904927 rad; bag_02 20/185 outside, min/max=-0.068039/0.805080 rad; bag_03 10/132 outside, min/max=-0.293672/0.508245 rad.
Direct CDR Float64 unpacking exactly matched `mcap_ros2` decoding for all readable steering samples (mismatches=0), so this is not a decoder interpretation artifact.

Probable-saturation audit (numeric plateaus only; no actuator meaning inferred):
- bag_01: repeated out-of-range plateau candidates: +0.651411 rad x11; repeated candidates at the confirmed +/-0.35 rad limits: 0.
- bag_02: repeated out-of-range plateau candidates: +0.805080 rad x8, +0.735348 rad x3; repeated candidates at the confirmed +/-0.35 rad limits: 0.
- bag_03: repeated out-of-range plateau candidates: none; repeated candidates at the confirmed +/-0.35 rad limits: 0.

Full repetition details are preserved in each `bag_*.json`.

## 6. Unresolved steering semantics

Radians and the approximate range are confirmed, but the observed out-of-range values require source-team reconciliation before extraction. Left/right sign convention and command-vs-actual-feedback meaning remain unresolved. Repository steering evidence describes simulator control and does not prove the provenance of these real-vehicle Float64 messages.

Provenance evidence checked separately: `configs/rosbag_collector_v1.json` and `src/physicar_e2e/rosbag_collector.py` only enumerate recorded topics; `src/physicar_e2e/expert_driver.py` documents simulator command semantics. The bag metadata and MCAP channel metadata contain no real publisher/node identity, so none of those files proves real steering sign or command/feedback meaning.

## 7. Speed numeric distributions

| Bag | Count/rate | min | p01 | p05 | p25 | mean | median | std | p75 | p95 | p99 | max | negative/zero/positive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bag_01 | 89/14.997 Hz | 0.000000 | 0.000000 | 0.000000 | 0.534203 | 0.611150 | 0.721779 | 0.253580 | 0.783008 | 0.852771 | 0.947253 | 0.998500 | 0/6/83 |
| bag_02 | 185/15.005 Hz | 0.000000 | 0.200000 | 0.200000 | 0.927171 | 0.878513 | 0.944713 | 0.197417 | 0.962933 | 0.991001 | 0.998500 | 0.998500 | 0/1/184 |
| bag_03 | 132/15.015 Hz | 0.633821 | 0.648831 | 0.693378 | 0.793514 | 0.847762 | 0.851469 | 0.084242 | 0.915497 | 0.963382 | 0.992676 | 0.993551 | 0/0/132 |

## 8. Unresolved speed semantics

Speed unit and message meaning remain unresolved. Exact zero is reported only as a diagnostic stationary candidate; no simulator `1.0 m/s` assumption or permanent threshold is applied.

The same provenance check found no real-vehicle publisher implementation or channel metadata that establishes the speed unit or whether the value is a command, estimate, or feedback signal.

## 9. Chosen timestamp domain

MCAP `log_time` (bag-record time) is selected for camera, steering, and speed. Image headers exist, but Float64 has no header and camera header epochs differ substantially from record time. No mixed-clock synchronization is performed.

- bag_01 record-minus-header offset: mean 58635.025 s, median 58635.025 s, p95 58635.027 s, max 58635.043 s.
- bag_02 record-minus-header offset: mean 58978.089 s, median 58978.088 s, p95 58978.091 s, max 58978.096 s.
- bag_03 record-minus-header offset: mean 58642.673 s, median 58642.673 s, p95 58642.676 s, max 58642.679 s.

## 10-11. Causal steering/speed label ages and future-label violations

| Bag | Steering age | Speed age | Missing steer | Missing speed | Complete pairs | Future labels |
|---|---|---|---:|---:|---:|---:|
| bag_01 | mean 58.903 ms, median 59.820 ms, p95 63.471 ms, max 78.874 ms | mean 58.858 ms, median 59.746 ms, p95 63.416 ms, max 78.831 ms | 0 | 0 | 87 | **0** |
| bag_02 | mean 62.217 ms, median 62.586 ms, p95 67.491 ms, max 69.918 ms | mean 62.175 ms, median 62.529 ms, p95 67.443 ms, max 69.909 ms | 1 | 1 | 184 | **0** |
| bag_03 | mean 62.101 ms, median 61.473 ms, p95 68.132 ms, max 70.786 ms | mean 62.060 ms, median 61.413 ms, p95 68.124 ms, max 70.776 ms | 1 | 1 | 131 | **0** |

## 12. Three-frame temporal readiness

| Bag | Candidates | Strict causal | Label-ready | Order failures | Adjacent gaps | Oldest-current span | >0.120 s diagnostic | Assessment |
|---|---:|---:|---:|---:|---|---|---:|---|
| bag_01 | 85 | 85 | 85 | 0 | mean 0.067 s, median 0.067 s, p95 0.070 s, max 0.085 s | mean 0.133 s, median 0.133 s, p95 0.136 s, max 0.152 s | 0 | readable prefix is temporally compatible; full-bag compatibility is unresolved because the MCAP is incomplete |
| bag_02 | 183 | 183 | 183 | 0 | mean 0.067 s, median 0.067 s, p95 0.069 s, max 0.074 s | mean 0.133 s, median 0.133 s, p95 0.136 s, max 0.141 s | 0 | readable prefix is temporally compatible; full-bag compatibility is unresolved because the MCAP is incomplete |
| bag_03 | 130 | 130 | 130 | 0 | mean 0.067 s, median 0.067 s, p95 0.070 s, max 0.072 s | mean 0.133 s, median 0.133 s, p95 0.137 s, max 0.141 s | 0 | readable prefix is temporally compatible; full-bag compatibility is unresolved because the MCAP is incomplete |

Simulator train timing was approximately 0.066 s mean / 0.065 s median / 0.070 s p95 adjacent gap and 0.132 s mean oldest-to-current span. The existing 0.120 s adjacent gate was compared diagnostically, not enforced on real data.

## 13. Cross-bag consistency

readable prefixes share a camera contract; steering, speed, and timing consistency across complete bags remains unresolved. Full-bag steering, speed, and timing consistency cannot be concluded from differently sized readable prefixes.

- bag_01: 35 readable steering samples fall outside the confirmed approximate [-0.35,+0.35] rad range
- bag_02: 20 readable steering samples fall outside the confirmed approximate [-0.35,+0.35] rad range
- bag_03: 10 readable steering samples fall outside the confirmed approximate [-0.35,+0.35] rad range
- raw speed medians differ across readable prefixes: bag_01=0.721779, bag_02=0.944713, bag_03=0.851469
- readable steering medians differ: bag_01=+0.326348 rad, bag_02=+0.052025 rad, bag_03=+0.134024 rad
- exact-zero raw speed record counts in readable prefixes: bag_01=6, bag_02=1, bag_03=0; suffix stationarity is unobservable for every incomplete bag
- camera timing is similar; readable-prefix gap p95: bag_01=0.069727 s, bag_02=0.069463 s, bag_03=0.070125 s

## 14. Candidate active-driving windows

Stationary prefix/suffix diagnostics use exact zero only. Incomplete inputs make the true bag suffix unobservable; the windows below are nonzero-run candidates, not permanent filters.

- bag_01: +2.341 s to +5.868 s (3.527 s), raw speed min/median/max=0.524930/0.744394/0.867050. Exact-zero prefix/suffix counts=0/0; this window is prefix-censored and diagnostic only.
- bag_02: +5.897 s to +12.294 s (6.397 s), raw speed min/median/max=0.549662/0.958314/0.998500. Exact-zero prefix/suffix counts=0/0; this window is prefix-censored and diagnostic only.
- bag_03: +0.009 s to +8.733 s (8.725 s), raw speed min/median/max=0.633821/0.851469/0.993551. Exact-zero prefix/suffix counts=0/0; this window is prefix-censored and diagnostic only.

## 15. Simulator-vs-real camera differences

Resolution match: yes; aspect-ratio match: yes; encoding match: yes. The real preview uses the full frame and never applies simulator `y=160:360`.

The real camera is mounted very low and shows a wider, visibly barrel-distorted indoor track view with close barriers, spectators, cones, ceiling lights, motion blur, and curved lane geometry. The simulator reference is a clean rectilinear outdoor view with a distant horizon and substantial sky. Although dimensions and rgb8 encoding match, image content and projection do not.

ROI implication: Simulator y=160:360 would remove much of the real far-track and turn context visible above row 160. It is not approved for real data; retain uncropped previews until a complete-bag human ROI decision.

## 16. Unresolved items before dataset extraction

- complete, footer-valid copies of bag_01, bag_02, and bag_03
- reconcile readable steering values above +0.35 rad with the confirmed approximate [-0.35,+0.35] rad source contract
- steering left/right sign convention
- steering command-vs-actual-feedback meaning
- speed unit and message meaning
- human approval of the real-camera ROI after full-frame preview review

## 17. Files added/modified

- `configs/real_bag_audit_v1.json`
- `src/physicar_e2e/real_bag_audit.py`
- `scripts/run_real_bag_audit_v1.py`
- `tests/test_real_bag_audit.py`
- `results/real_bag_audit_v1/{summary.json,REPORT.md,bag_01.json,bag_02.json,bag_03.json,sync.json}`
- bounded previews under `/home/a/physicar-e2e-artifacts/real_bag_audit_v1/previews/` (outside Git)

## 18. Tests

Focused unit tests cover mappings, Float64 record timestamps, Image header/record comparison, causal ZOH, zero future labels, cross-bag consistency, and scope guards. Execution status is reported in the task handoff after the audit artifacts are generated.

## 19. Git status

No commit or push was performed. Final worktree status is reported in the task handoff.

## Scope attestation

No training, fine-tuning, simulator driving, Docker modification, bag modification, odometry requirement, dataset extraction, speed-unit assumption, steering-sign assumption, or real-camera ROI application occurred.
