# Real PhysiCar Camera–Steering Phase Audit V1

## Result

**Semantic decision: `INCONCLUSIVE`.**

MCAP timing strongly resembles a camera-triggered control cycle, but the exact deployed `/steering` publisher could not be identified from the requested source location, recovered archive, platform source, or bag metadata. Code candidates support both post-camera and independent-control semantics. Therefore timing alone is not used to relabel the dataset.

`REAL_DATASET_V1` remains preserved and no `REAL_DATASET_V2` extraction is recommended until the deployed publisher source/provenance is recovered.

## Bag timing (MCAP log_time only)

All offsets below are milliseconds. PREV is `camera - previous`; NEXT is `next - camera`; NEAREST is signed `command - camera`.

| Bag | Frames | PREV mean / median / p05 / p95 / min / max | NEXT mean / median / p05 / p95 / min / max | NEAREST signed mean / median / p05 / p95 / min / max |
|---|---:|---|---|---|
| bag_01 | 651 | 60.615 / 60.401 / 55.058 / 66.726 / 40.550 / 78.874 | 6.017 / 6.370 / 1.143 / 8.487 / 0.836 / 26.062 | 6.017 / 6.370 / 1.143 / 8.487 / 0.836 / 26.062 |
| bag_02 | 1068 | 62.237 / 61.283 / 56.813 / 67.745 / 31.777 / 356.701 | 4.639 / 5.933 / 1.048 / 7.846 / 0.728 / 21.798 | 4.639 / 5.933 / 1.048 / 7.846 / 0.728 / 21.798 |
| bag_03 | 454 | 62.622 / 61.518 / 56.768 / 68.188 / 49.378 / 250.872 | 4.393 / 5.669 / 1.060 / 7.618 / 0.959 / 9.336 | 4.393 / 5.669 / 1.060 / 7.618 / 0.959 / 9.336 |

For every frame with a NEXT candidate, NEAREST is NEXT. Contiguous-decile medians satisfy the recorded stability rule in all three bags. Rare large PREV/period maxima are stream dropouts; the p05–p95 phase remains narrow.

## Control-cycle structure

- `bag_01`: PREV→camera median 60.401 ms; camera→NEXT median 6.370 ms; complete ordered triples 651/651.
- `bag_02`: PREV→camera median 61.283 ms; camera→NEXT median 5.933 ms; complete ordered triples 1067/1068.
- `bag_03`: PREV→camera median 61.518 ms; camera→NEXT median 5.669 ms; complete ordered triples 453/454.

This is strong phase evidence, not publisher provenance.

## Code-semantics audit

Requested source `/home/a/real_physicar_handoff/track_drive`: **not present**.

| Audit question | Evidence-led answer |
|---|---|
| Who publishes `/camera/image_raw`? | Platform real.launch.py starts physicar_camera/camera_node and remaps its output to /camera/image_raw. It is launched separately from the webserver. |
| Who publishes `/steering`? | The only exact /steering publisher found is the platform webserver RosBridge publisher, callable from independent HTTP/WebSocket endpoints. Available evidence does not establish that it produced the recorded bag stream. |
| Who publishes `/speed`? | The same RosBridge creates /speed and publish_speed. Available evidence does not establish that it produced the recorded bag stream. |
| Where is steering computed? | Unknown for the recorded bags. RosBridge.publish_steering forwards an externally supplied scalar and does no camera computation. Recovered e2e_sched.cb computes from a current camera frame but publishes the different /xycar_motor topic. |
| Image→compute→publish ordering? | Unknown for the recorded exact topics. The recovered e2e_sched candidate executes image conversion/preprocessing/inference and then publishes /xycar_motor inside cb. The exact-topic web control path is independent of camera callbacks. |
| Does `data_logger` publish? | Recovered data_logger only subscribes, caches the latest /xycar_motor command, and logs it during its image callback; it does not republish commands. |
| Is teleoperation independent? | Recovered teleop_key independently publishes /xycar_motor at 50 Hz, not /steering. The exact /steering web endpoints are also independent external-control inputs. |
| Are commands held? | The platform driver applies each /steering command to the servo. Its watchdog comment explicitly stops stale speed while holding steering/camera, so steering remains held between updates on this driver path. |

Additional provenance evidence:

- Recovered `TalkFile_track_drive.zip.zip`: candidate only. `data_logger.py` subscribes to camera and `/xycar_motor` and logs the cached command; it does not publish. `teleop_key.py` publishes independently of camera. `e2e_sched.py` processes the current camera callback and then publishes, but all use different bag topics.
- Read-only platform source shows the web bridge publishing exact `/steering` and `/speed` from independent HTTP/WebSocket requests, while the camera is a separate node. The motor driver subscribes and holds steering between updates; only stale speed is stopped by its watchdog.
- No `/lane/debug/path` publisher source was found. This prevents attribution of the recorded 15 Hz lane/control stream to either source architecture.
- The exact-topic web bridge documents its scalar as radians, whereas the frozen bag contract identifies recorded steering as normalized command multiplied by 0.35. This is additional evidence that the discovered bridge cannot be assumed to be the recording producer without launch/runtime provenance.

All file/function/line excerpts and source hashes are embedded in `summary.json`.

## Steering-change analysis

PREV/NEXT differences use `recorded × 0.35` exactly once, with no clipping. The fixed audit groups are low `<0.01 rad`, medium `0.01–<0.05 rad`, and high `≥0.05 rad`.

| Group | Count | absolute difference mean / median / p95 / max (rad) |
|---|---:|---|
| low abs delta lt 0p01 rad | 1239 | 0.002577 / 0.001556 / 0.008707 / 0.009859 |
| medium abs delta 0p01 to lt 0p05 rad | 837 | 0.017431 / 0.012146 / 0.038890 / 0.049808 |
| high abs delta ge 0p05 rad | 95 | 0.097714 / 0.094747 / 0.131488 / 0.281778 |

Magnitude-threshold and strict sign-change subsets are reported in `timing.json`.

## Six high-steering bag_03 cases

| Sequence | Camera log_time ns | PREV rad / age ms | NEXT rad / delay ms | NEAREST rad | Current label rad | Scratch V1 prediction rad |
|---|---:|---|---|---:|---:|---:|
| bag_03_t000154 | 1787685117777616348 | +0.250368 / 58.471 | +0.194535 / 7.908 | +0.194535 | +0.250368 | 0.204194 |
| bag_03_t000265 | 1787685125174812584 | -0.268414 / 57.107 | -0.268414 / 1.260 | -0.268414 | -0.268414 | 0.125076 |
| bag_03_t000266 | 1787685125241916253 | -0.268414 / 65.844 | -0.240376 / 3.707 | -0.240376 | -0.268414 | 0.108930 |
| bag_03_t000271 | 1787685125575362776 | -0.278143 / 56.265 | -0.266484 / 4.127 | -0.266484 | -0.278143 | 0.115289 |
| bag_03_t000272 | 1787685125644001933 | -0.266484 / 64.512 | -0.251570 / 6.527 | -0.251570 | -0.266484 | 0.124332 |
| bag_03_t000273 | 1787685125708887763 | -0.251570 / 58.359 | -0.234453 / 1.052 | -0.234453 | -0.251570 | 0.115786 |

See `high_steering_contact_sheet.png` and `high_steering_cases.json` for image paths.

Historical Scratch V1 evidence remains: bag_03 MAE 0.047842 rad, RMSE 0.081893 rad, sign agreement 0.8089; six-sample `|steering| ≥ 0.25` MAE 0.328104 rad. No model was trained or modified by this audit.

## Causality and dataset consequence

A command timestamp after image capture is not future sensor information when the command is computed from that image. A model using `[t-2,t-1,t]` remains camera-causal. That distinction does not resolve this audit because the deployed computation path is missing.

Current consequence: preserve V1 unchanged; recover the exact deployed lane/control publisher or recording launch provenance before selecting PREV, NEXT, or an independent-stream alignment policy. Do not create V2 solely because NEXT is closer in time.

## Scope and preservation

- Manifest SHA-256: `ba82ae5f1f7c606f5f516ea006148f033ab95ec9097d2f6aaa300c2ab91f5597`
- Scratch ONNX SHA-256: `b860afe396c8e48001339b4f99c8b3daa272500725d48d79b9c22b859c6fd339`
- Artifact and raw-bag before/after snapshots matched exactly.
- No training, driving, simulation, collection, raw-bag changes, dataset changes, checkpoint changes, commit, or push occurred.
