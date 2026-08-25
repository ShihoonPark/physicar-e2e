# Real Dataset Extraction V1

Extraction/QC result: **PASS**. No training was performed.

## 1. Exact source bag hashes

| Bag | Bytes | SHA-256 | Verified |
|---|---:|---|---|
| bag_01 | 678405210 | `c98fd035af40fb5e698e94f0a87fe3f8c979f4b473c16807621c70aa0b038f04` | yes |
| bag_02 | 1112101420 | `856bcdebb65db968a549625abb528a80ee863e03540fa458b43c711333826636` | yes |
| bag_03 | 472764359 | `f7b3622c0d35e04fcc31e01aa6041bc5ae46c1ed6fdcb700c30900aa7e34438f` | yes |

## 2. Extraction configuration

MCAP `log_time`; causal scalar ZOH; steering `recorded * 0.35` rad without clipping; steering age and each adjacent camera gap must be <=0.120 s. Speed is metadata-only with semantics `UNKNOWN_COMMAND_OR_FEEDBACK`. Real ROI is `x=0:480, y=80:360` -> RGB 200x66 bilinear.

## 3. Accepted temporal sequence count

**2163** of 2167 candidates (0.998154).

## 4. Rejected count and reasons

Rejected unique candidates: **4**.

Reason occurrences (overlap allowed): `{'non_increasing_camera_timestamps': 0, 'adjacent_camera_gap_gt_0p120_s': 4, 'no_causal_steering': 0, 'future_steering_label': 0, 'steering_age_gt_0p120_s': 2}`.

Mutually exclusive reason combinations: `{'adjacent_camera_gap_gt_0p120_s': 2, 'adjacent_camera_gap_gt_0p120_s+steering_age_gt_0p120_s': 2}`.

## 5. Per-bag sequence counts

| Bag | Camera frames | Candidates | Accepted | Rejected |
|---|---:|---:|---:|---:|
| bag_01 | 651 | 649 | 649 | 0 |
| bag_02 | 1068 | 1066 | 1064 | 2 |
| bag_03 | 454 | 452 | 450 | 2 |

## 6. Steering raw/radian distributions

Recorded normalized command: min=-0.934881, mean=0.089761, median=0.069822, p95=0.726640, max=0.910556.

Physical radians: min=-0.327208, mean=0.031416, median=0.024438, p95=0.254324, max=0.318695; signs={'negative_RIGHT': 667, 'zero': 19, 'positive_LEFT': 1477}; outside +/-0.35=0. Conversion-once QC=True.

## 7. Steering label-age distribution

min=0.040550, mean=0.061627, median=0.060945, p95=0.067565, max=0.078874 s.

## 8. Camera gap distribution

Accepted sequence adjacent gaps: min=0.039547, mean=0.066622, median=0.066654, p95=0.069694, max=0.085031 s. Oldest-to-current spans: min=0.099006, mean=0.133245, median=0.133284, p95=0.136735, max=0.151985 s.

## 9. Exact >120 ms rejects

- bag_02_t000900: gaps=0.064795920/0.357921377 s, steering_age=0.356701130 s, reasons=['adjacent_camera_gap_gt_0p120_s', 'steering_age_gt_0p120_s']
- bag_02_t000901: gaps=0.357921377/0.039547205 s, steering_age=0.031776791 s, reasons=['adjacent_camera_gap_gt_0p120_s']
- bag_03_t000447: gaps=0.067391582/0.252107132 s, steering_age=0.250871544 s, reasons=['adjacent_camera_gap_gt_0p120_s', 'steering_age_gt_0p120_s']
- bag_03_t000448: gaps=0.252107132/0.056709768 s, steering_age=0.050113621 s, reasons=['adjacent_camera_gap_gt_0p120_s']

## 10. Speed metadata completeness/staleness

Accepted: available=2163, missing=0, valid=2163, stale=0. All candidates: available=2167, missing=0, valid=2165, stale=2. Semantics remain `UNKNOWN_COMMAND_OR_FEEDBACK`; speed is not an input, target, or filter.

## 11. ROI and image QC

2173 images verified as RGB 200x66 PNG; missing=0, orphaned=0, corrupt=0. Simulator ROI, horizontal crop, and undistortion were not used.

## 12. Future-label violations

**0**.

## 13. Dataset manifest hash

`ba82ae5f1f7c606f5f516ea006148f033ab95ec9097d2f6aaa300c2ab91f5597` (2163 rows, `manifests/real_dataset_v1.csv`).

## 14. REAL_DATASET_V1 QC decision

**PASS**: all QC gates={'all_three_source_hashes_match': True, 'candidate_accounting_exact': True, 'nonzero_accepted_sequences_each_bag': True, 'whole_stream_steering_scale_exactly_once': True, 'physical_steering_range_qc': True, 'future_steering_label_violations_zero': True, 'accepted_steering_ages_within_0p120_s': True, 'accepted_adjacent_camera_gaps_within_0p120_s': True, 'no_episode_crossing': True, 'no_duplicate_frame_padding': True, 'every_stored_image_passes_qc': True, 'speed_semantics_preserved_unknown': True, 'speed_not_used_as_filter_input_or_target': True, 'no_raw_mcap_copies': True, 'training_not_invoked': True}.

## 15. Whether training can be considered next

**CAN BE CONSIDERED AFTER HUMAN REVIEW AND FINAL SUBSET SELECTION**. Training remains unauthorized and was not invoked.

## 16. Tests

The extractor itself invoked no tests or training. Post-extraction verification:

- `pytest -q tests/test_real_dataset.py`: **20 passed in 0.04 s**.
- `pytest -q`: **489 passed, 39 subtests passed in 82.49 s**; 2 pre-existing ONNX-export deprecation warnings.
- Independent validation re-read all 2,163 manifest rows and passed causality, timestamp arithmetic, gap/age, steering scaling, speed-state, and image-reference checks.

## 17. External artifacts

Dataset root: `/home/a/physicar-e2e-artifacts/real_dataset_v1`. Manifest, image inventory, rejection records, derived images, metadata, and bounded contact sheets are stored there; no raw MCAP was copied.

## 18. Git status

Branch: `feature/real-bag-intake-prep`. Audit and dataset source/config/test/result paths are untracked. No commit or push was performed; raw bags and the 51 MiB derived image dataset remain outside Git.
