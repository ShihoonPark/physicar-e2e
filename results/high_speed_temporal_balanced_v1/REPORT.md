# High-Speed Temporal PilotNet Late-Region Balance V1 — Final Report

Result: **HARD-GATE STOP — insufficient late-region diversity**.

The pre-registered balancing count was `K=6`, below the required `K>=20`. Per the experiment contract, no balanced manifest was created and Temporal PilotNet V10 was not trained, exported, preflighted, or driven. This is a negative data-sufficiency result, not a V10 policy failure. V9 remains canonical.

## 1. Preserved V9 evidence

Temporal PilotNet V9 remains unchanged:

- architecture: causal `[t-2,t-1,t]`, `9x66x200`, 255,819 parameters;
- training: from scratch on preserved High-Speed data;
- live result: 3/3 valid same-map, same-spawn 1.80 m/s `POLICY_PASS`;
- checkpoint: 1,032,805 bytes, SHA-256 `1cded5fcc7f3d13242de096c4868fc576d03fa6bc86df6e5c8c7c235d9faa6cc`;
- ONNX: 1,026,938 bytes, SHA-256 `7f6aa4c2d8c9b3615c580f065660c674efff94ff4cd0b9bdc9357df904000888`.

The audit rehashed both external artifacts successfully. Preserved V9 training, live, and temporal-dataset summary hashes are respectively `4c22c0b7f2d408b44b4698ff98d394ff3bde3f8d40c5dc34e6edd0a30d906f87`, `56829cfc312f5cfe353458c60afcd146a54f406374d2e02546e20406cccaa6d2`, and `a959e21140f639c1cc66afb5080da76ed7c71acf2a8c1fced6fe31590fa1bfbd`.

The user's separate visual comparison remains important context: the 1.80 m/s Expert passed at approximately 30.305 m with mean/max CTE approximately 0.0483/0.1780 m and appeared substantially farther inside on the final corner than V9. This experiment did not create another Expert run.

## 2. Route-metadata audit and imbalance

The audit inspected the exact preserved V9 temporal manifests. It joined every DAgger3 current frame to preserved numeric `completion_fraction` and `route_s_m` using the current simulator timestamp and absolute image path. It did not infer progress from filenames or pixels. The causal order, same-source identity, target identity, source MCAP hash, source-manifest hash, and adjacent gaps `<=0.120 s` were verified.

| DAgger3-A V9 temporal training bin | Sequence count |
|---|---:|
| 85% <= completion < 90% | 1,474 |
| 90% <= completion < 95% | 15 |
| 95% <= completion <= 100% | 6 |
| Total | 1,495 |

The maximum/minimum imbalance is `1474/6 = 245.67:1`. The 85–90% bin is 98.27 times the 90–95% bin and 245.67 times the 95–100% bin.

`K = min(1474,15,6) = 6`. Because `6 < 20`, deterministic selection was not performed. Selected counts are therefore **not applicable**, not zero-sample training bins. No counterfactual 6/6/6 manifest was written, and there is no V10 training-sequence count.

Manifest identities:

- V9 train temporal manifest SHA-256: `07d2f6cb6dd668352ae988dcfb771fa42739faed745681c0ed9b682b669834b4`;
- DAgger3-A source manifest SHA-256: `a39e32afc78da81783e7a91fd14ea0ad3a94e47f1e7ebe48abc7cf8ab9d66670`;
- V9 DAgger3-B temporal manifest SHA-256: `8872571653a5e3d340ad535c38ba9ca332d9c3efef5268c8544394ed1195d954`;
- DAgger3-B source manifest SHA-256: `5bf2401270ca7f2a76a4acaa3e8472b72d022fd2dcc804841e9675388367a107`.

## 3. Preserved V9 offline audit

All values are `MAE / RMSE / bias / max error / correlation / corrective magnitude ratio`, in radians where applicable. These are preserved matched V9 metrics; V10 columns do not exist because the diversity gate stopped before training.

| Major matched stratum (count) | V9 | V10 |
|---|---|---|
| Nominal validation (482) | .01063 / .01784 / .00108 / .09578 / .99329 / .9898 | Not trained |
| Nominal holdout (481) | .01073 / .01747 / .00140 / .07622 / .99349 / .9990 | Not trained |
| DAgger1-B (41) | .02130 / .03271 / .00127 / .11467 / .98865 / .9955 | Not trained |
| DAgger2-B (44) | .03533 / .05200 / -.01165 / .16576 / .98021 / .9573 | Not trained |
| DAgger3-B overall (1,011) | .01817 / .04060 / .00225 / .48289 / .97680 / .9799 | Not trained |

The exact DAgger3-B route-bin audit is:

| Bin (count) | MAE | RMSE | Bias | Max error | Correlation | Corrective ratio |
|---|---:|---:|---:|---:|---:|---:|
| 85–90% (990) | .01801 | .04072 | .00199 | .48289 | .97640 | .9803 |
| 90–95% (7) | .02067 | .02886 | -.00646 | .06754 | .38538 | .9813 |
| 95–100% (14) | .02837 | .03721 | .02529 | .07448 | .98926 | .9544 |

The combined preserved V9 90–100% subset has 21 samples and MAE `0.02580 rad`. The 95–100% MAE is 1.575 times the 85–90% MAE, consistent with the final region remaining harder, although the two upper-bin sample counts are very small.

## 4. V10 and downstream hard gates

The planned V10 contract was verified before the stop: the unchanged Temporal PilotNet builder has exactly 255,819 parameters, `9x66x200` input, and the same V9 training configuration (MSE, Adam, `1e-3`, batch 64, maximum 35 epochs, seed 20260824, early stopping, from-scratch initialization). No V10 model instance was optimized.

| Stage | Result |
|---|---|
| Balanced DAgger3-A manifest | Not created — K gate failed |
| Final V10 train sequence count | Not applicable |
| V10 training | Not executed |
| Matched V9/V10 evaluation | Not executed |
| Offline live gate | Not reached |
| V10 checkpoint / hash | Not created |
| V10 ONNX / hash / equivalence | Not created / not run |
| Simulator preflight | Not executed |
| V10 live run #1 | Not executed |
| Conditional runs #2/#3 | Not executed |
| V10 90–95% / 95–100% live CTE diagnostics | Not available; no live run |
| V10 repeatability | No result |

No new training data, raw bag, Expert lap, neural rollout, DAgger iteration, synthetic sample, label, or external V10 artifact directory was created. No automatic longer training, alternate bins/K, oversampling, weighting, DAgger, or later experiment was attempted.

## 5. Decision and practical interpretation

V10 does not beat V9 under measured evidence because V10 was correctly prevented from existing. V9 remains the canonical 1.80 m/s policy and all of its history is preserved. There is no V10 demo for the user to compare visually; the user's concern about V9's final-corner road margin remains unresolved by this experiment.

This balancing hypothesis cannot be tested cleanly with the preserved DAgger3-A source because the 95–100% bin contains only six temporal sequences. The correct next action is not another automatic optimization. Any future work would require a separately authorized decision about obtaining sufficient final-region evidence or accepting V9 against the actual product/competition rule.

Cone Avoidance V1 remains technically permitted by preserved V9's existing 3/3 simulator-repeatability gate if the user accepts V9 under the practical requirement that at least one wheel may remain on the road. This stopped experiment neither strengthens nor invalidates that product-level decision. It does not justify claiming that V9 follows the Expert's final-corner line closely.

## 6. Verification, files, and limitations

Focused tests: 14/14 PASS. Full regression: 290/290 PASS. Python compilation passed. `git diff --check` and the tracked-diff check passed.

Added path groups:

- `configs/high_speed_temporal_balanced_v1.json`;
- `docs/high_speed_temporal_balanced_v1.md`;
- `results/high_speed_temporal_balanced_v1/`;
- `scripts/run_high_speed_temporal_balanced_v1.py`;
- `src/physicar_e2e/high_speed_temporal_balanced.py`;
- `tests/test_high_speed_temporal_balanced.py`.

Final Git state: branch `experiment/pilotnet-high-speed-temporal-balanced-v1`, HEAD `d57effe7e91a9d20ffe3fd7ed71f919a1ba28bff`. The tracked tree is unchanged and the six intended experiment path groups are untracked. Nothing is staged, committed, or pushed.

Limitations: the upper DAgger3-A and DAgger3-B bins are extremely small; no V10 causal claim is possible. CTE is a centerline metric, not a formal wheel/road-footprint or visible-road-margin metric. The Expert visual comparison metrics were supplied as context, not generated by this audit. All preserved success evidence is same-map, same-spawn simulation evidence only, not real-robot performance.
