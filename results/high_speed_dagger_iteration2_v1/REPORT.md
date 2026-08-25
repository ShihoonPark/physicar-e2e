# High-Speed PilotNet DAgger Iteration 2 V1

Result: **PARTIAL SUPPORT**. Preserved V6 failed at 25.439 m (83.40%). V6 shadow rollout A failed at 25.364 m (83.15%) and passed the >=60% reproduction gate; B failed at 25.268 m (82.83%) after traversing the frozen A region. A window was route-s 21.028–25.364 m, 45 samples; B holdout contains 46 samples. Raw MCAPs are external: A 116,354,356 bytes (`4a3a1174...9be14c`), B 117,393,092 bytes (`e04a7c97...5ce4c`).

Both collections were V6-controlled, Expert-shadow-only, same world/spawn, 1.80 m/s, 0.90 m lookahead, 15 Hz. Safe stop and infrastructure health passed. Causal ZOH alignment had zero future labels, zero stale rejects, zero decode failures; A label age mean/median/p95/max 50.33/50/99/100 ms, B 51.63/50/95/105 ms.

V7 was trained from scratch: 1,940 nominal train + 50 DAgger1-A + 45 DAgger2-A = 2,035 samples; DAgger1-B (43), DAgger2-B (46), nominal validation/holdout excluded. Architecture remained 252,219 parameters. Checkpoint SHA256 `99a4671c...b4ea93`; ONNX SHA256 `d6925fa4...4fe6bad`; checker and PyTorch/ONNX equivalence passed (max 8.32e-8 rad).

Offline DAgger2-B: V6 MAE 0.13740 rad, corrective ratio 0.8074; V7 MAE 0.05018 rad, ratio 0.9393. DAgger1-B retention: V6 0.03526 rad, V7 0.03608 rad. Nominal validation/holdout V7 MAE 0.00976/0.00982 rad.

Live V7: run #1 POLICY_PASS, 30.24 m / 99.13%; run #2 POLICY_FAIL, 29.30 m / 96.05%, safe stop PASS, no infrastructure failures. This exceeds V6 by 3.86 m (+12.66%), but is not 3/3; V7 is not frozen and Cone Avoidance V1 is not justified. Further automatic DAgger is not authorized; no Iteration 3 was run.

Progression: Expert 1.80 m/s 3/3 PASS; V4 0.50 m/s 3/3 PASS; V4 1.80 FAIL 10.48%; V5 FAIL 67.82%; V6 FAIL 83.40%; V7 run1 PASS/run2 FAIL.

Tests: focused Iteration2 + DAgger1 contracts 20/20 PASS; final full regression follows. No V4/V5/Expert artifacts were overwritten; raw data/models remain external. No commit or push.
