# High-Speed Temporal PilotNet Late-Region Balance V1

This bounded experiment asks whether deterministic undersampling of the overrepresented DAgger3-A 85–90% route region can improve final-corner imitation while preserving Temporal PilotNet V9's 1.80 m/s repeatability.

The intervention is data distribution only. It preserves V9's causal three-frame representation, `9x66x200` architecture, 255,819 parameters, preprocessing, labels, optimizer, loss, learning rate, batch size, seed, early stopping, and 35-epoch maximum. It permits no new driving data, oversampling, weighting, sweep, DAgger, or automatic follow-up optimization.

The three frozen route bins are `[0.85,0.90)`, `[0.90,0.95)`, and `[0.95,1.00]`. Route completion comes from the preserved DAgger3 source metadata joined to each V9 temporal sequence by the current frame's simulator timestamp and image path. The intended equal count is `K = min(n1,n2,n3)`, with a hard minimum of 20. Had that gate passed, each bin would have been deterministically undersampled at evenly spaced ordered indices without replacement.

The audit found 1,474, 15, and 6 DAgger3-A V9 temporal sequences in the three bins. Therefore `K=6`, which fails the pre-registered `K>=20` gate. The experiment stopped before creating a balanced training manifest. V10 was not trained, exported, preflighted, or driven. V9 remains preserved and canonical; no later balancing or tuning experiment is automatically authorized.
