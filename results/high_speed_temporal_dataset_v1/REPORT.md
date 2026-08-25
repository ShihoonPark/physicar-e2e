# High-Speed Temporal Dataset V1

PASS. No rosbag, Expert lap, neural rollout, DAgger iteration, synthetic image, or duplicated PNG was created. Temporal manifests reference only the preserved V8 source images and hashes under external userdata.

The sample definition is strictly causal `[t-2, t-1, t] → Expert steering[t]`, within one episode/rollout and source trajectory. Adjacent timestamps must be increasing and no more than 0.120 s apart. The first two frames of each source are boundary rejects; no history is padded.

| Stratum | Candidates | Accepted | Gap rejects | Boundary rejects | Gap mean/median/p95/max (s) | Span mean/median/p95/max (s) |
|---|---:|---:|---:|---:|---|---|
| Train | 3,512 | 3,510 | 2 | 22 | .066/.065/.070/.070 | .132/.130/.135/.135 |
| Nominal validation | 482 | 482 | 0 | 4 | .066/.065/.070/.070 | .132/.130/.135/.135 |
| Nominal holdout | 481 | 481 | 0 | 4 | .066/.065/.070/.070 | .132/.130/.135/.135 |
| DAgger1-B | 41 | 41 | 0 | 2 | .066/.065/.070/.070 | .132/.130/.135/.135 |
| DAgger2-B | 44 | 44 | 0 | 2 | .066/.065/.070/.070 | .132/.130/.135/.135 |
| DAgger3-B | 1,013 | 1,011 | 2 | 2 | .066/.065/.070/.070 | .132/.130/.135/.135 |

Training sources are nominal episodes 001–008 and DAgger1/2/3 rollout A. Evaluation sources retain nominal episodes 009–012 and each historical rollout B. Source-ID and source-hash overlap are zero; low-speed data and all B holdouts are excluded from training.
