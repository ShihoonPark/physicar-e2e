# Dataset Extractor V1

Dataset Extractor V1 is an offline, deterministic conversion of the three
Rosbag Collector V1 pilot episodes into camera-to-steering samples. It does not
run the simulator or vehicle and never modifies the canonical raw MCAP files.

## Reader and clock domains

The host reader uses the lightweight `mcap` and `mcap-ros2-support` Python
packages. ROS 2 CDR messages are decoded from the schemas embedded in each MCAP;
no host ROS installation is required. Pillow performs checked RGB decoding,
cropping, bilinear resizing, PNG writing, and preview generation.

MCAP record timestamps are the sole synchronization domain. `/steering` and
`/speed` are headerless `std_msgs/msg/Float64` messages. Camera header stamps
and `/clock` values are retained only as diagnostics. Collector UTC timestamps
are used only for a clock-domain cross-check, not for extraction boundaries.

## Drive window and synchronization

The active drive window is the longest contiguous segment of `/speed` records
for which `abs(speed) >= 0.10 m/s`. Its first and last active MCAP record times
are the inclusive extraction boundaries. This removes the stationary safety
prefix and suffix without comparing unrelated clocks.

For every camera record at `t_cam`, causal zero-order hold selects the latest
steering and speed records whose MCAP record timestamps are at or before
`t_cam`. The extractor asserts both causal inequalities. A frame is rejected
and counted if either state is absent, older than 0.15 seconds, or the selected
speed is below the drive threshold. No nearest-neighbor or future label is used.

## Image artifact

The input must be a checked `480×360` `rgb8` image. The decoder validates
dimensions, encoding, row stride, and payload length, and strips any row padding
without reinterpreting color. The fixed full-width ROI is `y=160:360`, yielding
`480×200`. Pillow bilinear resize produces exactly `200×66` RGB and stores it as
lossless PNG. Stored images are not YUV; future model preprocessing must make
RGB-to-YUV explicit.

## Running the three-pilot extraction

From the repository checkout:

```bash
python3 scripts/run_dataset_extractor_v1.py --sim-root /path/to/physicar-ai-sim-docker
```

Alternatively supply both `--input-root` and `--output-root`. Machine-specific
paths are CLI inputs and are not embedded in the canonical configuration. The
default derived output is under simulator userdata at
`physicar_e2e/dataset_extractor_v1_pilot/`. Existing output causes a failure;
`--force` explicitly replaces only the resolved output dataset directory.

Images, manifests, dataset metadata, and contact sheets remain outside Git.
Only compact episode and summary metrics are written under `results/`.

## Determinism and split policy

Episodes are processed in numeric order, camera records are ordered by MCAP
record time, and filenames are stable (`frame_000000.png`, etc.). V1 performs no
random sampling, downsampling, balancing, augmentation, or splitting.

Do not make random frame-level train/validation/test splits. Adjacent frames are
strongly correlated and would leak temporal information. Any future split must
assign complete episodes/runs as indivisible groups using `episode_id`.

Low record-time label age demonstrates the observed causal record relationship;
it does not prove perfect physical sensor/actuator synchronization or equality
among MCAP record time, camera header time, simulator `/clock`, and collector
host UTC.
