# Automated Rosbag Collector V1

This milestone records the unchanged canonical geometric expert for one lap per
independent ROS 2 MCAP bag. It is simulator-only collection evidence and is not
evidence of real-robot performance.

## Recorded baseline

The exact topics are:

- `/camera/image_raw`
- `/steering`
- `/speed`
- `/cmd_vel`
- `/odom`
- `/clock`
- `/tf`
- `/tf_static`

The raw `sensor_msgs/msg/Image` stream is recorded at its published resolution
and encoding. The collector does not record the compressed camera stream and
does not crop, resize, convert, resample, synchronize, or trim data.

## Execution and storage

The host does not need ROS. All ROS commands run with the known Jazzy and
PhysiCar setup inside the configured, healthy Docker Compose service. The
collector inspects the live container mounts and requires
`/opt/physicar/userdata` to be a writable bind mount whose source is the
`userdata` directory beneath the supplied `--sim-root`.

Raw episodes use this external layout:

```text
<sim-root>/userdata/physicar_e2e/rosbag_collector_v1_pilot/
  episode_001/bag/
  episode_002/bag/
  episode_003/bag/
```

Each episode directory also contains a compact recorder log and a scoped PID
file. They remain external to this Git repository. Compact result JSON is
stored in `results/rosbag_collector_v1_pilot/` with one pilot summary in
`results/rosbag_collector_v1_pilot_summary.json`.

## Lifecycle and failure behavior

Before collection the command read-only verifies canonical and derived lane
assets, container identity and health, the installed bag commands, all required
topics, the runtime world, route, and absence of cones. Each episode then:

1. safe-stops, resets through the existing expert lifecycle, settles, and
   repeats preflight;
2. checks free space and starts one recorder;
3. confirms its exact process is alive and its bag directory exists;
4. runs the canonical expert directly, without a second reset;
5. safe-stops, sends SIGINT only to the episode PID, and waits for finalization;
6. runs installed `ros2 bag info`, checks every required topic/count and a
   meaningful raw-camera count, then measures host bytes; and
7. writes timing, expert, topic, integrity, cleanup, and storage metadata.

Any episode failure is preserved and stops the sequence. There is no automatic
retry. A graceful-shutdown timeout is reported as failure; cleanup escalates
only the recorder PID to SIGTERM and, if still necessary, SIGKILL. No broad
process matching is used.

The installed Jazzy `ros2 bag info` has no structured-output option, so V1
parses its topic/count lines and duration while treating any nonzero command
exit as an integrity failure.

## Commands

Read-only preflight:

```bash
python3 scripts/run_rosbag_collector_v1.py \
  --sim-root ~/physicar-ai-sim-docker --preflight-only
```

Authorized three-episode pilot:

```bash
python3 scripts/run_rosbag_collector_v1.py \
  --sim-root ~/physicar-ai-sim-docker --episodes 3
```

An existing external episode directory is never overwritten. Move it aside or
choose a deliberately different external data root before a separately
authorized future run. V1 does not implement dataset extraction, timestamp
synchronization assessment, PilotNet preprocessing, or a 50-episode run.
