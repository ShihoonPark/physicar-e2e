# Expert 1.80 m/s Lookahead Characterization V1

This bounded characterization reuses the canonical privileged Pure Pursuit implementation at 1.80 m/s, 15 Hz, ±0.349066 rad steering, and 0.18 m wheelbase. Only `lookahead_m` changes through the pre-registered ascending candidates 0.60, 0.75, and 0.90 m. The canonical 0.45 m failure is preserved and not rerun.

Each candidate receives one safe-stop/reset/full-preflight/live-run/safe-stop lifecycle. The sweep stops at the first `EXPERT_PASS` or any `INFRA_FAIL`; otherwise all three failures are retained. No repeatability is attempted here.

The actual steering target remains visible in radians through the unchanged `/steering` topic:

```bash
docker exec -it physicar-sim bash -lc '
source /opt/ros/jazzy/setup.bash
source /opt/physicar/install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///opt/physicar/src/physicar-ros/deploy/cyclonedds.xml
ros2 topic echo /steering
'
```
