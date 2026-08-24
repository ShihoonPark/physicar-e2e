# Expert Driver 1.80 m/s Feasibility Gate V1

This one-run gate reuses the canonical privileged Pure Pursuit Expert. The sole runtime difference is fixed speed: 0.50 m/s to 1.80 m/s. Lookahead remains 0.45 m, wheelbase 0.18 m, control frequency 15 Hz, and steering authority ±0.349066 rad. Route, world, spawn, geometry, liveness, lap completion, and safe-stop behavior remain canonical.

A valid Expert failure permanently ends the gate. An infrastructure failure makes the result inconclusive; neither classification permits an automatic retry.

The actual steering target remains observable in radians on the existing `/steering` topic:

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
