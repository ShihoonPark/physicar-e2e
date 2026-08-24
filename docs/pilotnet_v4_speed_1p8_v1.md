# PilotNet V4 1.80 m/s Validation V1

This bounded same-map experiment reuses the canonical camera-only PilotNet V4 inference and safety loop. Its sole control variable is fixed speed: 0.50 m/s to 1.80 m/s. The model, HTTP JPEG camera contract, preprocessing, ROI, steering clamp, 15 Hz loop, route, spawn, and safety semantics remain canonical.

At 15 Hz, nominal travel per neural update is 0.0333 m at 0.50 m/s and 0.1200 m at 1.80 m/s, a 3.6× increase. This is context, not a tuning input.

The first policy failure stops the experiment. Infrastructure failures are classified separately: the initial evaluation allows one infrastructure retry, repeatability replacements remain bounded, and at most five total attempts can occur. Three valid policy passes are required for `PASS`.

The actual steering target remains available as radians on `/steering`:

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
