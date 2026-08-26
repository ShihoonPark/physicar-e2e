# Real PhysiCar Runtime V1 — next-milestone physical test plan

Status: procedure only. Do not execute during Runtime V1. The exact real
traffic-light interface must be verified and adapted before motion.

1. If practical, secure the car with driven wheels off the ground. Keep an
   operator at the hardware emergency stop and begin with control publishing
   and physical-motion authorization disabled.
2. Verify `/camera/image_raw` is `sensor_msgs/msg/Image`, 480×360 `rgb8`, near
   15 Hz. Confirm the live core remains healthy through buffer warm-up and safe
   stops on camera interruption.
3. With speed forced to 0, verify steering sign at the vehicle interface:
   direct `/steering` `std_msgs/msg/Float64` positive normalized/model output
   turns LEFT; negative turns RIGHT. Confirm no `/teleop/steering`,
   `/teleop/speed`, or `/cmd_vel` routing exists.
4. Verify scaling at several points, including the limits: bounded +0.35 rad
   maps to +1.0 normalized and −0.35 rad maps to −1.0 exactly once.
5. Trigger camera timeout, bad ordering, inference failure, gate revocation, and
   node shutdown; each must command speed 0 and neutral steering.
6. Connect the verified GREEN-signal adapter and demonstrate only
   `WAITING_FOR_START → WARMING_TEMPORAL_BUFFER → RUNNING` on GREEN. Confirm no
   motion authorization on non-GREEN or missing signal.
7. On a controlled test surface with a spotter and emergency stop, authorize a
   direct `/speed` `std_msgs/msg/Float64` command materially below 1.0 m/s for
   separate straight and commanded-turn checks. Stop immediately on sign,
   scale, latency, or liveness mismatch.
8. Only after the bench and very-low-speed checks pass, prepare a separately
   reviewed closed-course E2E procedure. Do not jump directly to 1.0 m/s.
