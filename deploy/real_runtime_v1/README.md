# Real PhysiCar Temporal PilotNet Runtime V1 deploy bundle

This is a hardware-neutral CPU/ROS 2 deployment recipe for the selected
REAL-SCRATCH-V1 model. It is an offline integration candidate, not evidence of
real-vehicle safety or success. Runtime V1 has not physically driven the car.

Do not copy a simulator model. Copy these repository files while preserving the
`physicar_e2e` package layout:

- `src/physicar_e2e/pilotnet.py`
- `src/physicar_e2e/real_runtime.py`
- `src/physicar_e2e/real_runtime_ros2.py`
- `configs/real_runtime_v1.json`
- `scripts/run_real_runtime_ros2_v1.py`
- this deploy directory

Copy, outside Git, only the frozen selected runtime evidence from
`/home/a/physicar-e2e-artifacts/real_temporal_pilotnet_v1/selected/`:

- `real_temporal_pilotnet_v1_selected.onnx`
- `freeze.json`
- `freeze_seal.json`

Update only the three artifact paths in the copied config. Do not change their
hashes or any camera/preprocessing/model contract. The checkpoint is not needed
for inference; its frozen hash remains preserved in the manifest and freeze.

Install `requirements.txt` in a virtual environment appropriate to the actual
target CPU. Install ROS 2 and its Python message packages through the target's
supported ROS distribution rather than pip. No Raspberry Pi is assumed.

Before any ROS launch, run the offline startup audit and a stored-image/bag
check on the target. The canonical config keeps all motion keys safe:

```text
publish_control = false
physical_motion_authorized = false
start_gate.required = true
start_gate.adapter = null
```

The unverified real traffic-light API is an explicit blocker. The ROS node
therefore remains `WAITING_FOR_START`; it does not subscribe to a guessed topic.
A future verified adapter must call
`authorize_green_from_verified_adapter(green)`.

`--publish-control` is the explicit publisher opt-in. Never combine it with
`--development-start-bypass`; the adapter rejects that combination. Runtime V1
must stay in its default no-publish configuration. Physical-motion
authorization and the configured low speed belong to the next milestone only.

See `docs/real_runtime_v1_first_physical_test.md` before changing either safety
switch.
