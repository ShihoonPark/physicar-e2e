# PhysiCar PilotNet E2E — 빠른 확인 방법

이 저장소는 NVIDIA PilotNet/DAVE-2 스타일의 end-to-end 조향 파이프라인을 구현합니다.

```text
camera image → PilotNet → steering angle [rad]
```

원래 데이터 흐름은 다음과 같습니다.

```text
Expert driving
→ camera와 steering을 별도 timestamp topic으로 담은 ROS bag
→ causal camera/steering pairing
→ PilotNet training
→ camera-only closed-loop steering
```

조향값 단위는 radian이며 simulator의 물리 범위는 약 `-0.349066–+0.349066 rad`
(`-20–+20°`)입니다. PilotNet에는 camera만 입력됩니다. Route, GT pose, CTE 등은
expert 제어 또는 안전/평가에만 사용되며 neural observation에 포함되지 않습니다.

## 친구가 제일 먼저 하면 되는 것

1. Simulator를 실행하고 아래 cone-free world가 선택됐는지 확인합니다.
2. Expert 또는 PilotNet으로 차량을 주행합니다.
3. 별도 terminal에서 `/steering`을 실시간 확인합니다.
4. 필요하면 기존 bag의 `/camera/image_raw`와 `/steering` 존재 여부를 확인합니다.
5. 한 장의 480×360 camera image로 PilotNet V4의 `steering_rad`를 확인합니다.
6. Raspberry Pi 담당자는 [`deploy/pi5/README.md`](deploy/pi5/README.md)를 봅니다.

아래 명령은 저장소 root에서 실행합니다. Python 도구 전체를 확인하려면:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[pilotnet]'
python -m unittest discover -s tests
```

## 1. Simulator 실행 및 화면 보기

Simulator는 별도 checkout인 `~/physicar-ai-sim-docker`에서 실행됩니다.

```bash
cd ~/physicar-ai-sim-docker
./physicar
```

준비된 뒤 다음 화면을 사용할 수 있습니다.

- PhysiCar 앱: <http://localhost:8080/app>
- Gazebo 3D 화면: <http://localhost:8080/sim/>
- VNC 화면: <http://localhost:8080/vnc/>
- API 문서: <http://localhost:8080/docs>

이 프로젝트의 검증 world는 다음 cone-free derived world입니다.

```text
custom_71e69ee938032295503bfed557fde18c_e2e_lane_follow_v1
```

PhysiCar 앱에서 이 world를 선택합니다. Asset을 생성하거나 읽기 전용으로 검증하는
실제 명령은 다음과 같습니다. 이 스크립트는 tracked simulator source를 수정하지
않고 simulator가 사용하는 generated asset만 다룹니다.

```bash
python3 scripts/setup_lane_follow_environment_v1.py \
  --sim-root ~/physicar-ai-sim-docker

python3 scripts/setup_lane_follow_environment_v1.py \
  --sim-root ~/physicar-ai-sim-docker \
  --verify-only
```

## 2. 주행 중 steering 값을 실시간으로 보기

주행/simulator terminal은 그대로 두고 새 terminal을 엽니다. ROS 2 Jazzy와
PhysiCar overlay, CycloneDDS 설정은 `physicar-sim` container 안에 있습니다.

```bash
docker exec -it physicar-sim bash -lc 'source /opt/ros/jazzy/setup.bash && source /opt/physicar/install/setup.bash && export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file:///opt/physicar/src/physicar-ros/deploy/cyclonedds.xml && ros2 topic echo /steering'
```

`/steering`의 확인된 type은 `std_msgs/msg/Float64`입니다. 출력의 `data:`가 simulator
steering interface로 보내는 조향 target radian 값입니다. PilotNet 주행 중에는
neural model이 계산해 보낸 실제 command를 여기서 볼 수 있습니다.

한 번만 출력하고 종료하려면 주행 중 다음 명령을 사용합니다.

```bash
docker exec -it physicar-sim bash -lc 'source /opt/ros/jazzy/setup.bash && source /opt/physicar/install/setup.bash && export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file:///opt/physicar/src/physicar-ros/deploy/cyclonedds.xml && ros2 topic echo /steering --once'
```

예를 들어 `data: 0.12`는 약 `0.12 rad ≈ 6.9°`입니다.

고정 속도도 같은 방법으로 확인할 수 있습니다. 검증된 V4 조건에서는 주행 중 약
`data: 0.5`, 즉 `0.50 m/s`가 표시됩니다.

```bash
docker exec -it physicar-sim bash -lc 'source /opt/ros/jazzy/setup.bash && source /opt/physicar/install/setup.bash && export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST RMW_IMPLEMENTATION=rmw_cyclonedds_cpp CYCLONEDDS_URI=file:///opt/physicar/src/physicar-ros/deploy/cyclonedds.xml && ros2 topic echo /speed'
```

## 3. ROS bag에서 camera + steering 확인

Canonical collector가 기록하는 topic은 실제
[`configs/rosbag_collector_v1.json`](configs/rosbag_collector_v1.json)에 고정돼 있습니다.

- `/camera/image_raw` — `sensor_msgs/msg/Image`
- `/steering` — `std_msgs/msg/Float64`
- `/speed` — `std_msgs/msg/Float64`
- `/cmd_vel`
- `/odom`
- `/clock`
- `/tf`
- `/tf_static`

Steering 값은 camera message 안에 들어 있지 않습니다. Bag에는 camera와 steering이
서로 다른 timestamp topic으로 저장됩니다.

```text
camera(t) → latest steering where steering_time <= camera_time
```

Dataset Extractor는 MCAP record time을 기준으로 최신 과거 steering을 선택하는
`causal ZOH`/causal synchronization을 수행합니다. 미래 steering label은 사용하지
않습니다.

### 기존 pilot bag 확인

Canonical host root는 `--sim-root`를 기준으로 다음과 같이 해석됩니다.

```text
~/physicar-ai-sim-docker/userdata/physicar_e2e/rosbag_collector_v1_pilot/
  episode_001/bag/
  episode_002/bag/
  episode_003/bag/
```

`userdata`는 container의 `/opt/physicar/userdata`로 bind mount됩니다. 따라서 실제
episode 001 bag 정보는 다음 명령으로 확인합니다.

```bash
docker exec -it physicar-sim bash -lc 'source /opt/ros/jazzy/setup.bash && source /opt/physicar/install/setup.bash && ros2 bag info /opt/physicar/userdata/physicar_e2e/rosbag_collector_v1_pilot/episode_001/bag'
```

출력의 `Topic information`에 `/camera/image_raw`와 `/steering`, 각 type과 message
count가 함께 있어야 합니다. 현재 보존된 episode 001에서 실제로 확인된 count는
camera 881, steering 873입니다.

### 새 expert bag 수집 방법

먼저 simulator, 올바른 world, topic을 읽기 전용으로 검사합니다.

```bash
python3 scripts/run_rosbag_collector_v1.py \
  --sim-root ~/physicar-ai-sim-docker \
  --preflight-only
```

별도로 승인된 수집에서만 다음 명령을 실행합니다.

```bash
python3 scripts/run_rosbag_collector_v1.py \
  --sim-root ~/physicar-ai-sim-docker \
  --episodes 3
```

Canonical Pure Pursuit expert가 주행하고 camera와 steering을 별도 ROS topic으로 한
bag에 기록합니다. Raw bag은 Git이 아니라 simulator `userdata`에 남습니다. 기존
episode directory는 덮어쓰지 않으며, 이 명령은 50개 lap을 수집하지 않습니다.
자세한 lifecycle은 [`docs/rosbag_collector_v1.md`](docs/rosbag_collector_v1.md)에 있습니다.

## 4. Camera와 steering을 PilotNet sample로 만들기

Dataset Extractor V1의 실제 명령은 다음과 같습니다.

```bash
python3 scripts/run_dataset_extractor_v1.py \
  --sim-root ~/physicar-ai-sim-docker
```

기본 input/output은 각각 다음과 같습니다.

```text
<sim-root>/userdata/physicar_e2e/rosbag_collector_v1_pilot/
<sim-root>/userdata/physicar_e2e/dataset_extractor_v1_pilot/
```

기존 output이 있으면 안전하게 실패합니다. 기존 dataset을 다시 만들기 위해
`--force`를 사용하기 전에 반드시 외부 artifact 보존 여부를 확인하십시오.

Image 변환은 다음과 같습니다.

```text
480×360 rgb8
→ crop x=0:480, y=160:360
→ Pillow bilinear resize 200×66
→ RGB PNG 저장
→ model input에서 direct RGB→YUV 및 normalization
```

초기 세 pilot episode에서는 **2,632개** usable camera/steering sample과 미래 label
위반 0건이 검증됐습니다. 상세 결과는
[`results/dataset_extractor_v1_pilot_summary.json`](results/dataset_extractor_v1_pilot_summary.json),
설명은 [`docs/dataset_extractor_v1.md`](docs/dataset_extractor_v1.md)에 있습니다.

## 5. 학습된 PilotNet에서 steering 값 확인

### A. Offline: camera 한 장 → steering 값

V4 ONNX는 대형 generated artifact라 Git에 포함되지 않습니다. 원래 workstation의
canonical 위치는 다음과 같으며, 다른 PC에서는 파일을 별도로 전달받아 원하는
외부 경로에 둡니다.

```text
~/physicar-ai-sim-docker/userdata/physicar_e2e/pilotnet_dagger_iteration2_v1/v4/onnx/pilotnet_v4_dagger.onnx
```

반드시 다음 identity를 확인합니다.

```text
SHA-256: 5dd2b88b50c43aed44361229dea34e8981cb0a34d05b01eef81a9ccdf63f396a
size: 1,012,518 bytes
```

Simulator의 현재 camera snapshot을 저장해 한 장 inference를 실행할 수 있습니다.

```bash
export V4_ONNX=/path/to/pilotnet_v4_dagger.onnx
sha256sum "$V4_ONNX"
curl -fsS http://localhost:8080/camera -o /tmp/physicar_camera.jpg

python3 scripts/pilotnet_pi5_infer.py \
  --model "$V4_ONNX" \
  --image /tmp/physicar_camera.jpg
```

CLI는 startup에서 ONNX hash/크기를 다시 검증하고 `steering_rad`, `steering_deg`,
`decode_ms`, `preprocess_ms`, `inference_ms`, `total_ms`를 출력합니다. 이것이 가장
작은 standalone `camera → PilotNet V4 → steering` 확인 방법입니다.

### B. Closed loop: Simulator에서 V4 주행

먼저 주행하지 않는 preflight를 실행합니다.

```bash
python3 scripts/run_pilotnet_v4_dagger.py \
  --config configs/pilotnet_inference_v4_dagger.json \
  --onnx "$V4_ONNX" \
  --preflight-only \
  --result /tmp/pilotnet_v4_handoff_preflight.json
```

다음 명령은 차량을 실제로 reset하고 0.50 m/s로 주행합니다. 사용자 승인과 안전한
simulator 상태가 확인된 경우에만 실행하십시오. 첫 run이 PASS하면 구현된 bounded
runner가 최대 세 개의 conditional run을 수행할 수 있습니다. 같은 result path는
재사용할 수 없습니다.

```bash
python3 scripts/run_pilotnet_v4_dagger.py \
  --config configs/pilotnet_inference_v4_dagger.json \
  --onnx "$V4_ONNX" \
  --run \
  --result /tmp/pilotnet_v4_handoff_run.json
```

주행 중 simulator 화면과 별도 terminal의 `/steering` 출력을 함께 보면 camera-only
PilotNet이 계산한 조향을 가장 직접적으로 확인할 수 있습니다.

## 6. 이미 검증된 결과

| 단계 | 측정된 결과 |
|---|---|
| Automated Expert | 0.50 m/s, 독립 5/5 lap PASS |
| PilotNet V1 | 0.30 m/s full lap PASS; 0.50 m/s는 2.953 m / 9.68%에서 FAIL |
| Synthetic Recovery V2 | 0.50 m/s는 2.591 m / 8.49%에서 FAIL |
| On-policy DAgger V3 | 0.50 m/s는 19.819 m / 64.97%까지 진행 |
| Cumulative DAgger V4 | 0.50 m/s full lap PASS |
| V4 repeatability | valid same-map/same-spawn 0.50 m/s lap 3/3 PASS |
| V4 repeatability mean CTE | 세 valid run 평균 `0.01868 m` |
| PilotNet architecture | 252,219 parameters |
| V4 ONNX | 1,012,518 bytes, 약 1.01 MB |

상세 증거:

- Expert: [`results/expert_repeatability_v1_summary.json`](results/expert_repeatability_v1_summary.json)
- V1/V2: [`results/pilotnet_e2e_v2/comparison.json`](results/pilotnet_e2e_v2/comparison.json)
- V1/V2/V3: [`results/pilotnet_e2e_v3/comparison.json`](results/pilotnet_e2e_v3/comparison.json)
- V4: [`results/pilotnet_e2e_v4/REPORT.md`](results/pilotnet_e2e_v4/REPORT.md)
- V4 3/3: [`results/pilotnet_v4_repeatability_v1/REPORT.md`](results/pilotnet_v4_repeatability_v1/REPORT.md)

### 왜 expert는 반복 주행하는데 초기 PilotNet은 실패했나?

Expert는 privileged route/pose를 사용하지만 PilotNet은 camera-only입니다. 작은
closed-loop prediction error가 누적되면서 nominal training 분포 밖의 camera 상태로
진입하는 on-policy distribution shift가 발생했습니다. 실제 neural policy가 방문한
상태에 shadow-expert label을 붙인 DAgger가 진행 거리를 `V1 약 2.95 m → V3 약
19.82 m → V4 full lap`으로 확장했습니다. 이는 동일 map 조건의 simulator 결과이며
일반적인 자율주행 또는 real-robot 성공을 뜻하지 않습니다.

## 7. Raspberry Pi 5 담당자 handoff

Standalone bundle은 [`deploy/pi5/`](deploy/pi5/)에 있습니다.

```text
deploy/pi5/
  README.md
  requirements.txt
  pilotnet_pi5.py
  pilotnet_pi5_core.py
  benchmark_pi5.py
  model_manifest.json
```

Runtime은 Python, NumPy, Pillow, ONNX Runtime만 필요하며 PyTorch/ROS/CUDA가
필요하지 않습니다. ONNX는 Git에 없으므로 위 canonical 파일을 별도로 전달받아
`deploy/pi5/pilotnet_v4_dagger.onnx`에 둡니다.

```bash
cd deploy/pi5
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python pilotnet_pi5.py \
  --model pilotnet_v4_dagger.onnx \
  --image frame.jpg

python benchmark_pi5.py \
  --model pilotnet_v4_dagger.onnx \
  --image frame.jpg \
  --warmup 20 \
  --iterations 250 \
  --output pi5_benchmark.json
```

현재 검증된 것은 x86 deployment parity와 host CPU benchmark입니다. **실제
Raspberry Pi 5 성능은 아직 측정하지 않았습니다.** Pi-side contract는 `RGB image →
PilotNet → steering_rad`입니다. 물리 차량의 steering protocol이 정해지지 않았으므로
ROS/serial/HTTP actuator integration은 의도적으로 만들지 않았습니다.

## 8. Repository 구조

```text
configs/             canonical/experimental JSON configuration
deploy/pi5/          standalone Raspberry Pi 5 inference bundle
docs/                각 milestone의 설계, gate, 사용법
results/             compact JSON/Markdown evidence와 provenance
scripts/             source-checkout CLI entry points
src/physicar_e2e/    expert, collector, extractor, PilotNet, deployment code
tests/               simulator 없이 실행 가능한 unit/regression tests
```

Raw rosbags, image dataset, checkpoints, ONNX, generated model과 대형 log는 Git 밖
simulator `userdata`에 둡니다. Source, tests, configs, docs와 compact metrics만 이
저장소에 보존합니다.

---

# 상세 구현 및 연구 문서

아래는 기존 README가 제공하던 milestone별 사용법입니다. 빠른 handoff 이후 더
자세한 연구 맥락이 필요할 때 사용합니다.

## Expert Driver V1

Source-checkout launcher는 canonical config를 자동으로 사용합니다.

```bash
python3 scripts/run_expert_driver_v1.py --preflight-only
python3 scripts/run_expert_driver_v1.py --dry-run 3
python3 scripts/run_expert_driver_v1.py --result results/expert_driver_v1_latest.json
```

Installed command는 explicit config를 요구합니다.

```bash
physicar-expert-v1 --config /path/to/expert_driver_v1.json --preflight-only
```

Expert는 privileged geometry를 사용하고, 안전 정지를 `finally`에서 수행합니다.
상세 내용은 [`docs/expert_driver_v1.md`](docs/expert_driver_v1.md)에 있습니다.

## PilotNet E2E Smoke V1

V1은 episode 001–002로 학습하고 episode 003 전체로 검증했습니다. Generated
checkpoint/ONNX/plot/log는 userdata에, compact provenance만 Git에 둡니다.

```bash
python3 scripts/train_pilotnet_v1.py \
  --config configs/pilotnet_training_v1.json \
  --dataset-root /path/to/dataset_extractor_v1_pilot \
  --artifact-root /path/to/userdata/physicar_e2e/pilotnet_v1 \
  --result results/pilotnet_training_v1/summary.json
```

이 명령은 역사적 재현 방법일 뿐 handoff 확인을 위해 재학습할 필요는 없습니다.
자세한 내용은 [`docs/pilotnet_v1.md`](docs/pilotnet_v1.md)에 있습니다.

## PilotNet Recovery / Diagnosis / DAgger

V2는 12개 fixed-anchor recovery trajectory를 평가한 보존된 negative result입니다.
Failure Diagnosis는 on-policy distribution shift를 지지했고, V3/V4는 실제 neural
policy 방문 상태에 expert label을 추가했습니다. Architecture와 preprocessing은 모든
버전에서 유지됐습니다.

- [`docs/recovery_data_v1.md`](docs/recovery_data_v1.md)
- [`docs/pilotnet_failure_diagnosis_v1.md`](docs/pilotnet_failure_diagnosis_v1.md)
- [`docs/pilotnet_dagger_v1.md`](docs/pilotnet_dagger_v1.md)
- [`docs/pilotnet_dagger_iteration2_v1.md`](docs/pilotnet_dagger_iteration2_v1.md)
- [`docs/pilotnet_v4_repeatability_v1.md`](docs/pilotnet_v4_repeatability_v1.md)

## Dataset Extractor / Rosbag Collector / Environment

- [`docs/dataset_extractor_v1.md`](docs/dataset_extractor_v1.md)
- [`docs/rosbag_collector_v1.md`](docs/rosbag_collector_v1.md)
- [`docs/lane_follow_environment_v1.md`](docs/lane_follow_environment_v1.md)

모든 simulator 성공은 simulator evidence일 뿐 real-robot 성공이 아닙니다.
