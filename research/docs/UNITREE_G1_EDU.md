# Running S2R on Unitree G1 Edu

This guide deploys the real-to-sim ZMQ pipeline on a **Unitree G1 Edu** humanoid for missions like:

> “Go bring me a pen from the table somewhere in the lab.”

## Architecture on G1

```text
G1 cameras ──► camera/vision (YOLO + Qwen2.5-VL)
                      │
                      ▼
              mission + Qwen reasoner ──► decisions (logged + GUI)
                      │
                      ▼
                 VLA @ ~2Hz ──► ESN @ 50Hz+ ──► g1_bridge
                                                    │
                                        unitree_sdk2py LocoClient / arm hooks
                                                    │
                                               G1 Edu body
```

Everything (perception, decisions, commands, metrics) is logged by `data_collector` and shown on the monitor GUI.

## 1. Network & access

Lab whiteboard credentials (transcribed 2026-08-05; source image: [`assets/g1_lab_whiteboard_2026-08-05.png`](assets/g1_lab_whiteboard_2026-08-05.png)):

| Link | SSH | Notes |
|---|---|---|
| Ethernet | `ssh unitree@192.168.123.161` | Preferred for deploy / DDS |
| WiFi | `ssh unitree@10.54.182.34` | DHCP — re-check if unreachable |
| Password | `123` | Unitree default; change if rotated on-site |

1. Connect host PC ↔ G1 Ethernet (this lab robot: `192.168.123.161`; other units often use `192.168.123.164`).
2. SSH into the G1 compute unit (Ubuntu + ROS2 Humble on Edu images).
3. Confirm interface name (`ip link`) — often `eth0` / `enp*`.
4. Set in config:

```yaml
g1:
  mock: false
  mode: high_level
  iface: eth0
  network: "192.168.123.161"
  control_hz: 50
```

Pub/sub socket map from the same whiteboard (demo / teaching topology — not identical to `config/platforms/g1_edu.yaml` ZMQ names): see [G1_LAB_WHITEBOARD_NOTES.md](G1_LAB_WHITEBOARD_NOTES.md).

Official docs: [Unitree developer quick start](https://support.unitree.com/home/en/developer/Quick_start)

## 2. Install SDK & R2S

On the G1 (or a companion NUC on the same LAN):

```bash
# Unitree Python SDK2
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .

# S2R / research pipeline
cd /path/to/research_summer_2026/research
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

Optional ROS2 path: [`unitree_ros2`](https://github.com/unitreerobotics/unitree_ros2) for `/lowstate` `/lowcmd` debugging.

## 3. AI models on G1 / Jetson Thor

G1 Edu AI compute is intended for onboard perception/reasoning. Recommended:

| Process | Model | Serving |
|---|---|---|
| Detector | YOLOv8n | in-process ultralytics |
| VLM | Qwen2.5-VL-3B | vLLM `:9000` |
| Reasoner | Qwen2.5-3B-Instruct | vLLM `:8000` |
| Control upsample | ESN | in-process CPU/CUDA |

Use platform config:

```bash
python -m s2r.cli deploy -c config/platforms/g1_edu.yaml
```

Before first live run, keep:

```yaml
g1:
  mock: true   # dry-run on robot network without sending loco commands
```

Then set `mock: false` after verifying GUI decisions look sane.

## 4. Safety checklist (do not skip)

1. Clear workspace; e-stop reachable.
2. Start in **high_level** mode only (`g1.mode: high_level`).
3. Confirm `allow_motion` gating in GUI before enabling mock:false.
4. Low-level `LowCmd` (@ ~500 Hz) is **not** enabled by default — implement only after `MotionSwitchClient.ReleaseMode()` and PD tuning.
5. Begin with stand / zero velocity; then explore intents.
6. Keep data collection on for post-run audit.

## 5. Bring-pen mission runbook

```bash
# 1) Start model servers (Thor/G1 GPU)
vllm serve Qwen/Qwen2.5-3B-Instruct --port 8000
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --port 9000 --trust-remote-code

# 2) Deploy pipeline
python -m s2r.cli deploy -c config/platforms/g1_edu.yaml

# 3) Open monitor
# http://<g1-or-host-ip>:8080
```

Expected phase timeline in GUI **Mission** panel:

1. `explore` — mapping / search  
2. `locate` — table found  
3. `approach` — pen detected  
4. `grasp` — grasp zone  
5. `return` — carrying  
6. `handoff` — person detected  
7. `done`

## 6. What `g1_bridge` does today

File: `src/s2r/nodes/g1_bridge.py`

- Subscribes to `joint_cmd`, `decision`, `mission`
- In high-level mode, maps intents → `LocoClient.Move(vx, vy, yaw_rate)`
- Keeps arm joint vector for your site-specific arm retarget hook
- Publishes metrics to GUI (`mock`, `intent`, `hz`)

You should customize arm grasp / Dex3 hand control for your Edu configuration (23/29 DoF + hand options).

## 7. Logging & replay

Episodes land in `data/raw/episode_*.jsonl` with:

- perception captions / detections
- Qwen decisions + risk
- mission phase transitions
- VLA tokens + ESN joint commands
- latency metrics

Use these for ESN training (`python -m s2r.cli train`) and failure analysis.

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| SDK import fails | Install `unitree_sdk2_python`, check Python version |
| No loco motion | `g1.mock:false`, correct `iface`, robot not estopped |
| DDS conflicts with ROS2 | unset `CYCLONEDDS_HOME`; avoid domain collisions |
| VLM OOM | use 3B + quantization; lower `gpu-memory-utilization` |
| Jerky arms | keep ESN target_hz at 50; lower VLA amplitudes |

## References

- Unitree SDK2 Python: https://github.com/unitreerobotics/unitree_sdk2_python  
- Unitree ROS2: https://github.com/unitreerobotics/unitree_ros2  
- Support portal: https://support.unitree.com/  
- Model guide: [MODELS.md](MODELS.md)  
- Profiling: [PROFILING_V100_THOR.md](PROFILING_V100_THOR.md)
