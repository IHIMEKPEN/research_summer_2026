# Real G1 stages (sense first, then oracle)

Safe order for bringing the wipe / ESN pipeline onto the Unitree G1.
**Do not send motion until Stage 2.** Stages **−2** and **−1** are perception-only.

Lab access: [G1_LAB_WHITEBOARD_NOTES.md](G1_LAB_WHITEBOARD_NOTES.md) · Deploy guide: [UNITREE_G1_EDU.md](UNITREE_G1_EDU.md)

**Why these stages (vs full physical agent)?** See [PHYSICAL_AGENT_AND_RESEARCH_STAGES.md](PHYSICAL_AGENT_AND_RESEARCH_STAGES.md).  
Short version: Stages 0–3 test whether a **known-good oracle plan** can run on G1 (research). Observability (instruction → Qwen → tools → arms) is the parallel agent track in S2R.

```text
  Stage −2  Connect (SSH / LAN)
       │
       ▼
  Stage −1  Live camera + depth + timing   ← done when FOV/fps known
       │
       ▼
  Stage 0   Laptop dry-run (oracle → ESN → g1.mock)
       │
       ▼
  Stage 1   Same on robot net, still mock
       │
       ▼
  Stage 2   Hardware, clamps on, short arm trial
       │
       ▼
  Stage 3   Full oracle wipe trial
```

Oracle vs live wipe (sim): oracle = smooth left-arm drag; live VLA often = grasp/bunch/lift. Stages 0–3 isolate **hardware+ESN** from that VLA intent failure.

**Today (arm + e-stop only):** see [TODAY_ARM_INIT_AND_ESTOP.md](TODAY_ARM_INIT_AND_ESTOP.md) — present robot (5-finger, arms down) → demo init (ep.160 arms up).

---

## Stage −2 — Connect to the Unitree G1

**Goal:** Host PC can reach the robot compute unit. No motion, no models.

| Link | SSH | Notes |
|------|-----|--------|
| Ethernet (preferred) | `ssh unitree@192.168.123.161` | Stable for DDS / sensors |
| WiFi | `ssh unitree@10.54.182.34` | DHCP — re-check if down |
| Password | `123` | Lab default; rotate if changed |

```bash
# From laptop on the robot LAN
ping -c 3 192.168.123.161
ssh unitree@192.168.123.161

# On the robot: confirm NIC + video devices
ip -br link
ls -la /dev/video* 2>/dev/null || echo "no /dev/video*"
hostname; uname -a
```

**Exit criteria**
- [ ] SSH works over Ethernet
- [ ] You know which interface DDS/control will use (`eth0` / `enp*`)
- [ ] At least one `/dev/video*` (or documented RealSense/Unitree camera path) appears

---

## Stage −1 — Live camera + depth (what the robot sees)

**Goal:** See RGB + depth from `G1_camera_v2.py`, measure receive FPS, optionally run YOLO-World. Prove data flow before VLA / ESN / wipe joints.

### Publisher + teammate viewer (preferred)

Terminal 1 on G1 (`teleimager` env):

```bash
conda activate teleimager
python G1_camera_v2.py --fps 10
# expects: Streaming on tcp://*:5555 | Topic: 'head_camera'
```

Terminal 2 (NoMachine / display) — **Stage −1 default is view-only (no motion)**:

```bash
# on G1:
python scripts/G1_yolo_world_follow_v2.py --host 127.0.0.1

# from laptop:
python scripts/G1_yolo_world_follow_v2.py --host 10.54.182.34
```

Shows RGB | depth heatmap side-by-side, YOLO person track, FPS overlay. Press `q` / `s`.

Only if you intentionally want loco + hug (clear space, e-stop ready):

```bash
python scripts/G1_yolo_world_follow_v2.py --host 127.0.0.1 --enable-motion --iface enP2p1s0
```

Lightweight RGB/depth viewer (no YOLO): `scripts/G1_camera_sub.py`.

| Signal | Meaning |
|--------|---------|
| Publisher `Streaming at ~10 fps` | Grab + pub rate |
| Viewer `Receiving at X fps` | ZMQ + decode (+ YOLO) rate |
| Depth heatmap | LiDAR/depth channel from multipart part 3 |

**Exit criteria**
- [x] Live RGB + depth look correct *(bring-up done; FOV is strongly downward)*
- [ ] Receive fps logged (≈ publisher fps; YOLO may be slower)
- [ ] Path understood: `G1_camera_v2 → tcp://G1:5555 topic head_camera → G1_yolo_world_follow_v2` — motion only with `--enable-motion`

**FOV note (paper + deploy):** Head camera looks mostly at the floor / near field, not straight ahead. That is fine for table wipe demos but weak for horizon obstacles and for VLA observations that are already temporally stale. Documented in both NeurIPS WRL and ICRA Discussion (\emph{Onboard sensing FOV}).

**Do not proceed to Stage 0 until Stage −1 view-only looks good.**

---

## Stage 0 — Laptop dry-run (no robot motion)

**Implemented:** `scripts/G1_oracle_demo_stage0.py`

```bash
cd research/scripts
source .venv/bin/activate

# Wiring dry-run (no HF):
python G1_oracle_demo_stage0.py --synthetic --seconds 12

# Real Dex1 ep 160 tokens (needs `datasets` + network):
python G1_oracle_demo_stage0.py --episode 160 --seconds 15

# Visual reference (already measured MuJoCo oracle):
open ../results/step4_mujoco_evaluation/table_wipe_ep160_oracle_esn.mp4
```

Publishes oracle `action_token` @ ~2 Hz and `joint_cmd` @ ~100 Hz on localhost ZMQ.
**Does not move the G1** (arms not wired for wipe yet). Config stub: `config/platforms/g1_oracle_demo.yaml` (`g1.mock: true`).

---

## Stage 1 — Robot network, still mock

Same as Stage 0 on `192.168.123.161` LAN with `g1.mock: true`. GUI / logs look sane.

---

## Stage 2 — Hardware, short trial

`g1.mock: false`, stance freeze + rate clamps **on**, arms-first / short duration, e-stop ready.

---

## Stage 3 — Full oracle wipe

Held-out wipe episode on real G1 with full safety checklist from [UNITREE_G1_EDU.md](UNITREE_G1_EDU.md).

---

## Data-flow sketch (Stage −1 focus)

```text
G1 RGB  ──► OpenCV / camera_node ──► camera_pub (ZMQ) ──► sense-check FPS + sample JPEG
G1 LiDAR ─► ROS2 / Unitree DDS   ──► (optional) point cloud rate

Later (Stages 0+):
  demo/VLA tokens ──► ESN ──► joint_cmd ──► g1_bridge ──► Unitree SDK
```

Perception timing here is **independent** of the ~1.75 Hz UnifoLM vs 100 Hz control gap; Stage −1 only answers: *can we see, and how fast can we ingest frames on this box?*
