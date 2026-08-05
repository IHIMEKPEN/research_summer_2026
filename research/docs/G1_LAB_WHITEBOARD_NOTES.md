# G1 lab whiteboard notes (real robot)

Transcribed from lab whiteboard photo on **2026-08-05**.

Source image: [`assets/g1_lab_whiteboard_2026-08-05.png`](assets/g1_lab_whiteboard_2026-08-05.png)

Primary deploy guide: [UNITREE_G1_EDU.md](UNITREE_G1_EDU.md)

Left-side DSP notes on the board (`fs`, bandwidths, symbol/bit rates) are unrelated coursework and are **not** recorded here.

## Access

| Field | Value |
|---|---|
| Robot | Unitree **G1** |
| Ethernet SSH | `unitree@192.168.123.161` |
| WiFi SSH | `unitree@10.54.182.34` |
| Password | `123` |

```bash
# preferred (tethered)
ssh unitree@192.168.123.161

# wifi (IP may change with DHCP)
ssh unitree@10.54.182.34
```

## Socket pub/sub topology (whiteboard demo)

Rule written on the board:

- **1 Pub per Port**
- **Multi Sub per Port**

```text
Gesture Pub/Sub ──► Socket 2 (port 5556) ──┐
                                           │
Camera Pub ───────► Socket 1 (port 5555) ──┼──► Display Sub
                                           │         │
Yolo Sub / Yolo Pub ► Socket 3 (port 5554)─┘         ▼
                                              Decision Pub/Sub
                                                     │
                                                     ▼
                                              Socket 4 (port 5557)
                                                     │
                                                     ▼
                                              Motor Control Sub
```

| Socket | Port | Publisher side | Subscribers (on board) |
|---|---|---|---|
| 1 | **5555** | Camera Pub | Display Sub (+ Decision path) |
| 2 | **5556** | Gesture Pub/Sub | Display Sub / Decision |
| 3 | **5554** | Yolo Sub/Yolo Pub | Display Sub / Decision |
| 4 | **5557** | Decision Pub/Sub | Motor Control Sub |

## Relation to this repo’s S2R ZMQ map

`config/platforms/g1_edu.yaml` uses overlapping port numbers with **different roles**:

| Port | Whiteboard role | S2R `g1_edu.yaml` role |
|---|---|---|
| 5554 | YOLO | *(unused in default map)* |
| 5555 | Camera | `state_pub` |
| 5556 | Gesture | `action_token_pub` |
| 5557 | Motor / Decision out | `joint_cmd_pub` |
| 5558 | — | `decision_pub` |

Do **not** assume the whiteboard names match S2R node names. Use whiteboard ports only when running that demo stack; use `g1_edu.yaml` for ICRA S2R deploy.

## Config reminder for live G1

```yaml
g1:
  mock: false
  mode: high_level
  iface: eth0
  network: "192.168.123.161"
```
