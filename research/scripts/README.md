# scripts/ — G1 sense & debug

Long-term physical agent stack lives in `src/s2r/` (mission → vision → Qwen reason → decision → VLA/ESN → g1_bridge → GUI/logs).

These scripts are **bring-up / observe** tools for Stage −1 and day-to-day FOV checks.

| Script | Purpose |
|--------|---------|
| `G1_camera_view.py` | Live RGB + depth from `G1_camera_v2.py` (no YOLO, no motion) |
| `G1_camera_sub.py` | Same stream + optional sample JPEG / timing |
| `G1_yolo_world_follow_v2.py` | RGB + depth + YOLO-World; **motion off** unless `--enable-motion` |
| `G1_arm_goto_init.py` | ZMQ-only hang→init ramp (no SDK motion) |
| `G1_arm_goto_init_live.py` | **Real arms** via `rt/arm_sdk` (run on G1, `--enable-motion`) |
| `G1_joint_cmd_live_sub.py` | Live SUB for joint_cmd (log-only; run on G1) |

```bash
# Mac
source .venv/bin/activate
python G1_camera_view.py --host 10.54.182.34

# G1 (publisher already up)
python G1_camera_view.py --host 127.0.0.1
```

How this connects to research Stages 0–3 and the Qwen/tool-call agent goal:  
**[docs/PHYSICAL_AGENT_AND_RESEARCH_STAGES.md](../docs/PHYSICAL_AGENT_AND_RESEARCH_STAGES.md)**
