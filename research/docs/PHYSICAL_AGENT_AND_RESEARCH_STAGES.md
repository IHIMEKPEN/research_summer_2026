# Physical agents + research stages (how they fit)

Two tracks share the same G1 / S2R stack. Do not conflate them.

```text
TRACK A — Research (ICRA / NeurIPS frequency-gap paper)
  Prove: slow VLA intent + fast ESN bridge on wipe demos
  Stages −2 → 3 below

TRACK B — Physical agent (long-term)
  Observe: what it sees, instruction, Qwen reason, decision, tool calls (arms/wrist)
  Debug: when wipe fails, know *which layer* failed
```

Your live head_camera stream (Stage −1) is the first brick of **both**.

---

## What your sim videos already showed

| | Oracle ESN (demo tokens) | Live UnifoLM |
|--|--------------------------|--------------|
| Strategy | Left arm linear drag; right “ready” | Both arms; grasp→bunch→lift→press→reset |
| Cloth | Flat, table contact | Scrunch / lift off table |
| Time | ~10.6 s, coherent | ~28.7 s, incomplete |
| Diagnosis | Plan is correct → bridge can wipe | VLA treats wipe as **grasp/manipulate**, not drag |

So failures are often **outer-loop intent / contact**, not “ESN too slow.”  
Stages 0–3 exist to **separate** those causes on hardware.

---

## TRACK A — What each stage achieves

| Stage | Goal | Pass means |
|-------|------|------------|
| **−2** Connect | SSH / LAN to G1 | Robot reachable |
| **−1** Sense | Live RGB+depth (`G1_camera_v2` → viewer) | You know FOV (downward), fps, wire format |
| **0** Oracle dry-run | HF/demo episode → ESN → `g1.mock` | Rate + clamps OK **without** blaming VLA vision |
| **1** Mock on robot net | Same as 0 on G1 LAN | DDS/ZMQ path OK, still no motion |
| **2** Short hardware | Arms/wrists, clamps on, e-stop | Real joints track a **known-good** oracle plan |
| **3** Full oracle wipe | Episode 160-class wipe on G1 | Hardware can execute the *good* video behavior |

**Why oracle before live VLA on G1?**  
If Stage 3 looks like Video 1 (smooth left-arm drag) and live VLA still looks like Video 2, the bug is **policy/intent/priors** (press contact, unilateral wipe), not ESN rate or motor wiring.

That is research testing of *your* claim: frequency bridge + honest live limits — not yet the full agent product.

---

## TRACK B — Physical agent observability (your long-term goal)

You want, when something goes wrong, a single timeline:

```text
HUMAN (speech/GUI/typed)
    │  instruction e.g. "Go to that table and wipe the dirt
    │  off using the cloth on the table."
    ▼
mission_node  ──► mission_pub {instruction, phase}
    │
    ├──────────────────────────────┐
    ▼                              ▼
vision (RGB+depth, VLM caption)   Qwen reasoner
    │                              │
    │                              ▼
    │                     decision_pub {
    │                       intent: approach_table | wipe | hold | ...
    │                       reason: "cloth on table, safe to wipe"
    │                       allow_motion: true/false
    │                       risk: ...
    │                     }
    │                              │
    └──────────► VLA (~2 Hz) ◄─────┘  (language + image + gated by allow_motion)
                      │
                      ▼
                 ESN (100 Hz) → joint_cmd / arm-wrist tools
                      │
                      ▼
                 logged JSONL + GUI  (replay: why did it bunch the cloth?)
```

### Instruction path (important)

Slightly different from “Qwen alone talks to the VLA”:

| Source | What it carries |
|--------|------------------|
| **Human → mission** | Full natural-language instruction (Scenario 1 text) |
| **Qwen reasoner** | *Grounded decision*: intent, short `reason`, `allow_motion`, risk — supervisor / phase machine |
| **VLA (UnifoLM)** | Same instruction (or phase-conditioned goal) **+** camera frames → sparse action tokens |

So for human interaction: **Qwen decides *whether / which phase***; **VLA executes *how* to move** under that language+vision. Tool calls (arm/wrist) sit beside or below that, also gated by `allow_motion`.

Today in code: `reasoning_node` publishes `Decision`; `vla_node` reads mission + decision (`allow_motion`, goal/phase). UnifoLM still needs to be plugged into `VLANode.infer()` for real wipe language.

### Scenario 1 — table wipe (next real-robot agent test)

**Setup:** G1 facing the table; cloth on table; clear workspace; e-stop ready.

**Instruction (mission):**
> Go to that table and wipe the dirt off the table using the cloth on the table.

**Success looks like Video 1 (oracle):** unilateral drag, cloth flat on surface — not Video 2 (bunch/lift/re-grasp).

**Minimum observability to log every tick / VLA step:**
1. Latest head RGB (+ whether cloth/table visible given downward FOV)
2. Mission `instruction`
3. Qwen `intent` + `reason` + `allow_motion`
4. VLA token / goal
5. Optional tool: `wipe_stroke` / arm cmd
6. Contact / EE height proxy if available

**Safe ladder on hardware (do not skip):**
1. Stage −1: confirm FOV actually sees **table + cloth** (pitch head if floor-only).
2. Stage 0–3: oracle replay once so motors+ESN match Video 1.
3. Scenario 1 live: mission text → Qwen gate → UnifoLM → ESN, with same `press_table`-class contact prior / clamps as sim, short duration.

If (3) still bunches cloth while (2) was smooth, fix VLA/priors/instruction grounding — not the ESN rate.

### Already in this repo (S2R)


| Need | Where |
|------|--------|
| Instruction / phase | `mission_node` → `mission_pub` |
| See + caption | `camera` / `G1_camera_v2` + `vision_node` (YOLO / Qwen2.5-VL) |
| Reason | `reasoning_node` + Qwen2.5 Instruct → `decision_pub` |
| Gate motion | `allow_motion` consumed by VLA / ESN / `g1_bridge` |
| Arms (high-level) | `g1_bridge` loco + arm hooks; teammate hug via `G1ArmActionClient` |
| Inspect later | `data_collector` JSONL + GUI `:8080` |

### Still thin / next to build for agents

- **Tool-call schema** for arms/wrist (named tools: `move_ee`, `open_hand`, `wipe_stroke`) logged next to Qwen’s `reason`
- **Agent trace UI**: side-by-side image + instruction + decision text + last tool + joint snapshot
- Wire Stage −1 camera multipart into S2R `camera` / vision (not only standalone scripts)
- FOV / head pitch check before trusting wipe-from-vision (noted in both papers)

### Scripts folder role (now)

| Script | Track | Role |
|--------|-------|------|
| `G1_camera_view.py` | A+B | See what the robot sees (no YOLO) |
| `G1_camera_sub.py` | A+B | Same, with optional timing JSON |
| `G1_yolo_world_follow_v2.py` | B (caution) | See + detect; motion only with `--enable-motion` |

These are **sensors / debug**, not the full agent. Full agent = `python -m s2r.cli deploy -c config/platforms/g1_edu.yaml` with GUI + collector + Qwen servers.

---

## How to debug wipe using both tracks

When live wipe looks like Video 2:

1. **Stage −1** — Was the cloth even in FOV? (your feed is floor-heavy.)
2. **Stage 0–3 oracle** — With demo tokens, does G1 do Video‑1-like drag?  
   - Yes → VLA/prior problem (grasp vs wipe, need `press_table`-class contact prior or better instruction).  
   - No → bridge / IK / clamps / hardware.
3. **Agent log** — Did Qwen say `wipe` or `grasp`? Was `allow_motion` true? Which tool fired?

That is the point of observability: **fix the right layer**.

---

## Recommended order from today

1. Keep using `G1_camera_view.py` — for Scenario 1, **pitch until table+cloth are in frame** (not floor-only).
2. **Stage 0–3** oracle → ESN on G1 once (research + safety): prove Video‑1 motion is achievable on hardware.
3. **Scenario 1 live agent test:** mission instruction → Qwen decision log → UnifoLM → ESN, short horizon, clamps + contact prior, full GUI/JSONL trace.
4. Harden arm/wrist **tool calls** next to Qwen intents (`wipe_stroke`, etc.).

Target instruction for (3):
`Go to that table and wipe the dirt off the table using the cloth on the table.`

