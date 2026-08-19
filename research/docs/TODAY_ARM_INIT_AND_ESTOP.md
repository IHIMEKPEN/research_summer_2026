# Today’s success: arm go-to-init + e-stop

**Date focus:** Arm movement toward the wipe **demo initial pose**, plus knowing how to emergency-stop.  
Not today: full wipe, live VLA, Dex1 finger replay, 100 Hz ESN on hardware.

---

## Your present robot (lab, now)

| Item | State |
|------|--------|
| Platform | Unitree G1, overhead **tether on** |
| Hands | **5-finger Inspire RH56DFX-2L/2R** (not Dex1 2-finger in the HF wipe repo) — see [INSPIRE_RH56DFX_HAND.md](INSPIRE_RH56DFX_HAND.md) |
| Default pose | Standing, **arms hanging down** at sides |
| Workspace | White table in front; **dark cloth** + small **black rings** (“dirt”) |
| Camera | Head RGB+depth, **looking down** at table (`G1_camera_v2` `:5555`) |
| Control ZMQ | Oracle / arm cmds on **`:5557`** (do not reuse `:5555`) |
| Fingers today | **Do not drive** 5-finger joints — keep fixed / vendor default |

**Human instruction (mission text for later Scenario 1):**  
> Go to that table and wipe the dirt off the table using the cloth on the table.

---

## Demo initial position (what the sim video starts with)

Oracle wipe (`G1_Dex1_Wipe_Table` **episode 160**, first `observation.body` frame):

- Arms already **raised toward the table** (left arm ready to grasp/wipe; right often in a “ready” pose).
- Body standing so hands are in the table workspace — **not** hang-down.

That pose is the **go-to-init goal**. Your hang-down pose is the **start**. Today we only ramp arms (slowly) toward that init — not the full wipe trajectory.

Load goal joints with:

```bash
# when HF datasets available (on DGX / after pip install datasets):
python scripts/G1_arm_goto_init.py --episode 160 --live --seconds 20
# offline preset (no HF): approximate arms-up ready
python scripts/G1_arm_goto_init.py --preset arms_ready --live --seconds 20
```

---

## E-stop (do this before any motion)

**Your lab remote card (authoritative):**

| Action | Buttons | Press |
|--------|---------|--------|
| **① Damping (e-stop)** | **`L2 + B`** | short ★ |
| Damping protection | hold **①** ~5 s while in motion control | long |
| **② Locked Standing** | **`L2 + Up`** | short ★ |
| **③ Running Mode** | **`R2 + A`** | long ▲ (~2 s) |
| Regular Mode | `R1 + X` (1-DoF waist) / `R1 + Y` (3-DoF waist) | long ▲ |
| ⑤ Lie → Stand | `L2 + X` | short ★ |
| ⑥ Squat ↔ Stand | `L2 + A` | long ▲ |
| Stepping/Standing | Double-click **START** | ★ |

**Recommended seat power-on (from the card):**  
Turn on → ① Damping → ② Locked Standing → ③ Running Mode → demo → ④ Seated → off.  
**Tip on card:** ② and ③ need **manual help** to stand upright; keep the **tether** on.

**If you just hit damping (`L2 + B`) and want standing again:**
1. Support the robot (tether + hands on shoulders).
2. **`L2 + Up`** → Locked Standing (②) — assist it upright.
3. Long-press **`R2 + A`** → Running Mode (③) when you need walking/motion control.
4. For arm work at the table, **Locked Standing (②)** is usually enough to hold pose against gravity.
5. E-stop again anytime with **`L2 + B`**.

Do **not** use generic internet combos (`L1+A`, etc.) — **this sticker wins**.

---

## Today’s pass / fail

| Pass | Fail |
|------|------|
| Remote e-stop practiced (damping) with tether on | Unknown e-stop |
| Away from table / clear floor for first motion | Arms near table edge |
| Locked Standing (`L2+Up`), **not** Running Mode | Walking enabled |
| Live arm ramp script moves arms slowly toward ready | Fast jump / flail |
| Stop with **`L2+B`** or Ctrl+C | Continues after stop attempt |

### Live arm motion (on G1, open space)

```bash
# ON the G1, Locked Standing, tether on, remote L2+B ready, AWAY FROM TABLE:
python G1_arm_goto_init_live.py --iface enP2p1s0                 # dry-run print
python G1_arm_goto_init_live.py --iface enP2p1s0 --enable-motion # real move
# optional: --yes to skip Enter; --seconds 25 for slower ramp
```

Uses Unitree `rt/arm_sdk` (official arm PD path). Legs not commanded. E-stop = **`L2+B`**.
