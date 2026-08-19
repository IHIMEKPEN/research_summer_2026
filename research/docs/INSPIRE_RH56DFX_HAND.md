# Inspire Robots RH56DFX dexterous hands (lab G1)

Source: manufacturer page / phone screenshot (en.inspire-robots.com), saved 2026-08-13.

## Models on this robot
| Side | Model |
|------|--------|
| Left | **RH56DFX-2L** |
| Right | **RH56DFX-2R** |

Branding on hardware: **INSPIRE-ROBOTS** (white palm, silver finger links).

## Specs (RH56DFX series)
| Parameter | Value |
|-----------|--------|
| Degrees of freedom | **6** |
| Number of joints | **12** |
| Weight | **540 g** |
| Repeatability | **±0.20 mm** |
| Wrist (integrated) | null (no integrated wrist yaw/pitch/load torque in this table) |
| Control interface | **RS485** |
| Operating voltage | **DC 24 V ±10%** |
| Quiescent current | **0.09 A** |
| Peak current | **2 A** |
| Thumb fingertip strength | **15 N** |
| Other fingertips strength | **10 N** |
| Force resolution | **0.50 N** |
| Thumb lateral rotation range | **> 65°** |
| Thumb lateral rotation speed | **107°/s** |
| Thumb flexion speed | **70°/s** |
| Four-finger flexion speed | **260°/s** |

## Control stack notes (lab)
- Official Unitree bridge for DFX: [`unitreerobotics/dfx_inspire_service`](https://github.com/unitreerobotics/dfx_inspire_service) → DDS `rt/inspire/cmd` / `rt/inspire/state` (12 motors both hands; `q` only).
- FTP variants need different drivers; this unit is **DFX** per model numbers above.
- **Not** Dex1 2-finger. Wipe UnifoLM / `G1_Dex1_Wipe_Table` demos do **not** match this hand’s action space.

## Research implications
- Wipe UnifoLM / `G1_Dex1_Wipe_Table` use **Dex1-1 2-finger** grippers. MuJoCo now attaches Dex1-1 (and refuses Inspire / Dex3 `with_hand` models) so contact matches the VLA/ESN corpus.
- Lab G1 currently has these Inspire RH56DFX hands. **Purchase and attach Dex1-1** before claiming hardware wipe transfer; do not replay Dex1 gripper channels onto Inspire fingers.
- Safe bring-up until Dex1-1 is mounted: scripted Inspire open/close via inspire service (hands only); arms separate.
- VLA for Inspire would need G1+Inspire datasets/ckpts (e.g. community GR00T fine-tunes), which is a different embodiment than this paper.
