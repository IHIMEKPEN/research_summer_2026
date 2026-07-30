# Data layout

```text
data/
  raw/                      # live JSONL from data_collector
  processed/                # profiles, aggregates
  models/                   # deployable ESN weights, etc.
  benchmark/
    tasks.yaml              # 12-task catalog (Unitree G1 / UnifoLM-VLA)
    tasks/
      01_stack_block/
      ...
      12_dualrobot_clean_table/
        episodes/           # task episode logs
        videos/
        annotations/
        metrics/            # per-episode score JSON
    results/                # flat score dump for leaderboards
    splits/                 # train/val/test episode id lists
  unitree_vla/
    checkpoints/            # UnifoLM-VLA weights
    predictions/            # action token dumps
    evals/                  # eval summaries
  esn/
    train/pairs.npz
    val/pairs.npz
    checkpoints/
    curves/
  datasets/
    raw_teleop/
    converted/
    cache/
```

## 12 benchmark tasks

Defined in [`benchmark/tasks.yaml`](benchmark/tasks.yaml), aligned with Unitree open datasets:

| ID | Task | HF dataset |
|---|---|---|
| 01 | Stack Block | `unitreerobotics/G1_Stack_Block` |
| 02 | Bag Insert | `unitreerobotics/G1_Bag_Insert` |
| 03 | Erase Board | `unitreerobotics/G1_Erase_Board` |
| 04 | Clean Table | `unitreerobotics/G1_Clean_Table` |
| 05 | Pack Pencil Box | `unitreerobotics/G1_Pack_PencilBox` |
| 06 | Pour Medicine | `unitreerobotics/G1_Pour_Medicine` |
| 07 | Pack Ping Pong | `unitreerobotics/G1_Pack_PingPong` |
| 08 | Prepare Fruit | `unitreerobotics/G1_Prepare_Fruit` |
| 09 | Organize Tools | `unitreerobotics/G1_Organize_Tools` |
| 10 | Fold Towel | `unitreerobotics/G1_Fold_Towel` |
| 11 | Wipe Table | `unitreerobotics/G1_Wipe_Table` |
| 12 | Dual-Robot Clean Table | `unitreerobotics/G1_DualRobot_Clean_Table` |

Use notebooks in `../notebooks/` to train ESN and score models.
