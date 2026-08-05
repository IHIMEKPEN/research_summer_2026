# Dataset Card: G1_Dex1_Wipe_Table

**Source:** `unitreerobotics/G1_Dex1_Wipe_Table` (split=`train`)

## Summary

| Metric | Value |
|---|---|
| Rows (frames) | 75909 |
| Episodes | 200 |
| Episode indices | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199] |
| FPS (est.) | 30.000028610256777 |
| Total duration | 2523.633 s (0.7010 h)

## Columns / Features

```
observation.left_arm: List(Value('float32'), length=7)
observation.right_arm: List(Value('float32'), length=7)
observation.left_gripper: Value('float32')
observation.right_gripper: Value('float32')
observation.left_ee: List(Value('float32'), length=6)
observation.right_ee: List(Value('float32'), length=6)
observation.body: List(Value('float32'), length=29)
action.left_arm: List(Value('float32'), length=7)
action.right_arm: List(Value('float32'), length=7)
action.left_gripper: Value('float32')
action.right_gripper: Value('float32')
action.left_ee: List(Value('float32'), length=6)
action.right_ee: List(Value('float32'), length=6)
action.body: List(Value('float32'), length=7)
timestamp: Value('float32')
frame_index: Value('int64')
episode_index: Value('int64')
index: Value('int64')
task_index: Value('int64')
```

## Frames per episode

| episode_index | frames | duration_s |
|---:|---:|---:|
| 0 | 370 | 12.300 |
| 1 | 459 | 15.267 |
| 2 | 401 | 13.333 |
| 3 | 413 | 13.733 |
| 4 | 462 | 15.367 |
| 5 | 445 | 14.800 |
| 6 | 305 | 10.133 |
| 7 | 426 | 14.167 |
| 8 | 399 | 13.267 |
| 9 | 316 | 10.500 |
| 10 | 401 | 13.333 |
| 11 | 373 | 12.400 |
| 12 | 335 | 11.133 |
| 13 | 381 | 12.667 |
| 14 | 362 | 12.033 |
| 15 | 300 | 9.967 |
| 16 | 399 | 13.267 |
| 17 | 341 | 11.333 |
| 18 | 377 | 12.533 |
| 19 | 458 | 15.233 |
| 20 | 445 | 14.800 |
| 21 | 412 | 13.700 |
| 22 | 366 | 12.167 |
| 23 | 389 | 12.933 |
| 24 | 346 | 11.500 |
| 25 | 453 | 15.067 |
| 26 | 368 | 12.233 |
| 27 | 318 | 10.567 |
| 28 | 382 | 12.700 |
| 29 | 311 | 10.333 |
| 30 | 335 | 11.133 |
| 31 | 384 | 12.767 |
| 32 | 355 | 11.800 |
| 33 | 405 | 13.467 |
| 34 | 345 | 11.467 |
| 35 | 364 | 12.100 |
| 36 | 391 | 13.000 |
| 37 | 387 | 12.867 |
| 38 | 375 | 12.467 |
| 39 | 331 | 11.000 |
| 40 | 466 | 15.500 |
| 41 | 402 | 13.367 |
| 42 | 428 | 14.233 |
| 43 | 420 | 13.967 |
| 44 | 302 | 10.033 |
| 45 | 449 | 14.933 |
| 46 | 322 | 10.700 |
| 47 | 350 | 11.633 |
| 48 | 374 | 12.433 |
| 49 | 506 | 16.833 |
| 50 | 328 | 10.900 |
| 51 | 417 | 13.867 |
| 52 | 431 | 14.333 |
| 53 | 305 | 10.133 |
| 54 | 405 | 13.467 |
| 55 | 406 | 13.500 |
| 56 | 450 | 14.967 |
| 57 | 322 | 10.700 |
| 58 | 328 | 10.900 |
| 59 | 403 | 13.400 |
| 60 | 382 | 12.700 |
| 61 | 327 | 10.867 |
| 62 | 424 | 14.100 |
| 63 | 342 | 11.367 |
| 64 | 350 | 11.633 |
| 65 | 372 | 12.367 |
| 66 | 350 | 11.633 |
| 67 | 479 | 15.933 |
| 68 | 330 | 10.967 |
| 69 | 380 | 12.633 |
| 70 | 302 | 10.033 |
| 71 | 328 | 10.900 |
| 72 | 386 | 12.833 |
| 73 | 338 | 11.233 |
| 74 | 385 | 12.800 |
| 75 | 306 | 10.167 |
| 76 | 456 | 15.167 |
| 77 | 417 | 13.867 |
| 78 | 339 | 11.267 |
| 79 | 307 | 10.200 |
| 80 | 354 | 11.767 |
| 81 | 359 | 11.933 |
| 82 | 374 | 12.433 |
| 83 | 409 | 13.600 |
| 84 | 312 | 10.367 |
| 85 | 416 | 13.833 |
| 86 | 406 | 13.500 |
| 87 | 380 | 12.633 |
| 88 | 389 | 12.933 |
| 89 | 316 | 10.500 |
| 90 | 365 | 12.133 |
| 91 | 330 | 10.967 |
| 92 | 362 | 12.033 |
| 93 | 435 | 14.467 |
| 94 | 325 | 10.800 |
| 95 | 338 | 11.233 |
| 96 | 364 | 12.100 |
| 97 | 456 | 15.167 |
| 98 | 449 | 14.933 |
| 99 | 430 | 14.300 |
| 100 | 374 | 12.433 |
| 101 | 320 | 10.633 |
| 102 | 348 | 11.567 |
| 103 | 378 | 12.567 |
| 104 | 317 | 10.533 |
| 105 | 331 | 11.000 |
| 106 | 378 | 12.567 |
| 107 | 348 | 11.567 |
| 108 | 355 | 11.800 |
| 109 | 413 | 13.733 |
| 110 | 335 | 11.133 |
| 111 | 388 | 12.900 |
| 112 | 415 | 13.800 |
| 113 | 459 | 15.267 |
| 114 | 355 | 11.800 |
| 115 | 405 | 13.467 |
| 116 | 490 | 16.300 |
| 117 | 320 | 10.633 |
| 118 | 419 | 13.933 |
| 119 | 417 | 13.867 |
| 120 | 297 | 9.867 |
| 121 | 445 | 14.800 |
| 122 | 338 | 11.233 |
| 123 | 349 | 11.600 |
| 124 | 378 | 12.567 |
| 125 | 376 | 12.500 |
| 126 | 425 | 14.133 |
| 127 | 402 | 13.367 |
| 128 | 426 | 14.167 |
| 129 | 411 | 13.667 |
| 130 | 355 | 11.800 |
| 131 | 434 | 14.433 |
| 132 | 316 | 10.500 |
| 133 | 370 | 12.300 |
| 134 | 414 | 13.767 |
| 135 | 396 | 13.167 |
| 136 | 415 | 13.800 |
| 137 | 487 | 16.200 |
| 138 | 308 | 10.233 |
| 139 | 426 | 14.167 |
| 140 | 323 | 10.733 |
| 141 | 341 | 11.333 |
| 142 | 402 | 13.367 |
| 143 | 363 | 12.067 |
| 144 | 440 | 14.633 |
| 145 | 373 | 12.400 |
| 146 | 433 | 14.400 |
| 147 | 451 | 15.000 |
| 148 | 367 | 12.200 |
| 149 | 317 | 10.533 |
| 150 | 470 | 15.633 |
| 151 | 354 | 11.767 |
| 152 | 397 | 13.200 |
| 153 | 384 | 12.767 |
| 154 | 392 | 13.033 |
| 155 | 333 | 11.067 |
| 156 | 302 | 10.033 |
| 157 | 380 | 12.633 |
| 158 | 332 | 11.033 |
| 159 | 408 | 13.567 |
| 160 | 320 | 10.633 |
| 161 | 304 | 10.100 |
| 162 | 349 | 11.600 |
| 163 | 377 | 12.533 |
| 164 | 365 | 12.133 |
| 165 | 419 | 13.933 |
| 166 | 300 | 9.967 |
| 167 | 369 | 12.267 |
| 168 | 405 | 13.467 |
| 169 | 364 | 12.100 |
| 170 | 360 | 11.967 |
| 171 | 417 | 13.867 |
| 172 | 307 | 10.200 |
| 173 | 412 | 13.700 |
| 174 | 362 | 12.033 |
| 175 | 371 | 12.333 |
| 176 | 425 | 14.133 |
| 177 | 338 | 11.233 |
| 178 | 332 | 11.033 |
| 179 | 369 | 12.267 |
| 180 | 396 | 13.167 |
| 181 | 447 | 14.867 |
| 182 | 397 | 13.200 |
| 183 | 324 | 10.767 |
| 184 | 440 | 14.633 |
| 185 | 358 | 11.900 |
| 186 | 375 | 12.467 |
| 187 | 370 | 12.300 |
| 188 | 305 | 10.133 |
| 189 | 418 | 13.900 |
| 190 | 439 | 14.600 |
| 191 | 297 | 9.867 |
| 192 | 436 | 14.500 |
| 193 | 353 | 11.733 |
| 194 | 412 | 13.700 |
| 195 | 388 | 12.900 |
| 196 | 452 | 15.033 |
| 197 | 405 | 13.467 |
| 198 | 446 | 14.833 |
| 199 | 423 | 14.067 |

### Stats

- frames/ep: {'min': 297.0, 'max': 506.0, 'mean': 379.545, 'median': 377.0, 'std': 47.23852215088868, 'sum': 75909.0, 'p25': 340.5, 'p75': 415.0}
- duration_s/ep: {'min': 9.866666793823242, 'max': 16.83333396911621, 'mean': 12.618166661262512, 'median': 12.533333778381348, 'std': 1.5746174101996837, 'sum': 2523.6333322525024, 'p25': 11.3166663646698, 'p75': 13.800000190734863}

## Observation / action dims

```json
{
  "observation.left_arm": {
    "shape": [
      7
    ],
    "dtype": "float64"
  },
  "observation.right_arm": {
    "shape": [
      7
    ],
    "dtype": "float64"
  },
  "observation.left_gripper": {
    "type": "float",
    "value_preview": "4.49760103225708"
  },
  "observation.right_gripper": {
    "type": "float",
    "value_preview": "4.495545387268066"
  },
  "observation.left_ee": {
    "shape": [
      6
    ],
    "dtype": "float64"
  },
  "observation.right_ee": {
    "shape": [
      6
    ],
    "dtype": "float64"
  },
  "observation.body": {
    "shape": [
      29
    ],
    "dtype": "float64"
  },
  "action.left_arm": {
    "shape": [
      7
    ],
    "dtype": "float64"
  },
  "action.right_arm": {
    "shape": [
      7
    ],
    "dtype": "float64"
  },
  "action.left_gripper": {
    "type": "float",
    "value_preview": "4.5"
  },
  "action.right_gripper": {
    "type": "float",
    "value_preview": "4.489001274108887"
  },
  "action.left_ee": {
    "shape": [
      6
    ],
    "dtype": "float64"
  },
  "action.right_ee": {
    "shape": [
      6
    ],
    "dtype": "float64"
  },
  "action.body": {
    "shape": [
      7
    ],
    "dtype": "float64"
  }
}
```

## Language / task fields

```json
{
  "task_index": {
    "n_unique": 1,
    "value_counts": {
      "0": 75909
    }
  }
}
```

## Recommended train / held-out split

Hold out last 20% of episodes by index (40/200).

```
TRAIN_EPISODES=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159]
HELDOUT_EPISODES=[160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
```

- train frames: 60763 (80.0%)
- held-out frames: 15146 (20.0%)

## When model claims are valid / invalid

### Valid when
- Success/metrics reported on held-out episodes never used for ESN/readout training or hyperparameter selection.
- Episode-level split used (not random frame split) to avoid temporal leakage.
- Dataset scale disclosed: n_episodes=200, total_frames=75909, total_hours=0.7010092589590284.
- Sim vs real domain stated; wipe-table demos only support claims for this task distribution.

### Invalid when
- Reporting train-episode success as generalization.
- Random frame/train-test split across interleaved episode frames (leakage).
- Claiming broad G1 dexterity / multi-task competence from this single wipe-table corpus alone.
- Closed-loop UnifoLM success rates claimed without live Step-3 logs (repo research rule).
- Overstating robustness when n_episodes is small or held-out N < 2.

_Machine-readable twin: `/home/aihimekpen/research_summer_2026/research/results/step2_training/dataset_card_wipe_table.json`_

## Robotics data-declaration checklist

Declare these whenever you report a model number:

1. **Task / embodiment:** Unitree G1 Dex1 wipe-table demos only (`task_index=0`); 29-DoF `observation.body`.
2. **Scale:** 200 episodes, 75,909 frames, ~30 FPS raw, ~0.70 hours total.
3. **Split:** episode-level train `0–159` / held-out `160–199` (no random frame split).
4. **Train vs eval leakage:** ESN `W_out` must list `train_episodes` in `models/esn_cuda_ridge/config.json`; held-out metrics only for generalization.
5. **Oracle vs live:** Step 2/3-offline/4 use **dataset tokens** (oracle). Step 3 dual-thread uses **live UnifoLM** (timing/closed-loop).
6. **Domain:** sim MuJoCo replay ≠ real G1; S2R is a separate claim.
7. **Duration variability:** episodes ~9.9–16.8 s — report mean±std over episodes, not a single demo.
8. **Failure modes to watch:** grasp success without table contact; good joint RMSE with poor wipe path; train-ep videos sold as held-out.

