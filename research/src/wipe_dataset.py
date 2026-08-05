"""
G1_Dex1_Wipe_Table episode splits + light dataset utilities.

Canonical split (episode-level, last 20% held out):
  train   : episodes 0 .. 159  (160 eps, ~80% frames)
  held-out: episodes 160 .. 199 (40 eps, ~20% frames)

Regenerate the human/JSON cards:
  python3 -m src.wipe_dataset --write-card
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.paths import results_path

logger = logging.getLogger(__name__)

DATASET_ID = "unitreerobotics/G1_Dex1_Wipe_Table"
N_EPISODES = 200
TRAIN_EPISODES: Tuple[int, ...] = tuple(range(0, 160))
HELDOUT_EPISODES: Tuple[int, ...] = tuple(range(160, 200))


def parse_episode_spec(spec: str) -> List[int]:
    """
    Parse episode lists from CLI/notebook strings.

    Examples:
      "0" → [0]
      "0,1,5" → [0,1,5]
      "0-3" → [0,1,2,3]
      "train" / "heldout" / "all"
      "0-7,160-162"
    """
    text = (spec or "").strip().lower()
    if not text:
        raise ValueError("Empty episode spec")
    if text in {"train", "training"}:
        return list(TRAIN_EPISODES)
    if text in {"heldout", "held-out", "test", "eval"}:
        return list(HELDOUT_EPISODES)
    if text == "all":
        return list(range(N_EPISODES))

    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a_str, b_str = part.split("-", 1)
            a, b = int(a_str), int(b_str)
            if b < a:
                raise ValueError(f"Bad range {part!r}")
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    if not out:
        raise ValueError(f"No episodes parsed from {spec!r}")
    # preserve order, drop dups
    seen = set()
    uniq: List[int] = []
    for ep in out:
        if ep not in seen:
            seen.add(ep)
            uniq.append(ep)
    return uniq


def split_name_for_episodes(episodes: Sequence[int]) -> str:
    eps = list(episodes)
    if eps == list(TRAIN_EPISODES):
        return "train"
    if eps == list(HELDOUT_EPISODES):
        return "heldout"
    if eps == list(range(N_EPISODES)):
        return "all"
    if len(eps) == 1:
        return f"ep{eps[0]}"
    return f"ep{eps[0]}-{eps[-1]}_n{len(eps)}"


def load_wipe_dataset(dataset_id: str = DATASET_ID, split: str = "train"):
    from datasets import load_dataset

    return load_dataset(dataset_id, split=split)


def summarize_dataset(dataset) -> Dict:
    """Compute episode-level robotics data stats for declaration / cards."""
    eps = [int(r["episode_index"]) for r in dataset]
    ctr = Counter(eps)
    episode_ids = sorted(ctr)

    by_ep_ts: Dict[int, List[float]] = {e: [] for e in episode_ids}
    if "timestamp" in dataset.column_names:
        for r in dataset:
            by_ep_ts[int(r["episode_index"])].append(float(r["timestamp"]))

    frames = np.asarray([ctr[e] for e in episode_ids], dtype=np.float64)
    durs = []
    for e in episode_ids:
        ts = by_ep_ts.get(e) or []
        if len(ts) >= 2:
            ts_arr = np.sort(np.asarray(ts, dtype=np.float64))
            durs.append(float(ts_arr[-1] - ts_arr[0]))
        else:
            durs.append(float("nan"))
    durs_arr = np.asarray(durs, dtype=np.float64)
    finite = durs_arr[np.isfinite(durs_arr)]
    fps = float(frames.sum() / finite.sum()) if finite.size and finite.sum() > 0 else float("nan")

    body0 = np.asarray(dataset[0]["observation.body"], dtype=np.float64)
    task_vals = Counter(int(r["task_index"]) for r in dataset) if "task_index" in dataset.column_names else {}

    def _stats(x: np.ndarray) -> Dict[str, float]:
        x = x[np.isfinite(x)]
        return {
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "std": float(np.std(x)),
            "sum": float(np.sum(x)),
            "p25": float(np.percentile(x, 25)),
            "p75": float(np.percentile(x, 75)),
        }

    return {
        "dataset_id": DATASET_ID,
        "split": "train",
        "n_rows": int(len(dataset)),
        "n_episodes": int(len(episode_ids)),
        "episode_indices": episode_ids,
        "frames_per_episode": {str(e): int(ctr[e]) for e in episode_ids},
        "duration_s_per_episode": {str(e): float(d) for e, d in zip(episode_ids, durs_arr)},
        "frames_stats": _stats(frames),
        "duration_s_stats": _stats(finite) if finite.size else {},
        "fps_estimate": fps,
        "total_duration_s": float(finite.sum()) if finite.size else float("nan"),
        "total_hours": float(finite.sum() / 3600.0) if finite.size else float("nan"),
        "observation_body_dim": int(body0.size),
        "task_index_counts": {str(k): int(v) for k, v in sorted(task_vals.items())},
        "train_episodes": list(TRAIN_EPISODES),
        "heldout_episodes": list(HELDOUT_EPISODES),
        "train_n_episodes": len(TRAIN_EPISODES),
        "heldout_n_episodes": len(HELDOUT_EPISODES),
        "claims": {
            "valid_when": [
                "Metrics reported on held-out episodes never used for ESN readout training or hyperparameter selection.",
                "Episode-level split (not random frames) to avoid temporal leakage within demos.",
                "Dataset scale disclosed (n_episodes, hours, FPS, single task_index).",
                "Sim vs real domain stated; wipe-table demos only support this task distribution.",
            ],
            "invalid_when": [
                "Train-episode success reported as generalization.",
                "Random frame split across interleaved episode frames (leakage).",
                "Broad multi-task / G1 dexterity claims from this wipe-only corpus alone.",
                "Closed-loop UnifoLM success rates without live Step-3 logs.",
            ],
        },
    }


def write_dataset_card(out_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    out_dir = Path(out_dir or results_path("step2_training"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_wipe_dataset()
    summary = summarize_dataset(ds)
    json_path = out_dir / "dataset_card_wipe_table.json"
    md_path = out_dir / "DATASET_CARD.md"
    json_path.write_text(json.dumps(summary, indent=2))

    fs = summary["frames_stats"]
    ds_ = summary["duration_s_stats"]
    md = f"""# Dataset Card: G1_Dex1_Wipe_Table

**Source:** `{DATASET_ID}` (split=`train`)

## Summary

| Metric | Value |
|---|---|
| Rows (frames) | {summary['n_rows']} |
| Episodes | {summary['n_episodes']} |
| FPS (est.) | {summary['fps_estimate']:.3f} |
| Total duration | {summary['total_duration_s']:.1f} s ({summary['total_hours']:.3f} h) |
| observation.body dim | {summary['observation_body_dim']} |
| task_index | {summary['task_index_counts']} |

## Frames / duration per episode

- frames/ep: min={fs['min']:.0f} median={fs['median']:.0f} mean={fs['mean']:.1f} max={fs['max']:.0f} std={fs['std']:.1f}
- duration_s/ep: min={ds_['min']:.2f} median={ds_['median']:.2f} mean={ds_['mean']:.2f} max={ds_['max']:.2f}

## Canonical split (episode-level, last 20% held out)

```
TRAIN_EPISODES = 0 .. 159   # {len(TRAIN_EPISODES)} episodes
HELDOUT_EPISODES = 160 .. 199  # {len(HELDOUT_EPISODES)} episodes
```

Use held-out episodes for generalization claims (Step 2 eval, Step 3 offline baselines, Step 4 oracle).

## When claims are valid / invalid

### Valid when
{chr(10).join('- ' + x for x in summary['claims']['valid_when'])}

### Invalid when
{chr(10).join('- ' + x for x in summary['claims']['invalid_when'])}

_Machine-readable:_ `{json_path}`
"""
    md_path.write_text(md)
    logger.info("Wrote %s and %s", json_path, md_path)
    return json_path, md_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="Wipe-table dataset card / episode split helpers")
    parser.add_argument("--write-card", action="store_true", help="Regenerate DATASET_CARD.md + JSON")
    parser.add_argument("--parse", type=str, default=None, help="Parse an episode spec and print list")
    args = parser.parse_args()
    if args.parse is not None:
        eps = parse_episode_spec(args.parse)
        print(json.dumps({"spec": args.parse, "n": len(eps), "episodes": eps}, indent=2))
        return
    if args.write_card:
        write_dataset_card()
        return
    parser.print_help()


if __name__ == "__main__":
    main()
