#!/usr/bin/env python3
"""Go/no-go oracle benchmark gate for the stabilized wipe simulator.

Pass criterion (preregistered): demonstration replay (oracle_linear or oracle_pd)
achieves stable contact and a plausible path on ≥ 90% of evaluated episodes
with zero NaN/QACC terminations.

Usage (from research/):

  PYTHONPATH=. MUJOCO_GL=egl python notebooks/run_oracle_benchmark_gate.py \\
      --episodes 0,1,2,3,4,5,160,161,162 --press-table
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from datasets import load_dataset

NOTEBOOKS = Path(__file__).resolve().parent
ROOT = NOTEBOOKS.parent
import sys
sys.path.insert(0, str(NOTEBOOKS))
sys.path.insert(0, str(ROOT))

from wipe_control_baselines import BASELINE_NAMES, run_baseline
from wipe_esn_experiment import MAX_PLAUSIBLE_WIPE_PATH_M, pack_episodes

MJCF = ROOT / "unitree_mujoco/unitree_robots/g1/g1_29dof.xml"
OUT = ROOT / "results/main_independent_esn/oracle_benchmark_gate"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", default="0,1,2,3,4,5,160,161,162")
    p.add_argument("--baselines", default="stationary,oracle_zoh,oracle_linear,oracle_pd")
    p.add_argument("--press-table", action="store_true")
    p.add_argument("--pass-rate", type=float, default=0.90)
    args = p.parse_args()
    episodes = [int(x) for x in args.episodes.split(",") if x.strip()]
    baselines = [x.strip() for x in args.baselines.split(",") if x.strip()]
    for b in baselines:
        if b not in BASELINE_NAMES:
            raise SystemExit(f"Unknown baseline {b}")

    OUT.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("unitreerobotics/G1_Dex1_Wipe_Table")["train"]
    packed = pack_episodes(ds, episodes)
    rows = []
    t0 = time.perf_counter()
    for name in baselines:
        for ep in episodes:
            print(f"[gate] {name} ep={ep}", flush=True)
            r = run_baseline(name, packed[ep], MJCF, press_table=args.press_table, teacher_weight=0.0)
            assert r["teacher_source"] == "none", "held-out/task eval must not use proxy teacher loss"
            assert r["L_teacher"] == 0.0
            ok = (
                (not r["terminated_unstable"])
                and r["plausible_path"]
                and r["wipe_path_length_m"] <= MAX_PLAUSIBLE_WIPE_PATH_M
                and r["grasp_success"]
                and r["table_contact_ratio"] >= 0.15  # soft contact evidence for gate diagnostics
            )
            rows.append({
                "baseline": name,
                "episode": ep,
                "press_table": args.press_table,
                "gate_ok": ok,
                "terminated_unstable": r["terminated_unstable"],
                "grasp_success": r["grasp_success"],
                "wipe_path_length_m": r["wipe_path_length_m"],
                "table_contact_ratio": r["table_contact_ratio"],
                "wipe_coverage_m2": r["wipe_coverage_m2"],
                "max_cloth_jump_m": r["max_cloth_jump_m"],
                "plausible_path": r["plausible_path"],
                "L_task": r["L_task"],
                "L_grasp": r["L_grasp"],
                "L_path": r["L_path"],
                "L_contact": r["L_contact"],
                "L_coverage": r["L_coverage"],
                "L_smooth": r["L_smooth"],
                "L_limits": r["L_limits"],
                "L_teacher": r["L_teacher"],
                "teacher_source": r["teacher_source"],
                "task_success": r["task_success"],
            })

    summary = {"baselines": {}}
    for name in baselines:
        subset = [r for r in rows if r["baseline"] == name]
        n = len(subset)
        n_ok = sum(1 for r in subset if r["gate_ok"])
        n_nan = sum(1 for r in subset if r["terminated_unstable"])
        summary["baselines"][name] = {
            "n": n,
            "gate_ok": n_ok,
            "gate_rate": n_ok / n if n else 0.0,
            "nan_terminations": n_nan,
            "mean_path": float(np.mean([r["wipe_path_length_m"] for r in subset])),
            "mean_contact": float(np.mean([r["table_contact_ratio"] for r in subset])),
            "mean_L_task": float(np.mean([r["L_task"] for r in subset])),
        }

    # Go/no-go focuses on continuous demo replay.
    focus = "oracle_pd" if "oracle_pd" in summary["baselines"] else "oracle_linear"
    focus_stats = summary["baselines"].get(focus, {})
    passed = (
        focus_stats.get("gate_rate", 0.0) >= args.pass_rate
        and focus_stats.get("nan_terminations", 1) == 0
    )
    report = {
        "schema": "oracle_benchmark_gate_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "press_table": args.press_table,
        "pass_rate_threshold": args.pass_rate,
        "focus_baseline": focus,
        "passed": passed,
        "elapsed_s": time.perf_counter() - t0,
        "summary": summary,
        "rows": rows,
        "next_step": (
            "Proceed to hierarchical contact controller."
            if passed else
            "STOP: fix simulator/thresholds before any ESN optimizer comparison."
        ),
    }
    out_json = OUT / ("gate_press.json" if args.press_table else "gate_nopress.json")
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("passed", "focus_baseline", "summary", "next_step")}, indent=2))
    print(out_json)


if __name__ == "__main__":
    main()
