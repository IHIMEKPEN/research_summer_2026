"""
============================================================
Step 3 — Sim comparison: offline baselines + live UnifoLM bridges
Research Plan: VLA + ESN for Real-Time Humanoid Control
Author: Osemudiamen Andrew Ihimekpen | PVAMU CREDIT Center
============================================================

Completes the missing sim experiments for this week:
  1) Offline ZOH / linear (/ PID) vs demo joints  → results/step3_baselines/
  2) Optional live UnifoLM MuJoCo runs for esn/zoh/linear/pid
     → results/step3_dual_thread/dual_thread_report_{bridge}_live.json

Usage (from research/):
  # Offline only (CPU/dataset; no UnifoLM GPU load)
  python3 -m src.step3_sim_comparison --episode 0 --offline_only

  # Full sim suite: offline table + live UnifoLM for all bridges
  python3 -m src.step3_sim_comparison --episode 0 --duration_s 10

  # Live only, ESN then baselines
  python3 -m src.step3_sim_comparison --skip_offline --bridges esn,zoh,linear,pid --duration_s 10

Next week (hardware): Jetson AGX Thor + G1 via s2r — do not block on that here.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import List, Sequence

from src.paths import results_path
from src.step3_control_baselines import run_all_baselines, write_comparison_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OFFLINE_DIR = results_path("step3_baselines")
LIVE_DIR = results_path("step3_dual_thread")
EVAL_DIR = results_path("step3_evaluation")


def _run_offline(episode: int) -> Path:
    logger.info("Running offline ZOH / linear / PID baselines (episode=%d) ...", episode)
    results = run_all_baselines(episode)
    for r in results:
        out = OFFLINE_DIR / f"baseline_{r.method}_ep{episode}.json"
        OFFLINE_DIR.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(r), indent=2))
        logger.info("%s RMSE=%.6f jerk=%.3e → %s", r.method, r.rmse, r.jerk, out)
    csv_path = write_comparison_table(results, OFFLINE_DIR)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    # Mirror into step3_evaluation for the paper pipeline.
    write_comparison_table(results, EVAL_DIR)
    return csv_path


def _run_live_bridge(
    bridge: str,
    *,
    duration_s: float,
    episode: int,
    mock: bool,
    record_video: bool,
    extra_args: Sequence[str],
) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "src.step3_dual_thread_mujoco",
        "--bridge",
        bridge,
        "--duration_s",
        str(duration_s),
        "--episode",
        str(episode),
        *extra_args,
    ]
    if mock:
        cmd.append("--mock")
    if record_video:
        cmd.append("--record_video")
    logger.info("Launching live dual-process: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    tag = "mock" if mock else "live"
    report = LIVE_DIR / f"dual_thread_report_{bridge}_{tag}.json"
    if not report.exists():
        raise FileNotFoundError(f"Expected report missing: {report}")
    return report


def _write_summary(offline_csv: Path | None, live_reports: List[Path]) -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "offline_baselines_csv": str(offline_csv) if offline_csv else None,
        "live_reports": [],
    }
    for path in live_reports:
        summary["live_reports"].append(json.loads(path.read_text()))
    out = EVAL_DIR / "sim_comparison_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote summary: %s", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run offline ZOH/linear baselines + optional live UnifoLM bridge comparison",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--duration_s", type=float, default=10.0)
    parser.add_argument(
        "--bridges",
        type=str,
        default="esn,zoh,linear,pid",
        help="Comma-separated bridge modes for live MuJoCo runs",
    )
    parser.add_argument(
        "--offline_only",
        action="store_true",
        help="Only fill ZOH/linear/PID offline tables (no UnifoLM)",
    )
    parser.add_argument(
        "--skip_offline",
        action="store_true",
        help="Skip offline baselines; only run dual-process bridges",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock VLA in dual-process (smoke test only; default is live UnifoLM)",
    )
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to step3_dual_thread_mujoco after --",
    )
    args = parser.parse_args()

    extra = list(args.extra)
    if extra and extra[0] == "--":
        extra = extra[1:]

    offline_csv = None
    if not args.skip_offline:
        offline_csv = _run_offline(args.episode)

    live_reports: List[Path] = []
    if not args.offline_only:
        bridges = [b.strip() for b in args.bridges.split(",") if b.strip()]
        for bridge in bridges:
            report = _run_live_bridge(
                bridge,
                duration_s=args.duration_s,
                episode=args.episode,
                mock=args.mock,
                record_video=args.record_video,
                extra_args=extra,
            )
            live_reports.append(report)

    summary_path = _write_summary(offline_csv, live_reports)
    print("\n" + "=" * 60)
    print("  Step 3 — Sim comparison complete")
    print("=" * 60)
    if offline_csv:
        print(f"  Offline baselines : {offline_csv}")
    for r in live_reports:
        print(f"  Live report       : {r}")
    print(f"  Summary           : {summary_path}")
    if args.mock:
        print("  NOTE: --mock was set. Re-run without --mock for paper timing.")
    else:
        print("  Next week: deploy same bridges on Jetson AGX Thor → G1 (s2r).")
    print("=" * 60)


if __name__ == "__main__":
    main()
