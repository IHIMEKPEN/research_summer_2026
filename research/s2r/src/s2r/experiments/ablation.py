"""Compare pipeline runs with ESN vs without (passthrough baselines)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from s2r.experiments.inspect_data import (
    extract_series,
    load_jsonl_rows,
    save_report,
    smoothness_score,
    summarize_array,
)
from s2r.experiments.paths import PROCESSED, RAW


def _engine_from_rows(rows: list[dict[str, Any]]) -> str:
    for r in rows:
        if r.get("topic") != "joint_cmd":
            continue
        p = r.get("payload") or {}
        src = str(p.get("source", ""))
        if src.startswith("passthrough"):
            return src
        if src == "esn" or p.get("engine") == "esn":
            return "esn"
    # fallback via metrics
    for r in rows:
        if r.get("topic") != "metrics":
            continue
        p = r.get("payload") or {}
        node = p.get("node")
        if node == "esn":
            return "esn"
        if node == "passthrough":
            mode = (p.get("extras") or {}).get("mode", "zoh")
            return f"passthrough_{mode}"
    return "unknown"


def analyze_run(path: Path | str, label: str | None = None) -> dict[str, Any]:
    rows = load_jsonl_rows(Path(path))
    series = extract_series(rows)
    engine = label or _engine_from_rows(rows)
    jerk = smoothness_score(series["cmds"])
    tracking = float("nan")
    if series["joint_pos"].size and series["cmds"].size:
        n = min(len(series["joint_pos"]), len(series["cmds"]))
        if n > 0:
            tracking = float(np.mean(np.linalg.norm(series["joint_pos"][:n] - series["cmds"][:n], axis=1)))

    return {
        "label": engine,
        "path": str(path),
        "n_rows": series["n_rows"],
        "rates_hz": {
            "action_token": series["token_hz"],
            "joint_cmd": series["cmd_hz"],
            "state": series["state_hz"],
        },
        "upsample_ratio": (
            series["cmd_hz"] / series["token_hz"] if series["token_hz"] > 1e-9 else float("nan")
        ),
        "cmd_jerk_proxy": jerk.to_dict(),
        "latency_ms": summarize_array("latency_ms", series["latencies_ms"]).to_dict(),
        "tracking_err_mean": tracking,
        "intent_counts": series["intent_counts"],
        "topic_counts": series["by_topic"],
    }


def compare_runs(paths: dict[str, Path | str]) -> dict[str, Any]:
    """paths: {label: episode_jsonl_or_dir}"""
    runs = {label: analyze_run(path, label=label) for label, path in paths.items()}
    # ranked by smoothness (lower jerk better) then tracking
    ranking = sorted(
        runs.values(),
        key=lambda r: (
            r["cmd_jerk_proxy"].get("p95") if not _nan(r["cmd_jerk_proxy"].get("p95")) else 1e9,
            r["tracking_err_mean"] if not _nan(r["tracking_err_mean"]) else 1e9,
        ),
    )
    report = {
        "runs": runs,
        "ranking_smoothest_to_roughest": [r["label"] for r in ranking],
        "research_notes": [
            "ESN should raise joint_cmd Hz above VLA token Hz while reducing jerk vs raw 2Hz steps.",
            "ZOH keeps high rate but holds piecewise-constant commands (no learned dynamics).",
            "Linear is a simple non-learning upsampler baseline.",
            "Raw exposes the control stack to sparse 2Hz steps (stress test without buffer).",
        ],
    }
    return report


def _nan(x: Any) -> bool:
    try:
        return x is None or (isinstance(x, float) and np.isnan(x))
    except Exception:
        return True


def latest_raw_episode(raw_dir: Path | None = None) -> Path | None:
    root = raw_dir or RAW
    eps = sorted(root.glob("episode_*.jsonl"))
    return eps[-1] if eps else None


def save_ablation_report(report: dict[str, Any], name: str = "esn_ablation_compare.json") -> Path:
    return save_report(report, name=name)
