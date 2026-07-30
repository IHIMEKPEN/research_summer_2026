"""Inspect episode JSONL / benchmark data for robotics suitability."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from s2r.experiments.paths import BENCHMARK_TASKS, PROCESSED, RAW


TOPIC_ROBOTICS = {
    "state",
    "action_token",
    "joint_cmd",
    "decision",
    "perception",
    "mission",
    "map",
    "metrics",
}


@dataclass
class DistributionSummary:
    name: str
    count: int
    mean: float
    std: float
    min: float
    p25: float
    p50: float
    p75: float
    p95: float
    max: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RoboticsFitness:
    score: float  # 0..1
    applies: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checklist: dict[str, bool] = field(default_factory=dict)


def _safe_percentile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def summarize_array(name: str, values: list[float] | np.ndarray) -> DistributionSummary:
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return DistributionSummary(name, 0, *(float("nan"),) * 8)
    return DistributionSummary(
        name=name,
        count=int(arr.size),
        mean=float(arr.mean()),
        std=float(arr.std()),
        min=float(arr.min()),
        p25=_safe_percentile(arr, 25),
        p50=_safe_percentile(arr, 50),
        p75=_safe_percentile(arr, 75),
        p95=_safe_percentile(arr, 95),
        max=float(arr.max()),
    )


def load_jsonl_rows(sources: list[Path] | Path | None = None) -> list[dict[str, Any]]:
    if sources is None:
        sources = [RAW]
    if isinstance(sources, Path):
        sources = [sources]
    rows: list[dict[str, Any]] = []
    for src in sources:
        src = Path(src)
        paths = [src] if src.is_file() else sorted(src.rglob("*.jsonl"))
        for path in paths:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        row["_file"] = str(path)
                        rows.append(row)
    return rows


def estimate_hz(timestamps: list[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    ts = np.sort(np.asarray(timestamps, dtype=np.float64))
    dt = np.diff(ts)
    dt = dt[dt > 1e-6]
    if dt.size == 0:
        return 0.0
    return float(1.0 / np.median(dt))


def extract_series(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_topic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_topic[str(r.get("topic", "unknown"))].append(r)

    joint_pos, joint_vel, actions, cmds = [], [], [], []
    token_ts, cmd_ts, state_ts = [], [], []
    decisions, intents, phases = [], [], []
    latencies = []

    for r in by_topic.get("state", []):
        p = r.get("payload") or {}
        if "joint_pos" in p:
            joint_pos.append(p["joint_pos"])
            state_ts.append(float(r.get("ts", 0.0)))
        if "joint_vel" in p:
            joint_vel.append(p["joint_vel"])

    for r in by_topic.get("action_token", []):
        p = r.get("payload") or {}
        if "action" in p:
            actions.append(p["action"])
            token_ts.append(float(r.get("ts", 0.0)))

    for r in by_topic.get("joint_cmd", []):
        p = r.get("payload") or {}
        if "q" in p:
            cmds.append(p["q"])
            cmd_ts.append(float(r.get("ts", 0.0)))

    for r in by_topic.get("decision", []):
        p = r.get("payload") or {}
        decisions.append(p)
        if "intent" in p:
            intents.append(str(p["intent"]))
        if "latency_ms" in p:
            latencies.append(float(p["latency_ms"]))

    for r in by_topic.get("mission", []):
        p = r.get("payload") or {}
        if "phase" in p:
            phases.append(str(p["phase"]))

    for r in by_topic.get("metrics", []):
        p = r.get("payload") or {}
        if "latency_ms" in p:
            latencies.append(float(p["latency_ms"]))

    def _stack(xs: list) -> np.ndarray:
        if not xs:
            return np.zeros((0, 0), dtype=np.float64)
        dim = len(xs[0])
        clean = [x for x in xs if len(x) == dim]
        if not clean:
            return np.zeros((0, 0), dtype=np.float64)
        return np.asarray(clean, dtype=np.float64)

    return {
        "by_topic": {k: len(v) for k, v in by_topic.items()},
        "joint_pos": _stack(joint_pos),
        "joint_vel": _stack(joint_vel),
        "actions": _stack(actions),
        "cmds": _stack(cmds),
        "token_hz": estimate_hz(token_ts),
        "cmd_hz": estimate_hz(cmd_ts),
        "state_hz": estimate_hz(state_ts),
        "intent_counts": dict(Counter(intents)),
        "phase_counts": dict(Counter(phases)),
        "latencies_ms": np.asarray(latencies, dtype=np.float64),
        "n_files": len({r.get("_file") for r in rows}),
        "n_rows": len(rows),
    }


def joint_distributions(arr: np.ndarray, prefix: str) -> list[DistributionSummary]:
    if arr.ndim != 2 or arr.shape[0] == 0:
        return [summarize_array(f"{prefix}_empty", [])]
    out = []
    for j in range(arr.shape[1]):
        out.append(summarize_array(f"{prefix}_j{j}", arr[:, j]))
    # also magnitude
    out.append(summarize_array(f"{prefix}_norm", np.linalg.norm(arr, axis=1)))
    return out


def smoothness_score(cmds: np.ndarray) -> DistributionSummary:
    """Mean absolute jerk/accel proxy from discrete joint commands."""
    if cmds.ndim != 2 or cmds.shape[0] < 2:
        return summarize_array("cmd_jerk_proxy", [])
    vel = np.diff(cmds, axis=0)
    if cmds.shape[0] < 3:
        mag = np.linalg.norm(vel, axis=1)
        return summarize_array("cmd_jerk_proxy", mag)
    acc = np.diff(vel, axis=0)
    if cmds.shape[0] < 4:
        mag = np.linalg.norm(acc, axis=1)
        return summarize_array("cmd_jerk_proxy", mag)
    jerk = np.diff(acc, axis=0)
    mag = np.linalg.norm(jerk, axis=1)
    return summarize_array("cmd_jerk_proxy", mag)


def robotics_fitness(series: dict[str, Any], joint_limits: dict[str, list[float]] | None = None) -> RoboticsFitness:
    reasons: list[str] = []
    warnings: list[str] = []
    checks: dict[str, bool] = {}

    topics = series["by_topic"]
    checks["has_state"] = topics.get("state", 0) > 0
    checks["has_action_token"] = topics.get("action_token", 0) > 0
    checks["has_joint_cmd"] = topics.get("joint_cmd", 0) > 0
    checks["has_decision_or_mission"] = topics.get("decision", 0) + topics.get("mission", 0) > 0
    checks["control_rate_ok"] = series["cmd_hz"] >= 20.0 or series["state_hz"] >= 20.0
    checks["vla_rate_sparse"] = 0.2 <= series["token_hz"] <= 10.0 if series["token_hz"] > 0 else False

    jp = series["joint_pos"]
    cmds = series["cmds"]
    if jp.size and joint_limits:
        lo = np.asarray(joint_limits.get("min", []), dtype=np.float64)
        hi = np.asarray(joint_limits.get("max", []), dtype=np.float64)
        if lo.size and hi.size and jp.shape[1] == lo.size:
            in_range = float(np.mean((jp >= lo) & (jp <= hi)))
            checks["joints_within_limits"] = in_range > 0.95
            if in_range <= 0.95:
                warnings.append(f"Only {in_range:.1%} joint samples within configured limits")
        else:
            checks["joints_within_limits"] = False
            warnings.append("Joint limit dims do not match state dim")
    else:
        checks["joints_within_limits"] = jp.size > 0

    if cmds.size and cmds.shape[0] >= 3:
        jerk = smoothness_score(cmds)
        checks["commands_not_pathological"] = bool(jerk.count and jerk.p95 < 50.0)
        if jerk.count and jerk.p95 >= 50.0:
            warnings.append(f"High command jerk p95={jerk.p95:.2f} — may be harsh for real robots")
    else:
        checks["commands_not_pathological"] = False
        warnings.append("Not enough joint_cmd samples to assess smoothness")

    # Coverage of robotics topics
    coverage = sum(1 for t in TOPIC_ROBOTICS if topics.get(t, 0) > 0) / len(TOPIC_ROBOTICS)
    checks["topic_coverage_ge_50pct"] = coverage >= 0.5

    score = sum(1 for v in checks.values() if v) / max(1, len(checks))
    applies = score >= 0.6 and checks.get("has_state", False) and (
        checks.get("has_joint_cmd", False) or checks.get("has_action_token", False)
    )

    if applies:
        reasons.append("Dataset contains robot state and action/command streams suitable for control learning")
    else:
        reasons.append("Missing core robotics streams (state + actions/commands) or rates look non-robotic")

    if series["token_hz"] > 0 and series["cmd_hz"] > 0:
        reasons.append(
            f"Observed rates: action_token≈{series['token_hz']:.2f}Hz, joint_cmd≈{series['cmd_hz']:.2f}Hz, "
            f"state≈{series['state_hz']:.2f}Hz"
        )
        if series["cmd_hz"] >= 5 * max(series["token_hz"], 1e-6):
            reasons.append("Command rate >> token rate — good candidate for ESN / upsampling research")
        else:
            warnings.append("Command rate is not much higher than token rate — limited upsampling signal")

    return RoboticsFitness(score=score, applies=applies, reasons=reasons, warnings=warnings, checklist=checks)


def inspect_dataset(
    sources: list[Path] | Path | None = None,
    joint_limits: dict[str, list[float]] | None = None,
) -> dict[str, Any]:
    rows = load_jsonl_rows(sources)
    series = extract_series(rows)
    report = {
        "n_rows": series["n_rows"],
        "n_files": series["n_files"],
        "topic_counts": series["by_topic"],
        "rates_hz": {
            "action_token": series["token_hz"],
            "joint_cmd": series["cmd_hz"],
            "state": series["state_hz"],
        },
        "intent_counts": series["intent_counts"],
        "phase_counts": series["phase_counts"],
        "distributions": {
            "joint_pos": [d.to_dict() for d in joint_distributions(series["joint_pos"], "joint_pos")],
            "joint_vel": [d.to_dict() for d in joint_distributions(series["joint_vel"], "joint_vel")],
            "action_token": [d.to_dict() for d in joint_distributions(series["actions"], "action")],
            "joint_cmd": [d.to_dict() for d in joint_distributions(series["cmds"], "joint_cmd")],
            "latency_ms": summarize_array("latency_ms", series["latencies_ms"]).to_dict(),
            "cmd_jerk_proxy": smoothness_score(series["cmds"]).to_dict(),
        },
        "robotics_fitness": asdict(robotics_fitness(series, joint_limits=joint_limits)),
    }
    return report


def inspect_benchmark_coverage() -> dict[str, Any]:
    coverage = []
    for task_dir in sorted(BENCHMARK_TASKS.glob("*")):
        if not task_dir.is_dir():
            continue
        coverage.append(
            {
                "task": task_dir.name,
                "episodes": len(list((task_dir / "episodes").glob("*"))),
                "metrics": len(list((task_dir / "metrics").glob("*.json"))),
                "videos": len(list((task_dir / "videos").glob("*"))),
                "annotations": len(list((task_dir / "annotations").glob("*"))),
            }
        )
    return {"tasks": coverage, "n_tasks": len(coverage)}


def save_report(report: dict[str, Any], name: str = "data_inspect.json") -> Path:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    path = PROCESSED / name
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
