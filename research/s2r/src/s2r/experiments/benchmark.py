"""12-task benchmark utilities for model vs Unitree VLA evaluation."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from s2r.experiments.paths import BENCHMARK, BENCHMARK_RESULTS, BENCHMARK_TASKS


@dataclass
class BenchmarkTask:
    id: str
    name: str
    hf_dataset: str
    instruction: str
    category: str
    success_metric: str

    @property
    def dir(self) -> Path:
        return BENCHMARK_TASKS / self.id


@dataclass
class EpisodeScore:
    task_id: str
    model_name: str
    episode_id: str
    success: bool
    completion_time_s: float = 0.0
    interventions: int = 0
    e2e_latency_ms: float = 0.0
    vla_hz: float = 0.0
    esn_hz: float = 0.0
    decision_latency_ms: float = 0.0
    extras: dict[str, Any] = field(default_factory=dict)


def load_tasks(catalog: str | Path | None = None) -> list[BenchmarkTask]:
    path = Path(catalog) if catalog else BENCHMARK / "tasks.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [BenchmarkTask(**t) for t in data.get("tasks", [])]


def task_by_id(task_id: str, catalog: str | Path | None = None) -> BenchmarkTask:
    for t in load_tasks(catalog):
        if t.id == task_id or t.id.endswith(task_id) or t.name.lower() == task_id.lower():
            return t
    raise KeyError(f"Unknown task: {task_id}")


def score_episode(
    task_id: str,
    model_name: str,
    episode_id: str,
    success: bool,
    completion_time_s: float = 0.0,
    interventions: int = 0,
    e2e_latency_ms: float = 0.0,
    vla_hz: float = 0.0,
    esn_hz: float = 0.0,
    decision_latency_ms: float = 0.0,
    extras: dict[str, Any] | None = None,
) -> EpisodeScore:
    return EpisodeScore(
        task_id=task_id,
        model_name=model_name,
        episode_id=episode_id,
        success=success,
        completion_time_s=completion_time_s,
        interventions=interventions,
        e2e_latency_ms=e2e_latency_ms,
        vla_hz=vla_hz,
        esn_hz=esn_hz,
        decision_latency_ms=decision_latency_ms,
        extras=extras or {},
    )


def save_score(score: EpisodeScore, out_dir: str | Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else BENCHMARK_RESULTS
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{score.model_name}__{score.task_id}__{score.episode_id}.json"
    path.write_text(json.dumps(asdict(score), indent=2), encoding="utf-8")
    # also append to task metrics folder
    task_metrics = BENCHMARK_TASKS / score.task_id / "metrics"
    task_metrics.mkdir(parents=True, exist_ok=True)
    (task_metrics / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def load_scores(results_dir: str | Path | None = None) -> list[EpisodeScore]:
    root = Path(results_dir) if results_dir else BENCHMARK_RESULTS
    scores: list[EpisodeScore] = []
    if not root.exists():
        return scores
    for p in sorted(root.glob("*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        scores.append(EpisodeScore(**data))
    return scores


def summarize_results(
    scores: list[EpisodeScore] | None = None,
    results_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    if scores is None:
        scores = load_scores(results_dir)
    by_key: dict[tuple[str, str], list[EpisodeScore]] = {}
    for s in scores:
        by_key.setdefault((s.model_name, s.task_id), []).append(s)
    rows = []
    for (model, task), xs in sorted(by_key.items()):
        n = len(xs)
        rows.append(
            {
                "model": model,
                "task_id": task,
                "n": n,
                "success_rate": sum(1 for x in xs if x.success) / n if n else 0.0,
                "avg_completion_time_s": sum(x.completion_time_s for x in xs) / n if n else 0.0,
                "avg_e2e_latency_ms": sum(x.e2e_latency_ms for x in xs) / n if n else 0.0,
                "avg_esn_hz": sum(x.esn_hz for x in xs) / n if n else 0.0,
                "avg_vla_hz": sum(x.vla_hz for x in xs) / n if n else 0.0,
            }
        )
    return rows


def make_leaderboard(scores: list[EpisodeScore] | None = None) -> list[dict[str, Any]]:
    """Aggregate mean success across the 12 tasks per model."""
    rows = summarize_results(scores)
    by_model: dict[str, list[float]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r["success_rate"])
    board = []
    for model, rates in by_model.items():
        board.append(
            {
                "model": model,
                "tasks_evaluated": len(rates),
                "mean_success_rate": sum(rates) / len(rates) if rates else 0.0,
                "timestamp": time.time(),
            }
        )
    return sorted(board, key=lambda x: x["mean_success_rate"], reverse=True)
